# Differentiable observables

Source: https://pianistpy.com/docs/guide/differentiable

`pianistpy.Differentiable`

## The object

`pp.Differentiable(model)` gives the observables of a model of one component as JAX functions of the component’s
anisotropy \(\theta = (\beta_{\rm c}, R_{\rm a})\), the central anisotropy and the anisotropy radius, passed as
`theta`. The model is a Method II model, a distribution function inverted on the truncated profile of a Method I
model ([Truncation in phase space](https://pianistpy.com/docs/guide/truncation)), as in the example below. Each method takes \(\theta\) as
the first argument, returns the value of the public function of the same name on the model rebuilt at
\(\theta\), and works under `jax.jit` and `jax.grad`:

```python
d = pp.Differentiable(model)
d.sigma_r(theta, r)                    # as pp.sigma_r(model at theta, r)
d.sigma_t(theta, r)                    # as pp.sigma_t
d.sigma_los(theta, r_2d)               # as pp.sigma_los
d.sigma_los_aperture(theta, aperture)  # as pp.sigma_los_aperture
d.losvd(theta, r_2d, v)                # as pp.losvd
d.pdf_v(theta, r, v)                   # as pp.pdf_v
d.aperture_losvd(theta, aperture, v)   # as pp.aperture_losvd
```

Every radius, velocity, aperture length and \(\theta\) that a method reads, and every value that a method
returns, is in internal units whatever the model’s unit system, as is the anisotropy `theta` of `likelihood`.
`d.theta` is the model’s own anisotropy.

`Differentiable` accepts an isotropic, constant-anisotropy, Osipkov-Merritt Type I or Type II, or
Cuddeford-Osipkov-Merritt (COM) Type I component. For an isotropic or constant-anisotropy component,
\(R_{\rm a}\) is infinite and stays at infinity at every \(\theta\), and the gradient in \(R_{\rm a}\) is
zero. For an Osipkov-Merritt component, the gradient of `losvd`, `aperture_losvd` and `pdf_v` in
\(\beta_{\rm c}\) is NaN. An Osipkov-Merritt component has \(\beta_{\rm c} = 0\), where the form of the
inversion changes and the derivatives of the distribution function from the two sides of \(\beta_{\rm c} = 0\)
differ by more than \(10^{-6}\). For an Osipkov-Merritt component, `sigma_r`, `sigma_t`, `sigma_los` and
`sigma_los_aperture` carry the derivative in \(\beta_{\rm c}\), computed from the Jeans equation.

The constructor raises `InputError`, with a message that states the cause, for:

- a COM Type II component ([End-to-end differentiability](https://pianistpy.com/docs/guide/differentiability)),

- a model of more than one component, such as a superposition (the components of a superposition are
differentiated through `likelihood`),

- a component that is not spatially finite,

- a component not on a truncated profile that carries the distribution function of the Method I model it was built from.

## What a method computes

A method’s value at \(\theta\) is the public call on the model rebuilt at \(\theta\). At the model’s own
\(\theta\), a method returns the value of the public call on the model itself, bit for bit. The derivative of a
method holds fixed the radii and velocities at which the tables at \(\theta\) are tabulated, as `likelihood`
does for a free component. Where the tabulated radii and velocities change from one \(\theta\) to another, the
value of a method changes by up to the tolerance of the tables, and the derivative omits that change.

The derivatives of `losvd`, `pdf_v` and `aperture_losvd` are taken through the derivative of the distribution
function ([End-to-end differentiability](https://pianistpy.com/docs/guide/differentiability)), with the energies of the distribution function’s table held fixed. For a
component whose distribution function has no derivative in the anisotropy, such as an isotropic component,
`losvd`, `pdf_v` and `aperture_losvd` raise `InputError` with a message that states the
reason.

At a \(\theta\) where rebuilding the model raises `InputError`,
`UnphysicalModelError`, `AccuracyError` or `ResultError`, the
value is NaN and the gradient is zero. A method is not differentiable in the radii or velocities that the method
reads, and the gradient in a radius or a velocity is NaN.

## The cost

The first call of a method at a new \(\theta\) rebuilds the model at \(\theta\). The rebuild inverts the
distribution function at \(\theta\) on the component’s truncated profile, then computes the dispersion profile
and the table of the line-of-sight dispersion. A rebuild costs about as much as the public call on a model built at
\(\theta\), as does the rebuild of a free component in `likelihood`. A `Differentiable` keeps the tables of
at most eight values of \(\theta\), the model’s own \(\theta\) and the seven other values first met most
recently, so the value and the gradient at one \(\theta\) share one rebuild.

The derivatives of `losvd` and `aperture_losvd` cost far more. The derivatives of `losvd` and
`aperture_losvd` recompute and differentiate each row of the line-of-sight table that the call reads, and take
minutes to hours, with the number of rows the call reads ([Release notes](https://pianistpy.com/docs/project/release-notes)).

The values of `losvd` and `aperture_losvd`, without a derivative, cost only the rebuild at \(\theta\) and the
public call of the same name.

`Settings(check_rows=True)`, off by default, checks each row that the derivatives of `losvd` and
`aperture_losvd` read. The check requires that:

- each recomputed row equals the stored row, bit for bit;

- the row built again one velocity at a time, each at the quadrature level that gave the velocity’s value, equals the
recomputed row, bit for bit;

- the JAX computation of the row that carries the derivative in the anisotropy equals the row to \(10^{-12}\) of
the row’s largest value.

A row that fails raises a `ResultError` that identifies the row. The gradient is the same with and without the checks. On a
GPU the checks may fail a row, since a GPU’s order of summation does not promise the same bits. On the CPU, every row
passes the checks.

## Example

The example computes the derivative in \(\beta_{\rm c}\) of the line-of-sight dispersion at one projected radius
twice, by `jax.grad` and by a central difference:

```python
import jax
import numpy as np

import pianistpy as pp

truncated = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy()), pp.Truncation(k_t=0.05)
)
model = pp.solve(pp.Component(truncated.profile, pp.Anisotropy(beta_c=0.25)))
d = pp.Differentiable(model)

rule = float(jax.grad(lambda t: d.sigma_los(t, 0.3))(d.theta)[0])
h = np.array([1e-3, 0.0])
difference = float(d.sigma_los(d.theta + h, 0.3) - d.sigma_los(d.theta - h, 0.3)) / 2e-3
print(rule, difference)
```

The `jax.grad` derivative and the central difference agree to the error of the central difference. Each of the two
shifted \(\theta\) costs one rebuild.
