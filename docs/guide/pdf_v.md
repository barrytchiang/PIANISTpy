# Local velocity distribution

Source: https://pianistpy.com/docs/guide/pdf_v

`pianistpy.pdf_v()`, `pianistpy.LosTable`

`pdf_v(source, r, v)` returns the local velocity distribution \(p(V\,|\,R)\) at user-specified 3D radii.
\(p(V\,|\,R)\) is the probability density of the velocity \(V = |\mathbf{V}|\) of the model’s mass at the
distance \(R\) from the center. As in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758), the density at a radius is written as an integral over the angle
\(\varphi\) between the velocity and the radial direction, with the angle density \(p_\varphi\), and over
the velocity magnitude, with its density \(p_V\),

\[\varrho(R) = 2\pi \int_0^\pi {\rm d}\varphi\; p_\varphi(\varphi\,|\,R)
\int_0^{V_{\rm esc}(R)} {\rm d}V\; p_V(V\,|\,R, \varphi),
\qquad V_{\rm esc}(R) = \sqrt{2P(R)},\]

and the velocity distribution is the product \(p_\varphi\,p_V\) integrated over the angle and multiplied by
\(2\pi/\varrho(R)\):

\[p(V\,|\,R) = \frac{2\pi}{\varrho(R)} \int_0^\pi p_\varphi(\varphi\,|\,R)\, p_V(V\,|\,R, \varphi)\,{\rm d}\varphi .\]

\(p(V\,|\,R)\) integrates to one over \(V \geq 0\). For an isotropic, constant-anisotropy or Type I model,
the support of \(p(V\,|\,R)\) ends at the escape velocity \(V_{\rm esc}(R)\). A Type II model also holds
unbound orbits, and the velocities of a Type II model reach \(V_{\rm esc}/\sqrt{1 - R^2/R_{\rm a}^2}\). For a Type
II model, a radius at or beyond the anisotropy radius \(R_{\rm a}\) raises `InputError`. The
radius \(R\) of
[Chiang+(2026d)](https://arxiv.org/abs/2610.28758) is `r` in the package.

## Arguments and units

- `source` is a `Model` or the model’s line-of-sight table (`LosTable`).

- `r` and `v` are arrays that broadcast against each other. `r` holds radii in the model’s unit of length:
internal units by default, parsecs or an Astropy quantity in the physical system. `v` holds velocities in the
model’s unit of velocity (km/s in the physical system).

- The result has the broadcast shape of `r` and `v` and is a probability density per unit velocity of the
model’s unit system. The result is zero at negative velocities and beyond the end of the support.

For a superposition, the components are mixed by their density at the radius,
\(p = \sum_k w_k \varrho_k p_k / \sum_k w_k \varrho_k\).

## Accuracy

Each distinct radius is computed once, with three checks:

- the norm of the distribution against the density;

- the second moment of the distribution against \(\sigma_{\rm r}^2 (3 - 2\beta_\sigma)\) from [Radial velocity dispersion](https://pianistpy.com/docs/guide/sigma_r);

- the change of the distribution when the number of nodes of the distribution’s angle integral doubles.

The number of nodes of the angle integral doubles, at most three times, until the change of the distribution is
under `velocity_tolerance`: \(10^{-6}\) with the default settings, \(10^{-4}\) with `Settings(quick=True)`.
A distribution whose relative error is above \(10^{-4}\) is returned with an
`AccuracyWarning`. A distribution whose relative error is above \(10^{-2}\) raises
`AccuracyError`, unless the model’s settings set `write_anyway=True`.

[`pianistpy.observables.stages.velocity_distribution()`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.stages.velocity_distribution) returns the velocity distribution of a one-component model at
one radius, in internal units in every unit system. The returned object also holds the record of the distribution’s
checks, the escape velocity, the end of the distribution’s support and the share of the distribution above the escape
velocity (`unbound_fraction`, nonzero only for a Type II model).

## Example

```python
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
v = np.linspace(0.0, 1.4, 1401)                           # beyond the escape velocity at each of the three radii
p = pp.pdf_v(model, np.array([[0.1], [1.0], [10.0]]), v)   # (3, 1401): radii by velocities
p.sum(axis=1) * (v[1] - v[0])                              # one at each radius, to the accuracy of the sum on this grid

physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.pdf_v(physical, 500.0, [10.0, 50.0, 100.0])             # per km/s, at 500 pc
```
