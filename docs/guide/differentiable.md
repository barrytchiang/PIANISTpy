# Differentiable observables

Source: https://pianistpy.com/docs/guide/differentiable.html

# Differentiable observables

pianistpy.Differentiable

## The object

pp.Differentiable(model) holds a model of one component on a truncated profile (a distribution function inverted
on its parent’s profile, as pianistpy.truncate() makes it) and serves its observables as JAX functions of the
component’s anisotropy, theta = (beta_c, R_a). Each method takes theta first and mirrors the public call of the
same name, traceable and differentiable in theta:

```
d = pp.Differentiable(model)
d.sigma_r(theta, r)                    # as pp.sigma_r(model at theta, r)
d.sigma_t(theta, r)                    # as pp.sigma_t
d.sigma_los(theta, r_2d)               # as pp.sigma_los
d.sigma_los_aperture(theta, aperture)  # as pp.sigma_los_aperture
d.losvd(theta, r_2d, v)                # as pp.losvd
d.pdf_v(theta, r, v)                   # as pp.pdf_v
d.aperture_losvd(theta, aperture, v)   # as pp.aperture_losvd

```

Everything is in internal units, whatever the model’s unit system: theta, as the likelihood’s, the radii and apertures
a method reads, and the velocities it returns. d.theta is the model’s own anisotropy.

The families are those of the distribution function’s own rule (Differentiability): constant anisotropy (beta_c; an
isotropic or constant component’s R_a is held at its own, infinite, value with a zero gradient), Cuddeford Type I
(beta_c and R_a), Osipkov-Merritt Type I and Type II (R_a). Refused at construction, each with an error that names it:
Cuddeford Type II (its interior point at P(R_a) moves with both parameters), a superposition or a model of several
components (its components are differentiated through the likelihood), a component that is not spatially finite, and
one not on a truncated profile that carries its parent’s distribution function.

## What a method computes

A method’s value at theta is the public call on the model rebuilt at theta: at the model’s own theta, the plain call’s
own value, to the byte. Its derivative is the observable’s tangent rows in theta on the tables’ own rows, their
placement frozen, as the likelihood’s free components carry it: between two placements the value moves by the tables’
tolerance, which the derivative does not carry. Over an aperture the derivative is the aperture’s integral of the
tangent rows of the squared dispersion, on the plain call’s own nodes and weights (the integral is linear in the
rows). The line-of-sight velocity distribution, the speed distribution and the aperture’s line-of-sight velocity
distribution take theirs through the distribution function’s own derivative (Differentiability), their placement frozen;
a member whose distribution function that rule refuses is named at construction, and these three methods raise with
it. At a theta where the model is refused the value is not a number and its gradient zero. A method is not
differentiable in its radii.

## The cost

The first time a method meets a theta, the host rebuilds the model there: the distribution function inverted at theta
on the component’s truncated profile, then the dispersion profile and the projected table with their tangent rows.
That is the cost of the likelihood’s own rebuild of a free component, about that of the plain call on the model at
theta. The last eight theta are kept, so the value and the gradient at one theta share one rebuild.

## Example

The derivative of the line-of-sight dispersion at one projected radius in beta_c, by the rule and by a central
difference of the same function:

```
import jax
import numpy as np

import pianistpy as pp

parent = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy()), pp.Truncation(k_t=0.05)
)
model = pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=0.25)))
d = pp.Differentiable(model)

rule = float(jax.grad(lambda t: d.sigma_los(t, 0.3))(d.theta)[0])
h = np.array([1e-3, 0.0])
difference = float(d.sigma_los(d.theta + h, 0.3) - d.sigma_los(d.theta - h, 0.3)) / 2e-3
print(rule, difference)

```

The two agree to the difference’s own error; each of the two shifted theta is one rebuild.

     On this page
