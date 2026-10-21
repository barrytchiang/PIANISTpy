# The package

Source: https://pianistpy.com/docs/reference/api/pianistpy

In PIANISTpy, a model (Model) is the equilibrium of one or more
components that share one total potential, untruncated or truncated in
phase space. A component (Component) is a density profile, such as a
DoublePowerLaw, with a velocity anisotropy (Anisotropy). solve
computes a model from one component: the component’s distribution
function, and the self-consistent density, mass and potential of the
model ([Chiang+(2026d)](https://pianistpy.com/docs/project/citing)).

Every observable is computed from a model. Sampler draws particles.
sigma_r and sigma_t give the radial and the tangential velocity
dispersion at 3D radii, sigma_los the line-of-sight velocity dispersion
at projected radii, and losvd the line-of-sight velocity distribution.
likelihood compares a model with observed data (Data).

A model, a distribution function, a LosTable and a sampler each carry a
validation report (pianistpy.validation). The report’s tier, such as
`model.validation.tier`, is the worst band among the relative errors
that the report’s checks measured and the tiers of the results it was
computed from: `"silent"` (at most \(10^{-6}\)), `"recorded"`
(at most \(10^{-4}\)), `"warned"` (at most \(10^{-2}\)) or
`"refused"` (above \(10^{-2}\), or not a number).

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> model.validation.tier
'silent'
```

Every public class and function of the package is documented on the page of its subpackage
([Functions by subpackage](https://pianistpy.com/docs/reference/api/index)).
