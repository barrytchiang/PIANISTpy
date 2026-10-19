# Observables

Source: https://pianistpy.com/docs/guide/observables

# Observables

pianistpy.sigma_los(), pianistpy.sigma_los_aperture(), pianistpy.aperture_losvd(),
pianistpy.Aperture, pianistpy.LosTable

Observables are functions of a model. A single request computes only what it needs; for repeated requests on one
model, the prepared object LosTable(model) tabulates the line-of-sight velocity distributions once and every
observable reads from it.

```
sigma = pp.sigma_los(model, [0.1, 1.0, 10.0])        # at projected radii

circle = pp.Aperture.circle(r_ape=1.0, r_off=0.0)
slit = pp.Aperture.rectangle(side_x=2.0, side_y=0.2, r_off=0.0, angle=0.0)
pp.sigma_los_aperture(model, circle)                 # averaged over an aperture
pp.sigma_los_aperture(model, slit)

table = pp.LosTable(model)                           # once, for many requests
pp.sigma_los(table, [0.1, 1.0, 10.0])
pp.sigma_los_aperture(table, circle)
pp.aperture_losvd(table, circle, [0.0, 0.1, 0.2])    # the distribution in the aperture

```

An Aperture is circular or rectangular, centered or offset, the rectangle at an angle; every aperture call
takes the same object. The intrinsic radial and tangential dispersions and the line-of-sight velocity
distribution itself are available in the same way; their public names are listed in the
API reference. The distribution’s Gauss-Hermite moments (h3, h4)
enter the inference as data (pianistpy.inference.Data); they are not an observable call of their own.

Pairs of projected radius and line-of-sight velocity are drawn by ProjectedSampler on an annulus and, if
given, in a window of velocity, exactly conditioned: the radius carries the window’s share of the distribution
at each radius, for every anisotropy. The ranges and the pairs are in the source’s unit system, as for the other
observables; the tables it builds carry their errors in its report.

```
from pianistpy.observables import ProjectedSampler

sampler = ProjectedSampler(table, R_range=(0.1, 3.0), v_range=(-0.2, 0.3))
pairs = sampler.sample(1_000_000, seed=1)   # pairs.r_2d, pairs.v_los, pairs.mass

```

An aperture of arbitrary shape is not an exact route: the notebook of the same name shows how a mask on
projected draws gives it, with the statistical error printed beside every number (Examples).

One page per call, each with its definition, its arguments and units, the tiers it reports and an example:

Radial velocity dispersion (sigma_r)

Tangential velocity dispersion (sigma_t)

Line-of-sight velocity distribution (losvd)

Line-of-sight velocity distribution in an aperture (aperture_losvd)

Local speed distribution (pdf_v)

The prepared table (LosTable)

End-to-end differentiability

Differentiable observables (Differentiable)
