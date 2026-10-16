# The prepared table: LosTable

Source: https://pianistpy.com/docs/guide/lostable.html

# The prepared table: LosTable

pianistpy.LosTable, pianistpy.losvd(), pianistpy.sigma_los(),
pianistpy.sigma_los_aperture(), pianistpy.sigma_r(), pianistpy.sigma_t(), pianistpy.pdf_v()

LosTable(model) computes once what the observables of a model read: the intrinsic dispersion profile
(Radial velocity dispersion), the surface density \(\Sigma(R_{\rm 2D})\) and the line-of-sight dispersion
\(\sigma_{\rm los}(R_{\rm 2D})\) over projected radius, and the table of line-of-sight velocity distributions
\(D(R_{\rm 2D}, V_{\rm los})\) (Line-of-sight velocity distribution) on rows of projected radius from the center to the model’s edge.
Every observable then takes the table in place of the model and reads from it, so a table is the object to build
when a model is asked for more than one observable, or for one observable many times (a fit, a grid of apertures).

As in the paper, the dispersion inside an aperture is the surface-density weighted average of
\(\sigma_{\rm los}^2\); a centered circle of radius \(R_{\rm ape}\) gives

\[\overline{\sigma}_{\rm los}^2(R_{\rm ape}) =
\frac{\int_0^{R_{\rm ape}} \Sigma(R_{\rm 2D})\,\sigma_{\rm los}^2(R_{\rm 2D})\,R_{\rm 2D}\,{\rm d}R_{\rm 2D}}
{\int_0^{R_{\rm ape}} \Sigma(R_{\rm 2D})\,R_{\rm 2D}\,{\rm d}R_{\rm 2D}},\]

and an offset circle or a rectangle the same average over its own area. Given the model,
sigma_los_aperture() takes this average and reads no velocity distribution; given the table, it
takes the second moment of the aperture’s velocity distribution, the rows of the table averaged over the aperture
with the same weight. The two routes are independent and agree to the accuracy goal.

## Arguments and units

LosTable(model) takes a Model of one component or a superposition; a superposition holds
one table per component (parts), each read with its weight. The table is computed in internal units at the
model’s own settings and is not recomputed for another unit system: table.to("physical", ...) returns the
same tables presented in the units of model.to("physical", ...), and the observables read from it take and
return quantities in those units.

The rows are computed by a pool of worker processes, which Python starts by spawning a fresh interpreter. In a
script, build the table (or call losvd() on a model, which builds one) under
if __name__ == "__main__":, as Python’s multiprocessing requires; in a notebook or an interactive session
nothing is needed.

What it holds: model; profile, projected, kept and table, the results of its four stages;
validation, its report; parts, one table per component.

## Tiers

The rows of the table are added until the table reads itself between rows within the tolerance of its settings,
and every row is checked against two quantities computed without it. The report table.validation holds four
checks, each with its relative error and the radius where it is worst: norm, each row’s integral against the
surface density; second_moment, each row’s second moment against the projected dispersion; unjudged, a row
with surface density that could not be checked (beyond the hard limit if any); and certificate, the table’s
estimate of its reading between rows. The report’s tier is the worst of these and of the model’s own; a table
above the goal says so in a warning when it is built (Accuracy). Intervals of projected radius that the
table could not certify are recorded in table.table.r_uncertified, and a request inside one is refused by name.

## Example

```
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=1.0)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
table = pp.LosTable(model)                     # built once
table.validation.tier                          # "goal", "ceiling" or "warning"
{key: check.error for key, check in table.validation.checks.items()}

circle = pp.Aperture.circle(1.0)
pp.sigma_los_aperture(model, circle), pp.sigma_los_aperture(table, circle)   # the two routes
pp.sigma_los(table, [0.1, 1.0, 10.0])
pp.losvd(table, 1.0, [0.0, 0.5, 1.0])

physical = table.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.sigma_los_aperture(physical, pp.Aperture.circle(500.0))                   # km/s, 500 pc circle

```

     On this page
