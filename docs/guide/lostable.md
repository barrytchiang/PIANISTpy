# Line-of-sight table

Source: https://pianistpy.com/docs/guide/lostable

`pianistpy.LosTable`, `pianistpy.losvd()`, `pianistpy.sigma_los()`,
`pianistpy.sigma_los_aperture()`, `pianistpy.sigma_r()`, `pianistpy.sigma_t()`, `pianistpy.pdf_v()`

`LosTable(model)` computes, once, everything the observables of a model read:

- the intrinsic dispersion profile ([Radial velocity dispersion](https://pianistpy.com/docs/guide/sigma_r));

- the surface density \(\Sigma(R_{\rm 2D})\) and the line-of-sight dispersion
\(\sigma_{\rm los}(R_{\rm 2D})\) over projected radius;

- the line-of-sight velocity distributions \(D(R_{\rm 2D}, V_{\rm los})\) ([Line-of-sight velocity distribution](https://pianistpy.com/docs/guide/losvd)), one row per projected
radius, from the center outward.

Every observable then takes the table in place of the model and reads its values from the table. Build a table when a model is asked
for more than one observable, or for one observable many times (a fit, a grid of apertures).

As in [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), the dispersion inside an aperture is the surface-density weighted average of
\(\sigma_{\rm los}^2\). A centered circle of radius \(R_{\rm ape}\) gives

\[\overline{\sigma}_{\rm los}^2(R_{\rm ape}) =
\frac{\int_0^{R_{\rm ape}} \Sigma(R_{\rm 2D})\,\sigma_{\rm los}^2(R_{\rm 2D})\,R_{\rm 2D}\,{\rm d}R_{\rm 2D}}
{\int_0^{R_{\rm ape}} \Sigma(R_{\rm 2D})\,R_{\rm 2D}\,{\rm d}R_{\rm 2D}},\]

and an offset circle or a rectangle gives the same average over its own area.
`sigma_los_aperture()` computes the dispersion inside an aperture in two ways:

- Given the model, `sigma_los_aperture` computes the surface-density weighted average of
\(\sigma_{\rm los}^2\) and reads no velocity distribution.

- Given the table, `sigma_los_aperture` takes the second moment of the aperture’s velocity distribution, which is
the table’s rows averaged over the aperture with the surface density as the weight.

The results on the model and on the table agree to a relative error of \(10^{-6}\). The call on the table
takes longer than the call on the model.

## Arguments and units

`LosTable(model)` takes a `Model` of one component or a superposition
(`superpose()`). For a superposition it holds one table per component in `parts`, and reads each
with its component’s weight.

The table is computed in internal units, at the model’s own settings. `table.to("physical", ...)` recomputes
nothing. `table.to` returns the same tables presented in the units of `model.to("physical", ...)`, and the
observables read from the returned table take and return quantities in those units.

The rows are computed in parallel processes. In a script, build the table under `if __name__ == "__main__":`, as
Python’s `multiprocessing` requires. A call of `losvd()` or `aperture_losvd()` on a
model, and a `ProjectedSampler` given a model, also build a table and need the same
`if __name__ == "__main__":` guard. A notebook or an interactive session needs no `if __name__ == "__main__":`
guard. If a parallel process ends during the computation, the call raises `pianistpy.exceptions.WorkerError` at
once and states the cause. The usual cause is that the machine ran out of memory. Fewer cores
(`set_cores()`) lower the memory that the call uses.

The table holds:

- `model`, the model the table was built from;

- `profile` (the intrinsic dispersion profile), `projected` (the surface density and the line-of-sight
dispersion over projected radius), `kept` (the ranges of projected radius of the velocity quadrature) and
`table` (the line-of-sight velocity distributions), each of the first component for a superposition;

- `validation`, the table’s accuracy report;

- `parts`, one table per component.

## Where the rows run

The rows stop at the first row beyond which less than `losvd.outer_mass_share` of the model’s total projected
mass lies. The default is \(10^{-6}\). `Settings(losvd={"outer_mass_share": 0})` keeps the
rows to the model’s edge.

By default the rows start at the innermost radius of the dispersion profile. The setting
`losvd.innermost_read_radius` sets the smallest projected radius the table will be read at. The rows then start
at the last row inside the projected radius whose disk holds \(10^{-6}\) times the projected mass inside
`innermost_read_radius`. An aperture’s integral then starts at the table’s first row. Each row kept is identical
to the row at the same projected radius of a table built with `innermost_read_radius` and `outer_mass_share`
set to 0.

A read outside the rows raises `InputError`. For a read beyond the last row or inside the first
row, at a point or over an aperture, the error message gives the radius of the end row and the setting that moves
the end row (`losvd.outer_mass_share` or `losvd.innermost_read_radius`). An aperture whose part inside the
first row carries more than \(10^{-6}\) of the aperture’s weight also raises `InputError`.
The error message gives the radius of the smallest centered circle that the table computes to a relative error of
\(10^{-6}\).

## Accuracy report

The table adds rows until the interpolation between rows meets the tolerance of the model’s settings. Every row is
checked against the surface density and the line-of-sight dispersion, both computed without the row. The report `table.validation` holds six checks, each with its relative error
and the radius where it is worst:

- `norm`: each row’s integral against the surface density;

- `second_moment`: each row’s second moment against the projected dispersion;

- `unjudged`: infinite if a row where the surface density is positive could not be checked, and 0 otherwise;

- `t_converged`: infinite if the integral of a row along the line of sight did not converge, and 0 otherwise;

- `certificate`: the table’s estimate of its error between rows;

- `layer_floor`: the error floor of the integration toward zero angular momentum. The floor is nonzero only for a
central anisotropy \(\beta_{\rm c}\) between 1/2 and 1, and a floor above \(10^{-6}\) raises
`ResultError`.

The report’s tier is the band of the worst relative error among the six checks and the model’s own report
([Accuracy](https://pianistpy.com/docs/guide/accuracy)). Building a table whose relative error is above \(10^{-4}\) issues an
`AccuracyWarning`. A table whose relative error is above \(10^{-2}\) (tier `"refused"`)
raises `AccuracyError`, unless the model’s settings set `write_anyway=True`. An aperture call
on a table (`aperture_losvd`, `sigma_los_aperture`) warns above \(10^{-4}\) and raises above
\(10^{-2}\) in the same way for the aperture’s distribution. `table.table.r_uncertified` records the
intervals of projected radius where the table could not meet the table’s tolerance. A read inside one of the
intervals in `table.table.r_uncertified` raises `ResultError`, whose message gives the
interval.

## Example

```python
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
table = pp.LosTable(model)                     # built once
table.validation.tier                          # "silent", "recorded", "warned" or "refused"
{key: check.error for key, check in table.validation.checks.items()}

circle = pp.Aperture.circle(1.0)
pp.sigma_los_aperture(model, circle), pp.sigma_los_aperture(table, circle)   # on the model and on the table
pp.sigma_los(table, [0.1, 1.0, 10.0])
pp.losvd(table, 1.0, [0.0, 0.5, 1.0])

physical = table.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.sigma_los_aperture(physical, pp.Aperture.circle(500.0))                   # km/s, 500 pc circle
```

The [Representative computing time](https://pianistpy.com/docs/start/computing-time) page gives the time of a table’s first build and
of the calls that read the table, measured on a workstation CPU, with a GPU, and on a laptop.
