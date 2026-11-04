# Distribution functions

Source: https://pianistpy.com/docs/guide/df

`pianistpy.compute_df()`, `pianistpy.DistributionFunction`

`compute_df(component)` returns the untruncated distribution function of a component as a
`DistributionFunction`. `compute_df` computes the distribution function by the analytic inversion
of the density. `compute_df` returns the distribution function even where the distribution function is negative or
not monotonic, and the result states the values of the energy-like variable \(\mathcal{K}\) at which the distribution function is negative or not monotonic.
The attribute `k_t_min` is the lowest truncation value \(\mathcal{K}_{\rm t}\) that `truncate` accepts. `k_t_min` is zero when the
distribution function is non-negative and non-decreasing everywhere and the model is not a Type II model. One result serves every Method I truncation of the
same component, so a scan over
truncation values needs one `compute_df` call.

```python
halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))
df = pp.compute_df(halo)
```

`DistributionFunction.from_function(component, f)` validates a distribution function from elsewhere, such as a
fitting function or the output of another code, with the checks of `compute_df`:

- `f(k, depth)` takes an array of values of \(\mathcal{K}\) in internal units and the array of their depths
below the center, \(P_0 - \mathcal{K}\). It returns the distribution function \(F\) at each \(\mathcal{K}\), not
its logarithm. The depth carries no rounding error at every \(\mathcal{K} \ge 1/2\), so a fit written in
the depth near the center can use `depth`. A function of \(\mathcal{K}\) alone may ignore `depth`.

- The function is tabulated on the \(\mathcal{K}\) grid that `compute_df` uses for the same component and settings. The
table is then continued beyond its ends, checked for sign and validated against the density, as for
`compute_df`.

- The function gives \(F\) only, so the validation report marks its check of the table’s logarithmic slope,
`slope_column`, as `"not judged"`.

- A file written from the result stores the table and the function’s name, not the function.

- A value that is not finite raises `InputError`, which states the first such \(\mathcal{K}\).

- `truncate` raises `InputError` for a distribution function from `from_function`, because a
truncation needs values of the inversion that a distribution function from `from_function` does not carry.

```python
import numpy as np
from scipy.special import beta

# the isotropic Plummer model's DF in closed form, F = A K^(7/2)
plummer = pp.Component(pp.DoublePowerLaw.plummer(), pp.Anisotropy())
amplitude = 3.0 / (4.0 * np.pi * np.sqrt(2.0) * beta(4.5, 1.5))
df = pp.DistributionFunction.from_function(plummer, lambda k, depth: amplitude * k**3.5)
```

`DistributionFunction.evaluate(k)` returns the distribution function at given values of \(\mathcal{K}\), in the
unit system of the result, through the package’s own lookup of the table and of its asymptotic forms. The optional
`depth`, \(P_0 - \mathcal{K}\) held to full precision, serves values near the center. A distribution function
that is negative somewhere raises `ResultError`.

```python
df.evaluate([0.1, 0.5, 0.9])                 # F(K) at three values of K, internal units
```

A `DistributionFunction` holds:

- the distribution function tabulated on its \(\mathcal{K}\) grid, with its logarithmic slope;

- the asymptotic forms of the distribution function at the center and at the outer end, which continue the
distribution function beyond the table;

- where it is negative or not monotonic, and `k_t_min`;

- the inputs and the settings it was computed from;

- its validation report and its identity string, a digest of the inputs, the settings and the package version.

A result can be written to a file and read back equal ([Input and output](https://pianistpy.com/docs/guide/input-output)). Only a `Model` can be
sampled or projected, and `truncate` builds the model from a distribution function
([Truncation in phase space](https://pianistpy.com/docs/guide/truncation)).

## Round trip

`pianistpy.round_trip()`, `pianistpy.RoundTrip`

`round_trip(df, radii=None, settings=None)` returns a `RoundTrip`: the density that the
distribution function gives, by the package’s own quadrature; the potential that this density generates, by Poisson’s
equation; and the relative difference between the density read at that potential and the analytic density, at each
radius of `radii` (by default the grid of the settings group `round_trip`). The steps are the same for a
distribution function from `compute_df` and from `from_function`, so a fitting function is compared with the
inverted distribution function on equal terms, as in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). `round_trip` takes an untruncated distribution
function of one component on an analytic profile; a Method II or a Type II distribution function raises
`InputError`, and a negative one `ResultError`. The result also holds the
residual of the potential, the enclosed mass, the closure before Poisson’s equation and the two accuracy budgets, and
can be written to a file and read back ([Input and output](https://pianistpy.com/docs/guide/input-output)).

```python
df = pp.compute_df(pp.Component(pp.DoublePowerLaw.nfw()))
trip = pp.round_trip(df, radii=[0.1, 1.0, 10.0])
trip.error                                    # relative difference of the density at each radius
```
