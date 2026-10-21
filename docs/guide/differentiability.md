# End-to-end differentiability

Source: https://pianistpy.com/docs/guide/differentiability

[`pianistpy.inference.likelihood()`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.likelihood), [`pianistpy.df.gradient.make_inversion_table()`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_table),
[`pianistpy.df.gradient.table_jacobian()`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.table_jacobian), [`pianistpy.df.gradient.make_inversion_rows()`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_rows),
[`pianistpy.df.gradient.rows_jacobian()`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.rows_jacobian)

## What is differentiable

- **The likelihood.** The log-likelihood that [`pianistpy.inference.likelihood()`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.likelihood) returns is a JAX function of
the components’ weights and of the model’s mass and length scales, which JAX traces and differentiates in each
weight and each scale. With `free=`, the anisotropy \((\beta_{\rm c}, R_{\rm a})\) of each component listed in
`free` is also a parameter of the log-likelihood (see [Which models](https://pianistpy.com/docs/guide/differentiability#which-models) below and the
[Inference](https://pianistpy.com/docs/reference/api/inference) page). With `free=`, data of the binned distributions
(`"losvd"`, `"losvd_aperture"`, `"pdf_v"`) or of the Gauss-Hermite moments (`"h3"`, `"h4"`) raise
`InputError`.

- **The observables.** `pianistpy.Differentiable` provides the observables of a one-component model as JAX
functions of the component’s anisotropy \((\beta_{\rm c}, R_{\rm a})\) ([Differentiable observables](https://pianistpy.com/docs/guide/differentiable)).

- **The distribution function.** The derivative of the distribution function in the central anisotropy
\(\beta_{\rm c}\) and the anisotropy radius \(R_{\rm a}\) is available at the inversion’s energies and for
the full table (see [The distribution function](https://pianistpy.com/docs/guide/differentiability#the-distribution-function) below).

The derivative of the distribution function is available for the following anisotropy models, with their parameters:

- isotropic and constant anisotropy \((\beta_{\rm c})\),

- Cuddeford-Osipkov-Merritt (COM) Type I (\(\beta_{\rm c}\) and \(R_{\rm a}\)),

- Osipkov-Merritt Type I and Type II \((R_{\rm a})\).

The following cases raise `InputError`:

- COM Type II. The COM Type II inversion has an interior point at the potential \(P(R_{\rm a})\), which moves
with both \(R_{\rm a}\) and \(\beta_{\rm c}\), so the derivative is not integrable at \(P(R_{\rm a})\).
The log-likelihood of `"sigma_los"` and `"sigma_los_aperture"` data still has a derivative in the anisotropy
of a free COM Type II component, computed from the Jeans equation. `Differentiable` raises
`InputError` for a COM Type II component. The distribution function of a COM Type II component
and the observables read from the distribution function (`losvd`, `aperture_losvd`, `pdf_v`) have no
derivative.

- A distribution function whose center lies on the critical branch, where the inner slope \(\gamma\) of the
density equals \(2\beta_{\rm c}\) and the form of the center changes with \(\beta_{\rm c}\).

- A component on a truncated profile that does not carry the distribution function of the model the profile was
taken from.

## Which models

The anisotropy is a parameter, through `likelihood` with `free=` and through `pp.Differentiable`, only for a
Method II model: a component whose distribution function is inverted on another model’s truncated profile. The
truncated profile is held fixed, so the profile and the potential do not change with the anisotropy, and the
derivative is complete. Other models raise `InputError`:

- A Method I truncated model. The truncated profile of a Method I model follows from the model’s own distribution
function, so the truncated profile changes with \(\beta_{\rm c}\) and \(R_{\rm a}\). Ver. 1.0.0 does not
compute the derivative of the truncated profile in the anisotropy.

- An untruncated model, or a model that is not spatially finite.

## The distribution function

The derivative of the distribution function is available in two forms:

- the rows (the distribution function on the inversion’s energy grid), through `make_inversion_rows` and
`rows_jacobian`, for a component on its own analytic profile only;

- the table, as the lookup reads it, through `make_inversion_table` and `table_jacobian`. The derivative covers
every array of the table: the rows (`log_df`, `log_df_slope`), the nodes below the table’s first energy
(`log_df_below`, `log_df_slope_below`), the closed form at the center (`central`) and the tail (`tail`).
For a Method II component, the arrays are the rows (`log_df`), the power law at the lowest energies
(`edge_law`) and the closed form at the center (`center`).

`make_inversion_rows` and `make_inversion_table` each return a JAX function of the anisotropy parameters, the
names of the parameters (`params`), and the energies or the table at the component’s own parameters. The value of
the returned function is the result of `compute_df()` at the given parameters, computed on the CPU.
The derivative of the returned function is the package’s analytic derivative (Eq. (T:7.1) of the
[Technical notes](https://pianistpy.com/docs/reference/notes)). The `*_jacobian` functions return the values and their derivatives
as arrays.

Both returned functions hold the energies at the component’s own parameters. The function that
`make_inversion_table` returns makes the anchor of the closed form at the center and the nodes kept again at each
new parameter value, except that a component on a truncated profile also holds the nodes kept at the component’s
own parameters. The derivative is thus the derivative of the table that `compute_df()` returns at the
given parameters, on the held energies. Where a discrete choice would change between two parameter values, the value
of the returned function differs from a new `compute_df()` call by up to the tolerance of the
inversion.

## Accuracy and the record

The derivative is exact except for the derivative of the inversion’s operator in \(\beta_{\rm c}\). The
derivative of the operator in \(\beta_{\rm c}\) has no closed form and is taken by finite differences of the
operator’s values at nearby \(\beta_{\rm c}\). `rows_jacobian` and `table_jacobian` return a record of the
finite differences:

- the kind of difference: central (a Richardson quotient) where the inversion’s form is the same on both sides of
\(\beta_{\rm c}\), or one-sided at a \(\beta_{\rm c}\) where the form changes (integer and half-integer
\(\beta_{\rm c}\));

- the step of the difference;

- the Richardson residual of the quotient and, for a one-sided difference, the spread between the derivatives from
the two sides, relative to the largest \(|\partial F/\partial\beta_{\rm c}|\) in the record of
`rows_jacobian` and absolute in the record of `table_jacobian`;

- for a component on a truncated profile, where the component’s energy grid is placed.

The Richardson residual is the accuracy of the derivative, read against the thresholds \(10^{-6}\),
\(10^{-4}\) and \(10^{-2}\) of the validation reports ([Accuracy](https://pianistpy.com/docs/guide/accuracy)). Where the difference is central, the
residual is below \(10^{-6}\). At the isotropic value \(\beta_{\rm c} = 0\), the difference is one-sided, and
the residual is about \(10^{-5}\) of the largest derivative, between \(10^{-6}\) and \(10^{-4}\). The
record gives the residual. A gradient at \(\beta_{\rm c} = 0\) is usable at an error of about \(10^{-5}\).
For a gradient with an error below \(10^{-6}\), take a \(\beta_{\rm c}\) away from 0, where the difference
is central.

## Example

The derivative of \(\log_{10} F\) at one node of a Hernquist component’s table, by the analytic derivative and by
a central difference of the same function:

```python
import jax
import numpy as np

import pianistpy as pp
from pianistpy.df.gradient import make_inversion_table

halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25))
table, params, values = make_inversion_table(halo, pp.Settings())
node = int(np.searchsorted(values["k"], 0.5))  # the first node at or above K = 0.5

def log_df(theta):
    return table(theta)["log_df"][node]

theta = np.array([0.25])  # params == ("beta_c",)
rule = float(jax.grad(log_df)(theta)[0])
h = 1e-3
difference = float(log_df(theta + h) - log_df(theta - h)) / (2 * h)
print(rule, difference)
```

The analytic derivative and the central difference agree to the error of the central difference, of order
\(h^2\) times the third derivative. The table is made once. Each
call of `table` solves the inversion again at the new parameters, on the held grid.
