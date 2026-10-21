# Flags and refusals

Source: https://pianistpy.com/docs/guide/flags-refusals

A result with an accuracy warning is returned with an `AccuracyWarning`, and the result’s
validation report has the tier `"warned"`. A refusal is an error that the package raises instead of returning a
result. The message of each error states the cause, and each section below gives the exception class.

## The distribution function is always returned

`compute_df` returns the distribution function of every component for which the inversion is defined, whatever
the sign of the distribution function. The result states where the distribution function is negative. The result
also states the energy at which the distribution function crosses zero, the distribution function’s minimum, and
`k_t_min`, the lowest truncation energy that gives a valid equilibrium.

## A model is always a valid equilibrium

`truncate` and `solve` return a `Model` only where one exists. A request with no physical
equilibrium raises `UnphysicalModelError`. The exception’s `result` attribute holds the
distribution function, or the model without an edge where the truncation gives no edge, so the distribution
function need not be recomputed. A Method II `Component` takes the truncated density of a Method I model. A Method
II `Component` whose central anisotropy \(\beta_{\rm c}\) violates the existence condition at the edge of that
truncated density raises
`UnphysicalModelError` when the `Component` is created. The message states the bound that
\(\beta_{\rm c}\) must exceed.

## A model that is not spatially finite is a model

An untruncated model, which `truncate` and `solve` return without a `Truncation`, has a density that does not
reach zero at a finite radius. Such a model is returned with `spatially_finite` false, an infinite `r_t` and a
finite total mass `mass_t`. A truncation whose model has no edge raises `UnphysicalModelError`,
and so does an untruncated model whose total mass diverges. A model that is not spatially finite raises no error
and issues no `AccuracyWarning` for that property. The calls that need a spatially finite model
act as follows:

- a `Truncation` by target mass or target radius that no spatially finite model reaches raises
`UnphysicalModelError`, whose message states the value reached at the limit of the search;

- sampling requires an explicit sampling radius, and records the mass inside it;

- an observable at a projected radius beyond the outermost projected radius of the model’s line-of-sight table
raises `InputError`, and the message states the radius that bounds the table.

## A distribution function that diverges at the truncation energy

A Method II model whose central anisotropy \(\beta_{\rm c}\) lies more than 1 below the \(\beta_{\rm c}\) of
the Method I model whose truncated density the Method II model uses has a distribution function that diverges,
integrably, at the truncation energy. The package returns the Method II model and writes a warning to the package’s log. The warning gives the
exponent and says the model is a valid equilibrium of finite density whose stability is not established.

## One component

In ver. 1.0.0, `compute_df` and `solve` take one `Component` and raise `InputError` for a
sequence of more than one `Component`. A model of several components comes only from `superpose`, which
combines models that share one truncated density and potential. Several components with different density
profiles in one potential, and tracer populations in a fixed potential, are on the
[Upcoming features](https://pianistpy.com/docs/project/upcoming-features) page.

## The attributes to read first in a result

- `validation`: the report and its tier;

- `spatially_finite`;

- `r_t` and `mass_t`: the model’s outer radius and total mass, in the model’s unit system;

- `identity`: what the result was computed from;

- `units`: the unit system in which the result is presented;

- `timing`: the wall-clock time of each step of the computation, in seconds;

- in a file that `write` produces, the hardware (CPU or GPU) and the software versions of the session that wrote
the file.
