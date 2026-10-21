# Truncation in phase space

Source: https://pianistpy.com/docs/guide/truncation

`pianistpy.Truncation`, `pianistpy.truncate()`, `pianistpy.solve()`, `pianistpy.Model`

## Method I

`truncate(df, truncation)` returns a `Model`, which holds the distribution function truncated at an
energy and the self-consistent potential-density pair of the truncated distribution function. If the untruncated
distribution function is everywhere non-negative and monotonic, the truncated distribution function inherits both
properties, and the truncation boundary is in dynamical equilibrium. A request with no physical model raises
`UnphysicalModelError`, which carries the distribution function, or the model without an edge, for
inspection.

At \(\beta_{\rm c} = -1/2\) and a steep inner density slope \(\gamma\), the iteration for the center of the
truncated model does not converge, and `truncate` raises `AccuracyError`. For example, with
\((\alpha, \beta) = (2, 4)\), \(\beta_{\rm c} = -1/2\) and \(\mathcal{K}_{\rm t} = 0.3\), `truncate`
raises `AccuracyError` from \(\gamma = 1.735\) upward.

`Truncation` takes exactly one of three inputs:

```python
pp.Truncation(k_t=0.2)                          # the truncation energy, in internal units
pp.Truncation(mass_t=1.0, concentration=10.0)   # a target mass, in units of M_0 (model units)
pp.Truncation(r_t=5.0)                          # a target radius, in units of the scale radius
```

A target mass is given in units of \(M_0\), so it needs the concentration that defines \(M_0\). A target
radius or a truncation energy needs no concentration, and a concentration given with `r_t` or `k_t` raises
`InputError`.

`solve(component, truncation)` runs `compute_df` and `truncate` in one call. Without a `Truncation`,
`truncate` and `solve` return the untruncated model, or, for a Method II component, the model that ends at the
edge of the truncated density that the component uses.

A model holds:

- the truncated distribution function;

- the truncated density, potential and enclosed mass on one radial grid (`model.profile.r`, `.rho_trunc`,
`.p_trunc`, `.mass_trunc`);

- the truncation radius and the total mass;

- the validation report and the identity string, a digest of the inputs, the settings and the package version.

An untruncated model is not spatially finite. The untruncated model’s `spatially_finite` is false, and the
untruncated model’s truncation radius `r_t` is infinite. A function that needs a spatially finite model, such as
`Sampler` without `r_sample`, raises `InputError` for a model that is not spatially finite
([Flags and refusals](https://pianistpy.com/docs/guide/flags-refusals)).

### Type II models

A Type II model (Osipkov-Merritt or Cuddeford-Osipkov-Merritt) must end inside its anisotropy radius
\(R_{\rm a}\), where the model’s density \(\varrho_{\rm aux} / (1 - R^2/R_{\rm a}^2)\), with
\(\varrho_{\rm aux}\) the auxiliary density, has a pole.

- The truncation energy must satisfy \(\mathcal{K}_{\rm t} > P(R_{\rm a})\). Where this bound sets `k_t_min`,
`k_t_min` is the next floating-point number above \(P(R_{\rm a})\).

- The bound is necessary, not sufficient. The bound keeps inside \(R_{\rm a}\) the radius at which the
untruncated potential equals \(\mathcal{K}_{\rm t}\), but the truncated model’s own edge lies farther out than
that radius. Where the truncated model’s edge would reach \(R_{\rm a}\), `truncate` raises
`UnphysicalModelError` with a message that begins `TYPE II EDGE` and states the potential at
\(R_{\rm a}\) or the radius of the edge. A larger \(\mathcal{K}_{\rm t}\) moves the truncated model’s edge
inside \(R_{\rm a}\).

### Osipkov-Merritt Type I models with a small anisotropy radius

An Osipkov-Merritt Type I model with a small \(R_{\rm a}\) and a low \(\mathcal{K}_{\rm t}\) can have no edge
inside the largest radius at which `truncate` solves for the potential. `truncate` then raises
`UnphysicalModelError` with a message that begins `NO EDGE`. On the Hernquist density with the
quick settings, `truncate` raises `UnphysicalModelError` (`NO EDGE`) for the following
Osipkov-Merritt Type I models:

- \(R_{\rm a} = 1\) at every \(\mathcal{K}_{\rm t}\) from 0.005 to 0.3;

- \(R_{\rm a} = 2\) below \(\mathcal{K}_{\rm t} = 0.2\);

- \(R_{\rm a} = 5\) below \(\mathcal{K}_{\rm t} = 0.03\);

- \(R_{\rm a} = 10\) at \(\mathcal{K}_{\rm t} = 0.005\).

The Osipkov-Merritt Type I models on the Hernquist density with \(R_{\rm a}\) from 1 to 10 and
\(\mathcal{K}_{\rm t}\) from 0.005 to 0.3 that `truncate` builds have the tier `"recorded"`, above
\(10^{-6}\) and at most \(10^{-4}\) ([Accuracy](https://pianistpy.com/docs/guide/accuracy)). The check `high_k_slope_mismatch` of the distribution
function of each built model is \(3.4 \times 10^{-6}\), at \(\mathcal{K} = 1\).

## Method II

Method II gives a new anisotropy to a density that is already truncated. The new component takes the truncated
profile of a Method I model as its profile. `solve`, or `compute_df` and `truncate`, without a
`Truncation`, build the Method II model from the new component.

```python
truncated = pp.solve(pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy()), pp.Truncation(k_t=0.2))
model = pp.solve(pp.Component(truncated.profile, pp.Anisotropy(beta_c=-0.25)))
```

A Method II model exists only when its anisotropy is compatible with the edge of the Method I model’s truncated
density. Where that density goes as \(\hat{P}^q\) at the edge, with \(\hat{P}\) the Method I model’s truncated
relative potential, which is zero at the edge, the central anisotropy of the new component must satisfy
\(\beta_{\rm c} > 1/2 - q\). A `Component` whose \(\beta_{\rm c}\) is not above \(1/2 - q\) raises
`UnphysicalModelError` when the `Component` is created, and the error message states the bound.
The [Technical notes](https://pianistpy.com/docs/reference/notes) derive the condition \(\beta_{\rm c} > 1/2 - q\).
