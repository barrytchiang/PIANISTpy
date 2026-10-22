# Radial velocity dispersion

Source: https://pianistpy.com/docs/guide/sigma_r

`pianistpy.sigma_r()`, `pianistpy.LosTable`

`sigma_r(source, r)` returns the intrinsic radial velocity dispersion \(\sigma_{\rm r}\) at user-specified 3D radii.
\(\sigma_{\rm r}^2\) is the second moment of the radial velocity over the model’s distribution function at a
fixed radius. As in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758), \(\sigma_{\rm r}^2\) is the velocity moment of the augmented density
\(\varrho_{\rm aug}\), integrated over the energy \(\mathcal{K}\) up to the relative potential
\(P = P(R)\) at the radius \(R\),

\[\sigma_{\rm r}^2(R) = \frac{1}{\varrho_{\rm aug}(P, R)} \int_0^{P} \varrho_{\rm aug}(\mathcal{K}, R)\,
{\rm d}\mathcal{K}, \qquad P = P(R).\]

For the anisotropy models of the package, the augmented density separates as
\(\varrho_{\rm aug}(P, R) = \varrho_{\rm aux}(P)\,g(R)\), with \(\varrho_{\rm aux}\) the auxiliary density
and \(g\) the anisotropy factor (`Anisotropy.factor`). \(\sigma_{\rm r}^2\) is then the Jeans integral of
\(\varrho_{\rm aux}\) over the potential. The radius \(R\) of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758) is `r` in the package.

## Arguments and units

- `source` is a `Model` (one component, or a superposition) or the model’s
line-of-sight table (`LosTable`), which holds the same intrinsic profile.

- `r` is an array of 3D radii in the model’s unit system. In the internal and the model systems, the unit of `r`
is the scale radius. After `model.to("physical", ...)`, `r` is in parsecs, or an Astropy quantity of any
length unit.

- The result has the shape of `r`, in the model’s velocity unit (km/s in the physical system).

The members of a superposition, Method II models on one truncated density, share that density, so the members’ second moments
add with the weights:
\(\sigma_{\rm r}^2 = \sum_k w_k \sigma_{{\rm r},k}^2\).

## Accuracy

The profile is tabulated in radius once per model. Radii are added to the profile until \(\sigma_{\rm r}\) at
the midpoint of every interval between the profile’s radii is within a relative `profile_tolerance`:
\(10^{-7}\) with the default settings and \(10^{-6}\) with `Settings(quick=True)`. The default value of
`profile_tolerance`, \(10^{-7}\), is a tenth of \(10^{-6}\), so the profile adds little to the error of the
projected quantities computed from the profile. As for every observable, a model whose relative error is above
\(10^{-2}\) raises `AccuracyError`, unless the model’s settings set `write_anyway=True`
([Accuracy](https://pianistpy.com/docs/guide/accuracy)).

## Example

```python
import numpy as np
import pianistpy as pp

# a Hernquist density with a constant anisotropy of 0.25, truncated at K_t = 0.05
model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
r = np.array([0.1, 1.0, 10.0])
pp.sigma_r(model, r)                          # internal units

physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.sigma_r(physical, [50.0, 500.0, 5000.0])   # parsecs in, km/s out
```
