# Tangential velocity dispersion

Source: https://pianistpy.com/docs/guide/sigma_t

`pianistpy.sigma_t()`, `pianistpy.LosTable`

`sigma_t(source, r)` returns the intrinsic tangential velocity dispersion \(\sigma_{\rm t}\) at user-specified 3D radii.
\(\sigma_{\rm t}^2\) is the sum of the squared dispersions of the two tangential velocity components,
\(\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2\), with \(\theta\) and \(\phi\) the polar and the
azimuthal directions. As in [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), \(\sigma_{\rm t}^2\) is the tangential velocity moment of the augmented
density,

\[\sigma_{\rm t}^2(R) = \frac{2}{\varrho_{\rm aug}(P, R)} \int_0^{P}
\frac{\partial}{\partial R^2}\Big[R^2 \varrho_{\rm aug}(\mathcal{K}, R)\Big]\,{\rm d}\mathcal{K},
\qquad P = P(R).\]

The velocity anisotropy is \(\beta_\sigma = 1 - \sigma_{\rm t}^2 / (2\sigma_{\rm r}^2)\). \(\beta_\sigma\)
is zero for an isotropic model, \(\beta_{\rm c}\) for a constant-anisotropy model, and
\(R^2 / (R^2 + R_{\rm a}^2)\) for an Osipkov-Merritt Type I model. The package evaluates \(\sigma_{\rm t}^2 = 2(1 - \beta_\sigma)\sigma_{\rm r}^2\),
with \(\sigma_{\rm r}\) from [Radial velocity dispersion](https://pianistpy.com/docs/guide/sigma_r) and \(\beta_\sigma\) in closed form from the model’s anisotropy.

## Arguments and units

As for [Radial velocity dispersion](https://pianistpy.com/docs/guide/sigma_r), `source` is a `Model` or the model’s line-of-sight table
(`LosTable`), `r` is in the model’s unit system, and the result has the shape of `r`, in the
model’s velocity unit. For a superposition,
\(\sigma_{\rm t}^2 = \sum_k w_k \sigma_{{\rm t},k}^2\).

## Accuracy

`sigma_t` has the accuracy of `sigma_r` ([Radial velocity dispersion](https://pianistpy.com/docs/guide/sigma_r)), because `sigma_t` reads the radial dispersion
profile of `sigma_r`. The factor \(2(1 - \beta_\sigma)\) is computed in closed form from the model’s
anisotropy.

## Example

The anisotropy computed from the two dispersions is the model’s own, here the constant
\(\beta_{\rm c} = 0.25\).

```python
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
r = np.array([0.1, 1.0, 10.0])
beta = 1.0 - pp.sigma_t(model, r) ** 2 / (2.0 * pp.sigma_r(model, r) ** 2)
np.max(np.abs(beta - 0.25))   # of the order of the floating-point rounding error
```
