# Tangential velocity dispersion

Source: https://pianistpy.com/docs/guide/sigma_t

# Tangential velocity dispersion

pianistpy.sigma_t(), pianistpy.LosTable

sigma_t(source, r) returns the intrinsic tangential velocity dispersion \(\sigma_{\rm t}\) at 3D radii, of both
tangential components together, \(\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2\). As in Chiang et al. (2026d), it is the
tangential velocity moment of the augmented density,

\[\sigma_{\rm t}^2(R) = \frac{2}{\varrho_{\rm aug}(P, R)} \int_0^{P}
\frac{\partial}{\partial R^2}\Big[R^2 \varrho_{\rm aug}(\mathcal{K}, R)\Big]\,{\rm d}\mathcal{K},
\qquad P = P(R),\]

so that the anisotropy is \(\beta_\sigma = 1 - \sigma_{\rm t}^2 / (2\sigma_{\rm r}^2)\): zero for an isotropic model,
\(\beta_{\rm c}\) for a constant anisotropy, and \(R^2 / (R^2 + R_{\rm a}^2)\) for an Osipkov-Merritt
Type I model. The package evaluates it as \(\sigma_{\rm t}^2 = 2(1 - \beta_\sigma)\sigma_{\rm r}^2\), with
\(\sigma_{\rm r}\) from Radial velocity dispersion and \(\beta_\sigma\) from the model’s anisotropy in closed form.

## Arguments and units

As for Radial velocity dispersion: source is a Model or its LosTable, r is in
the model’s unit system, and the result has the shape of r in the model’s velocity unit. Of a superposition,
\(\sigma_{\rm t}^2 = \sum_k w_k \sigma_{{\rm t},k}^2\).

## Tiers

Those of Radial velocity dispersion, whose profile it reads: the anisotropy factor is exact.

## Example

The anisotropy read back from the two dispersions is the model’s own, here constant with
\(\beta_{\rm c} = 0.25\).

```
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
r = np.array([0.1, 1.0, 10.0])
beta = 1.0 - pp.sigma_t(model, r) ** 2 / (2.0 * pp.sigma_r(model, r) ** 2)
np.max(np.abs(beta - 0.25))   # rounding

```

     On this page
