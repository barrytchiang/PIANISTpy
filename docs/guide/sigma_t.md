# Tangential velocity dispersion

Source: https://pianistpy.com/docs/guide/sigma_t.html

# Tangential velocity dispersion

pianistpy.sigma_t(), pianistpy.LosTable

sigma_t(source, r) returns the intrinsic tangential velocity dispersion \(\sigma_t\) at 3D radii, of both
tangential components together, \(\sigma_t^2 = \sigma_\theta^2 + \sigma_\phi^2\). As in the paper, it is the
tangential velocity moment of the augmented density,

\[\sigma_t^2(R) = \frac{2}{\varrho_{\rm aug}(P, R)} \int_0^{P}
\frac{\partial}{\partial R^2}\Big[R^2 \varrho_{\rm aug}(\mathcal{K}, R)\Big]\,{\rm d}\mathcal{K},
\qquad P = P(R),\]

so that the anisotropy is \(\beta_\sigma = 1 - \sigma_t^2 / (2\sigma_r^2)\): zero for an isotropic model,
\(\beta_{\rm c}\) for a constant anisotropy, and \(R^2 / (R^2 + R_{\rm a}^2)\) for an Osipkov-Merritt
Type I model. The package evaluates it as \(\sigma_t^2 = 2(1 - \beta_\sigma)\sigma_r^2\), with
\(\sigma_r\) from Radial velocity dispersion and \(\beta_\sigma\) from the model’s anisotropy in closed form.

## Arguments and units

As for Radial velocity dispersion: source is a Model or its LosTable, r is in
the model’s unit system, and the result has the shape of r in the model’s velocity unit. Of a superposition,
\(\sigma_t^2 = \sum_k w_k \sigma_{t,k}^2\).

## Tiers

Those of Radial velocity dispersion, whose profile it reads: the anisotropy factor is exact.

## Example

The anisotropy read back from the two dispersions is the model’s own, here Osipkov-Merritt with
\(R_{\rm a} = 1\).

```
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=1.0)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
r = np.array([0.1, 1.0, 10.0])
beta = 1.0 - pp.sigma_t(model, r) ** 2 / (2.0 * pp.sigma_r(model, r) ** 2)
np.max(np.abs(beta - r**2 / (r**2 + 1.0)))   # rounding

```

     On this page
