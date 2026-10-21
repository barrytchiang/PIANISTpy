# Line-of-sight velocity distribution in an aperture

Source: https://pianistpy.com/docs/guide/aperture_losvd

`pianistpy.aperture_losvd()`, `pianistpy.Aperture`, `pianistpy.LosTable`

`aperture_losvd(source, aperture, v)` returns the line-of-sight velocity distribution of the model’s mass inside
an aperture. \(\overline{D}\) is the line-of-sight velocity distribution \(D\) at each projected radius
([Line-of-sight velocity distribution](https://pianistpy.com/docs/guide/losvd)), averaged over the aperture with the surface density as the weight:

\[\overline{D}(V_{\rm los}) = \frac{\int_{\rm aperture} \Sigma(R_{\rm 2D})\, D(R_{\rm 2D}, V_{\rm los})\,{\rm d}A}
{\int_{\rm aperture} \Sigma(R_{\rm 2D})\,{\rm d}A}.\]

The distribution is even in \(V_{\rm los}\) and integrates to one over all \(V_{\rm los}\).
\(\overline{D}\) is zero at every \(|V_{\rm los}|\) above the largest velocity that the distribution function
reaches on the lines of sight through the aperture. The second moment of \(\overline{D}\) is the squared
dispersion inside the aperture, \(\overline{\sigma}_{\rm los}^2\). Given the same `LosTable`,
`pianistpy.sigma_los_aperture()` returns \(\overline{\sigma}_{\rm los}\).

## Arguments and units

- `source` is a `Model` or the model’s line-of-sight table (`LosTable`).
Given a model, `aperture_losvd` builds the model’s `LosTable` at every call. For more than one call, build
the `LosTable` once ([Line-of-sight table](https://pianistpy.com/docs/guide/lostable)).

- `aperture` is an `Aperture`: circular or rectangular, centered or offset ([Observables](https://pianistpy.com/docs/guide/observables)).
The aperture’s lengths are in the model’s unit of length.

- `v` is an array of line-of-sight velocities in the model’s unit of velocity (km/s in the physical system).

The result has the shape of `v` and is a probability density per unit velocity of the model’s unit system.

For a superposition, the call sums the components’ distributions in the aperture. The distribution of component
\(k\) is weighted by the component’s weight \(w_k\) times the component’s projected mass \(M_k\) inside
the aperture:
\(\overline{D} = \sum_k w_k M_k \overline{D}_k / \sum_k w_k M_k\).

## Accuracy report

The report of `aperture_losvd` checks the second moment of \(\overline{D}\) against
\(\overline{\sigma}_{\rm los}^2\) computed from the projected dispersion, without the velocity distributions. The
tier of the result is the worse of the aperture’s check and the tier of the `LosTable` ([Line-of-sight table](https://pianistpy.com/docs/guide/lostable)).
`aperture_losvd` issues an `AccuracyWarning` if the relative error of \(\overline{D}\) or
of the `LosTable` is above \(10^{-4}\).

The quick settings lose the most accuracy in a small aperture. Take an NFW halo of concentration 30 with a
Cuddeford-Osipkov-Merritt (COM) Type I anisotropy (\(\beta_{\rm c} = -0.25\), \(R_{\rm a} = 10\)):

- In a centered circle of radius 0.05 scale radii, the second moment with the quick settings differs from the
second moment with the default settings by a relative \(6.7 \times 10^{-6}\). The check of the
quick-settings result reports \(6.2 \times 10^{-6}\) (tier `"recorded"`). With the default settings the
check reports \(1.0 \times 10^{-7}\).

- In a circle of radius 1 scale radius, the second moments with the quick and with the default settings differ by
a relative \(1.5 \times 10^{-6}\). With the default settings the check reports \(6.4 \times 10^{-8}\).

For this NFW model, the quick settings build the `LosTable` in about a quarter of the time that the default
settings take.

## Example

```python
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
table = pp.LosTable(model)
circle = pp.Aperture.circle(r_ape=1.0)

v = np.linspace(-1.5, 1.5, 3001)
d = pp.aperture_losvd(table, circle, v)       # shape (3001,)
dv = v[1] - v[0]
d.sum() * dv                                   # one, to the accuracy of the sum on this grid
np.sqrt((v**2 * d).sum() * dv), pp.sigma_los_aperture(table, circle)

pp.aperture_losvd(table, pp.Aperture.circle(r_ape=1.0, r_off=10.0), v)   # an offset circle
```
