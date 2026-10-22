# Line-of-sight velocity distribution

Source: https://pianistpy.com/docs/guide/losvd

`pianistpy.losvd()`, `pianistpy.LosTable`

`losvd(source, r_2d, v)` returns the line-of-sight velocity distribution \(D(R_{\rm 2D}, V_{\rm los})\) at
user-specified projected radii. \(D\) is the probability density of the line-of-sight velocity of the model’s
mass seen at a projected distance \(R_{\rm 2D}\) from the center. As in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758), \(D\) is the distribution
function \(F(\mathcal{Z}, L)\), with \(L\) the angular momentum, integrated over the line of sight up to the
truncation radius \(R_{\rm t}\) and over the two plane-of-sky velocity components \(V_{\perp 1}\) and
\(V_{\perp 2}\), and divided by the surface density \(\Sigma\),

\[D(R_{\rm 2D}, V_{\rm los}) = \frac{2}{\Sigma(R_{\rm 2D})} \int_{R_{\rm 2D}}^{R_{\rm t}}
\frac{R\,{\rm d}R}{(R^2 - R_{\rm 2D}^2)^{1/2}} \iint F(\mathcal{Z}, L)\,{\rm d}V_{\perp 1}\,{\rm d}V_{\perp 2},
\qquad
\Sigma(R_{\rm 2D}) = 2 \int_{R_{\rm 2D}}^{R_{\rm t}} \frac{\varrho(R)\,R\,{\rm d}R}{(R^2 - R_{\rm 2D}^2)^{1/2}},\]

where the inner integral runs over the plane-of-sky velocities at which the truncated distribution function is
positive. The distribution is even in \(V_{\rm los}\) and integrates to one over all
\(V_{\rm los}\). The distribution is zero at every \(|V_{\rm los}|\) above the largest velocity that the
distribution function reaches on the line of sight. The second moment of \(D\) is \(\sigma_{\rm los}^2\)
(`pianistpy.sigma_los()`).

## Arguments and units

- `source` is a `Model` or the model’s line-of-sight table (`LosTable`).
Given a model, `losvd` builds the model’s `LosTable` at every call. For more than one call, build the
`LosTable` once ([Line-of-sight table](https://pianistpy.com/docs/guide/lostable)). In a script, a `LosTable` and a call of `losvd` on a model run under
`if __name__ == "__main__":`, as [Line-of-sight table](https://pianistpy.com/docs/guide/lostable) explains.

- `r_2d` and `v` are arrays that broadcast against each other. `r_2d` holds projected radii in the model’s
unit of length: internal units by default, parsecs or an Astropy quantity in the physical system. `v` holds
line-of-sight velocities in the model’s unit of velocity (km/s in the physical system).

- The result has the broadcast shape of `r_2d` and `v` and is a probability density per unit velocity of the
model’s unit system.

For a superposition, each component’s distribution is read from the component’s own table. The components are
summed with their weights times their surface densities, \(D = \sum_k w_k \Sigma_k D_k / \sum_k w_k \Sigma_k\).

## Accuracy

The accuracy of `losvd` is the accuracy of the `LosTable` ([Line-of-sight table](https://pianistpy.com/docs/guide/lostable)). The `LosTable` checks the norm
of the distribution at each tabulated projected radius (a row of the table) against the surface density, and the
second moment of each row against the projected dispersion. The `LosTable` also estimates the error of the distribution between rows. A
projected radius in an interval where the `LosTable` could not meet its tolerance raises
`ResultError`, whose message gives the interval and the reason.

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

v = np.linspace(-1.5, 1.5, 3001)
d = pp.losvd(table, 1.0, v)                    # at R_2D = 1, shape (3001,)
dv = v[1] - v[0]
d.sum() * dv                                   # one, to the accuracy of the sum on this grid
np.sqrt((v**2 * d).sum() * dv), pp.sigma_los(table, 1.0)

pp.losvd(table, np.array([[0.1], [1.0], [10.0]]), v).shape   # (3, 3001): radii by velocities
```
