# Line-of-sight velocity distribution in an aperture

Source: https://pianistpy.com/docs/guide/aperture_losvd.html

# Line-of-sight velocity distribution in an aperture

pianistpy.aperture_losvd(), pianistpy.Aperture, pianistpy.LosTable

aperture_losvd(source, aperture, v) returns the line-of-sight velocity distribution of the tracers inside an
aperture: the distribution at each projected radius (Line-of-sight velocity distribution) averaged over the aperture with the surface
density,

\[\overline{D}(V_{\rm los}) = \frac{\int_{\rm aperture} \Sigma(R_{\rm 2D})\, D(R_{\rm 2D}, V_{\rm los})\,{\rm d}A}
{\int_{\rm aperture} \Sigma(R_{\rm 2D})\,{\rm d}A}.\]

It is even in \(V_{\rm los}\), integrates to one over all \(V_{\rm los}\), and is zero beyond the escape
velocity along the line of sight. Its second moment is the dispersion inside the aperture,
\(\overline{\sigma}_{\rm los}^2\) (pianistpy.sigma_los_aperture() given the same table).

## Arguments and units

source is a Model or its prepared LosTable; given a model, its table
is built first, so for more than one call build the table once (The prepared table: LosTable). aperture is an
Aperture, circular or rectangular, centered or offset (Observables), its lengths in the
model’s unit of length. v is an array of line-of-sight velocities in the model’s unit of velocity (km/s in the
physical system); the result has its shape, per unit velocity of the model’s unit system.

Of a superposition, each component’s distribution in the aperture is summed with its weight times its projected
mass inside the aperture, \(\overline{D} = \sum_k w_k M_k \overline{D}_k / \sum_k w_k M_k\).

## Tiers

The distribution is computed on panels in velocity, refined until its norm and its second moment agree with the
table’s rows, and at any velocity it is read by each panel’s polynomial through its own nodes. Its second moment is
checked against the aperture’s dispersion from the projected tables, which reads no velocity distribution, so the
tier is the worse of that check and the table’s own (The prepared table: LosTable); a distribution above the goal says so in a
warning.

A small aperture costs the quick tier the most. Measured on an NFW halo of concentration 30 with a Cuddeford Type I
anisotropy (\(\beta_c = -0.25\), \(R_a = 10\)) in a centered circle of radius 0.05 in the scale radius, the
quick tier’s second moment differs from the default tier’s by 6.7e-6 of it, and its own check reports 6.2e-6 (above
the goal, with a warning); in a circle of radius 1 the two differ by 1.5e-6. At the default tier the check reports
1.0e-7 and 6.4e-8. The quick tier’s table took a quarter of the default’s time.

## Example

```
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=1.0)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
table = pp.LosTable(model)
circle = pp.Aperture.circle(r_ape=1.0)

v = np.linspace(-1.5, 1.5, 3001)
d = pp.aperture_losvd(table, circle, v)       # shape (3001,)
dv = v[1] - v[0]
d.sum() * dv                                   # one, to the grid's rule
np.sqrt((v**2 * d).sum() * dv), pp.sigma_los_aperture(table, circle)

pp.aperture_losvd(table, pp.Aperture.circle(r_ape=1.0, r_off=10.0), v)   # an offset circle

```

     On this page
