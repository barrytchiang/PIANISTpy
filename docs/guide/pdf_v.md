# Local speed distribution

Source: https://pianistpy.com/docs/guide/pdf_v

# Local speed distribution

pianistpy.pdf_v(), pianistpy.LosTable

pdf_v(source, r, v) returns the local speed distribution \(p(V\,|\,R)\) at 3D radii: the probability density
of the speed \(V = |\mathbf{V}|\) of the tracers at a distance \(R\) from the center. As in Chiang+(2026d), the
density at a radius is split into the velocity angle \(\varphi\) and the speed,

\[\varrho(R) = 2\pi \int_0^\pi {\rm d}\varphi\; p_\varphi(\varphi\,|\,R)
\int_0^{V_{\rm esc}(R)} {\rm d}V\; p_V(V\,|\,R, \varphi),
\qquad V_{\rm esc}(R) = \sqrt{2P(R)},\]

and the speed distribution is the inner density integrated over the angle, normalized by \(\varrho(R)\):

\[p(V\,|\,R) = \frac{2\pi}{\varrho(R)} \int_0^\pi p_\varphi(\varphi\,|\,R)\, p_V(V\,|\,R, \varphi)\,{\rm d}\varphi .\]

It integrates to one over \(V \geq 0\). Its support ends at the escape speed \(V_{\rm esc}(R)\) for an
isotropic, constant-anisotropy or Type I model; a Type II model holds orbits of negative energy as well, and its
speeds reach \(V_{\rm esc}/\sqrt{1 - R^2/R_{\rm a}^2}\). The radius \(R\) of Chiang+(2026d) is r in the
package.

## Arguments and units

source is a Model or its prepared LosTable, whose intrinsic second
moments the distribution is checked against. r and v are arrays that broadcast against each other: radii
in the model’s unit of length (internal units by default, parsecs or an Astropy quantity in the physical system)
and speeds in its unit of velocity (km/s in the physical system). The result has their broadcast shape, per unit
velocity of the model’s unit system, and is zero at negative speeds and beyond the end of the support.

Of a superposition, the components are mixed by their density at the radius,
\(p = \sum_k w_k \varrho_k p_k / \sum_k w_k \varrho_k\).

## Tiers

Each radius asked for is computed once, with three checks: the norm of the distribution against the density, its
second moment against \(\sigma_{\rm r}^2 (3 - 2\beta_\sigma)\) from Radial velocity dispersion, and the change of the
distribution when the nodes of its angle rule double. The nodes double, at most three times, until that change is
under speed_tolerance (the goal \(10^{-6}\) at the default tier, the ceiling \(10^{-4}\) at the quick tier). A distribution
above the goal is returned with a warning; one beyond the hard limit is refused with an
AccuracyError unless the model’s settings ask for it (write_anyway). The stage behind the
call, pianistpy.observables.stages.speed_distribution(), returns the distribution at one radius with its
record, its escape speed, the end of its support and, for a Type II model, the share of it beyond the escape
speed (unbound_fraction).

## Example

```
import numpy as np
import pianistpy as pp

model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
v = np.linspace(0.0, 1.4, 1401)                           # past the escape speed at every r here
p = pp.pdf_v(model, np.array([[0.1], [1.0], [10.0]]), v)   # (3, 1401): radii by speeds
p.sum(axis=1) * (v[1] - v[0])                              # each about one

physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.pdf_v(physical, 500.0, [10.0, 50.0, 100.0])             # per km/s, at 500 pc

```

     On this page
