# Radial velocity dispersion

Source: https://pianistpy.com/docs/guide/sigma_r.html

# Radial velocity dispersion

pianistpy.sigma_r(), pianistpy.LosTable

sigma_r(source, r) returns the intrinsic radial velocity dispersion \(\sigma_r\) at 3D radii: the second
moment of the radial velocity over the model’s distribution function at a fixed radius. As in the paper, it is
the velocity moment of the augmented density \(\varrho_{\rm aug}(P, R)\) taken at the potential of the radius,

\[\sigma_r^2(R) = \frac{1}{\varrho_{\rm aug}(P, R)} \int_0^{P} \varrho_{\rm aug}(\mathcal{K}, R)\,
{\rm d}\mathcal{K}, \qquad P = P(R),\]

which for the anisotropy models of the package, whose augmented density separates as
\(\varrho_{\rm aug}(P, R) = \varrho_{\rm aux}(P)\,g(R)\), is the Jeans integral of \(\varrho_{\rm aux}\)
over the potential. The radius \(R\) of the paper is r in the package.

## Arguments and units

source is a Model (one component, or a superposition) or its prepared
LosTable, which holds the same intrinsic profile. r is an array of radii in the model’s
unit system: internal units by default (the scale radius is 1), parsecs (a float or an Astropy quantity) after
model.to("physical", ...). The result has the shape of r, in the model’s velocity unit (km/s in the
physical system).

Of a superposition, whose components share one density, the second moments add with the weights:
\(\sigma_r^2 = \sum_k w_k \sigma_{r,k}^2\).

## Tiers

The profile is tabulated once per model in radius, with rows added until every interval between rows reads
within profile_tolerance at its midpoint (1e-7 at the default tier, 1e-6 at the quick tier; a tenth of the
goal, the profile’s share of the error of what is projected from it). A model beyond the hard limit is refused
unless its settings ask for it (write_anyway), as for every observable (Accuracy).

## Example

```
import numpy as np
import pianistpy as pp

# a Hernquist density with an Osipkov-Merritt anisotropy of radius 1, truncated at K_t = 0.05
model = pp.solve(
    pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=1.0)),
    pp.Truncation(k_t=0.05),
    settings=pp.Settings(quick=True),
)
r = np.array([0.1, 1.0, 10.0])
pp.sigma_r(model, r)                          # internal units

physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.sigma_r(physical, [50.0, 500.0, 5000.0])   # parsecs in, km/s out

```

     On this page
