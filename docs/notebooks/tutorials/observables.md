# Compute observables from a model

Source: https://pianistpy.com/docs/notebooks/tutorials/observables

# Compute observables from a model

The line-of-sight velocity dispersion at projected radii, its average over an aperture, and the same quantities in physical units. This tutorial uses the isotropic Hernquist model, whose distribution function is known in closed form.

```
import numpy as np
import pianistpy as pp
from pianistpy.df.analytic import analytic_model

model = analytic_model("hernquist")

```

sigma_los takes projected radii, in the model’s units (internal here: the scale radius is 1).

```
r_2d = np.array([0.1, 1.0, 10.0])
pp.sigma_los(model, r_2d)

```

```
array([0.30974035, 0.27598987, 0.1234161 ])

```

sigma_los_aperture averages the squared dispersion over an aperture, weighted by the surface density: a centered circle, or a rectangle at an offset and an angle.

```
circle = pp.Aperture.circle(1.0)
slit = pp.Aperture.rectangle(2.0, 0.5, r_off=1.0, angle=0.3)
pp.sigma_los_aperture(model, circle), pp.sigma_los_aperture(model, slit)

```

```
(0.3028872393345692, 0.3003436495218896)

```

A LosTable computes the line-of-sight velocity distributions once and serves every call made from it. Its aperture dispersion is the second moment of the aperture’s velocity distribution, an independent route that agrees with the one from the model.

```
table = pp.LosTable(model)
pp.sigma_los(table, r_2d), pp.sigma_los_aperture(table, circle)

```

```
(array([0.30974035, 0.27598987, 0.1234161 ]), 0.302887239623593)

```

In physical units, radii and apertures are given in parsecs (or as Astropy quantities) and the dispersions are returned in km/s. Here M_0 = 1e9 solar masses and r_0 = 500 pc, with a concentration of 10.

```
physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
pp.sigma_los(physical, [50.0, 500.0, 5000.0])

```

\[[31.599922,~28.156675,~12.590995] \; \mathrm{\frac{km}{s}}\]

```
pp.sigma_los_aperture(
    table.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0), pp.Aperture.circle(500.0)
)

```

\[30.900763 \; \mathrm{\frac{km}{s}}\]
