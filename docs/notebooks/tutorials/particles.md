# Particles from a model

Source: https://pianistpy.com/docs/notebooks/tutorials/particles.html

# Particles from a model

A Sampler holds the tables that drawing particles needs; it is built once from a Model and reused by every call of sample. This tutorial draws particles from the isotropic Hernquist model, whose distribution function is known in closed form, so that it runs without the inversion.

```
import numpy as np
import pianistpy as pp
from pianistpy.df.analytic import analytic_model

model = analytic_model("hernquist")
model.spatially_finite, model.validation.tier

```

```
(False, 'goal')

```

The Hernquist model is not truncated, so the sampler needs an outer radius, r_sample, in internal units (the scale radius is 1).

```
sampler = pp.Sampler(model, r_sample=100.0)
particles = sampler.sample(100_000, seed=1)
particles.position.shape, particles.velocity.shape, particles.units

```

```
((100000, 3),
 (100000, 3),
 {'position': 'internal', 'velocity': 'internal', 'mass': 'internal'})

```

The same seed gives the same particles. The enclosed mass of the draw follows the profile: for the Hernquist model the mass inside r is proportional to r^2 / (1 + r)^2.

```
r = np.linalg.norm(particles.position, axis=1)
radii = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
drawn = np.array([(r < x).mean() for x in radii])
exact = (radii**2 / (1 + radii) ** 2) / (100.0**2 / 101.0**2)
np.column_stack([radii, drawn, exact])

```

```
array([[1.00000000e-01, 8.72000000e-03, 8.43057851e-03],
       [3.00000000e-01, 5.43700000e-02, 5.43248521e-02],
       [1.00000000e+00, 2.57090000e-01, 2.55025000e-01],
       [3.00000000e+00, 5.73190000e-01, 5.73806250e-01],
       [1.00000000e+01, 8.40350000e-01, 8.43057851e-01],
       [3.00000000e+01, 9.53980000e-01, 9.55348595e-01]])

```

The particles are written to a file and read back. The default layout is the package’s own HDF5 file; the layouts of simulation codes are chosen by layout.

```
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "hernquist.h5")
particles.write(path)
pp.read(path).identity == particles.identity

```

```
True

```
