# Quick start

Source: https://pianistpy.com/docs/start/quickstart

The example below builds an NFW halo of concentration 10 with a constant velocity anisotropy of 0.25. Method I
([Truncation in phase space](https://pianistpy.com/docs/guide/truncation)) truncates the halo in phase space so that the truncated mass
equals the virial mass of the untruncated halo. The model is then given in physical units, with a virial mass of
\(10^{12}\,M_\odot\), about the mass of the Milky Way, and a scale radius of 20 kpc.

```python
import pianistpy as pp
import astropy.units as u

# an untruncated NFW density profile with a constant velocity anisotropy of 0.25
halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))

# the untruncated distribution function
df = pp.compute_df(halo)

# Method I: truncate the distribution function so that the truncated mass equals the virial mass of the
# untruncated NFW halo of concentration 10 (the mass inside 10 scale radii)
model = pp.truncate(df, pp.Truncation(mass_t=1.0, concentration=10))

# physical units: a halo of 1e12 solar masses with R_vir = c * r_0 = 200 kpc
model = model.to("physical", mass_0=1e12 * u.Msun, r_0=20 * u.kpc)

# draw ten million equal-mass particles centered at the origin and write them to a file
particles = pp.Sampler(model).sample(10_000_000, seed=1)
particles.write("NFW_Const0p25_Mt1e12_PARIC.h5")

# the line-of-sight velocity dispersion at three projected radii
sigma = pp.sigma_los(model, [2, 20, 100] * u.kpc)
```

`pp.solve(halo, pp.Truncation(mass_t=1.0, concentration=10))` runs `compute_df` and `truncate` in one call.

The [Examples](https://pianistpy.com/docs/notebooks/index) page shows self-contained Jupyter tutorial notebooks, and the notebooks
that reproduce the result figures of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758).
