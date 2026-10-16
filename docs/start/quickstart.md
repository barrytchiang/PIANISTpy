# Quick start

Source: https://pianistpy.com/docs/start/quickstart.html

# Quick start

The example of the paper’s first figure: an NFW halo with a constant velocity anisotropy of 0.25, truncated in
phase space at the virial mass of a concentration-10 halo, presented as a Milky Way-mass halo.

```
import pianistpy as pp
import astropy.units as u

# an untruncated NFW density profile with a constant velocity anisotropy of 0.25
halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))

# the untruncated distribution function
df = pp.compute_df(halo)

# Method I: truncate it so that the truncated mass equals the virial mass of the
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

pp.solve(halo, truncation) runs the two steps at once.

Every result carries a validation report with its accuracy tier, and a model is always a valid equilibrium; what
the package returns, flags and refuses is on one page, Flags and refusals.
