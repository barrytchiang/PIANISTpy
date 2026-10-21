<!-- Written by docs/tools/repo_texts.py from https://pianistpy.com. Edit the website, not this file. -->

<p align="center">
<img src="https://raw.githubusercontent.com/barrytchiang/PIANISTpy/main/docs/_static/logo.png" alt="PIANISTpy" width="320">
</p>

# PIANISTpy

[![Tests](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml/badge.svg)](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/barrytchiang/PIANISTpy/branch/main/graph/badge.svg)](https://codecov.io/gh/barrytchiang/PIANISTpy)
[![Documentation](https://img.shields.io/badge/docs-pianistpy.com-00356B)](https://pianistpy.com)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/barrytchiang/PIANISTpy/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pianistpy)](https://pypi.org/project/pianistpy/)
[![Python](https://img.shields.io/pypi/pyversions/pianistpy)](https://pypi.org/project/pianistpy/)

**P**article **I**nitial conditions with **ANIS**o**T**ropy.

PIANISTpy builds spherical, non-rotating, velocity-anisotropic equilibrium models, self-consistently truncated in phase space, for dark matter (sub)halos, dwarf galaxies, stellar spheroids and star clusters. The package provides distribution functions, particle initial conditions, radial and tangential velocity dispersion profiles and line-of-sight observables, with JAX-based end-to-end differentiability for gradient-based Monte Carlo inference.

**Ver. 1.0.0** is the initial public release of PIANISTpy, described in [Chiang, van den Bosch and Schive (2026d), MNRAS submitted](https://pianistpy.com/docs/project/citing), henceforth [Chiang+(2026d)](https://pianistpy.com/docs/project/citing).

## What PIANISTpy does

- [Density profiles.](https://pianistpy.com/docs/guide/profiles) Analytical double-power-law density profiles with a finite central
potential, including the Navarro-Frenk-White, Hernquist and Plummer profiles.

- [Velocity anisotropy.](https://pianistpy.com/docs/guide/anisotropy) Isotropic, constant-anisotropy, Osipkov-Merritt and
Cuddeford-Osipkov-Merritt (COM) models. The Osipkov-Merritt and COM models are radially (Type I) or tangentially
(Type II) biased.

- [Distribution functions.](https://pianistpy.com/docs/guide/df) Each anisotropy model’s distribution function, inverted from the
input density with exact symbolic expressions up to the final numerical integration, with extensively validated
numerical accuracy ([Accuracy tables](https://pianistpy.com/docs/reference/accuracy)) and checks that the distribution function is
non-negative (physical) and monotonic (a sufficient condition for dynamical stability).

- [Truncation in phase space.](https://pianistpy.com/docs/guide/truncation) Model truncation performed in phase space. The
truncation boundary is self-consistent and in dynamical equilibrium. Method I preserves the model’s anisotropy
profile, and Method II gives different anisotropy models an identical truncated density.

- [Particles.](https://pianistpy.com/docs/guide/particles) Particle initial conditions drawn from the truncated distribution
function, with a finite, dynamically stable truncation boundary. The particles are given in internal, model or
physical units.

- [Observables.](https://pianistpy.com/docs/guide/observables) Radial, tangential and line-of-sight velocity dispersions,
line-of-sight velocity distributions and their Gauss-Hermite moments, and the local velocity distribution at a 3D
radius. Projected quantities are averaged over centered or offset circular and rectangular apertures.

- [Superposition.](https://pianistpy.com/docs/guide/superposition#guide-superposition) Distribution functions of different anisotropy models on one
truncated density, linearly superposed into one equilibrium model. The superposed model’s anisotropy profile can
differ from every analytical family, for example an anisotropy profile fitted to simulated halos.

- [Inference.](https://pianistpy.com/docs/guide/superposition#guide-inference) End-to-end differentiable in JAX: the likelihood of a model against
kinematic data, with gradients in the superposition weights, the mass and length scales (*M*<sub>0</sub> and
*r*<sub>0</sub>) and, for Method II components, the anisotropy parameters, for gradient-based Monte Carlo inference.

## Getting started

```bash
pip install pianistpy
```

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

[Install](https://pianistpy.com/docs/start/install) · [Quick start](https://pianistpy.com/docs/start/quickstart)

## Documentation

- [What PIANISTpy does](https://pianistpy.com/docs/start/overview)
- [User guide](https://pianistpy.com/docs/guide/)
- [Examples](https://pianistpy.com/docs/notebooks/)
- [Functions by subpackage](https://pianistpy.com/docs/reference/api/)
- [Release notes](https://pianistpy.com/docs/project/release-notes)

## How to cite

Please cite the initial release paper, [Chiang, van den Bosch and Schive (2026d), MNRAS submitted](https://pianistpy.com/docs/project/citing), henceforth [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), in work that uses PIANISTpy.

## License

PIANISTpy is released under the BSD 3-Clause License, copyright 2026 Barry T. Chiang. The full text is in the
`LICENSE` file of the repository.

PIANISTpy is written and maintained by Barry T. Chiang (Yale University), [barrytchiang@gmail.com](mailto:barrytchiang%40gmail.com).
