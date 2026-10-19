<p align="center">
  <img src="https://raw.githubusercontent.com/barrytchiang/PIANISTpy/main/docs/_static/logo.png" alt="PIANISTpy" width="320">
</p>

# PIANISTpy

[![Tests](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml/badge.svg)](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/barrytchiang/PIANISTpy/branch/main/graph/badge.svg)](https://codecov.io/gh/barrytchiang/PIANISTpy)
[![Documentation](https://img.shields.io/badge/docs-pianistpy.com-00356B)](https://pianistpy.com)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

**P**article **I**nitial conditions with **ANIS**o**T**ropy.

PIANISTpy builds spherical, non-rotating, velocity-anisotropic equilibrium models, self-consistently truncated in
phase space, for dark matter (sub)halos, dwarf galaxies, stellar spheroids and star clusters. The package provides
distribution functions, particle initial conditions, various velocity profiles and line-of-sight observables, with
JAX-based end-to-end differentiability for rapid gradient-based Monte Carlo inference.

> Status: version 0.9.0, a pre-release. Version 1.0.0, released with the paper that describes the method,
> adds the certified accuracy tiers on both backends, the remaining capabilities and the citation.

## What PIANISTpy does

- **Density profiles.** Any double power law in its three slopes, with the Navarro-Frenk-White, Hernquist and
  Plummer profiles as named presets.
- **Velocity anisotropy.** Isotropic, constant, Osipkov-Merritt, and the generalized Osipkov-Merritt (Cuddeford)
  models, radially biased (Type I) or tangentially biased (Type II).
- **Distribution functions.** The distribution function (DF) of each model, from the analytic inversion of its
  density, returned whether or not it is physical, with a statement of where it is not.
- **Truncation in phase space.** Method I truncates a model's own DF at a user-specified energy, or at the energy
  that yields a target mass or a target radius. Method II gives a new anisotropy to a density that is already
  truncated.
- **Particles.** Positions and velocities drawn by inversion, with no rejection step, and reproducible particle
  by particle.
- **Observables.** Intrinsic and line-of-sight velocity dispersions, line-of-sight velocity distributions, and
  their averages over apertures.
- **Superposition.** Distribution functions inverted from one truncated density (Method II children of one
  parent), combined with positive weights that sum to one into one equilibrium model.
- **Inference.** The log-likelihood of a model against data, as a JAX function of the components' weights and
  the scales of the units.

## Installation

```
pip install pianistpy
```

PIANISTpy needs Python 3.10 or later, [NumPy](https://numpy.org), [SciPy](https://scipy.org),
[h5py](https://www.h5py.org), [Astropy](https://www.astropy.org), [mpmath](https://mpmath.org),
[SymPy](https://www.sympy.org), [JAX](https://docs.jax.dev), [Diffrax](https://docs.kidger.site/diffrax/),
[Optimistix](https://docs.kidger.site/optimistix/) and [filelock](https://py-filelock.readthedocs.io).
The oldest versions the test suite passes with are NumPy 2.0.0, SciPy 1.14.0, h5py 3.11.0, Astropy 7.2.0,
mpmath 0.17, SymPy 1.2, and together JAX 0.6.1, Diffrax 0.7.0 and Optimistix 0.0.11 (these three are bound as
one set; older Diffrax or Optimistix releases pass only with a newer JAX); `pip` enforces them. It runs on Linux,
macOS and Windows.

## Quick start

The example of the paper's first figure: an NFW halo with a constant velocity anisotropy of 0.25, truncated in
phase space at the virial mass of a concentration-10 halo, presented as a Milky Way-mass halo.

```python
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

`pp.solve(halo, truncation)` runs the two steps at once.

## Documentation

The user guide, the notebook tutorials, the technical notes and the API reference are at
[pianistpy.com](https://pianistpy.com).

## Example notebooks

Tutorials for each workflow, and notebooks that reproduce the paper's figures, are in
[PIANISTpy-notebooks](https://github.com/barrytchiang/PIANISTpy-notebooks); each notebook is stored with its
outputs.

## Results and files

Every result (a distribution function, a model, a set of particles) is an immutable object that holds what it
was computed from: the inputs, the numerical settings and the version of the code. Its `write` method stores
it in a single HDF5 file, and `pp.read(path)` returns an equal object, whatever kind of result the file holds.
Nothing is written unless requested.

A `Model` is always a valid equilibrium. A request for which none exists raises `UnphysicalModelError`, and the
exception carries the distribution function for inspection.

## Units

Everything is computed and returned in internal units, the dimensionless normalization of Widrow (2000) in which
the scale radius and the central potential of the untruncated profile are 1. A result's `to` method presents it
in model units ($G = M_0 = r_0 = 1$) or, through Astropy, in physical units, without recomputing it, and what is
computed from it is returned in that system.

Three inputs of a truncation are given in the units that are natural for them, as their documentation states: the
truncation energy `k_t` in internal units, a target radius `r_t` in units of the scale radius, and a target mass
`mass_t` in units of $M_0$, together with the concentration that defines $M_0$.

## Accuracy

Every result carries a validation report, a short and fixed summary of the checks it passed, with its tier: a
check within the accuracy goal (a relative error of 1e-6) is silent; above the goal, within the ceiling (1e-4) or
within the hard limit (1e-2), the result is returned with a warning in the package's log and as an
`AccuracyWarning` (Python's standard filter silences or escalates it), and the flag stays with the result and with
everything computed from it; beyond the hard limit the result is refused.

The numerical settings come in two tested tiers, both held to these checks by the test suite:

```python
pp.Settings()            # the default tier
pp.Settings(quick=True)  # a faster tier for exploration
```

## Runtime options

None of these changes a number, except the platform: a GPU and a CPU give results that differ at rounding, and
each file records which made it.

```python
pp.set_verbosity("info")   # "quiet" (the default), "info" or "debug"
pp.set_timing(True)        # print the wall time of each stage
pp.set_workers("half")     # "auto" (the default), "full", "half" or a number of cores
pp.set_compilation_cache(False)  # the compilation cache is on by default; a path moves it
pp.set_platform("cpu")     # "auto" (the default), "cpu" (the CPU-only mode) or "gpu"; before the first computation
```

The CPU-only mode gives the same numbers on every machine; the environment variable `PIANISTPY_PLATFORM=cpu`
sets it before the package is imported. Each file records the backend that made it.

Hardware notes. The workers' switch sets how many cores the package's parallel work uses; `"auto"` uses every
allowed core inside a batch job and half of them anywhere else, and no value changes a number. A GPU accelerates the
two batched table builds, the speed table (2.7 to 7 times an 8-core CPU on an RTX 3080 Ti) and the line-of-sight
velocity table (1.0 to 1.4 times a 32-core node). The distribution function and the potential solve gain nothing
from a GPU: in this version the solve runs where the package runs, and a model is solved fastest in the CPU-only
mode. The CPU-only install is complete and is the right one for a laptop. JAX's
CUDA builds are for Linux: on Windows the GPU is reached through the Windows Subsystem for Linux with the Linux
wheels, which pays only for large table builds; a native Windows Python runs the CPU-only mode. A GPU and a CPU give
results that differ at rounding on the kernels; the accuracy goal holds on both, and the file's provenance says which
made it. The compilation cache lives in the home folder by default (`~/.cache/pianistpy/jax`) and can fill a small
one; `set_compilation_cache` with a path moves it.

## Citing

Please cite the initial release paper, Chiang, van den Bosch and Schive (2026), in work that uses PIANISTpy. The
citation entry is added with the 1.0.0 release.

## License

BSD 3-Clause. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md` and the code conventions in `STYLE_GUIDE.md`.
