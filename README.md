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

PIANISTpy builds spherical, non-rotating, velocity-anisotropic equilibrium models, self-consistently truncated in
phase space, for dark matter (sub)halos, dwarf galaxies, stellar spheroids and star clusters. The package provides
distribution functions, particle initial conditions, various velocity profiles and line-of-sight observables, with
JAX-based end-to-end differentiability for rapid gradient-based Monte Carlo inference.

> Status: ver. 0.9.9, the code of the paper that describes the method,
> [Chiang, van den Bosch and Schive (2026d), MNRAS](https://pianistpy.com/docs/project/citing), submitted (henceforth [Chiang+(2026d)](https://pianistpy.com/docs/project/citing)). Ver. 1.0.0 follows with the citation entry, the DOI of the software release and the changes its release notes list.

## What PIANISTpy does

- **Density profiles.** Analytical double-power-law density profiles with a finite central potential, including
  the commonly used Navarro-Frenk-White, Hernquist and Plummer profiles.
- **Velocity anisotropy.** Isotropic, constant-anisotropy, Osipkov-Merritt and Cuddeford-Osipkov-Merritt models,
  the latter two radially (Type I) or tangentially (Type II) biased.
- **Distribution functions.** Each anisotropy model's distribution function, inverted from the input density
  with exact symbolic expressions up to the final numerical integration, with
  [extensively validated numerical accuracy](https://pianistpy.com/docs/reference/accuracy) and checks that it
  is non-negative (physical) and monotonic (dynamically stable).
- **Truncation in phase space.** Model truncation performed in phase space, such that the truncation boundary is
  fully self-consistent and in dynamical equilibrium, by two methods: one preserves the model's anisotropy
  profile (Method I), the other gives different anisotropy models an identical truncated density (Method II).
- **Particles.** Particle initial conditions with a finite, dynamically stable truncation boundary, drawn from
  the truncated distribution function, in internal, model or physical units.
- **Observables.** Radial, tangential and line-of-sight velocity dispersions, line-of-sight velocity
  distributions and their Gauss-Hermite moments, and the local 3D velocity distribution at any radius; projected
  quantities averaged over centered or offset circular and rectangular apertures.
- **Superposition.** Distribution functions of different anisotropy models on one truncated density, linearly
  superposed into one equilibrium model whose anisotropy profile lies beyond the analytical families, such as
  one fitted to simulated halos.
- **Inference.** End-to-end differentiable in JAX: the likelihood of a model against kinematic data, and every
  observable, with gradients in the superposition weights, the unit scales and the anisotropy parameters, for
  gradient-based Monte Carlo inference.

## Installation

```
pip install pianistpy
```

PIANISTpy needs Python 3.11 or later, [NumPy](https://numpy.org), [SciPy](https://scipy.org),
[h5py](https://www.h5py.org), [Astropy](https://www.astropy.org), [mpmath](https://mpmath.org),
[SymPy](https://www.sympy.org), [JAX](https://docs.jax.dev), [Diffrax](https://docs.kidger.site/diffrax/) and
[Optimistix](https://docs.kidger.site/optimistix/).
The oldest versions the test suite passes with, measured one at a time and then all together as one environment,
are NumPy 2.3.0, SciPy 1.15.0, h5py 3.11.0, Astropy 6.1.0, mpmath 1.1.0, SymPy 1.2, JAX 0.6.0, Diffrax 0.7.1 and
Optimistix 0.1.0; `pip` enforces them. It runs on Linux, macOS and Windows.

## Quick start

The example of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Fig. 3: an NFW halo with a concentration
of 10 and a constant velocity anisotropy of 0.25, truncated in phase space such that the total final mass equals the
untruncated virial mass, scaled physically to the Milky Way mass.

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

Tutorials for each workflow, and notebooks that reproduce the result figures of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), are in
[PIANISTpy-notebooks](https://github.com/barrytchiang/PIANISTpy-notebooks), tagged with the package version
they were run with.

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
in model units (<i>G</i> = <i>M</i><sub>0</sub> = <i>r</i><sub>0</sub> = 1) or, through Astropy, in physical units, without recomputing it, and what is
computed from it is returned in that system.

Three inputs of a truncation are given in the units that are natural for them, as their documentation states: the
truncation energy `k_t` in internal units, a target radius `r_t` in units of the scale radius, and a target mass
`mass_t` in units of <i>M</i><sub>0</sub>, together with the concentration that defines <i>M</i><sub>0</sub>.

## Accuracy

Every result carries a validation report, a short and fixed summary of the checks it passed, with its tier: a
check within the accuracy goal (a relative error of 10<sup>&minus;6</sup>) is silent; above the goal, within the ceiling (10<sup>&minus;4</sup>) or
within the hard limit (10<sup>&minus;2</sup>), the result is returned with a warning in the package's log and as an
`AccuracyWarning` (Python's standard filter silences or escalates it), and the flag stays with the result and with
everything computed from it; beyond the hard limit the result is refused.

The numerical settings come in two tested tiers, both held to these checks by the test suite:

```python
pp.Settings()            # the default tier
pp.Settings(quick=True)  # a faster tier for exploration
```

## Runtime options

These runtime options change only how the PIANISTpy numerical executions are carried out, while keeping results
bit-for-bit identical, except for the platform switch, where the CPU-only and GPU-enabled results can differ, but by
less than 10<sup>&minus;10</sup> in relative terms.

```python
pp.set_verbosity("info")   # "quiet" (the default), "info" or "debug"
pp.set_timing(True)        # print the wall-clock time of each stage
pp.set_workers("half")     # "auto" (the default), "full", "half" or a number of cores
pp.set_compilation_cache(False)  # the compilation cache is on by default; a path moves it
pp.set_platform("cpu")     # "auto" (the default), "cpu" (the CPU-only mode) or "gpu"; before the first computation
```

The CPU-only mode gives the same numbers on every machine; the environment variable `PIANISTPY_PLATFORM=cpu`
sets it before the package is imported. Each file records the backend that made it.

`set_workers` sets how many cores the package's parallel work uses; `"auto"` uses every allowed core inside a batch
job and half of them anywhere else, and no value changes a number.

PIANISTpy runs on the CPU with the default installation. With a CUDA build of JAX installed, it automatically
becomes GPU-enabled for the GPU-implemented features, with no manual setting needed in PIANISTpy. JAX's CUDA builds
are for Linux only, so macOS runs the CPU-only mode. A native Windows Python also runs the CPU-only mode; on
Windows the GPU is reached through the Windows Subsystem for Linux (or a Linux virtual machine). Results on the GPU
and the CPU agree to better than 10<sup>&minus;10</sup> in relative terms, and the accuracy goal holds on both.
On shared clusters, a GPU in the Exclusive_Process compute mode serves one process at a time; if another process
holds it, JAX stops with `Unable to initialize backend 'cuda'`. Free the GPU, share it through an MPS daemon
(`nvidia-cuda-mps-control -d`), or use the CPU-only mode.

The compilation cache lives in the home folder by default (`~/.cache/pianistpy/jax`) and can fill a small one;
`set_compilation_cache` with a path moves it.

## Citing

Please cite the initial release paper, [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), in work that uses PIANISTpy. The
citation entry is added with ver. 1.0.0.

## License

BSD 3-Clause. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md` and the code conventions in `STYLE_GUIDE.md`.
