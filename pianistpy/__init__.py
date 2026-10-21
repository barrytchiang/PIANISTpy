"""PIANISTpy: equilibrium models of anisotropic spherical systems, from the
distribution function to the observables.

In PIANISTpy, a model (`Model`) is the equilibrium of one or more
components that share one total potential, untruncated or truncated in
phase space. A component (`Component`) is a density profile, such as a
`DoublePowerLaw`, with a velocity anisotropy (`Anisotropy`). `solve`
computes a model from one component: the component's distribution
function, and the self-consistent density, mass and potential of the
model (Chiang+(2026d)).

Every observable is computed from a model. `Sampler` draws particles.
`sigma_r` and `sigma_t` give the radial and the tangential velocity
dispersion at 3D radii, `sigma_los` the line-of-sight velocity dispersion
at projected radii, and `losvd` the line-of-sight velocity distribution.
`likelihood` compares a model with observed data (`Data`).

A model, a distribution function, a `LosTable` and a sampler each carry a
validation report (`pianistpy.validation`). The report's tier, such as
``model.validation.tier``, is the worst band among the relative errors
that the report's checks measured and the tiers of the results it was
computed from: ``"silent"`` (at most :math:`10^{-6}`), ``"recorded"``
(at most :math:`10^{-4}`), ``"warned"`` (at most :math:`10^{-2}`) or
``"refused"`` (above :math:`10^{-2}`, or not a number).

>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> model.validation.tier
'silent'
"""

__all__ = [
    "AccuracyError",
    "AccuracyWarning",
    "Anisotropy",
    "Aperture",
    "Component",
    "Data",
    "Differentiable",
    "DistributionFunction",
    "DoublePowerLaw",
    "InputError",
    "LosTable",
    "Model",
    "Particles",
    "PianistpyError",
    "PianistpyWarning",
    "ProjectedSampler",
    "ResultError",
    "Sampler",
    "Settings",
    "Truncation",
    "UnphysicalModelError",
    "__version__",
    "aperture_losvd",
    "compute_df",
    "likelihood",
    "log",
    "losvd",
    "pdf_v",
    "platform",
    "read",
    "set_compilation_cache",
    "set_timing",
    "set_verbosity",
    "set_platform",
    "set_cores",
    "sigma_los",
    "sigma_los_aperture",
    "sigma_r",
    "sigma_t",
    "solve",
    "superpose",
    "truncate",
]

# Importing the subpackage inference registers the reader of data files, so
# that read() opens a data file.
from . import inference as inference
from ._version import __version__
from .anisotropy import Anisotropy
from .df import (
    Component,
    DistributionFunction,
    Model,
    Truncation,
    compute_df,
    solve,
    truncate,
)
from .exceptions import (
    AccuracyError,
    AccuracyWarning,
    InputError,
    PianistpyError,
    PianistpyWarning,
    ResultError,
    UnphysicalModelError,
)
from .inference import Data, likelihood
from .io import read
from .observables import (
    Aperture,
    Differentiable,
    LosTable,
    ProjectedSampler,
    aperture_losvd,
    losvd,
    pdf_v,
    sigma_los,
    sigma_los_aperture,
    sigma_r,
    sigma_t,
)
from .particles import Particles, Sampler
from .profiles import DoublePowerLaw
from .runtime import (
    configure_jax,
    log,
    platform,
    set_compilation_cache,
    set_cores,
    set_platform,
    set_timing,
    set_verbosity,
)
from .settings import Settings
from .superposition import superpose

configure_jax()
