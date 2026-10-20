"""PIANISTpy: equilibrium models of anisotropic spherical systems."""

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
    "set_workers",
    "sigma_los",
    "sigma_los_aperture",
    "sigma_r",
    "sigma_t",
    "solve",
    "superpose",
    "truncate",
]

# The reader of data files registers itself on import.
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
    set_platform,
    set_timing,
    set_verbosity,
    set_workers,
)
from .settings import Settings
from .superposition import superpose

configure_jax()
