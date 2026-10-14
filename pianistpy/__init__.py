"""PIANISTpy: equilibrium models of anisotropic spherical systems."""

__all__ = [
    "AccuracyError",
    "AccuracyWarning",
    "Anisotropy",
    "Aperture",
    "Component",
    "DistributionFunction",
    "DoublePowerLaw",
    "InputError",
    "LosTable",
    "Model",
    "PianistpyError",
    "PianistpyWarning",
    "ResultError",
    "Sampler",
    "Settings",
    "Truncation",
    "UnphysicalModelError",
    "__version__",
    "compute_df",
    "log",
    "platform",
    "read",
    "set_compilation_cache",
    "set_timing",
    "set_verbosity",
    "set_platform",
    "set_workers",
    "sigma_los",
    "sigma_los_aperture",
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
from .io import read
from .observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from .particles import Sampler
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
