"""Velocity dispersions and velocity distributions of a model.

The functions of pianistpy.observables return the radial, tangential and
line-of-sight velocity dispersions, the line-of-sight velocity distribution,
and the local velocity distribution of a model, at user-specified radii or in a
user-specified aperture on the plane of the sky.
"""

__all__ = [
    "Aperture",
    "Differentiable",
    "LosTable",
    "ProjectedPairs",
    "ProjectedSampler",
    "aperture_losvd",
    "losvd",
    "pdf_v",
    "sigma_los_aperture",
    "sigma_los",
    "sigma_r",
    "sigma_t",
]

from ._differentiable import Differentiable
from .aperture import Aperture
from .core import (
    LosTable,
    aperture_losvd,
    losvd,
    pdf_v,
    sigma_los,
    sigma_los_aperture,
    sigma_r,
    sigma_t,
)
from .projected import ProjectedPairs, ProjectedSampler
