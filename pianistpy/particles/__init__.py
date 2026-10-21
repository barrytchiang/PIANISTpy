r"""Particle sampling: the sampler, the particle result and the particle writers.

A user builds a `Sampler` from a model once, then calls `Sampler.sample` for
a `Particles` object and `Particles.write` to write the particles to a file
in one of the layouts of ``PARTICLE_WRITERS``.

Every particle is drawn by inversion, with no rejection step. The radius
:math:`r` of a particle solves :math:`M(r)/M(r_{\rm s}) = u_r`, where
:math:`M(r)` is the mass enclosed within :math:`r`, :math:`r_{\rm s}` is the
outer radius of the sampling and :math:`u_r` is a uniform random number
between 0 and 1. The velocity of a particle is drawn from the model's
distribution of velocity at the potential :math:`\Psi(r)` of the particle's
radius. The derivation is in Notes Section T:5.
"""

__all__ = [
    "PARTICLE_WRITERS",
    "Particles",
    "Sampler",
    "component_view",
    "is_superposition",
    "register_particle_writer",
    "write_particles",
]

from . import writers as _writers  # noqa: F401  (registers the particle writers)
from .core import (
    PARTICLE_WRITERS,
    Particles,
    Sampler,
    component_view,
    is_superposition,
    register_particle_writer,
    write_particles,
)
