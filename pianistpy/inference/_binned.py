"""Binned distributions of velocity in the likelihood: the line-of-sight
velocity distribution at a projected radius or in an aperture, and the local
velocity distribution at a 3D radius.

The likelihood computes each distribution in NumPy before any JAX call, as
the observables compute the distribution: the values at the nodes of each
velocity interval, which fix the polynomial of the distribution on the
interval. This module writes the polynomial on each velocity interval as
the Legendre series of the polynomial's antiderivative, so that JAX
evaluates the cumulative distribution and the slope of the cumulative
distribution exactly at any velocity. The fraction of the distribution in a
bin is the difference of the cumulative distribution at the two edges of the
bin, and the velocity scale of the likelihood moves both edges.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..df import Model
from ..observables._panels import Panels, panels
from ..observables.stages import (
    DispersionProfile,
    KeptRanges,
    LosvdTable,
    ProjectedDispersion,
    aperture_losvd,
    dispersion_profile,
    kept_ranges,
    losvd_row,
    losvd_table,
    projected_dispersion,
    velocity_distribution,
)
from ..settings import Settings
from ..typing import Array

# The three binned observables: the line-of-sight velocity distribution at a
# projected radius, the line-of-sight velocity distribution in an aperture,
# and the local velocity distribution at a 3D radius.
BINNED = ("losvd", "losvd_aperture", "pdf_v")


def padded(p: Panels, n: int) -> tuple[Array, Array, Array]:
    """Return the interval ends `a` and `b` and the Legendre series of `p`,
    padded to `n` velocity intervals. Each padding interval lies beyond the
    last interval of `p`, has unit width and carries a series of zeros, so
    the padding adds nothing to the cumulative distribution."""
    if p.a.size > n:
        raise ValueError(f"{p.a.size} sub-panels, more than the {n} padded to.")
    k = n - p.a.size
    end = float(p.b[-1])
    a = np.concatenate([p.a, end + 1.0 + np.arange(k)])
    b = np.concatenate([p.b, end + 2.0 + np.arange(k)])
    series = np.concatenate([p.series, np.zeros((k, p.series.shape[1]))])
    return a, b, series


def cumulative(a: Array, b: Array, series: Array, u: Array) -> Array:
    """Return the integral of the distribution from 0 to each velocity `u`
    (``u >= 0``), exact in JAX: the sum over the velocity intervals of each
    interval's antiderivative at `u` clipped into the interval."""
    half = 0.5 * (b - a)
    y = jnp.clip((u[:, None] - 0.5 * (a + b)[None, :]) / half[None, :], -1.0, 1.0)
    # The three-term recurrence of the Legendre polynomials,
    # (k + 1) P_{k+1} = (2k + 1) y P_k - k P_{k-1}, evaluates the series
    # degree by degree without a table of the polynomials.
    p_prev, p_k = jnp.ones_like(y), y
    total = series[None, :, 0] * p_prev
    for k in range(1, series.shape[1]):
        total = total + series[None, :, k] * p_k
        p_prev, p_k = p_k, ((2 * k + 1) * y * p_k - k * p_prev) / (k + 1)
    return jnp.sum(total, axis=1)


@dataclass
class Pieces:
    """The tables from which one component's binned distributions are read,
    computed once in NumPy: the component's intrinsic dispersion profile,
    the component's projected table, the component's kept ranges and, for
    distributions in apertures, the component's table of line-of-sight
    velocity distributions."""

    model: Model
    profile: DispersionProfile
    projected: ProjectedDispersion
    kept: KeptRanges
    table: LosvdTable | None = None


def pieces(
    model: Model,
    settings: Settings,
    with_table: bool,
    kept: KeptRanges | None = None,
    table: LosvdTable | None = None,
    profile: DispersionProfile | None = None,
    projected: ProjectedDispersion | None = None,
) -> Pieces:
    """Return the `Pieces` of one component, with the component's table of
    line-of-sight velocity distributions when `with_table` is true. The
    arguments `kept`, `table`, `profile` and `projected`, where given, are
    the tables the likelihood computed once for all the `Data`, and
    `pieces` reads the given tables instead of computing the tables again.
    `pieces` reads `projected` only together with the `profile` from which
    `projected` was computed."""
    if profile is None or projected is None:
        profile = dispersion_profile(model, settings)
        projected = projected_dispersion(model, profile, settings)
    if kept is None:
        kept = kept_ranges(model, projected, settings)
    if with_table and table is None:
        table = losvd_table(model, kept, projected, settings)
    return Pieces(model, profile, projected, kept, table if with_table else None)


def read(kind: str, where: Any, p: Pieces, settings: Settings) -> tuple[Panels, float]:
    """Return one component's distribution for one `Data` of observable
    `kind` at the position `where` (in internal units) as `Panels`, and the
    component's share of the mixture at `where`: the component's projected
    mass in the aperture for ``"losvd_aperture"``, the component's surface
    density at the projected radius for ``"losvd"``, or the component's
    density at the 3D radius for ``"pdf_v"``."""
    if kind == "losvd_aperture":
        assert p.table is not None
        aperture = aperture_losvd(p.model, p.table, p.projected, where, settings)
        return (
            panels(aperture.v_nodes, aperture.v_weights, aperture.losvd[0]),
            float(aperture.mass[0]),
        )
    if kind == "losvd":
        row = losvd_row(p.model, float(where), p.kept, p.projected, settings)
        share = float(p.projected.read(np.array([float(where)]))[0][0, 0])
        return panels(row.v_nodes, row.v_weights, row.losvd[0]), share
    velocity = velocity_distribution(p.model, float(where), settings, p.profile)
    share = float(
        np.atleast_2d(p.model.profile.density(np.array([float(where)])))[0, 0]
    )
    return panels(velocity.v_nodes, velocity.v_weights, velocity.pdf), share
