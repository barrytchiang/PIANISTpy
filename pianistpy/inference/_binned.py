"""Binned distributions of velocity in the likelihood (298 stage 2, the DF
path): the line-of-sight velocity distribution at a projected radius or in an
aperture, and the local speed distribution at a 3D radius.

Each distribution is read on the host as the observables give it: on the
sub-panels of its velocity, the values at each sub-panel's nodes, which fix
the polynomial the observables read it by. Here each sub-panel's polynomial is
written as the Legendre series of its antiderivative, so that the cumulative
distribution, and its slope, are exact at any velocity in JAX: a bin's share
is a difference of the cumulative at its edges, which the speed scale moves.
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
    speed_distribution,
)
from ..settings import Settings
from ..typing import Array

# The observables of this module, and what each is measured at.
BINNED = ("losvd", "losvd_aperture", "pdf_v")


def padded(p: Panels, n: int) -> tuple[Array, Array, Array]:
    """The ends and series of `p` padded to `n` sub-panels: a padding
    sub-panel lies beyond the last, of unit width, with a zero series."""
    if p.a.size > n:
        raise ValueError(f"{p.a.size} sub-panels, more than the {n} padded to.")
    k = n - p.a.size
    end = float(p.b[-1])
    a = np.concatenate([p.a, end + 1.0 + np.arange(k)])
    b = np.concatenate([p.b, end + 2.0 + np.arange(k)])
    series = np.concatenate([p.series, np.zeros((k, p.series.shape[1]))])
    return a, b, series


def cumulative(a: Array, b: Array, series: Array, u: Array) -> Array:
    """The integral of the distribution from 0 to each `u` (``u >= 0``), exact in
    JAX: each sub-panel's antiderivative at u clipped into it, summed."""
    half = 0.5 * (b - a)
    y = jnp.clip((u[:, None] - 0.5 * (a + b)[None, :]) / half[None, :], -1.0, 1.0)
    # the Legendre series by its three-term recurrence, degree by degree
    p_prev, p_k = jnp.ones_like(y), y
    total = series[None, :, 0] * p_prev
    for k in range(1, series.shape[1]):
        total = total + series[None, :, k] * p_k
        p_prev, p_k = p_k, ((2 * k + 1) * y * p_k - k * p_prev) / (k + 1)
    return jnp.sum(total, axis=1)


@dataclass
class Pieces:
    """What a member's binned reads need, built on the host once: its
    intrinsic and projected tables, its kept ranges and, for apertures, its
    table of rows."""

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
) -> Pieces:
    """The member's tables, its table of rows where an aperture reads it;
    `kept` and `table`, where given, are the likelihood's own (built once for
    all its Data) and are read, not built again."""
    profile = dispersion_profile(model, settings)
    projected = projected_dispersion(model, profile, settings)
    if kept is None:
        kept = kept_ranges(model, projected, settings)
    if with_table and table is None:
        table = losvd_table(model, kept, projected, settings)
    return Pieces(model, profile, projected, kept, table if with_table else None)


def read(kind: str, where: Any, p: Pieces, settings: Settings) -> tuple[Panels, float]:
    """One member's distribution for one Data (`kind`, at `where` in internal
    units) as `Panels`, and its share of the mixture: its projected mass in
    the aperture, its surface density at the radius, or its density at the
    3D radius."""
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
    speed = speed_distribution(p.model, float(where), settings, p.profile)
    share = float(
        np.atleast_2d(p.model.profile.density(np.array([float(where)])))[0, 0]
    )
    return panels(speed.v_nodes, speed.v_weights, speed.pdf), share
