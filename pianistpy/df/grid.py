"""The energies of the DF table: five zones and the nodes around a Type II
model's interior point.

The grid runs from the floor to the ceiling gap below the central potential
in five zones (A log-spaced in Q, AB a geometric buffer, B linear, BC a
geometric buffer, C log-spaced in P_0 - Q), extended below the floor to the
DF table's own floor. No node lies above the ceiling: the gap states where
float64 energies are separable from P_0, and the center's form carries the DF
above the last node. OM Type I models place zones A and AB around the potential of the
anisotropy radius; OM Type II and COM models around the feature their pilot
inversion finds (given here).

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
"""

__all__ = [
    "GridZones",
    "buffer_grid",
    "feature_zones",
    "hybrid_grid",
    "inversion_grid",
    "standard_zones",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..profiles import DoublePowerLaw
from ..settings import Settings
from ..typing import Array
from .truncation import CEILING_GAP, K_FLOOR

# The kernel's table reaches down to the grid's own floor, by definition: no
# table-only nodes below it.
TABLE_FLOOR = K_FLOOR
# The written table starts at the first grid node at or above this multiple of
# the floor.
OUTPUT_FLOOR_FACTOR = 2.0
BAND_REFINE = 5.0  # a COM Type II interior point's band: spacing refined this much
BAND_HALFWIDTH = 0.5  # and its half-width, as a fraction of the point's energy


@dataclass(frozen=True)
class GridZones:
    """The five zones' boundaries.

    Parameters
    ----------
    p0 : float
        The central potential.
    a_lo, a_hi, ab_hi, bc_lo, c_lo, c_hi : float
        Zone A's lower bound, the A/AB, AB/B, B/BC and BC/C boundaries, and
        zone C's upper bound.
    q_feature : float
        The feature the OM and COM zones are placed around (Psi(R_a) for OM
        Type I, the pilot inversion's for the others); zero for the standard
        zones.
    case : int
        0 standard, 1 (a minimum) or 2 (a zero crossing) for a feature.
    kappa_low : float
        The analytic low-Q exponent; NaN where not applicable.
    """

    p0: float
    a_lo: float
    a_hi: float
    ab_hi: float
    bc_lo: float
    c_lo: float
    c_hi: float
    q_feature: float
    case: int
    kappa_low: float


def standard_zones(p0: float, psi_extent: float, psi_anchor_low: float) -> GridZones:
    """Return the fixed-fraction zones of an isotropic or constant-beta model."""
    return GridZones(
        p0=float(p0),
        a_lo=float(psi_anchor_low),
        a_hi=float(p0) * float(psi_extent),
        ab_hi=float(p0) * 3.0 / 8.0,
        bc_lo=float(p0) * 5.0 / 8.0,
        c_lo=float(p0) * (1.0 - float(psi_extent)),
        c_hi=float(p0) - float(psi_anchor_low),
        q_feature=0.0,
        case=0,
        kappa_low=float("nan"),
    )


def feature_zones(
    p0: float,
    psi_extent: float,
    psi_anchor_low: float,
    q_feature: float,
    case: int,
    kappa_low: float,
) -> GridZones:
    """Return the zones of an OM or COM model placed around its feature:
    A to 0.7 q_feature, AB to min(1.5 q_feature, 3/8 P_0); the standard
    boundaries where the feature is absent or the placement invalid. The
    high-Q end is never moved."""
    if q_feature > 0.0:
        a_hi = 0.7 * q_feature
        ab_hi = min(1.5 * q_feature, p0 * 3.0 / 8.0)
        a_hi = max(a_hi, float(psi_anchor_low) * 100.0)
        a_hi = min(a_hi, 0.9 * ab_hi)
        if not (float(psi_anchor_low) * 100.0 < a_hi < ab_hi < p0 * 5.0 / 8.0):
            a_hi = p0 * psi_extent
            ab_hi = p0 * 3.0 / 8.0
    else:
        a_hi = p0 * psi_extent
        ab_hi = p0 * 3.0 / 8.0
    return GridZones(
        p0=p0,
        a_lo=float(psi_anchor_low),
        a_hi=a_hi,
        ab_hi=ab_hi,
        bc_lo=p0 * 5.0 / 8.0,
        c_lo=p0 * (1.0 - psi_extent),
        c_hi=p0 - float(psi_anchor_low),
        q_feature=q_feature,
        case=int(case),
        kappa_low=kappa_low,
    )


def buffer_grid(psi_lo: float, psi_hi: float, h_lo: float, h_hi: float) -> Array:
    """Return a grid over [psi_lo, psi_hi] whose step goes geometrically from
    h_lo to h_hi, the number of steps the one whose total is closest to the
    width, the steps rescaled to sum to it exactly."""
    if psi_hi <= psi_lo or h_lo <= 0 or h_hi <= 0:
        return np.array([psi_lo, psi_hi])
    W = psi_hi - psi_lo
    if abs(np.log(h_hi / h_lo)) < 1e-10:
        N = max(2, int(round(W / h_lo)))
        pts = np.linspace(psi_lo, psi_hi, N + 1)
        pts[0] = psi_lo
        pts[-1] = psi_hi
        return pts

    def total_distance(N):
        if N < 2:
            return 0.0
        k_arr = np.arange(N, dtype=np.float64)
        steps = h_lo * (h_hi / h_lo) ** (k_arr / (N - 1))
        return float(np.sum(steps))

    N = 2
    while total_distance(N) < W:
        N += 1
        if N > 100000:
            break
    if N > 2:
        t_lo = total_distance(N - 1)
        t_hi = total_distance(N)
        if abs(t_lo - W) < abs(t_hi - W):
            N = N - 1
    if N < 2:
        return np.array([psi_lo, psi_hi])
    k_arr = np.arange(N, dtype=np.float64)
    steps = h_lo * (h_hi / h_lo) ** (k_arr / (N - 1))
    steps = steps * (W / np.sum(steps))
    pts = np.empty(N + 1)
    pts[0] = psi_lo
    pts[1:] = psi_lo + np.cumsum(steps)
    pts[-1] = psi_hi
    return pts


def hybrid_grid(
    zones: GridZones,
    n_linear: int,
    n_log: int,
    q_lo_phys: float,
    psi_anchor_low: float,
    g_table_floor: float,
    g_table_nodes_per_decade: int,
    q_ceil_gap: float,
) -> Array:
    """Return the five-zone grid on [q_lo, P_0 - ceiling gap], extended below
    to the table's own floor where that lies below the anchor."""
    P0 = float(zones.p0)
    psi_a_outer = float(zones.a_hi)
    psi_ab_outer = float(zones.ab_hi)
    psi_bc_inner = float(zones.bc_lo)
    psi_c_inner = float(zones.c_lo)
    q_lo = max(float(zones.a_lo), float(q_lo_phys))
    if psi_a_outer >= psi_ab_outer:
        psi_ab_outer = 0.5 * (psi_a_outer + psi_bc_inner)
        psi_bc_inner = 0.5 * (psi_ab_outer + psi_c_inner)
    grid_a = np.logspace(np.log10(q_lo), np.log10(psi_a_outer), n_log)
    h_a_outer = float(grid_a[-1] - grid_a[-2])
    q_c_lo = psi_c_inner
    q_c_hi = P0 - psi_anchor_low
    one_minus_c = np.logspace(np.log10(P0 - q_c_hi), np.log10(P0 - q_c_lo), n_log)
    grid_c = P0 - one_minus_c[::-1]
    grid_c = grid_c[(grid_c >= q_c_lo) & (grid_c <= q_c_hi)]
    if len(grid_c) < 2:
        grid_c = np.array([q_c_lo, q_c_hi])
    h_c_inner = float(grid_c[1] - grid_c[0])
    b_pivot = 0.05 * P0
    if psi_ab_outer < 0.01 * P0 and psi_bc_inner > b_pivot:
        dec_bridge = np.log10(b_pivot / psi_ab_outer)
        n_bridge = max(int(np.ceil(120.0 * dec_bridge)), 8)
        grid_b_log = np.logspace(
            np.log10(psi_ab_outer), np.log10(b_pivot), n_bridge, endpoint=False
        )
        grid_b = np.concatenate(
            [grid_b_log, np.linspace(b_pivot, psi_bc_inner, n_linear)]
        )
        h_b_lo = float(grid_b[1] - grid_b[0])
        h_b_hi = float(grid_b[-1] - grid_b[-2])
    else:
        grid_b = np.linspace(psi_ab_outer, psi_bc_inner, n_linear)
        h_b_lo = float(grid_b[1] - grid_b[0])
        h_b_hi = h_b_lo
    grid_ab = buffer_grid(psi_a_outer, psi_ab_outer, h_a_outer, h_b_lo)
    grid_bc = buffer_grid(psi_bc_inner, psi_c_inner, h_b_hi, h_c_inner)
    zone_bounds = np.array([psi_a_outer, psi_ab_outer, psi_bc_inner, psi_c_inner])
    raw_merge = np.concatenate([grid_a, grid_ab, grid_b, grid_bc, grid_c])
    dedup_tol = min(h_a_outer, h_b_lo, h_c_inner) * 0.1
    keep = np.ones(len(raw_merge), dtype=bool)
    for zb in zone_bounds:
        near = np.abs(raw_merge - zb) < dedup_tol
        near[np.argmin(np.abs(raw_merge - zb))] = False  # keep the closest point
        keep &= ~near
    grid_all = np.unique(np.concatenate([raw_merge[keep], zone_bounds]))
    grid_all = grid_all[(grid_all >= q_lo) & (grid_all <= P0 - psi_anchor_low)]
    not_compact = float(q_lo_phys) <= float(psi_anchor_low)
    if g_table_floor < psi_anchor_low and not_compact and g_table_floor < q_lo:
        n_ext = int(np.ceil(g_table_nodes_per_decade * np.log10(q_lo / g_table_floor)))
        ext = np.logspace(
            np.log10(g_table_floor), np.log10(q_lo), n_ext, endpoint=False
        )
        grid_all = np.concatenate([ext, grid_all])
    q_ceil = float(P0) - float(q_ceil_gap)
    grid_all = grid_all[grid_all <= q_ceil]
    if len(grid_all) == 0 or grid_all[-1] < q_ceil:
        grid_all = np.append(grid_all, q_ceil)
    if np.any(np.diff(grid_all) <= 0):
        grid_all = np.unique(grid_all)
    return np.asarray(grid_all, dtype=np.float64)


def inversion_grid(
    profile: DoublePowerLaw,
    anisotropy: Anisotropy,
    settings: Settings,
    feature: Mapping[str, Any] | None = None,
) -> tuple[Array, GridZones]:
    """Return the energies the inversion tabulates its DF on, and the zones.

    Parameters
    ----------
    profile : DoublePowerLaw
        The component's profile (its potential, in units of P_0 = 1).
    anisotropy : Anisotropy
        Its anisotropy.
    settings : Settings
        The settings; the grid reads the ``inversion`` group's
        ``n_grid_rows``, ``n_grid_tail_rows``, ``grid_tail_extent`` and
        ``table_nodes_per_decade``.
    feature : mapping, optional
        The pilot inversion's feature of an OM Type II or COM model:
        ``q_feature``, ``case`` and ``kappa_low``.

    Raises
    ------
    InputError
        If the energies are not strictly increasing, a precondition of every
        stage after this one: the settings place nodes float64 cannot
        separate.
    """
    p0 = 1.0
    group = settings.inversion
    anchor = K_FLOOR
    extent = float(group["grid_tail_extent"])
    q_lo_phys = anchor
    family = anisotropy.family
    if family == "OM_I":
        psi_ra = float(profile.p(np.array([float(anisotropy.r_a)]))[0])
        zones = feature_zones(p0, extent, anchor, psi_ra, 1, -0.5)
    elif family in ("OM_II", "COM_I", "COM_II"):
        if feature is None:
            raise ValueError(
                f"An {family} model's grid needs its pilot inversion's feature."
            )
        zones = feature_zones(
            p0,
            extent,
            anchor,
            float(feature["q_feature"]),
            int(feature["case"]),
            float(feature["kappa_low"]),
        )
    else:
        zones = standard_zones(p0, extent, anchor)
    q = hybrid_grid(
        zones,
        int(group["n_grid_rows"]),
        int(group["n_grid_tail_rows"]),
        q_lo_phys,
        anchor,
        TABLE_FLOOR,
        int(group["table_nodes_per_decade"]),
        CEILING_GAP,
    )
    # the nodes around a full-domain COM Type II model's interior point
    power = 1.0 - float(anisotropy.beta_c)
    if family == "COM_II" and not (float(power).is_integer() and int(power) % 2 == 1):
        p_s = float(profile.p(np.array([float(anisotropy.r_a)]))[0])
        i = int(np.searchsorted(q, p_s))
        if 2 < i < len(q) - 3:
            gap = float(max(q[i - 1] - q[i - 2], q[i] - q[i - 1], q[i + 1] - q[i]))
            d = p_s * 10.0 ** -np.arange(6.0, -0.01, -0.25)
            d = d[d <= 2.0 * gap]
            if d.size:
                q = np.unique(np.concatenate([q, p_s - d, p_s + d]))
            refine = BAND_REFINE
            halfwidth = BAND_HALFWIDTH
            if refine > 1.0 and halfwidth > 0.0:
                hb = gap / refine
                kb = np.floor(halfwidth * p_s / hb)
                qb = p_s + hb * np.arange(-kb, kb + 0.5)
                qb = qb[(qb > q[0]) & (qb < q[-1]) & (np.abs(qb - p_s) > 0.5 * hb)]
                q = np.unique(np.concatenate([q, qb]))
    if np.any(np.diff(q) <= 0.0):
        raise InputError(
            "The inversion's energies are not strictly increasing: the grid's "
            f"settings place nodes float64 cannot separate (the ceiling gap "
            f"{CEILING_GAP!r})."
        )
    return np.asarray(q, dtype=np.float64), zones
