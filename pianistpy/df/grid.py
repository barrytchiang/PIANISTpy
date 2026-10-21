r"""Return the energies on which the inversion tabulates a DF.

`inversion_grid` returns the energies :math:`Q` of the DF table of one
component, with the central potential :math:`P_0 = 1`. The grid runs from
the potential floor `K_FLOOR` to :math:`P_0 - \delta`, with :math:`\delta`
equal to `CEILING_GAP`, in five zones: zone A is log-spaced in :math:`Q`,
zone AB is a buffer whose step changes geometrically, zone B is linear in
:math:`Q`, zone BC is a second geometric buffer, and zone C is log-spaced
in :math:`P_0 - Q`. No node lies above :math:`P_0 - \delta`. The closed form
of the DF at the center covers the energies above the last node.

An isotropic or constant-anisotropy model uses fixed fractions of
:math:`P_0` as zone boundaries (`standard_zones`). An OM Type I model
places zones A and AB around the potential at its anisotropy radius
:math:`R_{\rm a}`. An OM Type II or COM model places zones A and AB around
the energy of the feature its pilot inversion finds (`feature_zones`). A
COM Type II model whose :math:`1 - \beta_{\rm c}` is not an odd integer,
with :math:`\beta_{\rm c}` the central anisotropy, gets extra nodes around
its interior point :math:`P(R_{\rm a})`.
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

# The lowest energy of the inversion's table equals the grid's floor
# K_FLOOR, so `hybrid_grid` adds no nodes below K_FLOOR for the inversion.
TABLE_FLOOR = K_FLOOR
# The written DF table starts at the first grid node at or above
# OUTPUT_FLOOR_FACTOR times K_FLOOR.
OUTPUT_FLOOR_FACTOR = 2.0
# The node spacing in the band around a COM Type II interior point is the
# largest local grid spacing divided by BAND_REFINE.
BAND_REFINE = 5.0
# The band around a COM Type II interior point P(R_a) spans P(R_a) plus or
# minus BAND_HALFWIDTH times P(R_a).
BAND_HALFWIDTH = 0.5


@dataclass(frozen=True)
class GridZones:
    r"""The boundaries of the five zones of the energy grid.

    Parameters
    ----------
    p0 : float
        Central potential :math:`P_0`.
    a_lo : float
        Lower bound of zone A.
    a_hi : float
        Boundary between zone A and zone AB.
    ab_hi : float
        Boundary between zone AB and zone B.
    bc_lo : float
        Boundary between zone B and zone BC.
    c_lo : float
        Boundary between zone BC and zone C.
    c_hi : float
        Upper bound of zone C.
    q_feature : float
        Energy of the feature that zones A and AB are placed around: the
        potential :math:`P(R_{\rm a})` at the anisotropy radius for an OM
        Type I model, and the pilot inversion's feature for an OM Type II
        or COM model. Zero for the standard zones.
    """

    p0: float
    a_lo: float
    a_hi: float
    ab_hi: float
    bc_lo: float
    c_lo: float
    c_hi: float
    q_feature: float


def standard_zones(p0: float, psi_extent: float, psi_anchor_low: float) -> GridZones:
    """Return the zones of an isotropic or constant-anisotropy model.

    The boundaries are fixed fractions of the central potential ``p0``:
    zone A ends at ``psi_extent * p0``, zone AB at ``3/8 * p0``, zone B at
    ``5/8 * p0`` and zone BC at ``(1 - psi_extent) * p0``. Zone A starts at
    ``psi_anchor_low`` and zone C ends at ``p0 - psi_anchor_low``.
    """
    return GridZones(
        p0=float(p0),
        a_lo=float(psi_anchor_low),
        a_hi=float(p0) * float(psi_extent),
        ab_hi=float(p0) * 3.0 / 8.0,
        bc_lo=float(p0) * 5.0 / 8.0,
        c_lo=float(p0) * (1.0 - float(psi_extent)),
        c_hi=float(p0) - float(psi_anchor_low),
        q_feature=0.0,
    )


def feature_zones(
    p0: float,
    psi_extent: float,
    psi_anchor_low: float,
    q_feature: float,
) -> GridZones:
    """Return the zones of an OM or COM model, placed around a feature.

    For a positive ``q_feature``, zone A ends at ``0.7 * q_feature`` and
    zone AB at ``min(1.5 * q_feature, 3/8 * p0)``. The end of zone A is then
    raised to at least ``100 * psi_anchor_low`` and lowered to at most
    ``0.9`` times the end of zone AB. Where the order
    ``100 * psi_anchor_low < end of A < end of AB < 5/8 * p0`` fails, or
    where ``q_feature`` is not positive, zones A and AB take the boundaries
    of `standard_zones`. The boundaries from ``5/8 * p0`` upward are always
    those of `standard_zones`.
    """
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
    )


def buffer_grid(psi_lo: float, psi_hi: float, h_lo: float, h_hi: float) -> Array:
    """Return a grid over [psi_lo, psi_hi] whose step changes geometrically.

    The step goes geometrically from h_lo to h_hi. The number of steps N is
    the one whose unscaled steps sum closest to the width psi_hi - psi_lo,
    and the steps are then rescaled to sum to the width exactly. Where
    abs(ln(h_hi / h_lo)) is below :math:`10^{-10}`, the grid has
    max(2, round(width / h_lo)) equal steps. Where psi_hi is not above
    psi_lo, or h_lo or h_hi is not positive, the grid is the two end points.
    """
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
    r"""Return the five-zone grid from q_lo to :math:`P_0` minus `q_ceil_gap`.

    The lowest energy q_lo is the larger of ``zones.a_lo`` and `q_lo_phys`,
    and the central potential :math:`P_0` is ``zones.p0``. Zone A holds
    `n_log` nodes log-spaced in Q, and zone C holds `n_log` nodes log-spaced
    in :math:`P_0 - Q`. Zone B holds `n_linear` linear nodes. Where zone B
    starts below :math:`0.01 P_0` and ends above :math:`0.05 P_0`, zone B
    starts with nodes log-spaced at 120 per decade (at least 8) up to
    :math:`0.05 P_0`. Zones AB and BC are `buffer_grid` buffers whose end
    steps match the steps of the neighboring zones.

    Nodes closer to a zone boundary than a tenth of the smallest of zone A's
    last step, zone B's first step and zone C's first step are removed,
    except the node closest to the boundary, and the zone boundaries are
    added as nodes. Nodes below q_lo or above :math:`P_0` minus
    `psi_anchor_low` are then removed. Where `g_table_floor` lies below
    `psi_anchor_low` and below q_lo, and `q_lo_phys` is not above
    `psi_anchor_low`, nodes log-spaced at `g_table_nodes_per_decade` nodes
    per decade are added from `g_table_floor` up to q_lo. Nodes above :math:`P_0` minus
    `q_ceil_gap` are removed, and :math:`P_0` minus `q_ceil_gap` is the last
    node.
    """
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
        near[np.argmin(np.abs(raw_merge - zb))] = False  # The closest node stays.
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
    """Return the energies on which the inversion tabulates a DF, and the zones.

    Parameters
    ----------
    profile : DoublePowerLaw
        Profile of the component, with its potential in units of the
        central potential :math:`P_0`, so that :math:`P_0 = 1`.
    anisotropy : Anisotropy
        Anisotropy model of the component.
    settings : Settings
        Numerical settings. The grid reads ``n_grid_rows`` (the linear nodes
        of zone B), ``n_grid_tail_rows`` (the nodes of zone A and of zone C),
        ``grid_tail_extent`` and ``table_nodes_per_decade`` of the
        ``inversion`` group.
    feature : mapping, optional
        The pilot inversion's result for an OM Type II or COM model, read
        as ``feature["q_feature"]`` (`invert.pilot_feature`). Default:
        ``None``, which is valid only for the other families.

    Returns
    -------
    q : array
        The energies, strictly increasing, in internal units.
    zones : GridZones
        The zone boundaries the grid is built on.

    Raises
    ------
    ValueError
        If the family is OM Type II, COM Type I or COM Type II and `feature`
        is None.
    """
    p0 = 1.0
    group = settings.inversion
    anchor = K_FLOOR
    extent = float(group["grid_tail_extent"])
    q_lo_phys = anchor
    family = anisotropy.family
    if family == "OM_I":
        psi_ra = float(profile.p(np.array([float(anisotropy.r_a)]))[0])
        zones = feature_zones(p0, extent, anchor, psi_ra)
    elif family in ("OM_II", "COM_I", "COM_II"):
        if feature is None:
            raise ValueError(
                f"An {family} model's grid needs its pilot inversion's feature."
            )
        zones = feature_zones(p0, extent, anchor, float(feature["q_feature"]))
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
    # The derivative of a COM Type II model's auxiliary density is not smooth
    # at the interior point P(R_a), unless 1 - beta_c is an odd integer. The
    # grid gets nodes at P(R_a) plus and minus P(R_a) 10^-k, for k from 6 to 0
    # in steps of 0.25 and offsets no larger than twice the largest of the
    # three grid spacings nearest P(R_a), and a band of evenly spaced nodes
    # around P(R_a) (BAND_REFINE, BAND_HALFWIDTH). Both node sets are added
    # only where at least three grid nodes lie below P(R_a) and at least four
    # lie at or above P(R_a).
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
