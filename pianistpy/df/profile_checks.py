"""The checks a model's profile carries of its own accuracy.

Each is read on the profile's rows through `TruncatedProfile`'s methods, so
that a model from a run and a model from a file are checked alike.

* ``poisson_residual``: Poisson's equation in integral form on every interval
  between consecutive rows, the edge row's included,
  ``Phi(r_a) - Phi(r_b) = INT M / r^2 dr`` and ``M(r_b) - M(r_a) = INT rho r^2 dr``,
  each relative to the larger of its two
  rows' values; the largest over the intervals.
* ``round_trip``: the density the model's own DF gives at a row,
  ``rho = C(beta_c) INT_0^Phi F(Q) (2 (Phi - Q))^(1/2 - beta_c) dQ`` over the
  anisotropy's factor in r, F through the lookup (the table and its forms),
  against the row's density, at the rows nearest each half-decade of radius,
  and at every row beyond the last of them to the last row before the edge;
  the largest relative difference.
* ``row_density``: the rows per bin the fills keep, at least
  ``profile_rows_per_decade`` per decade in bins of ``profile_bin_decades`` of
  the potential and of the depth, over the profile's own span, the fill's
  bins (`truncation.row_bin_edges`): the potential up to ROW_BINS_TOP times
  :math:`P_0`, the depth up to the same fraction of :math:`P_0`, whole bins
  only; a count, not an accuracy, so the check is silent where every bin is
  full and refused at any shortfall.
"""

__all__ = [
    "PROFILE_CHECK_KEYS",
    "density_from_df",
    "poisson_residual",
    "round_trip",
    "row_density",
    "profile_checks",
]

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..anisotropy import Anisotropy
from ..profiles import TruncatedProfile
from ..typing import Array
from ..validation import Check
from .lookup import below_floor_mass, evaluate_lookup, prepare_lookup
from .poisson import EVENT_ATOL, EVENT_RTOL, cuddeford_prefactor
from .truncation import K_FLOOR, ORIGINS, ROW_BINS_TOP, row_bin_edges

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
PROFILE_CHECK_KEYS = ("poisson_residual", "round_trip", "row_density")


def _p_0(profile: TruncatedProfile) -> float:
    """The central potential: the first row's potential and its depth."""
    u_0 = float(profile.u_trunc[0]) if profile.u_trunc.size else 0.0
    return float(profile.p_trunc[0]) + u_0


# the record of a check that formed its depths by subtraction
DEPTH_FALLBACK = (
    "depth formed from p_0 - Phi: no depth column for these rows (no u column, "
    "or the solve's u record does not match the rows)"
)


def _row_depths(profile: TruncatedProfile, n: int) -> Array | None:
    """The depths ``P_0 - Phi`` of the first `n` rows from the profile's
    column u_trunc, the solve's own, exact near the center where ``P_0 - Phi``
    has lost its digits; None where the column does not cover them, and the
    caller then forms ``P_0 - Phi``."""
    u = np.asarray(profile.u_trunc, dtype=np.float64)
    return u[:n] if u.size >= n else None


def _interior(profile: TruncatedProfile) -> int:
    return profile.r.size - (1 if profile.has_edge_row else 0)


def _event_tolerance(profile: TruncatedProfile) -> tuple[Array, Array]:
    """Each row's absolute tolerance in the potential and in the mass: zero
    on every row but the floor event's (by the profile's `origin`), where the
    root finder's tolerance on ln r maps through dPhi/d ln r = -M/r and
    dM/d ln r = r^3 rho."""
    r = np.asarray(profile.r, dtype=np.float64)
    a_phi, a_mass = np.zeros(r.size), np.zeros(r.size)
    if profile.origin.size == r.size:
        ev = np.flatnonzero(profile.origin == ORIGINS["event"])
        if ev.size:
            re = r[ev]
            d_ln_r = EVENT_ATOL + EVENT_RTOL * np.abs(np.log(re))
            m = np.asarray(profile.mass_trunc, dtype=np.float64)[ev]
            rho = np.sum(np.atleast_2d(profile.rho_trunc), axis=0)[ev]
            a_phi[ev] = m / re * d_ln_r
            a_mass[ev] = re**3 * rho * d_ln_r
    return a_phi, a_mass


def poisson_residual(profile: TruncatedProfile, rtol: float) -> Check:
    """Return Poisson's equation's largest residual over every interval of
    the profile (the edge row's included), relative to what the solve
    promised there: the larger potential of its two rows plus the larger of
    their absolute tolerances over the solve's relative tolerance ``rtol``
    (`_event_tolerance`: zero but on the floor event's row), and the mass
    likewise."""
    r = np.asarray(profile.r, dtype=np.float64)
    a, b = r[:-1], r[1:]
    half = 0.5 * (b - a)
    rq = (0.5 * (a + b))[:, None] + half[:, None] * _GL_X[None, :]
    flat = rq.ravel()
    mass_q = np.asarray(profile.mass(flat)).reshape(rq.shape)
    rho_q = np.sum(np.atleast_2d(profile.density(flat)), axis=0).reshape(rq.shape)
    i_phi = half * np.sum(_GL_W[None, :] * mass_q / rq**2, axis=1)
    i_mass = half * np.sum(_GL_W[None, :] * rho_q * rq**2, axis=1)
    phi_a, phi_b = np.asarray(profile.potential(a)), np.asarray(profile.potential(b))
    m_a, m_b = np.asarray(profile.mass(a)), np.asarray(profile.mass(b))
    a_phi, a_mass = _event_tolerance(profile)
    e_phi = np.abs((phi_a - phi_b) - i_phi) / (
        np.maximum(np.abs(phi_a), np.abs(phi_b))
        + np.maximum(a_phi[:-1], a_phi[1:]) / rtol
    )
    e_mass = np.abs((m_b - m_a) - i_mass) / (
        np.maximum(np.abs(m_a), np.abs(m_b))
        + np.maximum(a_mass[:-1], a_mass[1:]) / rtol
    )
    e = np.maximum(e_phi, e_mass)
    j = int(np.nanargmax(e))
    return Check(error=float(e[j]), where=float(b[j]))


def _nearest(x: Array, targets: Array) -> Array:
    """The indices of the entries of increasing `x` nearest each target."""
    i = np.clip(np.searchsorted(x, targets), 1, x.size - 1)
    return np.where(np.abs(x[i - 1] - targets) <= np.abs(x[i] - targets), i - 1, i)


def _check_radii(profile: TruncatedProfile, phi_min: float) -> Array:
    """The rows nearest each half-decade of radius, and every row beyond the
    last of them to the last row (on a truncated model the floor event's):
    the rows approaching an edge are dense in the potential and not in the
    radius, and a sampling of them can miss a peak a few rows wide. The edge
    row, whose potential and density are zero, is not one."""
    n = _interior(profile)
    r = np.asarray(profile.r[:n], dtype=np.float64)
    phi = np.asarray(profile.p_trunc[:n], dtype=np.float64)
    r = r[phi >= phi_min]
    log_r = np.log10(r)
    targets = np.arange(np.ceil(2.0 * log_r[0]) / 2.0, log_r[-1] + 1e-12, 0.5)
    i = _nearest(log_r, targets)
    return r[np.unique(np.r_[i, np.arange(int(np.max(i)), r.size)])]


def round_trip(
    profile: TruncatedProfile,
    table: Any,
    form: Any,
    anisotropy: Anisotropy,
) -> Check:
    """Return the largest relative difference between the density the
    model's DF gives (its table and its forms, through the lookup) and the
    profile's, at the rows `_check_radii` names."""
    radii = _check_radii(profile, 0.0)
    phi = np.asarray(profile.potential(radii), dtype=np.float64)
    # the rows' own depths from the column (by the depth, the center keeps the
    # digits Phi loses); without the column, P_0 - Phi
    depth = None
    n = _interior(profile)
    u_rows = _row_depths(profile, n)
    if u_rows is not None:
        rows = np.asarray(profile.r[:n], dtype=np.float64)
        i = np.clip(np.searchsorted(rows, radii), 0, n - 1)
        depth = np.where(rows[i] == radii, u_rows[i], np.nan)
    log_df = _log_density_from_df(radii, phi, table, form, anisotropy, depth)
    rho = np.sum(np.atleast_2d(profile.density(radii)), axis=0)
    # in logarithms, so a row whose density is finite is read wherever the
    # DF's integral would leave float64 on the way
    e = np.abs(np.expm1(log_df - np.log(rho)))
    j = int(np.nanargmax(e))
    source = DEPTH_FALLBACK if u_rows is None else None
    return Check(error=float(e[j]), where=float(radii[j]), source=source)


_PANEL_X, _PANEL_W = np.polynomial.legendre.leggauss(64)


def _gauss(lo: Array, hi: Array) -> tuple[Array, Array]:
    """Nodes and weights of 64-point Gauss-Legendre on each panel [lo, hi]."""
    half = 0.5 * (hi - lo)
    x = (0.5 * (lo + hi))[..., None] + half[..., None] * _PANEL_X
    return x, half[..., None] * _PANEL_W


# the radii read together, a chunk at a time: some 4e6 nodes of the panels
_AUX_CHUNK = 2048


def _log_aux(
    phi: Array,
    first: float,
    arrays: Any,
    kinds: Any,
    n: float,
    depth: Array,
) -> Array:
    """ln rho_aux(Phi) at each Phi, rho_aux = INT_0^Phi F(Q) (2 (Phi - Q))^n dQ
    through the lookup: below ``Q_lo = 1e-20 Phi`` the below-floor form's mass
    in closed form; to 0.9 Phi panels of a decade in ln Q; above, Q = Phi - u^k
    (k = 2 where ``n >= 0``, 1 / (1 + n) otherwise) on panels geometric in u to
    the end. The terms are summed in logarithms with the largest factored out.

    Where `depth`, ``P_0 - Phi``, is held (finite; NaN where not), the DF
    above 0.9 Phi is read by its depth, depth + u^k, not by Q, whose
    difference from :math:`P_0` keeps one ulp of :math:`P_0` only.

    The radii are read together, those of one panel count at a time, in
    chunks of `_AUX_CHUNK`: one lookup call a chunk."""
    phi = np.asarray(phi, dtype=np.float64).ravel()
    depth = np.asarray(depth, dtype=np.float64).ravel()
    out = np.full(phi.size, -np.inf)
    q_lo, split = 1e-20 * phi, 0.9 * phi
    below = np.where(q_lo < first, q_lo, first)
    count = np.ceil(np.log10(split / below)).astype(int)
    for c in np.unique(count):
        rows = np.flatnonzero(count == c)
        for start in range(0, rows.size, _AUX_CHUNK):
            i = rows[start : start + _AUX_CHUNK]
            out[i] = _log_aux_rows(
                phi[i], below[i], split[i], depth[i], int(c), arrays, kinds, n
            )
    return out


def _log_aux_rows(
    phi: Array,
    below: Array,
    split: Array,
    depth: Array,
    count: int,
    arrays: Any,
    kinds: Any,
    n: float,
) -> Array:
    """`_log_aux` on rows of one panel count `count` below 0.9 Phi."""
    ln10 = np.log(10.0)
    p = phi[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mass = np.where(
            below > 0.0,
            below_floor_mass(arrays, kinds, np.where(below > 0.0, below, 1.0)),
            0.0,
        )
        t_mass = np.where(mass > 0.0, np.log(mass) + n * np.log(2.0 * phi), -np.inf)
        edges = np.linspace(np.log(below), np.log(split), count + 1, axis=1)
        x, w = _gauss(edges[:, :-1], edges[:, 1:])
        q = np.exp(x)
        log_f = ln10 * np.asarray(
            evaluate_lookup(arrays, kinds, q.ravel(), np), dtype=np.float64
        ).reshape(q.shape)
        t_lo = np.log(w) + log_f + x + n * np.log(2.0 * (p - q))
        k = 2.0 if n >= 0.0 else 1.0 / (1.0 + n)
        u_max = (phi - split) ** (1.0 / k)
        u_edges = np.concatenate(
            [np.zeros((phi.size, 1)), u_max[:, None] * np.logspace(-12.0, 0.0, 13)],
            axis=1,
        )
        x, w = _gauss(u_edges[:, :-1], u_edges[:, 1:])
        q = p - x**k
        by_depth = np.where(
            np.isfinite(depth)[:, None, None], depth[:, None, None] + x**k, np.nan
        )
        log_f = ln10 * np.asarray(
            evaluate_lookup(arrays, kinds, q.ravel(), np, u=by_depth.ravel()),
            dtype=np.float64,
        ).reshape(q.shape)
        t_hi = (
            np.log(w)
            + log_f
            + np.log(k)
            + (k - 1.0) * np.log(x)
            + n * np.log(2.0 * x**k)
        )
        t = np.concatenate(
            [
                t_mass[:, None],
                t_lo.reshape(phi.size, -1),
                t_hi.reshape(phi.size, -1),
            ],
            axis=1,
        )
        t = np.where(np.isfinite(t), t, -np.inf)
        top = np.max(t, axis=1)
        held = np.isfinite(top)
        top_s = np.where(held, top, 0.0)
        total = top_s + np.log(np.sum(np.exp(t - top_s[:, None]), axis=1))
    return np.where(held, total, -np.inf)


def _log_density_from_df(
    radii: Any,
    phi: Any,
    table: Any,
    form: Any,
    anisotropy: Anisotropy,
    depth: Any = None,
) -> Array:
    """ln of the density a model's DF gives at the radii and potentials:
    ln rho_aux through the lookup (`_log_aux`) plus the logarithm of the
    anisotropy's factor g(R).

    The DF near the top is read by the depth ``P_0 - Phi``: `depth` where
    the caller holds it exactly (NaN where not), else ``P_0 - Phi``, exact
    where ``Phi >= P_0 / 2``; below, by the energy, as nothing there nears
    :math:`P_0`."""
    radii = np.asarray(radii, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    p_0 = float(form.p_0)
    held = np.where(phi >= 0.5 * p_0, p_0 - phi, np.nan)
    if depth is not None:
        depth = np.broadcast_to(np.asarray(depth, dtype=np.float64), phi.shape)
        held = np.where(np.isfinite(depth), depth, held)
    bc = float(anisotropy.beta_c)
    model_type = _model_type(anisotropy)
    n = 0.5 if model_type in ("OM_I", "OM_II") else 0.5 - bc
    arrays, kinds = prepare_lookup(table, form, 0)
    first = float(table.k[0, 0])
    log_aux = _log_aux(phi, first, arrays, kinds, n, held).reshape(phi.shape)
    # the one definition of g(R): rho_aug = g rho_aux
    log_g = np.log(anisotropy.factor(radii))
    if model_type in ("OM_I", "OM_II"):
        return np.log(4.0 * np.pi) + log_aux + log_g
    return np.log(cuddeford_prefactor(bc)) + log_aux + log_g


def density_from_df(
    radii: Any,
    phi: Any,
    table: Any,
    form: Any,
    anisotropy: Anisotropy,
    depth: Any = None,
) -> Array:
    """Return the density a model's DF gives at the radii and potentials:
    rho_aux through the lookup times the anisotropy's factor g(R), formed in
    logarithms (`_log_density_from_df`). `depth`, ``P_0 - Phi`` where the
    caller holds it exactly (NaN where not), keeps the center's digits that
    Phi has lost; given Phi only, the density is read to one ulp of
    :math:`P_0` in the depth."""
    return np.exp(_log_density_from_df(radii, phi, table, form, anisotropy, depth))


def row_density(profile: TruncatedProfile, settings: Mapping[str, Any]) -> Check:
    """Return the row-count rule's check: error 0 where every bin of the
    potential and of the depth holds its rows, infinite at any shortfall,
    with the bin's upper edge (in the potential or the depth) as where.

    The bins are the fills', from the one definition both read
    (`truncation.row_bin_edges`): decades of ``profile_bin_decades`` from the
    floor. Only the bins whose lower edge lies in the profile's own span are
    counted: in the potential from its lowest row (a spatially non-finite
    model's potential stays above the floor), in the depth from its first
    row's. Only whole bins are judged: a depth bin reaching beyond
    ROW_BINS_TOP times :math:`P_0`, the deepest depth counted, is not."""
    n = _interior(profile)
    phi = np.asarray(profile.p_trunc[:n], dtype=np.float64)
    kept = np.isfinite(phi) & (phi >= K_FLOOR)
    phi = phi[kept]
    p_0 = _p_0(profile)
    # the depths from the column, without it P_0 - Phi
    u_rows = _row_depths(profile, n)
    u = u_rows[kept] if u_rows is not None else p_0 - phi
    source = DEPTH_FALLBACK if u_rows is None else None
    u = u[(u > 0) & (u <= ROW_BINS_TOP * p_0)]
    bins = row_bin_edges(p_0, settings, float(np.max(u)) if u.size else 0.0)
    need, edges = bins.min_per_bin, bins.psi_edges
    count, _ = np.histogram(phi, bins=edges)
    inside = edges[:-1] >= (float(np.min(phi)) if phi.size else np.inf)
    short = np.flatnonzero(inside & (count < need))
    if short.size:
        return Check(
            error=float("inf"), where=float(edges[short[-1] + 1]), source=source
        )
    u_first = max(K_FLOOR, float(profile.u_trunc[0]) if profile.u_trunc.size else 0.0)
    if bins.u_edges is not None and bins.u_top > u_first:
        edges_u = bins.u_edges
        count_u, _ = np.histogram(u, bins=edges_u)
        # whole bins only: a bin reaching beyond the depths counted is not judged
        whole = edges_u[1:] <= ROW_BINS_TOP * p_0 * (1.0 + 1e-12)
        inside_u = (edges_u[:-1] >= u_first) & whole
        short = np.flatnonzero(inside_u & (count_u < need))
        if short.size:
            where = p_0 - float(edges_u[short[-1] + 1])
            return Check(error=float("inf"), where=where, source=source)
    return Check(error=0.0, where=None, source=source)


def _model_type(anisotropy: Anisotropy) -> str:
    """The kernel's family: isotropic or constant (the same branch), or
    Osipkov-Merritt or Cuddeford-Osipkov-Merritt of the anisotropy's kind."""
    if not np.isfinite(anisotropy.r_a):
        return "constant"
    family = "OM" if anisotropy.beta_c == 0.0 else "COM"
    return f"{family}_{anisotropy.kind}"


def profile_checks(
    profile: TruncatedProfile,
    table: Any,
    form: Any,
    anisotropy: Anisotropy,
    settings: Mapping[str, Any],
) -> dict[str, Check]:
    """Return the profile's three checks by their report keys; the round
    trip is not measured (NaN) on a signed table, which has no lookup."""
    checks = {
        "poisson_residual": poisson_residual(profile, float(settings["ode_rtol"]))
    }
    if table.sign.size:
        checks["round_trip"] = Check(error=float("nan"), where=None)
    else:
        checks["round_trip"] = round_trip(profile, table, form, anisotropy)
    checks["row_density"] = row_density(profile, settings)
    return checks
