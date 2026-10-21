r"""Return the checks that a model's profile carries of its own accuracy.

`profile_checks` returns three checks, each read on the profile's rows
through the methods of `TruncatedProfile`, so that a model from a run and
a model from a file are checked alike.

* ``poisson_residual``: Poisson's equation in integral form on every
  interval between consecutive rows at radii :math:`R_1 < R_2`, the
  interval to the edge row included:
  :math:`\Phi(R_1) - \Phi(R_2) = \int_{R_1}^{R_2} M / R^2 \, dR` and
  :math:`M(R_2) - M(R_1) = \int_{R_1}^{R_2} \rho R^2 \, dR`, with
  :math:`\Phi` the relative potential, :math:`M` the enclosed mass and
  :math:`\rho` the density, all in internal units. The potential's residual
  is relative to the larger absolute potential at :math:`R_1` and
  :math:`R_2`, and the mass's residual to the larger mass, each plus an
  allowance on the floor event's row (`poisson_residual`). The check's value
  is the largest residual over the intervals.
* ``round_trip``: the density that the model's own DF gives at a row,
  :math:`C(\beta_{\rm c}) \int_0^{\Phi} F(Q) \, (2 (\Phi - Q))^{1/2 -
  \beta_{\rm c}} \, dQ` times the anisotropy factor :math:`g(R)`, against
  the row's density. Here :math:`F` is the DF read through the lookup of
  the table and its forms, :math:`Q` the energy, :math:`\beta_{\rm c}` the
  central anisotropy and :math:`C(\beta_{\rm c})` the prefactor of the
  density integral. The rows checked are the rows nearest each half-decade
  of radius and every row beyond the last of those rows up to the last row
  before the edge row. The check's value is the largest relative
  difference.
* ``row_density``: the number of rows per bin that the fills keep. Each bin
  of ``profile_bin_decades`` decades must hold at least
  ``profile_rows_per_decade`` times ``profile_bin_decades`` rows, rounded,
  in the potential and in the depth :math:`P_0 - \Phi`, over the profile's
  own span (`truncation.row_bin_edges`). Here :math:`P_0` is the central
  potential. A depth bin whose upper edge lies above `ROW_BINS_TOP` (0.1)
  times :math:`P_0` is not judged. The row-density check
  counts rows and measures no accuracy: the check's value is 0 where every
  bin is full and infinite where any judged bin holds too few rows, which
  places the check in the "refused" tier.
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
    """Return the central potential: the first row's potential plus its depth.

    The depth is the first entry of the column ``u_trunc``, and 0 where the
    column is empty.
    """
    u_0 = float(profile.u_trunc[0]) if profile.u_trunc.size else 0.0
    return float(profile.p_trunc[0]) + u_0


# The source note of a check that formed its depths as p_0 - Phi, because the
# profile has no depth column that matches its rows.
DEPTH_FALLBACK = (
    "depth formed from p_0 - Phi: no depth column for these rows (no u column, "
    "or the solve's u record does not match the rows)"
)


def _row_depths(profile: TruncatedProfile, n: int) -> Array | None:
    """Return the depths ``P_0 - Phi`` of the first `n` rows from ``u_trunc``.

    The column ``u_trunc`` holds the solve's own depths, which keep their
    relative precision near the center, where the difference ``P_0 - Phi``
    has lost its digits. `_row_depths` returns None where the column has
    fewer than `n` entries, and the caller then forms ``P_0 - Phi``.
    """
    u = np.asarray(profile.u_trunc, dtype=np.float64)
    return u[:n] if u.size >= n else None


def _interior(profile: TruncatedProfile) -> int:
    """Return the number of rows before the edge row, or all rows without one."""
    return profile.r.size - (1 if profile.has_edge_row else 0)


def _event_tolerance(profile: TruncatedProfile) -> tuple[Array, Array]:
    """Return each row's absolute tolerance in the potential and in the mass.

    Both tolerances are zero on every row except the floor event's row,
    found by the profile's `origin`. On the floor event's row, the root
    finder's tolerance d ln r = EVENT_ATOL + EVENT_RTOL |ln r| maps to the
    potential through |dPhi/d ln r| = M/r and to the mass through
    dM/d ln r = r^3 rho.
    """
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
    """Return the largest residual of Poisson's equation over the profile.

    The integrals over each interval between consecutive rows, the interval
    to the edge row included, use 8-point Gauss-Legendre quadrature. The
    potential's residual on an interval is relative to the larger absolute
    potential of the interval's two rows plus the larger of the two rows'
    absolute tolerances in the potential divided by ``rtol``
    (`_event_tolerance`). The mass's residual on an interval is relative to
    the larger absolute mass of the interval's two rows plus the larger of
    the two rows' absolute tolerances in the mass divided by ``rtol``. On
    each interval, the residual is the larger of the potential's and the
    mass's residuals. The check's value is the largest residual over the
    intervals, and the check's location is the outer radius of the interval
    with the largest residual.

    Parameters
    ----------
    profile : TruncatedProfile
        The profile to check.
    rtol : float
        Relative tolerance of the Poisson solve (``ode_rtol``).
    """
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
    """Return the indices of the entries of increasing `x` nearest each target."""
    i = np.clip(np.searchsorted(x, targets), 1, x.size - 1)
    return np.where(np.abs(x[i - 1] - targets) <= np.abs(x[i] - targets), i - 1, i)


def _check_radii(profile: TruncatedProfile, phi_min: float) -> Array:
    """Return the radii of the rows on which the round trip is checked.

    The radii are the rows nearest each half-decade of radius, and every row
    from the last of those rows to the last row before the edge row (on a
    truncated model, the floor event's row). The rows approaching an edge
    are dense in the potential and sparse in the radius, so a sampling by
    radius alone can miss an error peak a few rows wide. Only rows whose
    potential is at or above ``phi_min`` count. The edge row, whose
    potential and density are zero, is never checked.
    """
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
    """Return the round-trip check of a model's DF against its profile.

    The check's value is the largest relative difference between the density
    that the model's DF gives, through the lookup of its table and forms,
    and the profile's density, at the rows that `_check_radii` returns. The
    check records `DEPTH_FALLBACK` as its source where the profile has no
    matching depth column.
    """
    radii = _check_radii(profile, 0.0)
    phi = np.asarray(profile.potential(radii), dtype=np.float64)
    # The DF near the center is read by the rows' own depths from the depth
    # column, which keep the digits that Phi loses near P_0. Without the
    # depth column, _log_density_from_df forms the depth as P_0 - Phi where
    # Phi >= P_0 / 2, and reads the DF by the energy elsewhere.
    depth = None
    n = _interior(profile)
    u_rows = _row_depths(profile, n)
    if u_rows is not None:
        rows = np.asarray(profile.r[:n], dtype=np.float64)
        i = np.clip(np.searchsorted(rows, radii), 0, n - 1)
        depth = np.where(rows[i] == radii, u_rows[i], np.nan)
    log_df = _log_density_from_df(radii, phi, table, form, anisotropy, depth)
    rho = np.sum(np.atleast_2d(profile.density(radii)), axis=0)
    # The difference is formed from logarithms, so that a row with a finite
    # density is checked even where the DF's integral in linear form would
    # leave the range of float64.
    e = np.abs(np.expm1(log_df - np.log(rho)))
    j = int(np.nanargmax(e))
    source = DEPTH_FALLBACK if u_rows is None else None
    return Check(error=float(e[j]), where=float(radii[j]), source=source)


_PANEL_X, _PANEL_W = np.polynomial.legendre.leggauss(64)


def _gauss(lo: Array, hi: Array) -> tuple[Array, Array]:
    """Return the nodes and weights of 64-point Gauss-Legendre on [lo, hi]."""
    half = 0.5 * (hi - lo)
    x = (0.5 * (lo + hi))[..., None] + half[..., None] * _PANEL_X
    return x, half[..., None] * _PANEL_W


# The number of radii in one lookup call of `_log_aux`. Each radius has at
# least 33 panels of 64 nodes, so one chunk holds at least 4.3e6 nodes; the
# chunk size limits the size of the arrays of one lookup call.
_AUX_CHUNK = 2048


def _log_aux(
    phi: Array,
    first: float,
    arrays: Any,
    kinds: Any,
    n: float,
    depth: Array,
) -> Array:
    r"""Return :math:`\ln \rho_{\rm aux}(\Phi)` at each potential :math:`\Phi`.

    The integral is
    :math:`\rho_{\rm aux} = \int_0^{\Phi} F(Q) \, (2 (\Phi - Q))^n \, dQ`,
    with :math:`F` the DF read through the lookup of `arrays` and `kinds`,
    :math:`Q` the energy and :math:`n` the argument `n`, in three parts.
    Below :math:`Q_{\rm b}`, the smaller of :math:`10^{-20} \Phi` and the
    table's first energy `first`, the part is the closed-form mass of
    the below-floor form times :math:`(2 \Phi)^n`. From :math:`Q_{\rm b}` to
    :math:`0.9 \Phi`, the part is Gauss-Legendre quadrature on panels equal
    in :math:`\ln Q`, each at most a decade wide. Above :math:`0.9 \Phi`,
    the substitution :math:`Q = \Phi - u^k`, with :math:`k = 2` where
    :math:`n \geq 0` and :math:`k = 1 / (1 + n)` otherwise, makes the
    integrand finite at :math:`u = 0`. The part above :math:`0.9 \Phi` is
    Gauss-Legendre quadrature on 13 panels in :math:`u`: the first panel
    from 0 to :math:`10^{-12}` times the largest :math:`u`, and each later
    panel a decade wide. The three parts' quadrature terms are summed in
    logarithms with the largest term factored out.

    Where `depth`, the depth :math:`P_0 - \Phi` with :math:`P_0` the central
    potential, is finite, the DF above :math:`0.9 \Phi` is read by the
    node's depth :math:`P_0 - \Phi + u^k`, and not by :math:`Q`: the
    difference :math:`P_0 - Q` formed from :math:`Q` has an absolute error
    of one ulp of :math:`P_0`. NaN entries of `depth` mark the potentials
    whose DF is read by :math:`Q`.

    The potentials with the same number of panels below :math:`0.9 \Phi`
    are read together, in chunks of `_AUX_CHUNK` potentials, with one lookup
    call per chunk.
    """
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
    """Return `_log_aux` for potentials that share the panel count `count`."""
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
    r"""Return the logarithm of the density that a model's DF gives.

    The log density is the log of the prefactor (:math:`4 \pi` for OM
    models, `cuddeford_prefactor` otherwise), plus
    :math:`\ln \rho_{\rm aux}` through the lookup (`_log_aux`), plus the
    logarithm of the anisotropy factor :math:`g(R)`.

    Near the center, the DF is read by the depth :math:`P_0 - \Phi`, with
    :math:`P_0` the central potential and :math:`\Phi` the potential. The
    depth is `depth` where the caller gives a finite value, and
    :math:`P_0 - \Phi` where :math:`\Phi \geq P_0 / 2`, where the
    subtraction is exact in float64. For :math:`\Phi < P_0 / 2` without a
    given depth, the DF is read by the energy, because every energy of the
    integral is at most :math:`\Phi`, below :math:`P_0 / 2`.
    """
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
    # The density is the DF's integral times the anisotropy factor g(R) of
    # `Anisotropy.factor`, the package's one definition of g(R).
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
    r"""Return the density that a model's DF gives at given radii and potentials.

    The density is the exponential of `_log_density_from_df`: the DF's
    integral through the lookup times the anisotropy factor :math:`g(R)`.
    The argument `depth` holds the depth :math:`P_0 - \Phi` where the caller
    holds the depth to full relative precision, and NaN elsewhere, with
    :math:`P_0` the central potential and :math:`\Phi` the potential.
    `depth` keeps the digits near the center that :math:`\Phi` has lost.
    Without `depth`, the depth near the center is known to one ulp of
    :math:`P_0` only.
    """
    return np.exp(_log_density_from_df(radii, phi, table, form, anisotropy, depth))


def row_density(profile: TruncatedProfile, settings: Mapping[str, Any]) -> Check:
    r"""Return the check of the number of rows per bin.

    The check's value is 0 where every judged bin of the potential and of
    the depth holds at least ``RowBins.min_per_bin`` rows, and infinite
    where a judged bin holds fewer. The potential's bins are judged first.
    The check's location is the upper edge of the last potential bin that
    holds too few rows or, where every potential bin is full, the potential
    :math:`P_0` minus the upper edge of the last depth bin that holds too
    few rows. Here :math:`P_0` is the central potential.

    The bins are those of `truncation.row_bin_edges`, which the fills also
    read: bins of ``profile_bin_decades`` decades from `K_FLOOR`. Only the
    bins whose lower edge lies in the profile's own span are judged. In the
    potential, the span starts at the profile's lowest potential at or
    above `K_FLOOR` (a spatially non-finite model's potential stays above
    `K_FLOOR`). In the depth, the span starts at the larger of `K_FLOOR`
    and the first row's depth. A depth bin whose upper edge lies above
    `ROW_BINS_TOP` times :math:`P_0`, the deepest depth counted, is not
    judged.
    """
    n = _interior(profile)
    phi = np.asarray(profile.p_trunc[:n], dtype=np.float64)
    kept = np.isfinite(phi) & (phi >= K_FLOOR)
    phi = phi[kept]
    p_0 = _p_0(profile)
    # The depths come from the depth column, or from P_0 - Phi where the
    # profile has no depth column for its rows.
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
        # Only whole depth bins count: a depth bin whose upper edge lies above
        # ROW_BINS_TOP * P_0, with a relative slack of 1e-12 for rounding, is
        # not judged.
        whole = edges_u[1:] <= ROW_BINS_TOP * p_0 * (1.0 + 1e-12)
        inside_u = (edges_u[:-1] >= u_first) & whole
        short = np.flatnonzero(inside_u & (count_u < need))
        if short.size:
            where = p_0 - float(edges_u[short[-1] + 1])
            return Check(error=float("inf"), where=where, source=source)
    return Check(error=0.0, where=None, source=source)


def _model_type(anisotropy: Anisotropy) -> str:
    """Return the anisotropy's family as the density integral needs it.

    The family is "constant" for an isotropic or constant-anisotropy model,
    which share one branch, and "OM_I", "OM_II", "COM_I" or "COM_II" for an
    Osipkov-Merritt or Cuddeford-Osipkov-Merritt model of the anisotropy's
    kind.
    """
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
    """Return the profile's three checks, keyed by `PROFILE_CHECK_KEYS`.

    The round trip is NaN on a table with a sign column, because a signed
    table has no lookup.

    Parameters
    ----------
    profile : TruncatedProfile
        The model's profile.
    table : DfTable
        The model's DF table.
    form : AsymptoticForm
        The forms beyond the table's two ends.
    anisotropy : Anisotropy
        Anisotropy model of the component.
    settings : mapping
        The truncation's settings; ``ode_rtol`` and the row-count settings
        are read.

    Returns
    -------
    checks : dict of str to Check
        The checks ``poisson_residual``, ``round_trip`` and ``row_density``.
    """
    checks = {
        "poisson_residual": poisson_residual(profile, float(settings["ode_rtol"]))
    }
    if table.sign.size:
        checks["round_trip"] = Check(error=float("nan"), where=None)
    else:
        checks["round_trip"] = round_trip(profile, table, form, anisotropy)
    checks["row_density"] = row_density(profile, settings)
    return checks
