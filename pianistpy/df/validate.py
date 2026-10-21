r"""Return the validated rows of a model's profile after the Poisson solve.

`validate_rows` keeps the rows of the profile that the Poisson solve
reached reliably: from the innermost row, inner fill rows included, to the
last solve row whose potential is finite and at or above the floor
`K_FLOOR`. On an untruncated model, the solve's potential carries a
constant outer offset eps, the solve's integration constant, measured as
the mean difference from the source profile's potential over the rows
beyond :math:`r = 10^{3}`. The offset eps sets R_trust, the radius out to
which the estimated relative error
``|eps| / Phi * max(1, |d ln rho / d ln Phi|)`` of the rows with a
potential Phi below :math:`P_0 / 2` stays at or below a given accuracy,
with :math:`P_0` the central potential and rho the density. The outer rows
of an untruncated model, where Phi is below :math:`P_0 / 2` and the depth
``u = P_0 - Phi`` is well conditioned, are replaced by the source
profile's closed forms on a fixed logarithmic grid from
``R(Phi = P_0 / 2)`` to the smaller of ``R(Phi = K_FLOOR)`` and the
solve's largest radius, so that the table's outer end does not depend on
the solve's offset. A truncated model whose solve
stopped at the floor event gets an edge row at the truncation radius
:math:`R_{\rm t}`, where the potential reaches zero.
"""

__all__ = ["ValidatedRows", "validate_rows"]

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import AccuracyError
from ..profiles import DoublePowerLaw
from ..typing import Array
from .fill import ProfileRows
from .poisson import PoissonSolve
from .truncation import K_FLOOR, ORIGINS


@dataclass(frozen=True)
class ValidatedRows:
    """The rows of a model's profile after validation.

    Parameters
    ----------
    r : array, shape (n,)
        Radii of the rows the profile keeps, increasing, in internal units.
    phi : array, shape (n,)
        Potential Phi on the kept rows.
    rho : array, shape (n,)
        Density on the kept rows.
    mass : array, shape (n,)
        Enclosed mass on the kept rows.
    u_r : array
        Radii of the depth column ``u``. The radii are matched to the
        profile's rows by value, not by index.
    u : array
        Depth ``u = P_0 - Phi`` on the radii `u_r`: the solve's and the
        fill's own depths, and ``P_0 - Phi`` on the closed-form outer rows.
    eps : float
        Outer potential offset of the untruncated solve. NaN on a truncated
        model.
    r_trust : dict
        R_trust keyed by the relative accuracies :math:`10^{-4}`,
        :math:`10^{-5}` and :math:`10^{-6}`. Each value is NaN on a
        truncated model.
    exact_outer : dict
        For the closed-form outer rows: the join radius, the end radius, the
        row count, the rows per decade, and the solve's relative difference
        from the closed forms in potential, density and mass at the last
        kept row inside the join radius. Empty on a truncated model.
    r_t : float
        Edge radius of a spatially finite model, the radius of the
        profile's last row. The edge radius is where the Kepler exterior
        ``Phi = M (1/r - 1/r_t) / (4 pi)`` of the last kept row with Phi at
        or above `K_FLOOR` and a positive density reaches zero, with M that
        row's enclosed mass.
        Infinite where the solve found no edge.
    inner_rows : dict
        The rows inside the solve's requested inner radius, which are the
        inner fill's series rows and which the profile keeps: the requested
        radius, the row count, and the radius and depth ranges of the rows.
    origin : array of int8
        Origin of each row, a code of `truncation.ORIGINS`: the main solve,
        the edge continuation, the floor event, the inner fill, the outer
        fill, the closed-form outer rows, or the edge row.
    """

    r: Array
    phi: Array
    rho: Array
    mass: Array
    u_r: Array
    u: Array
    eps: float
    r_trust: dict[float, float]
    exact_outer: dict[str, Any]
    r_t: float = float("inf")
    inner_rows: dict[str, Any] = field(default_factory=dict)
    origin: Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))


def _pot(source: DoublePowerLaw, r: Any) -> Array:
    """Return the source profile's potential at the radii r, as an array."""
    return np.asarray(source.p(np.atleast_1d(np.asarray(r, dtype=np.float64))))


def _r_where(source: DoublePowerLaw, level: float, lo: float, hi: float) -> float:
    """Return the radius R where the source potential Phi(R) equals level.

    The radius comes from 200 bisection steps in ln R on [lo, hi], with lo
    and hi given as ln R, for a potential that decreases with R.
    """
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(_pot(source, np.exp(mid)).reshape(-1)[0]) > level:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def validate_rows(
    rows: ProfileRows,
    solve: PoissonSolve,
    k_t: float,
    p_center: float,
    source: DoublePowerLaw,
    settings: Mapping[str, Any],
) -> ValidatedRows:
    """Return the validated rows of a model's profile.

    Parameters
    ----------
    rows : ProfileRows
        The profile's rows, with mass and depth column (`profile_rows`).
    solve : PoissonSolve
        The Poisson solve. The call reads the solve's innermost requested
        radius, its rows, its floor event and its edge continuation.
    k_t : float
        Truncation energy, in internal units. Zero for an untruncated model.
    p_center : float
        Central potential of the untruncated model, in internal units.
    source : DoublePowerLaw
        The source profile, whose potential, density and enclosed mass the
        call reads.
    settings : mapping
        The truncation's settings (`Settings.truncation`). The call reads
        ``log10_r_max``, the largest radius of the solve as a base-10
        logarithm, and ``profile_rows_per_decade``.

    Returns
    -------
    rows : ValidatedRows
        The kept rows, the depth column, the offset eps, R_trust, the
        closed-form outer rows' record, the edge radius and the origin of
        each row.

    Raises
    ------
    AccuracyError
        If the profile of an untruncated model has no finite row beyond
        :math:`r = 10^{3}`, so that the offset eps is not finite. Without a
        finite offset eps, the DF does not rebuild the model's density.
    """
    r_out, psi_out = np.array(solve.r), np.array(solve.phi)
    reached = (psi_out >= K_FLOOR) & np.isfinite(psi_out)
    r_max_reliable = r_out[np.where(reached)[0][-1]] if np.any(reached) else r_out[-1]
    r_ode_min = float(solve.r_floor_request or r_out[0])
    p_0 = float(p_center) - float(k_t)

    eps = float("nan")
    if float(k_t) == 0.0:
        r0, p0s = (
            np.array(rows.r, dtype=np.float64),
            np.array(rows.phi, dtype=np.float64),
        )
        m0 = np.isfinite(r0) & np.isfinite(p0s) & (r0 > 1e3)
        if m0.any():
            eps = float(
                np.mean(p0s[m0] - np.asarray(source.p(r0[m0]), dtype=np.float64))
            )
        if not np.isfinite(eps):
            rs = np.asarray(rows.r, dtype=np.float64)
            raise AccuracyError(
                f"the untruncated solve's potential ends at R = "
                f"{float(np.nanmax(rs[np.isfinite(rs)])):.4e} without reaching the "
                f"outer region (R > 1e3): the DF does not rebuild the model's density"
            )

    r_np, p_np = np.array(rows.r), np.array(rows.phi)
    rho_np, m_np = np.array(rows.rho), np.array(rows.mass)
    u_r = np.asarray(rows.r, dtype=np.float64)
    u = np.asarray(rows.u, dtype=np.float64)
    clip = r_np <= r_max_reliable
    inside = r_np < r_ode_min
    u_in = np.asarray(rows.u, dtype=np.float64)[inside]
    inner_rows = dict(
        r_ode_min=r_ode_min,
        rows=int(np.sum(inside)),
        r_range=(float(np.min(r_np[inside])), float(np.max(r_np[inside])))
        if inside.any()
        else (float("nan"), float("nan")),
        u_range=(float(np.min(u_in)), float(np.max(u_in)))
        if inside.any()
        else (float("nan"), float("nan")),
    )
    r_trust = {1e-4: float("nan"), 1e-5: float("nan"), 1e-6: float("nan")}
    exact_outer: dict[str, Any] = {}
    closed = np.zeros(r_np.size, dtype=bool)
    if float(k_t) == 0.0 and np.isfinite(eps):
        ok = clip & (p_np > 0) & (rho_np > 0)
        rr, pr, rhor = r_np[ok], p_np[ok], rho_np[ok]
        slope = np.abs(np.gradient(np.log(rhor), np.log(pr)))
        rel = abs(eps) / pr * np.maximum(1.0, slope)
        outer = pr < 0.5 * float(p_center)

        def trust(g):
            """Return the radius of the row before the first outer row whose
            relative error |eps| / Phi * max(1, |d ln rho / d ln Phi|) exceeds
            g; the radius of the last row where no outer row exceeds g; NaN
            where the first outer row that exceeds g is the first row."""
            bad = np.flatnonzero(outer & (rel > g))
            if not bad.size:
                return float(rr[-1])
            return float(rr[bad[0] - 1]) if bad[0] > 0 else float("nan")

        r_trust = {g: trust(g) for g in (1e-4, 1e-5, 1e-6)}
        p_half = 0.5 * float(p_center)
        lo = np.log(float(np.min(r_np[clip & (r_np >= r_ode_min)])))
        hi = np.log(float(np.max(r_np[clip])))
        r_join = _r_where(source, p_half, lo, hi)
        lo, hi = np.log(r_join), float(settings["log10_r_max"]) * np.log(10.0)
        if float(_pot(source, np.exp(hi)).reshape(-1)[0]) < K_FLOOR:
            r_end = _r_where(source, K_FLOOR, lo, hi)
        else:
            r_end = float(np.exp(hi))
        per_dec = int(settings["profile_rows_per_decade"])
        n_out = int(np.ceil(float(per_dec) * np.log10(r_end / r_join))) + 1
        r_o = np.exp(np.linspace(np.log(r_join), np.log(r_end), n_out))
        p_o = np.asarray(source.p(r_o), dtype=np.float64)
        rho_o = np.asarray(source.rho(r_o), dtype=np.float64)
        m_o = 4.0 * np.pi * np.asarray(source.mass(r_o), dtype=np.float64)
        inner = clip & (r_np < r_join)
        i_last = int(np.flatnonzero(inner)[-1])
        r_last = np.atleast_1d(r_np[i_last])
        pj = float(np.asarray(source.p(r_last)).reshape(-1)[0])
        rj = float(np.asarray(source.rho(r_last)).reshape(-1)[0])
        mj = 4.0 * np.pi * float(np.asarray(source.mass(r_last)).reshape(-1)[0])
        exact_outer = dict(
            exact_outer_R_join=r_join,
            exact_outer_R_end=r_end,
            exact_outer_rows=int(n_out),
            exact_outer_rows_per_decade=per_dec,
            exact_outer_join_rel_psi=float(p_np[i_last] / pj - 1.0),
            exact_outer_join_rel_rho=float(rho_np[i_last] / rj - 1.0),
            exact_outer_join_rel_M=float(m_np[i_last] / mj - 1.0),
            R_trust_applied=False,
        )
        r_np = np.concatenate([r_np[inner], r_o])
        p_np = np.concatenate([p_np[inner], p_o])
        rho_np = np.concatenate([rho_np[inner], rho_o])
        m_np = np.concatenate([m_np[inner], m_o])
        closed = np.concatenate(
            [np.zeros(int(np.sum(inner)), dtype=bool), np.ones(r_o.size, dtype=bool)]
        )
        clip = np.ones(r_np.size, dtype=bool)
        keep_u = u_r < r_join
        u_r = np.concatenate([u_r[keep_u], r_o])
        u = np.concatenate([u[keep_u], p_0 - p_o])
    r_v, p_v, rho_v, m_v = r_np[clip], p_np[clip], rho_np[clip], m_np[clip]
    closed = closed[clip]
    r_t = float("inf")
    if float(k_t) > 0.0 and solve.event_hit:
        # The edge row sits at r_t, where the Kepler exterior of the last row
        # with Phi at or above K_FLOOR and a positive density reaches Phi = 0.
        last = np.flatnonzero((p_v >= K_FLOOR) & (rho_v > 0))[-1]
        r_l, p_l, m_l = float(r_v[last]), float(p_v[last]), float(m_v[last])
        r_t = 1.0 / (1.0 / r_l - 4.0 * np.pi * p_l / m_l)
        keep = np.arange(r_v.size) <= last
        closed = np.append(closed[keep], False)
        r_v = np.append(r_v[keep], r_t)
        p_v = np.append(p_v[keep], 0.0)
        rho_v = np.append(rho_v[keep], 0.0)
        m_v = np.append(m_v[keep], m_l)
        keep_u = u_r <= r_l
        u_r = np.append(u_r[keep_u], r_t)
        u = np.append(u[keep_u], p_0)
    origin = _origin(r_v, p_v, closed, solve, math.isfinite(r_t))
    return ValidatedRows(
        r=r_v,
        phi=p_v,
        rho=rho_v,
        mass=m_v,
        u_r=u_r,
        u=u,
        eps=eps,
        r_trust=r_trust,
        exact_outer=exact_outer,
        r_t=r_t,
        inner_rows=inner_rows,
        origin=origin,
    )


def _origin(
    r: Array, phi: Array, closed: Array, solve: PoissonSolve, edge: bool
) -> Array:
    """Return the origin code (`ORIGINS`) of each row.

    A row at one of the solve's radii has the origin "solve"; from the start
    radius ``R_c`` of the edge continuation outward, "continuation"; and at
    the floor event, where Phi equals K_FLOOR, "event". A closed-form outer
    row has the origin "closed_outer". The edge row, the last row, has the
    origin "edge". Every other row comes from a fill: "inner_fill" below
    the solve's smallest kept radius, and "outer_fill" otherwise.
    """
    origin = np.full(r.size, ORIGINS["outer_fill"], dtype=np.int8)
    from_solve = np.isin(r, np.asarray(solve.r, dtype=np.float64))
    first = float(np.min(r[from_solve])) if from_solve.any() else np.inf
    origin[~from_solve & (r < first)] = ORIGINS["inner_fill"]
    origin[from_solve] = ORIGINS["solve"]
    cont = solve.edge_continuation
    if cont is not None:
        origin[from_solve & (r >= float(cont["R_c"]))] = ORIGINS["continuation"]
    if solve.event_hit:
        origin[from_solve & (phi == K_FLOOR)] = ORIGINS["event"]
    origin[closed] = ORIGINS["closed_outer"]
    if edge:
        origin[-1] = ORIGINS["edge"]
    return origin
