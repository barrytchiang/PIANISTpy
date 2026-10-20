"""The validation of a model's profile rows.

The rows the profile keeps are those the Poisson solve reached reliably:
from the innermost radius the solve was asked for to the last row where the
potential is at or above the floor. On an untruncated model the solve's
potential carries a constant outer offset eps (its integration constant),
measured against the source profile's own potential beyond
:math:`r = 10^{3}`; it sets the radius out to which a relative accuracy holds
(R_trust). The outer rows of an untruncated model, where the depth
``u = P_0 - Phi`` is well conditioned (``Phi < P_0 / 2``), are then the source
profile's closed forms on a fixed logarithmic grid, from
``R(Phi = P_0 / 2)`` to ``R(Phi = floor)``, so that the
table's end does not move with the solve's offset.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
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
    """The profile's rows after validation.

    Parameters
    ----------
    r, phi, rho, mass : array, shape (n,)
        The rows the profile keeps, increasing in r.
    u_r, u : array
        The central coordinate ``u = P_0 - Phi`` on the solve's rows and the
        fill's, and on the closed-form outer rows; matched to the profile's
        rows by radius.
    eps : float
        The untruncated solve's outer potential offset; NaN on a truncated
        model.
    r_trust : dict
        R_trust at the relative accuracies :math:`10^{-4}`, :math:`10^{-5}`
        and :math:`10^{-6}` (NaN on a truncated model).
    inner_rows : dict
        The rows inside the solve's requested inner radius (the inner fill's
        series rows), which the profile keeps: their count and their radius
        and depth ranges.
    r_t : float
        The edge of a spatially finite model, where the profile's last row
        is: the radius where the Kepler exterior of the last row of the solve,
        ``Phi = M (1/r - 1/r_t) / (4 pi)``, reaches zero; infinite where the solve
        found no edge.
    exact_outer : dict
        The closed-form outer rows' join, end, row count and the solve's
        relative difference from the closed forms at its last inner row;
        empty on a truncated model.
    origin : array of int8
        Each row's origin, a code of `truncation.ORIGINS`: the main solve, the edge
        continuation, the floor event, the inner and the outer fills, the
        closed-form outer rows, the edge row.
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
    return np.asarray(source.p(np.atleast_1d(np.asarray(r, dtype=np.float64))))


def _r_where(source: DoublePowerLaw, level: float, lo: float, hi: float) -> float:
    """ln R bisection for Phi(R) = level, Phi decreasing in R."""
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
    """Return the profile's validated rows.

    Parameters
    ----------
    rows : ProfileRows
        The rows, mass and u column of the profile (`profile_rows`).
    solve : PoissonSolve
        The Poisson solve: its innermost requested radius and its rows'
        reach.
    k_t, p_center : float
        The truncation energy and the untruncated central potential.
    source : DoublePowerLaw
        The source profile: its potential, density and enclosed mass.
    settings : mapping
        The truncation's settings (`Settings.truncation`): ``log10_r_max``
        (the run's) and ``profile_rows_per_decade``.

    Raises
    ------
    AccuracyError
        Where an untruncated model's solve ends without reaching
        :math:`r = 10^{3}`: the
        DF does not rebuild the model's density.
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
        # the edge row: the last row's Kepler exterior reaches Phi = 0 at r_t
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
    """Each row's origin (`ORIGINS`): a row at one of the solve's radii is the
    solve's, the edge continuation's from its start, the event's at the floor;
    a closed-form outer row is marked as built; the edge row is the last; the
    other rows are the fills', inner below the solve's first kept row and
    outer beyond it."""
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
