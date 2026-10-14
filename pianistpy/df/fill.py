"""The fills of a model's profile at its two ends.

The Poisson solve's rows are sparse where the potential nears the floor
(the outer end) and where it nears the center (the inner end). Both ends are
filled to the profile's row density: the outer end by log-log interpolation
of the solve's rows in Psi, continued below its lowest rows by their own
slope (and by the edge's closed form below the Taylor switch); the inner end
by the inner series of the source density, lowered by the central deficit
where the model carries one, out to the join with the solve's rows, and
between the join and the fill's threshold by the solve's own interpolant.

The join is the depth to which the series is trusted, moved inward on a
lowered model to where the series rows and the solve's rows agree.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
"""

__all__ = ["FilledProfile", "ProfileRows", "fill_profile", "profile_rows"]

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.special import beta as _sbeta

from ..exceptions import AccuracyError, InputError
from ..typing import Array
from ..validation import GOAL
from .center import (
    CenterAmplitude,
    anisotropy_factor,
    ceiling_depth,
    central_deficit,
    inner_density_series,
    inner_depth_series,
    inner_mass_series,
    join_depth,
)
from .poisson import PoissonSolve, cuddeford_prefactor, profile_density, source_slopes
from .truncation import INNER_DEPTH, K_FLOOR, ROW_BINS_DEPTH_TOP, row_bin_edges


@dataclass(frozen=True)
class FilledProfile:
    """The profile's rows after the two fills.

    Parameters
    ----------
    r, phi, rho : array, shape (n,)
        Radius, potential and density of the rows, increasing in r.
    u_fill_r, u_fill : array or None
        The inner fill's rows and their central coordinate u = P_0 - Phi,
        without a subtraction; None where the inner fill did not run.
    join : float
        The depth of the series / solve join.
    threshold : float
        The inner fill's threshold depth, the larger of the join and ``u_2``.
    reliability : dict
        The depth ``u_2`` (the outermost underdense depth bin) and which of
        the join and ``u_2`` set the threshold (``set_by``).
    """

    r: Array
    phi: Array
    rho: Array
    u_fill_r: Array | None
    u_fill: Array | None
    join: float
    threshold: float
    reliability: dict[str, Any]


def _potential(r, c, g, a, b, p_0):
    return p_0 - inner_depth_series(r, c, g, a, b)


def _low_psi_fill(
    r_save,
    p_save,
    rho_save,
    psi_lo_threshold,
    r_sorted,
    p_sorted,
    rho_sorted,
    min_per_bin,
    dlog,
    valid,
    edge,
):
    """The outer end's fill below the outermost underdense bin."""
    floor = K_FLOOR
    if psi_lo_threshold is None:
        return r_save, p_save, rho_save
    if not np.any(valid):
        return r_save, p_save, rho_save
    r_np, p_np, rho_np = r_save.copy(), p_save.copy(), rho_save.copy()
    if psi_lo_threshold < p_sorted[-1] or psi_lo_threshold > p_sorted[0]:
        return r_save, p_save, rho_save
    r_inner = float(np.interp(psi_lo_threshold, p_sorted[::-1], r_sorted[::-1]))
    r_outer = float(r_sorted[-1])
    n_dec = np.log10(psi_lo_threshold / max(floor, 1e-300))
    n_pts = max(100, int(np.ceil(n_dec / dlog * min_per_bin)))
    step = np.log(psi_lo_threshold / floor) / (n_pts - 1)
    start = psi_lo_threshold * np.exp(-step * 0.5)
    psi_grid = np.geomspace(start, floor, n_pts)
    fit = (r_sorted >= r_inner) & (r_sorted <= r_outer) & (rho_sorted > 0)
    if int(np.sum(fit)) < 4:
        return r_save, p_save, rho_save
    p_fit, r_fit, rho_fit = p_sorted[fit], r_sorted[fit], rho_sorted[fit]
    log_p = np.log(p_fit[::-1])
    log_r = np.log(r_fit[::-1])
    log_rho = np.log(np.maximum(rho_fit[::-1], 1e-300))
    tgt = np.log(psi_grid)
    log_r_fill = np.interp(tgt, log_p, log_r, left=log_r[0], right=log_r[-1])
    if len(log_p) >= 2:
        i_a = min(
            int(np.searchsorted(log_p, np.log(3.0 * INNER_DEPTH), side="right")),
            len(log_p) - 2,
        )
        slope = (log_rho[i_a + 1] - log_rho[i_a]) / (
            log_p[i_a + 1] - log_p[i_a] + 1e-300
        )
        extrap = log_rho[i_a] + slope * (tgt - log_p[i_a])
        log_rho_fill = np.where(
            tgt < log_p[i_a],
            extrap,
            np.interp(tgt, log_p, log_rho, left=log_rho[0], right=log_rho[-1]),
        )
    else:
        log_rho_fill = np.interp(
            tgt, log_p, log_rho, left=log_rho[0], right=log_rho[-1]
        )
    r_fill = np.exp(log_r_fill)
    rho_fill = np.exp(log_rho_fill)
    psi_fill = psi_grid
    if edge is not None:
        # below the Taylor switch the truncated DF is F1 K + F2 K^2 / 2 exactly
        f1, f2, ksw, b_e, mt_e, ra_e = edge
        n_e = 0.5 - b_e
        mf = psi_fill < ksw
        if mf.any():
            pf, rf = psi_fill[mf], r_fill[mf]
            pcl = (2.0**n_e) * (
                f1 * _sbeta(2.0, n_e + 1.0) * pf ** (n_e + 2.0)
                + 0.5 * f2 * _sbeta(3.0, n_e + 1.0) * pf ** (n_e + 3.0)
            )
            gf = rf ** (2.0 * b_e)
            if mt_e == "OM_I" or mt_e == "COM_I":
                gf = gf * (1.0 + (rf / ra_e) ** 2) ** (
                    1.0 - b_e if mt_e == "COM_I" else 1.0
                )
            if mt_e == "OM_II" or mt_e == "COM_II":
                gf = gf * (1.0 - (rf / ra_e) ** 2) ** (
                    1.0 - b_e if mt_e == "COM_II" else 1.0
                )
            rho_cl = cuddeford_prefactor(b_e) * pcl / gf
            rho_fill = np.array(rho_fill, dtype=np.float64)
            rho_fill[mf] = rho_cl
    keep = p_np > psi_lo_threshold
    r_save = np.concatenate([r_np[keep], r_fill[1:]])
    p_save = np.concatenate([p_np[keep], psi_fill[1:]])
    rho_save = np.concatenate([rho_np[keep], rho_fill[1:]])
    order = np.argsort(r_save)
    return r_save[order], p_save[order], rho_save[order]


def _high_psi_fill(
    r_save,
    p_save,
    rho_save,
    u_hi_threshold,
    u_threshold,
    psi_hi_threshold,
    r_sorted_hi,
    p_sorted_hi,
    rho_sorted_hi,
    min_per_bin,
    dlog,
    center,
    profile,
    p_0,
    u_join,
    bc,
    factor,
    solve,
    source,
):
    """The inner end's fill above the threshold depth; returns the rows and
    the fill's (r, u), or None for the latter where it did not run."""
    u_join_eff = float(u_join)
    if u_hi_threshold is None and u_threshold <= u_join_eff:
        return r_save, p_save, rho_save, None
    c_t = center.c_trunc
    _, a_p, b_p, g_p = profile
    delta = float(center.delta)
    r_np2, p_np2, rho_np2 = r_save.copy(), p_save.copy(), rho_save.copy()
    keep2 = p_np2 <= psi_hi_threshold
    u_ode_min = max(float(p_0 - np.max(p_sorted_hi)), 0.0)
    u_split = max(u_join_eff, u_ode_min)
    case_a = u_threshold <= u_split
    u1_lo, u1_hi = K_FLOOR, u_split
    n1_dec = np.log10(u1_hi / max(u1_lo, 1e-300))
    n1 = int(np.ceil(n1_dec / dlog * min_per_bin * 1.01))
    u1_grid = np.geomspace(u1_lo, u1_hi, n1)
    a0_inv = c_t / ((3.0 - g_p) * (2.0 - g_p))
    r_sr1 = (u1_grid / max(a0_inv, 1e-300)) ** (1.0 / (2.0 - g_p))
    for _ in range(2):
        p_cur = _potential(r_sr1, c_t, g_p, a_p, b_p, p_0)
        dpdr = -(
            inner_mass_series(r_sr1, c_t, g_p, a_p, b_p)
            / np.maximum(r_sr1, 1e-300) ** 2
        )
        step = (p_cur - (p_0 - u1_grid)) / np.where(dpdr != 0.0, dpdr, np.nan)
        r_sr1 = np.where(
            np.isfinite(step) & (np.abs(step) < 0.5 * r_sr1), r_sr1 - step, r_sr1
        )
    if delta > 0.0:
        # a lowered center shifts each row's depth by du: place the rows where
        # their final depth is the grid's, so the grid's bins stay full up to
        # the threshold
        lc = center.lowered

        def du_at(r):
            if lc is not None:
                return lc["at"](r)[1]
            return (
                delta
                * r ** (2.0 - 2.0 * float(bc))
                / ((3.0 - 2.0 * float(bc)) * (2.0 - 2.0 * float(bc)))
            )

        for _ in range(50):
            p_cur = _potential(r_sr1, c_t, g_p, a_p, b_p, p_0) + du_at(r_sr1)
            dpdr = -(
                inner_mass_series(r_sr1, c_t, g_p, a_p, b_p)
                / np.maximum(r_sr1, 1e-300) ** 2
            )
            step = (p_cur - (p_0 - u1_grid)) / np.where(dpdr != 0.0, dpdr, np.nan)
            move = np.isfinite(step) & (np.abs(step) < 0.5 * r_sr1)
            r_sr1 = np.where(move, r_sr1 - step, r_sr1)
            if not np.any(move & (np.abs(step) > 1e-15 * r_sr1)):
                break
    rho_sr1 = inner_density_series(r_sr1, c_t, g_p, a_p, b_p)
    psi_sr1 = _potential(r_sr1, c_t, g_p, a_p, b_p, p_0)
    du_sr1 = None
    if delta > 0.0:
        omf = anisotropy_factor(r_sr1, bc, factor[0], factor[1])
        dr_sr1 = delta * r_sr1 ** (-2.0 * float(bc)) / omf
        du_sr1 = (
            delta
            * r_sr1 ** (2.0 - 2.0 * float(bc))
            / ((3.0 - 2.0 * float(bc)) * (2.0 - 2.0 * float(bc)))
        )
        lc = center.lowered
        if lc is not None and not (lc["P0"] == float(p_0) and lc["bc"] == float(bc)):
            raise InputError(
                f"the lowered center held is for P0 = {lc['P0']!r}, bc = {lc['bc']!r}; "
                f"these rows are P0 = {float(p_0)!r}, bc = {float(bc)!r}"
            )
        if lc is not None:
            d56, du56, _ = lc["at"](r_sr1)
            dr_sr1 = (
                d56
                * r_sr1 ** (-2.0 * float(bc))
                / anisotropy_factor(r_sr1, bc, factor[0], factor[1])
            )
            du_sr1 = du56
        rho_sr1 = rho_sr1 - dr_sr1
        psi_sr1 = psi_sr1 + du_sr1
    if case_a:
        r_fill2, psi_fill2, rho_fill2 = r_sr1, psi_sr1, rho_sr1
    else:
        u2_lo, u2_hi = u_split, u_threshold
        n2_dec = np.log10(u2_hi / max(u2_lo, 1e-300))
        n2 = int(np.ceil(n2_dec / dlog * min_per_bin * 1.05))
        u2_grid = np.geomspace(u2_lo, u2_hi, n2)
        psi_sr2_tgt = p_0 - u2_grid
        lu_src = np.log(np.maximum(p_0 - p_sorted_hi, 1e-300))[::-1]
        lu_tgt = np.log(u2_grid)
        r_sr2 = np.exp(
            np.interp(
                lu_tgt,
                lu_src,
                np.log(r_sorted_hi)[::-1],
                left=np.log(r_sorted_hi[-1]),
                right=np.log(r_sorted_hi[0]),
            )
        )
        log_rho_sr2 = np.interp(
            lu_tgt,
            lu_src,
            np.log(np.maximum(rho_sorted_hi, 1e-300))[::-1],
            left=np.log(max(rho_sorted_hi[-1], 1e-300)),
            right=np.log(max(rho_sorted_hi[0], 1e-300)),
        )
        psi_sr2, rho_sr2, u_sr2_rows = psi_sr2_tgt, np.exp(log_rho_sr2), u2_grid
        lo_ev, hi_ev = solve.evaluate_range
        if (
            solve.evaluate is not None
            and abs(float(source["p_0"]) - float(p_0)) == 0.0
            and float(np.min(r_sr2)) >= lo_ev
            and float(np.max(r_sr2)) <= hi_ev
        ):
            uw2 = solve.evaluate(r_sr2)
            p2n = float(p_0) - uw2[:, 2]
            order = "cubic"
            r2n = np.asarray(
                profile_density(
                    jnp.asarray(r_sr2),
                    jnp.asarray(p2n),
                    source["q"],
                    source["p_rhs"],
                    source["ra"],
                    source["bc"],
                    source["model_type"],
                    p_0=float(p_0),
                    m=source_slopes(
                        source["q"], source["p_rhs"], float(p_0), source["p_s"], order
                    ),
                    order=order,
                ),
                dtype=np.float64,
            )
            psi_sr2, rho_sr2, u_sr2_rows = p2n, r2n, uw2[:, 2]
        r_fill2 = np.concatenate([r_sr1, r_sr2[1:]])
        psi_fill2 = np.concatenate([psi_sr1, psi_sr2[1:]])
        rho_fill2 = np.concatenate([rho_sr1, rho_sr2[1:]])
    u_sr1 = inner_depth_series(r_sr1, c_t, g_p, a_p, b_p)
    if delta > 0.0:
        u_sr1 = u_sr1 - du_sr1
    u_fill2 = u_sr1 if case_a else np.concatenate([u_sr1, u_sr2_rows[1:]])
    r_save = np.concatenate([r_np2[keep2], r_fill2[:-1]])
    p_save = np.concatenate([p_np2[keep2], psi_fill2[:-1]])
    rho_save = np.concatenate([rho_np2[keep2], rho_fill2[:-1]])
    order2 = np.argsort(r_save)
    fill = (
        np.asarray(r_fill2, dtype=np.float64).copy(),
        np.asarray(u_fill2, dtype=np.float64).copy(),
    )
    return r_save[order2], p_save[order2], rho_save[order2], fill


def fill_profile(
    solve: PoissonSolve,
    rho: Any,
    p_0: float,
    center: CenterAmplitude,
    profile: tuple[float, float, float, float],
    bc: float,
    factor: tuple[str | None, float],
    edge: tuple[float, float, float, float, str, float] | None,
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> FilledProfile:
    """Return the profile's rows after the outer and the inner fill.

    Parameters
    ----------
    solve : PoissonSolve
        The Poisson solve; its rows and its interpolant.
    rho : array
        The density on the solve's rows (`profile_density`).
    p_0 : float
        The model's central potential.
    center : CenterAmplitude
        The inner amplitude and the central deficit.
    profile : tuple
        (C, alpha, beta, gamma) of the source density.
    bc, factor
        The anisotropy's beta_c, its factor's kind and R_a.
    edge : tuple or None
        The Taylor edge (F_1, F_2, K_switch, beta_c, model type, R_a); None
        on an untruncated model.
    source : mapping
        The solve's source table: ``q``, ``p_rhs``, ``p_0``, ``p_s``, ``ra``,
        ``bc``, ``model_type``.
    settings : mapping
        The truncation's settings (`Settings.truncation`).

    Raises
    ------
    AccuracyError
        Where the series rows and the solve's rows of a lowered model agree
        in no half-decade of depth.
    """
    c_t = center.c_trunc
    _, a_p, b_p, g_p = profile
    r_save = np.array(solve.r).copy()
    p_save = np.array(solve.phi).copy()
    rho_save = np.array(rho).copy()
    floor = K_FLOOR
    dlog = float(settings["profile_bin_decades"])
    saved = (p_save >= floor) & (rho_save > 0)
    psi_saved = p_save[saved]
    psi_saved = psi_saved[np.isfinite(psi_saved)]
    u_vals = p_0 - psi_saved
    u_vals = u_vals[(u_vals > 0) & (u_vals <= ROW_BINS_DEPTH_TOP)]
    # the row-count bins, the check's (profile_checks.row_density) by construction
    row_bins = row_bin_edges(
        p_0, settings, float(np.max(u_vals)) if len(u_vals) else 0.0
    )
    min_per_bin, psi_top, psi_edges = (
        row_bins.min_per_bin,
        row_bins.psi_top,
        row_bins.psi_edges,
    )
    counts_lo, _ = np.histogram(psi_saved, bins=psi_edges)
    counts_hi = u_edges = None
    if row_bins.u_edges is not None:
        u_edges = row_bins.u_edges
        counts_hi, _ = np.histogram(u_vals, bins=u_edges)
    valid = np.isfinite(p_save) & (p_save > 0) & (p_save <= psi_top)
    sort = np.argsort(r_save[valid])
    r_sorted, p_sorted, rho_sorted = (
        r_save[valid][sort],
        p_save[valid][sort],
        rho_save[valid][sort],
    )
    psi_lo_threshold = None
    for i in range(len(counts_lo) - 1, -1, -1):
        if int(counts_lo[i]) < min_per_bin:
            psi_lo_threshold = float(psi_edges[i + 1])
            break
    u_hi_threshold = None
    if counts_hi is not None and u_edges is not None:
        for i in range(len(counts_hi) - 1, -1, -1):
            if int(counts_hi[i]) < min_per_bin:
                u_hi_threshold = float(u_edges[i + 1])
                break

    u_join = float(join_depth(c_t, g_p, a_p, b_p, GOAL / 10.0)[0])
    lc = center.lowered if float(center.delta) > 0.0 else None
    if lc is not None:
        tol = 0.1 * GOAL
        ro, uo, rho_o = r_save, float(p_0) - p_save, rho_save
        sel = (
            (uo > 0.0)
            & (uo <= u_join)
            & (ro > 0.0)
            & (ro <= float(lc["R_hi"]))
            & np.isfinite(rho_o)
            & (rho_o > 0.0)
        )
        if int(sel.sum()) >= 10:
            ords = np.argsort(uo[sel])[::-1]
            rs_, us_, rhos_ = ro[sel][ords], uo[sel][ords], rho_o[sel][ords]
            ds, dus, _ = lc["at"](rs_)
            rho_a = inner_density_series(rs_, c_t, g_p, a_p, b_p) - ds * rs_ ** (
                -2.0 * float(bc)
            ) / anisotropy_factor(rs_, bc, factor[0], factor[1])
            u_a = (float(p_0) - _potential(rs_, c_t, g_p, a_p, b_p, p_0)) - dus
            ok = (np.abs(rho_a / rhos_ - 1.0) <= tol) & (np.abs(u_a / us_ - 1.0) <= tol)
            bin_ = np.floor(2.0 * np.log10(us_)).astype(int)
            bins = np.unique(bin_)[::-1]
            bok = np.array([bool(np.all(ok[bin_ == b])) for b in bins])
            if not bok.any():
                raise AccuracyError(
                    f"the lowered center's rows and the solve's rows agree in no "
                    f"half-decade of depth to {tol:.0e} over u in "
                    f"[{us_[-1]:.2e}, {us_[0]:.2e}]"
                )
            best, cur = (0, 0), None
            for i, okb in enumerate(bok):
                cur = (i if cur is None else cur) if okb else None
                if okb and cur is not None and (i - cur + 1) > (best[1] - best[0] + 1):
                    best = (cur, i)
            u_seam = float(np.max(us_[bin_ == bins[best[0]]]))
            if u_seam < u_join:
                u_join = u_seam
    u2 = u_hi_threshold if u_hi_threshold is not None else 0.0
    u_threshold = max(u_join, u2)
    psi_hi_threshold = p_0 - u_threshold
    ode = np.isfinite(p_save) & (p_save > 0) & (rho_save > 0)
    o_sort = np.argsort(p_save[ode])
    p_o, r_o, rho_o2 = p_save[ode][o_sort], r_save[ode][o_sort], rho_save[ode][o_sort]
    i_lo = max(0, int(np.searchsorted(p_o, psi_hi_threshold, side="left")) - 1)
    i_hi = min(len(p_o) - 1, int(np.searchsorted(p_o, p_0 - u_join, side="right")))
    reliability = dict(
        u_2=float(u2), set_by=("join", "u_2")[int(np.argmax([u_join, u2]))]
    )

    r_save, p_save, rho_save = _low_psi_fill(
        r_save,
        p_save,
        rho_save,
        psi_lo_threshold,
        r_sorted,
        p_sorted,
        rho_sorted,
        min_per_bin,
        dlog,
        valid,
        edge,
    )
    r_save, p_save, rho_save, fill = _high_psi_fill(
        r_save,
        p_save,
        rho_save,
        u_hi_threshold,
        u_threshold,
        psi_hi_threshold,
        r_o[i_lo : i_hi + 1],
        p_o[i_lo : i_hi + 1],
        rho_o2[i_lo : i_hi + 1],
        min_per_bin,
        dlog,
        center,
        profile,
        p_0,
        u_join,
        bc,
        factor,
        solve,
        source,
    )
    return FilledProfile(
        r=r_save,
        phi=p_save,
        rho=rho_save,
        u_fill_r=None if fill is None else fill[0],
        u_fill=None if fill is None else fill[1],
        join=float(u_join),
        threshold=float(u_threshold),
        reliability=reliability,
    )


@dataclass(frozen=True)
class ProfileRows:
    """The profile's rows with their mass and central coordinate.

    Parameters
    ----------
    r, phi, rho : array, shape (n,)
        The filled rows the profile keeps: finite, rho > 0 and Phi at or
        above the floor.
    mass : array, shape (n,)
        The enclosed mass (4 pi INT rho r^2 dr): the inner series' (less the
        central deficit's) inside the ceiling, cumulative Simpson outside it
        on the rows' density read as a cubic spline in the logarithms, its
        step halved until the mass at every row moves by less than a tenth
        of the goal.
    u : array, shape (n,)
        The central coordinate u = P_0 - Phi, with no subtraction on the
        solve's node rows and the inner fill's rows.
    central_deficit : tuple of float or None
        (D0, D1) of `central_deficit` on a truncated model; None otherwise or
        where it is not available.
    ceiling : dict
        The mass's zone boundary: the depth where the mass series' next term
        reaches a tenth of the goal (``u_rule``, its fraction of P_0 and the
        bound), the depth used (``u_ceil``, ``r_ceil``) and which of the three
        set it (``set_by``): ``"rule"``, or a structural bound, where series
        rows end (``"fill_threshold"``) or the lowered center's reach
        (``"center_reach"``).
    zone : dict
        The outer zone's convergence: the halvings of the step (``levels``),
        its nodes and the largest relative move of the mass at a row at the
        last halving (``change``).
    """

    r: Array
    phi: Array
    rho: Array
    mass: Array
    u: Array
    central_deficit: tuple[float, float] | None
    ceiling: dict[str, Any] = field(default_factory=dict)
    zone: dict[str, Any] = field(default_factory=dict)


def _mass(r, rho, center, profile, p_0, r_ceil, bc):
    c_t = center.c_trunc
    _, a_p, b_p, g_p = profile
    delta = float(center.delta)
    lc = center.lowered
    r_np = np.asarray(r, dtype=np.float64)
    rho_np = np.asarray(rho, dtype=np.float64)
    mass = np.zeros(len(r_np), dtype=np.float64)
    zone1 = r_np <= r_ceil
    if (
        delta > 0.0
        and lc is not None
        and not (lc["P0"] == float(p_0) and lc["bc"] == float(bc))
    ):
        raise InputError(
            f"the lowered center held is for P0 = {lc['P0']!r}, bc = {lc['bc']!r}; "
            f"this mass column is P0 = {float(p_0)!r}, bc = {float(bc)!r}"
        )
    if np.any(zone1):
        r1 = r_np[zone1]
        m1 = 4.0 * np.pi * inner_mass_series(r1, c_t, g_p, a_p, b_p)
        if delta > 0.0:
            m1 = m1 - 4.0 * np.pi * (
                lc["at"](r1)[2]
                if lc is not None
                else delta * r1 ** (3.0 - 2.0 * float(bc)) / (3.0 - 2.0 * float(bc))
            )
        mass[zone1] = m1
    m_b = float(
        4.0 * np.pi * inner_mass_series(np.array([r_ceil]), c_t, g_p, a_p, b_p)[0]
    )
    if delta > 0.0:
        m_b -= (
            4.0
            * np.pi
            * (
                float(lc["at"](np.array([float(r_ceil)]))[2][0])
                if lc is not None
                else delta
                * float(r_ceil) ** (3.0 - 2.0 * float(bc))
                / (3.0 - 2.0 * float(bc))
            )
        )
    zone2 = ~zone1
    if np.any(zone2):
        rho_b = float(
            np.interp(
                r_ceil, r_np, rho_np, left=float(rho_np[0]), right=float(rho_np[-1])
            )
        )
        r_aug = np.concatenate([[r_ceil], r_np[zone2]])
        rho_aug = np.concatenate([[rho_b], rho_np[zone2]])
        if r_aug.size >= 3:
            zone_m, zone = _zone_mass(r_aug, rho_aug, m_b, 0.1 * GOAL)
            mass[zone2] = zone_m
        else:
            integrand = r_aug**2 * rho_aug
            dm = 4.0 * np.pi * 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_aug)
            mass[zone2] = m_b + np.cumsum(dm)
            zone = {"levels": 0, "nodes": int(r_aug.size), "change": float("nan")}
    else:
        zone = {"levels": 0, "nodes": 0, "change": float("nan")}
    return mass, zone


_ZONE_LEVELS = 6  # at most 2^6 sub-steps between two rows


def _zone_mass(r_aug, rho_aug, m_b, tol):
    """The mass past the ceiling at the rows: cumulative Simpson on the rows'
    density as a cubic spline in ln r of ln rho, each step between rows cut
    into 2^level sub-steps (geometric in r), the level raised until the mass
    at every row moves by less than ``tol`` relative; at most `_ZONE_LEVELS`
    halvings, the move then reached is what the result states. Rows at one
    radius in ln r (a repeated row) share the first one's mass."""
    full_log_r = np.log(r_aug)
    distinct = np.concatenate([[True], np.diff(full_log_r) > 0.0])
    at = np.cumsum(distinct) - 1  # each row's distinct node
    r_aug, rho_aug, log_r = r_aug[distinct], rho_aug[distinct], full_log_r[distinct]
    spline = CubicSpline(log_r, np.log(rho_aug))
    previous, change, level = None, float("nan"), 0
    for level in range(_ZONE_LEVELS + 1):
        k = 2**level
        t = np.linspace(0.0, 1.0, k + 1)[:-1]
        x = np.concatenate(
            [
                (log_r[:-1, None] + np.diff(log_r)[:, None] * t[None, :]).ravel(),
                log_r[-1:],
            ]
        )
        rr, rho = (r_aug, rho_aug) if level == 0 else (np.exp(x), np.exp(spline(x)))
        cum = cumulative_simpson(rr**2 * rho, x=rr, initial=0.0)
        m = m_b + 4.0 * np.pi * cum[::k][1:]
        if previous is not None:
            change = float(np.max(np.abs(m / previous - 1.0)))
            if change < tol:
                break
        previous = m
    m = np.concatenate([[m_b], m])[at][1:]
    return m, {"levels": int(level), "nodes": int(rr.size), "change": change}


def profile_rows(
    filled: FilledProfile,
    solve: PoissonSolve,
    p_0: float,
    center: CenterAmplitude,
    profile: tuple[float, float, float, float],
    bc: float,
    k_t: float,
    pot_center: float,
    q: Any,
    df: Any,
    f_at_k_t: float,
) -> ProfileRows:
    """Return the profile's rows, their enclosed mass and their central
    coordinate, and the central deficit of a truncated model.

    Parameters
    ----------
    filled : FilledProfile
        The rows after the two fills.
    solve : PoissonSolve
        The Poisson solve (its rows place the mass's inner zone; its u at
        its nodes is the column's there).
    k_t, pot_center, q, df, f_at_k_t
        The truncation and the untruncated DF, for the central deficit.
    """
    phi_raw = np.array(solve.phi)
    r_raw = np.array(solve.r)
    _, a_p, b_p, g_p = profile
    u_rule, _, bound = ceiling_depth(center.c_trunc, g_p, a_p, b_p, 0.1 * GOAL)
    u_ceil, set_by = u_rule, "rule"
    if filled.threshold is not None and float(filled.threshold) < u_ceil:
        # the series is the rows' only where the inner fill laid series rows
        u_ceil, set_by = float(filled.threshold), "fill_threshold"
    r_ceil = float(
        np.exp(
            np.interp(
                np.log(max(p_0 - u_ceil, 1e-300)),
                np.log(np.maximum(phi_raw[::-1], 1e-300)),
                np.log(np.maximum(r_raw[::-1], 1e-300)),
            )
        )
    )
    lc = center.lowered if float(center.delta) > 0.0 else None
    if lc is not None and r_ceil > float(lc["R_hi"]):
        r_ceil, set_by = float(lc["R_hi"]), "center_reach"
    ceiling = {
        "u_rule": u_rule,
        "u_rule_fraction": u_rule / float(p_0),
        "bound": bound,
        "u_ceil": u_ceil,
        "r_ceil": r_ceil,
        "set_by": set_by,
    }
    r, phi, rho = filled.r, filled.phi, filled.rho
    keep = (
        np.isfinite(r)
        & np.isfinite(phi)
        & np.isfinite(rho)
        & (rho > 0.0)
        & (phi >= K_FLOOR)
    )
    if not np.all(keep):
        r, phi, rho = r[keep], phi[keep], rho[keep]
    mass, zone = _mass(r, rho, center, profile, p_0, r_ceil, bc)
    rm = np.asarray(r, dtype=np.float64)
    u = float(p_0) - np.asarray(phi, dtype=np.float64)
    for src_r, src_u in ((solve.u_r, solve.u), (filled.u_fill_r, filled.u_fill)):
        if src_r is None:
            continue
        rg = np.asarray(src_r, dtype=np.float64)
        ug = np.asarray(src_u, dtype=np.float64)
        ok = np.isfinite(ug) & (ug > 0)
        rg, ug = rg[ok], ug[ok]
        o = np.argsort(rg)
        rg, ug = rg[o], ug[o]
        i = np.clip(np.searchsorted(rg, rm), 0, rg.size - 1)
        hit = (rg[i] == rm) & (
            np.abs(ug[i] - u) <= 4.0 * np.finfo(np.float64).eps * float(p_0)
        )
        u[hit] = ug[i][hit]
    deficit = None
    if float(k_t) > 0.0:
        deficit = central_deficit(q, df, k_t, pot_center, bc, f_at_k_t=f_at_k_t)
    return ProfileRows(
        r=r,
        phi=phi,
        rho=rho,
        mass=mass,
        u=u,
        central_deficit=deficit,
        ceiling=ceiling,
        zone=zone,
    )
