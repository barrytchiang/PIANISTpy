"""The table of rows of the line-of-sight velocity distribution, computed in
a pool and checked for its reading between rows."""

__all__ = [
    "START_METHOD",
    "init_worker",
    "losvd_rows",
    "new_pool",
    "row_worker",
    "row_worker_or_none",
]

import contextlib
import multiprocessing as mp
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..exceptions import ResultError
from ..runtime import get_workers, log, worker_environment
from ..typing import Array, ArrayLike
from ._ellipse import T_SET_BY, EllipseRoute
from ._ellipse_rules import BREAK_PAST_END, ROW_TOL_M2, ROW_TOL_PW
from ._kept import WORKER
from ._panel_table import PanelTable

# The pool's start method: spawn, which every platform has and which JAX
# allows after it initializes (fork does neither). Each worker imports the
# package and takes a pickled copy of the route.
START_METHOD = "spawn"


def new_pool(n_workers: int, route: EllipseRoute) -> Any:
    """A spawn pool of `n_workers` processes, each holding a copy of the
    route (`init_worker`) and running its BLAS and OpenMP on one thread,
    on the CPU: the pool is the parallelism and a worker is one process.
    The workers' variables are the runtime service's
    (`worker_environment`), held for the spawn."""
    with worker_environment():
        return mp.get_context(START_METHOD).Pool(
            processes=n_workers, initializer=init_worker, initargs=(route,)
        )


def init_worker(route: EllipseRoute) -> None:
    """The pool's initializer: the route, once a process."""
    WORKER["route"] = route


def row_worker(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """One row: D on the row's V panels, INT D dV, INT V^2 D dV and the part
    beyond the escape speed on the same nodes, its breaks and its t rule's
    record. A fourth element of `args`, (vrad_kept, vk_kept), sets the kept
    ranges on the worker's route."""
    i_r2d, r2d = args[:2]
    route = WORKER["route"]
    if len(args) > 2:
        route.vrad_kept, route.vk_kept = args[2]
    t0 = time.perf_counter()
    rule, cusp = route.rule(r2d)
    vp, wp, br, vesc, vk, vend, vr, code = route.panels(r2d)
    trep: dict[str, Any] = {}
    route.angle_counts[:] = 0
    dp = route.row(r2d, vp, rule, report=trep, w=wp)
    trep["angle_counts"] = route.angle_counts.copy()
    norm = 2.0 * float(np.dot(wp, dp))
    m2 = 2.0 * float(np.dot(wp, vp * vp * dp))
    beyond = 2.0 * float(np.dot(wp[vp > vesc], dp[vp > vesc]))
    ve = float(route.v_edge_many(np.array([r2d]))[0])
    vc = float(route.v_c_many(np.array([r2d]))[0])
    return (
        i_r2d,
        vp,
        wp,
        dp,
        norm,
        m2,
        beyond,
        cusp,
        (vesc, vk, vend, vr, len(br) - 1, code, ve, vc),
        time.perf_counter() - t0,
        trep,
    )


def row_worker_or_none(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """A row the check between rows adds: `row_worker`, but a radius whose own scan
    puts a decided break on or past its V_end returns (index, None)."""
    try:
        return row_worker(args)
    except ResultError as e:
        if BREAK_PAST_END not in str(e):
            raise
        return (args[0], None)


def losvd_rows(
    route: EllipseRoute,
    r_2d: ArrayLike,
    sigma: ArrayLike | None,
    sigma_los2: ArrayLike | None,
    r_breaks: ArrayLike = (),
    vrad_kept: Array | None = None,
    vk_kept: Array | None = None,
    sigma_at: Callable[[Array], tuple[Array, Array]] | None = None,
    certify: bool = True,
    r_edge: float = float("inf"),
    pool: Any = None,
    df_scale: float = 1.0,
    read_range: tuple[float, float] = (0.0, float("inf")),
) -> tuple[PanelTable, Array, Array, Array, dict[str, Any]]:
    r"""Return the table of rows at radii `r_2d`, with its self-check.

    Each row is computed in a pool (`row_worker`) and normalized to unit
    integral over V. Where `certify`, the table is then read between its rows
    as an aperture reads it, the stencil at row_order_test rows against two
    fewer at every interval's midpoint and in every end gap, and a row is
    added at each that fails ROW_TOL_M2 or ROW_TOL_PW, until none fails, a
    round no longer halves the worst, or no failing interval can take a row;
    what is left over is recorded, and the table refuses a reading there.

    Parameters
    ----------
    route : EllipseRoute
        The route of the main process, which reads the table between rows;
        the pool's workers take a copy (`init_worker`).
    r_2d : array-like
        The base rows.
    sigma, sigma_los2 : array-like or None
        The surface density and the squared dispersion at the base rows, the
        references of the self-check.
    r_breaks : array-like
        Where the layout changes.
    vrad_kept, vk_kept : array or None
        The kept ranges of V_rad and V_k.
    sigma_at : callable, optional
        (Sigma, :math:`\sigma_{\rm los}^2`) at added radii.
    certify : bool
        Whether to check the reading between rows.
    r_edge : float
        The parent's end, a true end of the last stretch where finite.
    pool : optional
        A pool already started with `init_worker`; by default one of the
        package's worker count (`pianistpy.set_workers`).
    df_scale : float
        The constant factor the rows' DF carries (`EllipseDF.scale`): a row
        integrates to the surface density times it.
    read_range : tuple of float
        The radii the table is read over: an end gap whose end lies outside
        it is not read (the rows of a table cut there end before it).

    Returns
    -------
    table, norm_rel, m2_rel, beyond_rel, info
        The table; each row's norm against Sigma and second moment against
        :math:`\sigma_{\rm los}^2`, less one; its share beyond the escape
        speed; and the record of the rows and of the check.
    """
    rr = route.rules
    lat = route.lattice
    r_2d_arr = np.asarray(r_2d, dtype=np.float64).copy()
    n_r2d = len(r_2d_arr)
    rows: Any = [None] * n_r2d
    norm: Any = np.full(n_r2d, np.nan)
    m2: Any = np.full(n_r2d, np.nan)
    beyond: Any = np.zeros(n_r2d)
    cusp: Any = np.zeros(n_r2d, dtype=bool)
    nvp: Any = np.zeros(n_r2d, dtype=int)
    secs: Any = np.zeros(n_r2d)
    t_panels: Any = np.zeros(n_r2d, dtype=np.int64)
    t_change: Any = np.full(n_r2d, np.nan)
    t_conv: Any = np.zeros(n_r2d, dtype=bool)
    ang = np.zeros(len(rr.angle_counts) + 1, dtype=np.int64)
    t_change_pw: Any = np.full(n_r2d, np.nan)
    t_set_by: Any = np.zeros(n_r2d, dtype=np.int8)
    kept = (vrad_kept, vk_kept)
    args_list = [(i, float(r_2d_arr[i]), kept) for i in range(n_r2d)]
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=np.float64).copy()
    sl2_arr = (
        None if sigma_los2 is None else np.asarray(sigma_los2, dtype=np.float64).copy()
    )
    cert: dict[str, Any] = dict(
        rows_added=0,
        rows_levels=0,
        rows_seconds=0.0,
        est_first=float("nan"),
        est_last=float("nan"),
        rows_end="none",
        rows_unread=0,
        rows_refused=0,
    )
    t0 = time.perf_counter()
    started = (
        new_pool(get_workers(), route) if pool is None else contextlib.nullcontext(pool)
    )
    with started as pool_:
        for i, vp, wp, dp, nm, mm, bb, cc, brk, dt, trep in pool_.imap_unordered(
            row_worker, args_list, chunksize=1
        ):
            rows[i], norm[i], m2[i], beyond[i], cusp[i], nvp[i], secs[i] = (
                (vp, wp, dp) + brk,
                nm,
                mm,
                bb,
                cc,
                len(vp),
                dt,
            )
            t_panels[i] = trep.get("panels", 0)
            t_change[i] = trep.get("change", np.nan)
            t_conv[i] = bool(trep.get("converged", False))
            ang += trep.get("angle_counts", 0)
            t_change_pw[i] = trep.get("change_pw", np.nan)
            t_set_by[i] = T_SET_BY.get(trep.get("set_by", ""), 0)
        log.debug(
            "%d rows in %.1f s (%d cusp rows).",
            n_r2d,
            time.perf_counter() - t0,
            int(cusp.sum()),
        )
        if certify:
            (
                rows,
                norm,
                m2,
                beyond,
                cusp,
                nvp,
                secs,
                t_panels,
                t_change,
                t_conv,
                t_change_pw,
                t_set_by,
            ) = (
                list(a)
                for a in (
                    rows,
                    norm,
                    m2,
                    beyond,
                    cusp,
                    nvp,
                    secs,
                    t_panels,
                    t_change,
                    t_conv,
                    t_change_pw,
                    t_set_by,
                )
            )
            r_list = list(r_2d_arr)
            sig_list = None if sigma_arr is None else list(sigma_arr)
            sl2_list = None if sl2_arr is None else list(sl2_arr)
            prev = None
            gap_shut: dict[Any, tuple[float, float]] = {}
            gap_prev: dict[Any, float] = {}
            gaps_read: set[Any] = set()
            gaps_rows = 0
            gap_anchor: dict[Any, float] = {}
            gap_last: dict[Any, list[float]] = {}
            gap_why: dict[Any, str] = {}
            gap_ang: dict[float, Any] = {}
            cert["gaps_back"] = 0
            t1 = time.perf_counter()
            while True:
                ok_now = np.array([np.isfinite(x) and x > 0.0 for x in norm])
                table = PanelTable.from_rows(
                    lat,
                    np.array(r_list),
                    [
                        (r[0], r[1], r[2] / norm[i] if ok_now[i] else r[2])
                        + tuple(r[3:])
                        for i, r in enumerate(rows)
                    ],
                    r_breaks=r_breaks,
                    r_edge=r_edge,
                )
                fail: list[float] = []
                worst, unread = 0.0, 0
                own: dict[Any, list[tuple[float, float]]] = {}
                int_now: dict[float, tuple[float, float, float]] = {}
                todo = []
                for i in range(len(r_list) - 1):
                    if table.stretch[i] != table.stretch[i + 1]:
                        continue
                    st = int(table.stretch[i])
                    key = (
                        (st, False)
                        if table.R_2D[i + 1] <= gap_anchor.get((st, False), -np.inf)
                        else (st, True)
                        if table.R_2D[i] >= gap_anchor.get((st, True), np.inf)
                        else None
                    )
                    if key in gap_shut:
                        continue
                    todo.append((i, key))
                mids = {i: table.interval_mid(i) for i, _ in todo}
                mi = [i for i, _ in todo if mids[i] is not None]
                mc = (
                    table.mid_curves(route, np.array([mids[i] for i in mi]))
                    if mi
                    else {}
                )
                mcv = {i: {x: y[j] for x, y in mc.items()} for j, i in enumerate(mi)}
                for i, key in todo:
                    est = table.interval_estimate(
                        route, i, rr.row_order_test, cv=mcv.get(i)
                    )
                    if est is None:
                        unread += 1
                        continue
                    rm, em, ep = est
                    bad = max(em / ROW_TOL_M2, ep / ROW_TOL_PW)
                    if key is not None:
                        own.setdefault(key, []).append((rm, bad))
                        continue
                    worst = max(worst, bad)
                    if bad > 1.0:
                        fail.append(rm)
                        int_now[float(rm)] = (
                            float(table.R_2D[i]),
                            float(table.R_2D[i + 1]),
                            bad,
                        )
                gap_now: dict[float, tuple[Any, tuple[float, float]]] = {}
                reads: dict[
                    Any, tuple[float, list[tuple[float, float]], tuple[float, float]]
                ] = {}
                for st in range(len(table._st_lo)):
                    for top in (False, True):
                        if (st, top) in gap_shut:
                            continue
                        end = float(
                            np.exp(table._st_hi[st] if top else table._st_lo[st])
                        )
                        if not read_range[0] <= end <= read_range[1]:
                            continue
                        g = table.gap_estimate(route, st, top, rr.row_order_test)
                        if g is None and (st, top) not in own:
                            continue
                        if g is not None:
                            gap_anchor.setdefault((st, top), float(table.R_2D[g[3]]))
                        pts = own.get((st, top), []) + (
                            []
                            if g is None
                            else [(g[0], max(g[1] / ROW_TOL_M2, g[2] / ROW_TOL_PW))]
                        )
                        gaps_read.add((st, top))
                        re_ = float(
                            np.exp(table._st_hi[st] if top else table._st_lo[st])
                        )
                        ra_ = gap_anchor[(st, top)]
                        reads[(st, top)] = (
                            max(b for _, b in pts),
                            pts,
                            (min(re_, ra_), max(re_, ra_)),
                        )
                gworst = max([0.0] + [x[0] for x in reads.values()])
                back: list[float] = []
                for key, (bad, _pts, rng) in reads.items():
                    if bad > 1.0 and key in gap_prev and bad > 0.5 * gap_prev[key]:
                        gap_shut[key] = rng
                        gap_why[key] = (
                            f"stalled, {gap_prev[key]:.1e} to {bad:.1e} of the "
                            f"tolerance, "
                            f"{len(gap_last.get(key, []))} row(s) taken back"
                        )
                        back += gap_last.pop(key, [])
                if back:
                    keep = [k for k, r in enumerate(r_list) if r not in set(back)]
                    (
                        r_list,
                        rows,
                        norm,
                        m2,
                        beyond,
                        cusp,
                        nvp,
                        secs,
                        t_panels,
                        t_change,
                        t_conv,
                        t_change_pw,
                        t_set_by,
                    ) = (
                        [lst[k] for k in keep]
                        for lst in (
                            r_list,
                            rows,
                            norm,
                            m2,
                            beyond,
                            cusp,
                            nvp,
                            secs,
                            t_panels,
                            t_change,
                            t_conv,
                            t_change_pw,
                            t_set_by,
                        )
                    )
                    if sig_list is not None:
                        sig_list = [sig_list[k] for k in keep]
                    if sl2_list is not None:
                        sl2_list = [sl2_list[k] for k in keep]
                    for r in back:
                        ang -= gap_ang.pop(r)
                    cert["rows_added"] -= len(back)
                    cert["gaps_back"] += len(back)
                    gaps_rows -= len(back)
                    continue
                for key, (bad, pts, rng) in reads.items():
                    if bad > 1.0 and key not in gap_shut:
                        gap_prev[key] = bad
                        for rm, b in pts:
                            if b > 1.0:
                                fail.append(rm)
                                gap_now[float(rm)] = (key, rng)
                if cert["rows_levels"] == 0:
                    cert["est_first"], cert["gap_first"] = worst, gworst
                cert["est_last"], cert["rows_unread"], cert["gap_last"] = (
                    worst,
                    unread,
                    gworst,
                )
                if not fail:
                    cert["rows_end"] = "tolerance"
                    break
                if prev is not None and worst > 1.0 and worst > 0.5 * prev:
                    cert["rows_end"] = "stalled"
                    for key, rng in gap_now.values():
                        gap_shut[key], gap_why[key] = (
                            rng,
                            "still failing when the rows ended",
                        )
                    for lo, hi, b in int_now.values():
                        gap_shut[("interval", lo)] = (lo, hi)
                        gap_why[("interval", lo)] = (
                            f"still {b:.1e} of the tolerance when the rows ended "
                            f"(stalled)"
                        )
                    break
                prev = worst
                new = sorted(set(float(r) for r in fail))
                passing = [r for r in new if route.row_test(r)[0]]
                cert["rows_refused"] += len(new) - len(passing)
                for r in set(new) - set(passing):
                    if r in gap_now:
                        gap_shut[gap_now[r][0]] = gap_now[r][1]
                        gap_why[gap_now[r][0]] = (
                            f"the row test refuses its row ({route.row_test(r)[2]})"
                        )
                new = passing
                got = 0
                added: dict[Any, list[float]] = {}
                for res in pool_.imap_unordered(
                    row_worker_or_none,
                    [(k, r, kept) for k, r in enumerate(new)],
                    chunksize=1,
                ):
                    if res[1] is None:
                        cert["rows_refused"] += 1
                        if new[res[0]] in gap_now:
                            gap_shut[gap_now[new[res[0]]][0]] = gap_now[new[res[0]]][1]
                            gap_why[gap_now[new[res[0]]][0]] = (
                                f"its row refused by its own breaks {BREAK_PAST_END}"
                            )
                        continue
                    _, vp, wp, dp, nm, mm, bb, cc, brk, dt, trep = res
                    st_new = int(np.searchsorted(table.R_breaks, new[res[0]]))
                    if int(brk[5]) != int(
                        table._layout[np.flatnonzero(table.stretch == st_new)[0]]
                    ):
                        cert["rows_refused"] += 1
                        if new[res[0]] in gap_now:
                            gap_shut[gap_now[new[res[0]]][0]] = gap_now[new[res[0]]][1]
                            gap_why[gap_now[new[res[0]]][0]] = (
                                "its row's layout is not the stretch's"
                            )
                        continue
                    if new[res[0]] in gap_now:
                        gaps_rows += 1
                        added.setdefault(gap_now[new[res[0]]][0], []).append(
                            new[res[0]]
                        )
                        gap_ang[new[res[0]]] = trep.get("angle_counts", 0)
                    rows.append((vp, wp, dp) + brk)
                    norm.append(nm)
                    m2.append(mm)
                    beyond.append(bb)
                    cusp.append(cc)
                    nvp.append(len(vp))
                    secs.append(dt)
                    r_list.append(new[res[0]])
                    t_panels.append(trep.get("panels", 0))
                    t_change.append(trep.get("change", np.nan))
                    t_conv.append(bool(trep.get("converged", False)))
                    t_change_pw.append(trep.get("change_pw", np.nan))
                    t_set_by.append(T_SET_BY.get(trep.get("set_by", ""), 0))
                    ang += trep.get("angle_counts", 0)
                    got += 1
                gap_last = added
                if got == 0:
                    if any(k in gap_shut for k, _ in gap_now.values()):
                        continue
                    cert["rows_end"] = "no row could be added"
                    for lo, hi, b in int_now.values():
                        gap_shut[("interval", lo)] = (lo, hi)
                        gap_why[("interval", lo)] = (
                            f"still {b:.1e} of the tolerance when the rows ended "
                            f"(no row could be added)"
                        )
                    break
                if sig_list is not None or sl2_list is not None:
                    if sigma_at is None:
                        raise ResultError(
                            "Rows were added to the table and no sigma_at was given "
                            "for their self-check references."
                        )
                    sg, s2 = sigma_at(np.array(r_list[len(r_list) - got :]))
                    if sig_list is not None:
                        sig_list += list(np.asarray(sg, dtype=np.float64))
                    if sl2_list is not None:
                        sl2_list += list(np.asarray(s2, dtype=np.float64))
                cert["rows_added"] += got
                cert["rows_levels"] += 1
                order = np.argsort(np.array(r_list), kind="stable")
                (
                    r_list,
                    rows,
                    norm,
                    m2,
                    beyond,
                    cusp,
                    nvp,
                    secs,
                    t_panels,
                    t_change,
                    t_conv,
                    t_change_pw,
                    t_set_by,
                ) = (
                    [lst[k] for k in order]
                    for lst in (
                        r_list,
                        rows,
                        norm,
                        m2,
                        beyond,
                        cusp,
                        nvp,
                        secs,
                        t_panels,
                        t_change,
                        t_conv,
                        t_change_pw,
                        t_set_by,
                    )
                )
                if sig_list is not None:
                    sig_list = [sig_list[k] for k in order]
                if sl2_list is not None:
                    sl2_list = [sl2_list[k] for k in order]
            cert["rows_seconds"] = time.perf_counter() - t1
            cert["gaps_read"], cert["gaps_rows"] = len(gaps_read), gaps_rows
            u_ = sorted(gap_shut.items(), key=lambda kv: kv[1])
            cert["gaps_uncert"] = [rng for _, rng in u_]
            cert["gaps_why"] = [
                (
                    "an interval between two rows, "
                    if k[0] == "interval"
                    else "an end gap, "
                )
                + gap_why.get(k, "")
                for k, _ in u_
            ]
            cert["ints_uncert"] = sum(1 for k, _ in u_ if k[0] == "interval")
            r_2d_arr = np.array(r_list)
            norm, m2, beyond, secs, t_change = (
                np.array(a, dtype=np.float64)
                for a in (norm, m2, beyond, secs, t_change)
            )
            cusp, t_conv = np.array(cusp, dtype=bool), np.array(t_conv, dtype=bool)
            nvp, t_panels = np.array(nvp, dtype=int), np.array(t_panels, dtype=np.int64)
            t_change_pw = np.array(t_change_pw, dtype=np.float64)
            t_set_by = np.array(t_set_by, dtype=np.int8)
            sigma_arr = None if sig_list is None else np.array(sig_list)
            sl2_arr = None if sl2_list is None else np.array(sl2_list)
            n_r2d = len(r_2d_arr)
            log.debug(
                "The table read between its rows: the worst interval %.2e of the "
                "tolerance before and %.2e after; %d row(s) added in %d round(s), %d "
                "refused; ended: %s; %d rows; %d end gap(s) and %d interval(s) left "
                "above the tolerance.",
                cert["est_first"],
                cert["est_last"],
                cert["rows_added"],
                cert["rows_levels"],
                cert["rows_refused"],
                cert["rows_end"],
                n_r2d,
                len(cert["gaps_uncert"]) - cert["ints_uncert"],
                cert["ints_uncert"],
            )
    ok = np.isfinite(norm) & (norm > 0.0)
    norm_rel = np.full(n_r2d, np.nan)
    m2_rel = np.full(n_r2d, np.nan)
    if sigma_arr is not None:
        sig = np.asarray(sigma_arr, dtype=np.float64) * df_scale
        norm_rel = np.where(
            ok & (sig > 0.0), norm / np.where(sig > 0.0, sig, 1.0) - 1.0, np.nan
        )
    table = PanelTable.from_rows(
        lat,
        r_2d_arr,
        [
            (r[0], r[1], r[2] / norm[i] if ok[i] else r[2]) + tuple(r[3:])
            for i, r in enumerate(rows)
        ],
        r_breaks=r_breaks,
        r_edge=r_edge,
    )
    table.uncert = np.asarray(cert.get("gaps_uncert", []), dtype=np.float64).reshape(
        -1, 2
    )
    table.uncert_why = list(cert.get("gaps_why", []))
    if sl2_arr is not None:
        sl2 = np.asarray(sl2_arr, dtype=np.float64)
        m2_rel = np.where(
            ok & (sl2 > 0.0),
            (m2 / np.where(ok, norm, 1.0)) / np.where(sl2 > 0.0, sl2, 1.0) - 1.0,
            np.nan,
        )
    beyond_rel = np.where(ok, beyond / np.where(ok, norm, 1.0), 0.0)
    return (
        table,
        norm_rel,
        m2_rel,
        beyond_rel,
        dict(
            cusp=cusp,
            n_vpanel=nvp,
            row_seconds=secs,
            t_panels=t_panels,
            t_change=t_change,
            t_converged=t_conv,
            R_2D=r_2d_arr,
            Sigma_R2D=sigma_arr,
            siglos2_ref=sl2_arr,
            t_change_pw=t_change_pw,
            t_set_by=t_set_by,
            angle_counts=ang,
            **cert,
        ),
    )
