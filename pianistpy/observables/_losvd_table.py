r"""The rows of the line-of-sight velocity distribution, computed in a process
pool, and the check of the table's interpolation between rows.

The stage behind `pianistpy.losvd`, `pianistpy.aperture_losvd` and
`pianistpy.LosTable` calls `losvd_rows` to build the table. Each row holds
:math:`D(R_{\rm 2D}, V)`, the line-of-sight velocity distribution at one
projected radius :math:`R_{\rm 2D}` on the row's velocity panels. Each row is
normalized to unit integral over :math:`V`. The radii where the panel layout
changes split the rows into stretches. A true end of a stretch is a radius
where the layout changes, or the finite radius where the line of sight ends.

The check estimates the interpolation error at the midpoint of each interval
between neighboring rows, and in each gap between a true end of a stretch and
the stretch's nearest row. With :math:`D_k` the row that interpolation over
:math:`k` rows gives and :math:`D_{k-2}` the row that interpolation over
:math:`k - 2` rows gives, the two estimates are

.. math::

    e_2 = \frac{|m_2[D_k] - m_2[D_{k-2}]|}{|m_2[D_k]|}, \qquad
    e_\infty = \frac{\max |D_k - D_{k-2}|}{\max |D_k|},

with :math:`m_2[D] = \int V^2 D\,{\rm d}V` and :math:`k` the setting
``losvd.n_stencil_test_rows``. A midpoint fails where :math:`e_2` exceeds
ROW_TOL_M2 (:math:`10^{-8}`) or :math:`e_\infty` exceeds ROW_TOL_PW
(:math:`10^{-7}`). Notes Section T:6.5 treats the table of rows."""

__all__ = [
    "START_METHOD",
    "init_worker",
    "init_worker_from",
    "losvd_rows",
    "new_pool",
    "row_worker",
    "row_worker_or_none",
]

import contextlib
import multiprocessing as mp
import signal
import time
import weakref
from collections.abc import Callable
from multiprocessing import pool as mp_pool
from multiprocessing import util as mp_util
from typing import Any

import numpy as np

from ..exceptions import ResultError, WorkerError
from ..runtime import (
    drop_spill,
    get_cores,
    log,
    spill,
    unspill,
    worker_environment,
)
from ..typing import Array, ArrayLike
from ._ellipse import T_SET_BY, EllipseRoute
from ._ellipse_rules import BREAK_PAST_END, ROW_TOL_M2, ROW_TOL_PW
from ._kept import WORKER
from ._panel_table import PanelTable

# The pool's start method. Every platform offers "spawn", and JAX allows
# "spawn" after JAX has initialized. The method "fork" meets neither
# condition. Each spawned process imports the package and loads a pickled
# copy of the route.
START_METHOD = "spawn"
# The interval, in seconds, at which a pool call checks for an ended process
# while the call waits for results.
WORKER_POLL = 0.1


class _RoutePool(mp_pool.Pool):
    """A `multiprocessing.pool.Pool` whose processes load the route from a
    file, and whose calls raise `WorkerError` when a process ends.

    `new_pool` writes the route to a file in a folder of its own. The
    folder is removed when the pool is terminated or garbage-collected
    (``remove_route``).

    The methods `map` and `imap_unordered` check for an ended process every
    WORKER_POLL seconds while waiting for results, and raise `WorkerError`
    at the first check after a process of the pool has ended. The standard
    pool replaces an ended process with a new process, but the task that the
    ended process held is lost. The standard pool's wait for that task's
    result would never return. Both methods wait on the standard pool's own
    calls (`map_async`, and the iterator of the standard `imap_unordered`),
    so the results and the result order are the standard pool's.

    The pool records every process it starts, in ``workers_started``,
    through the pool's ``Process`` hook, which the standard library does not
    document. The record holds the first processes and each replacement
    process. A process that starts and ends between two checks is therefore
    also found.

    The pool's termination does not wait forever on an ended process. A
    process takes the lock of the pool's task pipe to read a task. A process
    that ends while holding the lock never releases the lock. CPython
    3.11's ``Pool._help_stuff_finish`` takes that lock with no time limit,
    so the termination would wait forever. `_help_stuff_finish` replaces the
    standard method with a bounded wait.
    """

    remove_route: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.workers_started: list[Any] = []
        self.workers_reported: set[Any] = set()
        super().__init__(*args, **kwargs)

    def _setup_queues(self) -> None:
        super()._setup_queues()  # type: ignore[misc]
        # The termination's helper `_help_stuff_finish` receives the task
        # queue and no pool, so the record of the processes is attached to
        # the task queue. A pickled queue carries only the queue's pipes and
        # locks, so no process of the pool receives the record.
        self._inqueue.workers_started = self.workers_started  # type: ignore[attr-defined]

    def Process(self, ctx: Any, *args: Any, **kwds: Any) -> Any:  # type: ignore[override]  # noqa: N802
        process = ctx.Process(*args, **kwds)
        self.workers_started.append(process)
        return process

    def terminate(self) -> None:
        super().terminate()
        if self.remove_route is not None:
            self.remove_route()

    @staticmethod
    def _help_stuff_finish(inqueue: Any, task_handler: Any, size: int) -> None:
        """Replace CPython 3.11's ``Pool._help_stuff_finish`` with a bounded
        acquire of the task pipe's read lock.

        The standard method acquires the read lock with no time limit, then
        drains the task pipe while the task handler is alive and the pipe
        holds data. This method tries to acquire the read lock for
        WORKER_POLL seconds at a time. After each failed try, three checks
        run in order:

        1. Where the task handler has ended, the method returns without
           draining the pipe.
        2. Where at least one recorded process has ended, the method kills
           every recorded process that is still alive. An ended process may
           hold the read lock forever. The pool starts no new processes
           during termination.
        3. Where every recorded process has ended, the method stops trying
           for the lock.

        The method then drains the task pipe as the standard method does,
        with or without the read lock. Without the read lock, no process
        that reads the pipe is left alive. The drain keeps the task handler
        from blocking on a full pipe.
        """
        mp_util.debug("removing tasks from inqueue until task handler finished")
        workers = getattr(inqueue, "workers_started", [])
        while not inqueue._rlock.acquire(timeout=WORKER_POLL):
            if not task_handler.is_alive():
                return
            started = list(workers)
            if any(p.exitcode is not None for p in started):
                for p in started:
                    if p.exitcode is None:
                        p.kill()
                if all(p.exitcode is not None for p in started):
                    break
        while task_handler.is_alive() and inqueue._reader.poll():
            inqueue._reader.recv()
            time.sleep(0)

    def raise_if_ended(self) -> None:
        """Raise `WorkerError` where a process of the pool has ended and no
        earlier call has reported that process.

        The message identifies each newly ended process by its process id, with
        its exit code or the signal that ended it. Each ended process is
        reported once.

        Raises
        ------
        WorkerError
            If a process of the pool has ended and has not been reported.
        """
        # The pool's own thread appends to workers_started as the pool starts
        # processes, so the check reads a copy of the list.
        ended = [
            p
            for p in list(self.workers_started)
            if p.exitcode is not None and p not in self.workers_reported
        ]
        if not ended:
            return
        self.workers_reported.update(ended)
        codes = ", ".join(
            f"process {p.pid}: "
            + (
                f"ended by {signal.Signals(-p.exitcode).name}"
                if p.exitcode < 0
                else f"exit code {p.exitcode}"
            )
            for p in ended
        )
        raise WorkerError(
            f"A parallel process ended during the computation ({codes}), and the "
            "task that the process held is lost, so the call stops instead of "
            "waiting for the task. The usual cause is memory: the operating system "
            "ends a process when the machine runs out of memory. Fewer cores lower "
            f"the load: set_cores (now {get_cores()}), or the environment "
            "variable PIANISTPY_CORES."
        )

    def map(self, func: Any, iterable: Any, chunksize: Any = None) -> list[Any]:
        result = self.map_async(func, iterable, chunksize)
        while not result.ready():
            self.raise_if_ended()
            result.wait(WORKER_POLL)
        return result.get()

    def imap_unordered(
        self, func: Any, iterable: Any, chunksize: int | None = 1
    ) -> Any:
        return self._watched(super().imap_unordered(func, iterable, chunksize))

    def _watched(self, results: Any) -> Any:
        """Yield the results of the standard `imap_unordered` as they arrive,
        and check for an ended process (`raise_if_ended`) before each wait of
        at most WORKER_POLL seconds."""
        while True:
            self.raise_if_ended()
            try:
                item = results.next(WORKER_POLL)
            except mp.TimeoutError:
                continue
            except StopIteration:
                return
            yield item


def new_pool(n_workers: int, route: EllipseRoute) -> Any:
    """Return a `_RoutePool` of `n_workers` spawned processes on the CPU,
    each process holding a copy of `route`.

    Each process runs OpenBLAS, MKL and OpenMP on one thread and runs JAX on
    the CPU. The environment variables that set these choices are those of
    `runtime.worker_environment`, set while the processes are spawned and
    restored afterward.

    The function pickles the route once, into a file in a folder of its own
    under the temporary directory (`runtime.spill`). Each process loads the
    route from the file (`init_worker_from`). The pool starts its processes
    one after another. A spawn writes the initializer's arguments into the
    new process's pipe. A write larger than the pipe's buffer completes only
    once the new process has started and read the arguments. Passing the
    route itself as the argument would therefore make the start time grow
    with the number of processes. The folder is removed when the pool is
    terminated or garbage-collected. A call of the pool raises `WorkerError`
    where a process of the pool has ended (`_RoutePool`).
    """
    path, folder = spill(route, "route")
    try:
        with worker_environment():
            pool = _RoutePool(
                n_workers,
                init_worker_from,
                (path,),
                context=mp.get_context(START_METHOD),
            )
    except BaseException:
        drop_spill(folder)
        raise
    pool.remove_route = weakref.finalize(pool, drop_spill, folder)
    return pool


def init_worker(route: EllipseRoute) -> None:
    """Store `route` as the route of the calling process (`_kept.WORKER`)."""
    WORKER["route"] = route


def init_worker_from(path: str) -> None:
    """Load the route from the file at `path`, which `new_pool` wrote, and
    store the route as the route of the calling process (`init_worker`).
    The pool's initializer."""
    init_worker(unspill(path))


def row_worker(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Compute one row in a pool process.

    The argument `args` is ``(index, R_2D)`` or
    ``(index, R_2D, (vrad_kept, vk_kept))``. A third element sets the kept
    ranges of V_rad and V_k on the process's route. The function returns the
    tuple ``(index, V, w, D, norm, m2, beyond, cusp, breaks, seconds, record)``.
    V, w and D are the row's nodes, weights and values of D on ``V >= 0``.
    norm is the integral of D over V. m2 is the integral of V**2 D over V.
    beyond is the integral of D over the velocities above the escape velocity. Each
    integral covers both signs of V. cusp is true where the row is of the
    cusp class. breaks is
    ``(v_esc, V_k, V_end, V_rad, n_panels, code, V_edge, V_c)``. seconds is
    the wall-clock time of the row. record is the dict that the route's row
    computation fills, with the counts of the angle rules added under
    "angle_counts".
    """
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
    """Compute one row that the check adds, as `row_worker` does.

    Where the row's layout puts a break at or above the row's own V_end,
    `row_worker` raises a ResultError that carries BREAK_PAST_END, and this
    function returns ``(index, None)`` instead. Every other ResultError
    propagates.
    """
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
    r"""Return the table of rows at the projected radii `r_2d`, after the
    check of the table's interpolation between rows.

    The pool computes each row (`row_worker`). The table normalizes each
    row's D to unit integral over V where the row's integral is finite and
    positive. Where `certify` is true, the check then runs in rounds. Each
    round computes a failure ratio at the midpoint of every interval between
    neighboring rows of one stretch, and in every end gap of a stretch whose
    end lies inside `read_range`. The failure ratio is
    ``max(e_2 / ROW_TOL_M2, e_inf / ROW_TOL_PW)``, with ``e_2`` and
    ``e_inf`` the second-moment estimate and the pointwise estimate of
    `PanelTable._estimate` over ``rules.row_order_test`` rows. A midpoint
    fails where its failure ratio exceeds 1, and each failed midpoint gets a
    new row. The rounds stop in one of three cases:

    1. no midpoint fails (recorded as "tolerance");
    2. the worst failure ratio of the intervals exceeds 1 and exceeds half
       of the previous round's worst (recorded as "stalled");
    3. no new row could be added and no end gap was closed in the round
       (recorded as "no row could be added").

    An end gap is closed where the gap's worst failure ratio exceeds 1 and
    exceeds half of the gap's worst failure ratio in the previous round. The
    rows that the previous round added to a closed end gap are removed. A
    new row is not added in three cases: `EllipseRoute.row_test` rejects the
    row; the row's layout puts a break at or above the row's V_end; or the
    row's layout code differs from the code of the row's stretch. Where such
    a row lies in an end gap, that end gap is closed. Every closed end gap,
    and every interval still failing when the rounds stop in case 2 or 3,
    is recorded in the table's ``uncert`` and ``uncert_why``. The table
    raises ResultError for a radius inside a recorded interval
    (`PanelTable._prepare`).

    Parameters
    ----------
    route : EllipseRoute
        The route of the calling process, which reads the table between
        rows. Each process of the pool holds a copy (`init_worker`).
    r_2d : array-like
        The projected radii of the base rows, in internal units.
    sigma : array-like or None
        The surface density at the base rows, the reference of each row's
        norm.
    sigma_los2 : array-like or None
        The squared line-of-sight velocity dispersion at the base rows, the
        reference of each row's second moment.
    r_breaks : array-like, optional
        The projected radii where the layout changes. Default: ``()``, which
        means one stretch.
    vrad_kept : array or None, optional
        The kept ranges of V_rad, shape (n, 2). Default: None, which means
        V_rad is kept wherever the lattice resolves V_rad.
    vk_kept : array or None, optional
        The kept ranges of V_k, shape (n, 2). Default: None, which means V_k
        is kept wherever the lattice resolves V_k.
    sigma_at : callable, optional
        The function that returns (:math:`\Sigma`,
        :math:`\sigma_{\rm los}^2`) at the radii of added rows. Default: None,
        which means no function. A call that adds rows while `sigma` or
        `sigma_los2` is given then raises ResultError.
    certify : bool, optional
        Whether to run the check. Default: ``True``.
    r_edge : float, optional
        The radius where the line of sight ends. A finite `r_edge` is a true
        end of the last stretch. Default: infinity.
    pool : optional
        A pool whose processes hold the route (`new_pool`). Default: None,
        which means a new pool of `get_cores` processes, terminated at the
        end of the call.
    df_scale : float, optional
        The constant factor of the rows' DF (`EllipseDF.scale`). A row
        integrates to the surface density times `df_scale`. Default: ``1.0``.
    read_range : tuple of float, optional
        The range of radii the table is read over. The check skips an end gap
        whose end lies outside `read_range`. Default: ``(0.0, inf)``.

    Returns
    -------
    table : PanelTable
        The table of normalized rows, sorted by radius.
    norm_rel : array
        Each row's integral over V divided by (`sigma` times `df_scale`),
        less one. The value is NaN where `sigma` is None, where `sigma` is not
        positive, or where the row's integral is not finite and positive.
    m2_rel : array
        Each row's normalized second moment in V divided by `sigma_los2`,
        less one. The value is NaN where `sigma_los2` is None, where
        `sigma_los2` is not positive, or where the row's integral is not
        finite and positive.
    beyond_rel : array
        Each row's fraction of its integral above the escape velocity. The value
        is 0 where the row's integral is not finite and positive.
    info : dict
        The record of the rows and of the check.

    Raises
    ------
    WorkerError
        If a process of the pool ends during the call.
    ResultError
        If rows are added while `sigma` or `sigma_los2` is given and
        `sigma_at` is None.
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
        new_pool(get_cores(), route) if pool is None else contextlib.nullcontext(pool)
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
            "The table computed %d projected radii in %.1f s, %d of the projected "
            "radii with a cusp.",
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
                            f"where the error estimate stalled from "
                            f"{gap_prev[key]:.1e} to {bad:.1e} times the tolerance "
                            f"and {len(gap_last.get(key, []))} added radii were "
                            f"removed"
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
                            "where the error estimate was still above the "
                            "tolerance when the table stopped adding radii",
                        )
                    for lo, hi, b in int_now.values():
                        gap_shut[("interval", lo)] = (lo, hi)
                        gap_why[("interval", lo)] = (
                            f"where the error estimate was still {b:.1e} times the "
                            f"tolerance when the table stopped adding radii (stalled)"
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
                            f"where the new radius cannot be stored, because the "
                            f"new radius has {route.row_test(r)[2]}"
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
                                f"where the new radius cannot be stored, because "
                                f"the new radius has a velocity break on or above "
                                f"V_end {BREAK_PAST_END}"
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
                                "where the new radius cannot be stored, because "
                                "the panel layout at the new radius differs from "
                                "the stretch's panel layout"
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
                            f"where the error estimate was still {b:.1e} times the "
                            f"tolerance when the table stopped adding radii (no "
                            f"radius could be added)"
                        )
                    break
                if sig_list is not None or sl2_list is not None:
                    if sigma_at is None:
                        raise ResultError(
                            "The table added projected radii, and no sigma_at was "
                            "given to compute the reference dispersions at the "
                            "added radii for the table's self-check."
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
                    "an interval between two tabulated radii, "
                    if k[0] == "interval"
                    else "an end gap of a stretch of tabulated radii, "
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
                "The table's check between tabulated radii gives the worst error "
                "estimate %.2e times the tolerance before adding radii and %.2e "
                "times the tolerance after. The table added %d radii in %d rounds "
                "and could not store %d candidate radii; the rounds ended with the "
                "status '%s'. The table has %d tabulated radii, and %d end gaps and "
                "%d intervals remain above the tolerance.",
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
