r"""The ranges of projected radius where V_rad and V_k are kept as breaks of
the velocity panels of a row.

The kept ranges are computed once per model by the stage behind the
line-of-sight velocity distribution of `pianistpy.losvd`,
`pianistpy.aperture_losvd` and `pianistpy.LosTable`. At a projected radius
:math:`R_{\rm 2D}`, let :math:`D(V)` be the unnormalized line-of-sight
velocity distribution at velocity :math:`V`, :math:`V_{\rm end}` the velocity above
which :math:`D` is zero, :math:`\Sigma` the surface density and
:math:`\sigma_{\rm los}^2` the squared line-of-sight velocity dispersion. A
panel :math:`[V_{\rm a}, V_{\rm b}]` next to a break :math:`V_{\rm x}` has
the share :math:`S = 2 D(V_{\rm x}) (V_{\rm b} - V_{\rm a})`, an estimate of
the panel's part of the integral of :math:`D` over :math:`-V_{\rm end} \le V
\le V_{\rm end}`. The bound of the panel is

.. math::

    B = \max\left(\frac{S}{\Sigma},
    \frac{S\, V_{\rm b}^2}{\Sigma\, \sigma_{\rm los}^2}\right).

A break is kept as a break where the bound exceeds VRAD_KEEP, which is
:math:`10^{-8}`, one hundredth of :math:`10^{-6}`. The bound of V_rad is the
bound of the panel :math:`[V_{\rm rad}, V_{\rm end}]`. The bound of V_k is the
smaller of the bounds of the panels :math:`[0, V_k]` and
:math:`[V_k, V_{\rm end}]`. Notes Section T:6.4 treats the line-of-sight
velocity distribution."""

__all__ = ["extrema_in_pool", "single_v", "vk_kept_ranges", "vrad_kept_ranges"]

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..typing import Array, ArrayLike
from ._ellipse import EllipseRoute
from ._ellipse_rules import VRAD_KEEP, gl01
from ._layout import LAYOUT_SEQ
from ._projection import END_TOLERANCE

# The ellipse route of a pool process, under the key "route". The pool's
# initializer (`_losvd_table.init_worker`) sets the route once per process.
WORKER: dict[str, EllipseRoute] = {}


def single_v_worker(args: tuple[float, float]) -> float:
    r"""Return D at one pair :math:`(R_{\rm 2D}, V)` in a pool process.

    The value is computed with the row rule that the process's route assigns
    to :math:`R_{\rm 2D}`.
    """
    r2, v = args
    route = WORKER["route"]
    rule, _ = route.rule(r2)
    return float(route.row(r2, np.array([v]), rule)[0])


def extrema_worker(r2: Array) -> list[Array]:
    """Return `EllipseRoute.f_extrema_many` at the projected radii `r2`, in a
    pool process."""
    return WORKER["route"].f_extrema_many(r2)


def extrema_in_pool(
    route: EllipseRoute, r2: ArrayLike, pool_of: Callable[[], Any], n_parts: int
) -> None:
    """Compute `EllipseRoute.f_extrema` in a pool at each projected radius of
    `r2` that the route has not cached, and cache the extrema on the route.

    The function calls `pool_of` for the pool only where at least one radius
    is not cached. The uncached radii are split into at most `n_parts`
    parts, one task per part. `EllipseRoute.f_extrema_many` forms the extrema
    at each radius as `EllipseRoute.f_extrema` forms them at that radius
    alone, so the cached extrema equal the extrema the route would compute
    itself.
    """
    todo = [
        x
        for x in np.unique(np.asarray(r2, dtype=np.float64)).tolist()
        if x not in route._fext
    ]
    if not todo:
        return
    parts = np.array_split(np.array(todo), min(n_parts, len(todo)))
    for part, found in zip(parts, pool_of().map(extrema_worker, parts), strict=True):
        for x, ext in zip(part.tolist(), found, strict=True):
            route._fext[x] = ext


def single_v(
    route_df: EllipseRoute, r2: Array, v: Array, pmap: Callable[..., Any] | None
) -> list[float]:
    """Return D at each pair ``(r2[i], v[i])``.

    Where `pmap` is a pool's map, the pairs are computed in the pool. Where
    `pmap` is None, the pairs are computed one after another with `route_df`
    in the calling process.
    """
    if pmap is not None:
        return list(
            pmap(
                single_v_worker,
                [(float(r), float(x)) for r, x in zip(r2, v, strict=True)],
            )
        )
    return [
        float(route_df.row(float(r), np.array([x]), route_df.rule(float(r))[0])[0])
        for r, x in zip(r2, v, strict=True)
    ]


def _ranges(
    rs: Array, bound: Callable[[Array], Array], n_bisect: int, info: dict[str, Any]
) -> Array:
    r"""Return the ranges of projected radius where ``bound`` exceeds
    VRAD_KEEP, shape (n, 2).

    The function evaluates ``bound`` at the samples `rs`. Where the test
    ``bound > VRAD_KEEP`` changes between two neighboring samples, the
    function bisects the interval `n_bisect` times in
    :math:`\ln R_{\rm 2D}` and places the range end at the geometric mean of
    the final bracket. A range that holds the first sample starts at 0. A
    range that holds the last sample ends at infinity. The function writes
    into `info` two counts: under "rows", the count of evaluations of D at
    the samples, read from ``info["calls"]``; under "changes", the count of
    changes between neighboring samples.
    """
    keep = bound(rs) > VRAD_KEEP
    info["rows"] = info["calls"]
    j = np.flatnonzero(keep[1:] != keep[:-1])
    info["changes"] = int(j.size)
    lo, hi = rs[j].copy(), rs[j + 1].copy()
    for _ in range(n_bisect if j.size else 0):
        mid = np.sqrt(lo * hi)
        same = (bound(mid) > VRAD_KEEP) == keep[j]
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    rx = np.sqrt(lo * hi)
    ranges: list[tuple[float, float]] = []
    start = 0.0 if keep[0] else None
    for t, jj in enumerate(j):
        if keep[jj]:
            ranges.append((start, float(rx[t])))  # type: ignore[arg-type]
            start = None
        else:
            start = float(rx[t])
    if start is not None:
        ranges.append((start, np.inf))
    return np.array(ranges, dtype=np.float64).reshape(-1, 2)


def vrad_kept_ranges(
    route: EllipseRoute,
    route_df: EllipseRoute,
    r_base: ArrayLike,
    sigma_fn: Callable[[Array], tuple[Array, Array]],
    pmap: Callable[..., Any] | None = None,
) -> tuple[Array, dict[str, Any]]:
    r"""Return (ranges, info): the ranges of :math:`R_{\rm 2D}` where V_rad
    is kept as a break, shape (n, 2), and a record of the computation.

    The bound of V_rad is
    :math:`\max(S/\Sigma, S V_{\rm end}^2/(\Sigma \sigma_{\rm los}^2))`,
    with :math:`S = 2 D(V_{\rm rad}) (V_{\rm end} - V_{\rm rad})`.
    :math:`D(V_{\rm rad})` comes from one evaluation of D at the single velocity
    V_rad (`single_v`).
    :math:`\Sigma` and :math:`\sigma_{\rm los}^2` come from `sigma_fn`. The
    bound is 0 where V_rad is not an interior break
    (`Lattice.interior_breaks`) or where :math:`S` is not positive. The bound
    is infinite where :math:`S` is positive and :math:`\Sigma` or
    :math:`\sigma_{\rm los}^2` is not positive. The bound is evaluated on the
    base rows `r_base`, and each change across VRAD_KEEP is bisected
    (`_ranges`). The ranges are empty where the route has no V_rad (``lfac``
    false).
    """
    r = np.asarray(r_base, dtype=np.float64)
    info: dict[str, Any] = {"rows": 0, "calls": 0, "changes": 0, "seconds": 0.0}
    if not route.lfac:
        return np.zeros((0, 2)), info
    t0 = time.perf_counter()

    def bound(r2: Array) -> Array:
        vk, vend, vr = route.breaks_many(r2)
        inside = route.lattice.interior_breaks(
            vk, vend, vr, route.v_edge_many(r2), route.v_c_many(r2)
        )[1]
        b = np.zeros(len(r2))
        idx = np.flatnonzero(inside)
        if idx.size:
            sig, sl2 = (
                np.atleast_1d(np.asarray(a, dtype=np.float64))
                for a in sigma_fn(r2[idx])
            )
            dr = single_v(route_df, r2[idx], vr[idx], pmap)
            for n, i in enumerate(idx):
                share = 2.0 * dr[n] * (vend[i] - vr[i])
                info["calls"] += 1
                if share > 0.0:
                    b[i] = (
                        max(share / sig[n], share * vend[i] ** 2 / (sig[n] * sl2[n]))
                        if sig[n] > 0.0 and sl2[n] > 0.0
                        else np.inf
                    )
        return b

    ranges = _ranges(r, bound, route.rules.rb_bisect, info)
    info["seconds"] = time.perf_counter() - t0
    return ranges, info


def vk_kept_ranges(
    route: EllipseRoute,
    route_df: EllipseRoute,
    r_base: ArrayLike,
    sigma_fn: Callable[[Array], tuple[Array, Array]],
    pmap: Callable[..., Any] | None = None,
    r_read_max: float = np.inf,
) -> tuple[Array, dict[str, Any]]:
    r"""Return (ranges, info): the ranges of :math:`R_{\rm 2D}` where V_k is
    kept as a break, shape (n, 2), and a record of the computation.

    The bound of V_k is the smaller of two panel bounds. The panel
    :math:`[V_k, V_{\rm end}]` has the share
    :math:`S = 2 D(V_k) (V_{\rm end} - V_k)` and the bound
    :math:`\max(S/\Sigma, S V_{\rm end}^2/(\Sigma \sigma_{\rm los}^2))`. The
    panel :math:`[0, V_k]` has the share :math:`S = 2 D(V_k) V_k` and the
    bound :math:`\max(S/\Sigma, S V_k^2/(\Sigma \sigma_{\rm los}^2))`.

    The first of five cases that applies at a sample sets the bound:

    1. 0 where V_k is not an interior break (`Lattice.interior_breaks`);
    2. infinity where V_k lies between the first node and the last node of
       the panel that holds V_k once V_k is removed from the breaks, with no
       evaluation of :math:`D(V_k)`;
    3. 0 where either share is not positive;
    4. infinity where :math:`\Sigma` or :math:`\sigma_{\rm los}^2` is not
       positive. Every sample beyond `r_read_max` times
       (1 + END_TOLERANCE) falls in case 4 or earlier, because the function
       does not call `sigma_fn` there and sets both to 0;
    5. the smaller of the two panel bounds.

    The samples are the base rows `r_base` and ``rules.rb_ext`` further
    samples between the last base row and the route's line-of-sight end
    ``R_t``, the j-th of which lies at ``R_t (1 - d 2**-j)`` with
    ``d = 1 - r_base[-1] / R_t``. Each change across VRAD_KEEP is bisected
    (`_ranges`).
    """
    r = np.asarray(r_base, dtype=np.float64)
    info: dict[str, Any] = {"rows": 0, "calls": 0, "changes": 0, "seconds": 0.0}
    t0 = time.perf_counter()
    rr = route.rules
    lat = route.lattice
    r_end = float(route.R_t)
    d0 = 1.0 - r[-1] / r_end
    ext = (
        r_end * (1.0 - d0 * 0.5 ** np.arange(1, rr.rb_ext + 1))
        if d0 > 0.0
        else np.zeros(0)
    )
    rs = np.concatenate([r, ext])
    xg = gl01(rr.vp_nodes)[0]

    def bound(r2: Array) -> Array:
        vk, vend, vr = route.breaks_many(r2)
        ve, vc = route.v_edge_many(r2), route.v_c_many(r2)
        inside = lat.interior_breaks(vk, vend, vr, ve, vc)[0]
        b = np.zeros(len(r2))
        k_, r_, e_, c_ = lat.resolved_breaks(
            np.full(len(r2), np.nan), vend, route.vrad_break(r2, vr), ve, vc
        )
        code = np.atleast_1d(lat.layout_code(k_, vend, r_, e_, c_))
        for i in np.flatnonzero(inside):
            br = np.array(
                [0.0]
                + [
                    {"k": k_, "r": r_, "e": e_, "c": c_}[s][i]
                    for s in LAYOUT_SEQ[int(code[i])]
                ]
                + [vend[i]]
            )
            p = int(np.searchsorted(br, vk[i])) - 1
            if 0 <= p < len(br) - 1:
                u = lat.panel_units(int(code[i]))[p]
                xi = (vk[i] - br[p]) / (br[p + 1] - br[p])
                if u[0] + (u[1] - u[0]) * xg[0] < xi < u[-2] + (u[-1] - u[-2]) * xg[-1]:
                    b[i] = np.inf
        idx = np.flatnonzero(inside & ~np.isinf(b))
        if idx.size:
            # sigma_fn gives no reference beyond r_read_max. A sample there keeps
            # Sigma = 0 and sigma_los^2 = 0, so V_k is kept wherever both panels
            # next to V_k have a positive share.
            read = r2[idx] <= r_read_max * (1.0 + END_TOLERANCE)
            sig, sl2 = np.zeros(idx.size), np.zeros(idx.size)
            if read.any():
                sig[read], sl2[read] = (
                    np.atleast_1d(np.asarray(a, dtype=np.float64))
                    for a in sigma_fn(r2[idx[read]])
                )
            dks = single_v(route_df, r2[idx], vk[idx], pmap)
            for n, i in enumerate(idx):
                dk = dks[n]
                info["calls"] += 1
                shares = (
                    (2.0 * dk * (vend[i] - vk[i]), vend[i] ** 2),
                    (2.0 * dk * vk[i], vk[i] ** 2),
                )
                if min(s for s, _ in shares) > 0.0:
                    b[i] = (
                        min(
                            max(s / sig[n], s * v2 / (sig[n] * sl2[n]))
                            for s, v2 in shares
                        )
                        if sig[n] > 0.0 and sl2[n] > 0.0
                        else np.inf
                    )
        return b

    ranges = _ranges(rs, bound, rr.rb_bisect, info)
    info["seconds"] = time.perf_counter() - t0
    return ranges, info
