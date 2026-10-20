"""Where V_rad and V_k are kept as breaks of a row, over projected radius."""

__all__ = ["extrema_in_pool", "single_v", "vk_kept_ranges", "vrad_kept_ranges"]

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..typing import Array, ArrayLike
from ._ellipse import EllipseRoute
from ._ellipse_rules import VRAD_KEEP, gl01
from ._layout import LAYOUT_SEQ

# The worker's route, built once a process by the pool's initializer.
WORKER: dict[str, EllipseRoute] = {}


def single_v_worker(args: tuple[float, float]) -> float:
    r"""D at one :math:`(R_{\rm 2D}, V)` by the row's own rule, in a pool worker."""
    r2, v = args
    route = WORKER["route"]
    rule, _ = route.rule(r2)
    return float(route.row(r2, np.array([v]), rule)[0])


def extrema_worker(r2: Array) -> list[Array]:
    """`EllipseRoute.f_extrema_many` at the radii, in a pool worker."""
    return WORKER["route"].f_extrema_many(r2)


def extrema_in_pool(
    route: EllipseRoute, r2: ArrayLike, pool_of: Callable[[], Any], n_parts: int
) -> None:
    """Compute `EllipseRoute.f_extrema` at the radii the route has not cached,
    in `n_parts` pieces in the pool `pool_of` returns (called only where a
    radius is not cached), and cache them on the route. Every number is the
    one `f_extrema` forms at that radius alone, so the route reads
    the same extrema as it would have formed."""
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
    """Return D(R2[i], V[i]) for each i: in the pool where `pmap` is its map,
    else one call after another."""
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
    r"""The ranges where bound > VRAD_KEEP: each change between neighboring
    samples bisected in :math:`\ln R_{\rm 2D}`, the ends 0 and inf where a
    range runs off."""
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
    r"""Return (ranges, info): the :math:`R_{\rm 2D}` ranges where V_rad is
    kept as a break.

    The bound is B = 2 D(V_rad) (V_end - V_rad) / Sigma, or V_end^2 times that
    over Sigma :math:`\sigma_{\rm los}^2` where larger, with D(V_rad) one
    single-V row and Sigma, :math:`\sigma_{\rm los}^2` from `sigma_fn`; 0
    where the floor merges V_rad, inf where D > 0 on a row without surface
    density. Taken on the base rows, each change bisected.
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
) -> tuple[Array, dict[str, Any]]:
    r"""Return (ranges, info): the :math:`R_{\rm 2D}` ranges where V_k is kept
    as a break.

    The bound of `vrad_kept_ranges` for both panels at V_k, [V_k, V_end] and
    [0, V_k], the smaller of the two; inf, and no single-V row, where V_k lies
    between the first and last node of the panel it would be merged into.
    Taken on the base rows and on rb_ext samples beyond the last toward the
    parent's end.
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
            sig, sl2 = (
                np.atleast_1d(np.asarray(a, dtype=np.float64))
                for a in sigma_fn(r2[idx])
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
