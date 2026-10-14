"""The table of rows of the line-of-sight velocity distribution and its
reading between rows."""

__all__ = ["PanelTable"]

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..exceptions import ResultError
from ..typing import Array, ArrayLike
from ._ellipse import EllipseRoute
from ._ellipse_rules import gl01
from ._layout import BREAK_LABELS, LAYOUT_SEQ, Lattice


class PanelTable:
    """The ragged table of rows, each on its own mirrored V panels.

    `V`, `w`, `D` are flat, row i is ``[off[i], off[i + 1])``, its V >= 0 half
    ordered (panel, sub-panel, node). Rows of one stretch between R-breaks
    share a layout, and the stencil reads between them in theta =
    arccos(1 - 2 xi) of ln R across the stretch (a square root only at a true
    end: an R-break, or `r_edge`).

    Parameters
    ----------
    lattice : Lattice
        The grading of the rows' panels.
    r_2d, off, v, w, d : array
        The rows' radii, offsets and their nodes, weights and values.
    vesc, vk, vend, vrad, npan : array
        Each row's escape speed, breaks and panel count.
    r_breaks : array-like
        Where the layout changes.
    layout : array-like, optional
        Each row's layout code as it was built; else decided from its breaks.
    vedge, vc : array-like, optional
        Each row's V_edge and V_c, NaN where absent.
    r_edge : float
        The parent's end: R_t, or an infinite table's own edge.
    uncert, uncert_why : optional
        The intervals the certificate left uncertified, (n, 2), and why.

    Raises
    ------
    ResultError
        Where rows of one stretch do not share a layout.
    """

    def __init__(
        self,
        lattice: Lattice,
        r_2d: ArrayLike,
        off: ArrayLike,
        v: ArrayLike,
        w: ArrayLike,
        d: ArrayLike,
        vesc: ArrayLike,
        vk: ArrayLike,
        vend: ArrayLike,
        vrad: ArrayLike,
        npan: ArrayLike,
        r_breaks: ArrayLike = (),
        layout: ArrayLike | None = None,
        vedge: ArrayLike | None = None,
        vc: ArrayLike | None = None,
        r_edge: float = float("inf"),
        uncert: ArrayLike | None = None,
        uncert_why: Sequence[str] | None = None,
    ) -> None:
        self.lattice = lattice
        self.R_2D, self.off, self.V, self.w, self.D = (
            np.asarray(a) for a in (r_2d, off, v, w, d)
        )
        self.vesc, self.Vk, self.Vend, self.Vrad, self.npan = (
            np.asarray(a) for a in (vesc, vk, vend, vrad, npan)
        )
        self.Vedge = (
            np.full(len(self.R_2D), np.nan)
            if vedge is None
            else np.asarray(vedge, dtype=np.float64)
        )
        self.Vc = (
            np.full(len(self.R_2D), np.nan)
            if vc is None
            else np.asarray(vc, dtype=np.float64)
        )
        self._layout = (
            np.asarray(layout, dtype=np.int64)
            if layout is not None
            else np.array(
                [
                    int(
                        lattice.layout_code(
                            self.Vk[i],
                            self.Vend[i],
                            self.Vrad[i],
                            self.Vedge[i],
                            self.Vc[i],
                        )
                    )
                    for i in range(len(self.R_2D))
                ],
                dtype=np.int64,
            )
        )
        self.R_max = float(self.R_2D[-1])
        self.order = 4
        self.R_breaks = np.asarray(r_breaks, dtype=np.float64)
        self.stretch = np.searchsorted(self.R_breaks, self.R_2D)
        self.R_edge = float(r_edge)
        self.uncert = (
            np.zeros((0, 2))
            if uncert is None
            else np.asarray(uncert, dtype=np.float64).reshape(-1, 2)
        )
        self.uncert_why = (
            [""] * len(self.uncert)
            if uncert_why is None or len(uncert_why) != len(self.uncert)
            else [str(s) for s in uncert_why]
        )
        nst = len(self.R_breaks) + 1
        self._st_lo, self._st_hi = np.zeros(nst), np.zeros(nst)
        self._st_lt, self._st_ht = np.zeros(nst, dtype=bool), np.zeros(nst, dtype=bool)
        for st in range(nst):
            sel = np.flatnonzero(self.stretch == st)
            if st > 0:
                self._st_lo[st], self._st_lt[st] = np.log(self.R_breaks[st - 1]), True
            else:
                self._st_lo[st] = np.log(self.R_2D[sel[0]]) if sel.size else -np.inf
            if st < len(self.R_breaks):
                self._st_hi[st], self._st_ht[st] = np.log(self.R_breaks[st]), True
            elif np.isfinite(self.R_edge):
                self._st_hi[st], self._st_ht[st] = np.log(self.R_edge), True
            else:
                self._st_hi[st] = np.log(self.R_2D[sel[-1]]) if sel.size else np.inf
        self.v = self._coord(self.R_2D, self.stretch)
        self._width = np.zeros((len(self.R_2D), len(BREAK_LABELS) + 1))
        for i in range(len(self.R_2D)):
            wd = np.diff(self.row_breaks(i))
            self._width[i, : len(wd)] = wd
        xg, _ = gl01(lattice.rules.vp_nodes)
        self._xg = xg
        self._bw = np.array(
            [1.0 / np.prod(xg[j] - np.delete(xg, j)) for j in range(len(xg))]
        )
        # the stencil's own record: queries, those read at a lower order, the
        # least population of a stretch read
        self.n_query = self.n_short = 0
        self.min_pop = len(self.R_2D)
        self._prep_key: Any = None
        self._prep: Any = None
        for st in np.unique(self.stretch):
            sel = np.flatnonzero(self.stretch == st)
            if len(sel) > 1 and not np.all(self._layout[sel] == self._layout[sel[0]]):
                bad = sel[np.flatnonzero(self._layout[sel] != self._layout[sel[0]])[0]]
                raise ResultError(
                    f"[PANEL TABLE REFUSAL] Rows of one stretch do not share a panel "
                    f"layout: stretch "
                    f"{int(st)} holds row {int(sel[0])} "
                    f"(R_2D = {self.R_2D[sel[0]]:.6e}, "
                    f"layout code {int(self._layout[sel[0]])}) and row {int(bad)} "
                    f"(R_2D = {self.R_2D[bad]:.6e}, code {int(self._layout[bad])}) "
                    f"[3-56]."
                )

    def _coord(self, r: ArrayLike, st: ArrayLike) -> Array:
        """The stencil's variable of radii in stretches: theta with the square
        root at each true end, linear at a false one."""
        r, st = np.asarray(r, dtype=np.float64), np.asarray(st)
        lo, hi = self._st_lo[st], self._st_hi[st]
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.where(
                hi > lo, (np.log(r) - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0
            )
        xi = np.clip(xi, 0.0, 1.0)
        lt, ht = self._st_lt[st], self._st_ht[st]
        return np.where(
            lt & ht,
            np.arccos(1.0 - 2.0 * xi),
            np.where(lt, np.sqrt(xi), np.where(ht, 1.0 - np.sqrt(1.0 - xi), xi)),
        )

    def _coord_inv(self, v: float, st: int) -> float:
        """The radius at the variable's value v in stretch st."""
        lt, ht = bool(self._st_lt[st]), bool(self._st_ht[st])
        if lt and ht:
            xi = 0.5 * (1.0 - np.cos(v))
        elif lt:
            xi = v * v
        elif ht:
            xi = 1.0 - (1.0 - v) ** 2
        else:
            xi = v
        return float(np.exp(self._st_lo[st] + xi * (self._st_hi[st] - self._st_lo[st])))

    @classmethod
    def from_rows(
        cls,
        lattice: Lattice,
        r_2d: ArrayLike,
        rows: Sequence[tuple[Any, ...]],
        r_breaks: ArrayLike = (),
        r_edge: float = float("inf"),
    ) -> "PanelTable":
        """Return the table of rows ``(V, w, D, vesc, V_k, V_end, V_rad,
        npan[, code[, V_edge[, V_c]]])`` on V >= 0, stored mirrored."""
        n = np.array([2 * len(r[0]) for r in rows])
        off = np.concatenate([[0], np.cumsum(n)])
        v = np.concatenate([np.concatenate([-r[0][::-1], r[0]]) for r in rows])
        w = np.concatenate([np.concatenate([r[1][::-1], r[1]]) for r in rows])
        d = np.concatenate([np.concatenate([r[2][::-1], r[2]]) for r in rows])
        lay = np.array([r[8] for r in rows]) if all(len(r) > 8 for r in rows) else None
        ved = (
            np.array([r[9] for r in rows], dtype=np.float64)
            if all(len(r) > 9 for r in rows)
            else None
        )
        vc = (
            np.array([r[10] for r in rows], dtype=np.float64)
            if all(len(r) > 10 for r in rows)
            else None
        )
        cols = [np.array([r[k] for r in rows]) for k in (3, 4, 5, 6, 7)]
        return cls(
            lattice,
            r_2d,
            off,
            v,
            w,
            d,
            cols[0],
            cols[1],
            cols[2],
            cols[3],
            cols[4],
            r_breaks=r_breaks,
            layout=lay,
            vedge=ved,
            vc=vc,
            r_edge=r_edge,
        )

    def row(self, i: int) -> tuple[Array, Array, Array]:
        s = slice(int(self.off[i]), int(self.off[i + 1]))
        return self.V[s], self.w[s], self.D[s]

    def row_breaks(self, i: int) -> list[float]:
        """Return [0, interior breaks, V_end] of row i.

        Raises
        ------
        ResultError
            Where its stored layout and panel count disagree.
        """
        val = {
            "k": float(self.Vk[i]),
            "r": float(self.Vrad[i]),
            "e": float(self.Vedge[i]),
            "c": float(self.Vc[i]),
        }
        br = (
            [0.0]
            + [val[s] for s in LAYOUT_SEQ[int(self._layout[i])]]
            + [float(self.Vend[i])]
        )
        if len(br) - 1 != int(self.npan[i]):
            raise ResultError(
                f"[PANEL TABLE REFUSAL] Row {i}: its break list gives {len(br) - 1} "
                f"panels but the table "
                f"stores n_panels = {int(self.npan[i])}."
            )
        return br

    def _row_at(self, i: int, p: Array, xi: Array) -> Array:
        """D of row i at panel index p and position xi of that panel: the
        sub-panel's own polynomial through its Gauss nodes, barycentric. A
        panel the row does not have is the zero-width panel at V_end."""
        out = np.zeros(len(xi))
        m = self.lattice.rules.vp_nodes
        half0 = int(self.off[i]) + (int(self.off[i + 1]) - int(self.off[i])) // 2
        units = self.lattice.panel_units(int(self._layout[i]))
        nsub = np.concatenate([[0], np.cumsum([len(u) - 1 for u in units])])
        if int(self.off[i + 1]) - int(self.off[i]) != 2 * m * int(nsub[-1]):
            raise ResultError(
                f"[PANEL TABLE REFUSAL] Row {i}: "
                f"{int(self.off[i + 1]) - int(self.off[i])} nodes where its "
                f"layout's {int(nsub[-1])} sub-panels at {m} nodes give "
                f"{2 * m * int(nsub[-1])}."
            )
        for pp in np.unique(p):
            if pp >= int(self.npan[i]):
                continue
            sel = np.flatnonzero(p == pp)
            e = units[pp]
            k = np.clip(np.searchsorted(e, xi[sel], side="right") - 1, 0, len(e) - 2)
            t = (xi[sel] - e[k]) / (e[k + 1] - e[k])
            base = half0 + m * int(nsub[pp])
            dd = t[:, None] - self._xg[None, :]
            hit = dd == 0.0
            dn = self.D[(base + m * k)[:, None] + np.arange(m)[None, :]]
            # a position on a node gives inf / inf there, replaced just below
            with np.errstate(divide="ignore", invalid="ignore"):
                c = self._bw[None, :] / dd
                val = np.sum(c * dn, axis=1) / np.sum(c, axis=1)
            anyhit = hit.any(axis=1)
            val[anyhit] = dn[anyhit][hit[anyhit]]
            out[sel] = val
        return out

    def _prepare(
        self,
        route: EllipseRoute,
        r2: Array,
        curves: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Return the stencil's setup at radii r2: each radius's breaks and
        layout, checked against its stretch, its rows and Lagrange weights at
        `k` rows (`order` where None).

        Raises
        ------
        ResultError
            Where a stretch holds no row, a radius's layout is not its
            stretch's, or a radius lies where the certificate left the table
            uncertified.
        """
        lat = self.lattice
        if curves is None:
            vk, vend, vr = route.breaks_many(r2)
            vk = route.vk_break(r2, vk)
            vr = route.vrad_break(r2, vr)
            ve = route.v_edge_many(r2)
            vc = route.v_c_many(r2)
        else:
            vk, vend, vr, ve, vc = (
                np.asarray(curves[x], dtype=np.float64)
                for x in ("V_k", "V_end", "V_rad", "V_edge", "V_c")
            )
        vk, vr, ve, vc = lat.resolved_breaks(vk, vend, vr, ve, vc)
        a_, b_, c_, d_ = lat.interior_breaks(vk, vend, vr, ve, vc)
        cand = np.stack(
            [
                np.where(a_, vk, np.inf),
                np.where(b_, vr, np.inf),
                np.where(c_, ve, np.inf),
                np.where(d_, vc, np.inf),
            ],
            axis=1,
        )
        cand.sort(axis=1)
        brk = np.concatenate(
            [np.zeros((len(r2), 1)), cand, np.full((len(r2), 1), np.inf)], axis=1
        )
        npan = 1 + np.isfinite(cand).sum(axis=1)
        brk[np.arange(len(r2)), npan] = vend
        st = np.asarray(np.searchsorted(self.R_breaks, r2))
        first = np.asarray(np.searchsorted(self.stretch, st, side="left"))
        last = np.asarray(np.searchsorted(self.stretch, st, side="right")) - 1
        if not np.all(last >= first):
            j = int(np.flatnonzero(last < first)[0])
            raise ResultError(
                f"The stretch between R-breaks that contains R_2D = {float(r2[j]):.6e} "
                f"holds no stored row (stretch {int(st[j])}) [PANEL TABLE REFUSAL]."
            )
        tcode = lat.layout_code(vk, vend, vr, ve, vc)
        scode = self._layout[np.clip(first, 0, len(self._layout) - 1)]
        mism = np.flatnonzero(tcode != scode)
        if mism.size:
            j = int(mism[0])
            raise ResultError(
                f"The target R_2D = {float(r2[j]):.6e} has panel layout code "
                f"{int(tcode[j])} while the stretch it falls in (index {int(st[j])}, "
                f"rows {int(first[j])} to {int(last[j])}) was stored with code "
                f"{int(scode[j])} [PANEL TABLE REFUSAL] [3-56]."
            )
        for (ua, ub), why in zip(self.uncert, self.uncert_why, strict=True):
            inside = np.flatnonzero((r2 > ua) & (r2 < ub))
            if inside.size:
                raise ResultError(
                    f"The target R_2D = {float(r2[inside[0]]):.9e} lies in "
                    f"[{ua:.9e}, {ub:.9e}], which the table's certificate left "
                    f"uncertified: "
                    + (
                        why
                        if why
                        else "an end gap, between a true end of its stretch and the "
                        "nearest row, where the stencil reads beyond its rows"
                    )
                    + " [PANEL TABLE REFUSAL]."
                )
        kk = int(self.order if k is None else k)
        n = np.minimum(kk, last - first + 1)
        self.n_query += len(r2)
        self.n_short += int(np.sum(n < kk))
        self.min_pop = min(self.min_pop, int((last - first + 1).min()))
        i1 = np.searchsorted(self.R_2D, r2, side="right")
        start = np.clip(i1 - n // 2, first, last - n + 1)
        rows = np.where(
            np.arange(kk)[None, :] < n[:, None],
            start[:, None] + np.arange(kk)[None, :],
            -1,
        )
        x, xm = self.v[np.maximum(rows, 0)], self._coord(r2, st)
        lag = (rows >= 0).astype(np.float64)
        live = rows >= 0
        for b in range(kk):
            with np.errstate(divide="ignore", invalid="ignore"):
                f = (xm - x[:, b])[:, None] / (x - x[:, b][:, None])
            ok = live & live[:, b][:, None]
            ok[:, b] = False
            lag *= np.where(ok, f, 1.0)
        return (r2, brk, npan, vend, rows, lag)

    def predict(
        self,
        route: EllipseRoute,
        r2: float,
        k: int,
        curves: dict[str, Any] | None = None,
    ) -> tuple[Array, Array, Array]:
        """Return the row the stencil at k rows gives at radius r2 on the
        stretch's own lattice: (V nodes, weights, D) on V >= 0."""
        rr, brk, npan, vend, rows, lag = self._prepare(
            route, np.array([float(r2)]), curves=curves, k=k
        )
        m = self.lattice.rules.vp_nodes
        i0 = int(rows[0][rows[0] >= 0][0])
        units = self.lattice.panel_units(int(self._layout[i0]))
        p = np.concatenate([np.full((len(u) - 1) * m, q) for q, u in enumerate(units)])
        br = brk[0, : int(npan[0]) + 1]
        wm = np.diff(br)[p]
        v0, w0, _ = self.row(i0)
        h = len(v0) // 2
        br0 = np.asarray(self.row_breaks(i0))
        xi = (v0[h:] - br0[p]) / (br0[p + 1] - br0[p])
        with np.errstate(divide="ignore", invalid="ignore"):
            vm = br[p] + xi * wm
            wmn = w0[h:] * np.where(
                self._width[i0, p] > 0.0,
                wm / np.where(self._width[i0, p] > 0.0, self._width[i0, p], 1.0),
                0.0,
            )
        pred = np.zeros(len(p))
        for a in range(rows.shape[1]):
            i = int(rows[0, a])
            if i < 0:
                continue
            di = self.row(i)[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                pred += (
                    lag[0, a]
                    * np.where(
                        wm > 0.0, self._width[i, p] / np.where(wm > 0.0, wm, 1.0), 0.0
                    )
                    * di[len(di) // 2 :]
                )
        return vm, wmn, pred

    def interval_mid(self, i: int) -> float | None:
        """Return the midpoint, in the stencil's variable, of the interval
        between rows i and i + 1 of one stretch; None where they are not of
        one stretch or it holds fewer than four rows."""
        st = int(self.stretch[i])
        if st != int(self.stretch[i + 1]) or int(np.sum(self.stretch == st)) < 4:
            return None
        rm = self._coord_inv(0.5 * (float(self.v[i]) + float(self.v[i + 1])), st)
        if not (self.R_2D[i] < rm < self.R_2D[i + 1]):
            rm = float(np.sqrt(self.R_2D[i] * self.R_2D[i + 1]))
        return rm

    @staticmethod
    def mid_curves(route: EllipseRoute, rm: ArrayLike) -> dict[str, Array]:
        """Return the breaks at radii rm, V_rad and V_k through their bounds,
        one scan for all of them."""
        r1 = np.asarray(rm, dtype=np.float64)
        vk, ve_, vr = route.breaks_many(r1)
        return {
            "V_k": route.vk_break(r1, vk),
            "V_end": ve_,
            "V_rad": route.vrad_break(r1, vr),
            "V_edge": route.v_edge_many(r1),
            "V_c": route.v_c_many(r1),
        }

    def _estimate(
        self, route: EllipseRoute, rm: float, k: int, cv: dict[str, Any] | None
    ) -> tuple[float, float] | None:
        try:
            vm, wm, dk = self.predict(route, rm, k, curves=cv)
            _, _, dl = self.predict(route, rm, k - 2, curves=cv)
        except ResultError as e:
            if "[PANEL TABLE REFUSAL]" not in str(e):
                raise
            return None
        m2k, m2l = float(np.dot(wm, vm * vm * dk)), float(np.dot(wm, vm * vm * dl))
        pk = float(np.max(np.abs(dk))) if dk.size else 0.0
        est_m2 = abs(m2k - m2l) / abs(m2k) if m2k != 0.0 else 0.0
        est_pw = float(np.max(np.abs(dk - dl))) / pk if pk > 0.0 else 0.0
        return est_m2, est_pw

    def interval_estimate(
        self, route: EllipseRoute, i: int, k: int, cv: dict[str, Any] | None = None
    ) -> tuple[float, float, float] | None:
        """Return (R_mid, the k against k - 2 difference in the second moment,
        the same pointwise over the largest D) at the midpoint of the interval
        between rows i and i + 1, or None where it cannot be read."""
        rm = self.interval_mid(i)
        if rm is None:
            return None
        st = int(self.stretch[i])
        k = min(int(k), int(np.sum(self.stretch == st)))
        if cv is None:
            cv = {x: y[0] for x, y in self.mid_curves(route, np.array([rm])).items()}
        cv = {x: np.atleast_1d(y) for x, y in cv.items()}
        est = self._estimate(route, rm, k, cv)
        return None if est is None else (rm, est[0], est[1])

    def gap_estimate(
        self, route: EllipseRoute, st: int, top: bool, k: int
    ) -> tuple[float, float, float, int] | None:
        """Return the stencil's estimate in the gap between stretch st's true
        end (the upper if `top`) and its nearest row: (R_mid, second moment,
        pointwise, the nearest row), or None where the end is not true, the
        stretch holds fewer than four rows or the middle cannot be read."""
        if not (self._st_ht[st] if top else self._st_lt[st]):
            return None
        sel = np.flatnonzero(self.stretch == st)
        if sel.size < 4:
            return None
        i = int(sel[-1] if top else sel[0])
        re_ = float(np.exp(self._st_hi[st] if top else self._st_lo[st]))
        ve = float(self._coord(np.array([re_]), np.array([st]))[0])
        rm = self._coord_inv(0.5 * (ve + float(self.v[i])), st)
        if not min(re_, float(self.R_2D[i])) < rm < max(re_, float(self.R_2D[i])):
            return None
        k = min(int(k), int(sel.size))
        est = self._estimate(route, rm, k, None)
        return None if est is None else (rm, est[0], est[1], i)

    def at_pairs(
        self,
        route: EllipseRoute,
        r2d_arr: ArrayLike,
        vq: ArrayLike,
        curves: dict[str, Any] | None = None,
    ) -> Array:
        """Return D at the pairs (r2d_arr[j], vq[j]): (panel width x D) at the
        pair's (panel, xi), blended between rows by the stencil, over the
        panel's width at the radius."""
        r2, brk, npan, vend, rows, lag = self._prepare(
            route, np.asarray(r2d_arr, dtype=np.float64), curves
        )
        a = np.abs(np.asarray(vq, dtype=np.float64))
        out = np.zeros(len(r2))
        live = np.flatnonzero(a < vend)
        if live.size == 0:
            return out
        p = np.minimum((brk[live, 1:] <= a[live, None]).sum(axis=1), npan[live] - 1)
        lo, hi = brk[live, p], brk[live, p + 1]
        xi = np.clip((a[live] - lo) / (hi - lo), 0.0, 1.0)
        val = np.zeros(live.size)
        for c in range(rows.shape[1]):
            for i in np.unique(rows[live, c]):
                if i < 0:
                    continue
                s = np.flatnonzero(rows[live, c] == i)
                val[s] += (
                    lag[live[s], c]
                    * self._width[int(i), p[s]]
                    / (hi[s] - lo[s])
                    * self._row_at(int(i), p[s], xi[s])
                )
        out[live] = val
        return out

    def at(
        self,
        route: EllipseRoute,
        r2d_arr: ArrayLike,
        v: float,
        curves: dict[str, Any] | None = None,
    ) -> Array:
        """Return D at radii and one speed V, as `at_pairs` does for a V of
        its own at each radius; the setup is kept for the next V at the same
        radii."""
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        key = (len(r2), float(r2[0]), float(r2[-1]), float(r2.sum()))
        if self._prep_key != key:
            self._prep_key, self._prep = key, self._prepare(route, r2, curves)
        r2, brk, npan, vend, rows, lag = self._prep
        a = abs(float(v))
        out = np.zeros(len(r2))
        live = np.flatnonzero(a < vend)
        if live.size == 0:
            return out
        p = np.minimum((brk[live, 1:] <= a).sum(axis=1), npan[live] - 1)
        lo, hi = brk[live, p], brk[live, p + 1]
        xi = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        val = np.zeros(live.size)
        for c in range(rows.shape[1]):
            for i in np.unique(rows[live, c]):
                if i < 0:
                    continue
                s = np.flatnonzero(rows[live, c] == i)
                val[s] += (
                    lag[live[s], c]
                    * self._width[int(i), p[s]]
                    / (hi[s] - lo[s])
                    * self._row_at(int(i), p[s], xi[s])
                )
        out[live] = val
        return out
