r"""The table of rows of the line-of-sight velocity distribution, read at any
projected radius between its rows.

`pianistpy.losvd`, `pianistpy.aperture_losvd`, `pianistpy.LosTable` and
`ProjectedSampler` read the distribution :math:`D(R_{\rm 2D}, V)` at a
projected radius :math:`R_{\rm 2D}` and a line-of-sight velocity :math:`V`
through `PanelTable`. Each row holds D on its own velocity panels at one
radius. The radii where the panel layout changes split the rows into
stretches, and the rows of one stretch share one layout. Between rows, D at
the panel position :math:`\xi` (in units of the panel's width) is the
Lagrange interpolation over :math:`k` neighboring rows of the stretch,

.. math::

    D(R_{\rm 2D}, V) = \sum_{i} \ell_i(x)\, \frac{w_{p,i}}{w_p(R_{\rm 2D})}\,
    D_i(p, \xi),

with :math:`\ell_i` the Lagrange basis polynomial of row :math:`i` in the
interpolation variable :math:`x`, :math:`p` the panel that holds
:math:`V`, :math:`w_{p,i}` the width of panel :math:`p` in row :math:`i`,
:math:`w_p(R_{\rm 2D})` the width of panel :math:`p` at :math:`R_{\rm 2D}`,
and :math:`D_i(p, \xi)` row :math:`i`'s value at position :math:`\xi` of
panel :math:`p`. The width ratio keeps the integral of D over each panel
continuous between rows. Let :math:`\eta` be the position of
:math:`\ln R_{\rm 2D}` across the stretch's range of :math:`\ln R`, from 0 at
the stretch's lower end to 1 at the stretch's upper end. The interpolation
variable is :math:`x = \arccos(1 - 2\eta)` where both ends of the stretch are
true ends, :math:`x = \eta^{1/2}` where only the lower end is a true end,
:math:`x = 1 - (1 - \eta)^{1/2}` where only the upper end is a true end, and
:math:`x = \eta` where neither end is a true end. A true end is a radius where
the layout changes, or the table's finite edge. Notes Section T:6.5 treats
the table of rows."""

__all__ = ["PanelTable"]

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..exceptions import InputError, ResultError
from ..typing import Array, ArrayLike
from ._ellipse import EllipseRoute
from ._ellipse_rules import PANEL_REFUSAL, gl01
from ._layout import BREAK_LABELS, LAYOUT_SEQ, Lattice
from ._projection import END_TOLERANCE


def refuse_outside_rows(r2: Array, first: float, last: float) -> None:
    """Raise InputError where a projected radius of `r2` lies outside the
    table's rows.

    Raises
    ------
    InputError
        If a radius is below `first` times (1 - END_TOLERANCE). The message
        gives the setting ``losvd.innermost_read_radius``, which sets the
        first row.
    InputError
        If a radius is above `last` times (1 + END_TOLERANCE). The message
        gives the setting ``losvd.outer_mass_share``, which sets the last
        row.
    """
    r2 = np.asarray(r2, dtype=np.float64)
    inside = r2 < first * (1.0 - END_TOLERANCE)
    if np.any(inside):
        raise InputError(
            f"The projected radius R_2D = {float(np.min(r2[inside])):.6e} lies "
            f"inside the table's innermost tabulated radius {first:.6e}, which the "
            f"setting losvd.innermost_read_radius sets; the table is not "
            f"extrapolated."
        )
    beyond = r2 > last * (1.0 + END_TOLERANCE)
    if np.any(beyond):
        raise InputError(
            f"The projected radius R_2D = {float(np.max(r2[beyond])):.6e} lies "
            f"beyond the table's outermost tabulated radius {last:.6e}, which the "
            f"setting losvd.outer_mass_share sets; the table is not extrapolated."
        )


class PanelTable:
    r"""The table of rows of D, each row on its own velocity panels, read at
    any projected radius between the rows.

    The arrays `v`, `w` and `d` hold all rows one after another. Row i
    occupies the index range ``[off[i], off[i + 1])``. Each row holds the
    velocities ``-V`` in reverse order, then the velocities ``V >= 0`` in the order
    (panel, sub-panel, node), with the weights and values of D to match. The
    radii `r_breaks` split the rows into stretches. The rows of one stretch
    share one layout code. The table reads D between rows by Lagrange
    interpolation over neighboring rows of one stretch (`at_pairs`).

    Parameters
    ----------
    lattice : Lattice
        The sub-panel grading of the rows' panels.
    r_2d : array, shape (n_rows,)
        The projected radius of each row, increasing.
    off : array, shape (n_rows + 1,)
        The index where each row starts in `v`, `w` and `d`, and the total
        length last.
    v : array
        The velocity nodes of all rows.
    w : array
        The quadrature weights of all rows.
    d : array
        The values of D of all rows.
    vesc : array, shape (n_rows,)
        The escape velocity of each row.
    vk : array, shape (n_rows,)
        V_k of each row.
    vend : array, shape (n_rows,)
        V_end of each row, the velocity above which D is zero.
    vrad : array, shape (n_rows,)
        V_rad of each row.
    npan : array, shape (n_rows,)
        The number of velocity panels of each row.
    r_breaks : array-like, optional
        The projected radii where the layout changes. Default: ``()``, which
        means one stretch.
    layout : array-like, optional
        The layout code of each row. Default: None, which means each code is
        computed from the row's break velocities (`Lattice.layout_code`).
    vedge : array-like, optional
        V_edge of each row, NaN where V_edge is no break. Default: None, which
        means NaN for every row.
    vc : array-like, optional
        V_c of each row, NaN where V_c is no break. Default: None, which means
        NaN for every row.
    r_edge : float, optional
        The radius where the line of sight ends, :math:`R_{\rm t}`. A finite
        `r_edge` is a true end of the last stretch. Default: infinity, which
        means the last stretch ends at its last row.
    uncert : array-like, optional
        The intervals of radius, shape (n, 2), that the table's check left
        above its tolerance. Default: None, which means no interval.
    uncert_why : sequence of str, optional
        The reason for each interval of `uncert`. Default: None, which means
        an empty reason for each interval. An empty reason is also used where
        the length differs from the number of intervals.

    Raises
    ------
    ResultError
        If two rows of one stretch have different layout codes.
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
        # The last radius the table reads: r_edge where r_edge is finite, else
        # the last row's radius.
        self.R_end = self.R_edge if np.isfinite(self.R_edge) else self.R_max
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
        # The table's record of its reads: the number of radii read
        # (n_query), the number read with fewer rows than the order (n_short),
        # and the smallest number of rows in a stretch read (min_pop).
        self.n_query = self.n_short = 0
        self.min_pop = len(self.R_2D)
        self._prep_key: Any = None
        self._prep: Any = None
        for st in np.unique(self.stretch):
            sel = np.flatnonzero(self.stretch == st)
            if len(sel) > 1 and not np.all(self._layout[sel] == self._layout[sel[0]]):
                bad = sel[np.flatnonzero(self._layout[sel] != self._layout[sel[0]])[0]]
                raise ResultError(
                    f"{PANEL_REFUSAL} The tabulated radii of one stretch do not "
                    f"share a panel layout. The stretch {int(st)} holds the radius "
                    f"index {int(sel[0])} (R_2D = {self.R_2D[sel[0]]:.6e}, layout "
                    f"code {int(self._layout[sel[0]])}) and the radius index "
                    f"{int(bad)} (R_2D = {self.R_2D[bad]:.6e}, layout code "
                    f"{int(self._layout[bad])})."
                )

    def _coord(self, r: ArrayLike, st: ArrayLike) -> Array:
        r"""Return the interpolation variable x at radii `r` in stretches `st`.

        With :math:`\eta` the position of ln r across the stretch's range of
        ln R, clipped to [0, 1], x is :math:`\arccos(1 - 2\eta)` where both
        ends of the stretch are true ends, :math:`\eta^{1/2}` where only the
        lower end is a true end, :math:`1 - (1 - \eta)^{1/2}` where only the
        upper end is a true end, and :math:`\eta` where neither end is a true
        end.
        """
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
        """Return the radius at which the interpolation variable of stretch
        `st` takes the value `v` (the inverse of `_coord`)."""
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
        """Return the table built from `rows`, one entry per radius of `r_2d`.

        Each entry is ``(V, w, D, vesc, V_k, V_end, V_rad, npan)`` on
        ``V >= 0``, optionally followed by the layout code, then V_edge, then
        V_c. The table stores each row's nodes, weights and values mirrored to
        ``V < 0``. An optional field is used only where every entry has it.
        """
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
        """Return the velocity nodes, weights and values of D of row `i`, with
        both signs of the velocity."""
        s = slice(int(self.off[i]), int(self.off[i + 1]))
        return self.V[s], self.w[s], self.D[s]

    def row_breaks(self, i: int) -> list[float]:
        """Return the panel ends of row `i`: 0, the row's interior breaks in
        the order of the row's layout code, and V_end.

        Raises
        ------
        ResultError
            If the number of panels that row `i`'s layout code gives differs
            from row `i`'s stored panel count.
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
                f"{PANEL_REFUSAL} At the radius index {i}, the velocity breaks give "
                f"{len(br) - 1} panels, and the table stores n_panels = "
                f"{int(self.npan[i])}."
            )
        return br

    def _row_at(self, i: int, p: Array, xi: Array) -> Array:
        """Return D of row `i` at panel indices `p` and positions `xi`, each
        position in units of the panel's width.

        The value is the polynomial through the sub-panel's Gauss-Legendre
        node values, evaluated in barycentric form. A panel index at or
        beyond row `i`'s panel count gives 0.

        Raises
        ------
        ResultError
            If row `i`'s node count differs from the count its layout gives.
        """
        out = np.zeros(len(xi))
        m = self.lattice.rules.vp_nodes
        half0 = int(self.off[i]) + (int(self.off[i + 1]) - int(self.off[i])) // 2
        units = self.lattice.panel_units(int(self._layout[i]))
        nsub = np.concatenate([[0], np.cumsum([len(u) - 1 for u in units])])
        if int(self.off[i + 1]) - int(self.off[i]) != 2 * m * int(nsub[-1]):
            raise ResultError(
                f"{PANEL_REFUSAL} At the radius index {i}, the table stores "
                f"{int(self.off[i + 1]) - int(self.off[i])} velocity nodes, and the "
                f"panel layout's {int(nsub[-1])} sub-panels at {m} nodes give "
                f"{2 * m * int(nsub[-1])} velocity nodes."
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
            # A position exactly on a node gives inf / inf in the barycentric
            # form. The node's own value replaces that result below.
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
        """Return the interpolation setup at the projected radii `r2`.

        The setup is the tuple (radii, panel ends, panel counts, V_end, row
        indices, Lagrange weights), with one entry per radius of `r2`. The
        break velocities at each radius come from `curves` where given.
        Otherwise the function computes the break velocities with `route`
        (`_breaks`) and applies the route's kept ranges of V_k and V_rad. The
        function then drops each break that double precision does not resolve
        (`Lattice.resolved_breaks`). The interpolation uses `k` rows of the
        radius's stretch, or ``self.order`` rows where `k` is None, or every
        row of the stretch where the stretch holds fewer rows. The function
        sets up each distinct radius once, so a radius repeated in `r2` adds
        no cost.

        Raises
        ------
        InputError
            If a radius lies below the first row's radius or above the last
            radius the table reads (`refuse_outside_rows`).
        ResultError
            If the stretch of a radius holds no row.
        ResultError
            If the layout code at a radius differs from the layout code of the
            radius's stretch.
        ResultError
            If a radius lies inside an interval that the table's check left
            above its tolerance.
        """
        refuse_outside_rows(r2, float(self.R_2D[0]), self.R_end)
        r_all = np.asarray(r2, dtype=np.float64)
        r2, first_of, inv = np.unique(r_all, return_index=True, return_inverse=True)
        lat = self.lattice
        if curves is None:
            vk, vend, vr = self._breaks(route, r2)
            vk = route.vk_break(r2, vk)
            vr = route.vrad_break(r2, vr)
            ve = route.v_edge_many(r2)
            vc = route.v_c_many(r2)
        else:
            vk, vend, vr, ve, vc = (
                np.asarray(curves[x], dtype=np.float64)[first_of]
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
                f"The stretch {int(st[j])} between R-breaks that contains R_2D = "
                f"{float(r2[j]):.6e} holds no tabulated radius {PANEL_REFUSAL}."
            )
        tcode = lat.layout_code(vk, vend, vr, ve, vc)
        scode = self._layout[np.clip(first, 0, len(self._layout) - 1)]
        mism = np.flatnonzero(tcode != scode)
        if mism.size:
            j = int(mism[0])
            raise ResultError(
                f"The projected radius R_2D = {float(r2[j]):.6e} has the panel layout "
                f"code {int(tcode[j])}, and the stretch that contains R_2D = "
                f"{float(r2[j]):.6e} (index {int(st[j])}, radius indices "
                f"{int(first[j])} to {int(last[j])}) was stored with the layout code "
                f"{int(scode[j])} {PANEL_REFUSAL}."
            )
        for (ua, ub), why in zip(self.uncert, self.uncert_why, strict=True):
            inside = np.flatnonzero((r2 > ua) & (r2 < ub))
            if inside.size:
                raise ResultError(
                    f"The projected radius R_2D = {float(r2[inside[0]]):.9e} lies in "
                    f"[{ua:.9e}, {ub:.9e}], a range where the table's error "
                    f"estimate is above the tolerance; the range is "
                    + (
                        why
                        if why
                        else "an end gap of a stretch of tabulated radii, where the "
                        "interpolation would read beyond the stretch's tabulated radii"
                    )
                    + f" {PANEL_REFUSAL}."
                )
        kk = int(self.order if k is None else k)
        n = np.minimum(kk, last - first + 1)
        self.n_query += len(r_all)
        self.n_short += int(np.sum((n < kk)[inv]))
        self.min_pop = min(self.min_pop, int(np.min(last - first + 1)))
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
        return (r_all, brk[inv], npan[inv], vend[inv], rows[inv], lag[inv])

    @staticmethod
    def _breaks(route: EllipseRoute, r2: Array) -> tuple[Array, Array, Array]:
        """Return `EllipseRoute.breaks_many` at the radii `r2`, computed in
        blocks of radii.

        Each block holds ``rules.chunk // rules.brk_scan`` radii, so each
        temporary array of shape (radii, scan points) holds at most
        ``rules.chunk`` elements, the setting ``losvd.n_block_elements``.
        """
        step = max(1, int(route.rules.chunk) // int(route.rules.brk_scan))
        parts = [route.breaks_many(r2[i : i + step]) for i in range(0, len(r2), step)]
        if not parts:
            return route.breaks_many(r2)
        vk, vend, vr = (np.concatenate([p[j] for p in parts]) for j in range(3))
        return vk, vend, vr

    def predict(
        self,
        route: EllipseRoute,
        r2: float,
        k: int,
        curves: dict[str, Any] | None = None,
    ) -> tuple[Array, Array, Array]:
        """Return the row that interpolation over `k` rows gives at the
        projected radius `r2`, as (velocity nodes, weights, D) on ``V >= 0``.

        The returned nodes lie at the same panel positions as the nodes of
        the first row used, with the panel ends at `r2`. Each weight is the
        first row's weight times the ratio of the panel's width at `r2` to the
        panel's width in the first row. Each value of D is the Lagrange sum
        over the `k` rows of each row's node value times the ratio of the
        panel's width in that row to the panel's width at `r2`.
        """
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
        """Return the radius at the midpoint, in the interpolation variable,
        between rows `i` and `i + 1`.

        Where that radius does not lie strictly between the two rows' radii,
        the function returns the geometric mean of the two rows' radii. The
        function returns None where rows `i` and `i + 1` lie in different
        stretches, or where their stretch holds fewer than four rows.
        """
        st = int(self.stretch[i])
        if st != int(self.stretch[i + 1]) or int(np.sum(self.stretch == st)) < 4:
            return None
        rm = self._coord_inv(0.5 * (float(self.v[i]) + float(self.v[i + 1])), st)
        if not (self.R_2D[i] < rm < self.R_2D[i + 1]):
            rm = float(np.sqrt(self.R_2D[i] * self.R_2D[i + 1]))
        return rm

    @staticmethod
    def mid_curves(route: EllipseRoute, rm: ArrayLike) -> dict[str, Array]:
        """Return the break velocities at the radii `rm`, keyed "V_k",
        "V_end", "V_rad", "V_edge" and "V_c".

        V_k and V_rad are NaN outside the route's kept ranges. One call of
        `EllipseRoute.breaks_many` serves every radius.
        """
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
        """Return (second-moment estimate, pointwise estimate) of the
        interpolation error at the radius `rm`, or None.

        The function interpolates the row at `rm` over `k` rows and over
        ``k - 2`` rows (`predict`). The second-moment estimate is the absolute
        difference of the two rows' second moments in V, divided by the
        absolute second moment of the `k`-row interpolation. The pointwise
        estimate is the largest absolute difference of the two rows' values,
        divided by the largest absolute value of the `k`-row interpolation.
        The function returns None where `predict` raises a ResultError that
        carries PANEL_REFUSAL.
        """
        try:
            vm, wm, dk = self.predict(route, rm, k, curves=cv)
            _, _, dl = self.predict(route, rm, k - 2, curves=cv)
        except ResultError as e:
            if PANEL_REFUSAL not in str(e):
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
        """Return (R_mid, second-moment estimate, pointwise estimate) at the
        midpoint R_mid between rows `i` and `i + 1` (`interval_mid`,
        `_estimate`).

        The number of rows `k` is capped at the number of rows of the
        stretch. The break velocities at R_mid come from `cv` where given.
        The function returns None where `interval_mid` or `_estimate` returns
        None.
        """
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
        """Return the error estimate in the end gap of stretch `st`.

        The end gap lies between an end of the stretch and the stretch's row
        nearest to that end. The end is the upper end where `top` is true and
        the lower end otherwise. The result is (R_mid, second-moment estimate,
        pointwise estimate, index of the nearest row), with R_mid the
        midpoint of the gap in the interpolation variable (`_estimate`). The
        function returns None in four cases: the end is not a true end; the
        stretch holds fewer than four rows; R_mid does not lie strictly inside
        the gap; or `_estimate` returns None.
        """
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
        """Return D at each pair ``(r2d_arr[j], vq[j])``.

        For each pair, the function finds the panel p that holds ``|vq[j]|``
        at the radius and the position xi of ``|vq[j]|`` in that panel, in
        units of the panel's width. D is the Lagrange sum over the rows of
        each row's value at (p, xi) times the ratio of panel p's width in that
        row to panel p's width at the radius. D is 0 where ``|vq[j]|`` is at
        or above V_end at the radius.

        Raises
        ------
        InputError
            If a radius lies outside the rows (`_prepare`).
        ResultError
            If the table cannot read a radius (`_prepare`).
        """
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
        """Return D at each radius of `r2d_arr` and the one velocity `v`, as
        `at_pairs` computes D.

        The table keeps the interpolation setup (`_prepare`) of the last
        call. The next call reuses the setup where the radii have the same
        length, first value, last value and sum. The check does not compare
        `curves`.

        Raises
        ------
        InputError
            If a radius lies outside the rows (`_prepare`).
        ResultError
            If the table cannot read a radius (`_prepare`).
        """
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
