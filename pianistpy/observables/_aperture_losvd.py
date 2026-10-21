r"""The line-of-sight velocity distribution averaged over an aperture.

`aperture_losvd`, and `sigma_los_aperture` given a `LosTable`, use this
module through `stages.aperture_losvd`. The aperture distribution at the
line-of-sight velocity :math:`V \ge 0` is

.. math::

    \bar D(V) = \frac{\int D(R, V)\,\Sigma(R)\,{\rm arc}(R)\,R\,{\rm d}R}
    {\int \Sigma(R)\,{\rm arc}(R)\,R\,{\rm d}R},

with :math:`D(R, V)` the line-of-sight velocity distribution of the table
of rows at projected radius :math:`R`, :math:`\Sigma` the surface density
and :math:`{\rm arc}(R)` the angle of the circle of radius :math:`R` inside
the aperture (`Aperture.arc`). The aperture distribution has its own
velocity panels, with breaks at the extreme values of the rows' break
velocities over the aperture. The panels are bisected until the norm and
the second moment of the aperture distribution agree with the norm and
the second moment from the rows' own velocity rules to a relative
:math:`10^{-8}`. The distribution in an aperture is the subject of Notes
Section T:6.6.
"""

__all__ = [
    "BreakCurves",
    "aperture_losvd_panels",
    "aperture_output_rule",
    "curve_radii",
    "rows_rule_moments",
]

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..typing import Array
from ..validation import TIER_BOUNDS
from ._aperture import general_aperture_nodes, outside_table, unit_rule
from ._ellipse import EllipseRoute
from ._ellipse_rules import BREAK_TOL, gl01
from ._panel_table import PanelTable
from ._projection import _refuse_below
from .aperture import Aperture

# The names of the five break velocities of a row, in the column order of
# BreakCurves.val.
CURVE_KEYS = ("V_k", "V_end", "V_rad", "V_edge", "V_c")


class BreakCurves:
    """The five break velocities of a row (`CURVE_KEYS`) as functions of
    projected radius, cached at the projected radii of one aperture.

    A call returns the break velocities at the requested projected radii.
    The route computes each projected radius once, and the cache keeps the
    result. A new radius R gets a bracketed search in three conditions: R
    lies between two cached radii; at the two cached radii, the positions
    of the line-of-sight maxima differ by at most one coarse scan step h;
    and, for a route with an L factor, the positions of the second maxima
    also differ by at most h. The bracket of the search runs from the
    smaller to the larger position, widened by h on each side. Every other
    new radius gets the full scan. The counters ``n_near`` and ``n_full``
    count the radii of each kind.

    The cache holds a merged sorted part (``R``, ``val``, ``t``) and a
    pending sorted part. When the pending part holds more than `_MERGE`
    radii, the pending part is merged into the merged part.
    """

    _MERGE = 4096

    def __init__(self, route: EllipseRoute) -> None:
        self.route, self.R = route, np.zeros(0)
        self.val, self.t = np.zeros((0, 5)), np.zeros((0, 2))
        self._pR, self._pv, self._pt = np.zeros(0), np.zeros((0, 5)), np.zeros((0, 2))
        self.n_full = self.n_near = 0

    def _find(self, r: Array) -> tuple[Array, Array, Array, Array]:
        j = np.searchsorted(self.R, r)
        jc = np.minimum(j, max(self.R.size - 1, 0))
        fm = (
            (j < self.R.size) & (self.R[jc] == r)
            if self.R.size
            else np.zeros(r.shape, dtype=bool)
        )
        k = np.searchsorted(self._pR, r)
        kc = np.minimum(k, max(self._pR.size - 1, 0))
        fp = (
            (k < self._pR.size) & (self._pR[kc] == r)
            if self._pR.size
            else np.zeros(r.shape, dtype=bool)
        )
        return fm, jc, fp, kc

    def _add(self, new: Array) -> None:
        e = self.route
        tmax = np.arccosh(np.maximum(e.R_t / new, 1.0))
        h = tmax / (e.rules.brk_scan - 1)
        near = np.zeros(len(new), dtype=bool)
        br = None
        if self.R.size + self._pR.size >= 2:
            lo_r, hi_r = np.full(len(new), -np.inf), np.full(len(new), np.inf)
            ta, tb = np.zeros((len(new), 2)), np.zeros((len(new), 2))
            for rs, ts in ((self.R, self.t), (self._pR, self._pt)):
                if rs.size:
                    j = np.searchsorted(rs, new)
                    a = np.clip(j - 1, 0, rs.size - 1).astype(np.intp, copy=False)
                    b = np.clip(j, 0, rs.size - 1).astype(np.intp, copy=False)
                    up = (j > 0) & (rs[a] > lo_r)
                    dn = (j < rs.size) & (rs[b] < hi_r)
                    lo_r, ta = (
                        np.where(up, rs[a], lo_r),
                        np.where(up[:, None], ts[a], ta),
                    )
                    hi_r, tb = (
                        np.where(dn, rs[b], hi_r),
                        np.where(dn[:, None], ts[b], tb),
                    )
            inside = np.isfinite(lo_r) & np.isfinite(hi_r)
            d = np.abs(ta - tb)
            agree = d[:, 0] <= h
            if e.lfac:
                agree &= d[:, 1] <= h
            near = inside & agree
            lo, hi = np.minimum(ta, tb) - h[:, None], np.maximum(ta, tb) + h[:, None]
            br = (lo[near, 0], hi[near, 0], lo[near, 1], hi[near, 1])
        vals, ts = np.zeros((len(new), 5)), np.zeros((len(new), 2))
        for sel, bracket in ((near, br), (~near, None)):
            if sel.any():
                vk, vend, vr, t_end, t_rad = e.breaks_many(
                    new[sel], with_t=True, bracket=bracket
                )
                vk = e.vk_break(new[sel], vk)
                vr = e.vrad_break(new[sel], vr)
                vals[sel, 0], vals[sel, 1], vals[sel, 2] = vk, vend, vr
                ts[sel, 0], ts[sel, 1] = t_end, t_rad
        vals[:, 3], vals[:, 4] = e.v_edge_many(new), e.v_c_many(new)
        vals[:, 0], vals[:, 2], vals[:, 3], vals[:, 4] = e.lattice.resolved_breaks(
            vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4]
        )
        self.n_near += int(near.sum())
        self.n_full += int((~near).sum())
        at: Any = np.searchsorted(self._pR, new)
        self._pR = np.insert(self._pR, at, new)
        self._pv = np.insert(self._pv, at, vals, axis=0)
        self._pt = np.insert(self._pt, at, ts, axis=0)
        if self._pR.size > self._MERGE:
            at = np.searchsorted(self.R, self._pR)
            self.R = np.insert(self.R, at, self._pR)
            self.val = np.insert(self.val, at, self._pv, axis=0)
            self.t = np.insert(self.t, at, self._pt, axis=0)
            self._pR, self._pv, self._pt = (
                np.zeros(0),
                np.zeros((0, 5)),
                np.zeros((0, 2)),
            )

    def __call__(self, r: Any) -> dict[str, Array]:
        r = np.asarray(r, dtype=np.float64)
        fm, _, fp, _ = self._find(r)
        new = np.unique(r[~(fm | fp)])
        if new.size:
            self._add(new)
        fm, jm, fp, kp = self._find(r)
        vals = np.where(
            fm[:, None],
            self.val[jm] if self.R.size else 0.0,
            self._pv[kp] if self._pR.size else 0.0,
        )
        return {k: vals[:, j] for j, k in enumerate(CURVE_KEYS)}


def curve_radii(
    curves: Callable[[Array], dict[str, Array]],
    name: str,
    v: Array,
    rs: Array,
    vs: Array,
) -> list[list[float]]:
    """Return, for each velocity in `v`, the projected radii where the break
    curve `name` equals the velocity.

    The sample (`rs`, `vs`) holds the break curve's values `vs` at the
    projected radii `rs`. A velocity strictly between the curve's values at two
    consecutive sample radii has one root between the two sample radii. The root
    search is regula falsi with the Illinois step, on the break curve
    itself (`curves`). The search ends when the root's bracket is at most
    one unit in the last place of the bracket's upper radius, or after 60
    steps. The returned root is the midpoint of the final bracket."""
    out: list[list[float]] = [[] for _ in range(len(v))]
    ok = np.isfinite(vs)
    for j in range(len(rs) - 1):
        if not (ok[j] and ok[j + 1]) or vs[j] == vs[j + 1]:
            continue
        a, b = min(vs[j], vs[j + 1]), max(vs[j], vs[j + 1])
        sel = np.flatnonzero((v > a) & (v < b))
        if sel.size == 0:
            continue
        vt = v[sel]
        lo, hi = np.full(sel.size, rs[j]), np.full(sel.size, rs[j + 1])
        flo, fhi = np.full(sel.size, vs[j]) - vt, np.full(sel.size, vs[j + 1]) - vt
        live = np.ones(sel.size, dtype=bool)
        side = np.zeros(sel.size, dtype=np.int8)
        for _ in range(60):
            with np.errstate(divide="ignore", invalid="ignore"):
                m = lo - flo * (hi - lo) / (fhi - flo)
            m = np.where(np.isfinite(m) & (m > lo) & (m < hi), m, 0.5 * (lo + hi))
            k = np.flatnonzero(live)
            if k.size == 0:
                break
            fm = curves(m[k])[name] - vt[k]
            same_as_lo = np.sign(fm) == np.sign(flo[k])
            for q, i in enumerate(k):
                if fm[q] == 0.0:
                    lo[i] = hi[i] = m[i]
                    live[i] = False
                    continue
                if same_as_lo[q]:
                    lo[i], flo[i] = m[i], fm[q]
                    if side[i] == -1:
                        fhi[i] *= 0.5
                    side[i] = -1
                else:
                    hi[i], fhi[i] = m[i], fm[q]
                    if side[i] == 1:
                        flo[i] *= 0.5
                    side[i] = 1
                if hi[i] - lo[i] <= np.spacing(hi[i]):
                    live[i] = False
        for q, s in enumerate(sel):
            out[s].append(0.5 * (lo[q] + hi[q]))
    return out


def aperture_output_rule(
    route: EllipseRoute, curves: dict[str, Array], v_top: float
) -> tuple[Array, Array, Array, Array]:
    """Return the velocity rule of the aperture distribution's own panels.

    The returned tuple holds the ``V >= 0`` nodes, the weights, the panel
    breaks, and the edges of the sub-panels. The candidate breaks are 0,
    the least and the largest value over the aperture of each break curve
    in `curves` inside (0, `v_top`), and `v_top`. A candidate break is kept
    where the candidate exceeds the last kept break by more than
    ``BREAK_TOL * v_top``. A least or largest value of ``V_k`` is kept
    wherever the value exceeds the last kept break. The last break is set
    to `v_top`. Each panel is split into sub-panels as a row's panels are
    (``route.lattice.panel_edges_unit``). Each sub-panel carries a
    Gauss-Legendre rule of ``route.rules.vp_nodes`` nodes."""
    ends = sorted(
        {
            float(x)
            for vv in curves.values()
            if np.isfinite(vv).any()
            for x in (np.nanmin(vv), np.nanmax(vv))
            if 0.0 < x < v_top
        }
    )
    vk = curves.get("V_k")
    keep = (
        {float(x) for x in (np.nanmin(vk), np.nanmax(vk)) if 0.0 < x < v_top}
        if vk is not None and np.isfinite(vk).any()
        else set()
    )
    br = [0.0]
    for x in [*ends, float(v_top)]:
        if x - br[-1] > BREAK_TOL * v_top or (x in keep and x > br[-1]):
            br.append(x)
    br[-1] = float(v_top)
    xg, wg = gl01(route.rules.vp_nodes)
    vn, wn, sub = [], [], []
    for i in range(len(br) - 1):
        e = br[i] + (br[i + 1] - br[i]) * route.lattice.panel_edges_unit(
            i == 0, None, None
        )
        e[0], e[-1] = br[i], br[i + 1]
        vn.append((e[:-1, None] + np.diff(e)[:, None] * xg).ravel())
        wn.append((np.diff(e)[:, None] * wg).ravel())
        sub.append(np.column_stack([e[:-1], e[1:]]))
    return np.concatenate(vn), np.concatenate(wn), np.array(br), np.concatenate(sub)


def rows_rule_moments(
    table: PanelTable, route: EllipseRoute, r_nodes: Array, w_nodes: Array
) -> tuple[float, float]:
    """Return the norm and the second moment of the aperture distribution by
    the rows' own velocity rules.

    At each radial node, the stencil combines rows of the table with the
    stencil's weights. Each row is integrated on the row's own ``V >= 0``
    nodes and weights. Each node of the row is moved to the velocity with the
    same panel and the same relative position xi in the panel, among the
    breaks at the radial node. The radial weights `w_nodes` weight the
    radial nodes. The factor 2 adds the ``V < 0`` half of the even
    distribution."""
    m = table.lattice.rules.vp_nodes
    table._prep_key = None
    table.at(route, r_nodes, 0.0)
    _, brk, _, _, rows, lag = table._prep
    n0 = m2 = 0.0
    for c in range(rows.shape[1]):
        for i in np.unique(rows[:, c]):
            if i < 0:
                continue
            s = np.flatnonzero(rows[:, c] == i)
            v, w, d = table.row(int(i))
            h = len(v) // 2
            v, w, d = v[h:], w[h:], d[h:]
            br = np.asarray(table.row_breaks(int(i)))
            units = table.lattice.panel_units(int(table._layout[int(i)]))
            p = np.concatenate(
                [np.full((len(u) - 1) * m, k) for k, u in enumerate(units)]
            )
            xi = (v - br[p]) / (br[p + 1] - br[p])
            vt = brk[s][:, p] + xi[None, :] * (brk[s][:, p + 1] - brk[s][:, p])
            wd = (w * d)[None, :] * (lag[s, c] * w_nodes[s])[:, None]
            n0 += 2.0 * wd.sum()
            m2 += 2.0 * (wd * vt * vt).sum()
    return n0, m2


def aperture_losvd_panels(
    surface: Any,
    table: PanelTable,
    route: EllipseRoute,
    aperture: Aperture,
    n_nodes: int,
    nodes_per_decade: int,
    inner_tolerance: float,
    even: Sequence[float] | None = None,
    record: list[tuple[Array, Array, dict[str, Array]]] | None = None,
    extrema: Callable[[Array], None] | None = None,
) -> tuple[Array, Array, Array, Array, dict[str, Any], Array | None, Array | None]:
    r"""Return the aperture distribution on the distribution's own velocity
    panels.

    The base radial rule over the aperture is `general_aperture_nodes`, up
    to the smaller of ``surface.r_max`` and the table's last radius
    ``table.R_max``. The base weights include the surface density and are
    normalized to unit sum. A velocity V that equals no break curve inside
    the aperture uses the base radial rule. Every other velocity V uses a
    radial rule of its own, with the projected radii where a break curve
    equals V (`curve_radii`) as extra panel breaks.

    The velocity rule starts from `aperture_output_rule`. The velocity rule
    is refined in rounds. Each round bisects every sub-panel wider than
    half of the widest sub-panel. The refinement ends at the first of three
    conditions: the relative differences of the norm and of the second
    moment from the rows' own rules (`rows_rule_moments`) are both at most
    :math:`10^{-8}`; the widest sub-panel is no wider than the narrowest
    sub-panel of the starting rule; or a round changes the norm and the
    second moment by at most :math:`10^{-8}` relative while the difference
    from the rows' own rules is still above :math:`10^{-8}`.

    Parameters
    ----------
    surface : callable
        The reader of the surface density (`general_aperture_nodes`).
    table : PanelTable
        The table of rows, read between rows by its stencil.
    route : EllipseRoute
        The route of the component, without the DF.
    aperture : Aperture
        The aperture, with lengths in internal units.
    n_nodes : int
        The least number of Gauss-Legendre nodes of each radial panel.
    nodes_per_decade : int
        Radial nodes per decade of a panel's span in ln R, where this count
        gives a panel more nodes than `n_nodes`.
    inner_tolerance : float
        The largest accepted estimated share of the aperture's weight inside
        the table's first row.
    even : sequence of float, optional
        A pair (count, v_max). With `even`, the distribution is also
        evaluated on ``count`` velocities evenly spaced on [-v_max, v_max], or
        on [-v_top, v_top] where v_max is not positive. Default: ``None``,
        which means no evenly spaced evaluation.
    record : list, optional
        With `record`, the call appends one entry per returned node, in node
        order. Each entry holds the projected radii at which the node reads
        the table, the normalized weights of those projected radii, and the
        break velocities at those projected radii. The node's value of D is
        the weighted sum of the table read at the entry's radii and the
        node's V. Default: ``None``, which means no record.
    extrema : callable, optional
        With `extrema`, the call computes the line-of-sight extrema
        (`EllipseRoute.f_extrema`) at all radii of the sample and of each
        velocity's own radial rule in one batch, before the break curves read
        the extrema. Each extremum at a radius equals the extremum that the
        radius forms when computed alone, so the result does not change.
        Default: ``None``, which means each break curve computes the
        extrema as the curve reads them.

    Returns
    -------
    v : array
        The ``V >= 0`` nodes of the velocity rule.
    w : array
        The weights of the velocity rule.
    d : array
        The aperture distribution D at the nodes.
    breaks : array
        The panel breaks of the velocity rule.
    info : dict
        The norm and second moment by the velocity rule (``norm_file``,
        ``m2_file``) and by the rows' own rules (``norm_rows``,
        ``m2_rows``), the node counts, the scan counts of the break curves,
        the stencil order and the stencil's error estimate
        (``stencil_K``, ``stencil_est``), and the refinement's rounds, end
        condition and first and last relative differences.
    v_even : array or None
        The evenly spaced velocities, or None without `even`.
    d_even : array or None
        D at the evenly spaced velocities, zero beyond v_top, or None without
        `even`.

    Raises
    ------
    InputError
        If the aperture encloses no tabulated surface density.
    InputError
        If the estimated share of the aperture's weight inside the table's
        first row is above `inner_tolerance`.
    InputError
        If a node of the base radial rule lies inside the first row of the
        table of rows, ``table.R_2D[0]`` (`_projection._refuse_below`).
    """
    t_gl, w_h = unit_rule(n_nodes)
    r_edge = min(surface.r_max, float(table.R_max))
    arc, gbreaks = aperture.arc, list(aperture.breaks())

    def radial(extra: Sequence[float] = ()) -> tuple[Array, Array, Array]:
        rn, wa, sg = general_aperture_nodes(
            surface,
            arc,
            gbreaks + [float(x) for x in extra],
            t_gl,
            w_h,
            nodes_per_decade,
            inner_tolerance,
            r_edge=r_edge,
        )
        ww = wa * sg
        live = ww > 0.0
        return rn[live], ww[live], wa[live]

    r_base, w_base, _ = radial()
    denom = float(w_base.sum())
    if not (denom > 0.0):
        raise outside_table(surface.r_min, r_edge)
    w_base = w_base / denom
    curves = BreakCurves(route)
    rs = np.unique(
        np.concatenate(
            [r_base, np.geomspace(float(r_base.min()), float(r_base.max()), 65)]
        )
    )
    if extrema is not None:
        extrema(rs)
    sample = curves(rs)
    v_top = float(np.nanmax(sample["V_end"]))
    vp, wp, breaks, sub = aperture_output_rule(route, sample, v_top)

    def radii_for(v: Array) -> list[list[float]]:
        extra: list[list[float]] = [[] for _ in range(len(v))]
        for name, vs in sample.items():
            if np.isfinite(vs).sum() > 1 and np.nanmax(vs) > np.nanmin(vs):
                for s, r in enumerate(curve_radii(curves, name, v, rs, vs)):
                    extra[s] += r
        return extra

    _refuse_below(r_base, float(table.R_2D[0]), "line-of-sight velocity distribution")

    def d_bar(
        v: Array,
        extra: list[list[float]],
        src: list[Any] | None = None,
    ) -> tuple[Array, int]:
        d = np.zeros(len(v))
        nr = len(r_base)
        base = [s for s in range(len(v)) if not extra[s]]
        if src is not None:
            src[:] = [None] * len(v)
        if base:
            cb = curves(r_base)
            dq = table.at_pairs(
                route,
                np.tile(r_base, len(base)),
                np.repeat(np.asarray(v, dtype=np.float64)[base], nr),
                curves={k: np.tile(x, len(base)) for k, x in cb.items()},
            ).reshape(len(base), nr)
            for t, s in enumerate(base):
                d[s] = float(np.dot(w_base, dq[t]))
                if src is not None:
                    src[s] = (r_base, w_base, cb)
        mine = [s for s in range(len(v)) if extra[s]]
        if mine:
            sets = [radial(extra[s]) for s in mine]
            if extrema is not None:
                extrema(np.concatenate([x[0] for x in sets]))
            for x in sets:
                curves(x[0])
            rq = np.concatenate([x[0] for x in sets])
            cq = curves(rq)
            dq = table.at_pairs(
                route,
                rq,
                np.concatenate(
                    [
                        np.full(len(x[0]), float(v[s]))
                        for s, x in zip(mine, sets, strict=True)
                    ]
                ),
                curves=cq,
            )
            o = 0
            for s, (rn, wn, _) in zip(mine, sets, strict=True):
                d[s] = float(np.dot(wn, dq[o : o + len(rn)])) / denom
                if src is not None:
                    part = slice(o, o + len(rn))
                    src[s] = (rn, wn / denom, {k: x[part] for k, x in cq.items()})
                o += len(rn)
        return d, len(mine)

    sp: list[Any] | None = None if record is None else []
    dp, own = d_bar(vp, radii_for(vp), sp)
    n_rows, m2_rows = rows_rule_moments(table, route, r_base, w_base)
    k_use = int(table.order)
    table.order = k_use - 2
    _, m2_low = rows_rule_moments(table, route, r_base, w_base)
    table.order, table._prep_key = k_use, None
    stencil_est = abs(m2_low / m2_rows - 1.0) if m2_rows != 0.0 else 0.0
    xg_o, wg_o = gl01(route.rules.vp_nodes)
    m_o = len(xg_o)
    w_floor = float(np.min(sub[:, 1] - sub[:, 0]))

    def gap_of(vn: Array, wn: Array, dn: Array) -> tuple[float, float, float]:
        nf, mf = 2.0 * float(np.dot(wn, dn)), 2.0 * float(np.dot(wn, vn * vn * dn))
        return nf, mf, max(abs(nf / n_rows - 1.0), abs(mf / m2_rows - 1.0))

    n_file, m2_file, gap0 = gap_of(vp, wp, dp)
    gap, rounds, end = gap0, 0, "tolerance"
    # The refinement tolerance 1e-2 * TIER_BOUNDS[0] = 1e-8 lies two decades
    # below the 1e-6 bound of the aperture distribution's checks.
    while gap > 1e-2 * TIER_BOUNDS[0]:
        wid = sub[:, 1] - sub[:, 0]
        if not float(wid.max()) > w_floor:
            end = "floor"
            break
        pick = wid > 0.5 * float(wid.max())
        v_new: list[Array] = []
        w_new: list[Array] = []
        d_new: list[Any] = []
        s_new: list[Any] = []
        sub_new: list[tuple[float, float]] = []
        todo: list[int] = []
        par: list[int] = []
        for k in range(len(sub)):
            a, b = float(sub[k, 0]), float(sub[k, 1])
            if pick[k]:
                for lo, hi in ((a, 0.5 * (a + b)), (0.5 * (a + b), b)):
                    todo.append(len(v_new))
                    par.append(k)
                    v_new.append(lo + (hi - lo) * xg_o)
                    w_new.append((hi - lo) * wg_o)
                    d_new.append(None)
                    s_new.append(None)
                    sub_new.append((lo, hi))
            else:
                v_new.append(vp[k * m_o : (k + 1) * m_o])
                w_new.append(wp[k * m_o : (k + 1) * m_o])
                d_new.append(dp[k * m_o : (k + 1) * m_o])
                if sp is not None:
                    s_new.append(sp[k * m_o : (k + 1) * m_o])
                sub_new.append((a, b))
        vq = np.concatenate([v_new[j] for j in todo])
        sq: list[Any] | None = None if sp is None else []
        dq, own_q = d_bar(vq, radii_for(vq), sq)
        for t, j in enumerate(todo):
            d_new[j] = dq[t * m_o : (t + 1) * m_o]
            if sq is not None:
                s_new[j] = sq[t * m_o : (t + 1) * m_o]
        moved = 0.0
        for t in range(0, len(todo), 2):
            k, j0, j1 = par[t], todo[t], todo[t + 1]
            s = slice(k * m_o, (k + 1) * m_o)
            dn = 2.0 * (
                float(np.dot(w_new[j0], d_new[j0]))
                + float(np.dot(w_new[j1], d_new[j1]))
                - float(np.dot(wp[s], dp[s]))
            )
            dm = 2.0 * (
                float(np.dot(w_new[j0], v_new[j0] ** 2 * d_new[j0]))
                + float(np.dot(w_new[j1], v_new[j1] ** 2 * d_new[j1]))
                - float(np.dot(wp[s], vp[s] ** 2 * dp[s]))
            )
            moved = max(moved, abs(dn) / n_rows, abs(dm) / m2_rows)
        vp, wp, dp = np.concatenate(v_new), np.concatenate(w_new), np.concatenate(d_new)
        if sp is not None:
            sp = [x for part in s_new for x in part]
        sub = np.array(sub_new)
        own += own_q
        rounds += 1
        n_file, m2_file, gap = gap_of(vp, wp, dp)
        if gap > 1e-2 * TIER_BOUNDS[0] and not moved > 1e-2 * TIER_BOUNDS[0]:
            end = "stalled"
            break
    info: dict[str, Any] = dict(
        norm_file=n_file,
        m2_file=m2_file,
        norm_rows=n_rows,
        m2_rows=m2_rows,
        n_radial=len(r_base),
        n_own=own,
        n_V=len(vp),
        R_lo=float(r_base.min()),
        R_hi=float(r_base.max()),
        n_scan_full=curves.n_full,
        n_scan_near=curves.n_near,
        stencil_K=k_use,
        stencil_est=stencil_est,
        refine_rounds=rounds,
        refine_end=end,
        gap_first=gap0,
        gap_last=gap,
    )
    if record is not None and sp is not None:
        record.extend(sp)
    v_even = d_even = None
    if even is not None:
        vt = float(even[1]) if float(even[1]) > 0.0 else v_top
        v_even = np.linspace(-vt, vt, int(even[0]))
        va = np.abs(v_even)
        d_even = np.zeros(len(va))
        ins = va <= v_top
        if ins.any():
            d_even[ins] = d_bar(va[ins], radii_for(va[ins]))[0]
    return vp, wp, dp, breaks, info, v_even, d_even
