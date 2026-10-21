r"""The velocity panels of one row of the line-of-sight velocity distribution.

The tables behind `pianistpy.losvd`, `pianistpy.aperture_losvd` and
`pianistpy.LosTable` store the distribution at a set of projected radii
:math:`R_{\rm 2D}`, one row per radius. A row covers the line-of-sight velocities
:math:`0 \le V \le V_{\rm end}`, with :math:`V_{\rm end}` the velocity above
which the row's distribution is zero. Up to four interior break velocities
split :math:`[0, V_{\rm end}]` into panels: :math:`V_k`, :math:`V_{\rm rad}`,
:math:`V_{\rm edge}` and :math:`V_{\rm c}` (`EllipseRoute.breaks_many`,
`EllipseRoute.v_edge_many` and `EllipseRoute.v_c_many` compute the four
values). This module codes the order of a row's interior breaks as an integer
layout code, grades the sub-panels of each panel toward the panel's ends, and
merges an interior break where the panel next to the break is too narrow for
double precision.

A panel graded with :math:`L_{\rm lo}` layers toward its lower end and
:math:`L_{\rm hi}` layers toward its upper end has the sub-panel edges, in
units of the panel's width from the panel's lower end,

.. math::

    0,\ \tfrac12 r^{L_{\rm lo}},\ \ldots,\ \tfrac12 r,\ \tfrac12,\
    1 - \tfrac12 r,\ \ldots,\ 1 - \tfrac12 r^{L_{\rm hi}},\ 1,

with :math:`r` the setting ``losvd.velocity_layer_ratio``. Each sub-panel
carries a Gauss-Legendre rule whose node count is the setting
``losvd.n_velocity_nodes``. Notes
Section T:6.4 treats the line-of-sight velocity distribution."""

__all__ = [
    "BREAK_LABELS",
    "LAYOUT_SEQ",
    "MERGE_ORDER",
    "SEQ_LAYOUT",
    "Lattice",
]

import itertools

import numpy as np

from ..typing import Array, ArrayLike
from ._ellipse_rules import (
    BREAK_TOL,
    DEEP_K,
    RB_MIN_GAP,
    EllipseModel,
    EllipseRules,
    gl01,
)

# The letters of the interior breaks: k is V_k, r is V_rad, e is V_edge and c
# is V_c. LAYOUT_SEQ maps a layout code to the letters of a row's interior
# breaks in increasing velocity. Codes 0 to 15 are the orders without V_c, as
# listed. Codes 16 to 64 are the orders that contain V_c, by length, each
# length in the order of itertools.permutations over "krec". New codes are only
# appended, so a code stored in a table keeps its meaning.
BREAK_LABELS = "krec"
LAYOUT_SEQ: dict[int, str] = {
    0: "",
    1: "k",
    2: "r",
    3: "kr",
    4: "rk",
    5: "e",
    6: "ke",
    7: "ek",
    8: "re",
    9: "er",
    10: "kre",
    11: "ker",
    12: "ekr",
    13: "rke",
    14: "rek",
    15: "erk",
}
LAYOUT_SEQ.update(
    {
        16 + j: s
        for j, s in enumerate(
            "".join(p)
            for n in range(1, 5)
            for p in itertools.permutations(BREAK_LABELS, n)
            if "c" in p
        )
    }
)
SEQ_LAYOUT = {s: c for c, s in LAYOUT_SEQ.items()}
# A code plus DEEP_K marks a row whose interior V_k lies below the first
# sub-panel edge of the panel [0, V_end] graded toward V = 0. The code plus
# DEEP_K has the same breaks as the code. The panel above V_k is then graded
# toward V_k with as many layers as the first panel has toward V = 0.
LAYOUT_SEQ.update({DEEP_K + c: s for c, s in list(LAYOUT_SEQ.items()) if "k" in s})
# The preference order of the interior break merged at a panel that double
# precision does not resolve: V_k first, then V_c, then V_rad, then V_edge.
MERGE_ORDER = "kcre"


class Lattice:
    """The sub-panel grading of a row's velocity panels, and the tests of
    whether double precision resolves the row's nodes.

    Parameters
    ----------
    rules : EllipseRules
        The numerical rules of the ellipse route, read from the settings'
        ``losvd`` group.
    model : EllipseModel
        The model's constants, among them the layer counts toward V_edge and
        toward V_c.
    """

    def __init__(self, rules: EllipseRules, model: EllipseModel) -> None:
        self.rules = rules
        self.model = model
        self._units: dict[int, list[Array]] = {}

    def interior_breaks(
        self,
        vk: ArrayLike,
        vend: ArrayLike,
        vrad: ArrayLike,
        vedge: ArrayLike = np.nan,
        vc: ArrayLike = np.nan,
    ) -> tuple[Array, ...]:
        """Return four boolean arrays that mark where V_k, V_rad, V_edge and
        V_c are interior breaks.

        A break velocity is an interior break where three conditions hold:
        the break velocity is finite; the break velocity is positive; and
        V_end minus the break velocity exceeds BREAK_TOL (``1e-9``) times
        V_end. The arguments broadcast against each other.
        """
        vk, vend, vrad, vedge, vc = (
            np.asarray(x, dtype=np.float64) for x in (vk, vend, vrad, vedge, vc)
        )
        with np.errstate(invalid="ignore"):
            a = np.isfinite(vk) & (vk > 0.0) & (vend - vk > BREAK_TOL * vend)
            b = np.isfinite(vrad) & (vrad > 0.0) & (vend - vrad > BREAK_TOL * vend)
            c = np.isfinite(vedge) & (vedge > 0.0) & (vend - vedge > BREAK_TOL * vend)
            d = np.isfinite(vc) & (vc > 0.0) & (vend - vc > BREAK_TOL * vend)
        return tuple(np.broadcast_arrays(a, b, c, d))

    def layout_code(
        self,
        vk: ArrayLike,
        vend: ArrayLike,
        vrad: ArrayLike,
        vedge: ArrayLike = np.nan,
        vc: ArrayLike = np.nan,
    ) -> Array:
        """Return the layout code of each row from the row's break velocities.

        The code is the entry of SEQ_LAYOUT for the row's interior breaks
        (`interior_breaks`) in increasing velocity. Equal break velocities keep
        the order V_k, V_rad, V_edge, V_c. The code is raised by DEEP_K where
        V_k is an interior break below ``0.5 r**L * V_end``, with ``r`` the
        rules' ``vp_ratio`` and ``L`` the rules' ``vp_layers_zero``.
        """
        fl = self.interior_breaks(vk, vend, vrad, vedge, vc)
        shape, nb = np.shape(fl[0]), len(BREAK_LABELS)
        vals = np.stack(
            [
                np.where(f, v, np.inf)
                for f, v in zip(
                    fl, np.broadcast_arrays(vk, vrad, vedge, vc), strict=True
                )
            ],
            axis=-1,
        ).reshape(-1, nb)
        n = (np.stack(fl, axis=-1).reshape(-1, nb)).sum(axis=1)
        order = np.argsort(vals, axis=1, kind="stable")
        code = np.array(
            [
                SEQ_LAYOUT["".join(BREAK_LABELS[j] for j in order[i, : n[i]])]
                for i in range(len(n))
            ],
            dtype=np.int64,
        )
        r = self.rules
        with np.errstate(invalid="ignore"):
            deep = fl[0] & (
                np.broadcast_to(vk, shape)
                < 0.5 * r.vp_ratio**r.vp_layers_zero * np.broadcast_to(vend, shape)
            )
        return code.reshape(shape) + DEEP_K * deep

    def panel_edges_unit(
        self, first: bool, lay_lo: int | None = None, lay_hi: int | None = None
    ) -> Array:
        """Return the sub-panel edges of one panel in units of the panel's
        width, from 0 at the panel's lower end to 1 at the panel's upper end.

        The edges are graded geometrically toward both ends with the ratio
        ``vp_ratio``: `lay_lo` layers toward the lower end and `lay_hi` layers
        toward the upper end. Where `lay_lo` is None, the lower end takes
        ``vp_layers_zero`` layers on the first panel (`first` true) and
        ``vp_layers_kink`` layers on any other panel. Where `lay_hi` is None,
        the upper end takes ``vp_layers_kink`` layers.
        """
        rr = self.rules
        r = rr.vp_ratio
        lo = (
            (rr.vp_layers_zero if first else rr.vp_layers_kink)
            if lay_lo is None
            else lay_lo
        )
        ua = 0.5 * r ** np.arange(lo, -1, -1)
        ub = 0.5 * r ** np.arange(
            rr.vp_layers_kink if lay_hi is None else lay_hi, -1, -1
        )
        return np.concatenate([[0.0], ua, 1.0 - ub[::-1][1:], [1.0]])

    def panel_units(self, code: int) -> list[Array]:
        """Return the sub-panel edges of every panel of layout `code`, each in
        units of the panel's width (`panel_edges_unit`), computed once per
        layout code."""
        if code not in self._units:
            self._units[code] = [
                self.panel_edges_unit(i == 0, lo, hi)
                for i, (lo, hi) in enumerate(self.panel_layers(code))
            ]
        return self._units[code]

    def panel_layers(self, code: int) -> list[tuple[int, int]]:
        """Return, for every panel of layout `code`, the pair (layers toward
        the panel's lower end, layers toward the panel's upper end).

        The lower end of a panel takes one of three layer counts: the model's
        ``vp_layers_c`` where the panel's lower end is V_c; the rules'
        ``vp_layers_zero`` where the panel is the first panel, or where the
        panel's lower end is V_k and the code is DEEP_K or above; and the
        rules' ``vp_layers_kink`` otherwise. The upper end of a panel takes
        the model's ``vp_layers_edge`` where the panel's upper end is V_edge,
        and the rules' ``vp_layers_kink`` otherwise.
        """
        rr, m = self.rules, self.model
        seq = LAYOUT_SEQ[int(code)]
        return [
            (
                m.vp_layers_c
                if i > 0 and seq[i - 1] == "c"
                else rr.vp_layers_zero
                if i == 0 or (seq[i - 1] == "k" and int(code) >= DEEP_K)
                else rr.vp_layers_kink,
                m.vp_layers_edge
                if i < len(seq) and seq[i] == "e"
                else rr.vp_layers_kink,
            )
            for i in range(len(seq) + 1)
        ]

    def lattice_min_gaps(self, br: Array, units: list[Array]) -> Array:
        """Return the smallest relative node gap of each panel, shape
        (m, panels), for m rows with panel ends `br`, shape (m, panels + 1).

        The relative gap of two neighboring nodes is the velocity difference
        of the two nodes divided by the velocity of the upper node. Panel i's
        nodes are the ``vp_nodes``-point Gauss-Legendre nodes of each
        sub-panel, with the sub-panel edges ``units[i]`` given in units of the
        panel's width. A gap between two nodes of one panel belongs to that panel. A
        gap across a break belongs to the narrower of the two panels at the
        break, and to the lower panel where the two panels have equal widths.
        The gaps are formed in blocks of nodes, each block holding at most
        ``rules.chunk`` elements.
        """
        xg = gl01(self.rules.vp_nodes)[0]
        w = np.diff(br, axis=1)
        m, npan = w.shape
        gmin = np.full((m, npan), np.inf)
        cols = max(1, int(self.rules.chunk) // max(m, 1))
        last = None
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(npan):
                u = units[i]
                un = (u[:-1, None] + np.diff(u)[:, None] * xg[None, :]).ravel()
                if last is not None:
                    cross = (w[:, i - 1] * (1.0 - last) + w[:, i] * un[0]) / (
                        br[:, i] + w[:, i] * un[0]
                    )
                    to_a = w[:, i - 1] <= w[:, i]
                    gmin[to_a, i - 1] = np.minimum(gmin[to_a, i - 1], cross[to_a])
                    gmin[~to_a, i] = np.minimum(gmin[~to_a, i], cross[~to_a])
                du = np.diff(un)
                for a in range(0, du.size, cols):
                    step = w[:, i : i + 1] * du[None, a : a + cols]
                    node = (
                        br[:, i : i + 1]
                        + w[:, i : i + 1] * un[None, a + 1 : a + 1 + cols]
                    )
                    gmin[:, i] = np.minimum(gmin[:, i], (step / node).min(axis=1))
                last = un[-1]
        return gmin

    def resolved_breaks(
        self,
        vk: ArrayLike,
        vend: ArrayLike,
        vrad: ArrayLike,
        vedge: ArrayLike = np.nan,
        vc: ArrayLike = np.nan,
    ) -> tuple[Array, ...]:
        r"""Return (V_k, V_rad, V_edge, V_c) with NaN in place of each interior
        break that the row's panels cannot carry in double precision.

        A panel fails in either of two cases: the panel's smallest relative
        node gap (`lattice_min_gaps`) is below RB_MIN_GAP (``1e-12``) or is
        NaN; or a positive end of the panel needs more layers than
        :math:`L_{\max} = \lfloor \ln(x_1 w / (2 \epsilon V))
        / \ln(1/r) \rfloor`, with :math:`x_1` the first Gauss-Legendre node
        on [0, 1], :math:`w` the panel's width, :math:`V` the end's velocity,
        :math:`\epsilon` the machine epsilon of doubles and :math:`r` the
        rules' ``vp_ratio``. In a row with a failed panel, the function takes
        the failed panel with the smallest node gap and sets to NaN one of
        that panel's interior break ends, the one earliest in MERGE_ORDER. The
        ends 0 and V_end are never merged. The function then tests the rows
        again, at most four times in all.
        """
        vk, vend, vrad, vedge, vc = np.broadcast_arrays(
            *(np.asarray(x, dtype=np.float64) for x in (vk, vend, vrad, vedge, vc))
        )
        shape = vk.shape
        vals = {
            s: np.array(v, dtype=np.float64).ravel()
            for s, v in zip(BREAK_LABELS, (vk, vrad, vedge, vc), strict=True)
        }
        ve_ = np.array(vend, dtype=np.float64).ravel()
        xg = gl01(self.rules.vp_nodes)[0]
        x1 = float(xg[0])
        for _ in range(len(BREAK_LABELS)):
            code = np.atleast_1d(
                self.layout_code(vals["k"], ve_, vals["r"], vals["e"], vals["c"])
            )
            merged = False
            for c in np.unique(code):
                seq = LAYOUT_SEQ[int(c)]
                if not seq:
                    continue
                idx = np.flatnonzero(code == c)
                br = np.stack(
                    [np.zeros(len(idx))] + [vals[s][idx] for s in seq] + [ve_[idx]],
                    axis=1,
                )
                w = np.diff(br, axis=1)
                units = self.panel_units(int(c))
                layers = self.panel_layers(int(c))
                npan = len(seq) + 1
                fail = np.zeros((len(idx), npan), dtype=bool)
                gmin = self.lattice_min_gaps(br, units)
                fail |= ~(gmin >= RB_MIN_GAP)
                for i, (lo, hi) in enumerate(layers):
                    for v, need in ((br[:, i], lo), (br[:, i + 1], hi)):
                        with np.errstate(divide="ignore", invalid="ignore"):
                            wv = w[:, i] / v
                            lmax = np.where(
                                wv > 0.0,
                                np.floor(
                                    np.log(x1 * wv / (2.0 * np.finfo(np.float64).eps))
                                    / np.log(1.0 / self.rules.vp_ratio)
                                ),
                                -1.0,
                            )
                        fail[:, i] |= (v > 0.0) & (lmax < need)
                bad = np.flatnonzero(fail.any(axis=1))
                for m in bad:
                    i = int(np.flatnonzero(fail[m])[np.argmin(gmin[m, fail[m]])])
                    cands = [seq[i - 1]] if i > 0 else []
                    cands += [seq[i]] if i < len(seq) else []
                    s = min(cands, key=MERGE_ORDER.index)
                    vals[s][idx[m]] = np.nan
                    merged = True
            if not merged:
                break
        return tuple(vals[s].reshape(shape) for s in BREAK_LABELS)
