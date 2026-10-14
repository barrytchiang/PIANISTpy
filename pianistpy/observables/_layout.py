"""The V panels of a row of the line-of-sight velocity distribution: which
breaks a row carries, in which order, and the lattice of its nodes."""

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

# The interior breaks: k = V_k, r = V_rad, e = V_edge, c = V_c. The layouts by
# code are append-only, so that no stored code changes meaning: 0 to 15 as
# written, 16 to 64 the orders that contain c, by length, each in the order of
# itertools.permutations over "krec".
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
# A kept V_k inside the zero grading's first sub-panel: the same breaks, the
# panel above V_k graded toward it as the first panel is toward V = 0.
LAYOUT_SEQ.update({DEEP_K + c: s for c, s in list(LAYOUT_SEQ.items()) if "k" in s})
# The order in which an unresolved panel's interior breaks are merged.
MERGE_ORDER = "kcre"


class Lattice:
    """The grading of a row's V panels and the tests on its nodes.

    Parameters
    ----------
    rules : EllipseRules
        The route's rules.
    model : EllipseModel
        The model's constants (the layers toward V_edge and V_c).
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
        """Return (V_k interior, V_rad interior, V_edge interior, V_c
        interior), elementwise: a break within BREAK_TOL of V_end coincides
        with it."""
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
        """Return the layout's code: the order of the interior breaks, ties
        ordering k before r before e before c, plus DEEP_K where a kept V_k
        lies inside the zero grading's first sub-panel."""
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
        """Return the sub-panel edges of one panel on [0, 1], graded toward
        both ends: vp_layers_zero toward 0 on the first panel and
        vp_layers_kink toward every break, unless `lay_lo` or `lay_hi` say."""
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
        """Return the unit sub-panel edges of every panel of a layout."""
        if code not in self._units:
            self._units[code] = [
                self.panel_edges_unit(i == 0, lo, hi)
                for i, (lo, hi) in enumerate(self.panel_layers(code))
            ]
        return self._units[code]

    def panel_layers(self, code: int) -> list[tuple[int, int]]:
        """Return (layers toward the lower end, toward the upper end) for
        every panel of a layout."""
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

    def lattice_gaps(self, br: Array, units: list[Array]) -> tuple[Array, Array]:
        """Return (gap, owner): the gap between each pair of neighboring
        nodes of rows with breaks `br` (m, panels + 1), from the unit offsets
        times the panel's width, over the upper node's value, and the panel of
        each node."""
        xg = gl01(self.rules.vp_nodes)[0]
        w = np.diff(br, axis=1)
        nodes, steps, owner, last = [], [], [], None
        for i in range(br.shape[1] - 1):
            u = units[i]
            un = (u[:-1, None] + np.diff(u)[:, None] * xg[None, :]).ravel()
            nodes.append(br[:, i : i + 1] + w[:, i : i + 1] * un[None, :])
            if last is not None:
                steps.append((w[:, i - 1] * (1.0 - last) + w[:, i] * un[0])[:, None])
            steps.append(w[:, i : i + 1] * np.diff(un)[None, :])
            owner.append(np.full(un.size, i))
            last = un[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.concatenate(steps, axis=1) / np.concatenate(nodes, axis=1)[:, 1:]
        return gap, np.concatenate(owner)

    def resolved_breaks(
        self,
        vk: ArrayLike,
        vend: ArrayLike,
        vrad: ArrayLike,
        vedge: ArrayLike = np.nan,
        vc: ArrayLike = np.nan,
    ) -> tuple[Array, ...]:
        """Return (V_k, V_rad, V_edge, V_c) with every break NaN whose panel
        the row's lattice does not resolve.

        A panel is carried where doubles resolve its nodes (RB_MIN_GAP of their
        own values) and its graded sides (`l_max`); otherwise one of its
        interior breaks is merged (MERGE_ORDER; a panel bounded by 0 or V_end
        loses its other break) and the row is tested again.
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
                gmin = np.full((len(idx), npan), np.inf)
                gap, owner = self.lattice_gaps(br, units)
                o0, o1 = owner[:-1], owner[1:]
                for i in range(npan):
                    inside = (o0 == i) & (o1 == i)
                    if inside.any():
                        gmin[:, i] = np.minimum(gmin[:, i], gap[:, inside].min(axis=1))
                for j in np.flatnonzero(o0 != o1):
                    a_, b_ = int(o0[j]), int(o1[j])
                    to_a = w[:, a_] <= w[:, b_]
                    gmin[to_a, a_] = np.minimum(gmin[to_a, a_], gap[to_a, j])
                    gmin[~to_a, b_] = np.minimum(gmin[~to_a, b_], gap[~to_a, j])
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
