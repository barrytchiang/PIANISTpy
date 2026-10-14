r"""The ellipse route of the line-of-sight velocity distribution.

On the plane :math:`v_{\rm los} = V` at :math:`(R_{\rm 2D}, 0, u)` the level
sets of Q are ellipses centerd on Q's maximum, so the velocity-plane integral
is :math:`\int_0^{Q_c}{\rm d}s\,F(Q_c - s)\,\Lambda(s)` times
:math:`(R/R_{\rm 2D})/\sqrt{Aq}`, with :math:`\Lambda(s)` the integral of
:math:`(L^2)^{-\beta_c}` around the level set (:math:`2\pi` without an L
factor, where the s integral is :math:`G(Q_c)`), and
:math:`D(R_{\rm 2D}, V) = 2\int{\rm d}u` of that.
"""

__all__ = ["EllipseRoute"]

from collections.abc import Callable
from typing import Any

import numpy as np

from ..exceptions import ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ..validation import GOAL
from ._angle import angle_count, angle_table, lam_half, lam_mhalf, lam_rule, power_of
from ._angle_rule import angle_rule
from ._ellipse_df import EllipseDF
from ._ellipse_rules import (
    BIRTH_SPAN,
    BIRTH_TOL,
    CUSP_FRAC,
    L_BELOW,
    RB_FILL,
    RB_MIN_GAP,
    RB_SAME,
    T_TOL,
    EllipseModel,
    EllipseRules,
    gl01,
    l_max,
    seg_unit,
)
from ._layout import BREAK_LABELS, LAYOUT_SEQ, Lattice

# The anisotropy's sign in q = 1 + SGN (R / R_a)^2, by family.
_SGN = {
    "isotropic": 0.0,
    "constant": 0.0,
    "OM_I": 1.0,
    "COM_I": 1.0,
    "OM_II": -1.0,
    "COM_II": -1.0,
}
# The codes of the test that set a row's t level; 0 is not converged.
T_SET_BY = {"first": 1, "l1": 2, "pointwise": 3, "both": 4}
# The starting intervals of one chunk of radii of f_extrema_many, which bounds
# its memory.
_EXTREMA_CHUNK = 2_000_000
_T_ZERO = np.zeros(1)


class EllipseRoute:
    """D(R_2D, V) by the ellipse route.

    Parameters
    ----------
    psi : callable
        The potential at radii. It carries ``spline``, the cubic spline of
        ln P in ln R it is read from, and ``R_knots``, the first and last radius
        of its table, outside which it is held constant.
    r_t : float
        Where the lines of sight end.
    model : EllipseModel
        The model's constants.
    df : EllipseDF or None
        The DF; None for a route that only places breaks.
    rules : EllipseRules
        The route's rules.
    """

    def __init__(
        self,
        psi: Callable[[Array], Array],
        r_t: float,
        model: EllipseModel,
        df: EllipseDF | None,
        rules: EllipseRules,
    ) -> None:
        self.psi, self.R_t, self.df = psi, float(r_t), df
        self.model, self.rules = model, rules
        self.lattice = Lattice(rules, model)
        self.RA, self.BC, self.SGN = (
            float(model.r_a),
            float(model.beta_c),
            _SGN[model.family],
        )
        self.lfac = model.family in ("constant", "COM_I", "COM_II") and self.BC != 0.0
        self.powm = power_of(self.BC)
        # beta_c = 1/2 and -1/2: the angle integral in closed form
        self.lam_closed = self.lfac and self.BC in (0.5, -0.5)
        self.angle_counts = np.zeros(len(rules.angle_counts) + 1, dtype=np.int64)
        if self.lfac and not self.lam_closed:
            angle_table()
        # the R_2D ranges where V_rad and V_k may be breaks, (n, 2); None
        # where no bound ran, and then the floor alone decides
        self.vrad_kept: Array | None = None
        self.vk_kept: Array | None = None
        self._fext: dict[float, Array] = {}
        self._live: Array | None = None
        self._top: Array | None = None
        self.crossings: tuple[Array, list[str]] = (np.zeros(0), [])

    def _df(self) -> EllipseDF:
        if self.df is None:
            raise ResultError(
                "This route was built without a DF: it places breaks only."
            )
        return self.df

    # The geometry of a line of sight.

    def q_of(self, r: Array) -> Array:
        return 1.0 + self.SGN * (r / self.RA) ** 2 if self.SGN else np.ones_like(r)

    def rule(self, r2d: float) -> tuple[tuple[int, ...], bool]:
        """Return the row's rule and whether it is of the cusp class."""
        df = self._df()
        cusp = df.Qtop - float(self.psi(np.array([r2d]))[0]) < CUSP_FRAC * df.Qtop
        return (self.rules.rule_cusp if cusp else self.rules.rule_regular), cusp

    def _s_layers(
        self,
        r2d: float,
        r: Array,
        q: Array,
        u: Array,
        v: Array,
        qc: Array,
        split: Array,
        m: int,
        floor: int,
    ) -> Array:
        """Each node pair's layers toward s = 0: the least K whose innermost
        panel there is no wider than the nearest singularity behind s = 0,
        F's top (P0 - Q_c, times min(1, m / |kappa|)) or the angle integral's
        pinch at s = -s_star."""
        df = self._df()
        k = abs(df.kappa_top)
        s_star = 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)
        dist = np.minimum((df.Qend - qc) * min(1.0, m / k if k > 0.0 else 1.0), s_star)
        h = 0.5 * split
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(h > 0.0, dist / h, 1.0)
        if not np.all(x > 0.0):
            i = int(np.argmin(np.where(np.isfinite(x), x, -np.inf)))
            raise ResultError(
                f"At R_2D {r2d:.6e}, u {float(u[i]):.6e}, V {float(v[i]):.6e} no "
                f"singularity lies behind s = 0 (Q_c {float(qc[i]):.15e}, P0 "
                f"{df.Qend:.15e}, s_star {float(s_star[i]):.6e})."
            )
        kk = np.ceil(np.log(np.minimum(x, 1.0)) / np.log(self.rules.s_ratio) - 1e-12)
        return np.maximum(kk.astype(np.int64), int(floor))

    def _ks_layers(
        self,
        r2d: float,
        r: Array,
        q: Array,
        u: Array,
        v: Array,
        qc: Array,
        split: Array,
        s0: Array,
        m: int,
        floor: int,
    ) -> tuple[Array, Array]:
        """Each node pair's layers toward the split, one count a side: piece a
        takes the L = 0 point's own count and the reach of the support's end at
        Q_c; piece b first shrinks to s0, then takes that count (and the map's
        where 0 < beta_c < 1), or reaches F's top and the pinch behind the
        split."""
        df = self._df()
        mdl = self.model
        rr = self.rules.s_ratio
        k = abs(df.kappa_top)
        at0 = (s0 >= 0.0) & (s0 < qc)
        v0 = at0 & ~(split > 0.0)
        s_star = 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)
        hb = 0.5 * (qc - split)
        fd = (split + df.Qend - qc) * min(1.0, m / k if k > 0.0 else 1.0)
        dist = np.where(v0, fd, np.minimum(fd, split + s_star))
        with np.errstate(divide="ignore", invalid="ignore"):
            xd = np.where(hb > 0.0, dist / np.where(hb > 0.0, hb, 1.0), 1.0)
            xs = np.where(
                at0 & ~v0 & (hb > 0.0), split / np.where(hb > 0.0, hb, 1.0), 1.0
            )
        if not np.all(xd > 0.0):
            i = int(np.argmin(np.where(np.isfinite(xd), xd, -np.inf)))
            raise ResultError(
                f"At R_2D {r2d:.6e}, u {float(u[i]):.6e}, V {float(v[i]):.6e} no "
                f"singularity lies behind the split (Q_c {float(qc[i]):.15e}, split "
                f"{float(split[i]):.6e}, P0 {df.Qend:.15e}, s_star "
                f"{float(s_star[i]):.6e})."
            )
        kd = np.ceil(np.log(np.minimum(xd, 1.0)) / np.log(rr) - 1e-12).astype(np.int64)
        ksig = np.ceil(np.log(np.minimum(xs, 1.0)) / np.log(rr) - 1e-12).astype(
            np.int64
        )
        kend = np.where(at0, np.where(v0, mdl.k_end_v0, mdl.k_end), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            xf = np.where(
                split > 0.0,
                (qc - split) / (0.5 * np.where(split > 0.0, split, 1.0)),
                1.0,
            )
        kf = np.ceil(np.log(np.clip(xf, 1e-300, 1.0)) / np.log(rr) - 1e-12).astype(
            np.int64
        )
        ksa = np.where(at0 & ~v0, np.maximum(mdl.k_end, kf), 0)
        ksb = np.maximum(
            np.maximum(kd, ksig + np.where(at0, np.maximum(kend, mdl.k_map), 0)),
            mdl.k_map,
        )
        return (
            np.maximum(ksa, int(floor)).astype(np.int64),
            np.maximum(ksb, int(floor)).astype(np.int64),
        )

    def _qc_t(self, r2d: float, v: Any, t: Any) -> Array:
        r, u = r2d * np.cosh(t), r2d * np.sinh(t)
        q = self.q_of(r)
        return self.psi(r) - 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)

    def f_extrema(self, r2d: float) -> Array:
        """Return the interior local extrema in t of f = 2 P (R_2D^2 + q u^2)
        / (q R^2) along the line of sight, each to rounding, sorted.

        With every interior extremum of f a point of the sign test, each step
        holds at most one edge of the support at any V. The extrema are the
        zeros of h = s'(y) + gamma(y), y = ln R, s = ln P a cubic spline: on
        each knot interval s' is a quadratic whose range is exact and gamma is
        monotone on each side of its turn, so an interval whose summed range
        excludes 0 holds none; every other is halved n_bisect times. Cached
        per R_2D. An interval's right end is read from the interval itself
        (the spline's ``left`` where it has one): the profile's Hermite has a
        second derivative that jumps at the rows.
        """
        return self.f_extrema_many(np.array([float(r2d)]))[0]

    def f_extrema_many(self, r2d_arr: ArrayLike) -> list[Array]:
        """Return `f_extrema` at each of the projected radii, the radii not
        yet cached taken at once: each radius's intervals carry its index, and
        every number is the one `f_extrema` forms at that radius alone (3-194).
        """
        r2 = np.asarray(r2d_arr, dtype=np.float64).ravel()
        out: list[Array] = [np.zeros(0)] * r2.size
        todo: dict[float, list[int]] = {}
        for i, x in enumerate(r2.tolist()):
            hit = self._fext.get(x)
            if hit is not None:
                out[i] = hit
            else:
                todo.setdefault(x, []).append(i)
        if not todo:
            return out
        sp = getattr(self.psi, "spline", None)
        if sp is None:
            raise ResultError(
                "The potential states no spline (psi.spline), which the extrema of "
                "the line of sight are bounded on."
            )
        r_lo, r_hi = self.psi.R_knots  # type: ignore[attr-defined]
        for r2d in todo:
            if r2d < r_lo or self.R_t > r_hi:
                raise ResultError(
                    f"At R_2D {r2d:.17g}, R_t {self.R_t:.17g} the line of sight "
                    f"leaves the potential's knots [{r_lo:.17g}, {r_hi:.17g}], where "
                    f"it is held constant and the range bound does not hold."
                )
        vals = np.array(list(todo), dtype=np.float64)
        kn = np.exp(sp.x)
        lo = np.searchsorted(kn, vals, side="right")
        hi = np.maximum(np.searchsorted(kn, self.R_t, side="left"), lo)
        # the knots a radius needs: those next to a knot interval that may be
        # live, between R_2D and R_t
        needed = self._needed_knots(sp)
        i0 = np.searchsorted(needed, lo)
        n_kn = np.searchsorted(needed, hi) - i0
        # radii a chunk, so that a chunk's intervals stay near _EXTREMA_CHUNK
        ends = np.cumsum(n_kn + 3)
        start = 0
        while start < vals.size:
            stop = max(
                start + 1,
                int(
                    np.searchsorted(
                        ends, ends[start] - n_kn[start] - 3 + _EXTREMA_CHUNK
                    )
                ),
            )
            stop = min(stop, vals.size)
            found = self._f_extrema_chunk(
                sp,
                kn,
                vals[start:stop],
                lo[start:stop],
                needed,
                i0[start:stop],
                n_kn[start:stop],
            )
            for r2d, ext in zip(vals[start:stop].tolist(), found, strict=True):
                self._fext[r2d] = ext
                for i in todo[r2d]:
                    out[i] = ext
            start = stop
        return out

    def _live_bound(self, sp: Any) -> Array:
        """Return, per knot interval of the potential, whether any line of
        sight can make an interval inside it live in `f_extrema`'s first bound
        test (3-194).

        The test keeps an interval only if max s' + max gamma >= 0 on it. On
        a knot interval s' is the Hermite's quadratic, whose maximum is at an
        end or at the zero of s'' inside, and gamma <= G(x) = 2 x^2 / (a^2
        (a^2 + SGN x)) at any R_2D (gamma grows with R_2D^2 <= x = R^2; 0 for
        SGN = 0). The maximum of s' is taken over the interval and its two
        neighbors and G at the right end of the next one, against the reads
        that fall across a knot by rounding, and a margin of 1e-9; a knot
        interval that fails holds no live interval at any radius, so dropping
        it changes no number. Cached per route.
        """
        if self._live is not None:
            return self._live
        x = np.asarray(sp.x, dtype=np.float64)
        if not hasattr(sp, "both"):
            self._live = np.ones(x.size - 1, dtype=bool)
            self._top = np.full(x.size - 1, np.inf)
            return self._live
        s1r, s2r = sp.both(x[:-1], "right")
        s1l, s2l = sp.both(x[1:], "left")
        top = np.maximum(s1r, s1l)
        turn = s2r * s2l < 0.0
        if turn.any():
            yv = x[:-1][turn] + (x[1:][turn] - x[:-1][turn]) * s2r[turn] / (
                s2r[turn] - s2l[turn]
            )
            top[turn] = np.maximum(top[turn], sp(yv, 1))
        top = np.where(np.isnan(top), np.inf, top)
        self._top = top = np.maximum(
            top, np.maximum(np.append(top[1:], top[-1]), np.insert(top[:-1], 0, top[0]))
        )
        if self.SGN:
            a2 = self.RA * self.RA
            xe = np.exp(2.0 * x[np.minimum(np.arange(top.size) + 2, x.size - 1)])
            den = a2 + self.SGN * xe
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                g = np.where(den > 0.0, 2.0 * xe * xe / (a2 * den), np.inf)
            g = np.where(np.isnan(g), np.inf, g)
        else:
            g = np.zeros(top.size)
        with np.errstate(invalid="ignore"):
            total = top + g
            self._live = ~(total < -1e-9 * (1.0 + np.abs(top) + g))
        return self._live

    def _needed_knots(self, sp: Any) -> Array:
        """Return the knots next to a knot interval `_live_bound` keeps."""
        can = self._live_bound(sp)
        near = np.zeros(can.size + 1, dtype=bool)
        near[:-1] |= can
        near[1:] |= can
        return np.flatnonzero(near)

    def _f_extrema_chunk(
        self,
        sp: Any,
        kn: Array,
        v: Array,
        lo: Array,
        needed: Array,
        i0: Array,
        n_kn: Array,
    ) -> list[Array]:
        # `f_extrema` at the radii v, element by element as at one radius: the
        # starting points, the bisection, the runs and their dedup carry the
        # index of their radius; of the knots between, only those `needed`
        tmax = np.arccosh(self.R_t / v)
        a2, sg = self.RA * self.RA, self.SGN
        sp_left = getattr(sp, "left", sp)

        def parts(
            t: Array, s: Array, end: bool = False
        ) -> tuple[Array, Array, Array, Array]:
            # the end's rounding to the last knot (one reading of Psi); a line of
            # sight outside the knots is refused above
            r2d, r2s = v[s], v[s] * v[s]
            y = np.clip(
                np.log(r2d) + np.log1p(2.0 * np.sinh(0.5 * t) ** 2), sp.x[0], sp.x[-1]
            )
            x = (r2d * np.cosh(t)) ** 2
            gam = (
                2.0 * x * r2s / ((a2 + sg * (x - r2s)) * (a2 + sg * x))
                if sg
                else np.zeros_like(t)
            )
            if hasattr(sp, "both"):
                s1, s2 = sp.both(y, "left" if end else "right")
                return s1, s2, gam, y
            read = sp_left if end else sp
            return read(y, 1), read(y, 2), gam, y

        m_r = v.size
        idx = np.arange(m_r)
        # the starting points: 0, t_max, the knots between, and r_a's point,
        # each with the knot interval it starts
        s_kn = np.repeat(idx, n_kn)
        pos = np.arange(s_kn.size) - np.repeat(np.cumsum(n_kn) - n_kn, n_kn)
        k_kn = needed[np.repeat(i0, n_kn) + pos]
        j_t = int(np.searchsorted(kn, self.R_t, side="right")) - 1
        pts = [np.zeros(m_r), tmax, np.arccosh(kn[k_kn] / v[s_kn])]
        segs = [idx, idx, s_kn]
        knot = [lo - 1, np.full(m_r, j_t), k_kn]
        if sg > 0.0:
            r2s = v * v
            has = a2 > r2s
            xp = np.full(m_r, np.nan)
            xp[has] = self.RA * np.sqrt(a2 - r2s[has])
            has &= (r2s < xp) & (xp < self.R_t**2)
            pts.append(np.arccosh(np.sqrt(xp[has]) / v[has]))
            segs.append(idx[has])
            knot.append(np.searchsorted(kn, np.sqrt(xp[has]), side="right") - 1)
        te, st = np.concatenate(pts), np.concatenate(segs)
        kt = np.concatenate(knot)
        o = np.lexsort((te, st))
        te, st, kt = te[o], st[o], kt[o]
        keep = np.concatenate([[True], (st[1:] != st[:-1]) | (te[1:] != te[:-1])])
        te, st, kt = te[keep], st[keep], kt[keep]
        pair = st[1:] == st[:-1]
        a, b, s = te[:-1][pair], te[1:][pair], st[:-1][pair]
        # an interval whose knot interval no radius can make live is one the
        # first bound test below drops: not formed (3-194)
        can = self._live_bound(sp)
        ja = np.clip(kt[:-1][pair], 0, can.size - 1)
        alive = can[ja]
        a, b, s, ja = a[alive], b[alive], s[alive], ja[alive]
        if sg and a.size:
            # gamma at this radius: monotone on each side of its turn x_t
            # (r_a's point, SGN > 0) or increasing below a^2 (SGN < 0), so its
            # largest on the knot interval and its neighbors is at an end or at
            # x_t; an interval that holds a pole of gamma is kept
            top_nb = self._top if self._top is not None else np.full(can.size, np.inf)
            r2s = v[s] * v[s]
            xl = kn[np.clip(ja - 1, 0, kn.size - 1)] ** 2
            xh = kn[np.clip(ja + 2, 0, kn.size - 1)] ** 2

            def gam(x: Array) -> Array:
                den = (a2 + sg * (x - r2s)) * (a2 + sg * x)
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = 2.0 * x * r2s / den
                return np.where((den > 0.0) & np.isfinite(g), g, np.inf)

            if sg > 0.0:
                xt = np.sqrt(np.maximum((a2 - r2s) * a2, 0.0))
                gmax = np.where(xh <= xt, gam(xh), np.where(xl >= xt, gam(xl), gam(xt)))
            else:
                # increasing below a^2; poles at a^2 and a^2 + R_2D^2, past the
                # second positive and decreasing (q <= 0 there)
                gmax = np.where(
                    xh < a2,
                    gam(xh),
                    np.where(xl > a2 + r2s, gam(xl), np.inf),
                )
            top = top_nb[ja]
            with np.errstate(invalid="ignore"):
                on = ~(top + gmax < -1e-9 * (1.0 + np.abs(top) + gmax))
            a, b, s = a[on], b[on], s[on]
        fin_a, fin_b, fin_s = np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
        n_bisect = self.rules.n_bisect
        for lev in range(n_bisect + 1):
            s1a, s2a, ga, ya = parts(a, s)
            s1b, s2b, gb, yb = parts(b, s, end=True)
            lo_s, hi_s = np.minimum(s1a, s1b), np.maximum(s1a, s1b)
            vx = s2a * s2b < 0.0
            if vx.any():
                sv = sp(ya[vx] + (yb[vx] - ya[vx]) * s2a[vx] / (s2a[vx] - s2b[vx]), 1)
                lo_s[vx], hi_s[vx] = np.minimum(lo_s[vx], sv), np.maximum(hi_s[vx], sv)
            live = (lo_s + np.minimum(ga, gb) <= 0.0) & (
                hi_s + np.maximum(ga, gb) >= 0.0
            )
            a, b, s = a[live], b[live], s[live]
            if not a.size or lev == n_bisect:
                break
            m = 0.5 * (a + b)
            # an interval its midpoint no longer splits is at t's resolution
            ok = (m > a) & (m < b)
            fin_a = np.concatenate([fin_a, a[~ok]])
            fin_b = np.concatenate([fin_b, b[~ok]])
            fin_s = np.concatenate([fin_s, s[~ok]])
            a, b, m, s = a[ok], b[ok], m[ok], s[ok]
            a, b, s = (
                np.concatenate([a, m]),
                np.concatenate([m, b]),
                np.concatenate([s, s]),
            )
        a, b = np.concatenate([a, fin_a]), np.concatenate([b, fin_b])
        s = np.concatenate([s, fin_s])
        o = np.lexsort((a, s))
        a, b, s = a[o], b[o], s[o]
        cut = np.flatnonzero((s[1:] != s[:-1]) | (a[1:] > b[:-1])) + 1
        run = np.concatenate([[0], cut, [a.size]])
        t = 0.5 * (a[run[:-1]] + b[run[1:] - 1]) if a.size else a
        ts = s[run[:-1]] if a.size else s
        if t.size:
            keep = np.concatenate(
                [[True], (ts[1:] != ts[:-1]) | (np.diff(t) > 1e-12 * (1.0 + t[1:]))]
            )
            t, ts = t[keep], ts[keep]
        keep = (t > 0.0) & (t < tmax[ts])
        t, ts = t[keep], ts[keep]
        return np.split(t, np.cumsum(np.bincount(ts, minlength=m_r))[:-1])

    def v_esc(self, r2d: float) -> float:
        """Return the escape speed at the row's own radius, (2 P(R_2D))^(1/2)."""
        return float(np.sqrt(2.0 * self.psi(np.array([r2d]))[0]))

    def v_edge_many(self, r2d_arr: ArrayLike) -> Array:
        """Return V_edge at the line of sight's end R = R_t; NaN where the end
        is dead."""
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        if not self.model.edge_alive:
            return np.full(r2.shape, np.nan)
        rt = np.full(r2.shape, self.R_t)
        q = self.q_of(rt)
        v2 = 2.0 * self.psi(rt) * (r2 * r2 + q * (rt * rt - r2 * r2)) / (q * rt * rt)
        return np.sqrt(np.maximum(v2, 0.0))

    def v_c_many(self, r2d_arr: ArrayLike) -> Array:
        """Return V_c = (2 (P0 - P(R_2D)))^(1/2); NaN where V_c is no
        break."""
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        if not self.model.vc_alive:
            return np.full(r2.shape, np.nan)
        return np.sqrt(2.0 * np.maximum(self.model.p0_top - self.psi(r2), 0.0))

    # The breaks and the layout of a row.

    def breaks(self, r2d: float) -> tuple[list[float], float, float, float, float, int]:
        """Return (br, v_esc, V_k, V_end, V_rad, code): br the row's breaks
        [0, interior breaks increasing, V_end] and code its layout.

        Raises
        ------
        ResultError
            Where the decided breaks do not lie in order inside (0, V_end).
        """
        vesc = self.v_esc(r2d)
        vk, vend, vr = (float(x[0]) for x in self.breaks_many(np.array([float(r2d)])))
        ve = float(self.v_edge_many(np.array([float(r2d)]))[0])
        vc = float(self.v_c_many(np.array([float(r2d)]))[0])
        code = int(self.layout_many(np.array([float(r2d)]))[0])
        br = (
            [0.0]
            + [{"k": vk, "r": vr, "e": ve, "c": vc}[s] for s in LAYOUT_SEQ[code]]
            + [vend]
        )
        if not np.all(np.diff(br) > 0.0):
            raise ResultError(
                f"At R_2D = {r2d:.9e} layout code {code}, decided on the scan of "
                f"the breaks, does not order this row's own breaks inside (0, "
                f"V_end): {br} [3-76]."
            )
        return br, vesc, vk, vend, vr, code

    def panels(
        self, r2d: float
    ) -> tuple[Array, Array, list[float], float, float, float, float, int]:
        """Return the row's V >= 0 nodes and weights, ordered (panel,
        sub-panel, node), and its breaks: (V, w, br, v_esc, V_k, V_end, V_rad,
        code). Each panel's end edges are the break values."""
        br, vesc, vk, vend, vr, code = self.breaks(r2d)
        xg, wg = gl01(self.rules.vp_nodes)
        vn, wn = [], []
        units = self.lattice.panel_units(code)
        for i in range(len(br) - 1):
            e = br[i] + (br[i + 1] - br[i]) * units[i]
            e[0], e[-1] = br[i], br[i + 1]
            lo, hi = e[:-1], e[1:]
            vn.append((lo[:, None] + (hi - lo)[:, None] * xg).ravel())
            wn.append(((hi - lo)[:, None] * wg).ravel())
        return np.concatenate(vn), np.concatenate(wn), br, vesc, vk, vend, vr, code

    def breaks_many(
        self,
        r2d_arr: ArrayLike,
        with_t: bool = False,
        bracket: tuple[Array, Array, Array, Array] | None = None,
    ) -> tuple[Array, ...]:
        """Return (V_k, V_end, V_rad) at projected radii; with `with_t` also
        where the two maxima sit.

        V_end is the largest of f = 2 P (R_2D^2 + q u^2) / (q R^2) at t = 0, at
        the interior extrema of `f_extrema` (each to rounding) and at the end
        of the line of sight: one path at the rows and off them. V_rad is a
        coarse scan in t and brk_zoom local zooms around its maximum, inside
        `bracket` (lo_end, hi_end, lo_rad, hi_rad) where given (its first two
        are not read)."""
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        tmax = np.arccosh(np.maximum(self.R_t / r2, 1.0))
        rr = self.rules

        def f_end(t: Array) -> Array:
            r, u = r2[:, None] * np.cosh(t), r2[:, None] * np.sinh(t)
            q = self.q_of(r)
            return (
                2.0
                * self.psi(r.ravel()).reshape(r.shape)
                * (r2[:, None] ** 2 + q * u * u)
                / (q * r * r)
            )

        def f_end_many(r2d: Array, t: Array) -> Array:
            # f at each (R_2D, t) pair, as at one radius
            r, u = r2d * np.cosh(t), r2d * np.sinh(t)
            q = self.q_of(r)
            return 2.0 * self.psi(r) * (r2d * r2d + q * u * u) / (q * r * r)

        def f_rad(t: Array) -> Array:
            r, u = r2[:, None] * np.cosh(t), r2[:, None] * np.sinh(t)
            return (
                2.0
                * np.maximum(self.psi(r.ravel()).reshape(r.shape), 0.0)
                * (u / r) ** 2
            )

        def vmax(f: Callable[[Array], Array], br: Any) -> tuple[Array, Array]:
            if br is None:
                lo, hi = np.zeros_like(tmax), tmax.copy()
                n = rr.brk_scan
            else:
                lo, hi = np.clip(br[0], 0.0, tmax), np.clip(br[1], 0.0, tmax)
                n = 41
            rows = np.arange(len(r2))
            for _ in range(rr.brk_zoom + 1):
                t = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, n)[None, :]
                v = f(t)
                k = np.argmax(v, axis=1)
                lo, hi = (
                    t[rows, np.maximum(k - 1, 0)],
                    t[rows, np.minimum(k + 1, n - 1)],
                )
                best, tb = v[rows, k], t[rows, k]
                n = 41
            return np.sqrt(np.maximum(best, 0.0)), tb

        vk = np.sqrt(np.maximum(f_end(np.zeros((len(r2), 1)))[:, 0], 0.0))
        # V_end: f at 0, at each radius's extrema and at its end, all radii at
        # once; the first largest a radius, NaN first as np.argmax takes it
        ext = self.f_extrema_many(r2)
        pieces: list[Array] = []
        for e, tm in zip(ext, tmax.tolist(), strict=True):
            pieces.extend((_T_ZERO, e, np.array([tm])))
        t = np.concatenate(pieces) if pieces else np.zeros(0)
        n_t = np.array([e.size + 2 for e in ext], dtype=np.int64)
        f = f_end_many(np.repeat(r2, n_t), t)
        first = np.cumsum(n_t) - n_t
        at = np.arange(f.size)
        nan = np.isnan(f)
        f_num = np.where(nan, -np.inf, f)
        top = np.maximum.reduceat(f_num, first) if f.size else f_num
        k = (
            np.minimum.reduceat(
                np.where(f_num == np.repeat(top, n_t), at, f.size), first
            )
            if f.size
            else at
        )
        k_nan = np.minimum.reduceat(np.where(nan, at, f.size), first) if f.size else at
        k = np.where(k_nan < f.size, k_nan, k)
        best = f[k]
        vend = np.sqrt(np.where(0.0 > best, 0.0, best))
        t_end = t[k]
        if self.lfac:
            vr, t_rad = vmax(f_rad, None if bracket is None else bracket[2:])
        else:
            vr = t_rad = np.full(len(r2), np.nan)
        return (vk, vend, vr, t_end, t_rad) if with_t else (vk, vend, vr)

    def layout_many(self, r2d_arr: ArrayLike) -> Array:
        """Return the layout at projected radii: V_k and V_rad where their
        bounds keep them, V_edge and V_c, each where the lattice resolves its
        panel."""
        vk, vend, vr = self.breaks_many(r2d_arr)
        k_, r_, e_, c_ = self.lattice.resolved_breaks(
            self.vk_break(r2d_arr, vk),
            vend,
            self.vrad_break(r2d_arr, vr),
            self.v_edge_many(r2d_arr),
            self.v_c_many(r2d_arr),
        )
        return self.lattice.layout_code(k_, vend, r_, e_, c_)

    def raw_layout_many(self, r2d_arr: ArrayLike) -> Array:
        """Return the layout of the breaks themselves, with the floor only."""
        vk, vend, vr = self.breaks_many(r2d_arr)
        return self.lattice.layout_code(
            vk, vend, vr, self.v_edge_many(r2d_arr), self.v_c_many(r2d_arr)
        )

    def vrad_break(self, r2d_arr: ArrayLike, vr: Array) -> Array:
        """Return V_rad inside the kept ranges, NaN elsewhere."""
        if self.vrad_kept is None:
            return vr
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        inside = np.zeros(r2.shape, dtype=bool)
        for a, b in self.vrad_kept:
            inside |= (r2 >= a) & (r2 <= b)
        return np.where(inside, vr, np.nan)

    def vk_break(self, r2d_arr: ArrayLike, vk: Array) -> Array:
        """Return V_k inside the kept ranges, NaN elsewhere."""
        if self.vk_kept is None:
            return vk
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        inside = np.zeros(r2.shape, dtype=bool)
        for a, b in self.vk_kept:
            inside |= (r2 >= a) & (r2 <= b)
        return np.where(inside, vk, np.nan)

    def row_test(self, r2d: float) -> tuple[bool, float, str]:
        """Return (stored, node gap, why not) for a row as it would be
        stored: its nodes apart by RB_MIN_GAP of their own values, and no
        graded side asking for more layers than doubles resolve."""
        try:
            _, _, br, _, _, _, _, code = self.panels(float(r2d))
        except ResultError as e:
            if "[3-76]" not in str(e):
                raise
            return False, 0.0, "a decided break on or past its own V_end [3-76]"
        gap = float(
            np.min(
                self.lattice.lattice_gaps(
                    np.array([br], dtype=np.float64),
                    self.lattice.panel_units(int(code)),
                )[0]
            )
        )
        if gap < RB_MIN_GAP:
            return False, gap, f"node gap {gap:.3e} below RB_MIN_GAP"
        nm = (
            ["0"]
            + [
                {"k": "V_k", "r": "V_rad", "e": "V_edge", "c": "V_c"}[s]
                for s in LAYOUT_SEQ[int(code)]
            ]
            + ["V_end"]
        )
        for i, (lo, hi) in enumerate(self.lattice.panel_layers(code)):
            w = br[i + 1] - br[i]
            for v, need, side in ((br[i], lo, nm[i]), (br[i + 1], hi, nm[i + 1])):
                if v > 0.0 and l_max(w / v, self.rules) < need:
                    return (
                        False,
                        gap,
                        f"panel [{nm[i]}, {nm[i + 1]}] of width {w / v:.3e} of {side} "
                        f"asks for {need} layers toward it, doubles resolve "
                        f"{l_max(w / v, self.rules)}",
                    )
        return True, gap, ""

    def node_crossings(self, rs: Array, rb: Array) -> tuple[Array, list[str]]:
        """Return (radii, labels): where a merged break crosses a node of its
        row's lattice, each an R-break, bisected rb_bisect times between
        samples of one layout.

        A crossing splits where D's slope may be discontinuous there: a break
        merged by the resolution. A break merged by its own bound (V_rad or
        V_k outside its kept ranges, `vrad_break` and `vk_break`) has had its
        kink measured below the tolerance, and its crossing is smooth."""
        rr = self.rules
        lat = self.lattice
        xg = gl01(rr.vp_nodes)[0]
        units: dict[tuple[int, int], Array] = {}

        def unit_nodes(code: int, p: int) -> Array:
            if (code, p) not in units:
                u = lat.panel_units(int(code))[p]
                units[(code, p)] = (
                    u[:-1, None] + np.diff(u)[:, None] * xg[None, :]
                ).ravel()
            return units[(code, p)]

        def place(r2: Array) -> tuple[Array, dict[str, tuple[Array, Array]]]:
            vk, vend, vr = self.breaks_many(r2)
            ve, vc = self.v_edge_many(r2), self.v_c_many(r2)
            raw = {"k": vk, "r": vr, "e": ve, "c": vc}
            kept = dict(
                zip(
                    BREAK_LABELS,
                    lat.resolved_breaks(
                        self.vk_break(r2, vk), vend, self.vrad_break(r2, vr), ve, vc
                    ),
                    strict=True,
                )
            )
            code = np.atleast_1d(
                lat.layout_code(kept["k"], vend, kept["r"], kept["e"], kept["c"])
            )
            pos = {
                s: (np.full(len(r2), -1), np.full(len(r2), np.nan))
                for s in BREAK_LABELS
            }
            for i in range(len(r2)):
                seq = LAYOUT_SEQ[int(code[i])]
                br = np.array([0.0] + [kept[s][i] for s in seq] + [vend[i]])
                for s in BREAK_LABELS:
                    if s in seq or not np.isfinite(raw[s][i]):
                        continue
                    v = min(max(float(raw[s][i]), 0.0), float(vend[i]))
                    p = min(max(int(np.searchsorted(br, v)) - 1, 0), len(br) - 2)
                    pos[s][0][i], pos[s][1][i] = p, (v - br[p]) / (br[p + 1] - br[p])
            return code, pos

        ends = np.unique(
            np.concatenate([rs[:1], rb[(rb > rs[0]) & (rb < rs[-1])], rs[-1:]])
        )
        f = np.arange(1, rr.rb_interior + 1) / (rr.rb_interior + 1.0)
        samples = np.unique(
            np.concatenate(
                [
                    rs,
                    rb * (1.0 - RB_SAME),
                    rb * (1.0 + RB_SAME),
                    (
                        ends[:-1, None]
                        * (ends[1:, None] / ends[:-1, None]) ** f[None, :]
                    ).ravel(),
                ]
            )
        )
        samples = samples[(samples >= rs[0]) & (samples <= rs[-1])]
        code, pos = place(samples)
        lo_l: list[float] = []
        hi_l: list[float] = []
        lab_l: list[str] = []
        xi_l: list[float] = []
        for s in BREAK_LABELS:
            p, xi = pos[s]
            for a in np.flatnonzero(
                (code[:-1] == code[1:]) & (p[:-1] >= 0) & (p[:-1] == p[1:])
            ):
                un = unit_nodes(code[a], p[a])
                x0, x1 = sorted((xi[a], xi[a + 1]))
                for u in un[(un > x0) & (un < x1)]:
                    lo_l.append(samples[a])
                    hi_l.append(samples[a + 1])
                    lab_l.append(s)
                    xi_l.append(u)
        if not lo_l:
            return np.zeros(0), []
        lo, hi, node_xi, lab = (
            np.array(lo_l),
            np.array(hi_l),
            np.array(xi_l),
            np.array(lab_l),
        )
        below = np.array(
            [
                pos[s][1][np.searchsorted(samples, low)] < u
                for s, low, u in zip(lab, lo, node_xi, strict=True)
            ]
        )
        for _ in range(rr.rb_bisect):
            mid = np.sqrt(lo * hi)
            _, pm = place(mid)
            xm = np.array([pm[s][1][i] for i, s in enumerate(lab)])
            same = (xm < node_xi) == below
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        rx = np.sqrt(lo * hi)
        # the breaks merged by their own bound do not split
        one = np.ones(len(rx))
        kept = {"r": self.vrad_break(rx, one), "k": self.vk_break(rx, one)}
        split = np.array(
            [s not in kept or bool(np.isfinite(kept[s][i])) for i, s in enumerate(lab)],
            dtype=bool,
        )
        rx, lab = rx[split], lab[split]
        o = np.argsort(rx)
        return rx[o], [str(s) for s in lab[o]]

    def rows_with_r_breaks(self, r_base: ArrayLike) -> tuple[Array, Array]:
        """Return (radii, R-breaks): the base radii plus up to rb_layers rows
        approaching every R-break from each side, rt_layers toward the table's
        end and rows filling a stretch the layers leave short, each passing
        the row test.

        The R-breaks are the changes of the layout and of the raw layout (V_k's
        birth placed where V_end - V_k first exceeds BIRTH_TOL of V_end), and
        the radii where a merged break crosses a node, each bisected.
        """
        rr = self.rules

        def changes(
            fn: Callable[[Array], Array], rs: Array, lay: Array
        ) -> tuple[Array, Array, Array]:
            j_ = np.flatnonzero(lay[1:] != lay[:-1])
            lo, hi, c0, c1 = (
                rs[j_].copy(),
                rs[j_ + 1].copy(),
                lay[j_].copy(),
                lay[j_ + 1].copy(),
            )
            out_r, out_a, out_b = [], [], []
            for _ in range(len(LAYOUT_SEQ)):
                if lo.size == 0:
                    break
                top = hi.copy()
                for _ in range(rr.rb_bisect):
                    mid = np.sqrt(lo * hi)
                    same = fn(mid) == c0
                    lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
                ca = fn(hi)
                out_r.append(np.sqrt(lo * hi))
                out_a.append(c0)
                out_b.append(ca)
                more = ca != c1
                lo, hi, c0, c1 = hi[more], top[more], ca[more], c1[more]
            if not out_r:
                return (
                    np.zeros(0),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            rc, ac, bc = (
                np.concatenate(out_r),
                np.concatenate(out_a),
                np.concatenate(out_b),
            )
            o = np.argsort(rc)
            return rc[o], ac[o], bc[o]

        r = np.asarray(r_base, dtype=np.float64)
        f = np.arange(rr.rb_interior + 1) / (rr.rb_interior + 1.0)
        r_end = float(self.R_t)
        d0 = 1.0 - r[-1] / r_end
        ext = (
            r_end * (1.0 - d0 * 0.5 ** np.arange(1, rr.rb_ext + 1))
            if d0 > 0.0
            else np.zeros(0)
        )
        rs = np.concatenate(
            [
                (r[:-1, None] * (r[1:, None] / r[:-1, None]) ** f[None, :]).ravel(),
                r[-1:],
                ext,
            ]
        )
        lay = self.layout_many(rs)
        s = float(np.log(r[1] / r[0]))
        rb = changes(self.layout_many, rs, lay)[0]
        lr_ = self.raw_layout_many(rs)
        rr_list = []
        for rx0, ca_, cb_ in zip(*changes(self.raw_layout_many, rs, lr_), strict=True):
            a_, b_ = LAYOUT_SEQ[int(ca_)], LAYOUT_SEQ[int(cb_)]
            d_ = set(a_) ^ set(b_)
            if d_ == {"r"}:
                continue
            rx = float(rx0)
            if d_ == {"k"} and "k" in b_:

                def born(r2: Array) -> Array:
                    vk_b, vend_b, _ = self.breaks_many(r2)
                    return (vend_b - vk_b) > BIRTH_TOL * vend_b

                lo_b, hi_b = np.array([rx * np.exp(-BIRTH_SPAN)]), np.array([rx])
                if not born(lo_b)[0]:
                    for _ in range(rr.rb_bisect):
                        mid = np.sqrt(lo_b * hi_b)
                        up = born(mid)
                        lo_b, hi_b = np.where(up, lo_b, mid), np.where(up, mid, hi_b)
                    rx = float(np.sqrt(lo_b * hi_b)[0])
            rr_list.append(rx)
        if rr_list:
            allb = np.sort(np.concatenate([rb, np.asarray(rr_list, dtype=np.float64)]))
            rb = allb[np.concatenate([[True], np.diff(np.log(allb)) > RB_SAME])]
        rx_, lab_ = self.node_crossings(rs, rb)
        self.crossings = (rx_, lab_)
        if rx_.size:
            allb = np.sort(np.concatenate([rb, rx_]))
            rb = allb[np.concatenate([[True], np.diff(np.log(allb)) > RB_SAME])]
        r_top = float(rb[-1]) if (rb.size and rb[-1] > r[-1]) else r_end
        top_lay = r_top * (1.0 - s * 0.5 ** np.arange(1, rr.rt_layers + 1))
        top_lay = top_lay[(top_lay > r[0]) & (top_lay < r_top)]
        if rb.size == 0:
            cand = np.concatenate([r, top_lay])
            ok = self._rows_resolved(cand, np.isin(cand, r))
            return np.unique(cand[ok]), rb
        new = rb[:, None, None] * (
            1.0
            + np.array([-1.0, 1.0])[None, :, None]
            * s
            * 0.5 ** np.arange(1, rr.rb_layers + 1)[None, None, :]
        )
        side = np.broadcast_to(
            np.arange(len(rb))[:, None, None] + np.array([0, 1])[None, :, None],
            new.shape,
        )
        keep = (new > r[0]) & (new < r_top) & (np.searchsorted(rb, new) == side)
        cand = np.concatenate([r, new[keep], top_lay])
        fill = []
        for a_r, b_r in zip(rb[:-1], rb[1:], strict=True):
            if b_r > r_top:
                break
            if int(np.sum((cand > a_r) & (cand < b_r))) < RB_FILL:
                xi = 0.5 * (1.0 - np.cos(np.pi * (np.arange(RB_FILL) + 0.5) / RB_FILL))
                fill.append(a_r * (b_r / a_r) ** xi)
        if fill:
            cand = np.concatenate([cand, np.concatenate(fill)])
        ok = self._rows_resolved(cand, np.isin(cand, r))
        return np.unique(cand[ok]), rb

    def _rows_resolved(self, cand: Array, base: Array) -> Array:
        """Return which candidate rows pass the row test, naming the
        dropped."""
        res = [self.row_test(x) for x in cand]
        ok = np.array([t[0] for t in res], dtype=bool)
        if not np.all(ok):
            bad = np.flatnonzero(~ok)
            badb = bad[base[bad]]
            log.debug(
                "%d row(s) dropped (%d base row(s)): nodes closer than RB_MIN_GAP or a "
                "graded side asking for more layers than doubles resolve.",
                len(bad),
                len(badb),
            )
            for c in list(badb) + [c for c in bad if not base[c]][:12]:
                log.debug(
                    "%sR_2D = %.9e: %s",
                    "base row " if base[c] else "",
                    float(cand[c]),
                    res[c][2],
                )
        return ok

    # The row.

    def supports(self, r2d: float, va: Array) -> tuple[Array, Array, Array]:
        """Return flat arrays (jV, t_lo, t_hi) of the intervals in t where
        Q_c > 0, for every V of `va`: the sign test on 0, f's interior extrema
        and t_max, each edge bisected n_bisect times."""
        tmax = float(np.arccosh(self.R_t / r2d))
        tg = np.concatenate([[0.0], self.f_extrema(r2d), [tmax]])
        r, u = r2d * np.cosh(tg), r2d * np.sinh(tg)
        q = self.q_of(r)
        s = (
            self.psi(r)[None, :]
            - (va * va)[:, None] * (0.5 * q * r * r / (r2d * r2d + q * u * u))[None, :]
        )
        iv, ik = np.nonzero(np.sign(s[:, 1:]) != np.sign(s[:, :-1]))
        lo, hi, sl = tg[ik].copy(), tg[ik + 1].copy(), np.sign(s[iv, ik])
        for _ in range(self.rules.n_bisect):
            mid = 0.5 * (lo + hi)
            same = np.sign(self._qc_t(r2d, va[iv], mid)) == sl
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        cnt = np.bincount(iv, minlength=len(va))
        e = np.full((len(va), int(cnt.max(initial=0)) + 2), tmax)
        e[:, 0] = 0.0
        e[iv, 1 + np.arange(len(iv)) - (np.cumsum(cnt) - cnt)[iv]] = 0.5 * (lo + hi)
        a, b = e[:, :-1], e[:, 1:]
        pos = (b > a) & (
            self._qc_t(r2d, np.broadcast_to(va[:, None], a.shape), 0.5 * (a + b)) > 0.0
        )
        jv, k = np.nonzero(pos)
        return jv, a[jv, k], b[jv, k]

    def l0_split(
        self, r2d: float, va: Array, jv: Array, tl: Array, th: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Return the support intervals split at every root of s0(t) = Q_c(t),
        where the L = 0 point enters the s range, on the families with an L
        factor, and which intervals a root divided."""
        if not self.lfac or len(jv) == 0:
            return jv, tl, th, np.zeros(len(jv), dtype=bool)
        v = va[jv]
        live = v > 0.0
        if not live.any():
            return jv, tl, th, np.zeros(len(jv), dtype=bool)
        t_low = float(
            np.arcsinh(
                L_BELOW
                * float(v[live].min())
                / np.sqrt(2.0 * float(self.psi(np.array([r2d]))[0]))
            )
        )
        lo_t = np.where(tl > 0.0, tl, np.minimum(t_low, 0.5 * th))
        u01 = np.linspace(0.0, 1.0, self.rules.l_scan)
        ts = np.exp(
            np.log(lo_t)[:, None] + (np.log(th) - np.log(lo_t))[:, None] * u01[None, :]
        )

        def g(t: Array, vv: Array) -> Array:
            r, u = r2d * np.cosh(t), r2d * np.sinh(t)
            q = self.q_of(r)
            a = 1.0 + q * (u / r2d) ** 2
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ls0 = np.where(
                    u > 0.0, 2.0 * np.log(vv * r / u) - np.log(2.0 * a), np.inf
                )
                return ls0 - np.log(np.maximum(self._qc_t(r2d, vv, t), 1e-300))

        gg = g(ts, v[:, None])
        ok = np.isfinite(gg[:, 1:]) & np.isfinite(gg[:, :-1]) & live[:, None]
        ii, kk = np.nonzero(ok & (np.sign(gg[:, 1:]) != np.sign(gg[:, :-1])))
        if ii.size == 0:
            return jv, tl, th, np.zeros(len(jv), dtype=bool)
        lo, hi, sl = ts[ii, kk].copy(), ts[ii, kk + 1].copy(), np.sign(gg[ii, kk])
        for _ in range(self.rules.n_bisect):
            mid = 0.5 * (lo + hi)
            same = np.sign(g(mid, v[ii])) == sl
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        roots = 0.5 * (lo + hi)
        j_l, l_l, h_l, c_l = [], [], [], []
        for m in range(len(jv)):
            rt = np.sort(roots[ii == m])
            e = np.concatenate([[tl[m]], rt[(rt > tl[m]) & (rt < th[m])], [th[m]]])
            j_l.append(np.full(len(e) - 1, jv[m]))
            l_l.append(e[:-1])
            h_l.append(e[1:])
            c_l.append(np.full(len(e) - 1, len(e) > 2))
        return (
            np.concatenate(j_l),
            np.concatenate(l_l),
            np.concatenate(h_l),
            np.concatenate(c_l),
        )

    def _row_at(
        self,
        r2d: float,
        va: Array,
        jv: Array,
        tl: Array,
        th: Array,
        rule: tuple[int, ...],
        n_panels: int,
        n_lay: int = 0,
        cut: Array | None = None,
    ) -> Array:
        """Return D on the distinct |V| of `va` with `n_panels` composite
        Gauss panels in xi a support interval, the first of them replaced by
        `n_lay` layers graded toward the axis, under t = t_lo + (t_hi - t_lo)
        sin^2(pi xi / 2). The pieces of a split interval take half the
        count."""
        rr = self.rules
        if cut is not None and cut.any():
            d = self._row_at(
                r2d, va, jv[cut], tl[cut], th[cut], rule, max(n_panels // 2, 1), n_lay
            )
            if (~cut).any():
                d = d + self._row_at(
                    r2d, va, jv[~cut], tl[~cut], th[~cut], rule, n_panels, n_lay
                )
            return d
        xg, wg = gl01(rr.t_nodes)
        base = np.linspace(0.0, 1.0, n_panels + 1)
        if n_lay <= 0:
            e = base
        else:
            e = np.concatenate(
                [[0.0], base[1] * rr.t_ratio ** np.arange(n_lay - 1, -1, -1), base[2:]]
            )
        xi = (e[:-1, None] + np.diff(e)[:, None] * xg[None, :]).ravel()
        wxi = (np.diff(e)[:, None] * wg[None, :]).ravel()
        xs = np.sin(0.5 * np.pi * xi) ** 2
        ws = wxi * 0.5 * np.pi * np.sin(np.pi * xi)
        t = tl[:, None] + (th - tl)[:, None] * xs[None, :]
        wt = (th - tl)[:, None] * ws[None, :] * r2d * np.cosh(t)
        vp = np.broadcast_to(va[jv][:, None], t.shape).ravel()
        vals = self.plane(r2d, (r2d * np.sinh(t)).ravel(), vp, rule)
        return 2.0 * np.bincount(
            np.repeat(jv, xs.size), weights=wt.ravel() * vals, minlength=len(va)
        )

    def row(
        self,
        r2d: float,
        v: ArrayLike,
        rule: tuple[int, ...],
        report: dict[str, Any] | None = None,
        w: ArrayLike | None = None,
    ) -> Array:
        """Return D(R_2D, V) on `v` (any sign; evaluated on the distinct |V|).

        The row owns its rule: the panel count doubles, and the layers reach
        further toward the axis, until the change is within T_TOL in the row's
        own L1 norm and pointwise (with the row's weights `w`; else the largest
        change over the largest D), each node doubling only while its own
        change fails. `report` receives what the row needed.
        """
        rr = self.rules
        va, inv = np.unique(
            np.abs(np.asarray(v, dtype=np.float64)), return_inverse=True
        )
        jv, tl, th = self.supports(r2d, va)
        jv, tl, th, cut = self.l0_split(r2d, va, jv, tl, th)
        tm = float(th.max()) if th.size else 0.0
        # level 0's layers reach the knee t_a = arcsinh(R_a / R_2D) on Type I
        n0 = 0
        if self.SGN > 0.0 and np.isfinite(self.RA) and tm > 0.0:
            t_a = float(np.arcsinh(self.RA / r2d))
            while (
                tm
                * np.sin(
                    0.5 * np.pi * (1.0 / rr.t_panels) * rr.t_ratio ** max(n0 - 1, 0)
                )
                ** 2
                > t_a
                and n0 < 400
            ):
                n0 += 1
        n_p, n_lay = rr.t_panels, n0
        d = self._row_at(r2d, va, jv, tl, th, rule, n_p, n_lay, cut)
        change, converged = float("inf"), False
        change_pw, set_by, l1_prev, pw_prev = float("inf"), "", True, True
        wa = None
        if w is not None:
            wa = np.bincount(
                inv, weights=np.asarray(w, dtype=np.float64), minlength=len(va)
            )
        per_node = w is not None
        act = np.ones(len(va), dtype=bool)
        big = np.zeros(len(va), dtype=bool)
        for _ in range(rr.t_levels):
            n_p *= 2
            n_lay += rr.t_layers
            if per_node:
                sel = act[jv]
                d_new = d.copy()
                if sel.any():
                    d_new[act] = self._row_at(
                        r2d, va, jv[sel], tl[sel], th[sel], rule, n_p, n_lay, cut[sel]
                    )[act]
            else:
                d_new = self._row_at(r2d, va, jv, tl, th, rule, n_p, n_lay, cut)
            if wa is not None:
                scale = float(np.dot(wa, np.abs(d_new)))
                change = (
                    float(np.dot(wa, np.abs(d_new - d))) / scale if scale > 0.0 else 0.0
                )
                pk = float(np.max(np.abs(d_new))) if d_new.size else 0.0
                big = np.abs(d_new) > 1e-12 * pk
                change_pw = (
                    float(np.max(np.abs(d_new[big] - d[big]) / np.abs(d_new[big])))
                    if big.any()
                    else 0.0
                )
            else:
                scale = float(np.max(np.abs(d_new))) if d_new.size else 0.0
                change = (
                    float(np.max(np.abs(d_new - d))) / scale if scale > 0.0 else 0.0
                )
                change_pw = change
            if per_node:
                rel = np.zeros(len(va))
                rel[big] = np.abs(d_new[big] - d[big]) / np.abs(d_new[big])
                act &= big & (rel > T_TOL)
            d = d_new
            l1_ok, pw_ok = change <= T_TOL, change_pw <= T_TOL
            if l1_ok and pw_ok:
                converged = True
                set_by = (
                    "first"
                    if (l1_prev and pw_prev)
                    else "both"
                    if not (l1_prev or pw_prev)
                    else "l1"
                    if not l1_prev
                    else "pointwise"
                )
                break
            l1_prev, pw_prev = l1_ok, pw_ok
        if report is not None:
            xi_r = (
                (1.0 / n_p) * rr.t_ratio ** max(n_lay - 1, 0)
                if n_lay > 0
                else 1.0 / n_p
            )
            tm = float(th.max()) if th.size else 0.0
            report.update(
                panels=n_p,
                nodes=(n_p + max(n_lay - 1, 0)) * rr.t_nodes,
                change=change,
                converged=converged,
                tmax=tm,
                layers=n_lay,
                reach=tm * np.sin(0.5 * np.pi * xi_r) ** 2,
                t_a=float(np.arcsinh(self.RA / r2d)) if self.SGN else float("inf"),
                change_pw=change_pw,
                set_by=set_by,
                layers0=n0,
                intervals=int(len(jv)),
            )
        return d[inv]

    def plane(self, r2d: float, u: Array, v: Array, rule: tuple[int, ...]) -> Array:
        """Return the integral of F (L^2)^(-beta_c) over the velocity plane at
        the pairs (u, V)."""
        df = self._df()
        _, k0, ks, kq, m, nphi = rule
        r = np.sqrt(r2d * r2d + u * u)
        q = self.q_of(r)
        a = 1.0 + q * (u / r2d) ** 2
        qc = np.maximum(
            self.psi(r) - 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u), 0.0
        )
        jac = (r / r2d) / np.sqrt(a * q)
        if not self.lfac:
            return jac * 2.0 * np.pi * df.G(qc)
        v = np.broadcast_to(np.asarray(v, dtype=np.float64), np.shape(u))
        with np.errstate(divide="ignore", invalid="ignore"):
            s0 = np.where(u > 0.0, v * v * r * r / (2.0 * a * u * u), np.inf)
        split = np.where((s0 >= 0.0) & (s0 < qc), s0, 0.5 * qc)
        kp = self._s_layers(r2d, r, q, u, v, qc, split, m, k0)
        ka, kb = self._ks_layers(r2d, r, q, u, v, qc, split, s0, m, ks)
        key = (kp * 1000 + ka) * 1000 + kb
        out = np.empty(len(u))
        for k in np.unique(key):
            g = key == k
            i = int(np.argmax(g))
            out[g] = self._plane_s(
                r2d,
                u[g],
                v[g],
                r[g],
                q[g],
                a[g],
                qc[g],
                jac[g],
                split[g],
                int(kp[i]),
                int(ka[i]),
                int(kb[i]),
                kq,
                m,
                nphi,
            )
        return out

    def _plane_s(
        self,
        r2d: float,
        u: Array,
        v: Array,
        r: Array,
        q: Array,
        a: Array,
        qc: Array,
        jac: Array,
        split: Array,
        k0: int,
        ksa: int,
        ksb: int,
        kq: int,
        m: int,
        nphi: int,
    ) -> Array:
        """The (s, phi) quadrature of `plane` on pairs that share their
        layers: piece a on [0, split], piece b on [split, Q_c], piece b's
        first panel mapped where 0 < beta_c < 1 and its last where F's power at
        K = 0 asks (`EllipseDF.zero_map`)."""
        df = self._df()
        rr = self.rules
        sr = rr.s_ratio
        u1, u2 = seg_unit(k0, ksa, sr), seg_unit(ksb, kq, sr)
        xg, wg = gl01(m)
        n1, n2 = (len(u1) - 1) * m, (len(u2) - 1) * m
        xb = (u2[:-1, None] + np.diff(u2)[:, None] * xg).ravel()
        wb = (np.diff(u2)[:, None] * wg).ravel()
        if 0.0 < self.BC < 1.0:
            p = 1.0 / (1.0 - self.BC)
            d = float(u2[1] - u2[0])
            xb[:m] = u2[0] + d * xg**p
            wb[:m] = d * p * xg ** (p - 1.0) * wg
        omb = 1.0 - xb
        if df.zero_map(m, kq):
            p = 1.0 / (1.0 + df.p_zero)
            d = float(u2[-1] - u2[-2])
            lw = int(np.ceil(np.log(GOAL) / np.log(sr))) if df.k_zero != 0.0 else 0
            ew = np.concatenate([[0.0], sr ** np.arange(lw, -1, -1)])
            wn = (ew[:-1, None] + np.diff(ew)[:, None] * xg).ravel()
            ww = (np.diff(ew)[:, None] * wg).ravel()
            xb, wb, omb = xb[:-m], wb[:-m], omb[:-m]
            omb = np.concatenate([omb, d * wn**p])
            xb = np.concatenate([xb, 1.0 - d * wn**p])
            wb = np.concatenate([wb, d * p * wn ** (p - 1.0) * ww])
            n2 = xb.size
        sa = np.concatenate(
            [(u1[:-1, None] + np.diff(u1)[:, None] * xg).ravel(), np.ones(n2)]
        )
        ea = np.concatenate([[0.0], 0.5 * sr ** np.arange(k0, -1, -1)])
        eb = np.concatenate([[0.0], 0.5 * sr ** np.arange(ksa, -1, -1)])
        c1e = np.concatenate([1.0 - ea, eb[::-1][1:]])
        csa = np.concatenate(
            [
                (c1e[:-1, None] - (c1e[:-1] - c1e[1:])[:, None] * xg).ravel(),
                np.zeros(n2),
            ]
        )
        sb = np.concatenate([np.zeros(n1), xb])
        omb_all = np.concatenate([np.zeros(n1), omb])
        isb = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
        wa_ = np.concatenate([(np.diff(u1)[:, None] * wg).ravel(), np.zeros(n2)])
        wb_ = np.concatenate([np.zeros(n1), wb])
        ns = n1 + n2
        c1a = v * r / r2d / a
        c2a = u / r2d * np.sqrt(2.0 / a)
        c3a = 2.0 / q
        pre = jac * 2.0 * r ** (-2.0 * self.BC)
        out = np.empty(len(u))
        # the numpy rule forms a (nodes, angle nodes) block; the kernel forms
        # its own, so its route sizes a chunk by the s nodes alone
        step = max(
            1, rr.chunk // (ns * (1 if (self.lam_closed or rr.kernel) else nphi))
        )
        counts = tuple(rr.angle_counts)
        for i0 in range(0, len(u), step):
            sl = slice(i0, i0 + step)
            sp, rest = split[sl, None], (qc[sl] - split[sl])[:, None]
            s = sp * sa + rest * sb
            ws = sp * wa_ + rest * wb_
            fs = df.F(np.where(isb[None, :], rest * omb_all[None, :], qc[sl, None] - s))
            c1 = c1a[sl, None]
            c2 = c2a[sl, None] * np.sqrt(s)
            c3 = c3a[sl, None] * s
            kap = c1 * c2 + c3 - c2 * c2
            d2 = (c1 - c2) ** 2
            e2 = c2 * c2 - c3
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(kap > 0.0, np.sqrt(d2 / np.maximum(kap, 1e-300)), np.pi)
            np.clip(lam, 1e-15, np.pi, out=lam)
            with np.errstate(divide="ignore", invalid="ignore"):
                kpi = c3 - c1 * c2 - c2 * c2
                lpi = np.where(
                    kpi > 0.0, np.sqrt((c1 + c2) ** 2 / np.maximum(kpi, 1e-300)), np.pi
                )
            np.clip(lpi, 1e-15, np.pi, out=lpi)
            if self.lam_closed:
                with np.errstate(divide="ignore", invalid="ignore"):
                    s0 = np.where(
                        u[sl] > 0.0,
                        v[sl] * v[sl] * r[sl] * r[sl] / (2.0 * a[sl] * u[sl] * u[sl]),
                        np.inf,
                    )
                at0 = ((s0 >= 0.0) & (s0 < qc[sl]))[:, None]
                dm = np.where(
                    at0,
                    c2a[sl, None] ** 2
                    * np.where(isb[None, :], -rest * sb[None, :], sp * csa[None, :]),
                    c1 * c1 - c2 * c2,
                )
                e_ = (2.0 / (q[sl] * a[sl]))[:, None] * s
                lam_v = (
                    lam_half(c1, c3, e_, dm)
                    if self.BC > 0.0
                    else lam_mhalf(c1, c2, e_, dm)
                )
            else:
                lam_v = np.zeros(lam.shape)
                live = np.flatnonzero(ws > 0.0)
                f = [
                    np.broadcast_to(a_, lam.shape).ravel()[live]
                    for a_ in (c1, c2, c3, kap, d2, e2, lam, lpi)
                ]
                k = angle_count(
                    f[0],
                    f[1],
                    f[2],
                    ((2.0 / (q[sl] * a[sl]))[:, None] * s).ravel()[live],
                    counts,
                )
                for j, n in enumerate(counts + (nphi // 2,)):
                    g = np.flatnonzero(k == j)
                    if g.size:
                        args = (
                            f[0][g],
                            f[1][g],
                            f[2][g],
                            f[3][g],
                            f[4][g],
                            f[5][g],
                            f[6][g],
                            f[7][g],
                            n,
                        )
                        lam_v.flat[live[g]] = (
                            angle_rule(self.BC, *args)
                            if rr.kernel
                            else lam_rule(self.powm, *args)
                        )
                self.angle_counts += np.bincount(k, minlength=len(counts) + 1)
            out[sl] = pre[sl] * np.einsum(
                "ij,ij,ij->i", ws, fs, np.where(ws > 0.0, lam_v, 0.0)
            )
        return out
