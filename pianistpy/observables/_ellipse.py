r"""Compute rows of the line-of-sight velocity distribution by the ellipse route.

`EllipseRoute` computes the line-of-sight velocity distribution
:math:`D(R_{\rm 2D}, V)` (Chiang+(2026d), Eq. (B53)) row by row at projected
radii :math:`R_{\rm 2D}`. The public calls `pianistpy.losvd`,
`pianistpy.aperture_losvd` and `pianistpy.LosTable` read tables built from
the rows of `EllipseRoute`.

A point on the line of sight has depth u, radius
:math:`R = (R_{\rm 2D}^2 + u^2)^{1/2}` and :math:`q = 1 + s_A (R/R_{\rm a})^2`,
with :math:`R_{\rm a}` the anisotropy radius and :math:`s_A` = +1 for Type I,
-1 for Type II and 0 without :math:`R_{\rm a}`. On the plane of fixed
line-of-sight velocity V at the point, the level sets of the energy Q are
ellipses centered on the maximum

.. math::

    Q_c = \mathcal{P}(R) - \frac{q V^2 R^2}{2(R_{\rm 2D}^2 + q u^2)},

with :math:`\mathcal{P}` the relative potential. The integral of the DF F
over the plane is (Notes Eq. (T:6.2))

.. math::

    I(u) = \frac{R/R_{\rm 2D}}{\sqrt{A q}}
    \int_0^{Q_c} F(Q_c - s)\,\Lambda(s)\,{\rm d}s,
    \qquad A = 1 + q\,(u/R_{\rm 2D})^2,

with :math:`\Lambda(s)` the integral of :math:`(L^2)^{-\beta_{\rm c}}` around
the level set :math:`Q = Q_c - s`. Where the DF carries no L factor,
:math:`\Lambda = 2\pi`, and the s integral equals :math:`2\pi G(Q_c)`, with
:math:`G(Q) = \int_0^Q F\,{\rm d}K`. The distribution is

.. math::

    D(R_{\rm 2D}, V) = 2\int_0^{u_{\rm max}} I(u)\,{\rm d}u,

with :math:`u_{\rm max} = (R_{\rm t}^2 - R_{\rm 2D}^2)^{1/2}` and
:math:`R_{\rm t}` the radius where the lines of sight end. The panels of the
integral in u double until the row changes by at most :math:`10^{-6}`
(`T_TOL`). Notes Section T:6.4 describes the method.
"""

__all__ = ["EllipseRoute"]

from collections.abc import Callable
from typing import Any

import numpy as np

from ..exceptions import ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ..validation import TIER_BOUNDS
from ._angle import angle_count, angle_table, lam_half, lam_mhalf, lam_rule, power_of
from ._angle_rule import angle_rule
from ._ellipse_df import EllipseDF
from ._ellipse_rules import (
    BIRTH_SPAN,
    BIRTH_TOL,
    BREAK_PAST_END,
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

# The sign SGN in q = 1 + SGN (R / R_a)^2 for each anisotropy family.
_SGN = {
    "isotropic": 0.0,
    "constant": 0.0,
    "OM_I": 1.0,
    "COM_I": 1.0,
    "OM_II": -1.0,
    "COM_II": -1.0,
}
# The codes of the test that passed last when the panels in t of a row
# converged (`EllipseRoute.row`). Code 1: both tests passed at the first
# doubling. Code 2: the L1 test passed last. Code 3: the pointwise test passed
# last. Code 4: both tests passed at the same doubling after both failed at
# the doubling before. The readers of T_SET_BY write 0 for a row whose panels
# in t did not converge.
T_SET_BY = {"first": 1, "l1": 2, "pointwise": 3, "both": 4}
# The approximate number of starting intervals in one chunk of radii of
# `EllipseRoute.f_extrema_many`. The chunk size bounds the memory of the call.
_EXTREMA_CHUNK = 2_000_000
_T_ZERO = np.zeros(1)


def _velocities_kept(
    level: list[tuple[Any, ...]], keep: Array
) -> list[tuple[Any, ...]]:
    """Return the reads of one level of a row restricted to the velocities in `keep`.

    `level` holds the reads that `EllipseRoute._row_at` collects at one level.
    `keep` is a boolean mask over the distinct velocities of the row. Each read
    of F or G keeps the entries at the kept velocities. The read of the node
    pairs (kind ``"P"``) keeps the kept pairs in their order and renumbers
    the pairs' indices. The read of the node pairs also keeps the layer
    groups and the angle counts of the kept pairs, and drops a group or a
    count with no kept pair. Each read then equals the read that `_row_at`
    collects on the kept velocities alone.
    """
    out: list[tuple[Any, ...]] = []
    for kind, velocity, k, w in level:
        rows = keep[velocity]
        if not rows.any():
            continue
        if kind != "P":
            out.append((kind, velocity[rows], k[rows], w[rows]))
            continue
        # The new index of each kept pair, in the order of the kept pairs.
        new = np.cumsum(rows) - 1
        groups = []
        for at, layers in k["groups"]:
            on = rows[at]
            if on.any():
                groups.append((new[at[on]], layers))
        counts = []
        for at, chosen in k["counts"]:
            on = rows[at]
            if on.any():
                counts.append((new[at[on]], chosen[on]))
        pairs = dict(
            r2d=k["r2d"],
            u=k["u"][rows],
            v=k["v"][rows],
            rule=k["rule"],
            groups=groups,
            counts=counts,
        )
        out.append((kind, velocity[rows], pairs, w[rows]))
    return out


class EllipseRoute:
    r"""Rows of the line-of-sight velocity distribution by the ellipse route.

    A route places the velocity breaks of each row and the radii where the
    layout of the rows changes. A route with a DF also computes rows of
    :math:`D(R_{\rm 2D}, V)`.

    Parameters
    ----------
    psi : callable
        The relative potential at radii (`_ellipse_model.PotentialReading`).
        `psi` carries the attribute ``spline``, the derivatives of
        :math:`\ln\mathcal{P}` in ln R (`_ellipse_model.LogPotentialSlopes`).
        `psi` carries the attribute ``R_knots``, the first and the last radius
        of the potential's table. `psi` may carry the method ``depth``, the
        depth :math:`P_0 - \mathcal{P}` below the top energy.
    r_t : float
        The radius :math:`R_{\rm t}` where the lines of sight end
        (`_ellipse_model.floor_radius`).
    model : EllipseModel
        The model's constants.
    df : EllipseDF or None
        The DF. None builds a route that places velocity breaks only.
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
        # At beta_c = 1/2 and beta_c = -1/2 the angle integral has a closed
        # form, a complete elliptic integral.
        self.lam_closed = self.lfac and self.BC in (0.5, -0.5)
        self.angle_counts = np.zeros(len(rules.angle_counts) + 1, dtype=np.int64)
        if self.lfac and not self.lam_closed:
            angle_table()
        # The ranges of R_2D, shape (n, 2), where V_rad and V_k may be velocity
        # breaks. None means that no range was computed. With None, only the
        # tests of the lattice (Lattice.resolved_breaks) decide whether the
        # break is kept.
        self.vrad_kept: Array | None = None
        self.vk_kept: Array | None = None
        self._fext: dict[float, Array] = {}
        self._live: Array | None = None
        self._top: Array | None = None
        self.crossings: tuple[Array, list[str]] = (np.zeros(0), [])

    def _df(self) -> EllipseDF:
        """Return the route's DF.

        Raises
        ------
        ResultError
            If the route was built without a DF.
        """
        if self.df is None:
            raise ResultError(
                "The EllipseRoute was built without a DF, so the EllipseRoute "
                "places velocity breaks only."
            )
        return self.df

    # The geometry of a line of sight.

    def q_of(self, r: Array) -> Array:
        r"""Return :math:`q = 1 + s_A (r/R_{\rm a})^2` at the radii `r`.

        The sign :math:`s_A` is ``SGN``.
        """
        return 1.0 + self.SGN * (r / self.RA) ** 2 if self.SGN else np.ones_like(r)

    def rule(self, r2d: float) -> tuple[tuple[int, ...], bool]:
        """Return the rule of the row at `r2d` and whether the row is a cusp row.

        The row is of the cusp class where the table's top energy Q_top exceeds
        the potential at `r2d` by less than CUSP_FRAC times Q_top. A row of
        the cusp class takes ``rules.rule_cusp``. Any other row takes
        ``rules.rule_regular``.
        """
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
        """Return the layer count toward s = 0 of piece a for each node pair.

        Piece a covers [0, split] with ``h = split / 2``. Its innermost panel
        at s = 0 has width ``h s_ratio^K`` with K layers. The count is the
        least K whose innermost panel is no wider than the distance from
        s = 0 to the nearer of two singularities at s < 0, and at least
        `floor`. The first singularity is the top of F, at the distance
        ``(Q_end - Q_c) min(1, m / |kappa|)``. Here Q_end is the end energy of
        `EllipseDF`, and kappa is the exponent of F at the top. The second
        singularity is the pinch of the angle integral at ``s = -s_star``,
        with s_star the kinetic term ``q V^2 R^2 / (2 (R_2D^2 + q u^2))``.

        Raises
        ------
        ResultError
            If a node pair has no positive distance to a singularity at s < 0.
        """
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
                f"At R_2D = {r2d:.6e}, u = {float(u[i]):.6e} and V = "
                f"{float(v[i]):.6e}, the distance from s = 0 to the nearest "
                f"singularity at s < 0 is not positive (Q_c = {float(qc[i]):.15e}, "
                f"P0 = {df.Qend:.15e}, s_star = {float(s_star[i]):.6e})."
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
        r"""Return the layer counts toward the split of pieces a and b, per node pair.

        Piece a covers [0, split] and piece b covers [split, Q_c]. The split
        is the L = 0 point s0 where s0 lies in [0, Q_c), and Q_c / 2
        otherwise. A pair with s0 in [0, Q_c) is an L = 0 pair. An L = 0 pair
        with a split that is not positive is a V = 0 pair. Both counts are at
        least `floor`.

        The count ka of piece a is 0 except at an L = 0 pair that is not a
        V = 0 pair. At such a pair, ka is the larger of two counts. The first
        count is ``model.k_end``. The second count is the least K whose
        innermost panel of piece a at the split, of width
        ``0.5 split s_ratio^K``, is no wider than Q_c - split, the distance from
        the split to the end s = Q_c.

        The count kb of piece b is the largest of three counts. The first
        count is the least K whose innermost panel at the split, of width
        ``hb s_ratio^K`` with ``hb = (Q_c - split) / 2``, is no wider than the
        distance from the split to the nearest singularity below the split.
        That distance is ``(split + Q_end - Q_c) min(1, m / |kappa|)`` from the
        top of F, or ``split + s_star`` from the pinch of the angle integral
        where nearer. At a V = 0 pair only the top of F counts. The second
        count is 0 except at an L = 0 pair. At an L = 0 pair the second count
        is a shrink count plus the larger of ``model.k_map`` and the L = 0
        count. The shrink count is the least K whose innermost panel is no
        wider than the split, and 0 at a V = 0 pair. The L = 0 count is
        ``model.k_end_v0`` at a V = 0 pair and ``model.k_end`` otherwise. The
        third count is ``model.k_map``, which the map of the first panel of
        piece b needs where :math:`0 < \beta_{\rm c} < 1`.

        Raises
        ------
        ResultError
            If a node pair has no positive distance to a singularity below the
            split.
        """
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
                f"At R_2D = {r2d:.6e}, u = {float(u[i]):.6e} and V = "
                f"{float(v[i]):.6e}, the distance from the split to the nearest "
                f"singularity beyond the split is not positive (Q_c = "
                f"{float(qc[i]):.15e}, split = {float(split[i]):.6e}, P0 = "
                f"{df.Qend:.15e}, s_star = {float(s_star[i]):.6e})."
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
        """Return Q_c at velocity `v` at ``u = R_2D sinh t`` on the line of sight."""
        r, u = r2d * np.cosh(t), r2d * np.sinh(t)
        q = self.q_of(r)
        return self.psi(r) - 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)

    def f_extrema(self, r2d: float) -> Array:
        r"""Return the interior local extrema in t of f along the line of sight.

        The function is ``f = 2 P (R_2D^2 + q u^2) / (q R^2)``, with P the
        relative potential, ``u = R_2D sinh t`` and ``R = R_2D cosh t``. A
        velocity V has :math:`Q_c > 0` at t exactly where :math:`V^2 < f(t)`.
        The extrema are returned sorted, each located by bisection.
        `supports` places the extrema among the points of its sign test.
        Between two neighboring points of that test, f is monotone, so each
        step of the sign test holds at most one edge of the support at any V.

        The extrema of f are the zeros of ``h(y) = s'(y) + gamma(y)``, with
        ``y = ln R``, ``s = ln P`` and gamma the derivative in y of the
        remaining factors of ln f. On each knot interval of the potential, s'
        is a quadratic in y whose range is exact, and gamma is monotone on
        each side of its turning point. An interval whose range of h excludes
        0 holds no extremum. Every other interval is halved up to
        ``rules.n_bisect`` times. The upper end of an interval reads the
        derivatives from the interval below the end (``spline.left``),
        because the second derivative of the Hermite interpolant jumps at the
        rows. The result is cached for each :math:`R_{\rm 2D}`.

        Raises
        ------
        ResultError
            If the potential has no ``spline``, or the line of sight leaves
            the knots of the potential (`f_extrema_many`).
        """
        return self.f_extrema_many(np.array([float(r2d)]))[0]

    def f_extrema_many(self, r2d_arr: ArrayLike) -> list[Array]:
        r"""Return `f_extrema` at each of the projected radii `r2d_arr`.

        The radii absent from the cache are computed together, in chunks of about
        `_EXTREMA_CHUNK` starting intervals. Each interval carries the index
        of its radius. Every value equals the value that `f_extrema` computes
        for one radius at a time.

        Raises
        ------
        ResultError
            If the potential has no ``spline``. Also if a radius lies below
            the first knot of the potential, or :math:`R_{\rm t}` lies above the last
            knot.
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
                "The potential has no spline (psi.spline), and the bound on the "
                "potential's extrema along the line of sight needs the potential's "
                "spline."
            )
        r_lo, r_hi = self.psi.R_knots  # type: ignore[attr-defined]
        for r2d in todo:
            if r2d < r_lo or self.R_t > r_hi:
                raise ResultError(
                    f"At R_2D = {r2d:.17g} and R_t = {self.R_t:.17g}, the line of "
                    f"sight leaves the range [{r_lo:.17g}, {r_hi:.17g}] of the "
                    f"potential's knots. Outside the knots, the potential's spline is "
                    f"held constant, so the bound on the potential's extrema along "
                    f"the line of sight does not hold."
                )
        vals = np.array(list(todo), dtype=np.float64)
        kn = np.exp(sp.x)
        lo = np.searchsorted(kn, vals, side="right")
        hi = np.maximum(np.searchsorted(kn, self.R_t, side="left"), lo)
        # A radius needs the knots between R_2D and R_t that border a knot
        # interval that `_live_bound` keeps.
        needed = self._needed_knots(sp)
        i0 = np.searchsorted(needed, lo)
        n_kn = np.searchsorted(needed, hi) - i0
        # Each chunk takes radii until the chunk's starting intervals number
        # about _EXTREMA_CHUNK.
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
        r"""Return, per knot interval, whether an extremum of f may lie inside.

        The first bound test of `f_extrema` keeps an interval only where
        ``max s' + max gamma >= 0`` on the interval. On a knot interval, s' is
        the quadratic of the Hermite interpolant, whose maximum lies at an end
        or at the zero of s'' inside the interval. At any
        :math:`R_{\rm 2D}`, gamma is at most
        ``G(x) = 2 x^2 / (a^2 (a^2 + SGN x))``, with ``x = R^2`` and
        ``a = R_a``, because gamma grows with ``R_2D^2`` up to
        ``R_2D^2 = x``. G is 0 where SGN = 0.

        A knot interval is kept unless the sum of two bounds is below
        ``-1e-9 (1 + |max s'| + G)``. The first bound is the maximum of s'
        over the interval and its two neighbors. The second bound is G at the
        upper end of the next interval. The neighbors cover reads that fall
        across a knot by rounding. A knot interval that fails the test holds
        no interval that `f_extrema` keeps at any radius, so dropping the knot
        interval changes no value. Every knot interval is kept where the
        spline has no method ``both``. The result is cached on the route.
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
        """Return the indices of the knots next to a knot interval that is kept.

        A knot interval is kept where `_live_bound` is True.
        """
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
        # Compute `f_extrema` at the radii v together. Every element equals the
        # element computed at its radius alone. The starting points, the
        # bisection, the merged runs and the removal of duplicates carry the
        # index of their radius. Only the knots in `needed` start intervals.
        tmax = np.arccosh(self.R_t / v)
        a2, sg = self.RA * self.RA, self.SGN
        sp_left = getattr(sp, "left", sp)

        def parts(
            t: Array, s: Array, end: bool = False
        ) -> tuple[Array, Array, Array, Array]:
            # The clip to the knots takes the rounding of the end point to the
            # last knot: R_2D cosh(arccosh(R_t / R_2D)) can exceed R_t by one
            # unit in the last place. A line of sight that leaves the knots
            # raises ResultError in `f_extrema_many`.
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
        # The starting points are t = 0, t_max, the needed knots between them,
        # and the turning point of gamma where SGN > 0. Each starting point
        # carries the index of the knot interval that the point starts.
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
        # An interval inside a knot interval that `_live_bound` drops would fail
        # the first bound test below. Such an interval is not formed.
        can = self._live_bound(sp)
        ja = np.clip(kt[:-1][pair], 0, can.size - 1)
        alive = can[ja]
        a, b, s, ja = a[alive], b[alive], s[alive], ja[alive]
        if sg and a.size:
            # At the interval's own radius, gamma is monotone on each side of
            # its turning point x_t where SGN > 0, and increases below a^2
            # where SGN < 0. The largest gamma over the knot interval and its
            # neighbors therefore lies at an end or at x_t. An interval that
            # holds a pole of gamma is kept.
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
                # Where SGN < 0, gamma increases for x below a^2 and has poles at
                # x = a^2 and x = a^2 + R_2D^2. Above a^2 + R_2D^2, gamma is
                # positive and decreasing, and q <= 0.
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
            # An interval whose midpoint equals an end in doubles is final.
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
        """Return the escape velocity ``(2 P(R_2D))^(1/2)`` at the radius `r2d`."""
        return float(np.sqrt(2.0 * self.psi(np.array([r2d]))[0]))

    def v_edge_many(self, r2d_arr: ArrayLike) -> Array:
        r"""Return the velocity V_edge at the end of each line of sight.

        V_edge is :math:`f^{1/2}` at the end of the line of sight,
        :math:`R = R_{\rm t}`, with f the function of `f_extrema`. Every value
        is NaN where ``model.edge_alive`` is False.
        """
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        if not self.model.edge_alive:
            return np.full(r2.shape, np.nan)
        rt = np.full(r2.shape, self.R_t)
        q = self.q_of(rt)
        v2 = 2.0 * self.psi(rt) * (r2 * r2 + q * (rt * rt - r2 * r2)) / (q * rt * rt)
        return np.sqrt(np.maximum(v2, 0.0))

    def v_c_many(self, r2d_arr: ArrayLike) -> Array:
        """Return the velocity ``V_c = (2 (P0 - P(R_2D)))^(1/2)`` at projected radii.

        P0 is ``model.p0_top``. Every value is NaN where ``model.vc_alive`` is
        False, which means that V_c is no velocity break.
        """
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        if not self.model.vc_alive:
            return np.full(r2.shape, np.nan)
        return np.sqrt(2.0 * np.maximum(self.model.p0_top - self.psi(r2), 0.0))

    # The breaks and the layout of a row.

    def breaks(self, r2d: float) -> tuple[list[float], float, float, float, float, int]:
        """Return the velocity breaks and the layout of the row at `r2d`.

        The result is the tuple (br, v_esc, V_k, V_end, V_rad, code). The list
        br holds 0, the interior breaks of the layout in increasing order, and
        V_end. The integer code is the row's layout code (`layout_many`).

        Raises
        ------
        ResultError
            If the breaks of br do not increase strictly from 0 to V_end. The
            message carries the marker BREAK_PAST_END.
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
                f"At R_2D = {r2d:.9e}, the layout code {code}, chosen on the scan of "
                f"the breaks, does not put the velocity breaks at R_2D = {r2d:.9e} "
                f"in increasing order inside (0, V_end); the breaks are {br} "
                f"{BREAK_PAST_END}."
            )
        return br, vesc, vk, vend, vr, code

    def panels(
        self, r2d: float
    ) -> tuple[Array, Array, list[float], float, float, float, float, int]:
        r"""Return the velocity nodes and weights of the row at `r2d`, and its breaks.

        The result is the tuple (V, w, br, v_esc, V_k, V_end, V_rad, code),
        with br to code as in `breaks`. The nodes V cover :math:`V \ge 0` and are
        ordered by panel, then sub-panel, then node. The outer edges of each
        panel equal the break values exactly.

        Raises
        ------
        ResultError
            If `breaks` raises ResultError.
        """
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
        """Return the velocities (V_k, V_end, V_rad) at the projected radii `r2d_arr`.

        With f the function of `f_extrema`, V_k is ``f^(1/2)`` at t = 0.
        V_end is ``f^(1/2)`` at the largest of the values of f at t = 0, at the
        interior extrema of `f_extrema` and at the end of the line of sight.
        V_end is the largest velocity with Q_c > 0 somewhere on the line of
        sight. V_end is NaN where one of the values of f is NaN.

        V_rad is the largest value over t of ``(2 max(P, 0))^(1/2) u / R``.
        V_rad is found by a scan in t followed by ``rules.brk_zoom`` zooms of
        41 points around the largest value of the scan. Without `bracket`, the
        scan covers [0, t_max] at ``rules.brk_scan`` points. With `bracket`,
        given as (lo_end, hi_end, lo_rad, hi_rad), the scan covers
        [lo_rad, hi_rad] at 41 points, and lo_end and hi_end are not read.
        V_rad is NaN on a route whose DF has no L factor.

        With `with_t` True, the result also holds t_end and t_rad, the values
        of t where V_end and V_rad are attained.
        """
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
            # f at each pair (R_2D, t), equal to f computed for one radius alone.
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
        # V_end takes f at t = 0, at the extrema of each radius and at the end
        # of each line of sight, for all radii together. Each radius takes the
        # first largest value of f. A NaN value of f is taken first, as
        # np.argmax takes a NaN.
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
        """Return the layout code of the rows at the projected radii `r2d_arr`.

        The candidate breaks are V_k and V_rad inside their kept ranges
        (`vk_break`, `vrad_break`), V_edge and V_c. The lattice then drops each
        candidate whose panel the lattice does not resolve (`Lattice.resolved_breaks`).
        The code orders the remaining breaks (`Lattice.layout_code`).
        """
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
        """Return the raw layout code of the rows at the projected radii `r2d_arr`.

        The raw layout orders V_k, V_rad, V_edge and V_c without the kept
        ranges and without the resolution test of the lattice. Only the test
        of `Lattice.layout_code` against V_end applies (BREAK_TOL).
        """
        vk, vend, vr = self.breaks_many(r2d_arr)
        return self.lattice.layout_code(
            vk, vend, vr, self.v_edge_many(r2d_arr), self.v_c_many(r2d_arr)
        )

    def vrad_break(self, r2d_arr: ArrayLike, vr: Array) -> Array:
        """Return `vr` at the radii inside the kept ranges of V_rad, NaN elsewhere.

        `vr` is returned unchanged where no kept ranges are set.
        """
        if self.vrad_kept is None:
            return vr
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        inside = np.zeros(r2.shape, dtype=bool)
        for a, b in self.vrad_kept:
            inside |= (r2 >= a) & (r2 <= b)
        return np.where(inside, vr, np.nan)

    def vk_break(self, r2d_arr: ArrayLike, vk: Array) -> Array:
        """Return `vk` at the radii inside the kept ranges of V_k, NaN elsewhere.

        `vk` is returned unchanged where no kept ranges are set.
        """
        if self.vk_kept is None:
            return vk
        r2 = np.asarray(r2d_arr, dtype=np.float64)
        inside = np.zeros(r2.shape, dtype=bool)
        for a, b in self.vk_kept:
            inside |= (r2 >= a) & (r2 <= b)
        return np.where(inside, vk, np.nan)

    def row_test(self, r2d: float) -> tuple[bool, float, str]:
        """Return whether the row at `r2d` can be stored, its node gap, and the reason.

        The result is the tuple (stored, gap, reason). The gap is the least
        gap between neighboring velocity nodes over the value of the upper
        node. A row is stored where three conditions hold: `breaks` raises no
        ResultError with the marker BREAK_PAST_END; the gap is at least
        RB_MIN_GAP; and no graded side of a panel needs more layers than
        `l_max` allows. The reason is empty for a stored row, and states the
        failed condition otherwise.

        Raises
        ------
        ResultError
            If `panels` raises a ResultError without the marker
            BREAK_PAST_END.
        """
        try:
            _, _, br, _, _, _, _, code = self.panels(float(r2d))
        except ResultError as e:
            if BREAK_PAST_END not in str(e):
                raise
            return (
                False,
                0.0,
                f"a velocity break on or above V_end {BREAK_PAST_END}",
            )
        gap = float(
            np.min(
                self.lattice.lattice_min_gaps(
                    np.array([br], dtype=np.float64),
                    self.lattice.panel_units(int(code)),
                )
            )
        )
        if gap < RB_MIN_GAP:
            return (
                False,
                gap,
                f"a gap {gap:.3e} between velocity nodes, below RB_MIN_GAP",
            )
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
                        f"the panel [{nm[i]}, {nm[i + 1]}], whose width over {side} "
                        f"is {w / v:.3e} and which needs {need} layers toward "
                        f"{side}, more than the {l_max(w / v, self.rules)} layers "
                        f"that double precision resolves",
                    )
        return True, gap, ""

    def node_crossings(self, rs: Array, rb: Array) -> tuple[Array, list[str]]:
        r"""Return the radii where a merged break crosses a velocity node of its row.

        A merged break is a break that the row's layout does not carry. The
        sample radii are the radii `rs`, the radii just below and above each
        R-break of `rb`, and ``rules.rb_interior`` radii inside each interval
        between neighboring points of the set formed by the first radius of
        `rs`, the R-breaks and the last radius of `rs`. Two neighboring
        samples are compared where both samples have one layout code and the
        merged break lies in one panel at both samples. The merged break
        crosses every velocity node of the panel that lies between the
        break's positions at the two samples. Each crossing is bisected
        ``rules.rb_bisect`` times in :math:`R_{\rm 2D}`. Each crossing radius becomes an
        R-break, where the slope of D in :math:`R_{\rm 2D}` may jump.

        A crossing of V_rad or V_k outside the break's kept ranges
        (`vrad_break`, `vk_break`) is dropped. Outside the kept ranges of a
        break, the share of the row next to the break is at most VRAD_KEEP.

        Returns
        -------
        radii : array
            The crossing radii in increasing order.
        labels : list of str
            The label of the crossing break at each radius (`BREAK_LABELS`).
        """
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
        # A crossing of V_rad or V_k outside its kept ranges is no R-break.
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
        r"""Return the radii of the rows and the R-breaks, from the base radii `r_base`.

        An R-break is a radius where the layout of the rows changes. The
        R-breaks are found on samples of :math:`R_{\rm 2D}`. The samples are
        ``rules.rb_interior`` samples inside each interval between base radii,
        the last base radius, and ``rules.rb_ext`` samples beyond the last base
        radius toward :math:`R_{\rm t}`.

        The R-breaks are of three kinds, each bisected in :math:`R_{\rm 2D}`.
        The first kind is a change of `layout_many`. The second kind is a
        change of `raw_layout_many`, except a change of V_rad alone. Where V_k
        enters the raw layout, the R-break moves down to the radius where
        V_end - V_k first exceeds BIRTH_TOL times V_end, searched within
        BIRTH_SPAN in :math:`\ln R_{\rm 2D}`. The third kind is a crossing of a
        merged break over a velocity node (`node_crossings`). R-breaks closer
        than RB_SAME in :math:`\ln R_{\rm 2D}` are merged. The layout is also
        read at midpoints near the raw changes and near the ends of the kept
        ranges. A layout that leaves a code and returns to the code between
        two samples is found at the midpoints.

        The rows are the base radii and three sets of added radii. The first
        set holds up to ``rules.rb_layers`` radii on each side of each
        R-break, each between the R-break and its neighboring R-break. The
        second set holds ``rules.rt_layers`` radii toward the top radius. The
        top radius is the last R-break where the last R-break lies beyond the
        last base radius, and :math:`R_{\rm t}` otherwise. The third set fills
        each stretch between neighboring R-breaks that holds fewer than
        RB_FILL radii with RB_FILL more radii. Only rows that pass `row_test`
        are returned.

        Returns
        -------
        radii : array
            The radii of the rows that pass `row_test`, sorted and unique.
        r_breaks : array
            The R-breaks, sorted.
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
        rb, _, rb_code = changes(self.layout_many, rs, lay)
        lr_ = self.raw_layout_many(rs)
        raw = changes(self.raw_layout_many, rs, lr_)
        # Between two samples the layout can leave a code and return to the code
        # of the far sample, for example the codes 3, 2, 4, 2. The bisection
        # finds only the first change of such a sequence. The layout changes
        # where a raw break changes, where a kept range ends, or where the
        # lattice merges a break near a raw change or a range end. The raw
        # changes and the ends of the kept ranges cut the samples into
        # intervals. The layout is read at the midpoint of each interval next
        # to such a cut, and compared with the code that the changes found so
        # far give at the midpoint. Where the two codes differ, the changes
        # between the two samples around the midpoint are searched again with
        # the midpoint added.
        ends = [
            x
            for ranges in (self.vk_kept, self.vrad_kept)
            if ranges is not None
            for x in np.ravel(ranges)
            if np.isfinite(x)
        ]
        anchors = np.concatenate([raw[0], np.asarray(ends, dtype=np.float64)])
        anchors = anchors[(anchors > rs[0]) & (anchors < rs[-1])]
        cut = np.unique(np.concatenate([rs, anchors]))
        near = np.isin(cut[:-1], anchors) | np.isin(cut[1:], anchors)
        probes = np.sqrt(cut[:-1] * cut[1:])[near]
        for _ in range(len(LAYOUT_SEQ)):
            if probes.size == 0:
                break
            k = np.searchsorted(rb, probes) - 1
            given = np.where(k >= 0, rb_code[np.maximum(k, 0)], lay[0])
            off = self.layout_many(probes) != given
            if not np.any(off):
                break
            j = np.searchsorted(rs, probes[off]) - 1
            for jj in np.unique(j):
                pts = np.concatenate(
                    [rs[jj : jj + 1], probes[off][j == jj], rs[jj + 1 : jj + 2]]
                )
                rc, _, bc = changes(self.layout_many, pts, self.layout_many(pts))
                out = (rb <= rs[jj]) | (rb >= rs[jj + 1])
                rb = np.concatenate([rb[out], rc])
                rb_code = np.concatenate([rb_code[out], bc])
                o = np.argsort(rb)
                rb, rb_code = rb[o], rb_code[o]
        rr_list = []
        for rx0, ca_, cb_ in zip(*raw, strict=True):
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
        """Return a mask of the candidate rows `cand` that pass `row_test`.

        `base` marks the candidates that are base rows. The debug log lists
        every dropped base row, and up to 12 dropped added rows, each with the
        reason from `row_test`.
        """
        res = [self.row_test(x) for x in cand]
        ok = np.array([t[0] for t in res], dtype=bool)
        if not np.all(ok):
            bad = np.flatnonzero(~ok)
            badb = bad[base[bad]]
            log.debug(
                "The table drops %d candidate projected radii, of which %d are base "
                "radii. At each dropped radius, two velocity nodes are "
                "closer than RB_MIN_GAP, or a graded side of a velocity panel needs "
                "more layers than double precision resolves.",
                len(bad),
                len(badb),
            )
            for c in list(badb) + [c for c in bad if not base[c]][:12]:
                log.debug(
                    "The table drops the %sprojected radius R_2D = %.9e, which has %s.",
                    "base " if base[c] else "",
                    float(cand[c]),
                    res[c][2],
                )
        return ok

    # The row.

    def supports(self, r2d: float, va: Array) -> tuple[Array, Array, Array]:
        """Return the intervals in t where Q_c > 0, for every velocity of `va`.

        The result is three flat arrays (jv, t_lo, t_hi): the index into `va`
        of each interval's velocity, and the interval's ends. The sign test reads
        Q_c at t = 0, at the interior extrema of f (`f_extrema`) and at t_max.
        Each change of sign is bisected ``rules.n_bisect`` times. An interval
        between two edges is kept where Q_c > 0 at its midpoint.
        """
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
        e = np.full((len(va), int(np.max(cnt, initial=0)) + 2), tmax)
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
        """Return the support intervals split at every L = 0 crossing.

        The L = 0 point is ``s0 = V^2 R^2 / (2 A u^2)``. A crossing is a root
        in t of ``s0(t) = Q_c(t)``, where s0 enters the range [0, Q_c] of s.
        The roots are found as changes of sign of ``ln s0 - ln Q_c`` on
        ``rules.l_scan`` points spaced evenly in ln t, then bisected
        ``rules.n_bisect`` times. An interval that starts at t = 0 is scanned
        from the lesser of ``t_hi / 2`` and
        ``t_low = arcsinh(L_BELOW V_min / (2 P(R_2D))^(1/2))``, with V_min the
        least positive velocity. Only velocities V > 0 are scanned. The intervals are
        returned unchanged on a route whose DF has no L factor, and where no
        root is found.

        The result is (jv, t_lo, t_hi, cut). `cut` marks each piece of an
        interval that a root divided.
        """
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
        collect: list[tuple[Any, ...]] | None = None,
    ) -> Array:
        """Return D at the distinct velocities `va` at one level of the rule in t.

        Each support interval maps to xi in [0, 1] by
        ``t = t_lo + (t_hi - t_lo) sin^2(pi xi / 2)``. The rule in xi has
        `n_panels` equal Gauss-Legendre panels of ``rules.t_nodes`` nodes.
        Where `n_lay` is positive, `n_lay` panels graded toward xi = 0 at the
        ratio ``rules.t_ratio`` replace the first panel. The pieces of an
        interval that `cut` marks take ``max(n_panels // 2, 1)`` panels.

        With `collect`, the function appends the reads of `plane` to
        `collect`. Each read of F or G carries the index into `va` of its
        velocity and its weight in D. A final read of kind ``"P"`` holds the node
        pairs, the layer groups and the angle counts that `plane` chose.
        """
        rr = self.rules
        if cut is not None and cut.any():
            d = self._row_at(
                r2d,
                va,
                jv[cut],
                tl[cut],
                th[cut],
                rule,
                max(n_panels // 2, 1),
                n_lay,
                collect=collect,
            )
            if (~cut).any():
                d = d + self._row_at(
                    r2d,
                    va,
                    jv[~cut],
                    tl[~cut],
                    th[~cut],
                    rule,
                    n_panels,
                    n_lay,
                    collect=collect,
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
        nodes: list[tuple[Any, ...]] | None = None if collect is None else []
        u_pairs = (r2d * np.sinh(t)).ravel()
        vals = self.plane(r2d, u_pairs, vp, rule, collect=nodes)
        if collect is not None and nodes is not None:
            velocity, weight = np.repeat(jv, xs.size), 2.0 * wt.ravel()
            for kind, at, k, w in nodes:
                if kind in ("F", "G"):
                    collect.append((kind, velocity[at], k, weight[at][:, None] * w))
            # The node pairs, with the layer groups and angle counts of `plane`.
            pairs = dict(
                r2d=r2d,
                u=u_pairs,
                v=vp,
                rule=rule,
                groups=[(at, w) for kind, at, _, w in nodes if kind == "S"],
                counts=[(at, w) for kind, at, _, w in nodes if kind == "K"],
            )
            collect.append(("P", velocity, pairs, weight))
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
        trace: list[tuple[int, int, Array]] | None = None,
        collect: list[tuple[Any, ...]] | None = None,
    ) -> Array:
        r"""Return :math:`D(R_{\rm 2D}, V)` at the velocities `v` of the row at `r2d`.

        The velocities `v` may have either sign. D is computed at the distinct
        values of |V|. Each level of the rule in t doubles the panel count and
        adds ``rules.t_layers`` graded panels toward the lower end of each
        support interval, up to ``rules.t_levels`` doublings. The doubling
        stops once the change of the row from the previous level is at most
        T_TOL in two tests. With the row's weights `w`, the L1 test takes the
        weighted sum of the changes over the weighted sum of |D|, and the
        pointwise test takes the largest relative change at the nodes where
        |D| exceeds :math:`10^{-12}` times the largest |D|. With `w`, a velocity
        is recomputed at the next level only while its own relative change
        exceeds T_TOL. Without `w`, both tests take the largest change over
        the largest |D|.

        On a Type I model, the first level adds graded panels until the outer
        edge of the innermost graded panel lies at or below
        ``t_a = arcsinh(R_a / R_2D)``. That edge is mapped to t on [0, t_m],
        with t_m the largest upper end of the support intervals.

        `report`, where given, receives the panel count, the node count, the
        changes, the convergence flag, the test that passed last (`T_SET_BY`)
        and the layer counts. `trace`, where given, receives at each level the
        panel count, the layer count and the mask of the velocities computed at
        that level. With `collect`, each level's reads (`_row_at`) are kept
        for the velocities whose value comes from that level. Each kept read
        equals the read that `_row_at` at the level collects on the kept velocities
        alone.
        """
        rr = self.rules
        va, inv = np.unique(
            np.abs(np.asarray(v, dtype=np.float64)), return_inverse=True
        )
        jv, tl, th = self.supports(r2d, va)
        jv, tl, th, cut = self.l0_split(r2d, va, jv, tl, th)
        tm = float(th.max()) if th.size else 0.0
        # On a Type I model, the graded panels of level 0 reach t at or below
        # t_a = arcsinh(R_a / R_2D), with at most 400 layers.
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
        # `level` holds the reads of the last level computed, and `computed`
        # marks the velocities that the last level computed. The reads of a level
        # are kept for the velocities of that level that no later level computes.
        level: list[tuple[Any, ...]] = []
        d = self._row_at(
            r2d,
            va,
            jv,
            tl,
            th,
            rule,
            n_p,
            n_lay,
            cut,
            None if collect is None else level,
        )
        computed = np.ones(len(va), dtype=bool)
        if trace is not None:
            trace.append((n_p, n_lay, np.ones(len(va), dtype=bool)))
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
            if trace is not None:
                trace.append(
                    (n_p, n_lay, act.copy() if per_node else np.ones(len(va), bool))
                )
            if collect is not None:
                now = act.copy() if per_node else np.ones(len(va), bool)
                collect += _velocities_kept(level, computed & ~now)
                level, computed = [], now
            if per_node:
                sel = act[jv]
                d_new = d.copy()
                if sel.any():
                    d_new[act] = self._row_at(
                        r2d,
                        va,
                        jv[sel],
                        tl[sel],
                        th[sel],
                        rule,
                        n_p,
                        n_lay,
                        cut[sel],
                        None if collect is None else level,
                    )[act]
            else:
                d_new = self._row_at(
                    r2d,
                    va,
                    jv,
                    tl,
                    th,
                    rule,
                    n_p,
                    n_lay,
                    cut,
                    None if collect is None else level,
                )
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
        if collect is not None:
            collect += _velocities_kept(level, computed)
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

    def plane(
        self,
        r2d: float,
        u: Array,
        v: Array,
        rule: tuple[int, ...],
        collect: list[tuple[Any, ...]] | None = None,
    ) -> Array:
        """Return the integral of ``F (L^2)^(-beta_c)`` over the velocity plane.

        Each node pair is a depth u and a velocity V on the line of sight at
        `r2d`. The integral is the plane integral I(u) of the module
        docstring. Without an L factor, the integral is
        ``2 pi G(Q_c)`` times the jacobian factor. With an L factor, the
        pairs are grouped by their layer counts (`_s_layers`, `_ks_layers`),
        and each group is integrated by `_plane_s`.

        With `collect`, the function appends four kinds of reads to
        `collect`. A read ``("F", pairs, K, W)`` gives the value at each pair
        as the sum over the pair's row of W F(K). A read
        ``("G", pairs, Q_c, W)`` gives the value at each pair as W G(Q_c). A
        read ``("S", pairs, None, (k0, ka, kb, kq, m, nphi))`` gives the layer
        counts of a group. A read ``("K", pairs, None, index)`` gives the
        index of the angle count of each node, -1 where the node is not read.
        """
        df = self._df()
        _, k0, ks, kq, m, nphi = rule
        r = np.sqrt(r2d * r2d + u * u)
        q = self.q_of(r)
        a = 1.0 + q * (u / r2d) ** 2
        kinetic = 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)
        qc = np.maximum(self.psi(r) - kinetic, 0.0)
        # The depth of Q_c below P0, from the depth of the potential
        # (`PotentialReading.depth`) plus the kinetic term.
        depth = getattr(self.psi, "depth", None)
        dqc = depth(r) + kinetic if depth is not None else np.full(len(u), np.nan)
        jac = (r / r2d) / np.sqrt(a * q)
        if not self.lfac:
            if collect is not None:
                at = np.arange(len(u))
                collect.append(("G", at, qc[:, None], (jac * 2.0 * np.pi)[:, None]))
            return jac * 2.0 * np.pi * df.G(qc, dqc)
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
            nodes: list[tuple[Any, ...]] | None = None if collect is None else []
            out[g] = self._plane_s(
                r2d,
                u[g],
                v[g],
                r[g],
                q[g],
                a[g],
                qc[g],
                dqc[g],
                jac[g],
                split[g],
                int(kp[i]),
                int(ka[i]),
                int(kb[i]),
                kq,
                m,
                nphi,
                collect=nodes,
            )
            if collect is not None and nodes is not None:
                where_g = np.flatnonzero(g)
                for kind, at, k_, w_ in nodes:
                    collect.append((kind, where_g[at], k_, w_))
                layers = (int(kp[i]), int(ka[i]), int(kb[i]), kq, m, nphi)
                collect.append(("S", where_g, None, layers))
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
        dqc: Array,
        jac: Array,
        split: Array,
        k0: int,
        ksa: int,
        ksb: int,
        kq: int,
        m: int,
        nphi: int,
        collect: list[tuple[Any, ...]] | None = None,
    ) -> Array:
        r"""Return the (s, phi) quadrature of `plane` on pairs that share layers.

        Piece a covers s in [0, split] on the panels of
        ``seg_unit(k0, ksa)``. Piece b covers s in [split, Q_c] on the panels
        of ``seg_unit(ksb, kq)``. Where :math:`0 < \beta_{\rm c} < 1`, the
        first panel of piece b is mapped by ``s - split = d w^p`` with
        :math:`p = 1/(1 - \beta_{\rm c})`. Where `EllipseDF.zero_map` asks,
        the last panel of piece b is mapped so that the map carries
        :math:`F \sim K^p` at K = 0 exactly. Piece a reads F by the depth
        ``dqc + s`` at the energies K = Q_c - s above half the end energy of
        the DF. `dqc` is the depth of Q_c below P0, NaN where the potential
        gives no depth. With `collect`, the function appends the reads of
        `plane`.
        """
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
            lw = (
                int(np.ceil(np.log(TIER_BOUNDS[0]) / np.log(sr)))
                if df.k_zero != 0.0
                else 0
            )
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
        # The numpy rule forms a block of shape (s nodes, angle nodes) for each
        # pair, so a chunk of pairs is sized by both counts. The JAX kernel and
        # the closed form form no such block, so a chunk is sized by the s
        # nodes alone.
        step = max(
            1, rr.chunk // (ns * (1 if (self.lam_closed or rr.kernel) else nphi))
        )
        counts = tuple(rr.angle_counts)
        for i0 in range(0, len(u), step):
            sl = slice(i0, i0 + step)
            sp, rest = split[sl, None], (qc[sl] - split[sl])[:, None]
            s = sp * sa + rest * sb
            ws = sp * wa_ + rest * wb_
            k_s = np.where(isb[None, :], rest * omb_all[None, :], qc[sl, None] - s)
            u_s = np.where(
                ~isb[None, :] & (k_s > 0.5 * df.Qend), dqc[sl, None] + s, np.nan
            )
            fs = df.F(k_s, u_s) if np.any(np.isfinite(u_s)) else df.F(k_s)
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
                if collect is not None:
                    chosen = np.full(lam.shape, -1, dtype=np.int64)
                    chosen.flat[live] = k
                    collect.append(("K", np.arange(len(u))[sl], None, chosen))
            out[sl] = pre[sl] * np.einsum(
                "ij,ij,ij->i", ws, fs, np.where(ws > 0.0, lam_v, 0.0)
            )
            if collect is not None:
                at = np.arange(len(u))[sl]
                weight = pre[sl, None] * ws * np.where(ws > 0.0, lam_v, 0.0)
                collect.append(("F", at, k_s, weight))
        return out
