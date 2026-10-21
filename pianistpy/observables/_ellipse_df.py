r"""Read the DF of a model and its cumulative integral for the ellipse route.

The ellipse route (`EllipseRoute`) computes the rows of the line-of-sight
velocity distribution that the public calls `pianistpy.losvd`,
`pianistpy.aperture_losvd` and `pianistpy.LosTable` read. `EllipseDF` gives the
route the DF :math:`F(Q)` at energies Q and the cumulative integral

.. math::

    G(Q) = \int_0^Q F(K)\,{\rm d}K,

with K the energy variable of the DF. A family without an L factor in its DF
needs only :math:`G(Q_c)` at the largest energy :math:`Q_c` of each node pair
(Notes Eq. (T:6.2)). The table of G uses Simpson's rule on three branches.
"""

__all__ = ["EllipseDF"]

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..exceptions import InputError, ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ..validation import TIER_BOUNDS
from ._ellipse_rules import EllipseRules, gl01


class EllipseDF:
    r"""The DF F of a model and the table of its cumulative integral G.

    Parameters
    ----------
    q_grid : array-like
        Energies of the DF's table. The first energy is the floor
        :math:`Q_0`. The last energy is the top :math:`Q_{\rm top}`.
    f : callable
        F at energies, 0 where the lookup gives no finite value of F. The
        route passes `f` a second argument, the depth :math:`P_0 - Q` below
        the top energy :math:`P_0`, at the energies where the route holds
        the depth (`_ellipse_model.DfReading`).
    log10_f : callable
        The base-10 logarithm of F from the lookup. The values of `f` may
        equal :math:`10^{\log_{10} F}` times a constant factor (`scale`).
    cumulative : callable
        The lookup's own integral :math:`\int_0^Q F\,{\rm d}K` at one energy
        Q, without the constant factor of `f`.
    info : mapping
        The lookup's statement of the ends of the DF. ``below_floor`` is
        the rule of F below the floor (``"tail"``, ``"edge_law"`` or
        another). ``tail_a`` and ``tail_k`` are the exponents of the tail.
        ``edge_law_p`` is the exponent of the edge law. ``kappa`` is the
        exponent :math:`\kappa` of :math:`F \sim (P_0 - Q)^\kappa` at the
        top. ``P0`` is the top energy :math:`P_0`.
    rules : EllipseRules
        The route's rules (``g_nodes``, ``s_ratio``).

    Raises
    ------
    InputError
        If the exponent p of :math:`F \sim K^p` at :math:`K \to 0` is not
        above -1, so that F is not integrable at K = 0.

    Notes
    -----
    The table of G has three branches of ``g_nodes`` nodes each. Each branch
    integrates F with Simpson's rule (`_cum`) on a uniform grid.

    - The lower branch runs in ln Q from :math:`10^{-40} Q_0` to
      :math:`Q_0`. The lower branch starts from the lookup's cumulative
      integral at :math:`10^{-40} Q_0`.
    - The middle branch runs in ln Q from :math:`Q_0` to
      :math:`Q_{\rm top}/2`. The middle branch starts from the lookup's
      cumulative integral at :math:`Q_0`.
    - The upper branch runs in :math:`\ln(Q_{\rm end} - Q)` from
      :math:`Q_{\rm top}/2` to :math:`Q_{\rm end}(1 - 10^{-13})`. The upper
      branch starts from the value of the middle branch at
      :math:`Q_{\rm top}/2`.

    The end energy :math:`Q_{\rm end}` is :math:`P_0` where :math:`P_0` is
    finite and above :math:`Q_{\rm top}`, and :math:`Q_{\rm top}` otherwise.
    """

    def __init__(
        self,
        q_grid: ArrayLike,
        f: Callable[..., Array],
        log10_f: Callable[[Array], Array],
        cumulative: Callable[[float], float],
        info: Mapping[str, Any],
        rules: EllipseRules,
    ) -> None:
        q_grid = np.asarray(q_grid, dtype=np.float64)
        n = int(rules.g_nodes)
        self.rules = rules
        self.F = f
        self.Q0, self.Qtop = float(q_grid[0]), float(q_grid[-1])
        self.Qmid = 0.5 * self.Qtop
        self.F0 = float(f(np.array([self.Q0]))[0])
        l10q0 = float(log10_f(np.array([self.Q0]))[0])
        self._scale = (
            self.F0 / 10.0**l10q0 if np.isfinite(l10q0) and self.F0 > 0.0 else 1.0
        )
        base = float(cumulative(self.Q0)) * self._scale
        lq = np.linspace(np.log(self.Q0), np.log(self.Qmid), n)
        gl = self._cum(f(np.exp(lq)) * np.exp(lq), lq[1] - lq[0], base)
        p0 = float(info.get("P0", float("nan")))
        self.Qend = p0 if np.isfinite(p0) and p0 > self.Qtop else self.Qtop
        # Where Q_end is the lookup's P0, the depth P0 - Q given to G replaces the
        # difference Q_end - Q as the variable of the upper branch.
        self.depth_end = bool(np.isfinite(p0) and p0 > self.Qtop)
        y = np.linspace(np.log(self.Qend - self.Qmid), np.log(1e-13 * self.Qend), n)
        gu = self._cum(f(self.Qend - np.exp(y)) * np.exp(y), y[0] - y[1], gl[-1])
        self._lq, self._lGl = lq, np.log(np.maximum(gl, 1e-300))
        self._yu, self._lGu = y[::-1], np.log(np.maximum(gu, 1e-300))[::-1]
        q_low = 1e-40 * self.Q0
        lb = np.linspace(np.log(q_low), np.log(self.Q0), n)
        gb = self._cum(
            f(np.exp(lb)) * np.exp(lb),
            lb[1] - lb[0],
            float(cumulative(q_low)) * self._scale,
        )
        self._lb, self._lGb = lb, np.log(np.maximum(gb, 1e-300))
        self._sb = (self._lGb[1] - self._lGb[0]) / (lb[1] - lb[0])
        log.debug(
            "The cumulative integral G at the floor Q0 = %.6e is %.6e from the DF "
            "lookup, and the branch of G below Q0 gives %.9e at Q0. G is continued "
            "to Q_end = %.12f.",
            self.Q0,
            base,
            gb[-1],
            self.Qend,
        )
        bf = info.get("below_floor")
        # The exponent p_zero of F ~ K^p at K -> 0, as the lookup states it, and
        # the exponent k_zero of the logarithmic factor of the tail.
        self.p_zero = (
            -float(info.get("tail_a", 0.0))
            if bf == "tail"
            else float(info.get("edge_law_p", 0.0))
            if bf == "edge_law"
            else 0.0
        )
        self.k_zero = float(info.get("tail_k", 0.0)) if bf == "tail" else 0.0
        if not self.p_zero > -1.0:
            raise InputError(
                f"The DF behaves as F ~ K^p as K -> 0 with p = {self.p_zero:g} (the "
                f"rule {bf} below the floor), and p is at most -1, so F is not "
                f"integrable at K = 0."
            )
        self._zmap: dict[tuple[int, int], bool] = {}
        # The exponent kappa_top of F ~ (P0 - Q)^kappa at the top energy P0.
        self.kappa_top = float(info.get("kappa", 0.0))

    @property
    def scale(self) -> float:
        """The constant factor c in ``f = c 10**log10_f``, read at the floor Q_0.

        The factor is 1 where ``log10_f`` is not finite at Q_0 or F is not
        positive at Q_0. A row of D computed with `f` integrates over V to the
        surface density times the factor.
        """
        return self._scale

    def zero_map(self, m: int, kq: int) -> bool:
        r"""Return whether the s panel that touches K = 0 takes the map for F ~ K^p.

        The map carries the power :math:`F \sim K^p` at :math:`K \to 0`
        exactly. The panel takes the map where the error estimate e exceeds
        :math:`10^{-8}`. The estimate is
        :math:`e = |(p + 1)\sum_i w_i x_i^p - 1|\,(0.5\,r^{k_q})^{p + 1}`,
        with :math:`x_i` and :math:`w_i` the `m`-node Gauss-Legendre nodes and
        weights on [0, 1], r = ``rules.s_ratio`` and :math:`k_q` = `kq`. The
        first factor is the relative error of the rule on :math:`x^p`. The
        second factor is the share of the integral of :math:`K^p` that the
        panel carries. The result is cached for each pair (`m`, `kq`).
        """
        key = (int(m), int(kq))
        if key not in self._zmap:
            xg, wg = gl01(int(m))
            p = self.p_zero
            e = abs((p + 1.0) * float(np.dot(wg, xg**p)) - 1.0) * (
                0.5 * self.rules.s_ratio ** int(kq)
            ) ** (p + 1.0)
            self._zmap[key] = bool(e > 1e-2 * TIER_BOUNDS[0])
        return self._zmap[key]

    @staticmethod
    def _cum(g: Array, h: float, base: float) -> Array:
        """Return the cumulative integral of samples `g` on a uniform grid of step h.

        The integral starts from `base` at the first node. At the even nodes
        the integral is the composite Simpson sum. At an odd node the integral
        is the cubic Hermite interpolant of the integral through the two
        neighboring even nodes, with the samples `g` as its slopes. An odd last
        node, which an even node count leaves, keeps the trapezoid sum.
        """
        c = np.empty_like(g)
        c[0] = base
        c[1:] = base + np.cumsum(0.5 * h * (g[1:] + g[:-1]))
        c[2::2] = base + np.cumsum(h / 3.0 * (g[0:-2:2] + 4.0 * g[1:-1:2] + g[2::2]))
        c[1:-1:2] = 0.5 * (c[0:-2:2] + c[2::2]) + (h / 4.0) * (g[0:-2:2] - g[2::2])
        return c

    def G(self, q: ArrayLike, depth: ArrayLike | None = None) -> Array:
        r"""Return :math:`G(Q) = \int_0^Q F\,{\rm d}K` at the energies `q`.

        G is 0 at energies at or below 0. Below the floor :math:`Q_0`, G is
        read from the lower branch, and extended in ln Q by the branch's first
        slope below :math:`10^{-40} Q_0`. On the upper branch, `depth` gives
        :math:`P_0 - Q` where the caller holds the depth exactly, and NaN
        elsewhere. Where `depth` is finite and :math:`Q_{\rm end} = P_0`, the
        depth replaces the difference :math:`Q_{\rm end} - Q`, which loses
        digits near :math:`P_0`.

        Raises
        ------
        ResultError
            If an energy lies above :math:`Q_{\rm end}`, where the DF has no
            value.
        """
        q = np.asarray(q, dtype=np.float64)
        out = np.zeros_like(q)
        a = (q > 0.0) & (q < self.Q0)
        lqa = np.log(np.where(a, q, self.Q0))
        out[a] = np.exp(
            np.where(
                lqa >= self._lb[0],
                np.interp(lqa, self._lb, self._lGb),
                self._lGb[0] + self._sb * (lqa - self._lb[0]),
            )
        )[a]
        b = (q >= self.Q0) & (q <= self.Qmid)
        out[b] = np.exp(np.interp(np.log(q[b]), self._lq, self._lGl))
        c = q > self.Qmid
        if np.any(q > self.Qend):
            raise ResultError(
                f"Q = {float(q.max()):.15f} lies above P0 = {self.Qend:.15f}, and the "
                f"DF has no value above P0; the profile's potential exceeds the "
                f"model's own P0."
            )
        below = self.Qend - q[c]
        if depth is not None and self.depth_end:
            held = np.broadcast_to(np.asarray(depth, dtype=np.float64), q.shape)[c]
            below = np.where(np.isfinite(held), held, below)
        out[c] = np.exp(
            np.interp(
                np.log(np.maximum(below, 1e-13 * self.Qend)),
                self._yu,
                self._lGu,
            )
        )
        return out
