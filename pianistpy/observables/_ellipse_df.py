r"""The DF as the ellipse route reads it: :math:`F(Q)` and
:math:`G(Q) = \int_0^Q F\,{\rm d}K` on three branches."""

__all__ = ["EllipseDF"]

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..exceptions import InputError, ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ..validation import GOAL
from ._ellipse_rules import EllipseRules, gl01


class EllipseDF:
    r"""F from a lookup, and :math:`G(Q) = \int_0^Q F\,{\rm d}K` tabulated
    from the same F.

    Parameters
    ----------
    q_grid : array-like
        The DF table's energies; its first and last are its floor and top.
    f : callable
        F at energies, zero where the DF has no value; the route also passes
        it the depths below P0 near the top, where it holds them, as a second
        argument (`DfReading`).
    log10_f : callable
        The lookup's base-10 logarithm of F, which `f` may carry a constant
        factor against.
    cumulative : callable
        The lookup's own :math:`\int_0^Q F\,{\rm d}K` at one energy, whose
        below-floor piece is in closed form.
    info : mapping
        The lookup's statement of its ends: ``below_floor`` (``"tail"``,
        ``"edge_law"`` or another), ``tail_a``, ``tail_k``, ``edge_law_p``,
        ``kappa`` and ``P0``.
    rules : EllipseRules
        The route's rules (``g_nodes``, ``s_ratio``).

    Notes
    -----
    G runs on three branches with Simpson's rule: in ln Q from
    :math:`10^{-40}` Q_0 to the floor Q_0, in ln Q from Q_0 to half the top,
    and in ln(Q_end - Q) from there to :math:`10^{-13}` Q_end, Q_end the
    lookup's P0 where it lies above the top.
    Each branch starts from the lookup's own cumulative integral.
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
        # a depth below the lookup's P0 is the top branch's own variable
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
            "G's base at the floor Q0 = %.6e is %.6e from the lookup; the branch "
            "below it reads %.9e there; continued to Q_end = %.12f.",
            self.Q0,
            base,
            gb[-1],
            self.Qend,
        )
        bf = info.get("below_floor")
        # F ~ K^p at K -> 0 as the lookup states it, and the tail's logarithm
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
                f"F ~ K^p at K -> 0 with p = {self.p_zero:g} ({bf}) is not "
                f"integrable at K = 0."
            )
        self._zmap: dict[tuple[int, int], bool] = {}
        # and the exponent at the top, F ~ (P0 - Q)^kappa
        self.kappa_top = float(info.get("kappa", 0.0))

    @property
    def scale(self) -> float:
        """The constant factor `f` carries against ``10**log10_f``: a row of D
        integrates to the surface density times it."""
        return self._scale

    def zero_map(self, m: int, kq: int) -> bool:
        r"""Return whether the s panel that touches K = 0 takes the map that
        carries :math:`F \sim K^p` exactly: where m-node Gauss-Legendre's own
        error on :math:`K^p` over that panel, :math:`|(p+1)\sum w x^p - 1|`
        times its share :math:`(0.5\,r^{K_q})^{p+1}`, exceeds :math:`10^{-2}` of the
        goal."""
        key = (int(m), int(kq))
        if key not in self._zmap:
            xg, wg = gl01(int(m))
            p = self.p_zero
            e = abs((p + 1.0) * float(np.dot(wg, xg**p)) - 1.0) * (
                0.5 * self.rules.s_ratio ** int(kq)
            ) ** (p + 1.0)
            self._zmap[key] = bool(e > 1e-2 * GOAL)
        return self._zmap[key]

    @staticmethod
    def _cum(g: Array, h: float, base: float) -> Array:
        """The cumulative integral on a uniform grid of step h: Simpson on the
        even nodes, the cubic Hermite midpoint on the odd ones."""
        c = np.empty_like(g)
        c[0] = base
        c[1:] = base + np.cumsum(0.5 * h * (g[1:] + g[:-1]))
        c[2::2] = base + np.cumsum(h / 3.0 * (g[0:-2:2] + 4.0 * g[1:-1:2] + g[2::2]))
        c[1:-1:2] = 0.5 * (c[0:-2:2] + c[2::2]) + (h / 4.0) * (g[0:-2:2] - g[2::2])
        return c

    def G(self, q: ArrayLike, depth: ArrayLike | None = None) -> Array:
        r"""Return :math:`\int_0^Q F\,{\rm d}K` at energies.

        `depth`, :math:`P_0 - Q` where the caller holds it exactly (NaN where
        not), is the top branch's variable in place of the difference, which
        has lost the digits near :math:`P_0`.

        Raises
        ------
        ResultError
            Where an energy lies above Q_end, where the DF has no value.
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
                f"Q = {float(q.max()):.15f} lies above P0 = {self.Qend:.15f}, where "
                f"the DF has no value: the profile's potential exceeds the model's "
                f"own P0."
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
