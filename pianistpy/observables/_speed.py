r"""The local speed distribution of a model at a 3D radius: the DF's integral
over the direction of the velocity at each speed.

With :math:`t = \cos\eta`, :math:`\eta` the angle between the velocity and
the radius, :math:`\xi = (r/r_a)^2` and the sign :math:`s` of the anisotropy
(+1 for Type I, -1 for Type II, 0 without :math:`r_a`),

.. math::

    p(V\,|\,r) = \frac{4\pi}{\varrho(r)}\, r^{-2b} V^{2-2b}
        \int_0^1 (1-t^2)^{-b} f(Q)\,{\rm d}t, \qquad
    Q = \mathcal{P}(r) - \frac{V^2}{2}(1 + s\xi) + \frac{s\xi V^2}{2}\,t^2,

where :math:`b` is :math:`\beta_c` of the constant and COM classes and 0
otherwise, and :math:`f` is read by the DF's lookup over its own support
:math:`Q > K_c` (:math:`K_c` the lookup's floor where the DF is zero below
it, else 0). :math:`Q` is quadratic in :math:`t`, so the support's edge is an
end of the interval in :math:`t` and never an interior point; the interval
is also split where :math:`Q` crosses the table's first energy, where the
DF's table joins its form below. For :math:`b > 0` the piece that ends at
:math:`t = 1` is integrated in :math:`w`, :math:`1 - t = w^p`,
:math:`p = 1/(1-b)`, where :math:`(1-t)^{-b}{\rm d}t = p\,{\rm d}w`.
"""

__all__ = [
    "SpeedModel",
    "angle_integral",
    "angle_pieces",
    "graded_rule",
    "speed_breaks",
    "speed_model",
]

import functools
import math
from dataclasses import dataclass

import numpy as np

from ..df import Model
from ..df.lookup import K_0, LookupArrays, LookupKinds, evaluate_lookup, prepare_lookup
from ..typing import Array, ArrayLike
from ._tables import _kernel_beta

# The smallest panel of a graded rule, as a fraction of half its interval.
GRADE_MIN = 1e-12


@dataclass(frozen=True)
class SpeedModel:
    r"""What the angle integral reads of a model of one component.

    Parameters
    ----------
    arrays, kinds
        The DF's prepared lookup.
    k_c : float
        The DF's floor: the lookup's first energy where the DF is zero below
        it, else 0.
    k_w : float
        The table's first energy, where it joins its form below.
    b : float
        The L-factor exponent: :math:`\beta_{\rm c}` of the constant and COM
        classes, else 0.
    sign : float
        +1 for Type I, -1 for Type II, 0 without an anisotropy radius.
    r_a : float
        The anisotropy radius.
    """

    arrays: LookupArrays
    kinds: LookupKinds
    k_c: float
    k_w: float
    b: float
    sign: float
    r_a: float


def speed_model(model: Model) -> SpeedModel:
    """Return what the angle integral reads of a model of one component."""
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    anisotropy = model.components[0].anisotropy
    table = model.table
    return SpeedModel(
        arrays=arrays,
        kinds=kinds,
        k_c=float(arrays.params[K_0]) if kinds.below_floor == "zero" else 0.0,
        k_w=float(table.k[0, table.i_first[0]]),
        b=_kernel_beta(model),
        sign=0.0
        if math.isinf(anisotropy.r_a)
        else (1.0 if anisotropy.kind == "I" else -1.0),
        r_a=float(anisotropy.r_a),
    )


@functools.cache
def _unit_rule(n_panels: int, n_nodes: int) -> tuple[Array, Array, Array]:
    """The grading and the unit Gauss-Legendre rule of `graded_rule`, made
    once per (panels, nodes) and read only."""
    grade = np.geomspace(GRADE_MIN, 1.0, n_panels)
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    for a in (grade, x, w):
        a.setflags(write=False)
    return grade, x, w


def graded_rule(a: float, b: float, n_panels: int, n_nodes: int) -> tuple[Array, Array]:
    """Gauss-Legendre of `n_nodes` on 2 `n_panels` panels of [a, b], graded
    geometrically toward both ends, the smallest `GRADE_MIN` of half the
    interval. Empty where the interval is."""
    if not b > a:
        return np.zeros(0), np.zeros(0)
    grade, x, w = _unit_rule(int(n_panels), int(n_nodes))
    h = 0.5 * (b - a) * grade
    edges = np.unique(np.concatenate([[a, b], a + h, b - h]))
    lo, hi = edges[:-1], edges[1:]
    half = 0.5 * (hi - lo)
    return (
        ((0.5 * (lo + hi))[:, None] + half[:, None] * x).ravel(),
        (half[:, None] * w).ravel(),
    )


def speed_breaks(sm: SpeedModel, psi: float, xi: float) -> tuple[float, float, float]:
    r"""Return :math:`(V_{\rm esc}, V_*, V_{\rm max})` at a radius: the escape
    speed, the speed from which the support cuts the angle interval (Type I
    :math:`\sqrt{2A/(1+\xi)}`, Type II :math:`\sqrt{2A}`, with
    :math:`A = \mathcal{P} - K_c`; :math:`V_{\rm max}` without an anisotropy
    radius) and the speed where the support ends (:math:`\sqrt{2A}`, Type II
    :math:`\sqrt{2A/(1-\xi)}`)."""
    a = psi - sm.k_c
    v_esc = math.sqrt(2.0 * psi)
    if sm.sign > 0.0:
        return v_esc, math.sqrt(2.0 * a / (1.0 + xi)), math.sqrt(2.0 * a)
    if sm.sign < 0.0:
        return v_esc, math.sqrt(2.0 * a), math.sqrt(2.0 * a / (1.0 - xi))
    v_max = math.sqrt(2.0 * a)
    return v_esc, v_max, v_max


def _f(sm: SpeedModel, q: Array) -> Array:
    """F at energies, zero outside the support."""
    log_f = np.asarray(evaluate_lookup(sm.arrays, sm.kinds, np.maximum(q, 0.0)))
    return np.where((q > sm.k_c) & np.isfinite(log_f), 10.0**log_f, 0.0)


def angle_pieces(
    sm: SpeedModel, psi: float, xi: float, v: float
) -> tuple[float, float, list[float], list[str]]:
    """Return the coefficients of :math:`Q = c_0 + c_2 t^2` at one speed, the
    edges of the angle integral's pieces in t and, for each edge, what sets
    it: ``"0"`` and ``"1"`` the interval's ends, ``"root"`` the support's
    edge :math:`\\sqrt{(K_c - c_0)/c_2}`, ``"join"`` the table's first energy
    :math:`\\sqrt{(K_w - c_0)/c_2}`. No edges where the support is empty."""
    c0 = psi - 0.5 * v * v * (1.0 + sm.sign * xi)
    c2 = 0.5 * sm.sign * xi * v * v
    lo, hi = 0.0, 1.0
    lo_by, hi_by = "0", "1"
    cuts = []
    if c2 != 0.0:
        # the support Q > K_c in t: above its root for Type I, below it
        # for Type II
        root = (sm.k_c - c0) / c2
        if c2 > 0.0 and root > 0.0:
            lo = math.sqrt(min(root, 1.0))
            lo_by = "root" if root < 1.0 else "1"
        elif c2 < 0.0:
            hi = math.sqrt(min(max(root, 0.0), 1.0))
            hi_by = "root" if 0.0 < root < 1.0 else ("0" if root <= 0.0 else "1")
        join = (sm.k_w - c0) / c2
        if 0.0 < join < 1.0:
            cuts.append(math.sqrt(join))
    elif c0 <= sm.k_c:
        return c0, c2, [], []
    kept = sorted(c for c in cuts if lo < c < hi)
    return c0, c2, [lo, *kept, hi], [lo_by, *("join" for _ in kept), hi_by]


def angle_integral(
    sm: SpeedModel, psi: float, xi: float, v: ArrayLike, n_nodes: int, n_panels: int
) -> Array:
    r"""Return :math:`\int_0^1 (1-t^2)^{-b} f(Q)\,{\rm d}t` at speeds `v`.

    Parameters
    ----------
    sm : SpeedModel
        The model's reading.
    psi, xi : float
        The potential and :math:`(r/r_a)^2` (0 without an anisotropy radius)
        at the radius.
    v : array-like
        Speeds, in internal units.
    n_nodes, n_panels : int
        The rule of each interval: `graded_rule`.
    """
    b = sm.b
    speeds = np.asarray(v, dtype=np.float64)
    out = np.empty(speeds.size)
    for j, vj in enumerate(speeds.ravel()):
        c0, c2, edges, _ = angle_pieces(sm, psi, xi, float(vj))
        if not edges:
            out[j] = 0.0
            continue
        total = 0.0
        last = len(edges) - 2
        for i, (a, e) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if b > 0.0 and i == last and e == 1.0:
                # 1 - t = w^p: (1 - t)^(-b) dt = p dw, no node on t = 1
                p = 1.0 / (1.0 - b)
                w, ww = graded_rule(0.0, (1.0 - a) ** (1.0 - b), n_panels, n_nodes)
                om = w**p
                t = 1.0 - om
                total += p * float(
                    np.dot(ww, (2.0 - om) ** (-b) * _f(sm, c0 + c2 * t * t))
                )
            else:
                t, wt = graded_rule(a, e, n_panels, n_nodes)
                total += float(
                    np.dot(wt, (1.0 - t * t) ** (-b) * _f(sm, c0 + c2 * t * t))
                )
        out[j] = total
    return out.reshape(speeds.shape)
