r"""The angle integral of the local velocity distribution of a model.

`pdf_v` reads the local velocity distribution through
`stages.velocity_distribution`, which calls `_velocity.angle_integral`.
The distribution at the 3D radius :math:`r` and the velocity :math:`V` is

.. math::

    p(V\,|\,r) = \frac{4\pi}{\varrho(r)}\, r^{-2b} V^{2-2b}
        \int_0^1 (1-t^2)^{-b} F(Q)\,{\rm d}t, \qquad
    Q = \mathcal{P}(r) - \frac{V^2}{2}(1 + s\xi) + \frac{s\xi V^2}{2}\,t^2.

The symbols are the following. :math:`\varrho(r)` is the density and
:math:`\mathcal{P}(r)` the relative potential at :math:`r`. :math:`F` is the
distribution function (DF) of the model's one component, and :math:`Q` is
the DF's energy variable. :math:`t = \cos\eta`, with :math:`\eta` the angle
between the velocity and the radius vector. :math:`\xi = (r/r_{\rm a})^2`,
with :math:`r_{\rm a}` the anisotropy radius. The sign :math:`s` is
:math:`+1` for a Type I model, :math:`-1` for a Type II model and 0 for a
model without an anisotropy radius. The exponent :math:`b` is the central
anisotropy :math:`\beta_{\rm c}` for the constant-anisotropy and
Cuddeford-Osipkov-Merritt (COM) models, and 0 for every other model.

The DF's lookup reads :math:`F` on the DF's support :math:`Q > K_c`, and
:math:`F` is zero outside the support. :math:`K_c` is the lookup's floor
:math:`K_0` where the DF is zero below :math:`K_0`, and 0 otherwise.
:math:`Q` is quadratic in :math:`t`, so the edge of the support in
:math:`t` is an end of an integration piece and never lies inside a piece.
The interval in :math:`t` is also split where :math:`Q` equals the DF
table's first energy :math:`K_w`. Below :math:`K_w` the lookup reads the
DF's form below the table instead of the table. For :math:`b > 0`, the
piece that ends at :math:`t = 1` is integrated in the variable :math:`w`
defined by :math:`1 - t = w^p`, with :math:`p = 1/(1 - b)`. In :math:`w`,
:math:`(1-t)^{-b}{\rm d}t = p\,{\rm d}w`, so the integrable singularity of
the kernel :math:`(1-t^2)^{-b}` at :math:`t = 1` is removed.

The tolerance of the distribution is the setting ``velocity_tolerance``, which
`stages.velocity_distribution` applies (:math:`10^{-6}` in the default
settings). Chiang+(2026d), Eq. (B40), and Notes Eq. (T:6.4) state the
distribution.
"""

__all__ = [
    "VelocityModel",
    "angle_integral",
    "angle_pieces",
    "graded_rule",
    "velocity_breaks",
    "velocity_model",
]

import functools
import math
from dataclasses import dataclass

import numpy as np

from ..df import Model
from ..df.lookup import K_0, LookupArrays, LookupKinds, evaluate_lookup, prepare_lookup
from ..typing import Array, ArrayLike
from ._tables import _kernel_beta

# The width of the smallest panel of graded_rule next to each end of an
# interval, as a fraction of half the interval's length.
GRADE_MIN = 1e-12


@dataclass(frozen=True)
class VelocityModel:
    r"""The quantities of a one-component model that `angle_integral` reads.

    Parameters
    ----------
    arrays : LookupArrays
        The arrays of the DF's prepared lookup (`prepare_lookup`).
    kinds : LookupKinds
        The kinds of the DF's prepared lookup (`prepare_lookup`).
    k_c : float
        The DF's floor :math:`K_c`: the lookup's floor :math:`K_0` where the
        DF is zero below :math:`K_0`, and 0 otherwise. Internal units.
    k_w : float
        The DF table's first energy :math:`K_w`, below which the lookup reads
        the DF's form below the table. Internal units.
    b : float
        The exponent :math:`b` of the factor :math:`(L^2)^{-b}`: the central
        anisotropy :math:`\beta_{\rm c}` for the constant-anisotropy and COM
        models, and 0 for every other model.
    sign : float
        The sign :math:`s`: :math:`+1` for a Type I model, :math:`-1` for a
        Type II model, and 0 for a model without an anisotropy radius.
    r_a : float
        The anisotropy radius :math:`r_{\rm a}`, in internal units. Infinite
        for a model without an anisotropy radius.
    """

    arrays: LookupArrays
    kinds: LookupKinds
    k_c: float
    k_w: float
    b: float
    sign: float
    r_a: float


def velocity_model(model: Model) -> VelocityModel:
    """Return the `VelocityModel` of a model, read from the model's first
    component."""
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    anisotropy = model.components[0].anisotropy
    table = model.table
    return VelocityModel(
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
    """Return the panel grading and the Gauss-Legendre rule on [-1, 1] that
    `graded_rule` uses.

    The result is cached once for each pair of `n_panels` and `n_nodes`, and
    its three arrays are read-only."""
    grade = np.geomspace(GRADE_MIN, 1.0, n_panels)
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    for a in (grade, x, w):
        a.setflags(write=False)
    return grade, x, w


def graded_rule(a: float, b: float, n_panels: int, n_nodes: int) -> tuple[Array, Array]:
    """Return the nodes and weights of a graded Gauss-Legendre rule on [a, b].

    The rule splits [a, b] at its midpoint. Each half holds `n_panels` panels
    whose edges are graded geometrically toward the end of [a, b] on that
    side. The smallest panel next to each end has the width `GRADE_MIN` times
    half the length of [a, b]. Each panel takes `n_nodes` Gauss-Legendre
    nodes. Where b is not above a, the nodes and the weights are empty
    arrays."""
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


def velocity_breaks(
    sm: VelocityModel, psi: float, xi: float
) -> tuple[float, float, float]:
    r"""Return the three break velocities :math:`(V_{\rm esc}, V_*, V_{\rm max})`
    of the velocity distribution at one radius.

    The argument `psi` is the relative potential :math:`\mathcal{P}` at the
    radius, and the argument `xi` is :math:`\xi = (r/r_{\rm a})^2`. With
    :math:`A = \mathcal{P} - K_c`, the three velocities are the following.

    - :math:`V_{\rm esc} = \sqrt{2\mathcal{P}}` is the escape velocity.
    - :math:`V_*` is the velocity above which the DF's support covers only part
      of the interval :math:`0 \le t \le 1`. :math:`V_* = \sqrt{2A/(1+\xi)}`
      for a Type I model and :math:`V_* = \sqrt{2A}` for a Type II model.
      :math:`V_* = V_{\rm max}` for a model without an anisotropy radius.
    - :math:`V_{\rm max}` is the velocity above which the DF's support is
      empty. :math:`V_{\rm max} = \sqrt{2A}`, except for a Type II model,
      where :math:`V_{\rm max} = \sqrt{2A/(1-\xi)}`.
    """
    a = psi - sm.k_c
    v_esc = math.sqrt(2.0 * psi)
    if sm.sign > 0.0:
        return v_esc, math.sqrt(2.0 * a / (1.0 + xi)), math.sqrt(2.0 * a)
    if sm.sign < 0.0:
        return v_esc, math.sqrt(2.0 * a), math.sqrt(2.0 * a / (1.0 - xi))
    v_max = math.sqrt(2.0 * a)
    return v_esc, v_max, v_max


def _f(sm: VelocityModel, q: Array) -> Array:
    """Return the DF F at the energies `q`.

    F is zero where `q` is at or below the floor :math:`K_c` and where the
    lookup's value of log10 F is not finite."""
    log_f = np.asarray(evaluate_lookup(sm.arrays, sm.kinds, np.maximum(q, 0.0)))
    return np.where((q > sm.k_c) & np.isfinite(log_f), 10.0**log_f, 0.0)


def angle_pieces(
    sm: VelocityModel, psi: float, xi: float, v: float
) -> tuple[float, float, list[float], list[str]]:
    """Return the coefficients of Q and the pieces of the angle integral at
    one velocity.

    The return value is ``(c0, c2, edges, labels)``. The coefficients `c0`
    and `c2` define :math:`Q = c_0 + c_2 t^2` at the velocity `v`. The list
    `edges` holds the edges of the integration pieces in t, in increasing
    order. The list `labels` gives, for each edge, the condition that sets
    the edge. The labels are the following.

    - ``"0"`` labels the end :math:`t = 0` of the interval.
    - ``"1"`` labels the end :math:`t = 1` of the interval.
    - ``"root"`` labels the edge of the DF's support,
      :math:`t = \\sqrt{(K_c - c_0)/c_2}`.
    - ``"join"`` labels the point where Q equals the DF table's first
      energy, :math:`t = \\sqrt{(K_w - c_0)/c_2}`.

    Where :math:`c_2 = 0` and :math:`c_0 \\le K_c`, the support is empty, and
    `edges` and `labels` are empty lists."""
    c0 = psi - 0.5 * v * v * (1.0 + sm.sign * xi)
    c2 = 0.5 * sm.sign * xi * v * v
    lo, hi = 0.0, 1.0
    lo_by, hi_by = "0", "1"
    cuts = []
    if c2 != 0.0:
        # The support Q > K_c in t lies above the root t^2 = (K_c - c0) / c2
        # for a Type I model (c2 > 0), and below that root for a Type II
        # model (c2 < 0).
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
    sm: VelocityModel, psi: float, xi: float, v: ArrayLike, n_nodes: int, n_panels: int
) -> Array:
    r"""Return :math:`\int_0^1 (1-t^2)^{-b} F(Q)\,{\rm d}t` at the velocities `v`.

    Parameters
    ----------
    sm : VelocityModel
        The quantities of the model that the integral reads.
    psi : float
        The relative potential :math:`\mathcal{P}` at the radius, in internal
        units.
    xi : float
        :math:`\xi = (r/r_{\rm a})^2` at the radius, and 0 for a model
        without an anisotropy radius.
    v : array-like
        Velocities, in internal units.
    n_nodes : int
        Gauss-Legendre nodes on each panel of `graded_rule` on each piece.
    n_panels : int
        Panels of `graded_rule` on each half of each piece.

    Returns
    -------
    array
        The integral at each velocity, with the shape of `v`. The integral is
        zero at a velocity where the DF's support is empty.
    """
    b = sm.b
    velocities = np.asarray(v, dtype=np.float64)
    out = np.zeros(velocities.size)
    # The lists hold one entry for each piece of each velocity: the index of the
    # velocity, the factor p of a piece integrated in w (None for a piece
    # integrated in t), the weights, the kernel at the nodes, and Q at the
    # nodes. The lookup reads F once, at the nodes of all pieces together.
    owner, scale, weights, kernels, qs = [], [], [], [], []
    for j, vj in enumerate(velocities.ravel()):
        c0, c2, edges, _ = angle_pieces(sm, psi, xi, float(vj))
        last = len(edges) - 2
        for i, (a, e) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if b > 0.0 and i == last and e == 1.0:
                # With 1 - t = w^p, (1 - t)^(-b) dt = p dw, and the kernel
                # (1 - t^2)^(-b) reduces to (2 - w^p)^(-b). No node lies on t = 1.
                p = 1.0 / (1.0 - b)
                w, ww = graded_rule(0.0, (1.0 - a) ** (1.0 - b), n_panels, n_nodes)
                om = w**p
                t = 1.0 - om
                kernels.append((2.0 - om) ** (-b))
            else:
                p = None
                t, ww = graded_rule(a, e, n_panels, n_nodes)
                kernels.append((1.0 - t * t) ** (-b))
            owner.append(j)
            scale.append(p)
            weights.append(ww)
            qs.append(c0 + c2 * t * t)
    if not qs:
        return out.reshape(velocities.shape)
    f = _f(sm, np.concatenate(qs))
    at = 0
    for j, p, ww, kernel in zip(owner, scale, weights, kernels, strict=True):
        piece = float(np.dot(ww, kernel * f[at : at + ww.size]))
        out[j] += piece if p is None else p * piece
        at += ww.size
    return out.reshape(velocities.shape)
