r"""A velocity distribution given at the nodes of Gauss-Legendre sub-panels,
read at any velocity between the nodes.

Three callers use this module. `pianistpy.aperture_losvd` reads an aperture's
line-of-sight velocity distribution at user-specified velocities.
`pianistpy.likelihood` integrates a binned velocity distribution over each
bin. `Differentiable` carries a cotangent of the aperture distribution back to
the values at the nodes.

On a sub-panel :math:`[a, b]` with :math:`m` nodes, the distribution is the
polynomial of degree :math:`m - 1` through the :math:`m` node values,

.. math::

    D(v) = \sum_{j=0}^{m-1} c_j P_j(y), \qquad y = \frac{2v - a - b}{b - a},

with :math:`P_j` the Legendre polynomial of degree :math:`j` and :math:`c_j`
the coefficients that reproduce the node values. The :math:`m`-point
Gauss-Legendre rule integrates this polynomial exactly, so the integral of
:math:`D` over a sub-panel equals the quadrature sum over the sub-panel's
nodes. The module stores the Legendre series of the antiderivative of
:math:`D` in :math:`v`, which is zero at :math:`v = a`.

A set of nodes and weights counts as a rule of Gauss-Legendre sub-panels when
two conditions hold: every node differs from the Gauss-Legendre node by at
most :math:`10^{-12}` times the largest absolute node value; and every weight
differs from the Gauss-Legendre weight by at most :math:`10^{-12}` times the
Gauss-Legendre weight. Notes Section T:6.6 treats the distribution in an
aperture."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from ..typing import Array

# A velocity this close to a sub-panel's end, relative to the larger of the
# sub-panel's own ends, is read as inside the sub-panel: the ends come from the
# nodes and weights in floating point, and the first sub-panel's lower end,
# zero, comes out as a residue of order 1e-24 (the same relative tolerance as
# the rule's own matching).
EDGE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Panels:
    """A velocity distribution on Gauss-Legendre sub-panels.

    Attributes
    ----------
    a : array, shape (n_panels,)
        The lower end of each sub-panel.
    b : array, shape (n_panels,)
        The upper end of each sub-panel.
    nodes : array, shape (n_panels * m,)
        The velocity nodes, ``m`` nodes on each sub-panel, in sub-panel order.
    values : array, shape (n_panels * m,)
        The distribution's value at each node.
    series : array, shape (n_panels, m + 1)
        For each sub-panel, the Legendre coefficients in
        ``y = (2 v - a - b) / (b - a)`` of the antiderivative of the
        distribution in ``v``. Each antiderivative is zero at the
        sub-panel's lower end ``a``.
    """

    a: Array
    b: Array
    nodes: Array
    values: Array
    series: Array


def panels(nodes: Array, weights: Array, values: Array) -> Panels:
    """Return the distribution given by `values` at `nodes` as `Panels`.

    The sub-panels are found from `nodes` and `weights` alone. The function
    tries each node count ``m`` from 2 upward that divides the number of
    nodes. A count ``m`` is accepted when every group of ``m`` consecutive
    nodes and weights is the ``m``-point Gauss-Legendre rule on
    ``[c - h, c + h]``, with ``c`` the midpoint of the group's first and last
    node and ``h`` half the sum of the group's weights. A node matches when
    it differs from the Gauss-Legendre node by at most ``1e-12`` times the
    largest absolute node value. A weight matches when it differs from the
    Gauss-Legendre weight by at most ``1e-12`` times the Gauss-Legendre
    weight.

    Raises
    ------
    ValueError
        If no node count ``m`` makes every group a Gauss-Legendre rule.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(nodes)))
    for m in range(2, nodes.size + 1):
        if nodes.size % m:
            continue
        x, w = legendre.leggauss(m)
        v, wv = nodes.reshape(-1, m), weights.reshape(-1, m)
        half = 0.5 * wv.sum(axis=1)
        center = 0.5 * (v[:, 0] + v[:, -1])
        if np.allclose(
            v, center[:, None] + half[:, None] * x, rtol=0.0, atol=1e-12 * scale
        ) and np.allclose(wv, half[:, None] * w, rtol=1e-12, atol=0.0):
            return _fitted(center - half, center + half, nodes, values)
    raise ValueError(
        f"No node count per panel makes the {nodes.size} nodes a rule of "
        "Gauss-Legendre panels."
    )


def _fitted(a: Array, b: Array, nodes: Array, values: Array) -> Panels:
    """Return the distribution given by `values` at `nodes` on the sub-panels
    ``[a, b]`` as `Panels`, with the same number of nodes on every sub-panel.
    """
    n_panels = a.size
    m = nodes.size // n_panels
    half = 0.5 * (b - a)
    y = (nodes.reshape(n_panels, m) - 0.5 * (a + b)[:, None]) / half[:, None]
    series = np.empty((n_panels, m + 1))
    for p in range(n_panels):
        c = np.linalg.solve(
            legendre.legvander(y[p], m - 1), values.reshape(n_panels, m)[p]
        )
        series[p] = legendre.legint(c, lbnd=-1.0) * half[p]
    return Panels(a=a, b=b, nodes=nodes, values=values, series=series)


def read_panels(p: Panels, v: Array) -> Array:
    """Return the distribution `p` at velocities `v`.

    A velocity inside a sub-panel takes the value of the sub-panel's
    polynomial through its node values. A velocity outside every sub-panel
    takes the value zero. The sub-panels' ends are inferred from the nodes
    and weights in floating point (`panels`), so a velocity that no sub-panel
    contains but that lies within `EDGE_TOLERANCE` times the larger of a
    sub-panel's ends of that end is read from that sub-panel: the velocity
    zero, at the lower end of a distribution's first sub-panel, reads that
    sub-panel. A velocity a sub-panel contains is read from it exactly, so no
    value a sub-panel serves depends on the tolerance.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape)
    # The loop runs from the last sub-panel to the first. A velocity at the end
    # that two sub-panels share therefore takes the lower sub-panel's value.
    # The second pass places only the velocities the first pass left out.
    taken = np.zeros(v.shape, dtype=bool)
    for slack in (0.0, EDGE_TOLERANCE):
        for i in reversed(range(p.a.size)):
            tol = slack * max(abs(float(p.a[i])), abs(float(p.b[i])))
            inside = (v >= p.a[i] - tol) & (v <= p.b[i] + tol) & ~taken
            if np.any(inside):
                half = 0.5 * (p.b[i] - p.a[i])
                y = (v[inside] - 0.5 * (p.a[i] + p.b[i])) / half
                out[inside] = legendre.legval(y, legendre.legder(p.series[i]) / half)
        taken |= _contained(p, v, slack)
    return out


def _contained(p: Panels, v: Array, slack: float) -> Array:
    """Return which velocities some sub-panel contains, within `slack` times the
    larger of the sub-panel's ends."""
    tol = slack * np.maximum(np.abs(p.a), np.abs(p.b))
    return ((v[..., None] >= p.a - tol) & (v[..., None] <= p.b + tol)).any(axis=-1)


def read_panels_vjp(p: Panels, v: Array, cotangent: Array) -> Array:
    """Return the vector-Jacobian product of `read_panels` in the node values.

    The input `cotangent` holds one value per velocity in `v`. The output
    holds one value per node of `p`, in the order of ``p.nodes``. The value
    that `read_panels` gives inside a sub-panel is linear in that sub-panel's
    node values, so the product is exact. A velocity at the end that two
    sub-panels share is assigned to the lower sub-panel, as in `read_panels`,
    and a velocity no sub-panel contains is placed by the same tolerance.
    """
    v = np.asarray(v, dtype=np.float64)
    c = np.asarray(cotangent, dtype=np.float64)
    m = p.nodes.size // p.a.size
    out = np.zeros((p.a.size, m))
    taken = np.zeros(v.shape, dtype=bool)
    for slack in (0.0, EDGE_TOLERANCE):
        for i in range(p.a.size):
            tol = slack * max(abs(float(p.a[i])), abs(float(p.b[i])))
            inside = (v >= p.a[i] - tol) & (v <= p.b[i] + tol) & ~taken
            taken |= inside
            if np.any(inside):
                half = 0.5 * (p.b[i] - p.a[i])
                center = 0.5 * (p.a[i] + p.b[i])
                y_nodes = (p.nodes.reshape(-1, m)[i] - center) / half
                y = (v[inside] - center) / half
                at_v = legendre.legvander(y, m - 1)
                out[i] += np.linalg.solve(
                    legendre.legvander(y_nodes, m - 1).T, at_v.T @ c[inside]
                )
    return out.ravel()
