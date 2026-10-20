"""A distribution of velocity on Gauss-Legendre panels, read between its
nodes: each panel's polynomial through its own nodes, the polynomial the
quadrature integrates, written as the Legendre series of its antiderivative.
The rule of an aperture's distribution (`aperture_losvd`) and of the binned
distributions of the likelihood."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from ..typing import Array


@dataclass(frozen=True)
class Panels:
    """A distribution on its sub-panels: their ends, the nodes and the values
    there, and the Legendre coefficients of each sub-panel's antiderivative
    in y = (2 v - a - b) / (b - a), zero at the sub-panel's start."""

    a: Array
    b: Array
    nodes: Array
    values: Array
    series: Array


def panels(nodes: Array, weights: Array, values: Array) -> Panels:
    """The distribution with values at `nodes` of a rule of Gauss-Legendre
    panels with `weights`, as `Panels`: its sub-panels are the rule's panels,
    found from the rule itself (the m with which every m consecutive nodes
    and weights are an m-point Gauss-Legendre rule on [c - h, c + h], h half
    their weights' sum), so that each is read by its polynomial through its
    own nodes, as the quadrature integrates it."""
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
    raise ValueError(f"The {nodes.size} nodes are not a rule of Gauss-Legendre panels.")


def _fitted(a: Array, b: Array, nodes: Array, values: Array) -> Panels:
    """The distribution with values at `nodes` on the sub-panels [a, b], the
    same number of nodes on each, as `Panels`."""
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
    """The distribution at velocities `v` inside its sub-panels, by each
    sub-panel's polynomial through its nodes; zero outside them."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape)
    # at an end two sub-panels share, the first one's polynomial
    for i in reversed(range(p.a.size)):
        inside = (v >= p.a[i]) & (v <= p.b[i])
        if np.any(inside):
            half = 0.5 * (p.b[i] - p.a[i])
            y = (v[inside] - 0.5 * (p.a[i] + p.b[i])) / half
            out[inside] = legendre.legval(y, legendre.legder(p.series[i]) / half)
    return out


def read_panels_vjp(p: Panels, v: Array, cotangent: Array) -> Array:
    """The transpose of `read_panels` in the values: a cotangent on the
    distribution at velocities `v` carried to its values at the nodes. Each
    sub-panel's reading is linear in its own values, through the polynomial
    through them; at an end two sub-panels share, the first one's, as the
    reading takes it."""
    v = np.asarray(v, dtype=np.float64)
    c = np.asarray(cotangent, dtype=np.float64)
    m = p.nodes.size // p.a.size
    out = np.zeros((p.a.size, m))
    taken = np.zeros(v.shape, dtype=bool)
    for i in range(p.a.size):
        inside = (v >= p.a[i]) & (v <= p.b[i]) & ~taken
        taken |= inside
        if np.any(inside):
            half = 0.5 * (p.b[i] - p.a[i])
            center = 0.5 * (p.a[i] + p.b[i])
            y_nodes = (p.nodes.reshape(-1, m)[i] - center) / half
            y = (v[inside] - center) / half
            at_v = legendre.legvander(y, m - 1)
            out[i] = np.linalg.solve(
                legendre.legvander(y_nodes, m - 1).T, at_v.T @ c[inside]
            )
    return out.ravel()
