"""Quadrature nodes and weights, computed once and reused."""

__all__ = ["gauss_legendre", "gauss_legendre_unit"]

from functools import cache

import numpy as np

from ..exceptions import InputError
from ..typing import Array


@cache
def gauss_legendre(n_nodes: int) -> tuple[Array, Array]:
    """Return the Gauss-Legendre nodes and weights on ``[-1, 1]``.

    The nodes of a given count are computed once per session and returned as
    read-only arrays, so that no caller can change them for another.

    Parameters
    ----------
    n_nodes : int
        Number of nodes, at least 1.

    Returns
    -------
    nodes : array, shape (n_nodes,)
        Nodes in increasing order.
    weights : array, shape (n_nodes,)
        Weights; they sum to 2.

    Examples
    --------
    >>> nodes, weights = gauss_legendre(2)
    >>> float(weights.sum())
    2.0
    """
    if n_nodes < 1:
        raise InputError(f"n_nodes must be at least 1; got {n_nodes}.")
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@cache
def gauss_legendre_unit(n_nodes: int) -> tuple[Array, Array]:
    """Return the Gauss-Legendre nodes and weights on ``[0, 1]``.

    Parameters
    ----------
    n_nodes : int
        Number of nodes, at least 1.

    Returns
    -------
    nodes : array, shape (n_nodes,)
        Nodes in increasing order.
    weights : array, shape (n_nodes,)
        Weights; they sum to 1.
    """
    nodes, weights = gauss_legendre(n_nodes)
    nodes_unit = 0.5 * (nodes + 1.0)
    weights_unit = 0.5 * weights
    nodes_unit.setflags(write=False)
    weights_unit.setflags(write=False)
    return nodes_unit, weights_unit
