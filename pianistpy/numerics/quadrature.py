"""Gauss-Legendre nodes and weights on ``[-1, 1]`` and on ``[0, 1]``."""

__all__ = ["gauss_legendre", "gauss_legendre_unit"]

from functools import cache

import numpy as np

from ..exceptions import InputError
from ..typing import Array


@cache
def gauss_legendre(n_nodes: int) -> tuple[Array, Array]:
    r"""Return the nodes and weights of the Gauss-Legendre rule on ``[-1, 1]``.

    The nodes and weights of each node count are computed once per Python
    session and then reused. The arrays are read-only, so that no caller can
    change the nodes or weights that another caller receives.

    Parameters
    ----------
    n_nodes : int
        Number of nodes :math:`n`. `n_nodes` must be 1 or larger.

    Returns
    -------
    nodes : array, shape (n_nodes,)
        Nodes :math:`x_j` in increasing order, read-only.
    weights : array, shape (n_nodes,)
        Weights :math:`w_j`, read-only. The weights sum to 2.

    Raises
    ------
    InputError
        If `n_nodes` is smaller than 1.

    See Also
    --------
    gauss_legendre_unit : The Gauss-Legendre rule mapped to ``[0, 1]``.

    Notes
    -----
    The :math:`n`-point rule integrates every polynomial of degree up to
    :math:`2n - 1` exactly:

    .. math::

        \int_{-1}^{1} f(x)\,{\rm d}x \approx \sum_{j=1}^{n} w_j f(x_j).

    Examples
    --------
    The 2-point rule integrates :math:`x^2` over ``[-1, 1]`` to 2/3:

    >>> from pianistpy.numerics import gauss_legendre
    >>> nodes, weights = gauss_legendre(2)
    >>> float(weights.sum())
    2.0
    >>> round(float(weights @ nodes**2), 15)
    0.666666666666667
    """
    if n_nodes < 1:
        raise InputError(f"n_nodes must be at least 1; got {n_nodes}.")
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@cache
def gauss_legendre_unit(n_nodes: int) -> tuple[Array, Array]:
    r"""Return the nodes and weights of the Gauss-Legendre rule on ``[0, 1]``.

    The nodes and weights are those of `gauss_legendre` mapped from
    ``[-1, 1]`` to ``[0, 1]``. The nodes and weights of each node count are
    computed once per Python session and then reused, as read-only arrays.

    Parameters
    ----------
    n_nodes : int
        Number of nodes :math:`n`. `n_nodes` must be 1 or larger.

    Returns
    -------
    nodes : array, shape (n_nodes,)
        Nodes :math:`t_j = (x_j + 1)/2` in increasing order, read-only, with
        :math:`x_j` the nodes of `gauss_legendre`.
    weights : array, shape (n_nodes,)
        Weights :math:`w_j/2`, read-only, with :math:`w_j` the weights of
        `gauss_legendre`. The weights sum to 1.

    Raises
    ------
    InputError
        If `n_nodes` is smaller than 1.

    See Also
    --------
    gauss_legendre : The Gauss-Legendre rule on ``[-1, 1]``.

    Notes
    -----
    The :math:`n`-point rule integrates every polynomial of degree up to
    :math:`2n - 1` exactly:

    .. math::

        \int_{0}^{1} f(t)\,{\rm d}t \approx \sum_{j=1}^{n} \frac{w_j}{2}\,
        f(t_j).

    Examples
    --------
    The 1-point rule on ``[0, 1]`` has its node at 1/2 and its weight 1:

    >>> from pianistpy.numerics import gauss_legendre_unit
    >>> nodes, weights = gauss_legendre_unit(1)
    >>> nodes.tolist(), weights.tolist()
    ([0.5], [1.0])
    """
    nodes, weights = gauss_legendre(n_nodes)
    nodes_unit = 0.5 * (nodes + 1.0)
    weights_unit = 0.5 * weights
    nodes_unit.setflags(write=False)
    weights_unit.setflags(write=False)
    return nodes_unit, weights_unit
