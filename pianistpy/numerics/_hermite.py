"""Cubic Hermite interpolation (`hermite`) and its inverse (`hermite_inverse`)."""

__all__ = ["hermite", "hermite_inverse"]

import numpy as np

from ..typing import Array, ArrayLike

# Number of bisections of a root's bracket in hermite_inverse. Sixty halvings
# shrink the bracket's width by the factor 2^-60, about 8.7e-19, which is
# below the relative spacing 2^-52 of doubles. The bisection thereby serves
# the 1e-6 bound with a wide margin.
BISECTIONS = 60
# Number of Newton steps that refine the bisection's midpoint toward the root
# of the interpolant.
NEWTON_STEPS = 2
# The Newton step divides by the interpolant's slope, which is clamped from
# below at SLOPE_MIN so that a zero slope gives no division by zero.
SLOPE_MIN = 1e-300
# A Newton step is taken only when the step's size is below the width of the
# final bracket plus BRACKET_SLACK, so that a Newton step never moves a root
# out of the root's bracket by more than BRACKET_SLACK.
BRACKET_SLACK = 1e-12


def hermite(
    x: ArrayLike, x_nodes: Array, f_nodes: Array, df_dx_nodes: Array
) -> tuple[Array, Array]:
    r"""Return the cubic Hermite interpolant and its derivative at given points.

    Between two neighboring nodes, the interpolant is the cubic polynomial
    that takes the given values and the given slopes at both nodes. Below
    the first node and above the last node, the interpolant is the straight
    line through the end node with the slope given at the end node. Between
    the first and the last node, the interpolant reproduces every cubic
    polynomial exactly.

    Parameters
    ----------
    x : float or array-like
        Points :math:`x` at which the interpolant is evaluated, in the unit
        of `x_nodes`.
    x_nodes : array, shape (n,)
        Nodes :math:`x_i`, strictly increasing, with :math:`n \ge 2`.
    f_nodes : array, shape (n,)
        Values :math:`f_i` of the interpolant at the nodes, in the unit of
        :math:`f`.
    df_dx_nodes : array, shape (n,)
        Slopes :math:`f'_i` of the interpolant at the nodes, in the unit of
        `f_nodes` per unit of `x_nodes`.

    Returns
    -------
    f : array
        The interpolant at `x`, with the shape of `x`.
    df_dx : array
        The derivative of the interpolant in :math:`x` at `x`, with the
        shape of `x`.

    See Also
    --------
    hermite_inverse : The points where an increasing interpolant takes
        given values.

    Notes
    -----
    On the interval :math:`x_i \le x \le x_{i+1}`, with
    :math:`h = x_{i+1} - x_i` and :math:`t = (x - x_i)/h`, the interpolant
    is

    .. math::

        f(x) = (2t^3 - 3t^2 + 1) f_i + (t^3 - 2t^2 + t)\,h f'_i
        + (-2t^3 + 3t^2) f_{i+1} + (t^3 - t^2)\,h f'_{i+1}.

    Examples
    --------
    The cubic :math:`x^3` is reproduced exactly from its values and slopes
    at the two nodes 0 and 2:

    >>> import numpy as np
    >>> from pianistpy.numerics import hermite
    >>> x_nodes = np.array([0.0, 2.0])
    >>> f, df_dx = hermite(1.0, x_nodes, x_nodes**3, 3.0 * x_nodes**2)
    >>> float(f), float(df_dx)
    (1.0, 3.0)
    """
    x = np.asarray(x, dtype=np.float64)
    i = np.clip(np.searchsorted(x_nodes, x, side="right") - 1, 0, x_nodes.size - 2)
    h = x_nodes[i + 1] - x_nodes[i]
    t = (x - x_nodes[i]) / h
    t2, t3 = t * t, t * t * t
    f = (
        (2 * t3 - 3 * t2 + 1) * f_nodes[i]
        + (t3 - 2 * t2 + t) * h * df_dx_nodes[i]
        + (-2 * t3 + 3 * t2) * f_nodes[i + 1]
        + (t3 - t2) * h * df_dx_nodes[i + 1]
    )
    df_dx = (
        (6 * t2 - 6 * t) / h * f_nodes[i]
        + (3 * t2 - 4 * t + 1) * df_dx_nodes[i]
        + (-6 * t2 + 6 * t) / h * f_nodes[i + 1]
        + (3 * t2 - 2 * t) * df_dx_nodes[i + 1]
    )
    below, above = x < x_nodes[0], x > x_nodes[-1]
    f = np.where(
        below,
        f_nodes[0] + df_dx_nodes[0] * (x - x_nodes[0]),
        np.where(above, f_nodes[-1] + df_dx_nodes[-1] * (x - x_nodes[-1]), f),
    )
    df_dx = np.where(below, df_dx_nodes[0], np.where(above, df_dx_nodes[-1], df_dx))
    return f, df_dx


def hermite_inverse(
    f: ArrayLike, x_nodes: Array, f_nodes: Array, df_dx_nodes: Array
) -> tuple[Array, float]:
    """Return the points where an increasing cubic Hermite interpolant takes values f.

    The interpolant is the cubic Hermite interpolant that `hermite`
    evaluates. Each point is found between the two neighboring nodes whose
    values enclose the given value, or on the straight line beyond an end
    node for a value outside the values at the nodes. For a value between
    the values at two neighboring nodes, the returned point lies between
    the two nodes, or outside the two nodes by less than :math:`10^{-12}`
    plus :math:`2^{-60}` times the distance between the two nodes, in the
    unit of `x_nodes`.

    Parameters
    ----------
    f : float or array-like
        Values of the interpolant for which the points are sought, in the
        unit of `f_nodes`.
    x_nodes : array, shape (n,)
        Nodes, strictly increasing, with n of 2 or more.
    f_nodes : array, shape (n,)
        Values of the interpolant at the nodes, increasing.
    df_dx_nodes : array, shape (n,)
        Slopes of the interpolant at the nodes, in the unit of `f_nodes`
        per unit of `x_nodes`. The slopes at the first and the last node
        must be positive.

    Returns
    -------
    x : array
        Points at which the interpolant equals `f`, with the shape of `f`,
        in the unit of `x_nodes`.
    residual : float
        Largest absolute difference between the interpolant at `x` and `f`,
        in the unit of `f_nodes`.

    See Also
    --------
    hermite : The interpolant and its derivative.

    Examples
    --------
    The interpolant of the line :math:`f = 2x` takes the value 1.5 at
    :math:`x = 0.75`:

    >>> import numpy as np
    >>> from pianistpy.numerics import hermite_inverse
    >>> x_nodes = np.array([0.0, 1.0, 2.0])
    >>> x, residual = hermite_inverse(1.5, x_nodes, 2.0 * x_nodes, np.full(3, 2.0))
    >>> float(x), residual
    (0.75, 0.0)
    """
    f = np.asarray(f, dtype=np.float64)
    i = np.clip(np.searchsorted(f_nodes, f, side="right") - 1, 0, x_nodes.size - 2)
    x_lo = np.where(
        f < f_nodes[0], x_nodes[0] + (f - f_nodes[0]) / df_dx_nodes[0] - 1.0, x_nodes[i]
    )
    x_hi = np.where(
        f > f_nodes[-1],
        x_nodes[-1] + (f - f_nodes[-1]) / df_dx_nodes[-1] + 1.0,
        x_nodes[i + 1],
    )
    for _ in range(BISECTIONS):
        x_mid = 0.5 * (x_lo + x_hi)
        f_mid, _ = hermite(x_mid, x_nodes, f_nodes, df_dx_nodes)
        is_below = f_mid < f
        x_lo, x_hi = np.where(is_below, x_mid, x_lo), np.where(is_below, x_hi, x_mid)
    x = 0.5 * (x_lo + x_hi)
    for _ in range(NEWTON_STEPS):
        f_x, df_dx = hermite(x, x_nodes, f_nodes, df_dx_nodes)
        step = (f_x - f) / np.maximum(df_dx, SLOPE_MIN)
        x = np.where(np.abs(step) < (x_hi - x_lo) + BRACKET_SLACK, x - step, x)
    f_x, _ = hermite(x, x_nodes, f_nodes, df_dx_nodes)
    return x, float(np.max(np.abs(f_x - f)))
