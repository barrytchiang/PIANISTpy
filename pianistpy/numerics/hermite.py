"""Cubic Hermite interpolation on increasing nodes, and its inverse."""

__all__ = ["hermite", "hermite_inverse"]

import numpy as np

from ..typing import Array, ArrayLike

# Halvings of the bracket before the Newton steps: enough to exhaust the
# bits of a double on any bracket.
BISECTIONS = 60
NEWTON_STEPS = 2
# Guards of the Newton step: the slope it divides by, and the slack beyond
# the final bracket within which a step is kept.
SLOPE_MIN = 1e-300
BRACKET_SLACK = 1e-12


def hermite(
    x: ArrayLike, x_nodes: Array, f_nodes: Array, df_dx_nodes: Array
) -> tuple[Array, Array]:
    r"""Return the cubic Hermite interpolant and its derivative.

    Between two nodes the interpolant is the cubic that takes the values and
    the slopes given at both. Outside the nodes it continues as the straight
    line through the end node with the end slope.

    Parameters
    ----------
    x : array-like
        Points of evaluation.
    x_nodes : array, shape (n,)
        Nodes, strictly increasing.
    f_nodes : array, shape (n,)
        Values at the nodes.
    df_dx_nodes : array, shape (n,)
        Slopes at the nodes.

    Returns
    -------
    f : array
        The interpolant at `x`.
    df_dx : array
        Its derivative at `x`.

    Examples
    --------
    A cubic is reproduced exactly from its values and slopes at two nodes:

    >>> import numpy as np
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
    """Return where an increasing cubic Hermite interpolant takes given values.

    The root is bracketed by the two nodes whose values enclose the target,
    or by the straight continuation beyond the end nodes, and found by
    bisection. Newton steps then refine it; a step that would leave the
    bracket is not taken, so a locally steep interpolant cannot throw a root
    out of its interval.

    Parameters
    ----------
    f : array-like
        Values to invert for.
    x_nodes : array, shape (n,)
        Nodes, strictly increasing.
    f_nodes : array, shape (n,)
        Values at the nodes, increasing.
    df_dx_nodes : array, shape (n,)
        Slopes at the nodes, positive at the two end nodes.

    Returns
    -------
    x : array
        Points at which the interpolant equals `f`.
    residual : float
        Largest absolute difference between the interpolant at `x` and `f`.

    Examples
    --------
    >>> import numpy as np
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
