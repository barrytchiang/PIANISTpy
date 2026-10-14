"""Derivatives of a function known through the derivatives of its logarithm."""

__all__ = ["bell_partitions", "derivative_from_log"]

from collections import Counter
from collections.abc import Sequence
from functools import cache
from math import factorial

import numpy as np

from ..exceptions import InputError
from ..typing import Array


@cache
def bell_partitions(order: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return the terms of the complete exponential Bell polynomial.

    Parameters
    ----------
    order : int
        Order of the polynomial, at least 1.

    Returns
    -------
    terms : tuple of (int, tuple of int)
        One entry per partition of `order` into positive integers: its
        coefficient and the parts, in decreasing order. The polynomial is
        the sum over the entries of the coefficient times the product of the
        variables indexed by the parts.

    Examples
    --------
    >>> bell_partitions(3)
    ((1, (3,)), (3, (2, 1)), (1, (1, 1, 1)))
    >>> sum(coefficient for coefficient, _ in bell_partitions(4))
    15
    """
    if order < 1:
        raise InputError(f"order must be at least 1; got {order}.")

    def partitions(n, largest):
        if n == 0:
            yield ()
            return
        for part in range(min(n, largest), 0, -1):
            for rest in partitions(n - part, part):
                yield (part, *rest)

    terms = []
    for partition in partitions(order, order):
        denominator = 1
        for part, multiplicity in Counter(partition).items():
            denominator *= factorial(part) ** multiplicity * factorial(multiplicity)
        terms.append((factorial(order) // denominator, partition))
    return tuple(terms)


def derivative_from_log(log_derivatives: Sequence[Array], values: Array) -> Array:
    r"""Return a derivative of a positive function from those of its logarithm.

    Parameters
    ----------
    log_derivatives : sequence of array, length N
        The derivatives of orders 1 to N of the logarithm of the function,
        each at the same points.
    values : array
        The function itself at those points.

    Returns
    -------
    derivative : array
        The derivative of order N of the function.

    Notes
    -----
    With :math:`S = \ln f`, the chain rule of Faà di Bruno gives

    .. math::

        \frac{\mathrm{d}^N f}{\mathrm{d}x^N}
            = f\,Y_N\!\left(S', S'', \dots, S^{(N)}\right),

    where :math:`Y_N` is the complete exponential Bell polynomial.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.5, 1.0])
    >>> f = np.exp(3.0 * x)
    >>> ones, zeros = np.ones_like(x), np.zeros_like(x)
    >>> bool(np.allclose(derivative_from_log([3 * ones, zeros], f), 9.0 * f))
    True
    """
    order = len(log_derivatives)
    total = np.zeros_like(values)
    for coefficient, partition in bell_partitions(order):
        term = float(coefficient) * np.ones_like(values)
        for part, multiplicity in Counter(partition).items():
            term = term * (log_derivatives[part - 1] ** multiplicity)
        total += term
    return values * total
