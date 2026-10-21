r"""Gauss-Legendre quadrature and cubic Hermite interpolation.

A user calls `gauss_legendre` or `gauss_legendre_unit` for the nodes
:math:`x_j` and weights :math:`w_j` of the :math:`n`-point Gauss-Legendre
rule, which integrates a polynomial of degree up to :math:`2n - 1` exactly:

.. math::

    \int_{-1}^{1} f(x)\,{\rm d}x \approx \sum_{j=1}^{n} w_j f(x_j).

A user calls `hermite` for the piecewise cubic interpolant that takes given
values and slopes at given nodes, and `hermite_inverse` for the points where
an increasing piecewise cubic Hermite interpolant takes given values.
"""

__all__ = [
    "gauss_legendre",
    "gauss_legendre_unit",
    "hermite",
    "hermite_inverse",
]

from .hermite import hermite, hermite_inverse
from .quadrature import gauss_legendre, gauss_legendre_unit
