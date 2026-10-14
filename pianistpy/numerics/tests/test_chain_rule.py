"""Tests of the chain rule for a function known through its logarithm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import InputError
from pianistpy.numerics import bell_partitions, derivative_from_log

BELL_NUMBERS = {1: 1, 2: 2, 3: 5, 4: 15, 5: 52, 6: 203}


@pytest.mark.parametrize("order", sorted(BELL_NUMBERS))
def test_coefficients_sum_to_the_bell_number(order):
    terms = bell_partitions(order)
    assert sum(coefficient for coefficient, _ in terms) == BELL_NUMBERS[order]
    assert all(sum(parts) == order for _, parts in terms)
    with pytest.raises(InputError):
        bell_partitions(0)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_derivatives_of_a_gaussian(order):
    # f = exp(-x^2 / 2): S' = -x, S'' = -1, higher derivatives of S vanish, and
    # the derivative of order N is (-1)^N He_N(x) f.
    x = np.linspace(-2.0, 2.0, 9)
    f = np.exp(-0.5 * x * x)
    log_derivatives = [-x, -np.ones_like(x)] + [np.zeros_like(x)] * 3
    hermite = np.polynomial.hermite_e.hermeval(x, [0.0] * order + [1.0])
    assert_allclose(
        derivative_from_log(log_derivatives[:order], f),
        (-1.0) ** order * hermite * f,
        rtol=1e-13,
        atol=1e-15,
    )
