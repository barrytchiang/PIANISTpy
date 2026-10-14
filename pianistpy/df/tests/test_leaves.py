"""Tests of the leaves of the DF subpackage."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import InputError
from pianistpy.df.asymptotics import (
    scaled_upper_gamma,
    tail_mass_below,
)
from pianistpy.df.inversion import (
    forward_prefactor,
    inversion_prefactor,
    order_pair,
)

BETA_C = [0.75, 0.6, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -2.5]


@pytest.mark.parametrize(
    ("beta_c", "expected"),
    [
        (0.0, (1, 0.5, 1, 0.5)),
        (0.5, (1, 0.0, 1, 0.0)),
        (-0.5, (2, 0.0, 2, 0.0)),
        (-0.25, (1, 0.75, 2, -0.25)),
        (0.75, (0, 0.75, 1, -0.25)),
    ],
)
def test_order_pair(beta_c, expected):
    pair = order_pair(beta_c)
    assert (pair.order_m, pair.order_n, pair.order_j, pair.kernel_exponent) == expected
    assert -0.5 < pair.kernel_exponent <= 0.5
    with pytest.raises(InputError):
        order_pair(1.0)


def test_prefactors_at_their_known_values():
    # Isotropic: 1 / (sqrt(8) pi^2) and 4 pi; beta_c = 1/2: 1 / (2 pi^2), 2 pi^2.
    assert_allclose(inversion_prefactor(0.5, 0.0), 1.0 / (math.sqrt(8.0) * math.pi**2))
    assert_allclose(forward_prefactor(0.0), 4.0 * math.pi)
    assert inversion_prefactor(0.0, 0.5) == 1.0 / (2.0 * math.pi**2)
    assert_allclose(forward_prefactor(0.5), 2.0 * math.pi**2)
    assert_allclose(forward_prefactor(-0.5), math.pi**2)


def test_tail_mass_of_a_pure_power_law():
    k = np.array([1e-8, 1e-4])
    assert_allclose(tail_mass_below(2.0, 0.5, k), 2.0 * k**0.5 / 0.5, rtol=1e-15)


@pytest.mark.parametrize("b", [1.0086, 1.0111, 1.5])
def test_the_tail_mass_in_a_kernel_is_the_numpy_one(b):
    # the one function in NumPy and jitted in jax.numpy: the same algorithm,
    # its loops' exits by the same masks, so the two differ by rounding
    # (6e-13 at these orders)
    import jax
    import jax.numpy as jnp

    k = np.geomspace(1e-12, 0.9, 200)
    kernel = jax.jit(lambda z: tail_mass_below(1.0, 0.5, z, b, 1.0, jnp))
    assert_allclose(
        np.asarray(kernel(jnp.asarray(k))),
        tail_mass_below(1.0, 0.5, k, b, 1.0),
        rtol=1e-11,
    )


def test_the_scaled_gamma_at_order_minus_one_half_is_its_closed_form():
    # Gamma(-1/2, x) = 2 e^-x / sqrt(x) - 2 sqrt(pi) erfc(sqrt(x)), scaled by
    # e^x: the continued fraction above x = 1 and the series below it (the
    # closed form's own two terms cancel as x grows: 5e-14 at 500)
    from scipy.special import erfcx

    x = np.concatenate(
        [np.geomspace(1e-4, 1.0, 50, endpoint=False), np.geomspace(1.0, 30.0, 200)]
    )
    exact = 2.0 / np.sqrt(x) - 2.0 * math.sqrt(math.pi) * erfcx(np.sqrt(x))
    assert_allclose(scaled_upper_gamma(-0.5, x), exact, rtol=1e-13)


@pytest.mark.parametrize("s", [-0.0111, -0.0086, -0.25])
def test_the_scaled_gamma_at_a_small_negative_order_holds_its_digits(s):
    # the recurrence Gamma(s, x) = [Gamma(s + 1, x) - x^s e^-x] / s
    # read 1e-11 relative at s = -0.01; at the order itself, rounding
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    x = np.concatenate(
        [np.geomspace(1e-3, 1.0, 40, endpoint=False), np.linspace(1.0, 60.0, 120)]
    )
    exact = np.array([float(mp.e ** mp.mpf(v) * mp.gammainc(s, mp.mpf(v))) for v in x])
    assert_allclose(scaled_upper_gamma(s, x), exact, rtol=1e-13)
