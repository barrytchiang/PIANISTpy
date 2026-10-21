"""Tests of the JAX PCHIP interpolant against SciPy's PCHIP and closed forms."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from pianistpy.exceptions import InputError
from pianistpy.numerics.pchip import hermite_eval, interp1d, pchip_slopes


def _nodes(seed, n=17, monotone=False):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.05, 1.0, n))
    f = np.cumsum(rng.uniform(0.0, 1.0, n)) if monotone else rng.standard_normal(n)
    return x, f


@pytest.mark.parametrize("monotone", [False, True])
def test_the_slopes_are_scipys(monotone):
    x, f = _nodes(5, monotone=monotone)
    m = np.asarray(pchip_slopes(jnp.asarray(x), jnp.asarray(f)))
    assert np.allclose(m, PchipInterpolator(x, f)(x, 1), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("monotone", [False, True])
def test_the_interpolant_is_scipys_inside_the_nodes(monotone):
    x, f = _nodes(11, monotone=monotone)
    xq = np.linspace(x[0], x[-1], 1001)
    y = np.asarray(interp1d(jnp.asarray(xq), jnp.asarray(x), jnp.asarray(f), 0.0, 0.0))
    assert np.allclose(y, PchipInterpolator(x, f)(xq), rtol=0.0, atol=1e-13)


def test_a_monotone_table_gives_a_monotone_interpolant():
    x, f = _nodes(2, monotone=True)
    xq = np.linspace(x[0], x[-1], 4001)
    y = np.asarray(interp1d(jnp.asarray(xq), jnp.asarray(x), jnp.asarray(f), 0.0, 0.0))
    assert np.all(np.diff(y) >= 0.0)


def test_exact_slopes_reproduce_a_cubic():
    x = np.array([-1.0, 0.3, 1.1, 2.5, 3.0])
    f = 0.5 * x**3 - x**2 + 2.0 * x + 1.0
    m = 1.5 * x**2 - 2.0 * x + 2.0
    xq = np.linspace(-1.0, 3.0, 101)
    y = np.asarray(
        interp1d(
            jnp.asarray(xq), jnp.asarray(x), jnp.asarray(f), 0.0, 0.0, m=jnp.asarray(m)
        )
    )
    assert np.allclose(y, 0.5 * xq**3 - xq**2 + 2.0 * xq + 1.0, rtol=0.0, atol=4e-15)


def test_a_power_law_is_exact_in_log_log():
    x = np.logspace(-6.0, 2.0, 9)
    xq = np.logspace(-6.0, 2.0, 333)
    y = np.asarray(
        interp1d(
            jnp.asarray(xq),
            jnp.asarray(x),
            jnp.asarray(x**-1.5),
            0.0,
            0.0,
            logx=True,
            logf=True,
        )
    )
    assert np.allclose(y, xq**-1.5, rtol=1e-13, atol=0.0)


def test_outside_the_nodes_the_fill_values_are_returned():
    x, f = _nodes(7)
    y = np.asarray(
        interp1d(
            jnp.asarray([x[0] - 1.0, x[-1] + 1.0]),
            jnp.asarray(x),
            jnp.asarray(f),
            -3.0,
            4.0,
        )
    )
    assert np.array_equal(y, [-3.0, 4.0])
    y = np.asarray(
        interp1d(
            jnp.asarray([x[0] / 2.0]),
            jnp.asarray(x),
            jnp.asarray(np.abs(f) + 1.0),
            0.0,
            1.0,
            logf=True,
        )
    )
    assert np.allclose(y, [1e-300], rtol=1e-12, atol=0.0)


def test_the_nodes_are_returned_exactly():
    x, f = _nodes(13)
    m = pchip_slopes(jnp.asarray(x), jnp.asarray(f))
    y = np.asarray(
        hermite_eval(jnp.asarray(x), jnp.asarray(x), jnp.asarray(f), m, 0.0, 0.0)
    )
    assert np.allclose(y, f, rtol=0.0, atol=1e-15)


def test_the_linear_order_is_numpys_interp():
    x, f = _nodes(3)
    xq = np.linspace(x[0] - 1.0, x[-1] + 1.0, 257)
    y = np.asarray(
        interp1d(
            jnp.asarray(xq), jnp.asarray(x), jnp.asarray(f), -1.0, 2.0, order="linear"
        )
    )
    assert np.allclose(
        y, np.interp(xq, x, f, left=-1.0, right=2.0), rtol=0.0, atol=1e-15
    )


def test_fewer_than_three_nodes_are_refused():
    with pytest.raises(InputError):
        pchip_slopes(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0]))
