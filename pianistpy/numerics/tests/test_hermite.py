"""Tests of the cubic Hermite interpolant and its inverse."""

import numpy as np

from pianistpy.numerics import hermite, hermite_inverse


def _cubic(x):
    return 0.5 * x**3 - x**2 + 2.0 * x + 1.0, 1.5 * x**2 - 2.0 * x + 2.0


def test_a_cubic_is_reproduced_with_its_derivative():
    x_nodes = np.array([-1.0, 0.3, 1.1, 2.5])
    f_nodes, df_dx_nodes = _cubic(x_nodes)
    x = np.linspace(-1.0, 2.5, 101)
    f, df_dx = hermite(x, x_nodes, f_nodes, df_dx_nodes)
    f_exact, df_dx_exact = _cubic(x)
    assert np.allclose(f, f_exact, rtol=0.0, atol=4e-15)
    assert np.allclose(df_dx, df_dx_exact, rtol=0.0, atol=2e-14)


def test_the_interpolant_takes_the_values_and_slopes_at_the_nodes():
    rng = np.random.default_rng(3)
    x_nodes = np.cumsum(rng.uniform(0.1, 1.0, 12))
    f_nodes = rng.standard_normal(12)
    df_dx_nodes = rng.standard_normal(12)
    f, df_dx = hermite(x_nodes, x_nodes, f_nodes, df_dx_nodes)
    assert np.array_equal(f, f_nodes)
    assert np.array_equal(df_dx, df_dx_nodes)


def test_outside_the_nodes_the_interpolant_is_the_end_line():
    x_nodes = np.array([0.0, 1.0, 2.0])
    f_nodes = np.array([1.0, 2.0, 5.0])
    df_dx_nodes = np.array([0.5, 2.0, 4.0])
    f, df_dx = hermite(np.array([-2.0, 3.5]), x_nodes, f_nodes, df_dx_nodes)
    assert np.array_equal(f, [1.0 - 2.0 * 0.5, 5.0 + 1.5 * 4.0])
    assert np.array_equal(df_dx, [0.5, 4.0])


def test_the_inverse_returns_the_points_of_an_increasing_interpolant():
    x_nodes = np.linspace(-4.0, 4.0, 33)
    f_nodes = np.sinh(x_nodes)
    df_dx_nodes = np.cosh(x_nodes)
    x = np.linspace(-5.0, 5.0, 257)
    f, _ = hermite(x, x_nodes, f_nodes, df_dx_nodes)
    x_back, residual = hermite_inverse(f, x_nodes, f_nodes, df_dx_nodes)
    assert residual <= 4e-14
    assert np.allclose(x_back, x, rtol=0.0, atol=1e-13)


def test_the_inverse_stays_in_its_bracket_on_a_steep_interval():
    # An interval whose end slopes are far steeper than its secant: a plain
    # Newton step from its middle leaves the interval.
    x_nodes = np.array([0.0, 1.0, 2.0])
    f_nodes = np.array([0.0, 1.0, 2.0])
    df_dx_nodes = np.array([1.0, 1e-9, 1.0])
    f = np.array([0.999999, 1.000001])
    x, residual = hermite_inverse(f, x_nodes, f_nodes, df_dx_nodes)
    assert residual <= 1e-12
    assert 0.0 < x[0] < 1.0 < x[1] < 2.0
