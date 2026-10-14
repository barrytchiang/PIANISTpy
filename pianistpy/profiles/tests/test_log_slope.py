"""The potential's logarithmic slope and its derivative, from the profile."""

import numpy as np

from pianistpy.df.analytic import analytic_model


def test_the_log_slope_is_exact_in_the_rows_quantities():
    profile = analytic_model("hernquist", n_nodes=201).profile
    r = np.geomspace(1e-3, 50.0, 40)
    q = profile.potential_log_slope(r)
    assert q.shape == r.shape
    assert np.all(q < 0.0)
    # Hernquist in internal units: P = 1 / (1 + r), M = r^2 / (1 + r)^2 up to
    # the profile's normalization; the ratio M / (r P) is -d ln P / d ln r = r / (1 + r)
    expected = -r / (1.0 + r)
    assert np.allclose(q, expected, rtol=1e-6, atol=0.0)


def test_the_derivative_of_the_log_slope_matches_central_differences():
    profile = analytic_model("hernquist", n_nodes=201).profile
    r = np.geomspace(1e-2, 20.0, 12)
    step = 1e-5 * r
    central = (
        profile.potential_log_slope(r + step) - profile.potential_log_slope(r - step)
    ) / (2.0 * step)
    exact = profile.potential_log_slope_derivative(r)
    # the exact form reads the density's spline while the central difference reads
    # the potential's Hermite: they agree to the representation's own consistency
    assert np.allclose(exact, central, rtol=1e-4, atol=1e-12)
    # the closed form of the Hernquist case: d/dr [-r / (1 + r)] = -1 / (1 + r)^2
    assert np.allclose(exact, -1.0 / (1.0 + r) ** 2, rtol=1e-5, atol=0.0)
