"""Tests of the radial dispersion and the anisotropy profile against closed forms."""

import numpy as np
import pytest

from pianistpy import Anisotropy
from pianistpy.observables._dispersion import (
    beta_sigma,
    radial_dispersion,
    radial_dispersion_reader,
)

ANISOTROPIES = [
    Anisotropy(),
    Anisotropy(beta_c=0.3),
    Anisotropy(r_a=2.0),
    Anisotropy(r_a=2.0, kind="II"),
    Anisotropy(beta_c=-0.25, r_a=2.0),
    Anisotropy(beta_c=0.25, r_a=2.0, kind="II"),
]


@pytest.mark.parametrize("anisotropy", ANISOTROPIES, ids=str)
def test_beta_sigma_is_the_slope_of_the_factor(anisotropy):
    # beta_sigma = -R^2 d ln g / d R^2 (paper Eq. B44), by a central difference
    r = np.array([0.1, 0.5, 1.0, 1.5])
    x = r**2
    step = 1e-6 * x
    slope = (
        np.log(anisotropy.factor(np.sqrt(x + step)))
        - np.log(anisotropy.factor(np.sqrt(x - step)))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        beta_sigma(r, anisotropy), -x * slope, rtol=1e-6, atol=1e-9
    )


def _plummer():
    """The isotropic Plummer sphere in internal units: P = (1 + r^2)^(-1/2),
    rho = 3 P^5, so that rho sigma_r^2 = INT_0^P 3 K^5 dK and sigma_r^2 = P / 6."""

    def potential(r):
        return (1.0 + np.asarray(r, dtype=np.float64) ** 2) ** -0.5

    def density(r):
        return 3.0 * potential(r) ** 5

    def radius(p):
        p = np.asarray(p, dtype=np.float64)
        return np.sqrt(1.0 / p**2 - 1.0)

    return potential, density, radius


def test_radial_dispersion_of_plummer():
    potential, density, radius = _plummer()
    r = np.logspace(-3, 2, 40)
    p_outer = float(potential(np.array([100.0]))[0])
    boundary_term = 0.5 * p_outer**6
    sigma_r2, fraction = radial_dispersion(
        r, potential, density, radius, Anisotropy(), p_outer, boundary_term, 200
    )
    np.testing.assert_allclose(sigma_r2, potential(r) / 6.0, rtol=1e-12)
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))


def test_reader_returns_the_table_at_its_rows():
    potential, density, radius = _plummer()
    r = np.logspace(-3, 2, 40)
    table = potential(r) / 6.0
    read = radial_dispersion_reader(r, table, potential, False, float("nan"))
    np.testing.assert_allclose(read(r), table, rtol=1e-14)
