"""Tests of `beta_sigma` and `radial_dispersion` against closed forms."""

import numpy as np
import pytest

from pianistpy import Anisotropy
from pianistpy.observables._dispersion import (
    beta_sigma,
    radial_dispersion,
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
    """Assert that `beta_sigma` equals -R^2 d ln g / d R^2 from a central
    difference in R^2.

    The factor g(R) is `Anisotropy.factor` (Chiang+(2026d), Eq. (B44)), and
    beta_sigma = -R^2 d ln g / d R^2 is Chiang+(2026d), Eq. (B45). The step in
    R^2 is :math:`10^{-6} R^2`. The test holds to ``rtol=1e-6`` and ``atol=1e-9``
    at R in (0.1, 0.5, 1, 1.5), for each of the six models in ANISOTROPIES.
    """
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
    """Return the potential, the density and the radius at a potential of the
    isotropic Plummer sphere in internal units.

    The potential is P = (1 + r^2)^(-1/2), and the density is rho = 3 P^5. The
    isotropic Jeans integral ``rho sigma_r^2 = INT_0^P 3 K^5 dK`` gives
    ``sigma_r^2 = P / 6``. The radius at the potential P is
    ``(1 / P^2 - 1)^(1/2)``.
    """

    def potential(r):
        return (1.0 + np.asarray(r, dtype=np.float64) ** 2) ** -0.5

    def density(r):
        return 3.0 * potential(r) ** 5

    def radius(p):
        p = np.asarray(p, dtype=np.float64)
        return np.sqrt(1.0 / p**2 - 1.0)

    return potential, density, radius


def test_radial_dispersion_of_plummer():
    r"""Assert that `radial_dispersion` gives :math:`\sigma_{\rm r}^2 = P / 6` for the
    Plummer sphere.

    The integral starts at the potential P_outer at r = 100. The boundary term
    0.5 P_outer^6 is the integral of 3 K^5 dK from zero to P_outer. The test
    asserts :math:`\sigma_{\rm r}^2 = P / 6` to ``rtol=1e-12`` at 40 radii from
    :math:`10^{-3}` to 100, and a boundary fraction between 0 and 1 at each of the
    40 radii.
    """
    potential, density, radius = _plummer()
    r = np.logspace(-3, 2, 40)
    p_outer = float(potential(np.array([100.0]))[0])
    boundary_term = 0.5 * p_outer**6
    sigma_r2, fraction = radial_dispersion(
        r, potential, density, radius, Anisotropy(), p_outer, boundary_term, 200
    )
    np.testing.assert_allclose(sigma_r2, potential(r) / 6.0, rtol=1e-12)
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))
