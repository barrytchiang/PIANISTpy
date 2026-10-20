"""Tests of the dispersion and projection stages against the Plummer sphere."""

import numpy as np
import pytest

from pianistpy import Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    CERTIFICATE_TOLERANCE,
    aperture_dispersion,
    dispersion_profile,
    projected_dispersion,
)
from pianistpy.validation import GOAL

# Smaller grids than the default tier, enough for the goal on this model.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)


@pytest.fixture(scope="module")
def plummer():
    """The untruncated isotropic Plummer model, its dispersions and projection."""
    model = analytic_model("plummer")
    profile = dispersion_profile(model, SETTINGS)
    projected = projected_dispersion(model, profile, SETTINGS)
    return model, profile, projected


def test_dispersion_profile_of_plummer(plummer):
    # rho = 3 P^5 and isotropy give sigma_r^2 = P / 6 (Chiang+(2026d), Eq. B42)
    model, profile, _ = plummer
    r = np.logspace(-4, 3, 15)
    sigma_r, sigma_t, beta = profile.read(r)
    p = (1.0 + r**2) ** -0.5
    np.testing.assert_allclose(sigma_r[0] ** 2, p / 6.0, rtol=GOAL)
    np.testing.assert_allclose(sigma_t[0] ** 2, 2.0 * p / 6.0, rtol=GOAL)
    np.testing.assert_array_equal(beta, 0.0)
    # the rows are placed until every interval reads within the tolerance
    certificate = profile.certificate
    assert certificate["end"] == "tolerance"
    assert certificate["worst_last"] <= certificate["tolerance"]
    assert profile.sigma_r.shape == (1, certificate["rows"])
    assert np.isnan(profile.edge_limit[0])


def test_projected_dispersion_of_plummer(plummer):
    # Sigma = 4 (1 + R^2)^-2 and sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2) in
    # internal units (the internal density is 4 pi times Plummer's)
    _, _, projected = plummer
    r_2d = np.logspace(-4, 2, 13)
    surface_density, sigma_los = projected.read(r_2d)
    np.testing.assert_allclose(
        surface_density[0], 4.0 / (1.0 + r_2d**2) ** 2, rtol=GOAL
    )
    np.testing.assert_allclose(
        sigma_los[0] ** 2, 3.0 * np.pi / 64.0 / np.sqrt(1.0 + r_2d**2), rtol=GOAL
    )
    # The certificate states how its rows ended; the table is held above by the
    # closed forms, not by the certificate's own tolerance.
    assert projected.certificate["end"] in ("tolerance", "stalled")
    assert np.isfinite(projected.certificate["worst_last"])
    assert projected.certificate["worst_last"] < 100.0 * CERTIFICATE_TOLERANCE


def test_aperture_dispersion_of_plummer(plummer):
    # Over a centered circle of radius A the mean square is
    # (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^-1].
    _, _, projected = plummer
    for a in (0.05, 0.5, 3.0, 50.0):
        sigma = aperture_dispersion(projected, Aperture.circle(a), SETTINGS)
        exact = (
            np.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        assert sigma.shape == (1,)
        assert abs(sigma[0] ** 2 / exact - 1.0) < GOAL
