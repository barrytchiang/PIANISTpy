"""Tests of `dispersion_profile`, `projected_dispersion` and
`aperture_dispersion` against the closed forms of the isotropic Plummer model,
to a relative :math:`10^{-6}` (TIER_BOUNDS[0])."""

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
from pianistpy.validation import TIER_BOUNDS

# Grids smaller than the grids of the default settings. With these grids, the
# Plummer model's dispersions in this module meet the closed forms to 1e-6.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)


@pytest.fixture(scope="module")
def plummer():
    """Return the untruncated isotropic Plummer model, the model's
    `DispersionProfile` and the model's `ProjectedDispersion`, computed with
    SETTINGS."""
    model = analytic_model("plummer")
    profile = dispersion_profile(model, SETTINGS)
    projected = projected_dispersion(model, profile, SETTINGS)
    return model, profile, projected


def test_dispersion_profile_of_plummer(plummer):
    r"""Assert the radial dispersion profile of the Plummer model.

    The test asserts four properties:

    - :math:`\sigma_{\rm r}^2` equals P / 6 and :math:`\sigma_{\rm t}^2` equals
      2 P / 6 to a relative :math:`10^{-6}` at 15 radii from :math:`10^{-4}` to
      :math:`10^3`, and beta equals 0;
    - the row placement ended with every interval within its tolerance
      (``end == "tolerance"`` and ``worst_last`` not above ``tolerance``);
    - the stored ``sigma_r`` has one component and one value per row;
    - ``edge_limit`` is NaN, because the untruncated model is not spatially
      finite.
    """
    # The internal density of the Plummer model is rho = 3 P^5, with the
    # potential P = (1 + r^2)^(-1/2). The isotropic moment integrals of
    # Chiang+(2026d), Eq. (B42) then give sigma_r^2 = P / 6.
    model, profile, _ = plummer
    r = np.logspace(-4, 3, 15)
    sigma_r, sigma_t, beta = profile.read(r)
    p = (1.0 + r**2) ** -0.5
    np.testing.assert_allclose(sigma_r[0] ** 2, p / 6.0, rtol=TIER_BOUNDS[0])
    np.testing.assert_allclose(sigma_t[0] ** 2, 2.0 * p / 6.0, rtol=TIER_BOUNDS[0])
    np.testing.assert_array_equal(beta, 0.0)
    # The rows of the radial dispersion are added until the midpoint of every
    # interval between rows reads within the record's tolerance.
    certificate = profile.certificate
    assert certificate["end"] == "tolerance"
    assert certificate["worst_last"] <= certificate["tolerance"]
    assert profile.sigma_r.shape == (1, certificate["rows"])
    assert np.isnan(profile.edge_limit[0])


def test_projected_dispersion_of_plummer(plummer):
    r"""Assert the projected table of the Plummer model.

    The test asserts that :math:`\Sigma` and :math:`\sigma_{\rm los}^2` equal the
    closed forms to a relative :math:`10^{-6}` at 13 projected radii from
    :math:`10^{-4}` to 100. The test also asserts that the row placement ended
    as "tolerance" or "stalled", with a finite ``worst_last`` below 100
    CERTIFICATE_TOLERANCE.
    """
    # In internal units, the Plummer model has the surface density
    # Sigma = 4 (1 + R^2)^(-2) and sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2).
    # The internal density is 4 pi times the density of the Plummer model of
    # unit mass and unit scale radius.
    _, _, projected = plummer
    r_2d = np.logspace(-4, 2, 13)
    surface_density, sigma_los = projected.read(r_2d)
    np.testing.assert_allclose(
        surface_density[0], 4.0 / (1.0 + r_2d**2) ** 2, rtol=TIER_BOUNDS[0]
    )
    np.testing.assert_allclose(
        sigma_los[0] ** 2,
        3.0 * np.pi / 64.0 / np.sqrt(1.0 + r_2d**2),
        rtol=TIER_BOUNDS[0],
    )
    # The record `projected.certificate` states how the row placement ended.
    # The closed forms test the values of the projected table. The record's
    # worst_last is only bounded above, by 100 CERTIFICATE_TOLERANCE.
    assert projected.certificate["end"] in ("tolerance", "stalled")
    assert np.isfinite(projected.certificate["worst_last"])
    assert projected.certificate["worst_last"] < 100.0 * CERTIFICATE_TOLERANCE


def test_aperture_dispersion_of_plummer(plummer):
    """Assert that `aperture_dispersion` over centered circles of radius A in
    (0.05, 0.5, 3, 50) returns one value whose square equals the closed form
    to a relative :math:`10^{-6}`."""
    # Over a centered circle of radius A, the Sigma-weighted mean of
    # sigma_los^2 is (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^(-1)].
    _, _, projected = plummer
    for a in (0.05, 0.5, 3.0, 50.0):
        sigma = aperture_dispersion(projected, Aperture.circle(a), SETTINGS)
        exact = (
            np.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        assert sigma.shape == (1,)
        assert abs(sigma[0] ** 2 / exact - 1.0) < TIER_BOUNDS[0]
