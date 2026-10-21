"""Tests of the observables of the isotropic Plummer model with the quick
settings (``Settings(quick=True)``) against closed forms, to a relative
:math:`10^{-4}`."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from pianistpy.validation import TIER_BOUNDS


@pytest.fixture(scope="module")
def plummer():
    """Return the isotropic Plummer model with the quick settings, and the
    `LosTable` of the model."""
    model = dataclasses.replace(
        analytic_model("plummer"), settings=Settings(quick=True)
    )
    return model, LosTable(model)


def test_the_quick_settings_hold_1e_4_against_the_default(plummer):
    r"""Assert that the quick settings reproduce the Plummer closed forms to a
    relative :math:`10^{-4}` (TIER_BOUNDS[1]).

    The test asserts five properties:

    - the model's settings level is "quick" and is a tested settings level;
    - the table's validation report has the tier "silent" or "recorded";
    - the table's :math:`\sigma_{\rm r}^2` equals P / 6 at 11 radii from
      :math:`10^{-3}` to 100;
    - `sigma_los` on the model and on the table equals
      ((3 pi / 64) (1 + R^2)^(-1/2))^(1/2) at the 11 radii;
    - `sigma_los_aperture` on the model and on the table equals the closed form
      over centered circles of radius 0.5 and 3.
    """
    model, table = plummer
    assert model.settings.level == "quick" and model.settings.is_tested_level
    assert table.validation.level <= 1
    # The isotropic Plummer sphere has sigma_r^2 = P / 6, with the potential
    # P = (1 + r^2)^(-1/2) (Chiang+(2026d), Eq. (B42)).
    r = np.logspace(-3, 2, 11)
    p = (1.0 + r**2) ** -0.5
    np.testing.assert_allclose(
        table.profile.read(r)[0][0] ** 2, p / 6.0, rtol=TIER_BOUNDS[1]
    )
    # The Plummer sphere has sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2).
    exact = np.sqrt(3.0 * math.pi / 64.0 / np.sqrt(1.0 + r**2))
    np.testing.assert_allclose(sigma_los(model, r), exact, rtol=TIER_BOUNDS[1])
    np.testing.assert_allclose(sigma_los(table, r), exact, rtol=TIER_BOUNDS[1])
    # Over a centered circle of radius A, the Sigma-weighted mean of
    # sigma_los^2 is (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^(-1)].
    # Given a LosTable, sigma_los_aperture reads the aperture's velocity
    # distribution from the table.
    for a in (0.5, 3.0):
        aperture = Aperture.circle(a)
        exact = math.sqrt(
            math.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        assert sigma_los_aperture(model, aperture) == pytest.approx(
            exact, rel=TIER_BOUNDS[1]
        )
        assert sigma_los_aperture(table, aperture) == pytest.approx(
            exact, rel=TIER_BOUNDS[1]
        )
