"""The quick tier of the observables against the default tier."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from pianistpy.validation import CEILING


@pytest.fixture(scope="module")
def plummer():
    """The isotropic Plummer model at the quick tier, and its table."""
    model = dataclasses.replace(
        analytic_model("plummer"), settings=Settings(quick=True)
    )
    return model, LosTable(model)


def test_the_quick_tier_holds_the_ceiling_against_the_default(plummer):
    # the quick values (the observables' n_radii, the losvd group's rows,
    # floor layers, sightline and velocity nodes and break scan) on the
    # Plummer sphere, whose closed forms the default tier holds to the goal:
    # every reading within the ceiling of them, and the table's report within
    # the ceiling
    model, table = plummer
    assert model.settings.tier == "quick" and model.settings.is_tested_tier
    assert table.validation.level <= 1
    # sigma_r^2 = P / 6 (Chiang+(2026d), Eq. B42)
    r = np.logspace(-3, 2, 11)
    p = (1.0 + r**2) ** -0.5
    np.testing.assert_allclose(table.profile.read(r)[0][0] ** 2, p / 6.0, rtol=CEILING)
    # sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2), from the model and the table
    exact = np.sqrt(3.0 * math.pi / 64.0 / np.sqrt(1.0 + r**2))
    np.testing.assert_allclose(sigma_los(model, r), exact, rtol=CEILING)
    np.testing.assert_allclose(sigma_los(table, r), exact, rtol=CEILING)
    # a centered circle of radius A by both routes: the second route reads the
    # aperture's velocity distribution from the table
    for a in (0.5, 3.0):
        aperture = Aperture.circle(a)
        exact = math.sqrt(
            math.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        assert sigma_los_aperture(model, aperture) == pytest.approx(exact, rel=CEILING)
        assert sigma_los_aperture(table, aperture) == pytest.approx(exact, rel=CEILING)
