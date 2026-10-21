"""Tests of the inner series of `pianistpy.profiles.series` and of its join."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import DoublePowerLaw
from pianistpy.profiles.series import (
    inner_density,
    inner_depth,
    inner_mass,
    inner_potential,
    series_join,
)

KEYS = [
    (1.0, 3.0, 1.0),
    (1.0, 4.0, 1.0),
    (2.0, 5.0, 0.0),
    (1.0, 3.0, 0.0),
    (1.5, 4.5, 0.5),
    (0.5, 3.5, 1.0),
]
R_INNER = np.logspace(-12, -1, 23)


@pytest.mark.parametrize("key", KEYS)
def test_series_converge_to_the_closed_forms_at_small_radii(key):
    profile = DoublePowerLaw(*key)
    r = np.logspace(-6, -3, 7) ** (1.0 / profile.rho_alpha)
    assert_allclose(inner_density(profile, r), profile.rho(r), rtol=1e-13)
    assert_allclose(inner_mass(profile, r), profile.mass(r), rtol=1e-9)
    # The reference depth 1 - P is a difference of nearly equal numbers, so
    # the reference carries an absolute rounding error of a few units in the
    # last place of 1, covered by atol = 1e-15.
    assert_allclose(inner_depth(profile, r), 1.0 - profile.p(r), rtol=1e-6, atol=1e-15)
    assert_allclose(inner_potential(profile, r, 1.0), profile.p(r), rtol=1e-9)


@pytest.mark.parametrize("key", KEYS)
def test_join_holds_its_bound(key):
    profile = DoublePowerLaw(*key)
    depth, r_join, bound = series_join(profile)
    assert bound <= 1e-7
    assert_allclose(depth, inner_depth(profile, r_join), rtol=1e-14)
    # At the join radius the exact depth 1 - P must agree with the series'
    # depth to ten times the returned bound, plus the rounding of the
    # difference 1 - P.
    exact = 1.0 - float(profile.p(r_join))
    assert abs(depth / exact - 1.0) <= 10.0 * bound + 1e-12 / exact
