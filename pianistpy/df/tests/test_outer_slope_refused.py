"""Tests of the input's refusal of an outer slope between 2 and 3.

The outer end of the inversion and the tail exponent assume Psi ~ 1/R, which a
profile with 2 < rho_beta < 3 does not have; such a component is refused by
name at its input. The profile itself stays available (its mass and potential).
"""

import math

import pytest

from pianistpy import Anisotropy, DoublePowerLaw
from pianistpy.df.inputs import Component
from pianistpy.exceptions import InputError


@pytest.mark.parametrize("beta", [2.01, 2.5, 2.99])
@pytest.mark.parametrize("anisotropy", [Anisotropy(), Anisotropy(beta_c=-0.5)])
def test_an_outer_slope_between_two_and_three_is_refused_by_name(beta, anisotropy):
    with pytest.raises(
        InputError, match="OUTER SLOPE: rho_beta = .* lies between 2 and 3"
    ):
        Component(DoublePowerLaw(1.0, beta, 1.0), anisotropy)


@pytest.mark.parametrize("beta", [3.0, 4.0, 5.0])
def test_an_outer_slope_of_three_or_more_is_taken(beta):
    Component(DoublePowerLaw(1.0, beta, 1.0))


def test_the_profile_alone_stays_available():
    profile = DoublePowerLaw(1.0, 2.5, 1.0)
    assert not math.isfinite(float(profile.mass_total))
