r"""Tests that `Component` raises InputError for a profile whose outer slope
lies strictly between 2 and 3.

The outer end of the DF inversion and the exponent of the DF's low-energy
tail assume a relative potential :math:`\Psi \propto 1/R` at large radius
:math:`R`. A profile with an outer density slope :math:`\beta` strictly between
2 and 3 has an infinite total mass, and its relative potential does not fall
as :math:`1/R`. `Component` therefore raises InputError for such a profile.
The profile itself is still constructed, with its mass and potential.
"""

import math

import pytest

from pianistpy import Anisotropy, DoublePowerLaw
from pianistpy.df.inputs import Component
from pianistpy.exceptions import InputError


@pytest.mark.parametrize("beta", [2.01, 2.5, 2.99])
@pytest.mark.parametrize("anisotropy", [Anisotropy(), Anisotropy(beta_c=-0.5)])
def test_an_outer_slope_between_two_and_three_is_refused_with_cause(beta, anisotropy):
    """For an outer slope strictly between 2 and 3, `Component` raises
    InputError whose message states the outer slope, for both an isotropic and
    an anisotropic model."""
    with pytest.raises(
        InputError, match="OUTER SLOPE: rho_beta = .* lies between 2 and 3"
    ):
        Component(DoublePowerLaw(1.0, beta, 1.0), anisotropy)


@pytest.mark.parametrize("beta", [3.0, 4.0, 5.0])
def test_an_outer_slope_of_three_or_more_is_taken(beta):
    """For an outer slope of 3 or more, `Component` raises no error."""
    Component(DoublePowerLaw(1.0, beta, 1.0))


def test_the_profile_alone_stays_available():
    """A double power law with outer slope 2.5 is constructed, and its total
    mass is infinite."""
    profile = DoublePowerLaw(1.0, 2.5, 1.0)
    assert not math.isfinite(float(profile.mass_total))
