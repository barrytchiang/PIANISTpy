"""Tests of the double-power-law profile."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import DoublePowerLaw, InputError
from pianistpy.profiles.core import (
    _mass_closed,
    _mass_general,
    _mass_or_p_mpmath,
    _p_closed,
    _p_general,
)

CLOSED_FORM_KEYS = [
    (1.0, 3.0, 1.0),
    (1.0, 3.0, 0.0),
    (1.0, 3.0, 0.5),
    (1.0, 3.0, 1.5),
    (2.0, 3.0, 0.0),
    (2.0, 3.0, 1.0),
    (1.0, 4.0, 1.0),
    (1.0, 4.0, 0.0),
    (1.0, 4.0, 1.5),
    (2.0, 5.0, 0.0),
]
GENERAL_KEYS = [(1.0, 3.5, 1.5), (1.5, 4.5, 0.5), (0.5, 3.5, 1.0)]
R_WIDE = np.logspace(-8, 8, 17)


def test_presets_are_parameter_sets_of_the_one_profile():
    assert DoublePowerLaw.nfw() == DoublePowerLaw(1.0, 3.0, 1.0)
    assert DoublePowerLaw.hernquist() == DoublePowerLaw(1.0, 4.0, 1.0)
    assert DoublePowerLaw.plummer() == DoublePowerLaw(2.0, 5.0, 0.0)


def test_dictionary_round_trip():
    profile = DoublePowerLaw(1.5, 3.5, 0.5)
    assert DoublePowerLaw.from_dict(profile.to_dict()) == profile


@pytest.mark.parametrize(
    ("rho_alpha", "rho_beta", "rho_gamma"),
    [(0.4, 3.0, 1.0), (1.0, 2.0, 1.0), (1.0, 3.0, 2.0), (1.0, 3.0, -0.1)],
)
def test_inadmissible_parameters_are_refused(rho_alpha, rho_beta, rho_gamma):
    with pytest.raises(InputError):
        DoublePowerLaw(rho_alpha, rho_beta, rho_gamma)


def test_hernquist_against_its_textbook_forms():
    # In model units the Hernquist profile has unit total mass and scale radius.
    profile = DoublePowerLaw.hernquist()
    r = np.logspace(-4, 4, 33)
    assert profile.rho_norm == 2.0
    assert profile.mass_total == 1.0
    assert_allclose(profile.rho(r), 2.0 / (r * (1.0 + r) ** 3), rtol=1e-14)
    assert_allclose(profile.mass(r), (r / (1.0 + r)) ** 2, rtol=1e-14)
    assert_allclose(profile.p(r), 1.0 / (1.0 + r), rtol=1e-14)


def test_plummer_against_its_textbook_forms():
    profile = DoublePowerLaw.plummer()
    r = np.logspace(-4, 4, 33)
    assert_allclose(profile.rho_norm, 3.0, rtol=1e-15)
    assert_allclose(profile.rho(r), 3.0 * (1.0 + r * r) ** -2.5, rtol=1e-14)
    assert_allclose(profile.p(r), 1.0 / np.sqrt(1.0 + r * r), rtol=1e-14)


def test_outputs_keep_the_shape_of_the_input():
    profile = DoublePowerLaw.nfw()
    assert np.shape(profile.p(1.0)) == ()
    assert np.shape(profile.mass([1.0, 2.0])) == (2,)
    assert np.shape(profile.rho(np.ones((2, 3)))) == (2, 3)
    assert math.isinf(profile.mass_total)


@pytest.mark.parametrize("key", CLOSED_FORM_KEYS + GENERAL_KEYS)
def test_mass_and_potential_against_arbitrary_precision(key):
    profile = DoublePowerLaw(*key)
    assert_allclose(
        profile.p(R_WIDE), _mass_or_p_mpmath(profile, R_WIDE, "p"), rtol=1e-10
    )
    assert_allclose(
        profile.mass(R_WIDE), _mass_or_p_mpmath(profile, R_WIDE, "mass"), rtol=1e-10
    )


@pytest.mark.parametrize("key", [k for k in CLOSED_FORM_KEYS if k[1] > 3.0])
def test_closed_forms_agree_with_the_general_path(key):
    # One result, one path: a closed form used inside the general functions is
    # held to the incomplete-Beta form.
    profile = DoublePowerLaw(*key)
    r = np.logspace(-6, 6, 25)
    assert_allclose(_p_closed(profile, r), _p_general(profile, r), rtol=1e-10)
    assert_allclose(_mass_closed(profile, r), _mass_general(profile, r), rtol=1e-10)


@pytest.mark.parametrize("key", [(1.0, 3.0, 1.0), (1.0, 4.0, 1.0), (1.5, 4.5, 0.5)])
def test_poisson_equation(key):
    # dP/dR = -M / R**2, by a centerd difference of the potential.
    profile = DoublePowerLaw(*key)
    r = np.logspace(-2, 2, 41)
    step = 1e-5 * r
    slope = (profile.p(r + step) - profile.p(r - step)) / (2.0 * step)
    assert_allclose(-slope * r**2, profile.mass(r), rtol=1e-7)


@pytest.mark.parametrize("rho_beta", [2.2, 2.5])
def test_the_far_outer_branch_holds_for_an_outer_slope_below_three(rho_beta):
    # Beyond r ~ 1e10 the complement branch is used. For 2 < beta < 3 the mass
    # grows as r^(3 - beta) and the potential falls as r^(2 - beta), with
    # coefficients fixed by the density C r^(-beta).
    profile = DoublePowerLaw(1.0, rho_beta, 1.0)
    r = np.array([1e9, 1e10, 1e11, 1e13, 1e16, 1e20])
    mass = profile.rho_norm * r ** (3.0 - rho_beta) / (3.0 - rho_beta)
    p = profile.rho_norm * r ** (2.0 - rho_beta) / ((3.0 - rho_beta) * (rho_beta - 2.0))
    assert np.allclose(np.asarray(profile.mass(r)) / mass, 1.0, rtol=0.0, atol=3e-2)
    assert np.allclose(np.asarray(profile.p(r)) / p, 1.0, rtol=0.0, atol=3e-2)
    ratio = np.asarray(profile.p(r)) / p
    assert np.all(np.abs(np.diff(ratio)) < 2e-2)


@pytest.mark.parametrize("rho_beta", [2.2, 2.5, 2.9, 3.0, 3.5])
def test_the_mass_and_the_potential_are_continuous_where_the_branch_changes(rho_beta):
    profile = DoublePowerLaw(1.0, rho_beta, 1.0)
    r = np.array([0.98e10, 0.99e10, 1.01e10, 1.02e10])
    for values in (np.asarray(profile.mass(r)), np.asarray(profile.p(r))):
        slope = np.diff(np.log(values)) / np.diff(np.log(r))
        # The middle interval spans the change of branch: its slope lies
        # between its neighbors', as on any smooth stretch.
        assert abs(slope[1] - 0.5 * (slope[0] + slope[2])) < 1e-6
