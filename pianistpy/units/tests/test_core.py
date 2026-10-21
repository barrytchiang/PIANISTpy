"""Tests of the conversion factors out of internal units (`unit_factors`)."""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pianistpy import DoublePowerLaw, InputError
from pianistpy.units import unit_factors

KEYS = [(1.0, 3.0, 1.0), (1.0, 4.0, 1.0), (2.0, 5.0, 0.0), (1.5, 4.5, 0.5)]
CONCENTRATION = 10.0
MASS_0_MSUN = 1.0e12
R_0_PC = 2.0e4


@pytest.mark.parametrize("key", KEYS)
def test_model_factors_against_a_direct_quadrature(key):
    # The number q is, by definition, the integral of r**2 times the density
    # in units of rho_0 from r = 0 to the concentration radius.
    alpha, beta, gamma = key
    mass_q, _ = quad(
        lambda r: r**2 * r ** (-gamma) * (1.0 + r**alpha) ** ((gamma - beta) / alpha),
        0.0,
        CONCENTRATION,
        limit=200,
    )
    profile = DoublePowerLaw(*key)
    factors = unit_factors(profile, CONCENTRATION)
    product = profile.rho_norm * mass_q
    assert_allclose(factors.mass_q, mass_q, rtol=1e-10)
    assert_allclose(factors.model_v, product**-0.5, rtol=1e-10)
    assert_allclose(factors.model_mass, 1.0 / product, rtol=1e-10)
    assert_allclose(factors.model_rho, 1.0 / (4.0 * math.pi * product), rtol=1e-10)
    assert factors.physical_v_kms is None


def test_physical_factors_against_their_definitions():
    profile = DoublePowerLaw.nfw()
    factors = unit_factors(profile, CONCENTRATION, MASS_0_MSUN, R_0_PC)
    mass_q = math.log(1.0 + CONCENTRATION) - CONCENTRATION / (1.0 + CONCENTRATION)
    rho_0 = MASS_0_MSUN * u.Msun / (4.0 * math.pi * (R_0_PC * u.pc) ** 3 * mass_q)
    rho_w = rho_0 / profile.rho_norm
    velocity = np.sqrt(4.0 * math.pi * G * rho_w * (R_0_PC * u.pc) ** 2).to(u.km / u.s)
    mass = (4.0 * math.pi * rho_w * (R_0_PC * u.pc) ** 3).to(u.Msun)
    assert_allclose(factors.physical_v_kms, velocity.value, rtol=1e-12)
    assert_allclose(factors.physical_mass_msun, mass.value, rtol=1e-12)
    assert_allclose(factors.physical_p_kms2, velocity.value**2, rtol=1e-12)
    # The model unit of velocity is sqrt(G M_0 / r_0), so the physical factor
    # divided by the model factor equals sqrt(G M_0 / r_0) in km/s.
    velocity_0 = np.sqrt(G * MASS_0_MSUN * u.Msun / (R_0_PC * u.pc)).to(u.km / u.s)
    assert_allclose(
        factors.physical_v_kms / factors.model_v, velocity_0.value, rtol=1e-12
    )


def test_inadmissible_inputs_are_refused():
    profile = DoublePowerLaw.nfw()
    with pytest.raises(InputError):
        unit_factors(profile, 0.0)
    with pytest.raises(InputError):
        unit_factors(profile, CONCENTRATION, mass_0_msun=MASS_0_MSUN)
