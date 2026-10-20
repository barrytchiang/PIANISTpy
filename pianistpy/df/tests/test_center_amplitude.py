"""Tests of the center's amplitude, N' C_s Gamma(nu - s) / Gamma(-s).

The Gamma ratio is formed from the log of each Gamma with its sign, so that a
steep cusp, whose two Gammas both overflow (gamma 1.99: Gamma(200.5) and
Gamma(199)), has the ordinary amplitude it holds, and an ordinary member's
amplitude is the one the quotient gave.
"""

import math

import numpy as np
import pytest

from pianistpy import Anisotropy, DoublePowerLaw
from pianistpy.df.central import center_form
from pianistpy.df.gradient import _center_jac
from pianistpy.df.inputs import Component


def _center(gamma, beta_c):
    component = Component(DoublePowerLaw(1.0, 4.0, gamma), Anisotropy(beta_c=beta_c))
    return center_form(component.unit_profile, component.anisotropy)


@pytest.mark.parametrize(
    ("beta_c", "kappa", "log10_amplitude"),
    [(0.0, -200.5, 0.26), (-0.25, -250.75, 0.97)],
)
def test_a_steep_cusp_has_its_finite_amplitude(beta_c, kappa, log10_amplitude):
    center = _center(1.99, beta_c)
    assert center["branch"] == "regular" and center["form"] == "power"
    assert center["kappa_sing"] == pytest.approx(kappa, abs=1e-9)
    assert math.isfinite(center["amplitude"]) and center["amplitude"] > 0.0
    assert round(math.log10(center["amplitude"]), 2) == log10_amplitude
    assert math.isfinite(center["next_coef"])


def test_an_ordinary_amplitude_is_unchanged():
    # Hernquist, isotropic: Gamma(5/2) / Gamma(1); the value the quotient gave
    assert _center(1.0, 0.0)["amplitude"].hex() == "0x1.59b8b1f4ecc99p-3"


def test_the_gradient_path_reads_the_steep_center():
    # gamma 1.99, beta_c -1: the next term's Gamma(201.5) / Gamma(199) overflow
    # both; the derivatives in beta_c equal the central difference of the center
    component = Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(beta_c=-1.0))
    _, jac = _center_jac(component, "beta_c")
    h = 1e-6
    up, down = _center(1.99, -1.0 + h), _center(1.99, -1.0 - h)
    keys = ("kappa_sing", "amplitude", "next_kappa", "next_coef")
    difference = [(up[key] - down[key]) / (2.0 * h) for key in keys]
    assert np.all(np.isfinite(jac))
    assert jac == pytest.approx(difference, rel=1e-6)
