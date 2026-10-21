r"""Tests of the amplitude of the DF's closed form at the center.

The amplitude is :math:`N' C_s \Gamma(\nu - s) / \Gamma(-s)`, with
:math:`s` the least non-integer power of the auxiliary density in the depth
:math:`u = P_0 - P` below the central potential :math:`P_0`, :math:`C_s` the
coefficient of that power, :math:`\nu = 3/2 - \beta_{\rm c}` and
:math:`N' = 2^{\beta_{\rm c}} / [(2\pi)^{3/2} \Gamma(1 - \beta_{\rm c})]`.
The ratio of the two Gamma functions is formed from the logarithm and the
sign of each Gamma function. For a cusp of slope
:math:`\gamma = 1.99`, both :math:`\Gamma(200.5)` and :math:`\Gamma(199)`
overflow float64, but their ratio is a finite number, and the tests require
that finite amplitude. For a Hernquist cusp the tests require the amplitude
to equal, bit for bit, the quotient of the two Gamma functions.
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
    # For the isotropic Hernquist model the Gamma ratio is Gamma(5/2) / Gamma(1),
    # and the amplitude equals the quotient of the two Gamma functions bit for bit.
    assert _center(1.0, 0.0)["amplitude"].hex() == "0x1.59b8b1f4ecc99p-3"


def test_the_gradient_path_reads_the_steep_center():
    # At gamma = 1.99 and beta_c = -1, Gamma(201.5) and Gamma(199) of the next
    # term both overflow float64. The Jacobian of the center's four numbers with
    # respect to beta_c must be finite and must equal their central difference
    # in beta_c.
    component = Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(beta_c=-1.0))
    _, jac = _center_jac(component, "beta_c")
    h = 1e-6
    up, down = _center(1.99, -1.0 + h), _center(1.99, -1.0 - h)
    keys = ("kappa_sing", "amplitude", "next_kappa", "next_coef")
    difference = [(up[key] - down[key]) / (2.0 * h) for key in keys]
    assert np.all(np.isfinite(jac))
    assert jac == pytest.approx(difference, rel=1e-6)
