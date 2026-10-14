import math

from pianistpy.anisotropy import Anisotropy
from pianistpy.df.central import center_form
from pianistpy.profiles import DoublePowerLaw


def test_hernquist_isotropic_center():
    # Hernquist (gamma = 1), isotropic: the regular branch, rho_aux ~ u^s with
    # s = -gamma / (2 - gamma) = -1, and F ~ A u^kappa, kappa = s - 3/2 = -5/2.
    form = center_form(DoublePowerLaw(1.0, 4.0, 1.0), Anisotropy())
    assert (form["branch"], form["form"]) == ("regular", "power")
    assert form["s"] == -1.0 and form["kappa"] == -2.5
    assert form["amplitude"] > 0.0 and form["sign"] == 1.0
    assert math.isnan(form["log_coef"])
