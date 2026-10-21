r"""Tests of the closed form of the DF and of the auxiliary density at the center.

`center_form` gives the branch, the exponent and the amplitude of the DF's
closed form near the central potential :math:`P_0`; `rho_aux_series` gives the
auxiliary density as a series in the depth :math:`u = P_0 - P`.
"""

import math

import numpy as np
import pytest

from pianistpy.anisotropy import Anisotropy
from pianistpy.df.central import center_form
from pianistpy.profiles import DoublePowerLaw


def test_hernquist_isotropic_center():
    # For the isotropic Hernquist model (gamma = 1) the center is on the regular
    # branch. The auxiliary density goes as u^s with s = -gamma / (2 - gamma) = -1,
    # and the DF goes as A u^kappa with kappa = s - 3/2 = -5/2.
    form = center_form(DoublePowerLaw(1.0, 4.0, 1.0), Anisotropy())
    assert (form["branch"], form["form"]) == ("regular", "power")
    assert form["s"] == -1.0 and form["kappa"] == -2.5
    assert form["amplitude"] > 0.0 and form["sign"] == 1.0
    assert math.isnan(form["log_coef"])


_ANCHOR_U = 1.0 - 0.9999999989985393


@pytest.mark.parametrize(
    "profile",
    [DoublePowerLaw.hernquist(), DoublePowerLaw.nfw()],
    ids=["hernquist", "nfw"],
)
@pytest.mark.parametrize(
    "anisotropy",
    [
        Anisotropy(),
        Anisotropy(beta_c=0.3),
        Anisotropy(beta_c=-0.25),
        Anisotropy(r_a=10.0),
        Anisotropy(beta_c=-0.25, r_a=10.0),
        Anisotropy(beta_c=0.25, r_a=10.0),
    ],
    ids=lambda a: a.family,
)
def test_rho_aux_series_matches_the_exact_derivatives_at_the_anchor(
    profile, anisotropy
):
    # The N-th P-derivative of the series, (-1)^N sum_s C_s s (s - 1) ...
    # (s - N + 1) u^(s - N) for N = 0 to 3, must match the N-th derivative that
    # ExactDerivs computes symbolically, to a relative difference below 1e-13.
    # The depth _ANCHOR_U = 1.0015e-9 below P_0 is the depth of the central
    # anchor node on the grid that compute_df builds.
    from pianistpy.df.central import _MODEL_TYPES, rho_aux_series
    from pianistpy.df.derivatives import ExactDerivs

    a, prof = anisotropy, profile
    bc = 0.0 if a.model == "om" else float(a.beta_c)
    ra = float(a.r_a) if np.isfinite(a.r_a) else 1.0
    ser = rho_aux_series(
        prof.rho_alpha,
        prof.rho_beta,
        prof.rho_gamma,
        bc,
        _MODEL_TYPES[a.family],
        ra,
        float(prof.rho_norm),
    )
    ex = ExactDerivs(prof, beta_c=bc, model=a.family, r_a=ra, max_order=3)
    u = _ANCHOR_U
    r = float(np.asarray(ex.r_of_p(np.array([1.0 - u])))[0])
    for n in range(4):
        s_val = (-1) ** n * sum(
            c * np.prod([s - i for i in range(n)]) * u ** (s - n) for s, c in ser
        )
        e_val = float(ex._eval(n, np.array([r]))[0])
        assert abs(s_val / e_val - 1) < 1e-13, n
