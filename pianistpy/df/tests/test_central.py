import math

import numpy as np
import pytest

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
    # The series' d^N/dP^N = (-1)^N sum C_s s (s - 1) ... u^(s - N), N = 0 to 3,
    # against ExactDerivs' symbolic derivatives at the central anchor. The anchor
    # is the grid node compute_df places, u = 1.0015e-9, the same at the default
    # and quick tiers for every case here (measured; Type II and gamma < 1 DPL
    # models carry no central anchor). Measured: at most 2.7e-15 relative (NFW
    # constant beta_c 0.3, order 3).
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
