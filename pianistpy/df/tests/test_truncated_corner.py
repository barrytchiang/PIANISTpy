r"""Tests of the truncated table's rows below the Taylor form's switch at
:math:`\beta_{\rm c} = \pm 1/2`.

There the table holds the exact shifted closed form ``F_u(K + K_t) - F_u(K_t)``,
whose difference cancels by ``F_u(K_t) / F(K)`` (about :math:`6 \times 10^{8}`
at the floor on a steep cusp), so its rounding set the sign of the Taylor
form's check. Below the switch the rows are the closed form's own Taylor sum
at :math:`\mathcal{K}_{\rm t}`. Measured here on a Hernquist component against
the shifted difference at 50 digits, from the closed form
``Psi = 1 / (1 + R)``: ``rho_aux = R^(2 beta_c) rho(R(Psi))``,
``F_u = d^m rho_aux / dPsi^m / (2 pi^2)``.
"""

import mpmath as mp
import numpy as np
import pytest

from pianistpy import Truncation
from pianistpy.anisotropy import Anisotropy
from pianistpy.df import compute_df, truncate
from pianistpy.df.inputs import Component
from pianistpy.profiles import DoublePowerLaw

K_T = 0.1


def _shifted(k, k_t, beta_c):
    """``F_u(k + k_t) - F_u(k_t)`` at 50 digits, Hernquist (C = 2, Psi(0) = 1)."""
    m = 1 if beta_c > 0 else 2
    with mp.workdps(50):
        b = mp.mpf(beta_c)

        def aux(p):
            r = 1 / p - 1
            return r ** (2 * b) * 2 / (r * (1 + r) ** 3)

        kt = mp.mpf(k_t)
        f_kt = mp.diff(aux, kt, m)
        return [
            float((mp.diff(aux, kt + mp.mpf(float(x)), m) - f_kt) / (2 * mp.pi**2))
            for x in k
        ]


@pytest.fixture(scope="module", params=[0.5, -0.5])
def corner(request):
    c = Component(DoublePowerLaw.hernquist(), Anisotropy(beta_c=request.param))
    return request.param, truncate(compute_df(c), Truncation(k_t=K_T))


def test_the_rows_below_the_switch_are_the_closed_form(corner):
    beta_c, model = corner
    f_1, f_2, k_switch = model.form.below_floor_coefficients[0]
    k = np.asarray(model.table.k[0])
    below = np.flatnonzero(k <= k_switch)
    assert below.size >= 1
    rows = below[np.unique(np.linspace(0, below.size - 1, 12).astype(int))]
    got = 10.0 ** np.asarray(model.table.log_df[0])[rows]
    want = np.array(_shifted(k[rows], K_T, beta_c))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_the_taylor_form_meets_the_bottom_row(corner):
    _, model = corner
    f_1, f_2, _ = model.form.below_floor_coefficients[0]
    k_0 = float(model.table.k[0][0])
    row = 10.0 ** float(model.table.log_df[0][0])
    assert abs((f_1 * k_0 + 0.5 * f_2 * k_0**2) / row - 1.0) <= 1e-9
