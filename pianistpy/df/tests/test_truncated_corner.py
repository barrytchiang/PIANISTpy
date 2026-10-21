r"""Tests of the truncated table at :math:`\beta_{\rm c} = \pm 1/2` below the
switch energy of the below-floor Taylor form.

At :math:`\beta_{\rm c} = \pm 1/2` the truncated table holds the exact shifted
closed form ``F_u(K + K_t) - F_u(K_t)``, with ``F_u`` the untruncated DF. In
float64 that difference loses digits in proportion to the ratio
``F_u(K_t) / F(K)``, which is large at small K. Below the Taylor form's switch
energy ``k_switch``, the table therefore holds the Taylor sum of ``F_u`` at
:math:`\mathcal{K}_{\rm t}` to the cubic term, whose terms are all positive.

The tests compare the table of a truncated Hernquist component with the shifted
difference evaluated at 50 digits. The Hernquist potential is
``Psi = 1 / (1 + R)``, the augmented density is
``rho_aux = R^(2 beta_c) rho(R(Psi))``, and the DF is
``F_u = d^m rho_aux / dPsi^m / (2 pi^2)``, with m = 1 at
:math:`\beta_{\rm c} = 1/2` and m = 2 at :math:`\beta_{\rm c} = -1/2`.
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
    """Return ``F_u(k + k_t) - F_u(k_t)`` of the Hernquist DF at 50 digits.

    The Hernquist profile has C = 2 and Psi(0) = 1. Each returned value is
    rounded to float64 only after the difference is formed at 50 digits.
    """
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
