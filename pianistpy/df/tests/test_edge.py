r"""Tests of the DF at the edge of a truncated density, in `pianistpy.df.edge`.

Near the edge, a truncated density goes as C P^q and the DF as A K^p. The tests
check the relation between the exponents q and p, the amplitude A from the
amplitude C, and the lowest central anisotropy :math:`\beta_{\rm c}` for which
a DF exists.
"""

import math

import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pianistpy import InputError, UnphysicalModelError
from pianistpy.df.analytic import hernquist_df
from pianistpy.df.edge import (
    check_edge_existence,
    density_edge_exponent,
    df_edge_amplitude,
    df_edge_exponent,
    lowest_beta_c,
)
from pianistpy.df.inversion import forward_prefactor

BETA_PARENT = [-0.5, 0.0, 0.25]
DF_EDGE = [1.0, 2.0]


@pytest.mark.parametrize("beta_parent", BETA_PARENT)
@pytest.mark.parametrize("df_edge", DF_EDGE)
def test_the_condition_is_one_on_the_difference(beta_parent, df_edge):
    # A parent whose DF goes as K^nu at the edge has a child DF of central
    # anisotropy beta_c exactly when beta_c - beta_parent > -1 - nu, for every
    # parent anisotropy beta_parent. The child DF then goes as
    # K^(nu + beta_c - beta_parent) at the edge.
    q = density_edge_exponent(df_edge, beta_parent)
    assert lowest_beta_c(q) - beta_parent == pytest.approx(-1.0 - df_edge)
    for beta_child in (-2.5, -1.0, 0.0, 0.4):
        assert df_edge_exponent(q, beta_child) == pytest.approx(
            df_edge + beta_child - beta_parent
        )


def test_a_parent_is_its_own_child():
    for beta_parent in BETA_PARENT:
        q = density_edge_exponent(1.0, beta_parent)
        assert df_edge_exponent(q, beta_parent) == pytest.approx(1.0)


def test_existence_is_refused_at_and_below_the_bound():
    q = 2.5
    check_edge_existence(q, -1.999)
    for beta_c in (-2.0, -2.5):
        with pytest.raises(UnphysicalModelError, match="beta_c > 1/2 - 2.5 = -2"):
            check_edge_existence(q, beta_c)
        with pytest.raises(UnphysicalModelError):
            df_edge_amplitude(1.0, q, beta_c)


def test_a_non_integrable_df_edge_is_refused():
    with pytest.raises(InputError):
        density_edge_exponent(-1.0, 0.0)


@pytest.mark.parametrize("q", [2.5, 3.5, 1.2])
@pytest.mark.parametrize("beta_c", [0.4, 0.0, -0.5, -0.9])
def test_amplitude_against_the_density_integral(q, beta_c):
    # The DF F(Q) = A Q^p, with A from df_edge_amplitude and p from
    # df_edge_exponent, must give back the density C P^q with C = 1 through
    # rho_aux(P) = K_f INT_0^P F(Q) (P - Q)^(1/2 - beta_c) dQ, where K_f is
    # the forward prefactor.
    if not beta_c > lowest_beta_c(q):
        pytest.skip("no DF")
    amplitude = df_edge_amplitude(1.0, q, beta_c)
    exponent = df_edge_exponent(q, beta_c)
    forward = 2.0 ** (0.5 - beta_c) * forward_prefactor(beta_c)
    p = 0.37
    integral = quad(
        lambda t: p ** (exponent + 1.5 - beta_c),
        0.0,
        1.0,
        weight="alg",
        wvar=(exponent, 0.5 - beta_c),
        epsabs=0.0,
        epsrel=1e-12,
    )[0]
    assert_allclose(forward * amplitude * integral, p**q, rtol=1e-9)


def test_isotropic_edge_of_the_truncated_hernquist_model():
    # The truncated isotropic DF F_t(K) = F(K + K_t) - F(K_t) goes as
    # F'(K_t) K at the edge. The truncated density therefore goes as C P^(5/2)
    # with C = 4 pi sqrt(2) F'(K_t) B(2, 3/2), where B(2, 3/2) = 4/15, and
    # df_edge_amplitude must return F'(K_t) from C.
    k_t = 0.05
    _, slope = hernquist_df(k_t)
    density_amplitude = 4.0 * math.pi * math.sqrt(2.0) * float(slope) * (4.0 / 15.0)
    assert df_edge_exponent(2.5, 0.0) == 1.0
    assert_allclose(
        df_edge_amplitude(density_amplitude, 2.5, 0.0), float(slope), rtol=1e-13
    )
