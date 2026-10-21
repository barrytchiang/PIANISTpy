"""Tests of the Method II edge law in `pianistpy.df.edge_law`.

The edge expansion is the expansion of a child's augmented density at the
parent's edge, built from the parent DF's Taylor coefficients F_1 and F_2. The
edge law is the child DF's power law that the edge expansion inverts to. The
tests compare the edge expansion, the edge law and the floor terms with
closed-form identities, so no stored product is needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import beta as beta_function

from pianistpy import Anisotropy, InputError
from pianistpy.df import edge_law
from pianistpy.df.inversion import forward_prefactor, inversion_prefactor, order_pair
from pianistpy.df.rows import method2_floor_tail

F_1, F_2, K_SWITCH, R_T, M_T = 3.0e-3, 7.0e-2, 2.0e-8, 31.9, 1.0


def _children():
    return [
        Anisotropy(),
        Anisotropy(beta_c=0.3),
        Anisotropy(beta_c=-0.5),
        Anisotropy(r_a=10.0),
        Anisotropy(r_a=100.0, kind="II"),
        Anisotropy(beta_c=0.25, r_a=10.0),
        Anisotropy(beta_c=-0.25, r_a=100.0, kind="II"),
    ]


def _expansion(anisotropy):
    return edge_law.edge_expansion(F_1, F_2, K_SWITCH, R_T, M_T, anisotropy)


def test_the_isotropic_expansion_is_the_parent_density_and_inverts_to_its_taylor_form():
    ex = _expansion(Anisotropy())
    a = 4.0 * np.pi * np.sqrt(2.0)
    assert ex.coefficients[2.5] == pytest.approx(a * 4.0 / 15.0 * F_1, rel=1e-15)
    assert ex.coefficients[3.5] == pytest.approx(a * 8.0 / 105.0 * F_2, rel=1e-15)
    assert ex.h_t == 1.0 and ex.h_1 == 0.0
    p, c0, c1 = edge_law.edge_law(ex)
    assert p == 1.0
    assert c0 == pytest.approx(F_1, rel=1e-14)
    assert c1 == pytest.approx(F_2 / 2.0, rel=1e-14)


@pytest.mark.parametrize("anisotropy", _children())
def test_the_inverse_factor_and_its_slope_match_the_package_factor(anisotropy):
    ex = _expansion(anisotropy)
    h = 1.0 / float(anisotropy.factor(np.array([R_T]))[0])
    assert ex.h_t == pytest.approx(h, rel=1e-14)
    step = 1e-6 * R_T
    hh = 1.0 / anisotropy.factor(np.array([R_T - step, R_T + step]))
    dh_dr = float((hh[1] - hh[0]) / (2.0 * step))
    r_prime = -(R_T**2) / M_T
    assert ex.h_1 == pytest.approx(dh_dr * r_prime, rel=1e-7, abs=1e-14)


@pytest.mark.parametrize("anisotropy", _children())
def test_the_edge_law_inverts_the_expansion_exactly(anisotropy):
    # For each power q of the edge expansion, the density integral of the
    # edge-law term C_q K^(q - 3/2 + bc), with the package's forward
    # prefactor, must give back the expansion's term a_q P^q. Here bc is the
    # child's central anisotropy beta_c.
    ex = _expansion(anisotropy)
    p, c0, c1 = edge_law.edge_law(ex)
    bc = ex.beta_c
    assert p == pytest.approx(1.0 + bc)
    nu = 0.5 - bc
    for q, c in ((2.5, c0), (3.5, c1)):
        alpha = q - 1.5 + bc
        forward = (
            forward_prefactor(bc) * c * 2.0**nu * beta_function(alpha + 1.0, nu + 1.0)
        )
        assert forward == pytest.approx(ex.coefficients[q], rel=1e-13)


@pytest.mark.parametrize("anisotropy", _children())
def test_the_floor_tail_feeds_the_kernel_with_the_floor_pair(anisotropy):
    ex = _expansion(anisotropy)
    tail = edge_law.floor_tail(ex)
    pair = order_pair(ex.beta_c)
    assert (tail["m_f"], tail["n_f"]) == (pair.order_m, pair.order_n)
    assert tail["pref"] == pytest.approx(inversion_prefactor(pair.order_n, ex.beta_c))
    assert tail["a_q"] == ex.coefficients
    q_min = 1e-9
    q = np.array([q_min, 1e-7, 1e-5])
    corr, _ = method2_floor_tail(q, tail)
    assert np.all(np.isfinite(corr))
    p, c0, c1 = edge_law.edge_law(ex)
    law = q**p * (c0 + c1 * q)
    # At the largest test energy, 1e-5, the floor correction is below 1e-3 of
    # the edge law.
    assert abs(corr[-1]) < 1e-3 * abs(law[-1]) + 1e-300


def test_floor_value_is_the_derivative_of_the_expansion():
    ex = _expansion(Anisotropy(beta_c=0.25, r_a=10.0))
    p = 1e-9
    step = 1e-3 * p
    numerical = (ex.density(p + step) - ex.density(p - step)) / (2.0 * step)
    assert edge_law.floor_value(ex, 1, p) == pytest.approx(float(numerical), rel=1e-6)
    assert edge_law.floor_value(ex, 0, p) == pytest.approx(
        float(ex.density(p)), rel=1e-15
    )
    with pytest.raises(InputError):
        edge_law.floor_value(ex, 1, 10.0 * K_SWITCH)


def test_a_parent_without_the_taylor_form_is_refused_and_one_with_it_is_read():
    form = SimpleNamespace(below_floor=("tail",), below_floor_coefficients=((),))
    parent = SimpleNamespace(form=form, profile=SimpleNamespace(r_t=R_T, mass_t=M_T))
    with pytest.raises(InputError, match="Taylor"):
        edge_law.edge_expansion_of_parent(parent, Anisotropy(r_a=10.0))
    form = SimpleNamespace(
        below_floor=("taylor",), below_floor_coefficients=((F_1, F_2, K_SWITCH),)
    )
    parent = SimpleNamespace(form=form, profile=SimpleNamespace(r_t=R_T, mass_t=M_T))
    ex = edge_law.edge_expansion_of_parent(parent, Anisotropy(r_a=10.0))
    assert ex == _expansion(Anisotropy(r_a=10.0))


def test_a_type_ii_child_whose_edge_lies_beyond_its_anisotropy_radius_is_refused():
    with pytest.raises(InputError, match="R_a"):
        _expansion(Anisotropy(r_a=10.0, kind="II"))
    with pytest.raises(InputError):
        edge_law.edge_expansion(F_1, F_2, K_SWITCH, -1.0, M_T, Anisotropy())


@pytest.mark.parametrize("anisotropy", _children())
def test_the_expansion_at_a_node_on_the_edge_is_the_edge_expansion(anisotropy):
    # edge_expansion_at_node takes the inverse factor at the edge as
    # h_t = h(P_node) - H_1 P_node, with H_1 the slope of h in P. At
    # P_node = 0, h_t is h at the edge, so the expansion at the node equals the
    # edge expansion.
    ex = _expansion(anisotropy)
    node = edge_law.edge_expansion_at_node(
        F_1, F_2, K_SWITCH, 0.0, ex.h_t, ex.h_1, anisotropy
    )
    assert node.coefficients == ex.coefficients
    assert (node.beta_c, node.k_switch, node.h_t, node.h_1) == (
        ex.beta_c,
        ex.k_switch,
        ex.h_t,
        ex.h_1,
    )
    # At P_node = 1e-10 inside the edge, with h(P_node) = h_t + H_1 P_node, the
    # expansion's h_t equals the edge expansion's h_t.
    p_node = 1e-10
    moved = edge_law.edge_expansion_at_node(
        F_1, F_2, K_SWITCH, p_node, ex.h_t + ex.h_1 * p_node, ex.h_1, anisotropy
    )
    assert moved.h_t == pytest.approx(ex.h_t, rel=1e-15, abs=1e-300)
    assert moved.h_1 == ex.h_1


@pytest.mark.parametrize("anisotropy", _children())
def test_an_om_parent_with_h_one_is_the_isotropic_parent(anisotropy):
    # For an OM Type I parent, the expansion divides the child's inverse factor
    # by the parent's inverse factor h_p = 1 + R^2 / r_a^2. With h_p = 1 and a
    # slope of h_p equal to zero, the expansion equals the expansion for an
    # isotropic parent at the same node.
    p_node, h_node, h_slope = 3e-10, 1.7, -2.5e3
    iso = edge_law.edge_expansion_at_node(
        F_1, F_2, K_SWITCH, p_node, h_node, h_slope, anisotropy
    )
    om = edge_law.edge_expansion_at_node(
        F_1, F_2, K_SWITCH, p_node, h_node, h_slope, anisotropy, parent_h=(1.0, 0.0)
    )
    assert om.coefficients == iso.coefficients
    assert (om.h_t, om.h_1) == (iso.h_t, iso.h_1)
    # With h_p = 2 and a slope of h_p equal to zero, the expansion equals the
    # expansion for an isotropic parent with the child's inverse factor
    # divided by 2.
    om2 = edge_law.edge_expansion_at_node(
        F_1, F_2, K_SWITCH, 0.0, h_node, 0.0, anisotropy, parent_h=(2.0, 0.0)
    )
    assert om2.h_t == pytest.approx(h_node / 2.0, rel=1e-15)
    assert om2.coefficients[2.5] == pytest.approx(
        edge_law.edge_expansion_at_node(
            F_1, F_2, K_SWITCH, 0.0, h_node / 2.0, 0.0, anisotropy
        ).coefficients[2.5],
        rel=1e-15,
    )
