"""Tests of `truncation_gates` on small synthetic tables of an untruncated DF.

Every table spans the energies Q from :math:`10^{-6}` to just below 1 in
internal units, with 61 nodes spaced evenly in log Q.
"""

import numpy as np

from pianistpy.df.gates import truncation_gates

Q = np.logspace(-6.0, -1e-3, 61)


def _gates(f, k_t, a=-0.5):
    """Return the gates of truncation energy `k_t` for the DF table `f` on Q.

    The same table serves as the sign table and as the rounding-test table,
    and `a` is the analytic exponent of the DF's tail.
    """
    return truncation_gates(Q, f, Q, f, k_t, a)


def test_a_positive_increasing_table_admits_any_truncation():
    g = _gates(Q**1.5, 0.01)
    assert g.k_0 == 0.0 and g.q_min == 0.0 and g.k_t_min == 0.0
    assert g.set_by == "none" and g.fired == ()


def test_a_negative_zone_sets_the_bound_by_its_last_zero_crossing():
    f = Q**1.5 - 1e-3
    k_0 = float(Q[np.flatnonzero(f < 0)[-1]])
    g = _gates(f, 0.5 * k_0)
    assert g.k_0 == k_0 and g.set_by == "K_0" and g.k_t_min == k_0
    assert g.fired == ("0 < K_t < K_0", "Z GATE")
    assert _gates(f, 2.0 * k_0).fired == ()


def test_a_last_minimum_above_the_zero_crossing_sets_the_bound():
    f = Q**1.5 + np.exp(-((np.log(Q) + 4.0) ** 2))
    g = _gates(f, 1e-6)
    assert g.k_0 == 0.0 and g.set_by == "Q_min" and g.q_min > 0.0
    assert g.fired == ("Z GATE",)


def test_untruncated_negatives_are_rounding_only_below_1e_4():
    f = Q**1.5
    f[:3] = -1e-30
    assert "Z_t = 0 GATE" not in _gates(f, 0.0).fired
    assert "Z_t = 0 GATE" in _gates(f, 0.0, a=0.5).fired
    f[:3] = -1e-2
    assert "Z_t = 0 GATE" in _gates(f, 0.0).fired


def test_the_domain_edge_bounds_the_truncation_and_the_gates_read_the_support():
    # The DF table is negative below the domain edge K = 0.3 and positive
    # above the domain edge. `truncation_gates` reads the table only above the
    # domain edge, so the DF has no last zero crossing and K_0 is 0.
    q = np.linspace(0.01, 0.99, 99)
    f = np.where(q < 0.3, -1.0, q)
    g = truncation_gates(q, f, q, f, 0.0, -1.0, k_edge=0.3)
    assert g.k_0 == 0.0 and g.set_by == "domain_edge"
    # The DOMAIN gate requires K_t strictly above the domain edge, so k_t_min
    # is the next double above 0.3, the smallest K_t that the DOMAIN gate
    # admits.
    assert g.k_t_min == np.nextafter(0.3, np.inf)
    assert truncation_gates(q, f, q, f, g.k_t_min, -1.0, k_edge=0.3).fired == ()
    assert g.fired == ("DOMAIN",)
    assert truncation_gates(q, f, q, f, 0.31, -1.0, k_edge=0.3).fired == ()
    assert "DOMAIN" in truncation_gates(q, f, q, f, 0.3, -1.0, k_edge=0.3).fired
