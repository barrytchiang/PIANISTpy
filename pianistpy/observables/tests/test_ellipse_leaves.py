"""Tests of the ellipse route's leaves: the layer counts, the angle integral,
the DF's cumulative table and the layout codes."""

import numpy as np
import pytest

from pianistpy.observables._angle import lam_half, lam_mhalf, lam_rule, power_of
from pianistpy.observables._ellipse_df import EllipseDF
from pianistpy.observables._ellipse_rules import (
    EllipseModel,
    EllipseRules,
    layer_counts,
)
from pianistpy.observables._layout import LAYOUT_SEQ, Lattice

RULES = EllipseRules()


@pytest.mark.parametrize(
    ("beta_c", "k_end"),
    [(-0.5, 1), (-0.25, 2), (0.25, 4), (0.5, 10)],
)
def test_layers_toward_the_l0_point(beta_c, k_end):
    # the counts measured when the rule was set (tolerance 1e-2 of the goal)
    assert layer_counts(beta_c, RULES)[0] == k_end


def test_layers_stop_at_the_rules_floor_above_one_half():
    # a negative power is self-similar under the layers: 19 layers on a floor
    # of 2.86e-8 at beta_c = 3/4
    k, floor, _, _, _ = layer_counts(0.75, RULES)
    assert k == 19
    assert floor == pytest.approx(2.86e-8, rel=1e-2)


def test_layers_of_the_map():
    assert layer_counts(0.25, RULES)[4] == 7
    assert layer_counts(0.5, RULES)[4] == 0
    assert layer_counts(-0.25, RULES)[4] == 0


def _angle_inputs(rng, n):
    c1 = rng.uniform(0.1, 2.0, n)
    c2 = rng.uniform(0.1, 2.0, n)
    c3 = c2 * c2 + rng.uniform(0.05, 2.0, n)
    kap = c1 * c2 + c3 - c2 * c2
    d2 = (c1 - c2) ** 2
    e2 = c2 * c2 - c3
    lam = np.clip(np.sqrt(d2 / kap), 1e-15, np.pi)
    kpi = c3 - c1 * c2 - c2 * c2
    lpi = np.clip(
        np.where(kpi > 0.0, np.sqrt((c1 + c2) ** 2 / np.maximum(kpi, 1e-300)), np.pi),
        1e-15,
        np.pi,
    )
    return c1, c2, c3, kap, d2, e2, lam, lpi


@pytest.mark.parametrize("beta_c", [0.5, -0.5])
def test_closed_forms_against_the_rule(beta_c):
    rng = np.random.default_rng(1)
    c1, c2, c3, kap, d2, e2, lam, lpi = _angle_inputs(rng, 200)
    rule = lam_rule(power_of(beta_c), c1, c2, c3, kap, d2, e2, lam, lpi, 64)
    e = c3 - c2 * c2
    dm = c1 * c1 - c2 * c2
    closed = lam_half(c1, c3, e, dm) if beta_c > 0 else lam_mhalf(c1, c2, e, dm)
    np.testing.assert_allclose(closed, rule, rtol=1e-12)


def test_rule_at_isotropy_is_pi():
    rng = np.random.default_rng(2)
    args = _angle_inputs(rng, 50)
    np.testing.assert_allclose(lam_rule(power_of(0.0), *args, 16), np.pi, rtol=1e-14)


def test_cumulative_of_a_flat_df():
    # F = 1 above and below the floor: G(Q) = Q on every branch. The upper
    # branch is read linearly in ln G against ln(Q_end - Q), second order in
    # its step: about 4e-9 at the default nodes, 4e-7 at a tenth of them.
    q_grid = np.linspace(1e-3, 1.0, 50)

    def f(q):
        return np.ones_like(np.asarray(q, dtype=np.float64))

    df = EllipseDF(
        q_grid,
        f,
        lambda q: np.zeros_like(np.asarray(q, dtype=np.float64)),
        float,
        {"below_floor": "none"},
        RULES,
    )
    q = np.concatenate([np.geomspace(1e-30, 1e-3, 20), np.linspace(1e-3, 0.999, 50)])
    np.testing.assert_allclose(df.G(q), q, rtol=1e-8)
    assert df.p_zero == 0.0 and not df.zero_map(8, 4)


def _lattice():
    model = EllipseModel(
        family="isotropic",
        r_a=float("inf"),
        beta_c=0.0,
        k_end=0,
        k_end_floor=None,
        k_end_v0=0,
        k_end_v0_floor=None,
        k_map=0,
        edge_alive=False,
        edge_gamma=float("nan"),
        vp_layers_edge=0,
        p0_top=float("nan"),
        kappa_top=float("nan"),
        vc_p=float("nan"),
        vc_alive=False,
        vp_layers_c=0,
    )
    return Lattice(RULES, model)


def test_layout_codes():
    lat = _lattice()
    nan = np.nan
    assert int(lat.layout_code(0.5, 1.0, nan)) == 1
    assert int(lat.layout_code(nan, 1.0, 0.5)) == 2
    assert LAYOUT_SEQ[int(lat.layout_code(0.3, 1.0, 0.6))] == "kr"
    assert LAYOUT_SEQ[int(lat.layout_code(0.6, 1.0, 0.3))] == "rk"
    # a break within BREAK_TOL of V_end coincides with it
    assert int(lat.layout_code(1.0 - 1e-12, 1.0, nan)) == 0
    # a kept V_k inside the zero grading's first sub-panel takes the deep code
    assert int(lat.layout_code(1e-9, 1.0, nan)) == 101


def test_panel_units_grade_toward_their_breaks():
    lat = _lattice()
    (units,) = lat.panel_units(0)
    assert units[0] == 0.0 and units[-1] == 1.0
    assert np.all(np.diff(units) > 0.0)
    # vp_layers_zero layers toward V = 0, vp_layers_kink toward V_end
    assert units[1] == pytest.approx(0.5 * RULES.vp_ratio**RULES.vp_layers_zero)
    assert 1.0 - units[-2] == pytest.approx(0.5 * RULES.vp_ratio**RULES.vp_layers_kink)
