r"""Tests of the DF table's derivative on a Method II child.

S_E1's parent (NFW isotropic, truncated at
:math:`\mathcal{K}_{\rm t} = 0.0079589980165958`) and two children on it,
COM Type I (-0.2, 1.8) and OM Type I (:math:`R_{\rm a} = 5`): the rule
against central differences of the child solve on the grid held at theta
(Richardson over two steps), the values against the solve's bit for bit, and
the VJP against the rule.
"""

import numpy as np
import pytest

from pianistpy import Truncation, platform
from pianistpy.anisotropy import Anisotropy
from pianistpy.df import compute_df, truncate
from pianistpy.df.central import rho_aux_series
from pianistpy.df.gradient import (
    _child_forward,
    _child_held,
    _dual_series,
    _with,
    make_inversion_table,
    table_jacobian,
)
from pianistpy.df.inputs import Component
from pianistpy.df.tabulated import tabulated_df
from pianistpy.exceptions import InputError
from pianistpy.profiles import DoublePowerLaw
from pianistpy.settings import Settings

K_T = 0.0079589980165958
# the transpose held to the bit is a property of a fixed reduction order,
# which a GPU does not promise; there the accuracy grid judges
CPU_ONLY = pytest.mark.skipif(
    platform() != "cpu",
    reason="deterministic recomputation, a property of a fixed reduction order; "
    "on a GPU the accuracy grid judges",
)
MEMBERS = {
    "com": Anisotropy(beta_c=-0.2, r_a=1.8),
    "om": Anisotropy(r_a=5.0),
}


@pytest.fixture(scope="module")
def parent():
    return truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy())), Truncation(k_t=K_T)
    )


@pytest.fixture(scope="module", params=sorted(MEMBERS))
def child(request, parent):
    c = Component(profile=parent.profile, anisotropy=MEMBERS[request.param])
    values, jac, record = table_jacobian(c, Settings())
    return c, values, jac, record


def test_the_values_are_the_solve(child):
    c, values, _, _ = child
    _, table, form, *_ = tabulated_df(c, Settings())
    assert np.array_equal(values["k"], table.k[0])
    assert np.array_equal(values["log_df"], table.log_df[0])
    np.testing.assert_array_equal(values["edge_law"], form.below_floor_coefficients[0])
    assert values["center"][0] == form.central_exponent[0]
    assert values["center"][1] == form.central_amplitude[0]


def test_the_rule_against_the_frozen_grid(child):
    # the solve at theta +- h and +- h/2 on the grid and kept nodes held at
    # theta, Richardson, h = 1e-3
    c, values, jac, _ = child
    s = Settings()
    held = _child_held(c, s)
    a = c.anisotropy
    for theta in jac:
        x0 = float(a.beta_c) if theta == "beta_c" else float(a.r_a)

        def at(x, theta=theta):
            return _with(
                c,
                x if theta == "beta_c" else a.beta_c,
                x if theta == "r_a" else a.r_a,
            )

        diff = {}
        for h in (1e-3, 5e-4):
            up, dn = (
                _child_forward(at(x0 + h), s, held),
                _child_forward(at(x0 - h), s, held),
            )
            diff[h] = {k: (up[k] - dn[k]) / (2 * h) for k in up}
        rich = {k: (4 * diff[5e-4][k] - diff[1e-3][k]) / 3 for k in diff[1e-3]}
        # every row, in beta_c the rows' exponent part at its measured step
        np.testing.assert_allclose(
            jac[theta]["log_df"], rich["log_df"], rtol=1e-6, atol=1e-6
        )
        for key in ("edge_law", "center"):
            np.testing.assert_allclose(
                jac[theta][key], rich[key], rtol=1e-6, atol=1e-12
            )


def test_the_dual_series_is_the_series(child):
    c, _, _, _ = child
    p, a = c.unit_profile, c.anisotropy
    bce = 0.0 if a.family.startswith("OM") else float(a.beta_c)
    xm = 3.0 * max(float(p.rho_alpha), 2.0)
    ref = rho_aux_series(
        p.rho_alpha, p.rho_beta, p.rho_gamma, bce, a.family, a.r_a, p.rho_norm, x_max=xm
    )
    dual = _dual_series(p, bce, a.family, float(a.r_a), "r_a", xm)
    assert [s for s, *_ in dual] == pytest.approx([s for s, _ in ref], abs=1e-12)
    assert [v for _, v, *_ in dual] == pytest.approx([v for _, v in ref], rel=1e-13)


@CPU_ONLY
def test_the_child_vjp_is_the_transpose(child):
    import jax

    c, values, jac, _ = child
    table, params, values2 = make_inversion_table(c, Settings())
    theta0 = np.array(
        [
            float(c.anisotropy.beta_c) if p == "beta_c" else float(c.anisotropy.r_a)
            for p in params
        ]
    )
    out, vjp = jax.vjp(table, theta0)
    for k in out:
        np.testing.assert_array_equal(np.asarray(out[k]), values2[k])
    rng = np.random.default_rng(3)
    v = {k: rng.normal(size=np.shape(out[k])) for k in out}
    (g,) = vjp(v)
    for i, p in enumerate(params):
        want = sum(float(np.sum(v[k] * jac[p][k])) for k in out)
        assert float(g[i]) == pytest.approx(want, rel=1e-12)


def test_cuddeford_type_ii_child_is_refused(parent):
    c = Component(
        profile=parent.profile, anisotropy=Anisotropy(beta_c=-0.2, r_a=100.0, kind="II")
    )
    with pytest.raises(InputError, match="Type II"):
        table_jacobian(c, Settings())
