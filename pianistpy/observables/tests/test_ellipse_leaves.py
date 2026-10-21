r"""Tests of the building blocks of the ellipse route of the LOSVD.

The tests cover the layer counts of `layer_counts`, the angle integral
:math:`\Lambda` of `lam_rule` and of its closed forms, Bulirsch's complete
elliptic integral `cel`, the cumulative DF integral of `EllipseDF`, the layout
codes of `Lattice.layout_code`, and the sub-panel grading and least node gaps
of `Lattice`.
"""

import numpy as np
import pytest

from pianistpy.observables._angle import lam_half, lam_mhalf, lam_rule, power_of
from pianistpy.observables._ellipse_df import EllipseDF
from pianistpy.observables._ellipse_rules import (
    EllipseModel,
    EllipseRules,
    gl01,
    layer_counts,
)
from pianistpy.observables._layout import LAYOUT_SEQ, Lattice

RULES = EllipseRules()


@pytest.mark.parametrize(
    ("beta_c", "k_end"),
    [(-0.5, 1), (-0.25, 2), (0.25, 4), (0.5, 10)],
)
def test_layers_toward_the_l0_point(beta_c, k_end):
    r"""Assert the layer count toward the L = 0 point for four values of
    :math:`\beta_{\rm c}`.

    The test asserts that ``k_end``, the first entry of `layer_counts`, equals
    1, 2, 4 and 10 at ``beta_c`` of -0.5, -0.25, 0.25 and 0.5. `layer_counts`
    sets each count at the tolerance :math:`10^{-8}`, which is
    :math:`10^{-2}` times :math:`10^{-6}`.
    """
    assert layer_counts(beta_c, RULES)[0] == k_end


def test_layers_stop_at_the_rules_floor_above_one_half():
    r"""Assert the layer count and the error floor at ``beta_c = 0.75``.

    At ``beta_c = 0.75`` the endpoint power :math:`1 - 2\beta_{\rm c}` is
    negative. A negative power is self-similar under the layers, so no layer
    count reaches the tolerance :math:`10^{-8}`. The test asserts that
    `layer_counts` returns 19 layers and an error floor of
    :math:`2.86 \times 10^{-8}` to a relative difference of :math:`10^{-2}`.
    """
    k, floor, _, _, _ = layer_counts(0.75, RULES)
    assert k == 19
    assert floor == pytest.approx(2.86e-8, rel=1e-2)


def test_layers_of_the_map():
    r"""Assert ``k_map``, the fifth entry of `layer_counts`, at three
    :math:`\beta_{\rm c}`.

    The test asserts that ``k_map`` is 7 at ``beta_c = 0.25``, 0 at
    ``beta_c = 0.5`` and 0 at ``beta_c = -0.25``.
    """
    assert layer_counts(0.25, RULES)[4] == 7
    assert layer_counts(0.5, RULES)[4] == 0
    assert layer_counts(-0.25, RULES)[4] == 0


def _angle_inputs(rng, n):
    r"""Return `n` random node arrays (c1, c2, c3, kap, d2, e2, lam, lpi) of
    the angle integral, drawn from `rng`, with c3 above c2**2 and the widths
    lam and lpi clipped to :math:`[10^{-15}, \pi]`."""
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
    """Assert that `lam_rule` matches the closed forms of the angle integral.

    The closed form is `lam_half` at ``beta_c = 0.5`` and `lam_mhalf` at
    ``beta_c = -0.5``. On 200 random inputs, the test asserts that `lam_rule`
    with 64 Gauss nodes per half agrees with the closed form to a relative
    difference of :math:`10^{-12}`.
    """
    rng = np.random.default_rng(1)
    c1, c2, c3, kap, d2, e2, lam, lpi = _angle_inputs(rng, 200)
    rule = lam_rule(power_of(beta_c), c1, c2, c3, kap, d2, e2, lam, lpi, 64)
    e = c3 - c2 * c2
    dm = c1 * c1 - c2 * c2
    closed = lam_half(c1, c3, e, dm) if beta_c > 0 else lam_mhalf(c1, c2, e, dm)
    np.testing.assert_allclose(closed, rule, rtol=1e-12)


def test_cel_against_the_complete_integrals():
    r"""Assert that `cel` matches SciPy's complete elliptic integrals.

    On 2000 random parameters m and n, the test asserts that `cel` gives E(m)
    against ``scipy.special.ellipe`` and K(m) against ``scipy.special.ellipk``
    to a relative difference of :math:`4 \times 10^{-15}`. The test asserts
    that `cel` gives ``k_c**2 * (Pi - K)`` against Carlson's R_J form to a
    relative difference of :math:`8 \times 10^{-15}`. The test asserts that
    the value of `cel` at one element is the same, bit for bit, whether the
    element is passed alone, in a 1-D array or in a broadcast 2-D block.
    """
    # The identity k_c**2 (Pi - K) = k_c**2 (n/3) R_J(0, k_c**2, 1, 1 - n), with
    # k_c**2 = 1 - m, gives the reference for the third integral.
    from scipy.special import ellipe, ellipk, elliprj

    from pianistpy.observables._angle import cel

    rng = np.random.default_rng(3)
    m = rng.uniform(1e-12, 1.0 - 1e-12, 2000)
    n = rng.uniform(1e-9, 1.0 - 1e-9, 2000)
    kc = np.sqrt(1.0 - m)
    np.testing.assert_allclose(cel(kc, 1.0, 1.0, kc**2), ellipe(m), rtol=4e-15)
    np.testing.assert_allclose(cel(kc, 1.0, 1.0, 1.0), ellipk(m), rtol=4e-15)
    rj = kc**2 * (n / 3.0) * elliprj(0.0, kc**2, 1.0, 1.0 - n)
    np.testing.assert_allclose(cel(kc, 1.0 - n, 0.0, kc**2 * n), rj, rtol=8e-15)
    # The rows of the ellipse route pass 2-D blocks broadcast between the
    # arguments of cel, so the value of cel at one element must not depend on
    # the other elements in the call or on the shape of the call.
    alone = np.array([cel(k, 1.0, 1.0, k * k) for k in kc[:20]])
    assert alone.tobytes() == cel(kc[:20], 1.0, 1.0, kc[:20] ** 2).tobytes()
    block = cel(kc[:20].reshape(4, 5), (1.0 - n[:5])[None, :], 0.0, 0.5)
    assert block.shape == (4, 5)
    assert block[0].tobytes() == cel(kc[:5], 1.0 - n[:5], 0.0, 0.5).tobytes()


def test_rule_at_isotropy_is_pi():
    r"""Assert that `lam_rule` at ``beta_c = 0`` equals pi.

    At ``beta_c = 0`` the factor :math:`P^{-\beta_{\rm c}}` of the angle
    integrand is 1, so the angle integral over :math:`[0, \pi]` is pi. On 50
    random inputs, the test asserts that `lam_rule` with 16 Gauss nodes per
    half equals pi to a relative difference of :math:`10^{-14}`.
    """
    rng = np.random.default_rng(2)
    args = _angle_inputs(rng, 50)
    np.testing.assert_allclose(lam_rule(power_of(0.0), *args, 16), np.pi, rtol=1e-14)


def test_cumulative_of_a_flat_df():
    """Assert that the cumulative integral G(Q) of a flat DF equals Q.

    The DF is F = 1 at every energy, above and below the DF's floor, so
    G(Q) equals Q on every branch of `EllipseDF.G`. At 70 energies from
    :math:`10^{-30}` to 0.999, the test asserts that G(Q) equals Q to a
    relative difference of :math:`10^{-8}`. The test also asserts that
    ``p_zero``, the exponent of F at K = 0, is 0, and that the s panel at
    K = 0 takes no map at 8 nodes and 4 layers.
    """
    # The upper branch of G interpolates ln G linearly in ln(Q_end - Q). The
    # error of the linear interpolation is second order in the interpolation
    # step. The tolerance rtol=1e-8 of the test allows for that error.
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
    """Return the `Lattice` of RULES and an isotropic `EllipseModel` with every
    layer count 0."""
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
    """Assert the layout codes of rows with breaks at ``V_k`` and ``V_rad``.

    With ``V_end = 1``, the test asserts code 1 for ``V_k = 0.5`` alone and
    code 2 for ``V_rad = 0.5`` alone. The test asserts the letters "kr" for
    ``V_k = 0.3`` below ``V_rad = 0.6`` and the letters "rk" for
    ``V_k = 0.6`` above ``V_rad = 0.3``. The test asserts code 0 for
    ``V_k = 1 - 1e-12`` and code 101 for ``V_k = 1e-9``.
    """
    lat = _lattice()
    nan = np.nan
    assert int(lat.layout_code(0.5, 1.0, nan)) == 1
    assert int(lat.layout_code(nan, 1.0, 0.5)) == 2
    assert LAYOUT_SEQ[int(lat.layout_code(0.3, 1.0, 0.6))] == "kr"
    assert LAYOUT_SEQ[int(lat.layout_code(0.6, 1.0, 0.3))] == "rk"
    # V_k = 1 - 1e-12 lies within BREAK_TOL (1e-9) of V_end = 1, so V_k
    # coincides with V_end and the row has no interior break.
    assert int(lat.layout_code(1.0 - 1e-12, 1.0, nan)) == 0
    # V_k = 1e-9 lies inside the first sub-panel of the grading toward V = 0,
    # so the code of V_k alone, 1, is raised by DEEP_K = 100 to 101.
    assert int(lat.layout_code(1e-9, 1.0, nan)) == 101


def _least_gaps_at_once(lat, br, units):
    """Return the least gap of each panel, from the whole lattice at once.

    The gap between two neighboring nodes is the step between the nodes over
    the upper node's value. A gap inside a panel belongs to that panel. A gap
    across a break belongs to the narrower of the two panels, the lower panel
    where both panels have equal width. The result is the reference for
    `Lattice.lattice_min_gaps`, which works on blocks of nodes.
    """
    xg = gl01(lat.rules.vp_nodes)[0]
    w = np.diff(br, axis=1)
    nodes, steps, owner, last = [], [], [], None
    for i in range(br.shape[1] - 1):
        u = units[i]
        un = (u[:-1, None] + np.diff(u)[:, None] * xg[None, :]).ravel()
        nodes.append(br[:, i : i + 1] + w[:, i : i + 1] * un[None, :])
        if last is not None:
            steps.append((w[:, i - 1] * (1.0 - last) + w[:, i] * un[0])[:, None])
        steps.append(w[:, i : i + 1] * np.diff(un)[None, :])
        owner.append(np.full(un.size, i))
        last = un[-1]
    gap = np.concatenate(steps, axis=1) / np.concatenate(nodes, axis=1)[:, 1:]
    owner = np.concatenate(owner)
    o0, o1 = owner[:-1], owner[1:]
    gmin = np.full(w.shape, np.inf)
    for i in range(w.shape[1]):
        inside = (o0 == i) & (o1 == i)
        gmin[:, i] = np.minimum(gmin[:, i], gap[:, inside].min(axis=1))
    for j in np.flatnonzero(o0 != o1):
        a, b = int(o0[j]), int(o1[j])
        to_a = w[:, a] <= w[:, b]
        gmin[to_a, a] = np.minimum(gmin[to_a, a], gap[to_a, j])
        gmin[~to_a, b] = np.minimum(gmin[~to_a, b], gap[~to_a, j])
    return gmin


@pytest.mark.parametrize("chunk", [1, 7, RULES.chunk])
def test_least_gaps_are_those_of_the_whole_lattice(chunk):
    """Assert that the blockwise least gaps equal the whole-lattice least gaps.

    For blocks of 1 node, 7 nodes and ``RULES.chunk`` nodes, and for layout
    codes 3 and 4, the test asserts that `Lattice.lattice_min_gaps` equals
    `_least_gaps_at_once` bit for bit. The 40 rows include two breaks
    :math:`10^{-15}` apart, three panels of equal width, and a break
    :math:`10^{-13}` below the upper end at 1.
    """
    import dataclasses

    lat = _lattice()
    lat = Lattice(dataclasses.replace(RULES, chunk=chunk), lat.model)
    rng = np.random.default_rng(300)
    m = 40
    a = np.sort(rng.uniform(0.0, 1.0, (m, 2)), axis=1)
    a[0] = (0.5, 0.5 + 1e-15)
    a[1] = (1.0 / 3.0, 2.0 / 3.0)
    a[2] = (0.25, 1.0 - 1e-13)
    br = np.concatenate([np.zeros((m, 1)), a, np.ones((m, 1))], axis=1)
    for code in (3, 4):
        units = lat.panel_units(code)
        got = lat.lattice_min_gaps(br, units)
        assert got.tobytes() == _least_gaps_at_once(lat, br, units).tobytes()


def test_panel_units_grade_toward_their_breaks():
    """Assert the unit sub-panel edges of the single panel of layout code 0.

    The test asserts that the edges run from 0 to 1 and increase strictly.
    The test asserts that the first interior edge is
    ``0.5 * vp_ratio**vp_layers_zero`` and that the last interior edge lies
    ``0.5 * vp_ratio**vp_layers_kink`` below 1.
    """
    lat = _lattice()
    (units,) = lat.panel_units(0)
    assert units[0] == 0.0 and units[-1] == 1.0
    assert np.all(np.diff(units) > 0.0)
    # The panel of code 0 has vp_layers_zero layers toward V = 0 and
    # vp_layers_kink layers toward V_end.
    assert units[1] == pytest.approx(0.5 * RULES.vp_ratio**RULES.vp_layers_zero)
    assert 1.0 - units[-2] == pytest.approx(0.5 * RULES.vp_ratio**RULES.vp_layers_kink)
