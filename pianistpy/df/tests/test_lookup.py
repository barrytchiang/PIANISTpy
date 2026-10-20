"""Tests of the DF lookup against closed forms."""

import dataclasses

import numpy as np
import pytest

from pianistpy.df.lookup import (
    LookupArrays,
    evaluate_lookup,
    log10_df,
    prepare_lookup,
)
from pianistpy.df.stages import AsymptoticForm, DfTable
from pianistpy.exceptions import ResultError

P_0 = 1.0
# a power law F = K^q: log10 F is linear in ln K, which the cubic Hermite reproduces
Q_EXPONENT = 1.5


def _table(slopes=True, k_below=False):
    k = np.geomspace(1e-3, 0.9, 200)
    # d log10 F / d ln K of F = K^q is q / ln 10 on every node
    slope = np.full(k.shape, Q_EXPONENT / np.log(10.0) if slopes else np.nan)
    below = {}
    if k_below:
        kb = np.geomspace(1e-5, 1e-3, 30)[:-1]
        below = dict(
            k_below=kb[None, :],
            n_below=(kb.size,),
            log_df_below=(Q_EXPONENT * np.log10(kb))[None, :],
            log_df_slope_below=np.full((1, kb.size), Q_EXPONENT / np.log(10.0)),
        )
    return DfTable(
        k=k[None, :],
        log_df=(Q_EXPONENT * np.log10(k))[None, :],
        log_df_slope=slope[None, :],
        i_first=(0,),
        i_last=(k.size - 1,),
        slopes_exact=(slopes,),
        **below,
    )


def _form(**fields):
    base = dict(
        p_0=P_0,
        central_label=("test",),
        central_exponent=(-0.5,),
        central_amplitude=(0.0,),
        tail_exponent=(0.0,),
        tail_amplitude=(0.0,),
        tail_log_exponent=(0.0,),
        central_form=("analytic",),
        central_log_coefficient=(np.nan,),
        central_next_form=("none",),
        central_next_exponent=(np.nan,),
        central_next_coefficient=(np.nan,),
        tail_log_scale=(1.0,),
        below_floor=("zero",),
        below_floor_coefficients=((),),
    )
    base.update(fields)
    return AsymptoticForm(**base)


@pytest.mark.parametrize("slopes", [True, False])
def test_inside_the_table_a_power_law_is_reproduced(slopes):
    table = _table(slopes=slopes)
    k = np.geomspace(1e-3, 0.9, 1001)
    np.testing.assert_allclose(
        log10_df(table, _form(), k), Q_EXPONENT * np.log10(k), rtol=0, atol=1e-13
    )


def test_the_nodes_are_returned_exactly():
    table = _table()
    assert np.array_equal(log10_df(table, _form(), table.k[0]), table.log_df[0])


def test_the_nodes_below_the_table_continue_it():
    table = _table(k_below=True)
    k = np.geomspace(1e-5, 1e-3, 301)
    np.testing.assert_allclose(
        log10_df(table, _form(), k), Q_EXPONENT * np.log10(k), rtol=0, atol=1e-13
    )


def test_zero_outside_the_support():
    table = _table()
    k = np.array([-1.0, 0.0, 1e-4, P_0, 2.0])
    assert np.all(log10_df(table, _form(), k) == -np.inf)


def _above_closed_form(kind, s, amplitude, log_coefficient, u):
    k_top = 0.9
    df_top = k_top**Q_EXPONENT
    u_top = P_0 - k_top
    if kind == "power":
        return np.log10(df_top + amplitude * (u**s - u_top**s))
    if kind == "log":
        return np.log10(df_top + log_coefficient * np.log(u_top / u))
    return np.full(u.shape, np.log10(df_top))


@pytest.mark.parametrize(
    "kind, s, amplitude, log_coefficient",
    [
        ("power", -0.5, 0.3, np.nan),
        ("power", -2.5, 1e-3, np.nan),
        ("log", 0.0, 0.0, 0.2),
        ("analytic", 0.0, 0.0, np.nan),
    ],
)
def test_above_the_table_the_center_form(kind, s, amplitude, log_coefficient):
    form = _form(
        central_form=(kind,),
        central_exponent=(s,),
        central_amplitude=(amplitude,),
        central_log_coefficient=(log_coefficient,),
    )
    k = P_0 - np.geomspace(0.1, 1e-9, 200)[1:]
    # the closed form at the u the energies hold, P_0 - k
    np.testing.assert_allclose(
        log10_df(_table(), form, k),
        _above_closed_form(kind, s, amplitude, log_coefficient, P_0 - k),
        rtol=0,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "rule, coefficients, closed_form",
    [
        ("taylor", (2.0, -3.0, 1e-3), lambda k: 2.0 * k - 1.5 * k * k),
        ("edge_law", (0.75, 2.0, -3.0), lambda k: k**0.75 * (2.0 - 3.0 * k)),
    ],
)
def test_below_the_table_the_rule(rule, coefficients, closed_form):
    form = _form(below_floor=(rule,), below_floor_coefficients=(coefficients,))
    k = np.geomspace(1e-12, 1e-3, 100)[:-1]
    np.testing.assert_allclose(
        log10_df(_table(), form, k), np.log10(closed_form(k)), rtol=1e-14
    )


def test_below_the_table_the_tail():
    amplitude, a, b, e_0 = 0.7, 0.4, 1.0, 1.0
    form = _form(
        below_floor=("tail",),
        tail_amplitude=(amplitude,),
        tail_exponent=(a,),
        tail_log_exponent=(b,),
        tail_log_scale=(e_0,),
    )
    k = np.geomspace(1e-300, 1e-3, 100)[:-1]
    expected = np.log10(amplitude) - a * np.log10(k) - b * np.log10(np.log(e_0 / k))
    np.testing.assert_allclose(log10_df(_table(), form, k), expected, rtol=1e-14)


def test_a_signed_table_is_refused():
    table = dataclasses.replace(_table(), sign=np.ones((1, 200), dtype=np.int8))
    with pytest.raises(ResultError):
        log10_df(table, _form(), np.array([0.5]))


def test_the_kernel_is_the_host_function():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    form = _form(
        central_form=("power",),
        central_exponent=(-0.5,),
        central_amplitude=(0.3,),
        below_floor=("taylor",),
        below_floor_coefficients=((2.0, -3.0, 1e-3),),
    )
    table = _table(k_below=True)
    k = np.concatenate(
        [
            [-1.0, 0.0, P_0, 2.0],
            np.geomspace(1e-8, 0.9, 2000),
            P_0 - np.geomspace(0.1, 1e-9, 200),
        ]
    )
    host = log10_df(table, form, k)
    kernel = np.asarray(jax.jit(lambda q: log10_df(table, form, q, xp=jax.numpy))(k))
    assert np.array_equal(np.isfinite(kernel), np.isfinite(host))
    finite = np.isfinite(host)
    np.testing.assert_allclose(kernel[finite], host[finite], rtol=0, atol=1e-13)


def _steep_top_table():
    """``F = u^-2`` in the depth ``u = P_0 - K``, tabulated up to ``u = 1e-7``
    below the top; each node's depth is formed from its K (exact there), so the table's
    values and slopes are the closed form's at the nodes themselves."""
    k = np.unique(P_0 - np.geomspace(0.5, 1e-7, 4000))
    u = P_0 - k
    return DfTable(
        k=k[None, :],
        log_df=(-2.0 * np.log10(u))[None, :],
        log_df_slope=(2.0 * k / (u * np.log(10.0)))[None, :],
        i_first=(0,),
        i_last=(k.size - 1,),
        slopes_exact=(True,),
    )


def test_the_entry_by_the_depth_is_the_entry_by_the_energy_where_no_depth_is_held():
    table = _table(k_below=True)
    arrays, kinds = prepare_lookup(table, _form())
    k = np.concatenate(
        [np.geomspace(1e-5, 0.9, 1000), P_0 - np.geomspace(0.1, 1e-9, 50)]
    )
    by_k = evaluate_lookup(arrays, kinds, k)
    assert np.array_equal(
        evaluate_lookup(arrays, kinds, k, u=np.full(k.size, np.nan)), by_k
    )


def test_by_the_depth_the_top_cells_read_the_depth_itself():
    # near the top a caller holds u exactly where K = P_0 - u has lost it: at
    # u = 1e-6, one ulp of K is 1.1e-10 of u, so by the energy F moves in steps
    # of K; by the depth it is the closed form to rounding
    arrays, kinds = prepare_lookup(_steep_top_table(), _form())
    node = int(np.argmin(np.abs((P_0 - arrays.k[: int(arrays.n)]) - 1e-6)))
    u_node = P_0 - float(arrays.k[node])
    u = u_node * (1.0 + np.arange(1, 33) * 1e-12)
    exact = -2.0 * np.log10(u)
    by_u = evaluate_lookup(arrays, kinds, P_0 - u, u=u)
    by_k = evaluate_lookup(arrays, kinds, P_0 - u)
    np.testing.assert_allclose(by_u, exact, rtol=0, atol=1e-14)
    assert np.max(np.abs(by_k - exact)) > 1e-12


def test_by_the_depth_a_depth_below_half_an_ulp_of_p_0_reads_the_center_form():
    # P_0 - u rounds to P_0 below u = 5.55e-17; by the depth the form
    # is still read there, by the energy P_0 itself is where F ends
    form = _form(
        central_form=("power",), central_exponent=(-2.5,), central_amplitude=(1e-3,)
    )
    arrays, kinds = prepare_lookup(_table(), form)
    u = np.array([1e-20, 3.3e-17, 5e-17, 1e-16, 1e-12])
    by_u = evaluate_lookup(arrays, kinds, P_0 - u, u=u)
    np.testing.assert_allclose(
        by_u, _above_closed_form("power", -2.5, 1e-3, np.nan, u), rtol=1e-14
    )
    assert np.all(np.isneginf(evaluate_lookup(arrays, kinds, P_0 - u[:3])))
    # at and beyond P_0 nothing, by either entry
    for u_end in (0.0, -1e-17):
        assert np.isneginf(evaluate_lookup(arrays, kinds, P_0, u=np.array([u_end])))[0]


def test_two_models_of_the_same_shapes_share_one_compiled_lookup():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    kernel = jax.jit(evaluate_lookup, static_argnames=("kinds", "xp"))
    k = jnp.asarray(np.geomspace(1e-4, 0.999, 513))
    results = []
    for exponent in (1.5, 2.0):
        table = dataclasses.replace(
            _table(), log_df=(exponent * np.log10(_table().k[0]))[None, :]
        )
        form = _form(central_form=("power",), central_amplitude=(2.0,))
        arrays, kinds = prepare_lookup(table, form)
        traced = LookupArrays(*(jnp.asarray(a) for a in arrays))
        results.append(np.asarray(kernel(traced, kinds, k, xp=jnp)))
        np.testing.assert_allclose(
            results[-1], log10_df(table, form, np.asarray(k)), rtol=0, atol=1e-13
        )
    assert kernel._cache_size() == 1
    assert not np.array_equal(results[0], results[1])


@pytest.mark.parametrize(
    "fields",
    [
        dict(central_form=("fitted",)),
        dict(below_floor=("linear",)),
        dict(central_next_form=()),
    ],
)
def test_a_form_it_is_not_given_is_refused(fields):
    with pytest.raises(ResultError):
        log10_df(_table(), _form(**fields), np.array([0.5]))
