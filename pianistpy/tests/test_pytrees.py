"""The input objects as pytrees: traced parameters, static configuration."""

import math

import numpy as np
import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df.analytic import analytic_model

jax = pytest.importorskip("jax")
jnp = jax.numpy


def test_anisotropy_leaves_are_its_parameters_and_its_family_is_static():
    a = Anisotropy(beta_c=-0.25, r_a=10.0, kind="II")
    leaves, treedef = jax.tree_util.tree_flatten(a)
    assert leaves == [-0.25, 10.0]
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back == a and back.family == "COM_II" and back.model == "com"
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, a)
    assert doubled.beta_c == -0.5 and doubled.r_a == 20.0
    assert doubled.family == "COM_II"


def test_profile_is_configuration_with_no_leaves():
    p = DoublePowerLaw.nfw()
    leaves, treedef = jax.tree_util.tree_flatten(p)
    assert leaves == []
    assert jax.tree_util.tree_unflatten(treedef, leaves) == p


def test_component_and_truncation_round_trip():
    c = Component(
        DoublePowerLaw.hernquist(), Anisotropy(beta_c=0.3), mass=2.0, name="halo"
    )
    leaves, treedef = jax.tree_util.tree_flatten(c)
    assert leaves == [0.3, math.inf, 2.0]
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back == c
    t = Truncation(k_t=0.2)
    leaves, treedef = jax.tree_util.tree_flatten(t)
    assert leaves == [0.2]
    assert jax.tree_util.tree_unflatten(treedef, leaves) == t


def test_gradient_through_the_anisotropy_factor_in_beta_c():
    def g(a):
        return a.factor(2.0, xp=jnp)

    a = Anisotropy(beta_c=0.3)
    grad = jax.grad(g)(a)
    # d/d beta_c of 2^(-2 beta_c) = -2 ln 2 2^(-2 beta_c)
    assert float(grad.beta_c) == pytest.approx(
        -2.0 * math.log(2.0) * 2.0**-0.6, rel=1e-12
    )
    assert grad.family == "constant"
    c = Anisotropy(beta_c=-0.25, r_a=10.0)
    grad = jax.grad(lambda a: a.factor(3.0, xp=jnp))(c)
    step = 1e-6
    central = (
        Anisotropy(beta_c=-0.25, r_a=10.0 + step).factor(3.0)
        - Anisotropy(beta_c=-0.25, r_a=10.0 - step).factor(3.0)
    ) / (2.0 * step)
    assert float(grad.r_a) == pytest.approx(float(central), rel=1e-6)


def test_jit_retraces_on_the_family_not_on_the_values():
    calls = []

    @jax.jit
    def g(a, r):
        calls.append(1)
        return a.factor(r, xp=jnp)

    r = jnp.asarray([0.5, 2.0])
    g(Anisotropy(beta_c=0.3), r)
    g(Anisotropy(beta_c=0.1), r)
    assert len(calls) == 1
    g(Anisotropy(beta_c=0.1, r_a=5.0), r)
    assert len(calls) == 2
    # beta_c traced, the power is XLA's, not NumPy's: at rounding (measured on
    # the CPU, 4000 radii 1e-3 to 1e3 and five beta_c: up to 8 ulp, 86 percent
    # to the bit; the form g = R^(-2 beta_c) u^(-s) the same, up to 7 ulp)
    got = np.asarray(g(Anisotropy(beta_c=0.1, r_a=5.0), r))
    want = Anisotropy(beta_c=0.1, r_a=5.0).factor(np.asarray([0.5, 2.0]))
    assert np.all(np.abs(got - want) <= 16 * np.spacing(np.abs(want)))


def test_a_model_is_a_pytree_of_its_arrays_with_its_records_static():
    model = analytic_model("plummer", n_nodes=201)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    assert len(leaves) > 10
    assert all(
        isinstance(x, (np.ndarray, float, int)) or hasattr(x, "shape") for x in leaves
    )
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.identity == model.identity
    assert back.settings is model.settings
    assert back.validation is model.validation
    np.testing.assert_array_equal(back.table.log_df, model.table.log_df)
    np.testing.assert_array_equal(back.profile.r, model.profile.r)
    assert back.form.central_form == model.form.central_form
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, model)
    np.testing.assert_array_equal(doubled.profile.r, 2 * model.profile.r)
    assert doubled.profile.source == model.profile.source
    assert doubled.components[0].anisotropy.family == "isotropic"


def test_a_jitted_function_takes_a_model_and_reuses_its_trace():
    model = analytic_model("plummer", n_nodes=201)
    calls = []

    @jax.jit
    def mass_at_edge(m):
        calls.append(1)
        return jnp.sum(m.profile.mass_trunc[-1:]) + 0.0 * jnp.sum(
            m.table.log_df[0, :10]
        )

    first = float(mass_at_edge(model))
    second = float(mass_at_edge(model))
    assert first == second == pytest.approx(float(model.profile.mass_trunc[-1]))
    assert len(calls) == 1
