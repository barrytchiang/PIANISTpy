"""Tests of the superposition of models that share one truncated pair."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy import Anisotropy, Component, InputError, Settings, read
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition import mixture, moments, superpose


def _member(parent, beta_c, n_keep=None):
    """A Method II model on `parent`'s truncated profile with its own
    anisotropy; with `n_keep`, its table cut to that many energies, a grid of
    its own."""
    component = Component(profile=parent.profile, anisotropy=Anisotropy(beta_c=beta_c))
    table = parent.table
    if n_keep is not None:
        table = dataclasses.replace(
            table,
            k=table.k[:, :n_keep],
            log_df=table.log_df[:, :n_keep],
            log_df_slope=table.log_df_slope[:, :n_keep],
            i_last=(n_keep - 1,),
        )
    return dataclasses.replace(parent, components=(component,), table=table)


@pytest.fixture(scope="module")
def parent():
    return _lowered_hernquist()


def test_the_members_are_stacked_on_their_own_grids(parent):
    a, b = _member(parent, 0.0), _member(parent, -0.2, n_keep=1500)
    model = superpose({a: 0.25, b: 0.75})
    # one component per member, its weight as its mass relative to the first
    assert [c.mass for c in model.components] == [1.0, 3.0]
    assert [c.anisotropy.beta_c for c in model.components] == [0.0, -0.2]
    table = model.table
    assert table.k.shape == (2, a.table.k.shape[1])
    assert table.k[0].tobytes() == a.table.k[0].tobytes()
    assert table.log_df[0].tobytes() == a.table.log_df[0].tobytes()
    # the shorter grid: its own energies, then its last one repeated, NaN there
    assert table.k[1, :1500].tobytes() == b.table.k[0].tobytes()
    assert np.all(table.k[1, 1500:] == b.table.k[0, -1])
    assert np.all(np.isnan(table.log_df[1, 1500:]))
    assert table.i_last == (a.table.i_last[0], 1499)
    assert model.form.below_floor == a.form.below_floor + b.form.below_floor
    assert model.form.p_0 == a.form.p_0
    # the shared density split by weight, its rows summing to it
    rho = parent.profile.rho_trunc[0]
    np.testing.assert_array_equal(model.profile.rho_trunc[0], 0.25 * rho)
    np.testing.assert_array_equal(model.profile.rho_trunc[1], 0.75 * rho)
    np.testing.assert_allclose(model.profile.rho_trunc.sum(axis=0), rho, rtol=1e-15)
    assert model.profile.r.tobytes() == parent.profile.r.tobytes()
    assert len(model.profile.edge_exponent) == 2
    assert model.profile.model == model.identity
    # the same request, the same identity; another weight, another
    assert superpose({a: 0.25, b: 0.75}).identity == model.identity
    assert superpose({a: 0.5, b: 0.5}).identity != model.identity


def test_the_weights_are_normalized_with_a_warning(parent, caplog):
    # the log, not the recent messages: those are a bounded queue, full after
    # the tests before this one, so that their length does not grow
    a, b = _member(parent, 0.0), _member(parent, -0.2)
    with caplog.at_level("WARNING", logger="pianistpy"):
        model = superpose({a: 1.0, b: 3.0})
    assert any("normalized" in r.getMessage() for r in caplog.records)
    assert [c.mass for c in model.components] == [1.0, 3.0]
    np.testing.assert_array_equal(
        model.profile.rho_trunc[1], 0.75 * parent.profile.rho_trunc[0]
    )


def test_a_superposed_model_is_its_file(parent, tmp_path):
    model = superpose({_member(parent, 0.0): 0.4, _member(parent, -0.2, 1500): 0.6})
    model.write(tmp_path / "s.h5")
    back = read(tmp_path / "s.h5")
    assert back.identity == model.identity
    for name in ("k", "log_df", "log_df_slope"):
        assert (
            np.asarray(getattr(back.table, name)).tobytes()
            == getattr(model.table, name).tobytes()
        )
    assert back.profile.rho_trunc.tobytes() == model.profile.rho_trunc.tobytes()
    assert [c.mass for c in back.components] == [c.mass for c in model.components]


def test_superpose_refuses_by_name(parent):
    a, b = _member(parent, 0.0), _member(parent, -0.2)
    with pytest.raises(InputError, match="at least two"):
        superpose({a: 1.0})
    with pytest.raises(InputError, match="positive and finite"):
        superpose({a: 1.0, b: 0.0})
    other = _lowered_hernquist(k_t=0.08)
    with pytest.raises(InputError, match="share one truncated pair"):
        superpose({a: 0.5, _member(other, -0.2): 0.5})
    with pytest.raises(InputError, match="other settings"):
        superpose({a: 0.5, dataclasses.replace(b, settings=Settings(quick=True)): 0.5})
    with pytest.raises(InputError, match="truncated profile"):
        superpose({a: 0.5, parent: 0.5})


def test_a_mixture_is_differentiable_in_its_weights():
    # the gradient contract (R-47, R-48): a quantity of the mixture is linear
    # in the members' values and, through the normalization, smooth in the
    # weights; reverse mode against the closed form and central differences
    values = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.5], [2.0, 2.0, 7.0]])
    weights = jnp.array([0.2, 0.5, 0.3])

    def moment(w):
        return jnp.sum(mixture(w, values, jnp) ** 2)

    grad = np.asarray(jax.grad(moment)(weights))
    total = float(weights.sum())
    m = np.asarray(mixture(weights, values, jnp))
    exact = np.array([2.0 * np.dot(m, (np.asarray(v) - m)) / total for v in values])
    np.testing.assert_allclose(grad, exact, rtol=1e-13)
    for j in range(3):
        step = jnp.zeros(3).at[j].set(1e-6)
        central = (float(moment(weights + step)) - float(moment(weights - step))) / 2e-6
        np.testing.assert_allclose(central, grad[j], rtol=1e-6)
    np.testing.assert_allclose(mixture(weights, values), m, rtol=1e-15)


def test_the_moments_of_a_mixture(parent):
    # isotropic members: the mixture is isotropic and its dispersion each
    # member's; an isotropic and a constant-beta member: the weighted sums
    r = np.geomspace(1e-3, 0.9 * float(parent.profile.r_t), 40)
    same = moments(superpose({_member(parent, 0.0): 0.3, _member(parent, 0.0): 0.7}), r)
    np.testing.assert_array_equal(same["beta_sigma"][same["sigma_r2"] > 0], 0.0)
    np.testing.assert_allclose(
        same["sigma_r2"], same["sigma_r2_components"][0], rtol=1e-14
    )
    mixed = moments(
        superpose({_member(parent, 0.0): 0.4, _member(parent, -0.2): 0.6}), r
    )
    s0, s1 = mixed["sigma_r2_components"]
    np.testing.assert_allclose(mixed["sigma_r2"], 0.4 * s0 + 0.6 * s1, rtol=1e-14)
    np.testing.assert_allclose(
        mixed["sigma_t2"], 2.0 * (0.4 * s0 + 0.6 * 1.2 * s1), rtol=1e-14
    )
    expected = 1.0 - (0.4 * s0 + 0.6 * 1.2 * s1) / (0.4 * s0 + 0.6 * s1)
    np.testing.assert_allclose(mixed["beta_sigma"], expected, rtol=1e-13, atol=1e-15)
    assert np.all((mixed["beta_sigma"] <= 0.0) & (mixed["beta_sigma"] >= -0.2))
