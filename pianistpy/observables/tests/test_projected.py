"""Tests of `ProjectedSampler`: pairs of projected radius and line-of-sight
velocity drawn on an annulus and in a window, against the Plummer sphere's
projected mass, the table's own exact cumulative at each pair's radius (the
reference without a table, A0), the distribution read by `losvd`, and on a
superposition its components' shares."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
    superpose,
)
from pianistpy.df import compute_df, truncate
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import LosTable, ProjectedSampler, losvd
from pianistpy.particles import stages as particle_stages
from pianistpy.superposition.tests.test_core import _member
from pianistpy.validation import GOAL

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
ANNULUS = (0.1, 3.0)
N_PAIRS = 20000


def _plummer_mass(r):
    # the projected mass inside R of the Plummer sphere of unit mass and scale
    return r**2 / (1.0 + r**2)


def _ks(sample, cdf):
    x = np.sort(sample)
    return float(np.max(np.abs(cdf(x) - np.arange(1, x.size + 1) / x.size)))


@pytest.fixture(scope="module")
def plummer():
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    table = LosTable(model)
    return table, ProjectedSampler(table, ANNULUS)


def test_the_annulus_mass_is_the_plummer_projected_mass(plummer):
    _, sampler = plummer
    exact = _plummer_mass(ANNULUS[1]) - _plummer_mass(ANNULUS[0])
    assert sampler.validation.tier == "goal"
    assert sampler.mass == pytest.approx(exact, rel=GOAL)
    pairs = sampler.sample(1000, seed=3)
    assert pairs.mass.sum() == pytest.approx(exact, rel=GOAL)


def test_a_pair_is_the_same_whatever_n(plummer):
    _, sampler = plummer
    big, small = sampler.sample(5000, seed=7), sampler.sample(300, seed=7)
    np.testing.assert_array_equal(small.r_2d, big.r_2d[:300])
    np.testing.assert_array_equal(small.v_los, big.v_los[:300])


def test_the_whole_window_is_the_unwindowed_draw_bit_for_bit(plummer):
    table, sampler = plummer
    whole = ProjectedSampler(table, ANNULUS, v_range=(-10.0, 10.0))
    a, b = sampler.sample(2000, seed=5), whole.sample(2000, seed=5)
    np.testing.assert_array_equal(a.r_2d, b.r_2d)
    np.testing.assert_array_equal(a.v_los, b.v_los)


def test_each_pair_inverts_the_exact_cumulative_at_its_own_radius(plummer):
    # A0: the table's own row read at the pair's radius, its cumulative exact
    # per sub-panel; the pair's speed is its number's inverse there
    _, sampler = plummer
    n = 200
    pairs = sampler.sample(n, seed=11)
    numbers = np.asarray(particle_stages.random_numbers_block(11, 0, 0, n).speed)
    part = sampler.parts[0]
    (stretch,) = part.stretches
    rows = part.rows(pairs.r_2d, stretch.st)
    got = np.array(
        [
            row.cdf(np.array([abs(v)]))[0]
            for row, v in zip(rows, pairs.v_los, strict=True)
        ]
    )
    np.testing.assert_allclose(
        got, np.abs(2.0 * numbers - 1.0), rtol=0.0, atol=10.0 * GOAL
    )
    np.testing.assert_array_equal(np.sign(pairs.v_los), np.sign(2.0 * numbers - 1.0))


def test_the_radii_follow_the_projected_mass(plummer):
    _, sampler = plummer
    pairs = sampler.sample(N_PAIRS, seed=1)
    lo, total = _plummer_mass(ANNULUS[0]), sampler.mass
    ks = _ks(pairs.r_2d, lambda r: (_plummer_mass(r) - lo) / total)
    assert ks < 1.63 / math.sqrt(N_PAIRS)


def test_a_window_conditions_the_pairs(plummer):
    table, sampler = plummer
    window = (-0.2, 0.3)
    inside = ProjectedSampler(table, ANNULUS, v_range=window)
    assert inside.validation.tier == "goal"
    assert 0.0 < inside.mass < sampler.mass
    pairs = inside.sample(N_PAIRS, seed=2)
    assert np.all((pairs.v_los >= window[0]) & (pairs.v_los <= window[1]))
    # the velocities at a narrow ring against the distribution in the window
    ring = (pairs.r_2d > 0.95) & (pairs.r_2d < 1.05)
    grid = np.linspace(window[0], window[1], 2001)
    pdf = np.asarray(losvd(table, np.ones(grid.size), grid))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    ks = _ks(pairs.v_los[ring], lambda v: np.interp(v, grid, cdf / cdf[-1]))
    assert ks < 1.63 / math.sqrt(int(ring.sum()))


def test_refusals(plummer):
    table, _ = plummer
    with pytest.raises(InputError, match="holds none"):
        ProjectedSampler(table, ANNULUS, v_range=(5.0, 6.0))
    with pytest.raises(InputError, match="last row"):
        ProjectedSampler(table, (1e9, 2e9))
    with pytest.raises(InputError, match="R_range"):
        ProjectedSampler(table, (2.0, 1.0))
    with pytest.raises(InputError, match="v_range"):
        ProjectedSampler(table, ANNULUS, v_range=(2.0, 1.0))


def test_a_sampler_from_the_model_is_the_samplers_from_its_table(plummer):
    table, sampler = plummer
    a = ProjectedSampler(table.model, ANNULUS).sample(500, seed=4)
    b = sampler.sample(500, seed=4)
    np.testing.assert_array_equal(a.r_2d, b.r_2d)
    np.testing.assert_array_equal(a.v_los, b.v_los)


@pytest.fixture(scope="module")
def superposed():
    """The certified quick-tier superposition of test_superposed: the truncated
    Hernquist model and the same on every other energy of its DF table."""
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))
    parent = dataclasses.replace(
        _member(truncate(df, Truncation(k_t=0.05)), 0.0), settings=Settings(quick=True)
    )
    t = parent.table
    n = t.k.shape[1]
    keep = np.unique(np.r_[np.arange(0, n, 2), n - 1])
    thinned = dataclasses.replace(
        parent,
        table=dataclasses.replace(
            t,
            k=t.k[:, keep],
            log_df=t.log_df[:, keep],
            log_df_slope=t.log_df_slope[:, keep],
            i_last=(keep.size - 1,),
        ),
    )
    model = superpose({parent: 0.3, thinned: 0.7})
    return LosTable(model)


def test_a_superposition_draws_its_components_by_their_shares(superposed):
    sampler = ProjectedSampler(superposed, (0.05, 2.0), v_range=(-0.1, 0.2))
    masses = np.array([p.mass for p in sampler.parts])
    np.testing.assert_allclose(
        sampler.shares, np.array([0.3, 0.7]) * masses / (np.array([0.3, 0.7]) @ masses)
    )
    pairs = sampler.sample(N_PAIRS, seed=4)
    frac = np.mean(pairs.component == 1)
    assert abs(frac - sampler.shares[1]) < 4.0 * math.sqrt(0.25 / N_PAIRS)
    assert np.all((pairs.v_los >= -0.1) & (pairs.v_los <= 0.2))
