"""Tests of `ProjectedSampler`.

`ProjectedSampler` draws pairs of projected radius and line-of-sight velocity
on an annulus, optionally in a user-specified range of line-of-sight velocity.
The tests compare the draws of the Plummer sphere with the Plummer sphere's
closed-form projected mass, with the exact cumulative distribution of the
line-of-sight velocity at each pair's radius, and with the distribution of
`losvd`. On a superposed model, the tests compare the fraction of pairs drawn
from each component with the component's share.
"""

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
from pianistpy.validation import TIER_BOUNDS

pytestmark = pytest.mark.slow

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
ANNULUS = (0.1, 3.0)
N_PAIRS = 20000


def _plummer_mass(r):
    """Return the projected mass inside projected radius `r` of the Plummer sphere.

    The Plummer sphere has unit mass and unit scale radius, so the projected
    mass is ``r**2 / (1 + r**2)``.
    """
    return r**2 / (1.0 + r**2)


def _ks(sample, cdf):
    """Return the Kolmogorov-Smirnov distance of `sample` from the CDF `cdf`."""
    x = np.sort(sample)
    return float(np.max(np.abs(cdf(x) - np.arange(1, x.size + 1) / x.size)))


@pytest.fixture(scope="module")
def plummer():
    """Return the `LosTable` of the isotropic Plummer model with SETTINGS, and
    the `ProjectedSampler` of the table on ANNULUS, without a velocity
    range."""
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    table = LosTable(model)
    return table, ProjectedSampler(table, ANNULUS)


def test_the_annulus_mass_is_the_plummer_projected_mass(plummer):
    """Assert that the annulus mass of the sampler is the Plummer projected mass.

    On the annulus from projected radius 0.1 to 3, the test asserts that the
    sampler's validation tier is "silent". The test asserts that the
    sampler's mass and the summed mass of 1000 drawn pairs each equal the
    closed-form projected mass of the annulus to a relative difference of
    :math:`10^{-6}`.
    """
    _, sampler = plummer
    exact = _plummer_mass(ANNULUS[1]) - _plummer_mass(ANNULUS[0])
    assert sampler.validation.tier == "silent"
    assert sampler.mass == pytest.approx(exact, rel=TIER_BOUNDS[0])
    pairs = sampler.sample(1000, seed=3)
    assert pairs.mass.sum() == pytest.approx(exact, rel=TIER_BOUNDS[0])


def test_a_pair_is_the_same_whatever_n(plummer):
    """Assert that the first 300 pairs of a draw do not depend on the draw size.

    With one seed, the test asserts that the 300 pairs of a draw of 300 equal,
    bit for bit, the first 300 pairs of a draw of 5000.
    """
    _, sampler = plummer
    big, small = sampler.sample(5000, seed=7), sampler.sample(300, seed=7)
    np.testing.assert_array_equal(small.r_2d, big.r_2d[:300])
    np.testing.assert_array_equal(small.v_los, big.v_los[:300])


def test_the_whole_window_is_the_unwindowed_draw_bit_for_bit(plummer):
    """Assert that a velocity range wider than the distribution changes no pair.

    With one seed, the test asserts that 2000 pairs drawn with the
    user-specified range from -10 to 10 equal, bit for bit, 2000 pairs drawn
    without a range.
    """
    table, sampler = plummer
    whole = ProjectedSampler(table, ANNULUS, v_range=(-10.0, 10.0))
    a, b = sampler.sample(2000, seed=5), whole.sample(2000, seed=5)
    np.testing.assert_array_equal(a.r_2d, b.r_2d)
    np.testing.assert_array_equal(a.v_los, b.v_los)


def test_each_pair_inverts_the_exact_cumulative_at_its_own_radius(plummer):
    """Assert that each pair's velocity inverts the exact cumulative distribution.

    For each of 200 pairs, the test reads the sampler's row at the pair's
    projected radius and evaluates the row's exact cumulative distribution of
    the line-of-sight velocity. The test asserts that the cumulative
    distribution at the pair's velocity equals ``|2u - 1|`` to an
    absolute difference of :math:`10^{-5}`, where u is the pair's uniform
    random number. The test asserts that the sign of the pair's velocity is
    the sign of ``2u - 1``.
    """
    _, sampler = plummer
    n = 200
    pairs = sampler.sample(n, seed=11)
    numbers = np.asarray(
        particle_stages.random_numbers_block(11, 0, 0, n).velocity_magnitude
    )
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
        got, np.abs(2.0 * numbers - 1.0), rtol=0.0, atol=10.0 * TIER_BOUNDS[0]
    )
    np.testing.assert_array_equal(np.sign(pairs.v_los), np.sign(2.0 * numbers - 1.0))


def test_the_radii_follow_the_projected_mass(plummer):
    """Assert that the drawn radii follow the Plummer projected mass.

    The test asserts that the Kolmogorov-Smirnov distance of 20000 drawn
    radii from the closed-form cumulative projected mass on the annulus is
    below ``1.63 / sqrt(20000)``, the critical distance at the 1% level.
    """
    _, sampler = plummer
    pairs = sampler.sample(N_PAIRS, seed=1)
    lo, total = _plummer_mass(ANNULUS[0]), sampler.mass
    ks = _ks(pairs.r_2d, lambda r: (_plummer_mass(r) - lo) / total)
    assert ks < 1.63 / math.sqrt(N_PAIRS)


def test_a_window_conditions_the_pairs(plummer):
    """Assert that a user-specified velocity range conditions the drawn pairs.

    With the range from -0.2 to 0.3, the test asserts that the sampler's
    validation tier is "silent", that the sampler's mass is positive and
    below the mass without a range, and that every drawn velocity lies in the
    range. For the pairs with projected radius between 0.95 and 1.05, the
    test asserts that the Kolmogorov-Smirnov distance of the velocities from
    the cumulative of `losvd` at projected radius 1, restricted to the range,
    is below the critical distance at the 1% level.
    """
    table, sampler = plummer
    window = (-0.2, 0.3)
    inside = ProjectedSampler(table, ANNULUS, v_range=window)
    assert inside.validation.tier == "silent"
    assert 0.0 < inside.mass < sampler.mass
    pairs = inside.sample(N_PAIRS, seed=2)
    assert np.all((pairs.v_los >= window[0]) & (pairs.v_los <= window[1]))
    # The reference CDF for the pairs in the ring 0.95 < R < 1.05 is the
    # trapezoidal integral of losvd at R = 1 over the user-specified range.
    ring = (pairs.r_2d > 0.95) & (pairs.r_2d < 1.05)
    grid = np.linspace(window[0], window[1], 2001)
    pdf = np.asarray(losvd(table, np.ones(grid.size), grid))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    ks = _ks(pairs.v_los[ring], lambda v: np.interp(v, grid, cdf / cdf[-1]))
    assert ks < 1.63 / math.sqrt(int(ring.sum()))


def test_refusals(plummer):
    r"""Assert InputError for four invalid arguments of `ProjectedSampler`.

    The test asserts InputError for a velocity range from 5 to 6 that holds
    none of the distribution, for an annulus from :math:`10^{9}` to
    :math:`2 \times 10^{9}` beyond the table's outermost projected radius,
    for an annulus whose lower end exceeds its upper end, and for a velocity
    range whose lower end exceeds its upper end.
    """
    table, _ = plummer
    with pytest.raises(InputError, match="holds none"):
        ProjectedSampler(table, ANNULUS, v_range=(5.0, 6.0))
    with pytest.raises(InputError, match="outermost tabulated radius"):
        ProjectedSampler(table, (1e9, 2e9))
    with pytest.raises(InputError, match="R_range"):
        ProjectedSampler(table, (2.0, 1.0))
    with pytest.raises(InputError, match="v_range"):
        ProjectedSampler(table, ANNULUS, v_range=(2.0, 1.0))


def test_a_slope_the_density_does_not_give_is_refused_not_filled():
    """Assert AccuracyError where the density gives the inverse no slope.

    The test builds a row with zero density and zero total mass, so the
    normalized density at the nodes of the inverse cumulative is 0 / 0. The
    slope of the inverse cumulative is then not finite. The test asserts that
    `_Row.table` raises AccuracyError, so the slope is not filled with a
    substitute value.
    """
    from pianistpy.exceptions import AccuracyError
    from pianistpy.observables._ellipse_rules import gl01
    from pianistpy.observables.projected import _Row

    row = object.__new__(_Row)
    xg, wg = gl01(2)
    row.xg, row.wg = np.asarray(xg), np.asarray(wg)
    row.bw = np.array([1.0 / (xg[0] - xg[1]), 1.0 / (xg[1] - xg[0])])
    row.width = np.array([1.0])
    row.lo, row.hi = np.array([0.0]), np.array([1.0])
    row.d = np.zeros((1, 2))
    row.cum, row.total, row.v_end = np.zeros(2), 0.0, 1.0
    row._tables = {}
    with pytest.raises(AccuracyError, match="no positive finite slope"):
        row.table(np.linspace(-1.0, 1.0, 3))


def test_a_sampler_from_the_model_is_the_samplers_from_its_table(plummer):
    """Assert that a sampler built from a model equals one built from its table.

    With one seed, the test asserts that 500 pairs drawn by a sampler built
    from the model equal, bit for bit, 500 pairs drawn by the sampler built
    from the model's `LosTable`.
    """
    table, sampler = plummer
    a = ProjectedSampler(table.model, ANNULUS).sample(500, seed=4)
    b = sampler.sample(500, seed=4)
    np.testing.assert_array_equal(a.r_2d, b.r_2d)
    np.testing.assert_array_equal(a.v_los, b.v_los)


@pytest.fixture(scope="module")
def superposed():
    """Return the `LosTable` of a superposed model of two components.

    The first component is the Hernquist model truncated by `truncate` with
    ``Truncation(k_t=0.05)``, with the quick settings. The second component is
    the first component with its DF table kept only at every other energy and
    at the last energy. The weights are 0.3 for the first component and 0.7
    for the second component. The model equals the fixture ``tabled`` of
    test_superposed.py.
    """
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
    """Assert that a superposed sampler draws each component by its share.

    On the annulus from 0.05 to 2 and the velocity range from -0.1 to 0.2,
    the test asserts that each component's share is the component's weight
    times the component's mass, normalized over the components. Of 20000
    pairs, the test asserts that the fraction drawn from the second
    component differs from the second component's share by less than
    ``4 sqrt(0.25 / 20000)``, 4 times the largest binomial standard deviation
    of the fraction. The test asserts that every drawn velocity lies in the
    velocity range.
    """
    sampler = ProjectedSampler(superposed, (0.05, 2.0), v_range=(-0.1, 0.2))
    masses = np.array([p.mass for p in sampler.parts])
    np.testing.assert_allclose(
        sampler.shares, np.array([0.3, 0.7]) * masses / (np.array([0.3, 0.7]) @ masses)
    )
    pairs = sampler.sample(N_PAIRS, seed=4)
    frac = np.mean(pairs.component == 1)
    assert abs(frac - sampler.shares[1]) < 4.0 * math.sqrt(0.25 / N_PAIRS)
    assert np.all((pairs.v_los >= -0.1) & (pairs.v_los <= 0.2))
