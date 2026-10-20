"""Tests of the sampler on a superposed model."""

import numpy as np
import pytest

from pianistpy import Sampler
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition import superpose
from pianistpy.superposition.tests.test_core import _member


@pytest.fixture(scope="module")
def members():
    parent = _lowered_hernquist()
    return _member(parent, 0.0), _member(parent, -0.2, n_keep=1500)


def test_each_particle_is_its_components_draw(members):
    # the component by the weights from a channel of its own; the particle is
    # then its component's one-component draw, to the bit
    a, b = members
    n = 4000
    particles = Sampler(superpose({a: 0.3, b: 0.7})).sample(n, seed=11)
    alone = [Sampler(m).sample(n, seed=11) for m in (a, b)]
    labels = particles.component
    assert particles.component_names == ("component", "component")
    for k, draw in enumerate(alone):
        mine = labels == k
        assert np.array_equal(particles.position[mine], draw.position[mine])
        assert np.array_equal(particles.velocity[mine], draw.velocity[mine])
    # the fraction of each component, binomial within four standard deviations
    share = float(np.mean(labels == 1))
    assert abs(share - 0.7) < 4.0 * np.sqrt(0.7 * 0.3 / n)
    # equal masses summing to the shared model's mass
    assert np.all(particles.mass == particles.mass[0])
    np.testing.assert_allclose(particles.mass.sum(), alone[0].mass.sum(), rtol=1e-12)


def test_a_weight_of_one_is_the_one_component_draw(members):
    a, b = members
    particles = Sampler(superpose({a: 1.0, b: 1e-300})).sample(3000, seed=5)
    single = Sampler(a).sample(3000, seed=5)
    assert np.all(particles.component == 0)
    assert np.array_equal(particles.position, single.position)
    assert np.array_equal(particles.velocity, single.velocity)
    assert np.array_equal(particles.mass, single.mass)


def test_a_truncated_model_is_drawn_in_model_units():
    # the units come from the component's analytic profile (`unit_profile`),
    # not its truncated one, which has no normalization: converted at the output
    # only, so the positions are the internal ones and the velocities scaled
    parent = _lowered_hernquist()
    in_model = Sampler(parent.to("model", concentration=10.0)).sample(2000, seed=3)
    internal = Sampler(parent).sample(2000, seed=3)
    factors = parent.to("model", concentration=10.0).units.factors(
        parent.components[0].unit_profile
    )
    assert np.array_equal(in_model.position, internal.position)
    np.testing.assert_allclose(
        in_model.velocity, internal.velocity * factors.model_v, rtol=1e-15
    )


def test_a_speed_window_on_a_superposition_is_its_components_by_their_shares(members):
    # the window's fraction is the weighted sum of the components' own; each
    # speed is one of its component's windowed speeds
    a, b = members
    r = np.geomspace(0.05, 5.0, 300)
    window = (0.2, 0.6)
    v, fraction = Sampler(superpose({a: 0.3, b: 0.7})).speeds(r, seed=9, v_range=window)
    own = [Sampler(m).speeds(r, seed=9, v_range=window) for m in (a, b)]
    np.testing.assert_allclose(fraction, 0.3 * own[0][1] + 0.7 * own[1][1], rtol=1e-14)
    assert np.all((v == own[0][0]) | (v == own[1][0]))
