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
    # Each particle's component is drawn by the weights from a random stream
    # of its own, and each particle then equals, bit for bit, the particle of
    # the one-component draw of that component.
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
    # The fraction of particles in component 1 follows a binomial
    # distribution, so the test bounds the fraction's distance from the weight
    # 0.7 by four binomial standard deviations.
    share = float(np.mean(labels == 1))
    assert abs(share - 0.7) < 4.0 * np.sqrt(0.7 * 0.3 / n)
    # Every particle has the same mass, and the particle masses sum to the
    # mass of the shared model.
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
    # The unit factors come from the component's analytic profile
    # (`unit_profile`) and not from the truncated profile, which has no density
    # normalization. The conversion applies only at the output, so the model
    # units leave the positions equal to the internal positions and scale the
    # velocities.
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


def test_a_velocity_window_on_a_superposition_is_its_components_by_their_shares(
    members,
):
    # The fraction of the velocity distribution inside the user-specified range
    # is the weighted sum of the components' fractions, and each drawn velocity
    # equals a velocity that the component's own draw in the same range gives.
    a, b = members
    r = np.geomspace(0.05, 5.0, 300)
    window = (0.2, 0.6)
    v, fraction = Sampler(superpose({a: 0.3, b: 0.7})).velocities(
        r, seed=9, v_range=window
    )
    own = [Sampler(m).velocities(r, seed=9, v_range=window) for m in (a, b)]
    np.testing.assert_allclose(fraction, 0.3 * own[0][1] + 0.7 * own[1][1], rtol=1e-14)
    assert np.all((v == own[0][0]) | (v == own[1][0]))
