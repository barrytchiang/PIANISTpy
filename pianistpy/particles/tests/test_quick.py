"""Tests of the sampler with the quick settings against the sampler with the
default settings."""

import dataclasses

import numpy as np

from pianistpy import Sampler, Settings
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.validation import TIER_BOUNDS


def test_the_quick_settings_hold_1e_4_against_the_default():
    # The quick settings set row_error_goal to 10^-4, halve the Gauss-Legendre
    # nodes and quarter the velocity table's nodes. On one truncated model, the
    # quick sampler's positions equal the default sampler's positions, because
    # the quick values of row_error_goal, n_gauss_legendre and n_nodes leave
    # the radius table unchanged; the quick
    # velocities lie within 10^-4 of the default velocities; and the quick sampler's
    # tier is "silent" or "recorded".
    model = _lowered_hernquist()
    default = Sampler(model)
    quick = Sampler(dataclasses.replace(model, settings=Settings(quick=True)))
    assert quick.validation.level <= 1
    a, b = default.sample(5000, seed=3), quick.sample(5000, seed=3)
    assert np.array_equal(a.position, b.position)
    velocity_a = np.linalg.norm(a.velocity, axis=1)
    velocity_b = np.linalg.norm(b.velocity, axis=1)
    assert np.max(np.abs(velocity_b / velocity_a - 1.0)) < TIER_BOUNDS[1]
