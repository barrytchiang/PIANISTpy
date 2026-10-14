"""The quick tier of the sampling against the default tier."""

import dataclasses

import numpy as np

from pianistpy import Sampler, Settings
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.validation import CEILING


def test_the_quick_tier_holds_the_ceiling_against_the_default():
    # the quick values (row_error_goal at the ceiling, half the Gauss-Legendre
    # nodes, a quarter of the speed table's nodes) on one truncated model: the
    # same radii (the radius table is the default's), the speeds within the
    # ceiling of the default's, and the quick report within the ceiling
    model = _lowered_hernquist()
    default = Sampler(model)
    quick = Sampler(dataclasses.replace(model, settings=Settings(quick=True)))
    assert quick.validation.level <= 1
    a, b = default.sample(5000, seed=3), quick.sample(5000, seed=3)
    assert np.array_equal(a.position, b.position)
    speed_a = np.linalg.norm(a.velocity, axis=1)
    speed_b = np.linalg.norm(b.velocity, axis=1)
    assert np.max(np.abs(speed_b / speed_a - 1.0)) < CEILING
