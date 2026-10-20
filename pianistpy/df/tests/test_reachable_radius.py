"""Tests of the smallest radius the exact derivatives' solve reaches.

r_of_p solves P(R) in host float64 (NumPy): below 2^-537 the square of R
rounds to zero and the slope dP/dR = -M/R^2 has no value, so a depth that
needs a smaller radius is refused instead of returning nan. Measured on
(alpha, beta, gamma) = (2, 4, 1.95): R is exact to :math:`10^{-14}` down to
2^-537.
"""

import numpy as np
import pytest

from pianistpy.df.derivatives import ExactDerivs, host_reachable_radius
from pianistpy.exceptions import InputError
from pianistpy.profiles import DoublePowerLaw


def test_the_host_floor_is_the_subnormal_one():
    r = host_reachable_radius()
    assert r == 2.0**-537
    assert r * r > 0.0 and (r / 2.0) * (r / 2.0) == 0.0


@pytest.fixture(scope="module")
def steep():
    prof = DoublePowerLaw(2.0, 4.0, 1.95)
    return prof, ExactDerivs(prof, beta_c=0.5, model="constant", max_order=1)


def test_a_depth_above_the_floor_is_solved(steep):
    prof, d = steep
    r_true = 4.0 * host_reachable_radius()
    u = float(prof.depth_inner(np.array([r_true]), terms=8)[0][0])
    r = float(np.asarray(d.r_of_p(np.array([1.0 - u])))[0])
    u_back = float(prof.depth_inner(np.array([r]), terms=8)[0][0])
    assert np.isfinite(r)
    assert u_back == pytest.approx(1.0 - (1.0 - u), rel=1e-12)


def test_a_depth_below_the_floor_is_refused(steep):
    prof, d = steep
    u = float(
        prof.depth_inner(np.array([0.5 * host_reachable_radius()]), terms=8)[0][0]
    )
    with pytest.raises(InputError, match="host float64"):
        d.r_of_p(np.array([0.5, 1.0 - u]))
