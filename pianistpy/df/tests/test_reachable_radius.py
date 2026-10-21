r"""Tests of the smallest radius that `ExactDerivs.r_of_p` reaches.

`ExactDerivs.r_of_p` solves for the radius R at which the potential P(R)
takes a given value, in float64 on the host with NumPy. For R below 2^-537,
the square of R rounds to zero in float64, and the slope
:math:`\mathrm{d}P/\mathrm{d}R = -M/R^2` has no finite value. A depth
u = 1 - P that needs a radius below 2^-537 therefore makes
`ExactDerivs.r_of_p` raise InputError instead of returning NaN.
"""

import numpy as np
import pytest

from pianistpy.df.derivatives import ExactDerivs, host_reachable_radius
from pianistpy.exceptions import InputError
from pianistpy.profiles import DoublePowerLaw


def test_the_host_floor_is_the_subnormal_one():
    """`host_reachable_radius` returns 2^-537: the square of 2^-537 is a
    positive subnormal double, and the square of 2^-538 is zero."""
    r = host_reachable_radius()
    assert r == 2.0**-537
    assert r * r > 0.0 and (r / 2.0) * (r / 2.0) == 0.0


@pytest.fixture(scope="module")
def steep():
    r"""Return the double power law with (alpha, beta, gamma) = (2, 4, 1.95)
    and its exact derivatives for a constant anisotropy
    :math:`\beta_{\rm c} = 1/2`, up to the first order."""
    prof = DoublePowerLaw(2.0, 4.0, 1.95)
    return prof, ExactDerivs(prof, beta_c=0.5, model="constant", max_order=1)


def test_a_depth_above_the_floor_is_solved(steep):
    """At the depth of the radius four times `host_reachable_radius`,
    `ExactDerivs.r_of_p` returns a finite radius whose depth equals the given
    depth to a relative difference of at most :math:`10^{-12}`."""
    prof, d = steep
    r_true = 4.0 * host_reachable_radius()
    u = float(prof.depth_inner(np.array([r_true]), terms=8)[0][0])
    r = float(np.asarray(d.r_of_p(np.array([1.0 - u])))[0])
    u_back = float(prof.depth_inner(np.array([r]), terms=8)[0][0])
    assert np.isfinite(r)
    assert u_back == pytest.approx(1.0 - (1.0 - u), rel=1e-12)


def test_a_depth_below_the_floor_is_refused(steep):
    """At the depth of the radius half of `host_reachable_radius`,
    `ExactDerivs.r_of_p` raises InputError whose message states host float64."""
    prof, d = steep
    u = float(
        prof.depth_inner(np.array([0.5 * host_reachable_radius()]), terms=8)[0][0]
    )
    with pytest.raises(InputError, match="host float64"):
        d.r_of_p(np.array([0.5, 1.0 - u]))
