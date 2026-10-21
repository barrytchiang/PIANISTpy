r"""Tests of the lowered center's map from a radius r to the radius r' at
which the untruncated depth equals a given depth, where the map's linear start
lies outside the interval (0, r].

The map's linear start is ``r - du r^2 / M``, with ``du`` the depth shift at
r and ``M`` the untruncated mass at r. For the double power law with
(alpha, beta, gamma) = (2, 4, 1.68), a model with :math:`\beta_{\rm c} = -1/2`
truncated at :math:`\mathcal{K}_{\rm t} = 0.3` has grid radii, such as
``R_NODE``, at which the depth shift is large enough that the linear start is
below zero, outside the interval (0, r] that holds r'. `_map_bracketed` finds
r' by Newton steps kept inside the bracket (0, r].
"""

import numpy as np
import pytest

from pianistpy.df.center import _map_bracketed, _series
from pianistpy.exceptions import AccuracyError
from pianistpy.profiles import DoublePowerLaw

GAMMA = 1.68
R_NODE = 4.774981e-01
U_HAT = 0.41532910194458217


@pytest.fixture(scope="module")
def cusp():
    """Return the density normalization, gamma, alpha and beta of the double
    power law with (alpha, beta, gamma) = (2, 4, ``GAMMA``)."""
    c = float(DoublePowerLaw(2.0, 4.0, GAMMA).rho_norm)
    return c, GAMMA, 2.0, 4.0


def test_the_linear_start_leaves_the_bracket_at_the_measured_node(cusp):
    """At the radius ``R_NODE`` and the depth ``U_HAT``, the depth shift is
    positive and the map's linear start ``r - du r^2 / M`` is below zero."""
    c, g, a, b = cusp
    r = np.array([R_NODE])
    du = _series(r, c, g, a, b, 2) - U_HAT
    start = r - du * r**2 / _series(r, c, g, a, b, 3)
    assert float(du[0]) > 0.0
    assert float(start[0]) < 0.0


def test_the_bracketed_map_finds_the_root(cusp):
    """`_map_bracketed` returns a radius r' strictly between 0 and r at which
    the untruncated depth series equals the given depth to a relative
    difference of at most :math:`10^{-14}`."""
    c, g, a, b = cusp
    r = np.array([R_NODE, 0.5])
    uh = np.array([U_HAT, 0.40])
    x = _map_bracketed(r, uh, np.array([-0.002, -0.01]), c, g, a, b)
    assert np.all((x > 0.0) & (x < r))
    assert _series(x, c, g, a, b, 2) == pytest.approx(uh, rel=1e-14)


def test_no_root_in_the_bracket_is_refused(cusp):
    """`_map_bracketed` raises AccuracyError when the given depth is negative,
    since no radius in (0, r] has a negative untruncated depth."""
    c, g, a, b = cusp
    with pytest.raises(AccuracyError, match="no root in"):
        _map_bracketed(
            np.array([R_NODE]), np.array([-1e-3]), np.array([-0.5]), c, g, a, b
        )
