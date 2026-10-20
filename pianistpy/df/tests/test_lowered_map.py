r"""Tests of the lowered center's map r -> r' where its linear start leaves
(0, r].

At :math:`\beta_{\rm c} = -1/2`, :math:`\mathcal{K}_{\rm t} = 0.3` on
(alpha, beta, gamma) = (2, 4, 1.68), the grid's outer nodes have a depth
shift du so large that the start r - du r^2 / M falls below zero (measured:
node 6711, r = 0.4774981, start -0.002), and Newton from it gave nan. The root
r' exists in (0, r]: the bracketed solve finds it.
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
    c = float(DoublePowerLaw(2.0, 4.0, GAMMA).rho_norm)
    return c, GAMMA, 2.0, 4.0


def test_the_linear_start_leaves_the_bracket_at_the_measured_node(cusp):
    c, g, a, b = cusp
    r = np.array([R_NODE])
    du = _series(r, c, g, a, b, 2) - U_HAT
    start = r - du * r**2 / _series(r, c, g, a, b, 3)
    assert float(du[0]) > 0.0
    assert float(start[0]) < 0.0


def test_the_bracketed_map_finds_the_root(cusp):
    c, g, a, b = cusp
    r = np.array([R_NODE, 0.5])
    uh = np.array([U_HAT, 0.40])
    x = _map_bracketed(r, uh, np.array([-0.002, -0.01]), c, g, a, b)
    assert np.all((x > 0.0) & (x < r))
    assert _series(x, c, g, a, b, 2) == pytest.approx(uh, rel=1e-14)


def test_no_root_in_the_bracket_is_refused(cusp):
    c, g, a, b = cusp
    with pytest.raises(AccuracyError, match="no root in"):
        _map_bracketed(
            np.array([R_NODE]), np.array([-1e-3]), np.array([-0.5]), c, g, a, b
        )
