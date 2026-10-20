r"""Tests of the lowered center's refusal where its grid reaches radii at which
``r^(3 - 2 beta_c)`` underflows.

At :math:`\beta_{\rm c} = -1/2`, :math:`\mathcal{K}_{\rm t} = 0.3` on
(alpha, beta, gamma) = (1, 4, 1.9) `center_amplitude` starts the grid at
:math:`r = 10^{-103}` (measured), where r^4 is zero in float64: the
Poisson depth's integrand dm / r^4 is 0/0 at the first iteration. The refusal
is named before the spline is built; the inputs here are that grid's ends with
a constant deficit (the underflow does not depend on it). `center_amplitude`
now starts the grid no lower than `representable_end`, so the refusal is
reached only by a grid asked to start below it, as here.
"""

import numpy as np
import pytest

from pianistpy.df.center import lowered_center, representable_end
from pianistpy.exceptions import AccuracyError

DEFICIT = np.polynomial.chebyshev.Chebyshev([1.0], domain=[0.0, 1.0])


def test_the_underflowing_grid_is_refused_by_name():
    with pytest.raises(AccuracyError, match="not defined at delta_1") as err:
        lowered_center(1e-103, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert "dm / r^4 is not finite at r = 1.000000e-103" in str(err.value)


def test_the_same_map_above_the_underflow_converges():
    out = lowered_center(1e-60, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert out["err"] < 1e-15 and not out["floor_stopped"]


@pytest.mark.parametrize(
    ("w3", "delta_1"), [(4.0, 1.0), (4.0, 1e-30), (3.0, 1.0), (2.0, 1e3)]
)
def test_the_representable_end_holds_normal_doubles(w3, delta_1):
    r = representable_end(w3, delta_1)
    tiny = float(np.finfo(np.float64).tiny)
    assert r**w3 >= tiny * (1.0 - 1e-12)
    assert abs(delta_1) * r**w3 / w3 >= tiny * (1.0 - 1e-12)
    assert (0.99 * r) ** w3 < tiny * max(1.0, w3 / abs(delta_1))


def test_the_map_from_the_representable_end_converges():
    lo = max(1e-103, representable_end(4.0, 1.0))
    out = lowered_center(lo, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert out["err"] < 1e-15 and not out["floor_stopped"]
