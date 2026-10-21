r"""Tests of the AccuracyError that `lowered_center` raises where its radial
grid reaches radii at which ``r^(3 - 2 beta_c)`` underflows to zero in
float64, and of `representable_end`, the radius below which that underflow
occurs.

For the double power law with (alpha, beta, gamma) = (1, 4, 1.9) and
:math:`\beta_{\rm c} = -1/2`, the exponent ``3 - 2 beta_c`` is 4. On a grid
that starts at :math:`r = 10^{-103}`, ``r^4`` is zero in float64, and the
integrand ``dm / r^4`` of the Poisson depth is 0/0 at the first iteration.
`lowered_center` raises AccuracyError for that integrand before it builds its
spline. The tests pass a constant deficit, since the underflow of ``r^4``
does not depend on the deficit. `center_amplitude` starts the grid no lower
than `representable_end`, so only a grid that a caller starts below
`representable_end` reaches the AccuracyError, as in these tests.
"""

import numpy as np
import pytest

from pianistpy.df.center import lowered_center, representable_end
from pianistpy.exceptions import AccuracyError

# A constant deficit D(u) = 1 on the depth interval [0, 1].
DEFICIT = np.polynomial.chebyshev.Chebyshev([1.0], domain=[0.0, 1.0])


def test_the_underflowing_grid_is_refused_with_cause():
    """A grid that starts at :math:`r = 10^{-103}` makes `lowered_center` raise
    AccuracyError that states ``dm / r^4`` is not finite at that radius."""
    with pytest.raises(AccuracyError, match="not defined at delta_1") as err:
        lowered_center(1e-103, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert "dm / r^4 is not finite at r = 1.000000e-103" in str(err.value)


def test_the_same_map_above_the_underflow_converges():
    """With the same inputs and a grid that starts at :math:`r = 10^{-60}`,
    where ``r^4`` is a normal double, the iteration of `lowered_center`
    converges without the stop at the rounding floor."""
    out = lowered_center(1e-60, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert out["err"] < 1e-15 and not out["floor_stopped"]


@pytest.mark.parametrize(
    ("w3", "delta_1"), [(4.0, 1.0), (4.0, 1e-30), (3.0, 1.0), (2.0, 1e3)]
)
def test_the_representable_end_holds_normal_doubles(w3, delta_1):
    """At the radius r that `representable_end` returns, ``r^w3`` and the
    deficit's mass ``|delta_1| r^w3 / w3`` are both at least the smallest
    normal double, up to a relative tolerance of :math:`10^{-12}`. At the
    radius ``0.99 r``, at least one of the two lies below the smallest normal
    double."""
    r = representable_end(w3, delta_1)
    tiny = float(np.finfo(np.float64).tiny)
    assert r**w3 >= tiny * (1.0 - 1e-12)
    assert abs(delta_1) * r**w3 / w3 >= tiny * (1.0 - 1e-12)
    assert (0.99 * r) ** w3 < tiny * max(1.0, w3 / abs(delta_1))


def test_the_map_from_the_representable_end_converges():
    """A grid that starts at the larger of :math:`10^{-103}` and
    `representable_end` lets the iteration of `lowered_center` converge
    without the stop at the rounding floor."""
    lo = max(1e-103, representable_end(4.0, 1.0))
    out = lowered_center(lo, 0.5, 1.0, 1.9, 1.0, 4.0, -0.5, DEFICIT, None, 1.0)
    assert out["err"] < 1e-15 and not out["floor_stopped"]
