r"""Tests of the AccuracyError that `lowered_center` raises where its
fixed-point iteration for the central deficit leaves the domain of its map.

``CASES`` holds, for each inner slope gamma of the double power law with
(alpha, beta) = (2, 4), the inputs that `center_amplitude` passes to
`lowered_center` in a call of `truncate`: the inner end of the radial grid,
the density normalization ``rho_norm``, the central anisotropy
:math:`\beta_{\rm c}`, the coefficients of the Chebyshev series of the deficit
D(u) in the depth u, and the upper end of the depth interval of that series.
The cases with gamma 1.735, 1.75 and 1.78 have :math:`\beta_{\rm c} = -1/2`
and :math:`\mathcal{K}_{\rm t} = 0.3`. At gamma 1.735 and 1.75, the iteration
from the first-order deficit delta_1 leaves the map's domain before it reaches
a fixed point. At gamma 1.78, the map has no root at delta_1 itself. The case
with gamma 1.70 has :math:`\beta_{\rm c} = +1/2` and
:math:`\mathcal{K}_{\rm t} = 0.05`, and its iteration converges.
"""

import numpy as np
import pytest

from pianistpy.df.center import lowered_center
from pianistpy.exceptions import AccuracyError

CASES = {
    1.735: (
        4.460650630018799e-41,
        0.26499999999999996,
        -0.5,
        [
            35.73304532464659,
            12.152237826739151,
            34.98730567181179,
            3.328596012409604e-14,
            1.360746797525939e-14,
            1.2770085330628046e-14,
            -1.0048591735576167e-14,
            -1.0885974380207509e-14,
            1.3398122314101557e-14,
        ],
        1.6441654073584455,
    ),
    1.75: (
        2.4414062500000006e-43,
        0.25,
        -0.5,
        [
            51.75862896001016,
            23.819343321443746,
            50.05859139006454,
            3.684483636377926e-14,
            -2.2609331405046357e-14,
            4.354389752083002e-14,
            1.3816813636417219e-14,
            3.6844836363779235e-14,
            1.42355049587329e-14,
        ],
        1.7011807451983132,
    ),
    1.78: (
        8.66960339322637e-49,
        0.22000000000000003,
        -0.5,
        [
            122.38498823728621,
            83.44626130727411,
            114.24944357635809,
            3.349530578525387e-15,
            1.674765289262693e-14,
            1.3398122314101545e-13,
            1.3398122314101545e-13,
            2.009718347115232e-14,
            8.038873388460929e-14,
        ],
        1.825575609574704,
    ),
    1.70: (
        1.1129556050964727e-36,
        0.30000000000000004,
        0.5,
        [
            0.009128298574004856,
            -0.03306307413215894,
            -7.359808400080195e-18,
            3.2710259555911984e-18,
            0.0,
            -6.542051911182395e-18,
            4.088782444488998e-19,
            -1.6355129777955985e-17,
            1.6355129777955996e-18,
        ],
        1.5140024797399083,
    ),
}


def _run(gamma):
    """Return the output of `lowered_center` on the inputs ``CASES[gamma]``,
    with the outer end of the radial grid at 0.5 and ``kind=None``, which
    means a constant anisotropy with no anisotropy radius."""
    r_lo, c, bc, coef, top = CASES[gamma]
    cheb = np.polynomial.chebyshev.Chebyshev(coef, domain=[0.0, top])
    return lowered_center(r_lo, 0.5, c, gamma, 2.0, 4.0, bc, cheb, None, 1.0)


@pytest.mark.parametrize("gamma", [1.735, 1.75])
def test_the_fixed_point_not_reached_is_refused_with_cause(gamma):
    """At gamma 1.735 and 1.75, `lowered_center` raises AccuracyError that
    states the fixed point is not reached from delta_1, gives the update's
    response to a uniform change of delta, and states that the map has no root
    in (0, r]."""
    with pytest.raises(AccuracyError, match="not reached from delta_1") as err:
        _run(gamma)
    assert "response to a uniform change of delta" in str(err.value)
    assert "no root in (0, r]" in str(err.value)


def test_no_map_at_delta_1_is_refused_with_cause():
    """At gamma 1.78, `lowered_center` raises AccuracyError that states the
    map is not defined at the first-order deficit delta_1."""
    with pytest.raises(AccuracyError, match="not defined at delta_1"):
        _run(1.78)


def test_the_neighbor_at_plus_one_half_converges():
    r"""At gamma 1.70 with :math:`\beta_{\rm c} = +1/2`, the iteration of
    `lowered_center` converges in 31 iterations to a last change below
    :math:`10^{-15}`, without the stop at the rounding floor."""
    out = _run(1.70)
    assert out["iters"] == 31
    assert out["err"] < 1e-15 and not out["floor_stopped"]
