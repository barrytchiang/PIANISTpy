r"""Tests of the lowered center's refusal where its fixed point is not reached.

The inputs of `lowered_center` as `center_amplitude` makes them (the deficit's
Chebyshev series, the grid's ends), captured from `truncate` on
(alpha, beta) = (2, 4) with C = rho_norm. At :math:`\beta_{\rm c} = -1/2`,
:math:`\mathcal{K}_{\rm t} = 0.3`, from gamma 1.735 the iteration leaves the
map's domain (measured: neither Picard nor Newton on the residual reaches a
fixed point from delta_1; at 1.78 the map has no root at delta_1 itself); at
+1/2, :math:`\mathcal{K}_{\rm t} = 0.05`, gamma 1.70 it converges.
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
    r_lo, c, bc, coef, top = CASES[gamma]
    cheb = np.polynomial.chebyshev.Chebyshev(coef, domain=[0.0, top])
    return lowered_center(r_lo, 0.5, c, gamma, 2.0, 4.0, bc, cheb, None, 1.0)


@pytest.mark.parametrize("gamma", [1.735, 1.75])
def test_the_fixed_point_not_reached_is_refused_by_name(gamma):
    with pytest.raises(AccuracyError, match="not reached from delta_1") as err:
        _run(gamma)
    assert "response to a uniform change of delta" in str(err.value)
    assert "no root in (0, r]" in str(err.value)


def test_no_map_at_delta_1_is_refused_by_name():
    with pytest.raises(AccuracyError, match="not defined at delta_1"):
        _run(1.78)


def test_the_neighbor_at_plus_one_half_converges():
    out = _run(1.70)
    assert out["iters"] == 31
    assert out["err"] < 1e-15 and not out["floor_stopped"]
