"""The edge row of a spatially finite product that holds none."""

import math

import numpy as np
import pytest

from pianistpy import ResultError
from pianistpy.df.assembly import edge_row_of
from pianistpy.df.truncation import ORIGINS

M_T = 0.3  # the floor row's mass over 4 pi


def _rows(psi_k=0.05):
    return dict(
        R=np.array([0.1, 1.0, 1.19]),
        Psi=np.array([0.9, psi_k, 1e-10]),
        rho=np.array([5.0, 0.1, 1e-30]),
        M=np.array([0.01, 0.5, 4.0 * math.pi * M_T]),
        u=np.array([1e-3, 0.851, 0.901]),
        origin=np.array([0, 0, 2], dtype=np.int8),
    )


def test_the_floor_row_is_replaced_by_the_edge_row_at_the_kepler_zero():
    out = edge_row_of(_rows())
    # Kepler through the last row above the floor, M_t (1/r - 1/r_e) = Psi_k
    r_e = 1.0 / (1.0 / 1.0 - 0.05 / M_T)
    assert out["R"][-1] == r_e and float(out["r_t"]) == r_e
    np.testing.assert_array_equal(out["R"][:-1], [0.1, 1.0])
    np.testing.assert_array_equal(out["Psi"], [0.9, 0.05, 0.0])
    np.testing.assert_array_equal(out["rho"], [5.0, 0.1, 0.0])
    np.testing.assert_array_equal(out["M"], [0.01, 0.5, 4.0 * math.pi * M_T])
    assert M_T * (1.0 / 1.0 - 1.0 / r_e) == pytest.approx(0.05, rel=1e-15)
    # the central potential less the potential, the potential zero at the edge
    assert out["u"][-1] == 0.9 + 1e-3 and out["u"][1] == 0.851
    assert out["origin"][-1] == ORIGINS["edge"] and out["origin"].dtype == np.int8


def test_rows_ending_on_an_edge_row_are_kept():
    rows = _rows()
    rows["Psi"][-1], rows["rho"][-1] = 0.0, 0.0
    out = edge_row_of(rows)
    for name, column in rows.items():
        np.testing.assert_array_equal(out[name], column)


def test_a_kepler_form_without_a_zero_or_an_unknown_column_is_refused():
    with pytest.raises(ResultError, match="does not reach zero"):
        edge_row_of(_rows(psi_k=0.5))
    rows = _rows()
    rows["w"] = np.ones(3)
    with pytest.raises(ResultError, match="no value for the column 'w'"):
        edge_row_of(rows)
