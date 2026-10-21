"""Tests of `table_checks`, `slope_column_excess` and `validity_certificate`.

`table_checks` reads the checks of a DF table: the exponent of the DF at the
center and the error that the stated slope column makes. `validity_certificate`
states, from closed forms, whether float64 holds the DF and the radius at the
table's top.
"""

import math

import numpy as np
import pytest

from pianistpy.df.analytic import hernquist_df
from pianistpy.df.checks import slope_column_excess, table_checks, validity_certificate
from pianistpy.exceptions import ResultError


def test_the_certificate_of_a_hernquist_table():
    # For inner slope gamma = 1, DF exponent kappa = -5/2 at the center and the
    # table's top at a depth of 1e-9 below the central potential, the projected
    # log10 F at the table's top is 22.5 and the projected log10 R there is -9.
    # Float64 represents both the DF and the radius at the table's top.
    cert = validity_certificate(1.0, -2.5, 1e-9)
    assert cert["valid_gamma"] and cert["linearF_representable"]
    assert cert["proj_log10F_at_ceiling"] == pytest.approx(22.5)
    assert cert["inner_R_exponent"] == 1.0
    assert cert["proj_log10R_at_ceiling"] == pytest.approx(-9.0)
    assert cert["R_representable"]


def test_no_exponent_without_a_valid_inner_slope():
    cert = validity_certificate(2.5, -2.5, 1e-9)
    assert not cert["valid_gamma"]
    assert math.isnan(cert["kappa_top"]) and not cert["linearF_representable"]


def _hernquist_table(n=400):
    # The nodes are uniform in x = ln(k / (1 - k)), from k = 1e-6 to the depth
    # 1 - k = 1e-6 below the central potential 1. The table holds k, log10 of the
    # closed-form Hernquist DF and the DF's derivative in k.
    x = np.linspace(np.log(1e-6), np.log((1.0 - 1e-6) / 1e-6), n)
    k = 1.0 / (1.0 + np.exp(-x))
    f, ddf = hernquist_df(k)
    return k, np.log10(f), ddf


def test_the_exact_slope_column_is_within_1e_6():
    k, log_df, ddf = _hernquist_table()
    slope = k * ddf * 10.0 ** (-log_df) / np.log(10.0)
    assert np.max(slope_column_excess(np.log(k), log_df, slope)) < 1e-6


def test_a_wrong_slope_column_is_refused():
    k, log_df, ddf = _hernquist_table()
    top = {"top_kappa": -2.5}
    with pytest.raises(ResultError):
        table_checks(k, log_df, -ddf, 1.0, top)


def test_the_stated_exponent_of_the_hernquist_center():
    # The Hernquist DF goes as u^(-5/2) in the depth u = 1 - k at the center, so
    # the slope of the table's top decade must be within 1e-4 of the stated
    # exponent -5/2.
    k, log_df, ddf = _hernquist_table()
    checks = table_checks(k, log_df, ddf, 1.0, {"top_kappa": -2.5})
    assert checks.kappa == -2.5
    assert checks.kappa_residual < 1e-4
    assert checks.k_top == k[-1]
    assert math.isnan(checks.edge_rel_at_floor)
