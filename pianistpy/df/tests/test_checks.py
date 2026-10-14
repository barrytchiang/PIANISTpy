"""Tests of the checks a DF table states about itself and of the certificate."""

import math

import numpy as np
import pytest

from pianistpy.df.analytic import hernquist_df
from pianistpy.df.checks import slope_column_excess, table_checks, validity_certificate
from pianistpy.exceptions import ResultError


def test_the_certificate_of_a_hernquist_table():
    # gamma = 1, kappa = -5/2 at the center, the table's top 1e-9 below it:
    # F spans 22.5 decades and R 9, both representable.
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
    # nodes uniform in ln(k / u) from 1e-6 to u = 1e-6, the closed form and its slope
    x = np.linspace(np.log(1e-6), np.log((1.0 - 1e-6) / 1e-6), n)
    k = 1.0 / (1.0 + np.exp(-x))
    f, ddf = hernquist_df(k)
    return k, np.log10(f), ddf


def test_the_exact_slope_column_is_within_the_goal():
    k, log_df, ddf = _hernquist_table()
    slope = k * ddf * 10.0 ** (-log_df) / np.log(10.0)
    assert np.max(slope_column_excess(np.log(k), log_df, slope)) < 1e-6


def test_a_wrong_slope_column_is_refused():
    k, log_df, ddf = _hernquist_table()
    top = {"top_kappa": -2.5}
    with pytest.raises(ResultError):
        table_checks(k, log_df, -ddf, 1.0, top)


def test_the_stated_exponent_of_the_hernquist_center():
    # F ~ u^(-5/2) at the center: the table's last decades converge to it
    k, log_df, ddf = _hernquist_table()
    checks = table_checks(k, log_df, ddf, 1.0, {"top_kappa": -2.5})
    assert checks.kappa == -2.5
    assert checks.kappa_residual < 1e-4
    assert checks.k_top == k[-1]
    assert math.isnan(checks.edge_rel_at_floor)
