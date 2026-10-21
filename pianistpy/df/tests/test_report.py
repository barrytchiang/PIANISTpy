"""Tests of the validation report of a closed-form untruncated DF."""

import numpy as np

from pianistpy.df.analytic import analytic_model
from pianistpy.df.report import DF_REPORT_KEYS, validate_df
from pianistpy.df.stages import PotentialDensityPair


def test_the_closed_form_hernquist_df_is_silent():
    """The report of the closed-form Hernquist DF has the keys of an
    untruncated DF with a tail, with no "eps" check, and its tier is "silent":
    every check is at most :math:`10^{-6}`. The Hernquist DF is non-negative
    and non-decreasing, so its lowest admissible truncation energy is zero and
    no truncation gate fires."""
    # The density round trip reads the lookup's interpolation of the table.
    # A table of 4000 nodes keeps that interpolation error below 1e-6, the
    # upper bound of the "silent" tier.
    model = analytic_model("hernquist", n_nodes=4000)
    pair = PotentialDensityPair(
        components=model.components,
        is_analytic=True,
        p_0=1.0,
        r=np.zeros(0),
        p=np.zeros(0),
        rho=np.zeros((1, 0)),
    )
    k_t_min, report, gates = validate_df(pair, model.table, model.form)
    assert report.keys == DF_REPORT_KEYS["untruncated_df_with_tail"]
    assert "eps" not in report.keys
    assert report.tier == "silent"
    assert k_t_min == 0.0 and gates.set_by == "none" and gates.fired == ()
