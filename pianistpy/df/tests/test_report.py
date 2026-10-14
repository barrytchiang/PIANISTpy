import numpy as np

from pianistpy.df.analytic import analytic_model
from pianistpy.df.report import DF_REPORT_KEYS, validate_df
from pianistpy.df.stages import PotentialDensityPair


def test_the_closed_form_hernquist_df_is_within_the_goal():
    # The closed-form table and its forms: every check of the DF's report
    # within the goal, no eps among them, and a non-negative, non-decreasing
    # DF admits every truncation energy.
    model = analytic_model("hernquist", n_nodes=400)
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
    assert report.tier == "goal"
    assert k_t_min == 0.0 and gates.set_by == "none" and gates.fired == ()
