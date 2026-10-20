"""Tests of the slope column of a table that states none: a Method II child's
check is not judged and says why; an inversion's table is judged as before."""

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df import compute_df, truncate
from pianistpy.df.report import NO_SLOPE_COLUMN_CHILD


def test_a_childs_slope_column_is_not_judged_and_says_why():
    parent = truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.05),
    )
    child = compute_df(Component(profile=parent.profile, anisotropy=Anisotropy()))
    assert not child.table.slopes_exact[0]
    report = child.validation
    slope = report.checks["slope_column"]
    assert slope.reason == NO_SLOPE_COLUMN_CHILD and slope.tier == "not judged"
    judged = [report.checks[k].level for k in report.keys if k != "slope_column"]
    assert report.level == max(judged)
    assert report.to_dict()["checks"]["slope_column"]["reason"] == NO_SLOPE_COLUMN_CHILD


def test_an_inversions_slope_column_is_judged():
    df = compute_df(Component(DoublePowerLaw.plummer(), Anisotropy()))
    assert df.table.slopes_exact[0]
    slope = df.validation.checks["slope_column"]
    assert slope.reason is None and slope.tier == "goal"
    assert "reason" not in df.validation.to_dict()["checks"]["slope_column"]
