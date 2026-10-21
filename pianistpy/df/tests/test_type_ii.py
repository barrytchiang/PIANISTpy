r"""Tests of the checks and the truncations of Type II anisotropy models.

The untruncated DF of a Type II model has its density round trip "not judged",
and its slope-column check excludes the first node of the DF's support. A
truncation of a Type II model at :math:`\mathcal{K}_{\rm t}` above
:math:`P(R_{\rm a})` has tier "silent", with a round-trip error at most
:math:`10^{-6}`. A truncation whose edge would lie at or beyond
:math:`R_{\rm a}` raises UnphysicalModelError.
"""

import math

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    Truncation,
    UnphysicalModelError,
)
from pianistpy.df import compute_df, truncate
from pianistpy.df.report import NO_ROUND_TRIP_TYPE_II, NO_SLOPE_AT_TYPE_II_EDGE

pytestmark = pytest.mark.slow

OM_II = Anisotropy(r_a=100.0, kind="II")


@pytest.fixture(scope="module")
def hernquist_om_ii():
    return compute_df(Component(DoublePowerLaw.hernquist(), OM_II, name="om2"))


@pytest.fixture(scope="module")
def nfw_om_ii():
    return compute_df(Component(DoublePowerLaw.nfw(), OM_II, name="om2"))


def test_an_untruncated_df_judges_its_density_round_trip():
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="iso"))
    check = df.validation.checks["round_trip"]
    assert check.reason is None and check.tier == "silent"
    assert 0.0 < check.where and check.error <= 1e-6


def test_an_untruncated_type_ii_df_states_its_round_trip_not_judged(hernquist_om_ii):
    # The Type II DF is returned on its support K > P(R_a) only. The density
    # inside R_a would read the DF's continuation at K below P(R_a), so the
    # round-trip check of the untruncated DF is not formed.
    check = hernquist_om_ii.validation.checks["round_trip"]
    assert check.reason == NO_ROUND_TRIP_TYPE_II
    assert math.isnan(check.error) and check.where is None
    assert check.tier == "not judged" and check.level == 0
    assert hernquist_om_ii.validation.tier == "silent"


@pytest.mark.parametrize("name", ["hernquist_om_ii", "nfw_om_ii"])
def test_a_type_ii_truncation_is_silent(name, request):
    df = request.getfixturevalue(name)
    model = truncate(df, Truncation(k_t=0.3))
    assert model.validation.tier == "silent"
    assert model.validation.checks["round_trip"].error <= 1e-6


def test_the_type_ii_truncation_nearest_its_edge_is_silent(hernquist_om_ii):
    # k_t_min is the next double above P(R_a), since the DOMAIN condition
    # K_t > P(R_a) is strict. A truncation at K_t = k_t_min raises TYPE II EDGE.
    # A truncation at K_t = 1.1 P(R_a) succeeds, with R_t below R_a.
    k_edge = float(DoublePowerLaw.hernquist().p(np.array([100.0]))[0])
    assert hernquist_om_ii.k_t_min == np.nextafter(k_edge, np.inf)
    with pytest.raises(UnphysicalModelError, match="TYPE II EDGE"):
        truncate(hernquist_om_ii, Truncation(k_t=hernquist_om_ii.k_t_min))
    model = truncate(hernquist_om_ii, Truncation(k_t=1.1 * k_edge))
    assert model.profile.r_t < 100.0
    assert model.validation.tier == "silent"
    assert model.validation.checks["round_trip"].error <= 1e-6


def test_a_type_ii_truncation_whose_edge_lies_beyond_r_a_is_refused(hernquist_om_ii):
    # At K_t = 0.0103636224, which is 1.0467 times P(R_a) = 1/101, the solve
    # passes R_a and would place the truncation edge beyond R_a, where the
    # Type II model is not defined.
    with pytest.raises(UnphysicalModelError, match="lies at or beyond R_a"):
        truncate(hernquist_om_ii, Truncation(k_t=0.0103636224))


@pytest.fixture(scope="module")
def nfw_com_ii():
    com_ii = Anisotropy(beta_c=0.25, r_a=100.0, kind="II")
    return compute_df(Component(DoublePowerLaw.nfw(), com_ii, name="com2"))


def test_a_com_ii_truncations_own_checks_are_silent(nfw_com_ii):
    df = nfw_com_ii
    assert df.validation.checks["round_trip"].reason == NO_ROUND_TRIP_TYPE_II
    model = truncate(df, Truncation(k_t=0.3))
    # Every check computed by the truncated model's own stages has tier
    # "silent", and the parent DF's report also has tier "silent".
    assert all(c.tier == "silent" for c in model.validation.checks.values())
    assert model.validation.checks["round_trip"].error <= 1e-6
    assert model.validation.tier == df.validation.tier == "silent"


def test_a_type_ii_slope_column_leaves_its_edge_node_unjudged(nfw_com_ii):
    # F goes as (K - P(R_a))^(-1/2) at the support's first node, which the
    # one-sided polynomial stencils of the slope check cannot read. The
    # slope-column check excludes that node and judges every other node.
    check = nfw_com_ii.validation.checks["slope_column"]
    assert check.excluded == NO_SLOPE_AT_TYPE_II_EDGE and check.reason is None
    assert check.tier == "silent" and check.error <= 1e-6
    record = nfw_com_ii.validation.to_dict()["checks"]["slope_column"]
    assert record["excluded"] == NO_SLOPE_AT_TYPE_II_EDGE
    back = type(nfw_com_ii.validation).from_dict(nfw_com_ii.validation.to_dict())
    assert back.checks["slope_column"] == check


def test_a_type_i_slope_column_judges_every_node():
    df = compute_df(Component(DoublePowerLaw.nfw(), Anisotropy(r_a=100.0), name="om1"))
    check = df.validation.checks["slope_column"]
    assert check.excluded is None
    assert "excluded" not in df.validation.to_dict()["checks"]["slope_column"]
