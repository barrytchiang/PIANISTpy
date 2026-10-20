r"""Tests of a Type II model: its untruncated DF's round trip is not judged, nor
its slope column at the support's first node, and its physical products, the
truncations above :math:`P(R_{\rm a})`, are at the goal."""

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
    assert check.reason is None and check.tier == "goal"
    assert 0.0 < check.where and check.error <= 1e-6


def test_an_untruncated_type_ii_df_states_its_round_trip_not_judged(hernquist_om_ii):
    # the DF is returned over its support K > P(R_a) only: the density
    # inside R_a reads the continuation below it, so the check is not formed
    check = hernquist_om_ii.validation.checks["round_trip"]
    assert check.reason == NO_ROUND_TRIP_TYPE_II
    assert math.isnan(check.error) and check.where is None
    assert check.tier == "not judged" and check.level == 0
    assert hernquist_om_ii.validation.tier == "goal"


@pytest.mark.parametrize("name", ["hernquist_om_ii", "nfw_om_ii"])
def test_a_type_ii_truncation_is_at_the_goal(name, request):
    df = request.getfixturevalue(name)
    model = truncate(df, Truncation(k_t=0.3))
    assert model.validation.tier == "goal"
    assert model.validation.checks["round_trip"].error <= 1e-6


def test_the_type_ii_truncation_nearest_its_edge_is_at_the_goal(hernquist_om_ii):
    # k_t_min is the next float above P(R_a) (DOMAIN is strict); the gates
    # admit it, the edge gate refuses it and up to K_t = (1 + 1e-3) P(R_a)
    # here; 1.1 P(R_a) is admitted, R_t < R_a
    k_edge = float(DoublePowerLaw.hernquist().p(np.array([100.0]))[0])
    assert hernquist_om_ii.k_t_min == np.nextafter(k_edge, np.inf)
    with pytest.raises(UnphysicalModelError, match="TYPE II EDGE"):
        truncate(hernquist_om_ii, Truncation(k_t=hernquist_om_ii.k_t_min))
    model = truncate(hernquist_om_ii, Truncation(k_t=1.1 * k_edge))
    assert model.profile.r_t < 100.0
    assert model.validation.tier == "goal"
    assert model.validation.checks["round_trip"].error <= 1e-6


def test_a_type_ii_truncation_whose_edge_lies_beyond_r_a_is_refused(hernquist_om_ii):
    # at K_t = 1.0467 P(R_a) the solve passes R_a without stalling and places
    # the edge at R_t = 100.06, where the model is not defined
    with pytest.raises(UnphysicalModelError, match="lies at or beyond R_a"):
        truncate(hernquist_om_ii, Truncation(k_t=0.0103636224))


@pytest.fixture(scope="module")
def nfw_com_ii():
    com_ii = Anisotropy(beta_c=0.25, r_a=100.0, kind="II")
    return compute_df(Component(DoublePowerLaw.nfw(), com_ii, name="com2"))


def test_a_com_ii_truncations_own_checks_are_at_the_goal(nfw_com_ii):
    df = nfw_com_ii
    assert df.validation.checks["round_trip"].reason == NO_ROUND_TRIP_TYPE_II
    model = truncate(df, Truncation(k_t=0.3))
    # every check the model's own stages compute is at the goal, and so is its
    # parent DF
    assert all(c.tier == "goal" for c in model.validation.checks.values())
    assert model.validation.checks["round_trip"].error <= 1e-6
    assert model.validation.tier == df.validation.tier == "goal"


def test_a_type_ii_slope_column_leaves_its_edge_node_unjudged(nfw_com_ii):
    # F goes as (K - P(R_a))^(-1/2) at the support's first node, which
    # the one-sided stencils misread by 2.2e-3; the other nodes judge it
    check = nfw_com_ii.validation.checks["slope_column"]
    assert check.excluded == NO_SLOPE_AT_TYPE_II_EDGE and check.reason is None
    assert check.tier == "goal" and check.error <= 1e-6
    record = nfw_com_ii.validation.to_dict()["checks"]["slope_column"]
    assert record["excluded"] == NO_SLOPE_AT_TYPE_II_EDGE
    back = type(nfw_com_ii.validation).from_dict(nfw_com_ii.validation.to_dict())
    assert back.checks["slope_column"] == check


def test_a_type_i_slope_column_judges_every_node():
    df = compute_df(Component(DoublePowerLaw.nfw(), Anisotropy(r_a=100.0), name="om1"))
    check = df.validation.checks["slope_column"]
    assert check.excluded is None
    assert "excluded" not in df.validation.to_dict()["checks"]["slope_column"]
