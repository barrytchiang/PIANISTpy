"""Tests of the truncated model from its untruncated DF."""

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


@pytest.fixture(scope="module")
def hernquist_df():
    return compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))


def test_write_anyway_returns_a_model_beyond_the_hard_limit_with_a_warning(
    hernquist_df, monkeypatch
):
    import dataclasses
    import warnings

    import pianistpy.df.truncated as truncated
    import pianistpy.validation as validation
    from pianistpy import AccuracyError, AccuracyWarning, Settings

    # the hard limit lowered below this model's round trip, at the refusal site
    # and in the tiers, so that the report reads "refused" as on a real miss
    monkeypatch.setattr(truncated, "HARD_LIMIT", 0.0)
    monkeypatch.setattr(validation, "HARD_LIMIT", 0.0)
    with pytest.raises(AccuracyError, match="beyond the hard limit"):
        truncate(hernquist_df, Truncation(k_t=0.05))
    asked = dataclasses.replace(hernquist_df, settings=Settings(write_anyway=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = truncate(asked, Truncation(k_t=0.05))
    said = [w for w in caught if issubclass(w.category, AccuracyWarning)]
    assert len(said) == 1 and "write_anyway" in str(said[0].message)
    assert model.validation.tier == "refused"  # the verdict is in the record
    # the model is the same model: its identity ignores the override, and the
    # file records that it was asked for anyway
    from pianistpy.df.core import model_identity

    assert model.identity == model_identity(
        hernquist_df.components, Truncation(k_t=0.05), hernquist_df.settings
    )
    assert model.validation.checks["round_trip"].error <= 1e-6  # the numbers unchanged
    assert model.settings.to_dict()["write_anyway"] is True


def test_truncate_builds_a_finite_model_whose_df_gives_its_density(hernquist_df):
    model = truncate(hernquist_df, Truncation(k_t=0.05))
    profile = model.profile
    assert profile.spatially_finite and math.isfinite(profile.r_t)
    assert profile.k_t == 0.05 and model.parent == hernquist_df.identity
    assert model.form.p_0 == pytest.approx(0.95)
    assert model.form.below_floor == ("taylor",)
    # the model's own DF rebuilds the density of its own rows
    assert model.validation.checks["round_trip"].error <= 1e-6


def test_a_mass_beyond_the_total_is_refused_before_any_solve(hernquist_df):
    # the Hernquist model's total mass is 1.21 times its mass inside r = 10
    with pytest.raises(UnphysicalModelError, match="M_0 = 1.21"):
        truncate(hernquist_df, Truncation(mass_t=1.5, concentration=10.0))


def test_a_center_negative_by_the_cusp_theorem_is_refused():
    cored = DoublePowerLaw(rho_alpha=1.0, rho_beta=4.0, rho_gamma=0.0)
    df = compute_df(Component(cored, Anisotropy(beta_c=0.5), name="cored"))
    with pytest.raises(UnphysicalModelError, match="CENTRAL VALIDITY") as raised:
        truncate(df, Truncation(k_t=0.05))
    assert raised.value.result is df


def test_a_type_ii_df_is_returned_over_its_support_and_refused_below_it():
    om_ii = Anisotropy(r_a=100.0, kind="II")
    hernquist = DoublePowerLaw.hernquist()
    df = compute_df(Component(hernquist, om_ii, name="om2"))
    # the support is K > P(R_a): the table keeps its grid, NaN at and below it
    k_edge = float(hernquist.p(np.array([100.0]))[0])
    k = np.asarray(df.table.k[0])
    log_df = np.asarray(df.table.log_df[0])
    first = int(df.table.i_first[0])
    assert np.all(np.isnan(log_df[k <= k_edge]))
    assert k[first] > k_edge and np.all(k[:first] <= k_edge)
    assert df.k_t_min >= k_edge
    assert df.inversion.continuation == ("u = 1 - R^2/R_a^2, Merritt's polynomial",)
    # below it, the general refusal names the domain edge
    with pytest.raises(UnphysicalModelError, match="DOMAIN"):
        truncate(df, Truncation(k_t=1e-4))


def test_the_untruncated_model_is_the_dfs_own_where_it_is_valid(hernquist_df):
    # Hernquist isotropic: a non-negative DF, a finite mass
    model = truncate(hernquist_df)
    hernquist = DoublePowerLaw.hernquist()
    assert model.truncation is None and not model.spatially_finite
    assert model.profile.k_t == 0.0 and model.profile.mass_t == hernquist.mass_total
    r = model.profile.r
    assert np.array_equal(model.profile.p_trunc, np.asarray(hernquist.p(r)))
    assert model.validation.checks["round_trip"].error < 1e-6
    assert model.validation.checks["row_density"].error == 0.0
    assert np.min(model.profile.p_trunc) >= 1e-10  # the rows end at the floor
    assert model.validation.tier == "goal"
    assert model.table is hernquist_df.table and model.form is hernquist_df.form
    assert model.r_trust == {"goal": math.inf, "ceiling": math.inf}
    # compute_df records each stage's wall time
    assert {"energy_grid", "invert_rows", "asymptotic_form", "validate_df"} <= set(
        hernquist_df.timing
    )


def test_the_untruncated_model_is_refused_by_name_where_it_is_not_valid():
    nfw = compute_df(Component(DoublePowerLaw.nfw(), Anisotropy(), name="nfw"))
    with pytest.raises(UnphysicalModelError, match="UNTRUNCATED MASS") as raised:
        truncate(nfw)
    assert raised.value.result is nfw
    om_ii = compute_df(
        Component(DoublePowerLaw.hernquist(), Anisotropy(r_a=100.0, kind="II"))
    )
    with pytest.raises(UnphysicalModelError, match="UNTRUNCATED: .*domain_edge"):
        truncate(om_ii)
    # a non-negative DF with a global minimum above zero: K_t >= Q_min, K_t = 0
    # included (the lower limit of Appendix B)
    shallow = DoublePowerLaw(rho_alpha=1.0, rho_beta=3.5, rho_gamma=1.0)
    com_i = compute_df(Component(shallow, Anisotropy(beta_c=-0.75, r_a=10.0)))
    assert com_i.k_t_min > 0.0 and not com_i.is_physical
    with pytest.raises(UnphysicalModelError, match="global minimum at Q = .*Q_min"):
        truncate(com_i)
