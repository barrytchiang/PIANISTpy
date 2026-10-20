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


def test_the_floor_rows_density_is_on_the_edge_law(hernquist_df):
    # the floor event's row, Phi = 1e-10, the source table's first node: its
    # density is read at the floor itself, so it continues the edge law
    # rho ~ Phi^q from the row inside it (a margin of 1e-14 on the potential
    # had put it 2.5e-4 above)
    from pianistpy.df.truncation import ORIGINS

    profile = truncate(hernquist_df, Truncation(k_t=0.05)).profile
    i = int(np.flatnonzero(profile.origin == ORIGINS["event"])[-1])
    phi = np.asarray(profile.p_trunc, dtype=np.float64)
    rho = np.asarray(profile.rho_trunc, dtype=np.float64)[0]
    assert phi[i] == 1e-10 and i == profile.r.size - 2  # the row before the edge
    law = rho[i - 1] * (phi[i] / phi[i - 1]) ** profile.edge_exponent[0]
    assert abs(rho[i] / law - 1.0) <= 1e-8


def test_the_round_trip_reads_the_edge_approach_and_the_last_row(hernquist_df):
    # beyond the last half-decade of radius the rows approach the edge densely
    # in the potential: the check reads every one of them, to the last row
    # before the edge
    from pianistpy.df.profile_checks import _check_radii

    profile = truncate(hernquist_df, Truncation(k_t=0.05)).profile
    radii = _check_radii(profile, 0.0)
    r = np.asarray(profile.r, dtype=np.float64)
    assert radii[-1] == r[-2] and r[-1] not in radii  # the edge row is not read
    # the row nearest the last half-decade of radius, and every row after it
    r_half = 10.0 ** (np.floor(2.0 * np.log10(r[-2])) / 2.0)
    j = int(np.argmin(np.abs(np.log10(r[:-1] / r_half))))
    assert np.array_equal(radii[radii >= r[j]], r[j:-1])
    assert radii.size == np.unique(radii).size


def test_the_round_trip_reads_each_radius_as_alone(hernquist_df):
    # the DF's density is read on all the radii together, in chunks of one
    # panel count: each radius reads as it does alone
    import pianistpy.df.profile_checks as checks

    model = truncate(hernquist_df, Truncation(k_t=0.05))
    profile, anisotropy = model.profile, model.components[0].anisotropy
    radii = checks._check_radii(profile, 0.0)[::50]
    phi = np.asarray(profile.potential(radii), dtype=np.float64)
    together = checks.density_from_df(radii, phi, model.table, model.form, anisotropy)
    alone = [
        float(
            checks.density_from_df(
                radii[i : i + 1], phi[i : i + 1], model.table, model.form, anisotropy
            )[0]
        )
        for i in range(radii.size)
    ]
    np.testing.assert_allclose(together, alone, rtol=1e-14)


def test_the_round_trip_reads_the_depth_column_whatever_the_rounded_sum(hernquist_df):
    import dataclasses

    from pianistpy.df.profile_checks import round_trip

    model = truncate(hernquist_df, Truncation(k_t=0.05))
    profile = model.profile
    assert profile.u_trunc.size == profile.r.size  # the column is carried
    # the first row's potential one ulp lower: P(r_0) + u(r_0) no longer rounds
    # to P_0, and the depths the check reads are the column's all the same
    p_trunc = np.array(profile.p_trunc, dtype=np.float64)
    p_trunc[0] = np.nextafter(p_trunc[0], -np.inf)
    missed = dataclasses.replace(profile, p_trunc=p_trunc)
    assert float(missed.p_trunc[0]) + float(missed.u_trunc[0]) != float(model.form.p_0)
    rest = (model.table, model.form, model.components[0].anisotropy)
    assert round_trip(missed, *rest) == round_trip(profile, *rest)


def test_a_check_that_forms_the_depth_by_subtraction_says_so(hernquist_df):
    import dataclasses

    from pianistpy.df.profile_checks import DEPTH_FALLBACK, round_trip, row_density
    from pianistpy.validation import ValidationReport

    model = truncate(hernquist_df, Truncation(k_t=0.05))
    rest = (model.table, model.form, model.components[0].anisotropy)
    rules = model.settings.truncation
    # with the column: nothing is written, and the record is as before
    assert round_trip(model.profile, *rest).source is None
    assert row_density(model.profile, rules).source is None
    assert "source" not in model.validation.to_dict()["checks"]["round_trip"]
    # without it: both checks say where their depths came from
    bare = dataclasses.replace(model.profile, u_trunc=np.zeros(0))
    checks = {
        "round_trip": round_trip(bare, *rest),
        "row_density": row_density(bare, rules),
    }
    assert all(c.source == DEPTH_FALLBACK for c in checks.values())
    report = ValidationReport(keys=tuple(checks), checks=checks)
    assert ValidationReport.from_dict(report.to_dict()).checks == checks


def test_the_marginal_cusp_carries_its_deficit_into_the_inner_rows():
    from pianistpy.df.profile_checks import density_from_df

    # NFW at beta_c = 0.5, 2 beta_c = gamma: the deficit has the cusp's power
    # and changes with the depth, so the window's median is not the amplitude
    # at the center; the lowered center gives the inner rows their density
    nfw = compute_df(Component(DoublePowerLaw.nfw(), Anisotropy(beta_c=0.5)))
    model = truncate(nfw, Truncation(k_t=0.19986231811501387))
    profile = model.profile
    fill = np.flatnonzero(np.asarray(profile.origin) == 3)
    rho = np.sum(np.atleast_2d(profile.rho_trunc), axis=0)[fill]
    from_df = density_from_df(
        profile.r[fill],
        profile.p_trunc[fill],
        model.table,
        model.form,
        model.components[0].anisotropy,
        profile.u_trunc[fill],
    )
    assert fill.size > 1000 and np.max(np.abs(from_df / rho - 1.0)) <= 1e-8
    assert model.validation.tier == "goal"


def test_a_mass_beyond_the_total_is_refused_before_any_solve(hernquist_df):
    # the Hernquist model's total mass is 1.21 times its mass inside r = 10
    with pytest.raises(UnphysicalModelError, match="M_0 = 1.21"):
        truncate(hernquist_df, Truncation(mass_t=1.5, concentration=10.0))


def test_a_target_radius_is_reached(hernquist_df):
    model = truncate(hernquist_df, Truncation(r_t=20.0))
    assert model.profile.r_t == pytest.approx(20.0, rel=1e-6)
    assert model.validation.tier == "goal"


def test_a_target_mass_well_inside_the_total_is_reached(hernquist_df):
    # in units of the mass inside the concentration radius, M(<10) = 100 / 121
    model = truncate(hernquist_df, Truncation(mass_t=0.3, concentration=10.0))
    m_c = float(DoublePowerLaw.hernquist().mass(np.array([10.0]))[0])
    assert model.profile.mass_t / m_c == pytest.approx(0.3, rel=1e-6)
    assert model.validation.tier == "goal"


def test_a_type_ii_target_radius_beyond_its_anisotropy_radius_is_refused():
    df = compute_df(
        Component(DoublePowerLaw.hernquist(), Anisotropy(r_a=100.0, kind="II"))
    )
    with pytest.raises(UnphysicalModelError, match="beyond R_a = 100"):
        truncate(df, Truncation(r_t=150.0))
    # a mass only a truncation radius beyond R_a would hold: M(<100) is 1.186
    # times M(<10), so 1.19 is refused by the search, with R_a its binding limit
    with pytest.raises(UnphysicalModelError, match="R_a"):
        truncate(df, Truncation(mass_t=1.19, concentration=10.0))


def test_a_center_negative_by_the_cusp_theorem_is_refused():
    cored = DoublePowerLaw(rho_alpha=1.0, rho_beta=4.0, rho_gamma=0.0)
    df = compute_df(Component(cored, Anisotropy(beta_c=0.5), name="cored"))
    with pytest.raises(UnphysicalModelError, match="CENTRAL VALIDITY") as raised:
        truncate(df, Truncation(k_t=0.05))
    assert raised.value.result is df


def test_a_degenerate_central_series_the_floor_alone_cuts_is_refused():
    from types import SimpleNamespace

    from pianistpy.df.truncated import central_series_refusal

    # gamma = 2 beta_c: the series' leading exponent vanishes; below gamma near
    # 2 the floor's depth lies below the table's top and nothing is refused
    for gamma in (1.0, 1.5, 1.9):
        component = Component(
            DoublePowerLaw(1.0, 4.0, gamma), Anisotropy(beta_c=gamma / 2.0)
        )
        central_series_refusal(SimpleNamespace(components=(component,)))
    steep = Component(
        DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(beta_c=0.995), name="steep"
    )
    df = compute_df(steep)
    # named before the untruncated model's own check, which would name the
    # spurious global minimum the degenerate table takes near the center
    with pytest.raises(UnphysicalModelError, match="CENTRAL SERIES") as raised:
        truncate(df)
    assert raised.value.result is df
    with pytest.raises(UnphysicalModelError, match="CENTRAL SERIES") as raised:
        truncate(df, Truncation(k_t=0.05))
    assert raised.value.result is df


def test_the_validity_certificate_is_judged_at_the_rows_top():
    # beta_c +1/4 at gamma 1.99: top_limit lies below the host floor's depth,
    # where the rows end; judged there the columns are representable
    floor_cut = compute_df(
        Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(beta_c=0.25))
    )
    assert floor_cut.validation.checks["validity"].error == 0.0
    # isotropic: top_limit lies above the floor's depth, the rows reach it, and
    # the linear F there leaves the certificate's margin
    reached = compute_df(Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy()))
    assert reached.validation.checks["validity"].error == np.inf


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


def test_a_type_ii_truncation_whose_edge_lies_beyond_r_a_is_refused():
    # K_t = 0.0102 lies above P(R_a) = 1/101, so the untruncated R(K_t) = 97 is
    # inside R_a = 100; the truncated model's own potential is still above the
    # floor at R_a, where rho_aux / (1 - r^2/R_a^2) has a pole
    om_ii = Anisotropy(r_a=100.0, kind="II")
    df = compute_df(Component(DoublePowerLaw.hernquist(), om_ii, name="om2"))
    with pytest.raises(UnphysicalModelError, match="TYPE II EDGE") as raised:
        truncate(df, Truncation(k_t=0.0102))
    assert raised.value.result is df


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


def test_the_half_integer_test_is_the_inversions_own():
    # the closed form's order where order_pair's kernel exponent is zero, and
    # nowhere else: no tolerance around +-1/2 (release audit 317 A2)
    from pianistpy import Anisotropy
    from pianistpy.df.truncated import _half_integer_order

    assert _half_integer_order(Anisotropy(beta_c=0.5)) == 1
    assert _half_integer_order(Anisotropy(beta_c=-0.5)) == 2
    assert _half_integer_order(Anisotropy(beta_c=0.5 - 1e-10)) is None
    assert _half_integer_order(Anisotropy(beta_c=-0.5 + 1e-10)) is None
    assert _half_integer_order(Anisotropy(beta_c=0.25)) is None
    assert _half_integer_order(Anisotropy(beta_c=0.5, r_a=10.0)) is None


def test_the_truncated_df_reports_its_clip():
    # release audit 317 A3: the largest negative of F(K + K_t) - F(K_t) set to
    # zero is a check of the model's report (zero where the difference never
    # goes below zero)
    from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
    from pianistpy.df import compute_df, truncate

    model = truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.2),
    )
    clip = model.validation.checks["df_clip"]
    assert clip.error == 0.0 and clip.tier == "goal"
