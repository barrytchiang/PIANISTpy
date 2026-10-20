r"""Tests of a Method II child on a parent no other test builds: a radially
biased child at :math:`\beta_{\rm c} = 0.5` on a cusp of slope 1, whose
central band takes the closed form."""

import numpy as np

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df import compute_df, truncate

K_T = 0.0079589980165958


def _finite_on_its_support(df):
    t = df.table
    support = slice(int(t.i_first[0]), int(t.i_last[0]) + 1)
    return np.all(np.isfinite(np.asarray(t.log_df)[0, support]))


def test_a_child_at_beta_c_one_half_on_a_cusp_of_slope_one():
    # gamma = 2 beta_c: the central band is the closed form's
    parent = truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy())), Truncation(k_t=K_T)
    )
    child = compute_df(Component(parent.profile, Anisotropy(beta_c=0.5)))
    assert child.inversion.critical_band
    assert _finite_on_its_support(child)
    model = truncate(child)
    assert model.validation.tier == "goal"
    # the model's parent is its DF: the DF's tier is the model's parents' tier
    assert model.validation.parents_level == child.validation.level


def test_a_truncation_with_no_edge_is_refused_and_so_is_a_child_on_it():
    # an NFW Osipkov-Merritt Type I parent truncated near its k_t_min (0.069):
    # its solve reaches the outer radius above the floor, so it has no edge;
    # truncate refuses it by name with the model as its result, and a component
    # on that model's profile is refused by name
    import pytest

    from pianistpy import UnphysicalModelError
    from pianistpy.df import Model

    om = Anisotropy(r_a=5.0)
    df = compute_df(Component(DoublePowerLaw.nfw(), om))
    with pytest.raises(
        UnphysicalModelError, match="NO EDGE: the solve reaches"
    ) as raised:
        truncate(df, Truncation(k_t=0.1))
    parent = raised.value.result
    assert isinstance(parent, Model) and not parent.spatially_finite
    for anisotropy in (om, Anisotropy()):
        with pytest.raises(UnphysicalModelError, match="PARENT NOT SPATIALLY FINITE"):
            compute_df(Component(parent.profile, anisotropy))


def test_the_closed_form_band_meets_the_rows_at_its_anchor():
    # Fig. 7's beta_c = +1/2 child: the band carries the parent's deficit to its
    # cubic term, so it meets the computed rows at the anchor. With the deficit
    # linear in the depth the join stepped by 2.45e-6 in log10 F; measured
    # 6.7e-9 with the cubic.
    from pianistpy import Settings
    from pianistpy.df.tabulated import tabulated_df

    parent = truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy())),
        Truncation(k_t=0.16502882092587715),
    )
    _, table, form, _, _, inversion, rows = tabulated_df(
        Component(parent.profile, Anisotropy(beta_c=0.5)), Settings()
    )
    band = dict(inversion.critical_band[0])
    assert band["critical_band"].startswith("closed-form series band")
    u_a = band["critical_band_u"]
    p_0 = float(form.p_0)
    support = slice(int(table.i_first[0]), int(table.i_last[0]) + 1)
    k, log_f = np.asarray(table.k)[0, support], np.asarray(table.log_df)[0, support]
    raw = np.interp(k, np.asarray(rows.k[0]), np.asarray(rows.f[0]))
    first = int(np.flatnonzero(p_0 - k <= u_a)[0])
    assert abs(log_f[first] - np.log10(raw[first])) < 1e-7


def test_a_childs_tier_carries_its_parent_models(tmp_path):
    # truncate records the model's tier on its profile, and a
    # Method II child's DF takes it as its parents' tier, so a child of a
    # warning parent reads warning however well its own checks do; the field
    # is written and read with the file
    import dataclasses

    from pianistpy import read
    from pianistpy.validation import TIER_NAMES

    parent = truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy())), Truncation(k_t=K_T)
    )
    assert parent.profile.model_level == parent.validation.level
    path = tmp_path / "parent.h5"
    parent.write(path)
    assert read(path).profile.model_level == parent.validation.level
    warning = TIER_NAMES.index("warning")
    profile = dataclasses.replace(parent.profile, model_level=warning)
    child = compute_df(Component(profile, Anisotropy()))
    assert child.validation.parents_level == warning
    assert child.validation.tier == "warning"
    model = truncate(child)
    assert model.validation.tier == "warning"
    # a profile that records no tier reads as level 0
    unrecorded = dataclasses.replace(parent.profile, model_level=None)
    assert compute_df(Component(unrecorded, Anisotropy())).validation.parents_level == 0
