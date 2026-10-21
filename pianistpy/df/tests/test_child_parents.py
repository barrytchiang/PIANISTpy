r"""Tests of Method II children on truncated NFW parents.

The tests cover a radially biased child with :math:`\beta_{\rm c} = 0.5` on a
cusp of slope :math:`\gamma = 1`, whose DF near the center is a closed-form
series; a parent truncation that has no edge; and the tier that a child takes
from its parent model.
"""

import numpy as np

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df import compute_df, truncate

K_T = 0.0079589980165958


def _finite_on_its_support(df):
    t = df.table
    support = slice(int(t.i_first[0]), int(t.i_last[0]) + 1)
    return np.all(np.isfinite(np.asarray(t.log_df)[0, support]))


def test_a_child_at_beta_c_one_half_on_a_cusp_of_slope_one():
    # The NFW cusp has gamma = 1 = 2 beta_c, the critical case, so the child's DF
    # near the center is the closed-form series band.
    parent = truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy())), Truncation(k_t=K_T)
    )
    child = compute_df(Component(parent.profile, Anisotropy(beta_c=0.5)))
    assert child.inversion.critical_band
    assert _finite_on_its_support(child)
    model = truncate(child)
    assert model.validation.tier == "silent"
    # The parent of the child's model is the child's DF, so the model's
    # parents_level equals the level of the child DF's validation report.
    assert model.validation.parents_level == child.validation.level


def test_a_truncation_with_no_edge_is_refused_and_so_is_a_child_on_it():
    # The NFW Osipkov-Merritt Type I DF with r_a = 5, truncated at k_t = 0.1,
    # gives a solve that reaches its outer radius with the potential above the
    # floor, so the truncated model has no edge. truncate raises
    # UnphysicalModelError with a message that starts with "NO EDGE" and carries
    # the model as the exception's result. A component on that model's profile
    # makes compute_df raise UnphysicalModelError with a message that contains
    # "PARENT NOT SPATIALLY FINITE".
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
    # The child with beta_c = 1/2 of Chiang+(2026d), Fig. 7: the closed-form
    # series band carries the parent's central deficit to its cubic term in the
    # depth. At the first table node inside the band, log10 F of the table must
    # differ from log10 F of the computed rows by less than 1e-7.
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
    # truncate records the level of the model's validation report on the
    # model's profile as model_level, and the file of the model stores
    # model_level. A Method II child's DF takes the profile's model_level as its
    # parents_level. A child of a parent whose tier is "warned" therefore has
    # the tier "warned", whatever the child's own checks give.
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
    warning = TIER_NAMES.index("warned")
    profile = dataclasses.replace(parent.profile, model_level=warning)
    child = compute_df(Component(profile, Anisotropy()))
    assert child.validation.parents_level == warning
    assert child.validation.tier == "warned"
    model = truncate(child)
    assert model.validation.tier == "warned"
    # A profile whose model_level is None gives the child parents_level 0.
    unrecorded = dataclasses.replace(parent.profile, model_level=None)
    assert compute_df(Component(unrecorded, Anisotropy())).validation.parents_level == 0
