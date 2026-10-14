"""A component built on the truncated profile of another model (Method II)."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw, Settings, read
from pianistpy.df import Model
from pianistpy.df.core import model_identity
from pianistpy.df.edge import density_edge_exponent
from pianistpy.df.tests.test_results import _form, _report, _same, _table
from pianistpy.exceptions import InputError, UnphysicalModelError
from pianistpy.profiles import TruncatedProfile

N = 16
SOURCE = DoublePowerLaw.hernquist()
PARENT_BETA_C = 0.25


def _parent_profile(n_components=1, spatially_finite=True, model="parent-identity"):
    # a truncated DF with a positive slope at its edge, F ~ K, of a parent
    # with central anisotropy PARENT_BETA_C: the density ends as P^(5/2 - 1/4)
    q = density_edge_exponent(1.0, PARENT_BETA_C)
    # the last row is the edge where the profile is finite: zero density there
    rho = np.ones((n_components, N))
    if spatially_finite:
        rho[:, -1] = 0.0
    return TruncatedProfile(
        r=np.logspace(-3, 1, N),
        p_trunc=np.linspace(0.9, 0.0, N),
        rho_trunc=rho,
        mass_trunc=np.linspace(0.0, 2.0, N),
        k_t=0.1,
        r_t=10.0 if spatially_finite else math.inf,
        mass_t=2.0 if spatially_finite else math.inf,
        spatially_finite=spatially_finite,
        source=SOURCE,
        source_anisotropy=Anisotropy(beta_c=PARENT_BETA_C),
        model=model,
        edge_exponent=(q,) * n_components if spatially_finite else (),
    )


def test_a_component_takes_a_truncated_profile():
    profile = _parent_profile()
    child = Component(profile, Anisotropy(beta_c=-0.5))
    assert child.profile is profile
    assert child.unit_profile is SOURCE
    assert Component(SOURCE).unit_profile is SOURCE


def test_existence_at_the_edge_is_the_general_condition():
    # beta_c - beta_p > -2 for a parent whose DF has a positive slope at its edge
    profile = _parent_profile()
    Component(profile, Anisotropy(beta_c=PARENT_BETA_C - 2.0 + 1e-9))
    for beta_c in (PARENT_BETA_C - 2.0, PARENT_BETA_C - 2.5):
        with pytest.raises(UnphysicalModelError, match="beta_c"):
            Component(profile, Anisotropy(beta_c=beta_c))


def test_a_profile_without_an_edge_has_no_condition():
    Component(_parent_profile(spatially_finite=False), Anisotropy(beta_c=-5.0))


def test_a_profile_of_several_components_is_refused():
    with pytest.raises(InputError, match="one"):
        Component(_parent_profile(n_components=2))


def test_the_record_identifies_the_profile_without_its_arrays():
    child = Component(_parent_profile(), Anisotropy(beta_c=-0.5))
    record = child.to_dict()
    assert record["profile"] == {
        "profile": "truncated",
        "model": "parent-identity",
        "source": SOURCE.to_dict(),
        "source_anisotropy": Anisotropy(beta_c=PARENT_BETA_C).to_dict(),
    }
    other = Component(_parent_profile(model="another"), Anisotropy(beta_c=-0.5))
    settings = Settings()
    assert model_identity((child,), None, settings) != model_identity(
        (other,), None, settings
    )
    with pytest.raises(InputError, match="arrays"):
        Component.from_dict(record)
    with pytest.raises(InputError, match="arrays"):
        Component.from_dict(record, profile=_parent_profile(model="another"))
    assert Component.from_dict(record, profile=child.profile) == child


def test_a_model_on_a_truncated_profile_is_its_file(tmp_path):
    parent = _parent_profile()
    components = (Component(parent, Anisotropy(beta_c=-0.5), name="child"),)
    settings = Settings()
    identity = model_identity(components, None, settings)
    model = Model(
        components=components,
        truncation=None,
        settings=settings,
        identity=identity,
        parent="df-identity",
        table=dataclasses.replace(
            _table(),
            log_df=_table().log_df[:1],
            log_df_slope=_table().log_df_slope[:1],
            i_first=(0,),
            i_last=(N - 1,),
            slopes_exact=(True,),
        ),
        form=_form(),
        profile=dataclasses.replace(parent, model=identity),
        validation=_report(),
        timing={},
        r_trust={},
        inversion=None,
    )
    path = tmp_path / "child.h5"
    model.write(path)
    back = read(path)
    assert back.components == model.components
    _same_fields(back.components[0].profile, parent)
    _same_fields(back.profile, model.profile)
    # the units come from the analytic profile behind the truncated one
    assert np.array_equal(back.radial_profiles()["r"], model.profile.r)
    assert float(back.r_t) == 10.0


def _same_fields(a, b):
    _same(a, b)
    assert a.source == b.source and a.model == b.model
