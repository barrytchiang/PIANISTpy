"""Tests of the inputs of the distribution-function computation."""

import pytest

import pianistpy.df as pp_df
from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
)
from pianistpy.df.core import model_identity, model_record


def test_component_round_trip():
    component = Component(
        DoublePowerLaw.nfw(), Anisotropy(beta_c=-0.25, r_a=10.0), mass=1.0, name="halo"
    )
    assert Component.from_dict(component.to_dict()) == component


def test_truncation_takes_exactly_one_target():
    assert Truncation(mass_t=2.0, concentration=10.0).mass_t == 2.0
    with pytest.raises(InputError):
        Truncation()
    with pytest.raises(InputError):
        Truncation(k_t=0.1, r_t=5.0)


def test_a_target_mass_needs_the_concentration_and_nothing_else_takes_it():
    with pytest.raises(InputError):
        Truncation(mass_t=2.0)
    with pytest.raises(InputError):
        Truncation(r_t=50.0, concentration=10.0)
    with pytest.raises(InputError):
        Truncation(mass_t=2.0, concentration=0.0)
    truncation = Truncation(mass_t=2.0, concentration=10.0)
    assert Truncation.from_dict(truncation.to_dict()) == truncation


def test_identity_follows_the_inputs_and_the_settings():
    halo = Component(DoublePowerLaw.nfw(), Anisotropy(beta_c=-0.25, r_a=10.0))
    truncation = Truncation(mass_t=2.0, concentration=10.0)
    base = model_identity([halo], truncation, Settings())
    assert base == model_identity(
        [halo], Truncation(mass_t=2.0, concentration=10.0), Settings()
    )
    assert base != model_identity(
        [halo], Truncation(mass_t=2.5, concentration=10.0), Settings()
    )
    assert base != model_identity([halo], truncation, Settings(quick=True))
    assert base != model_identity([halo], None, Settings())
    other = Component(
        DoublePowerLaw.nfw(), Anisotropy(beta_c=-0.25, r_a=10.0, kind="II")
    )
    assert base != model_identity([other], truncation, Settings())
    record = model_record([halo], truncation, Settings())
    assert Component.from_dict(record["inputs"]["components"][0]) == halo
    assert Truncation.from_dict(record["inputs"]["truncation"]) == truncation


def test_this_release_computes_one_component():
    halo = Component(DoublePowerLaw.nfw())
    stars = Component(DoublePowerLaw.hernquist(), mass=0.05, name="stars")
    for call in (pp_df.compute_df, pp_df.solve):
        with pytest.raises(InputError, match="one component"):
            call((halo, stars))


def test_truncate_says_so_where_the_df_diverges(monkeypatch, caplog):
    # the log, not the recent messages: those are a bounded queue, full after
    # the tests before this one, so that their length does not grow
    import dataclasses

    from pianistpy.df import core
    from pianistpy.df.analytic import analytic_model

    model = analytic_model("hernquist", n_nodes=11)
    divergent = dataclasses.replace(
        model,
        form=dataclasses.replace(
            model.form,
            below_floor=("edge_law",),
            below_floor_coefficients=((-0.5, 1.0, 0.0),),
        ),
    )

    class Backend:
        result = model

        @classmethod
        def truncate(cls, df, truncation):
            return cls.result

    monkeypatch.setattr(core, "_BACKEND", Backend)
    with caplog.at_level("DEBUG", logger="pianistpy"):
        assert pp_df.truncate(None) is model
    assert not caplog.records
    Backend.result = divergent
    with caplog.at_level("WARNING", logger="pianistpy"):
        assert pp_df.truncate(None).diverges_at_zero == (True,)
    assert any("K^-0.5" in r.getMessage() for r in caplog.records)
