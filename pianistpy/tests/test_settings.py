"""Tests of the settings container and the identity string."""

import pytest

from pianistpy import InputError, Settings
from pianistpy.identity import identity_string, short_identity


def test_tiers_and_overrides():
    assert Settings().tier == "default"
    assert Settings(quick=True).tier == "quick"
    assert Settings().is_tested_tier
    custom = Settings(inversion={"n_nodes": 256})
    assert not custom.is_tested_tier
    assert Settings.from_dict(custom.to_dict()) == custom


def test_write_anyway_is_recorded_but_not_part_of_the_identity():
    plain, anyway = Settings(), Settings(write_anyway=True)
    assert not plain.write_anyway and anyway.write_anyway
    assert "write_anyway" not in plain.to_dict()
    assert anyway.to_dict()["write_anyway"] is True
    assert Settings.from_dict(anyway.to_dict()) == anyway
    assert Settings.from_dict(plain.to_dict()) == plain
    assert anyway != plain
    assert repr(anyway) == "Settings(write_anyway=True)"
    # the numbers of a result do not depend on it, so the identity does not
    assert anyway.numerical() == plain.numerical()
    a = identity_string({"x": 1}, plain.numerical(), "1.0")
    b = identity_string({"x": 1}, anyway.numerical(), "1.0")
    assert a == b
    with pytest.raises(InputError):
        Settings(write_anyway="yes")


def test_settings_are_immutable_and_refuse_unknown_groups():
    with pytest.raises(AttributeError):
        Settings().tier = "quick"
    with pytest.raises(InputError):
        Settings(inversions={"n_nodes": 256})


def test_fields_are_declared_typed_and_read_through_the_group():
    settings = Settings(sampling={"n_rows": 256}, truncation={"ode_rtol": 1})
    assert settings.sampling["n_rows"] == 256
    assert settings.sampling["n_nodes"] == Settings().sampling["n_nodes"]
    assert settings.truncation["ode_rtol"] == 1.0
    assert isinstance(settings.truncation["ode_rtol"], float)
    assert set(settings.resolved()) == {
        "inversion",
        "truncation",
        "sampling",
        "observables",
        "losvd",
    }
    with pytest.raises(TypeError):
        settings.sampling["n_rows"] = 1
    for bad in (
        {"sampling": {"n_row": 256}},
        {"sampling": {"n_rows": 256.0}},
        {"sampling": {"n_rows": True}},
        {"sampling": {"n_rows": 0}},
        {"truncation": {"ode_rtol": "tight"}},
    ):
        with pytest.raises(InputError):
            Settings(**bad)
    with pytest.raises(InputError):
        settings.group("sample")


def test_identity_ignores_the_order_of_keys_and_sees_every_change():
    a = identity_string({"x": 1, "y": 2}, {"tier": "default"}, "1.0")
    b = identity_string({"y": 2, "x": 1}, {"tier": "default"}, "1.0")
    c = identity_string({"x": 1, "y": 2}, {"tier": "quick"}, "1.0")
    assert a == b
    assert a != c
    assert short_identity(a) == a[:12]
