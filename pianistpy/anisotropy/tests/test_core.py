"""Tests of `Anisotropy`: the model label, the dictionary form and the input checks."""

import pytest

from pianistpy import Anisotropy, InputError


@pytest.mark.parametrize(
    ("kwargs", "model"),
    [
        ({}, "isotropic"),
        ({"beta_c": 0.0}, "isotropic"),
        ({"beta_c": 1e-12}, "constant"),
        ({"r_a": 10.0}, "om"),
        ({"beta_c": -0.25, "r_a": 10.0, "kind": "II"}, "com"),
    ],
)
def test_model_label_is_decided_by_exact_comparison(kwargs, model):
    assert Anisotropy(**kwargs).model == model


def test_dictionary_round_trip_keeps_an_infinite_radius():
    anisotropy = Anisotropy(beta_c=0.3)
    record = anisotropy.to_dict()
    assert record["r_a"] is None
    assert Anisotropy.from_dict(record) == anisotropy


@pytest.mark.parametrize("kwargs", [{"beta_c": 1.0}, {"r_a": 0.0}, {"kind": "III"}])
def test_inadmissible_parameters_are_refused(kwargs):
    with pytest.raises(InputError):
        Anisotropy(**kwargs)


def test_the_kind_is_one_where_r_a_is_infinite():
    # A model with an infinite anisotropy radius R_a does not depend on the
    # type, so Anisotropy stores kind "I" for every such model. Two such models
    # that differ only in the type given are then equal and have equal
    # dictionary forms.
    for beta_c in (0.0, 0.3):
        typed = Anisotropy(beta_c=beta_c, kind="II")
        assert typed.kind == "I"
        assert typed == Anisotropy(beta_c=beta_c)
        assert typed.to_dict() == Anisotropy(beta_c=beta_c).to_dict()
    assert Anisotropy(r_a=5.0, kind="II").kind == "II"
