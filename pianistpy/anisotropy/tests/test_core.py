"""Tests of the anisotropy model."""

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
