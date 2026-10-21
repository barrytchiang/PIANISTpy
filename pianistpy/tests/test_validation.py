"""Tests of the bands, the validation report and the quadrature leaf."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import InputError, ResultError
from pianistpy.numerics import gauss_legendre, gauss_legendre_unit
from pianistpy.validation import (
    TIER_BOUNDS,
    Check,
    ValidationReport,
    tier_level,
)


@pytest.mark.parametrize(
    ("error", "level"),
    [
        (0.0, 0),
        (TIER_BOUNDS[0], 0),
        (2 * TIER_BOUNDS[0], 1),
        (TIER_BOUNDS[1], 1),
        (2 * TIER_BOUNDS[1], 2),
        (TIER_BOUNDS[2], 2),
        (2 * TIER_BOUNDS[2], 3),
        (float("nan"), 3),
        (float("inf"), 3),
    ],
)
def test_tier_boundaries(error, level):
    assert tier_level(error) == level


def test_report_has_fixed_keys_and_takes_the_worst_tier():
    keys = ("density", "potential")
    report = ValidationReport(
        keys, {"density": Check(3e-7, where=1.5), "potential": Check(2e-5)}
    )
    assert report.tier == "recorded"
    assert list(report.to_dict()["checks"]) == list(keys)
    assert ValidationReport(keys, report.checks, parents_level=2).tier == "warned"
    with pytest.raises(InputError):
        ValidationReport(keys, {"density": Check(1e-7)})
    with pytest.raises(InputError):
        ValidationReport(keys, {**report.checks, "extra": Check(1e-7)})


@pytest.mark.parametrize("n_nodes", [1, 2, 8, 16, 64])
def test_gauss_legendre_is_numpy_to_the_bit_and_cached(n_nodes):
    nodes, weights = gauss_legendre(n_nodes)
    reference = np.polynomial.legendre.leggauss(n_nodes)
    assert np.array_equal(nodes, reference[0])
    assert np.array_equal(weights, reference[1])
    assert gauss_legendre(n_nodes)[0] is nodes
    unit_nodes, unit_weights = gauss_legendre_unit(n_nodes)
    assert np.array_equal(unit_nodes, 0.5 * (reference[0] + 1.0))
    assert np.array_equal(unit_weights, 0.5 * reference[1])


def test_gauss_legendre_is_exact_for_polynomials_and_read_only():
    nodes, weights = gauss_legendre_unit(8)
    # An 8-node Gauss-Legendre rule integrates every polynomial of degree at
    # most 15 exactly.
    for degree in range(16):
        assert_allclose(weights @ nodes**degree, 1.0 / (degree + 1), rtol=1e-14)
    with pytest.raises(ValueError):
        nodes[0] = 0.0
    with pytest.raises(InputError):
        gauss_legendre(0)


def test_a_refused_result_is_consumed_only_when_asked_for_anyway():
    from pianistpy import AccuracyError, Settings
    from pianistpy.validation import refuse_unless_asked

    refused = ValidationReport(("a",), {"a": Check(error=2e-2)})
    warned = ValidationReport(("a",), {"a": Check(error=2e-3)})
    with pytest.raises(AccuracyError, match="does not take the result as input"):
        refuse_unless_asked("The model 'x'", refused, Settings())
    refuse_unless_asked("The model 'x'", refused, Settings(write_anyway=True))
    refuse_unless_asked("The model 'x'", warned, Settings())
    # A parent in the band refused makes a derived result's report refused,
    # through parents_level, so the derived result raises the same error.
    inherited = ValidationReport(("a",), {"a": Check(error=0.0)}, parents_level=3)
    with pytest.raises(AccuracyError):
        refuse_unless_asked("The model 'x'", inherited, Settings())


def test_announce_is_silent_records_and_warns_by_band(caplog):
    from pianistpy.exceptions import AccuracyWarning
    from pianistpy.validation import Check, ValidationReport, announce

    def one(error):
        return ValidationReport(
            keys=("row",),
            checks={"row": Check(error=error, where=1.0)},
            parents_level=0,
        )

    silent, recorded, warned = one(1e-8), one(3e-5), one(3e-4)
    with caplog.at_level("INFO", logger="pianistpy"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            announce("The sampler of 'x'", silent, "row 1.0e-08 at Psi = 1.0")
            announce("The sampler of 'x'", recorded, "row 3.0e-05 at Psi = 1.0")
        with pytest.warns(AccuracyWarning, match="tier 'warned'"):
            announce("The sampler of 'x'", warned, "row 3.0e-04 at Psi = 1.0")
    said = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert said == [
        (
            "INFO",
            "The sampler of 'x' is above 0.000001 (tier 'recorded'): "
            "row 3.0e-05 at Psi = 1.0.",
        ),
        (
            "WARNING",
            "The sampler of 'x' is above 0.0001 (tier 'warned'): "
            "row 3.0e-04 at Psi = 1.0.",
        ),
    ]


def test_a_report_of_an_earlier_version_reads_its_band_names():
    from pianistpy.validation import Check, ValidationReport

    record = ValidationReport(("a",), {"a": Check(1e-8)}, parents_level=1).to_dict()
    assert record["parents_tier"] == "recorded"
    for former, level in (("goal", 0), ("ceiling", 1), ("warning", 2), ("refused", 3)):
        read = ValidationReport.from_dict(dict(record, parents_tier=former))
        assert read.parents_level == level


def test_a_report_is_read_against_the_key_sets_of_its_kind():
    sets = (("a", "b"), ("a", "b", "c"))
    written = ValidationReport(("a", "b"), {"a": Check(1e-8), "b": Check(2e-8)})
    record = written.to_dict()
    assert ValidationReport.from_dict(record, sets).keys == ("a", "b")
    # A check that the key set requires and that the file does not hold is
    # read as a check that is not judged.
    older = dict(record, checks={"a": record["checks"]["a"]})
    read = ValidationReport.from_dict(older, (("a", "c"),))
    assert read.keys == ("a", "c")
    assert read.checks["c"].tier == "not judged"
    assert read.checks["c"].reason
    # A check that no key set of the kind holds raises ResultError, and the
    # message names the check.
    stray = dict(record, checks={**record["checks"], "z": record["checks"]["a"]})
    with pytest.raises(ResultError, match="'z'"):
        ValidationReport.from_dict(stray, sets)
