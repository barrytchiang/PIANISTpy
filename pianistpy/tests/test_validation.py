"""Tests of the tiers, the validation report and the quadrature leaf."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import InputError, ResultError
from pianistpy.numerics import gauss_legendre, gauss_legendre_unit
from pianistpy.validation import (
    CEILING,
    GOAL,
    HARD_LIMIT,
    Check,
    ValidationReport,
    tier_level,
)


@pytest.mark.parametrize(
    ("error", "level"),
    [
        (0.0, 0),
        (GOAL, 0),
        (2 * GOAL, 1),
        (CEILING, 1),
        (2 * CEILING, 2),
        (HARD_LIMIT, 2),
        (2 * HARD_LIMIT, 3),
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
    assert report.tier == "ceiling"
    assert list(report.to_dict()["checks"]) == list(keys)
    assert ValidationReport(keys, report.checks, parents_level=2).tier == "warning"
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
    # Eight nodes integrate polynomials up to degree 15 exactly.
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
    with pytest.raises(AccuracyError, match="not consumed"):
        refuse_unless_asked("The model 'x'", refused, Settings())
    refuse_unless_asked("The model 'x'", refused, Settings(write_anyway=True))
    refuse_unless_asked("The model 'x'", warned, Settings())
    # a parent's refusal is carried by the children's reports the same way
    inherited = ValidationReport(("a",), {"a": Check(error=0.0)}, parents_level=3)
    with pytest.raises(AccuracyError):
        refuse_unless_asked("The model 'x'", inherited, Settings())


def test_announce_logs_and_warns_above_the_goal_and_is_silent_within(caplog):
    from pianistpy.exceptions import AccuracyWarning
    from pianistpy.validation import Check, ValidationReport, announce

    within = ValidationReport(
        keys=("row",), checks={"row": Check(error=1e-8, where=1.0)}, parents_level=0
    )
    above = ValidationReport(
        keys=("row",), checks={"row": Check(error=3e-4, where=2.0)}, parents_level=0
    )
    with caplog.at_level("WARNING", logger="pianistpy"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            announce("The sampler of 'x'", within, "row 1.0e-08 at Psi = 1.0")
        with pytest.warns(AccuracyWarning, match="tier 'warning'"):
            announce("The sampler of 'x'", above, "row 3.0e-04 at Psi = 2.0")
    said = [r.getMessage() for r in caplog.records]
    assert said == [
        "The sampler of 'x' is above the goal (tier 'warning'): "
        "row 3.0e-04 at Psi = 2.0."
    ]


def test_a_report_is_read_against_the_key_sets_of_its_kind():
    sets = (("a", "b"), ("a", "b", "c"))
    written = ValidationReport(("a", "b"), {"a": Check(1e-8), "b": Check(2e-8)})
    record = written.to_dict()
    assert ValidationReport.from_dict(record, sets).keys == ("a", "b")
    # a check the file lacks (an earlier version's) is read as not judged
    older = dict(record, checks={"a": record["checks"]["a"]})
    read = ValidationReport.from_dict(older, (("a", "c"),))
    assert read.keys == ("a", "c")
    assert read.checks["c"].tier == "not judged"
    assert read.checks["c"].reason
    # a check no report of the kind has is refused by name
    stray = dict(record, checks={**record["checks"], "z": record["checks"]["a"]})
    with pytest.raises(ResultError, match="'z'"):
        ValidationReport.from_dict(stray, sets)
