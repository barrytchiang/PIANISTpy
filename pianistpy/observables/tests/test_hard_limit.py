"""One rule for a result beyond the hard limit: refused unless the settings ask
for it (``write_anyway``), at the LosTable, an aperture's distribution, the
likelihood and Differentiable."""

import dataclasses

import numpy as np
import pytest

from pianistpy import (
    AccuracyError,
    Anisotropy,
    Aperture,
    Component,
    Differentiable,
    DoublePowerLaw,
    Settings,
    Truncation,
    solve,
)
from pianistpy.inference import Data, log_likelihood
from pianistpy.observables import LosTable, aperture_losvd, core, sigma_los_aperture
from pianistpy.validation import ValidationReport

QUICK = Settings(quick=True)
ANYWAY = Settings(quick=True, write_anyway=True)
APERTURE = Aperture.circle(1.0)


def _refused(report):
    # the report of a result beyond the hard limit: its checks, tier "refused"
    return ValidationReport(keys=report.keys, checks=report.checks, parents_level=3)


@pytest.fixture(scope="module")
def model():
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)
    return solve(Component(parent.profile, Anisotropy(beta_c=0.25)), settings=QUICK)


def test_a_table_beyond_the_hard_limit_is_refused_unless_asked(model, monkeypatch):
    validate = core.validate_table
    monkeypatch.setattr(core, "validate_table", lambda *a: _refused(validate(*a)))
    with pytest.raises(AccuracyError, match="LosTable.*hard limit"):
        LosTable(model)
    table = LosTable(dataclasses.replace(model, settings=ANYWAY))
    assert table.validation.tier == "refused"


def test_an_aperture_beyond_the_hard_limit_is_refused_unless_asked(model, monkeypatch):
    table = LosTable(model)
    asked = LosTable(dataclasses.replace(model, settings=ANYWAY))
    validate = core.validate_aperture
    monkeypatch.setattr(core, "validate_aperture", lambda *a: _refused(validate(*a)))
    for call in (
        lambda t: aperture_losvd(t, APERTURE, np.array([0.0, 0.2])),
        lambda t: sigma_los_aperture(t, APERTURE),
    ):
        with pytest.raises(AccuracyError, match="aperture distribution.*hard limit"):
            call(table)
        assert np.all(np.isfinite(call(asked)))


def test_a_model_beyond_the_hard_limit_is_refused_by_the_likelihood_and_differentiable(
    model,
):
    refused = dataclasses.replace(model, validation=_refused(model.validation))
    data = Data("sigma_los", np.ones(2), r_2d=np.array([0.3, 1.0]), errors=np.ones(2))
    with pytest.raises(AccuracyError, match="Component 0.*hard limit"):
        log_likelihood(refused, data)
    with pytest.raises(AccuracyError, match="hard limit"):
        Differentiable(refused)
