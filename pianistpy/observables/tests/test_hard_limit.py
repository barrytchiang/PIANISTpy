"""Tests that a result whose validation report has the tier "refused" raises
AccuracyError unless the settings set ``write_anyway=True``.

The tier "refused" holds a result whose estimated relative error is above
:math:`10^{-2}`. The tests set the tier "refused" with ``monkeypatch`` or with a
replaced report. The tested calls are `LosTable`, `aperture_losvd`,
`sigma_los_aperture`, `log_likelihood` and `Differentiable`.
"""

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

pytestmark = pytest.mark.slow

QUICK = Settings(quick=True)
ANYWAY = Settings(quick=True, write_anyway=True)
APERTURE = Aperture.circle(1.0)


def _refused(report):
    # The returned report keeps the keys and checks of `report`, and
    # parents_level = 3 sets the returned report's tier to "refused".
    return ValidationReport(keys=report.keys, checks=report.checks, parents_level=3)


@pytest.fixture(scope="module")
def model():
    r"""Return a model with the density profile of a Hernquist model truncated
    by ``Truncation(k_t=0.05)`` and the constant anisotropy
    :math:`\beta_{\rm c} = 0.25`, solved with the quick settings."""
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)
    return solve(Component(parent.profile, Anisotropy(beta_c=0.25)), settings=QUICK)


def test_a_table_beyond_0_01_is_refused_unless_asked(model, monkeypatch):
    """Assert that `LosTable` raises AccuracyError when the table's report has
    the tier "refused".

    With ``write_anyway=True``, `LosTable` returns a table whose report has the
    tier "refused".
    """
    validate = core.validate_table
    monkeypatch.setattr(core, "validate_table", lambda *a: _refused(validate(*a)))
    with pytest.raises(AccuracyError, match="LosTable.*beyond 0.01"):
        LosTable(model)
    table = LosTable(dataclasses.replace(model, settings=ANYWAY))
    assert table.validation.tier == "refused"


def test_an_aperture_beyond_0_01_is_refused_unless_asked(model, monkeypatch):
    """Assert that `aperture_losvd` and `sigma_los_aperture` on a `LosTable`
    raise AccuracyError when the aperture distribution's report has the tier
    "refused".

    With ``write_anyway=True``, `aperture_losvd` and `sigma_los_aperture` return
    finite values.
    """
    table = LosTable(model)
    asked = LosTable(dataclasses.replace(model, settings=ANYWAY))
    validate = core.validate_aperture
    monkeypatch.setattr(core, "validate_aperture", lambda *a: _refused(validate(*a)))
    for call in (
        lambda t: aperture_losvd(t, APERTURE, np.array([0.0, 0.2])),
        lambda t: sigma_los_aperture(t, APERTURE),
    ):
        with pytest.raises(AccuracyError, match="aperture distribution.*beyond 0.01"):
            call(table)
        assert np.all(np.isfinite(call(asked)))


def test_a_model_beyond_0_01_is_refused_by_the_likelihood_and_differentiable(
    model,
):
    """Assert that `log_likelihood` and `Differentiable` raise AccuracyError for
    a model whose report has the tier "refused"."""
    refused = dataclasses.replace(model, validation=_refused(model.validation))
    data = Data("sigma_los", np.ones(2), r_2d=np.array([0.3, 1.0]), errors=np.ones(2))
    with pytest.raises(AccuracyError, match="Component 0.*beyond 0.01"):
        log_likelihood(refused, data)
    with pytest.raises(AccuracyError, match="beyond 0.01"):
        Differentiable(refused)
