"""Tests of the public observables: a model and its prepared table, against
each other, against the Plummer sphere and across unit systems."""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from pianistpy import AccuracyWarning, InputError, Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from pianistpy.validation import GOAL, ValidationReport

# Smaller grids than the default tier, as the tests of the stages take them.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
# The physical scales: M_0 in solar masses, r_0 in parsecs, M_0 the mass inside
# ten scale radii.
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    """The isotropic Plummer model at the smaller grids, and its table."""
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    return model, LosTable(model)


def test_sigma_los_from_a_model_and_from_its_table(plummer):
    # sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2) in internal units
    model, table = plummer
    r_2d = np.logspace(-3, 2, 11)
    exact = np.sqrt(3.0 * math.pi / 64.0 / np.sqrt(1.0 + r_2d**2))
    from_model, from_table = sigma_los(model, r_2d), sigma_los(table, r_2d)
    assert from_model.shape == r_2d.shape
    np.testing.assert_allclose(from_model, exact, rtol=GOAL)
    np.testing.assert_array_equal(from_table, from_model)


def test_sigma_los_aperture_by_its_two_routes(plummer):
    # over a centerd circle of radius A the mean square is
    # (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^-1]
    model, table = plummer
    for a in (0.5, 3.0):
        aperture = Aperture.circle(a)
        exact = math.sqrt(
            math.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        from_model = sigma_los_aperture(model, aperture)
        from_table = sigma_los_aperture(table, aperture)
        assert isinstance(from_model, float) and isinstance(from_table, float)
        assert from_model == pytest.approx(exact, rel=GOAL)
        assert from_table == pytest.approx(from_model, rel=GOAL)


def test_the_model_unit_system_is_read_and_returned(plummer):
    model, table = plummer
    physical = model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    factors = physical.units.factors(physical.components[0].unit_profile)
    r_2d = np.array([0.1, 1.0, 10.0])
    value = sigma_los(physical, r_2d * R_0)
    assert str(value.unit) == "km / s"
    np.testing.assert_allclose(
        value.value, sigma_los(model, r_2d) * factors.physical_v_kms, rtol=1e-14
    )
    # an aperture in parsecs, from the table presented in the same units
    in_parsecs = sigma_los_aperture(
        table.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0),
        Aperture.circle(1.0 * R_0),
    )
    assert in_parsecs.value == pytest.approx(
        sigma_los_aperture(table, Aperture.circle(1.0)) * factors.physical_v_kms,
        rel=1e-14,
    )


def test_a_model_of_two_components_is_refused(plummer):
    model, _ = plummer
    two = dataclasses.replace(model, components=model.components * 2)
    with pytest.raises(InputError, match="one component"):
        sigma_los(two, np.array([1.0]))


def test_the_projected_tables_traced_reader_gives_the_same_numbers(plummer):
    import jax
    import jax.numpy as jnp

    _, table = plummer
    projected = table.projected
    r_2d = np.geomspace(projected.r_2d[0], projected.r_2d[-1], 37)
    surface, sigma = projected.read(r_2d)
    traced = jax.jit(lambda r: projected.read_traced(r, jnp))(jnp.asarray(r_2d))
    # the surface density is the exponential of the cubic, whose value reaches
    # ln Sigma = -58 at the last row: its rounding there is some 60 ulp
    np.testing.assert_allclose(np.asarray(traced[0]), surface, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(traced[1]), sigma**2, rtol=1e-13)
    # beyond the last row both are zero
    beyond = projected.read_traced(np.array([2.0 * projected.r_2d[-1]]))
    assert float(beyond[0][0, 0]) == 0.0 and float(beyond[1][0, 0]) == 0.0


def test_a_table_carries_its_report_and_its_models_tier(plummer, caplog):
    model, table = plummer
    assert table.validation.keys == ("norm", "second_moment", "unjudged", "certificate")
    assert table.validation.tier == "goal"
    # a model above the goal: the table says so, and its report keeps the tier
    flagged = dataclasses.replace(
        model,
        validation=ValidationReport(
            keys=model.validation.keys, checks=model.validation.checks, parents_level=1
        ),
    )
    with (
        caplog.at_level("WARNING", logger="pianistpy"),
        pytest.warns(AccuracyWarning, match="LosTable"),
    ):
        above = LosTable(flagged)
    assert above.validation.tier == "ceiling"
    assert any(
        "LosTable" in r.getMessage() and "above the goal" in r.getMessage()
        for r in caplog.records
    )
    # an aperture read from that table says so too; one from the table at the
    # goal does not
    caplog.clear()
    with caplog.at_level("WARNING", logger="pianistpy"), warnings.catch_warnings():
        warnings.simplefilter("error", AccuracyWarning)
        sigma_los_aperture(table, Aperture.circle(1.0))
    assert not any("above the goal" in r.getMessage() for r in caplog.records)
    with (
        caplog.at_level("WARNING", logger="pianistpy"),
        pytest.warns(AccuracyWarning, match="aperture distribution"),
    ):
        sigma_los_aperture(above, Aperture.circle(1.0))
    assert any(
        "aperture distribution" in r.getMessage() and "'ceiling'" in r.getMessage()
        for r in caplog.records
    )
