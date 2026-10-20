"""Tests of the public intrinsic dispersions and the line-of-sight velocity
distribution, at projected radii and in apertures: a model and its prepared
table against each other, against the Plummer sphere and across unit
systems."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import InputError, Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import (
    Aperture,
    LosTable,
    aperture_losvd,
    losvd,
    sigma_los,
    sigma_los_aperture,
    sigma_r,
    sigma_t,
)
from pianistpy.observables.stages import aperture_losvd as aperture_stage
from pianistpy.validation import GOAL

# Smaller grids than the default tier, as the tests of the stages take them.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    """The isotropic Plummer model at the smaller grids, and its table."""
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    return model, LosTable(model)


def test_sigma_r_and_sigma_t_from_a_model_and_from_its_table(plummer):
    # isotropic Plummer: sigma_r^2 = Psi / 6 = (1 + r^2)^(-1/2) / 6 in internal
    # units, and sigma_t^2 = 2 sigma_r^2
    model, table = plummer
    r = np.logspace(-2, 2, 9)
    exact_r = np.sqrt((1.0 + r**2) ** -0.5 / 6.0)
    for call, exact in ((sigma_r, exact_r), (sigma_t, math.sqrt(2.0) * exact_r)):
        from_model, from_table = call(model, r), call(table, r)
        assert from_model.shape == r.shape
        np.testing.assert_allclose(from_model, exact, rtol=GOAL)
        np.testing.assert_array_equal(from_table, from_model)
    assert sigma_r(model, 1.0).shape == ()


def test_sigma_r_from_a_coarse_start_meets_the_closed_form(plummer):
    # four starting radii: the rows are added to the tolerance from there
    model, _ = plummer
    coarse = dataclasses.replace(
        model,
        settings=Settings(
            observables={
                "n_radii": 400,
                "n_jeans_nodes": 200,
                "n_projection_nodes": 200,
                "n_profile_start": 4,
            }
        ),
    )
    r = np.logspace(-2, 2, 9)
    exact = np.sqrt((1.0 + r**2) ** -0.5 / 6.0)
    np.testing.assert_allclose(sigma_r(coarse, r), exact, rtol=GOAL)


def test_the_losvd_integrates_to_one_and_to_the_projected_dispersion(plummer):
    # the distribution is even, integrates to one over all velocity, and its
    # second moment is the projected dispersion the table was certified against
    model, table = plummer
    half = np.linspace(0.0, 1.5, 3001)
    v = np.concatenate([-half[:0:-1], half])  # an exact mirror
    for r_2d in (0.3, 2.0):
        d = losvd(table, r_2d, v)
        assert d.shape == v.shape
        np.testing.assert_array_equal(d, d[::-1])
        assert np.all(d >= 0.0)
        norm = np.trapezoid(d, v)
        second = np.trapezoid(v**2 * d, v)
        assert norm == pytest.approx(1.0, rel=1e-5)
        assert math.sqrt(second) == pytest.approx(
            float(sigma_los(table, r_2d)), rel=1e-5
        )
    # the escape velocity along the line of sight bounds the support
    assert losvd(table, 0.3, 1.45) == 0.0
    # a model builds its own table and reads the same numbers
    np.testing.assert_array_equal(
        losvd(model, 2.0, v[::500]), losvd(table, 2.0, v[::500])
    )
    # the two arguments broadcast
    grid = losvd(table, np.array([[0.3], [2.0]]), v[::1000])
    assert grid.shape == (2, 7)


def test_the_aperture_losvd_integrates_to_one_and_to_the_aperture_dispersion(
    plummer,
):
    # in a centered and an offset circle: even, of unit integral, its second
    # moment the aperture's dispersion from the same table, and at the
    # stage's own nodes the stage's values
    model, table = plummer
    half = np.linspace(0.0, 1.5, 3001)
    v = np.concatenate([-half[:0:-1], half])  # an exact mirror
    for circle in (Aperture.circle(1.0), Aperture.circle(0.5, r_off=2.0)):
        d = aperture_losvd(table, circle, v)
        assert d.shape == v.shape
        np.testing.assert_array_equal(d, d[::-1])
        assert np.all(d >= 0.0)
        assert np.trapezoid(d, v) == pytest.approx(1.0, rel=1e-5)
        assert math.sqrt(np.trapezoid(v**2 * d, v)) == pytest.approx(
            float(sigma_los_aperture(table, circle)), rel=1e-5
        )
        # at the stage's nodes its values, each panel's polynomial through
        # them to rounding of the panel's scale
        stage = aperture_stage(model, table.table, table.projected, circle, SETTINGS)
        np.testing.assert_allclose(
            aperture_losvd(table, circle, stage.v_nodes),
            stage.losvd[0],
            rtol=1e-12,
            atol=1e-14 * float(np.max(stage.losvd[0])),
        )
    # the escape velocity along the line of sight bounds the support
    assert aperture_losvd(table, Aperture.circle(1.0), 1.45) == 0.0


def test_the_model_unit_system_is_read_and_returned(plummer):
    model, table = plummer
    physical = table.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    factors = physical.model.units.factors(physical.model.components[0].unit_profile)
    r = np.array([0.1, 1.0, 10.0])
    value = sigma_r(physical, r * R_0)
    assert str(value.unit) == "km / s"
    np.testing.assert_allclose(
        value.value, sigma_r(table, r) * factors.physical_v_kms, rtol=1e-14
    )
    value = sigma_t(physical, r * R_0)
    np.testing.assert_allclose(
        value.value, sigma_t(table, r) * factors.physical_v_kms, rtol=1e-14
    )
    # a velocity in km/s, a density per km/s
    v = np.array([0.0, 0.5, 1.0])
    density = losvd(physical, 2.0 * R_0, v * factors.physical_v_kms)
    assert str(density.unit) == "s / km"
    np.testing.assert_allclose(
        density.value, losvd(table, 2.0, v) / factors.physical_v_kms, rtol=1e-14
    )
    density = aperture_losvd(physical, Aperture.circle(R_0), v * factors.physical_v_kms)
    assert str(density.unit) == "s / km"
    np.testing.assert_allclose(
        density.value,
        aperture_losvd(table, Aperture.circle(1.0), v) / factors.physical_v_kms,
        rtol=1e-14,
    )


def test_a_source_that_is_neither_a_model_nor_a_table_is_refused():
    for call, args in (
        (sigma_r, (1.0,)),
        (sigma_t, (1.0,)),
        (losvd, (1.0, 0.0)),
        (aperture_losvd, (Aperture.circle(1.0), 0.0)),
    ):
        with pytest.raises(InputError, match="reads a Model or a LosTable"):
            call("plummer", *args)
