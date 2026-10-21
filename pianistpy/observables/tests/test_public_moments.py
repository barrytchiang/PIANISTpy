"""Tests of `sigma_r`, `sigma_t`, `losvd` and `aperture_losvd` on the isotropic
Plummer model.

The tests compare a `Model` with the model's `LosTable`, compare both with the
Plummer closed forms, convert the results to the physical unit system, and
check the InputError for a source that is neither a `Model` nor a `LosTable`.
"""

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
from pianistpy.validation import TIER_BOUNDS

# Grids smaller than the grids of the default settings, equal to the grids in
# test_stages.py. With these grids, sigma_r and sigma_t meet the Plummer closed
# forms to 1e-6.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    """Return the isotropic Plummer model with SETTINGS, and the `LosTable`
    of the model."""
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    return model, LosTable(model)


def test_sigma_r_and_sigma_t_from_a_model_and_from_its_table(plummer):
    """Assert that `sigma_r` and `sigma_t` on the model equal the closed forms
    to a relative :math:`10^{-6}` at nine radii from 0.01 to 100, with the
    shape of the radii. Assert that the two calls on the `LosTable` return the
    same values as the two calls on the model, and that `sigma_r` at a scalar
    radius returns a 0-dimensional array."""
    # The isotropic Plummer model has sigma_r^2 = P / 6 = (1 + r^2)^(-1/2) / 6
    # in internal units, with P the relative potential, and
    # sigma_t^2 = 2 sigma_r^2.
    model, table = plummer
    r = np.logspace(-2, 2, 9)
    exact_r = np.sqrt((1.0 + r**2) ** -0.5 / 6.0)
    for call, exact in ((sigma_r, exact_r), (sigma_t, math.sqrt(2.0) * exact_r)):
        from_model, from_table = call(model, r), call(table, r)
        assert from_model.shape == r.shape
        np.testing.assert_allclose(from_model, exact, rtol=TIER_BOUNDS[0])
        np.testing.assert_array_equal(from_table, from_model)
    assert sigma_r(model, 1.0).shape == ()


def test_sigma_r_from_a_coarse_start_meets_the_closed_form(plummer):
    """Assert that `sigma_r` from a profile started at n_profile_start = 4
    radii equals the closed form to a relative :math:`10^{-6}` at nine radii
    from 0.01 to 100."""
    # The radial dispersion starts from four logarithmic radii and adds rows
    # until every interval between rows meets the profile's tolerance.
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
    np.testing.assert_allclose(sigma_r(coarse, r), exact, rtol=TIER_BOUNDS[0])


def test_the_losvd_integrates_to_one_and_to_the_projected_dispersion(plummer):
    """Assert the properties of `losvd` on the Plummer model.

    The test asserts five properties:

    - at the projected radii 0.3 and 2, the distribution is even in the
      velocity v, not negative, of unit integral over v to a relative
      :math:`10^{-5}`, and has a second moment whose square root equals
      `sigma_los` to a relative :math:`10^{-5}`;
    - at the projected radius 0.3, the distribution is 0 at v = 1.45;
    - `losvd` on the model returns the same values as `losvd` on the table;
    - radii of shape (2, 1) and seven velocities broadcast to the shape
      (2, 7).
    """
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
    # The velocity 1.45 exceeds the escape velocity sqrt(2 P) = 1.38 at the
    # projected radius 0.3, the largest escape velocity along that line of sight.
    assert losvd(table, 0.3, 1.45) == 0.0
    # Given a model, losvd builds the model's table and returns the values of
    # losvd on that table.
    np.testing.assert_array_equal(
        losvd(model, 2.0, v[::500]), losvd(table, 2.0, v[::500])
    )
    # The projected radii and the velocities broadcast against each other.
    grid = losvd(table, np.array([[0.3], [2.0]]), v[::1000])
    assert grid.shape == (2, 7)


def test_the_aperture_losvd_integrates_to_one_and_to_the_aperture_dispersion(
    plummer,
):
    """Assert the properties of `aperture_losvd` on the Plummer model.

    For a centered circle of radius 1 and for a circle of radius 0.5 offset
    by 2, the test asserts that the distribution is even in the velocity v,
    not negative, of unit integral over v to a relative :math:`10^{-5}`, and has
    a second moment whose square root equals `sigma_los_aperture` to a
    relative :math:`10^{-5}`. The test also asserts that `aperture_losvd` at the
    velocity nodes of the stage `stages.aperture_losvd` equals the stage's
    values, and that the distribution in the centered circle of radius 1 is
    0 at v = 1.45.
    """
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
        # At the velocity nodes of the stage, the polynomial of each panel
        # passes through the stage's values. The two results agree to
        # rounding error relative to the largest value of the stage.
        stage = aperture_stage(model, table.table, table.projected, circle, SETTINGS)
        np.testing.assert_allclose(
            aperture_losvd(table, circle, stage.v_nodes),
            stage.losvd[0],
            rtol=1e-12,
            atol=1e-14 * float(np.max(stage.losvd[0])),
        )
    # The velocity 1.45 exceeds sqrt(2) = 1.41, the escape velocity at the
    # center of the Plummer model and the largest escape velocity in the model.
    assert aperture_losvd(table, Aperture.circle(1.0), 1.45) == 0.0


def test_the_model_unit_system_is_read_and_returned(plummer):
    """Assert that the calls on a table in the physical unit system read
    radii in parsecs and return quantities.

    The test converts the table with concentration 10, mass_0 = :math:`10^9`
    solar masses and r_0 = 500 pc. The test asserts that `sigma_r` and
    `sigma_t` return values in km / s. The test asserts that `losvd` and
    `aperture_losvd` read velocities in km / s and return densities in
    s / km. Each dispersion equals the dispersion in internal units times
    the velocity factor ``physical_v_kms`` to a relative :math:`10^{-14}`. Each
    density equals the density in internal units divided by ``physical_v_kms``
    to a relative :math:`10^{-14}`.
    """
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
    # In the physical system, losvd reads velocities in km/s and returns a
    # density per km/s.
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
    """Assert that `sigma_r`, `sigma_t`, `losvd` and `aperture_losvd` raise
    InputError when the source is the string "plummer", which is neither a
    `Model` nor a `LosTable`."""
    for call, args in (
        (sigma_r, (1.0,)),
        (sigma_t, (1.0,)),
        (losvd, (1.0, 0.0)),
        (aperture_losvd, (Aperture.circle(1.0), 0.0)),
    ):
        with pytest.raises(InputError, match="must be a Model or a LosTable"):
            call("plummer", *args)
