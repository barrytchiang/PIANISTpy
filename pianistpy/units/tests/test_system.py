"""Tests of the unit system in which a result presents the result's quantities."""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G
from numpy.testing import assert_allclose

from pianistpy import DoublePowerLaw, InputError, read
from pianistpy.df.analytic import analytic_model
from pianistpy.units import Units

NFW = DoublePowerLaw.nfw()


def test_internal_units_change_nothing():
    value = np.array([0.5, 2.0])
    for kind in ("length", "velocity", "mass", "density", "potential"):
        assert np.array_equal(Units().convert(kind, value, NFW), value)


def test_model_units_make_the_mass_inside_the_concentration_radius_one():
    units = Units.build("model", concentration=10.0)
    mass_inside = NFW.mass(np.array([10.0]))
    assert_allclose(units.convert("mass", mass_inside, NFW), 1.0, rtol=1e-14)
    # Internal units and model units both measure a length in units of r_0,
    # so the model units leave a length unchanged.
    assert np.array_equal(units.convert("length", [3.0], NFW), [3.0])


def test_physical_units_are_consistent_with_gravity():
    # The circular velocity v at the concentration radius c r_0 obeys
    # v**2 = G M_0 / (c r_0), with M_0 the mass inside c r_0.
    units = Units.build(
        "physical", concentration=10.0, mass_0=1e12 * u.Msun, r_0=20.0 * u.kpc
    )
    mass = units.convert("mass", NFW.mass(np.array([10.0])), NFW)
    radius = units.convert("length", [10.0], NFW)
    assert_allclose(mass.to_value(u.Msun), 1e12, rtol=1e-12)
    assert_allclose(radius.to_value(u.kpc), 200.0, rtol=1e-14)
    v_internal = np.sqrt(NFW.mass(np.array([10.0])) / 10.0)
    v = units.convert("velocity", v_internal, NFW)
    expected = np.sqrt(G * mass / radius).to_value(u.km / u.s)
    assert_allclose(v.to_value(u.km / u.s), expected, rtol=1e-10)
    # The potential factor is the square of the velocity factor, and the density
    # factor is the mass factor over the cube of the length factor divided by
    # 4 pi, because the internal unit of mass is 4 pi rho_w r_0**3.
    p = units.convert("potential", [1.0], NFW)
    assert_allclose(
        p.to_value(u.km**2 / u.s**2), v.to_value(u.km / u.s) ** 2 / v_internal**2
    )
    rho = units.convert("density", [1.0], NFW)
    scale = units.convert("mass", [1.0], NFW) / units.convert("length", [1.0], NFW) ** 3
    assert_allclose(
        rho.to_value(u.Msun / u.pc**3),
        scale.to_value(u.Msun / u.pc**3) / (4.0 * math.pi),
        rtol=1e-12,
    )


def test_r_vir_stands_for_the_scale_radius():
    a = Units.build("physical", concentration=10.0, mass_0=1e12, r_vir=200.0 * u.kpc)
    b = Units.build("physical", concentration=10.0, mass_0=1e12 * u.Msun, r_0=2.0e4)
    assert a.r_0_pc == pytest.approx(b.r_0_pc)
    assert a.mass_0_msun == b.mass_0_msun


@pytest.mark.parametrize(
    "build",
    [
        lambda: Units.build("model"),
        lambda: Units.build("physical", concentration=10.0, mass_0=1e12),
        lambda: Units.build("physical", concentration=10.0, r_0=1.0),
        lambda: Units.build(
            "physical", concentration=10.0, mass_0=1.0, r_0=1.0, r_vir=2.0
        ),
        lambda: Units.build("physical", mass_0=1.0, r_vir=2.0),
        lambda: Units.build("model", concentration=-1.0),
        lambda: Units(system="imperial"),
    ],
)
def test_incomplete_units_are_refused(build):
    with pytest.raises(InputError):
        build()


def test_a_model_keeps_its_numbers_and_carries_its_units(tmp_path):
    model = analytic_model("hernquist")
    physical = model.to(
        "physical", concentration=10.0, mass_0=1e12 * u.Msun, r_0=20.0 * u.kpc
    )
    assert physical.identity == model.identity
    assert physical.profile is model.profile
    assert model.units.system == "internal" and physical.units.system == "physical"
    profiles = physical.radial_profiles()
    assert profiles["r"].unit == u.pc and profiles["mass"].unit == u.Msun
    assert_allclose(profiles["r"].to_value(u.kpc), 20.0 * model.profile.r, rtol=1e-14)
    internal = model.radial_profiles()
    assert np.array_equal(internal["density"], model.profile.rho_trunc)
    # The file records the model's units, so the model read back from the file
    # carries units equal to the units of the model written.
    path = tmp_path / "model.h5"
    physical.write(path)
    assert read(path).units == physical.units
    assert read(path).to("internal").units == Units()
