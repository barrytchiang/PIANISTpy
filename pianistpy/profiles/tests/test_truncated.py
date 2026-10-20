"""Tests of the truncated profile's own evaluation."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pianistpy import Anisotropy, DoublePowerLaw
from pianistpy.exceptions import InputError, ResultError
from pianistpy.profiles import TruncatedProfile

HERNQUIST = DoublePowerLaw.hernquist()
ROWS_PER_DECADE = 200


def _untruncated(profile=HERNQUIST, decades=(-4.0, 4.0)):
    """The rows of an analytic profile, as an untruncated model's table."""
    r = np.geomspace(
        10.0 ** decades[0],
        10.0 ** decades[1],
        int(ROWS_PER_DECADE * (decades[1] - decades[0])) + 1,
    )
    return TruncatedProfile(
        r=r,
        p_trunc=np.asarray(profile.p(r)),
        rho_trunc=np.asarray(profile.rho(r))[np.newaxis, :],
        mass_trunc=np.asarray(profile.mass(r)),
        k_t=0.0,
        r_t=math.inf,
        mass_t=float(profile.mass_total),
        spatially_finite=False,
        source=profile,
        source_anisotropy=Anisotropy(),
        model="test-untruncated",
        edge_exponent=(),
    )


def _finite(q=2.5, r_t=2.0, n=2000):
    """A spatially finite profile with its edge row: a Kepler exterior joined
    to a power-law interior, so that every row is consistent with its slopes."""
    r = np.geomspace(1e-3, r_t, n)
    # a constant-density sphere inside r_t with p = 0 at the edge, in the
    # profile's unit: dM/dr = r^2 rho and dp/dr = -M/r^2
    mass_t = 1.0
    rho_0 = 3.0 * mass_t / r_t**3
    mass = rho_0 * r**3 / 3.0
    p = rho_0 * (r_t**2 - r**2) / 6.0
    rho = np.full(n, rho_0)
    # the last row is the edge: p = 0, rho = 0, r = r_t
    rho[-1] = 0.0
    p[-1] = 0.0
    return TruncatedProfile(
        r=r,
        p_trunc=p,
        rho_trunc=rho[np.newaxis, :],
        mass_trunc=mass,
        k_t=0.3,
        r_t=r_t,
        mass_t=mass_t,
        spatially_finite=True,
        source=HERNQUIST,
        source_anisotropy=Anisotropy(),
        model="test-finite",
        edge_exponent=(q,),
    )


def test_an_untruncated_profile_evaluates_its_closed_forms_between_the_rows():
    profile = _untruncated()
    r = np.geomspace(1.1e-4, 0.9e4, 301) * 1.0137  # off the rows
    assert_allclose(profile.potential(r), HERNQUIST.p(r), rtol=1e-9)
    assert_allclose(profile.mass(r), HERNQUIST.mass(r), rtol=1e-9)
    assert_allclose(profile.density(r)[0], HERNQUIST.rho(r), rtol=1e-7)


def test_the_rows_are_reproduced_to_rounding():
    # the interpolants pass through the rows; the logarithm and back costs ulps
    profile = _untruncated()
    assert_allclose(profile.potential(profile.r), profile.p_trunc, rtol=1e-14)
    assert_allclose(profile.mass(profile.r), profile.mass_trunc, rtol=1e-14)
    assert_allclose(profile.density(profile.r), profile.rho_trunc, rtol=1e-14)


def test_radius_inverts_the_potential():
    profile = _untruncated()
    r = np.geomspace(2e-4, 5e3, 50) * 1.0071
    assert_allclose(profile.radius(profile.potential(r)), r, rtol=1e-10)
    with pytest.raises(InputError):
        profile.radius(1.1 * profile.p_trunc[0])


def test_beyond_the_rows_the_power_laws_of_the_end_rows_continue():
    profile = _untruncated(decades=(-2.0, 2.0))
    r_end, p_end, m_end = profile.r[-1], profile.p_trunc[-1], profile.mass_trunc[-1]
    slope_p = -m_end / (r_end * p_end)
    slope_m = r_end**3 * profile.rho_trunc[0, -1] / m_end
    assert_allclose(
        profile.potential(1e3), p_end * (1e3 / r_end) ** slope_p, rtol=1e-13
    )
    assert_allclose(profile.mass(1e3), m_end * (1e3 / r_end) ** slope_m, rtol=1e-13)


def test_a_finite_profile_reads_its_last_interval_by_the_edge_forms():
    profile = _finite()
    assert profile.has_edge_row
    r_last = profile.r[-2]
    r = np.linspace(r_last, profile.r_t, 7)[1:-1]
    kepler = profile.mass_t * (1.0 / r - 1.0 / profile.r_t)
    assert_allclose(profile.potential(r), kepler, rtol=1e-14)
    p_last = profile.p_trunc[-2]
    assert_allclose(
        profile.density(r)[0],
        profile.rho_trunc[0, -2] * (kepler / p_last) ** 2.5,
        rtol=1e-13,
    )
    assert_allclose(profile.radius(kepler), r, rtol=1e-14)
    # at and beyond the edge
    assert profile.potential(profile.r_t) == 0.0
    assert np.all(profile.density(np.array([profile.r_t, 3.0]))[0] == 0.0)
    assert profile.mass(3.0) == profile.mass_t
    assert profile.radius(0.0) == profile.r_t
    assert profile.radius(-1.0) == profile.r_t
    # inside, the rows are read by the interpolants (the logarithm of the
    # potential steepens toward the edge, where the rows of this test are
    # coarse; a real profile is dense there)
    r_in = np.geomspace(2e-3, 0.5 * profile.r_t, 40)
    assert_allclose(profile.potential(r_in), (4.0 - r_in**2) / 16.0, rtol=1e-9)
    assert_allclose(profile.mass(r_in), 3.0 / 8.0 * r_in**3 / 3.0, rtol=1e-9)


def test_the_edge_row_invariant_is_checked():
    profile = _finite()
    rho = profile.rho_trunc.copy()
    rho[0, -1] = 1.0
    with pytest.raises(ResultError, match="zero density"):
        TruncatedProfile(**{**_fields(profile), "rho_trunc": rho})
    with pytest.raises(ResultError, match="truncation radius"):
        TruncatedProfile(**{**_fields(profile), "r_t": 2.5})
    with pytest.raises(ResultError, match="increasing"):
        TruncatedProfile(**{**_fields(profile), "r": profile.r[::-1]})


def test_a_profile_without_its_edge_row_is_read_to_its_last_row():
    # the working code's products end at a small positive potential: no edge
    # row, and the interpolants reach the last row
    profile = _finite()
    fields = _fields(profile)
    keep = slice(0, -1)
    trimmed = TruncatedProfile(
        **{
            **fields,
            "r": profile.r[keep],
            "p_trunc": profile.p_trunc[keep],
            "rho_trunc": profile.rho_trunc[:, keep],
            "mass_trunc": profile.mass_trunc[keep],
        }
    )
    assert not trimmed.has_edge_row
    assert_allclose(trimmed.potential(trimmed.r[-1]), trimmed.p_trunc[-1], rtol=1e-14)


def test_the_tables_hold_what_the_methods_evaluate():
    from pianistpy.numerics import hermite

    for profile in (_untruncated(decades=(-2.0, 2.0)), _finite()):
        tables = profile.interpolation_tables()
        assert tables.has_edge_row == profile.has_edge_row
        n = tables.log_r.size
        assert n == profile.r.size - (1 if profile.has_edge_row else 0)
        # inside the interior rows: beyond the last one a finite profile reads
        # its Kepler form, which the bare Hermite does not
        r = np.geomspace(profile.r[1] * 1.0003, profile.r[n - 1] / 1.0003, 25)
        log_p, _ = hermite(np.log(r), tables.log_r, tables.log_p, tables.log_p_slope)
        assert_allclose(np.exp(log_p), profile.potential(r), rtol=1e-14)
        log_m, _ = hermite(
            np.log(r), tables.log_r_all, tables.log_mass, tables.log_mass_slope
        )
        assert_allclose(np.exp(log_m), profile.mass(r), rtol=1e-14)
        # the density spline's coefficients, highest power first, per interval
        i = np.clip(
            np.searchsorted(tables.log_r, np.log(r), side="right") - 1, 0, n - 2
        )
        dx = np.log(r) - tables.log_r[i]
        c = tables.log_rho_coefficients[0][:, i]
        rho = np.exp(((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3])
        assert_allclose(rho, profile.density(r)[0], rtol=1e-13)
        if profile.has_edge_row:
            assert (tables.r_t, tables.mass_t) == (profile.r_t, profile.mass_t)
            assert (
                tables.r_last == profile.r[-2] and tables.p_last == profile.p_trunc[-2]
            )
            assert np.array_equal(
                tables.edge_exponent, np.asarray(profile.edge_exponent)
            )
        else:
            assert math.isinf(tables.r_t)


def test_the_anisotropy_of_the_source_is_stated():
    profile = _untruncated()
    assert profile.source_anisotropy == Anisotropy()
    assert profile.to_dict()["source_anisotropy"] == Anisotropy().to_dict()


def _fields(profile):
    import dataclasses

    return {
        item.name: getattr(profile, item.name) for item in dataclasses.fields(profile)
    }
