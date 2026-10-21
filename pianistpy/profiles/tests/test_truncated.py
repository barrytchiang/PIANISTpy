"""Tests of the evaluation of `TruncatedProfile` between and beyond its nodes."""

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
    """Return an untruncated profile tabulated from an analytic profile.

    The nodes are uniform in log r over the given decades, with
    ROWS_PER_DECADE nodes per decade.
    """
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
    """Return a spatially finite test profile with an edge node.

    The profile is a sphere of constant density, total mass 1 and radius
    `r_t`, on `n` nodes from r = :math:`10^{-3}` to r = `r_t`. The last node
    is the edge node, with zero potential and zero density. The argument `q`
    is the edge exponent of the single component.
    """
    r = np.geomspace(1e-3, r_t, n)
    # A sphere of constant density rho_0 inside r_t, with p = 0 at r_t, in
    # the profile's units, where dM/dr = r**2 rho and dp/dr = -M / r**2.
    mass_t = 1.0
    rho_0 = 3.0 * mass_t / r_t**3
    mass = rho_0 * r**3 / 3.0
    p = rho_0 * (r_t**2 - r**2) / 6.0
    rho = np.full(n, rho_0)
    # The last node is the edge node: p = 0 and rho = 0 at r = r_t.
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
    # The factor 1.0137 moves the radii off the nodes.
    r = np.geomspace(1.1e-4, 0.9e4, 301) * 1.0137
    assert_allclose(profile.potential(r), HERNQUIST.p(r), rtol=1e-9)
    assert_allclose(profile.mass(r), HERNQUIST.mass(r), rtol=1e-9)
    assert_allclose(profile.density(r)[0], HERNQUIST.rho(r), rtol=1e-7)


def test_the_rows_are_reproduced_to_rounding():
    # The interpolants pass through the nodes. Taking the logarithm and the
    # exponential back costs a few units in the last place.
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
    # At and beyond the edge.
    assert profile.potential(profile.r_t) == 0.0
    assert np.all(profile.density(np.array([profile.r_t, 3.0]))[0] == 0.0)
    assert profile.mass(3.0) == profile.mass_t
    assert profile.radius(0.0) == profile.r_t
    assert profile.radius(-1.0) == profile.r_t
    # Inside half the edge radius, the interpolants give the constant-density
    # potential and mass to 1e-9. The test stops at half the edge radius,
    # because the logarithm of the potential steepens toward the edge, where
    # the nodes of this test profile are too coarse for 1e-9.
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
    # A spatially finite profile whose last node has a positive potential has
    # no edge node, and the interpolants reach the last node.
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
        # The radii lie inside the interior nodes, because beyond the last
        # interior node a profile with an edge node evaluates the Kepler form,
        # which the Hermite interpolant of the tables does not hold.
        r = np.geomspace(profile.r[1] * 1.0003, profile.r[n - 1] / 1.0003, 25)
        log_p, _ = hermite(np.log(r), tables.log_r, tables.log_p, tables.log_p_slope)
        assert_allclose(np.exp(log_p), profile.potential(r), rtol=1e-14)
        log_m, _ = hermite(
            np.log(r), tables.log_r_all, tables.log_mass, tables.log_mass_slope
        )
        assert_allclose(np.exp(log_m), profile.mass(r), rtol=1e-14)
        # The density spline's coefficients, highest power first, per interval.
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
    """Return the dataclass fields of a profile as a dictionary."""
    import dataclasses

    return {
        item.name: getattr(profile, item.name) for item in dataclasses.fields(profile)
    }
