"""Tests of the local speed distribution (`pdf_v`): its angle integral against
closed forms and an independent quadrature in the angle, its certificates on
models of every family, a Type II model's speeds beyond escape, the public
call across unit systems and a superposition's mixture."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
    superpose,
)
from pianistpy.df import compute_df, truncate
from pianistpy.df.analytic import analytic_model
from pianistpy.df.lookup import log10_df
from pianistpy.observables import pdf_v
from pianistpy.observables._speed import angle_integral, speed_breaks, speed_model
from pianistpy.observables.stages import dispersion_profile, speed_distribution
from pianistpy.superposition.tests.test_core import _member
from pianistpy.validation import GOAL

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    return dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)


@pytest.fixture(scope="module")
def families():
    r"""Hernquist models of each family: isotropic (closed form), OM I and
    COM I (untruncated) and OM II with :math:`R_{\rm a} = 100`, truncated at
    :math:`\mathcal{K}_{\rm t} = 0.05` above :math:`P(R_{\rm a}) = 0.0099`
    (the domain gate), its truncation radius 23.3."""
    hernquist = DoublePowerLaw.hernquist()

    def built(anisotropy, truncation=None):
        df = compute_df(Component(hernquist, anisotropy, name="h"))
        return dataclasses.replace(truncate(df, truncation), settings=SETTINGS)

    return {
        "isotropic": dataclasses.replace(
            analytic_model("hernquist"), settings=SETTINGS
        ),
        "OM_I": built(Anisotropy(r_a=1.0)),
        "COM_I": built(Anisotropy(beta_c=0.25, r_a=1.0)),
        "OM_II": built(Anisotropy(r_a=100.0, kind="II"), Truncation(k_t=0.05)),
    }


def _f(sm, model, q):
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    log_f = np.asarray(log10_df(model.table, model.form, np.maximum(q, 0.0)))
    return np.where((q > sm.k_c) & np.isfinite(log_f), 10.0**log_f, 0.0)


@pytest.mark.parametrize("b", [0.0, 0.25, 0.5, -0.5])
def test_without_an_anisotropy_radius_the_angle_integral_is_the_closed_form(plummer, b):
    # Q does not depend on the angle: the integral is f(Psi - V^2/2) times
    # INT_0^1 (1 - t^2)^(-b) dt = sqrt(pi) Gamma(1 - b) / (2 Gamma(3/2 - b))
    sm = dataclasses.replace(speed_model(plummer), b=b)
    psi = float(plummer.profile.potential(np.array([1.0]))[0])
    v = np.linspace(0.0, math.sqrt(2.0 * psi), 41)[1:-1]
    exact = (
        _f(sm, plummer, psi - 0.5 * v * v)
        * math.sqrt(math.pi)
        * gamma(1.0 - b)
        / (2.0 * gamma(1.5 - b))
    )
    np.testing.assert_allclose(
        angle_integral(sm, psi, 0.0, v, 16, 16), exact, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("b", "sign", "r_a", "r"),
    [
        (0.0, 1.0, 1.0, 1.0),
        (0.25, 1.0, 1.0, 1.0),
        (-0.25, 1.0, 1.0, 2.0),
        (0.0, -1.0, 3.0, 1.0),
        (-0.25, -1.0, 3.0, 2.0),
    ],
)
def test_the_angle_integral_is_an_independent_integral_in_the_angle(
    plummer, b, sign, r_a, r
):
    # the same integral in eta, sin^(1-2b)(eta) f(Q(eta)), by adaptive
    # quadrature with the support's edge as a break, on Plummer's DF read
    # under each anisotropy (the integral does not ask that the DF be the
    # anisotropy's own)
    sm = dataclasses.replace(speed_model(plummer), b=b, sign=sign, r_a=r_a)
    psi = float(plummer.profile.potential(np.array([r]))[0])
    xi = (r / r_a) ** 2
    _, v_kink, v_max = speed_breaks(sm, psi, xi)
    v = np.concatenate(
        [np.linspace(0.02, 0.98, 13) * v_max, v_kink * (1.0 + np.array([-1e-3, 1e-3]))]
    )
    v = v[v < v_max]

    def reference(vj):
        def integrand(eta):
            s = math.sin(eta)
            q = psi - 0.5 * vj * vj * (1.0 + sign * xi * s * s)
            return s ** (1.0 - 2.0 * b) * float(_f(sm, plummer, q)[0])

        edge = 1.0 + 2.0 * sm.k_c / (vj * vj) - 2.0 * psi / (vj * vj)
        points = []
        if xi > 0.0:
            y = -edge / (sign * xi)
            if 0.0 < y < 1.0:
                points.append(math.asin(math.sqrt(y)))
        return quad(
            integrand,
            0.0,
            0.5 * math.pi,
            points=points or None,
            epsabs=0.0,
            epsrel=1e-12,
            limit=400,
        )[0]

    got = angle_integral(sm, psi, xi, v, 16, 16)
    expected = np.array([reference(vj) for vj in v])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9 * np.max(expected))


@pytest.mark.parametrize("family", ["isotropic", "OM_I", "COM_I", "OM_II"])
def test_the_distribution_holds_its_norm_and_second_moment(families, family):
    # the norm against the profile's density and the second moment against
    # the Jeans dispersion, sigma_r^2 (3 - 2 beta): two routes that share only
    # the model
    model = families[family]
    profile = dispersion_profile(model, SETTINGS)
    r_t = float(model.profile.r_table_max)
    for r in (0.1, 1.0, min(10.0, 0.9 * r_t)):
        result = speed_distribution(model, r, SETTINGS, profile)
        assert abs(result.record["norm"]) < GOAL
        assert abs(result.record["second_moment"]) < GOAL
        assert result.record["halving"] < GOAL
        assert result.record["tier"] == "goal"
        np.testing.assert_allclose(result.read(result.v_nodes), result.pdf, rtol=1e-14)


def test_a_type_ii_distribution_reaches_beyond_the_escape_speed(families):
    # Q_+ = Z + L^2 / (2 r_a^2) above its floor: speeds up to
    # V_esc / sqrt(1 - r^2 / r_a^2), the share beyond V_esc unbound
    model = families["OM_II"]
    r = 0.9 * float(model.profile.r_table_max)
    result = speed_distribution(model, r, SETTINGS)
    xi = (r / 100.0) ** 2
    assert result.v_max == pytest.approx(result.v_esc / math.sqrt(1.0 - xi), rel=1e-12)
    assert result.unbound_fraction > 0.0
    assert np.all(result.read(np.array([1.0001, 1.5]) * result.v_max) == 0.0)
    assert result.read(np.array([0.5 * (result.v_esc + result.v_max)]))[0] > 0.0
    for isotropic in ("isotropic", "OM_I", "COM_I"):
        other = speed_distribution(families[isotropic], 1.0, SETTINGS)
        assert other.v_max == pytest.approx(other.v_esc, rel=1e-12)
        assert other.unbound_fraction == 0.0


def test_the_unbound_fraction_is_an_independent_integral_beyond_escape(families):
    # the DF's share with Z < 0 at a radius, by nested adaptive quadrature in
    # (V, eta) over its own support split at Z = 0 (V = V_esc), against the
    # record's unbound fraction: no rule of the stage is shared
    model = families["OM_II"]
    r = 0.9 * float(model.profile.r_table_max)
    result = speed_distribution(model, r, SETTINGS)
    sm = speed_model(model)
    psi = float(model.profile.potential(np.array([r]))[0])
    rho = float(np.atleast_2d(model.profile.density(np.array([r])))[0, 0])
    xi = (r / 100.0) ** 2

    def inner(v):
        def integrand(eta):
            s = math.sin(eta)
            q = psi - 0.5 * v * v * (1.0 - xi * s * s)
            return s * float(_f(sm, model, q)[0])

        # the support Q > 0 holds where sin^2(eta) > (1 - 2 psi / v^2) / xi
        y = (1.0 - 2.0 * psi / (v * v)) / xi
        lo = math.asin(math.sqrt(min(max(y, 0.0), 1.0)))
        return 2.0 * quad(integrand, lo, 0.5 * math.pi, epsabs=0.0, epsrel=1e-12)[0]

    beyond = quad(
        lambda v: v * v * inner(v),
        result.v_esc,
        result.v_max,
        epsabs=0.0,
        epsrel=1e-10,
        limit=200,
    )[0]
    unbound = 2.0 * math.pi * beyond / rho
    assert result.unbound_fraction > 1e-4
    assert result.unbound_fraction == pytest.approx(unbound, rel=1e-6)


def test_pdf_v_reads_the_stage_in_each_unit_system(families):
    model = families["OM_I"]
    v = np.linspace(0.0, 1.4, 8)
    stage = speed_distribution(model, 1.0, SETTINGS)
    got = pdf_v(model, 1.0, v)
    assert got.shape == v.shape
    np.testing.assert_allclose(got, stage.read(v), rtol=1e-14)
    # r and v broadcast
    assert pdf_v(model, np.array([[0.5], [1.0]]), v).shape == (2, v.size)
    # in the physical system: r in parsecs, v in km/s, a density per km/s
    physical = model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    kms = _kms(physical)
    per_kms = pdf_v(physical, R_0, v * kms)
    np.testing.assert_allclose(
        np.asarray(getattr(per_kms, "value", per_kms)) * kms, got, rtol=1e-12
    )


def test_pdf_v_from_two_angle_nodes_agrees_with_the_default(families):
    # the angle rule starts at two nodes and doubles to its tolerance
    model = families["OM_I"]
    coarse = dataclasses.replace(
        model,
        settings=Settings(
            observables={
                "n_radii": 400,
                "n_jeans_nodes": 200,
                "n_projection_nodes": 200,
                "n_speed_angle_nodes": 2,
            }
        ),
    )
    v = np.linspace(0.0, 1.4, 8)
    want = pdf_v(model, 1.0, v)
    np.testing.assert_allclose(
        pdf_v(coarse, 1.0, v), want, rtol=GOAL, atol=GOAL * float(np.max(want))
    )


def _kms(model):
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return float(getattr(one, "value", one))


def test_a_superposition_mixes_by_the_density_at_the_radius():
    # the truncated Hernquist model and the same with its DF table on every
    # other energy, a grid of its own (as the superposed observables' tests
    # take them): each holds its own norm, so the mixture is read as built
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))
    parent = dataclasses.replace(
        _member(truncate(df, Truncation(k_t=0.05)), 0.0), settings=SETTINGS
    )
    t = parent.table
    keep = np.unique(np.r_[np.arange(0, t.k.shape[1], 2), t.k.shape[1] - 1])
    thinned = dataclasses.replace(
        parent,
        table=dataclasses.replace(
            t,
            k=t.k[:, keep],
            log_df=t.log_df[:, keep],
            log_df_slope=t.log_df_slope[:, keep],
            i_last=(keep.size - 1,),
        ),
    )
    members = [parent, thinned]
    weights = (0.4, 0.6)
    model = superpose(dict(zip(members, weights, strict=True)))
    r, v = 0.5, np.linspace(0.05, 1.2, 9)
    parts = [pdf_v(m, r, v) for m in members]
    rho = [
        float(np.atleast_2d(m.profile.density(np.array([r])))[0, 0]) for m in members
    ]
    w = np.asarray(weights) / sum(weights)
    expected = sum(wk * dk * pk for wk, dk, pk in zip(w, rho, parts, strict=True))
    expected = expected / sum(wk * dk for wk, dk in zip(w, rho, strict=True))
    np.testing.assert_allclose(pdf_v(model, r, v), expected, rtol=1e-12)


def test_the_call_refuses_what_it_does_not_read(plummer):
    with pytest.raises(InputError, match="pdf_v reads a Model or a LosTable"):
        pdf_v("plummer", 1.0, 0.5)
    with pytest.raises(InputError, match="speed distribution is read at a radius"):
        speed_distribution(plummer, 0.0, SETTINGS)
