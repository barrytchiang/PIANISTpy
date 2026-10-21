"""Tests of the local velocity distribution, computed by pdf_v and by
observables.stages.velocity_distribution.

The tests compare the angle integral with a closed form and with an adaptive
quadrature in the angle, check the validation records of velocity_distribution on
Hernquist models of each anisotropy family, check a Type II model's velocities above
the escape velocity, check pdf_v across unit systems, and check the velocity
distribution of a superposition as the density-weighted mixture of its members.
"""

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
from pianistpy.observables._velocity import (
    angle_integral,
    velocity_breaks,
    velocity_model,
)
from pianistpy.observables.stages import dispersion_profile, velocity_distribution
from pianistpy.superposition.tests.test_core import _member
from pianistpy.validation import TIER_BOUNDS

pytestmark = pytest.mark.slow

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    return dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)


@pytest.fixture(scope="module")
def families():
    r"""Return Hernquist models of four anisotropy families, keyed by family.

    The models are the isotropic model from its closed-form DF, the untruncated
    Osipkov-Merritt (OM) Type I model with :math:`R_{\rm a} = 1`, the untruncated
    Cuddeford-Osipkov-Merritt (COM) Type I model with
    :math:`\beta_{\rm c} = 0.25` and :math:`R_{\rm a} = 1`, and the OM Type II
    model with :math:`R_{\rm a} = 100` truncated at
    :math:`\mathcal{K}_{\rm t} = 0.05`. Every model carries SETTINGS.
    """
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
    # Without an anisotropy radius (xi = 0) the DF argument Q does not depend on
    # the angle variable t. The angle integral is then f(Psi - V^2/2) times
    # INT_0^1 (1 - t^2)^(-b) dt = sqrt(pi) Gamma(1 - b) / (2 Gamma(3/2 - b)),
    # with Psi the potential at r = 1, V the velocity and b the L-factor exponent.
    sm = dataclasses.replace(velocity_model(plummer), b=b)
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
    # The reference integrates sin^(1-2b)(eta) f(Q(eta)) over the angle eta
    # from 0 to pi/2 by adaptive quadrature, with the edge of the DF's support
    # as a break point. The DF is the Plummer model's DF under each anisotropy
    # of the parameters, because the angle integral does not require the DF to
    # belong to that anisotropy. angle_integral agrees with the reference to an
    # absolute tolerance of 1e-9 times the reference's maximum.
    sm = dataclasses.replace(velocity_model(plummer), b=b, sign=sign, r_a=r_a)
    psi = float(plummer.profile.potential(np.array([r]))[0])
    xi = (r / r_a) ** 2
    _, v_kink, v_max = velocity_breaks(sm, psi, xi)
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
    # The record's "norm" compares the distribution's norm with the profile's
    # density. The record's "second_moment" compares the distribution's second
    # moment with the Jeans dispersion sigma_r^2 (3 - 2 beta). Each comparison
    # uses two computations that share only the model. Each record entry is
    # below 1e-6, the tier is "silent", and the reader reproduces the stored
    # values at the velocity nodes to rtol = 1e-14.
    model = families[family]
    profile = dispersion_profile(model, SETTINGS)
    r_t = float(model.profile.r_table_max)
    for r in (0.1, 1.0, min(10.0, 0.9 * r_t)):
        result = velocity_distribution(model, r, SETTINGS, profile)
        assert abs(result.record["norm"]) < TIER_BOUNDS[0]
        assert abs(result.record["second_moment"]) < TIER_BOUNDS[0]
        assert result.record["halving"] < TIER_BOUNDS[0]
        assert result.record["tier"] == "silent"
        np.testing.assert_allclose(result.read(result.v_nodes), result.pdf, rtol=1e-14)


def test_a_type_ii_distribution_reaches_beyond_the_escape_velocity(families):
    # A Type II model's DF argument is Q_+ = Z + L^2 / (2 r_a^2), with Z = Psi - V^2/2
    # the relative energy. Q_+ above the DF's floor allows velocities up to
    # V_esc / sqrt(1 - r^2 / r_a^2), with V_esc the escape velocity at radius r.
    # The fraction of the distribution above V_esc is unbound and positive.
    # The isotropic, OM Type I and COM Type I models have v_max = V_esc and a
    # zero unbound fraction.
    model = families["OM_II"]
    r = 0.9 * float(model.profile.r_table_max)
    result = velocity_distribution(model, r, SETTINGS)
    xi = (r / 100.0) ** 2
    assert result.v_max == pytest.approx(result.v_esc / math.sqrt(1.0 - xi), rel=1e-12)
    assert result.unbound_fraction > 0.0
    assert np.all(result.read(np.array([1.0001, 1.5]) * result.v_max) == 0.0)
    assert result.read(np.array([0.5 * (result.v_esc + result.v_max)]))[0] > 0.0
    for isotropic in ("isotropic", "OM_I", "COM_I"):
        other = velocity_distribution(families[isotropic], 1.0, SETTINGS)
        assert other.v_max == pytest.approx(other.v_esc, rel=1e-12)
        assert other.unbound_fraction == 0.0


def test_the_unbound_fraction_is_an_independent_integral_beyond_escape(families):
    # The reference is the fraction of the DF with relative energy Z < 0 at the
    # radius, by nested adaptive quadrature in the velocity V from V_esc to v_max
    # and the angle eta over the DF's support. The reference uses no quadrature
    # rule of velocity_distribution. The unbound fraction exceeds 1e-4 and agrees
    # with the reference to rtol = 1e-6.
    model = families["OM_II"]
    r = 0.9 * float(model.profile.r_table_max)
    result = velocity_distribution(model, r, SETTINGS)
    sm = velocity_model(model)
    psi = float(model.profile.potential(np.array([r]))[0])
    rho = float(np.atleast_2d(model.profile.density(np.array([r])))[0, 0])
    xi = (r / 100.0) ** 2

    def inner(v):
        def integrand(eta):
            s = math.sin(eta)
            q = psi - 0.5 * v * v * (1.0 - xi * s * s)
            return s * float(_f(sm, model, q)[0])

        # The DF's support Q > 0 is where sin^2(eta) > (1 - 2 psi / v^2) / xi.
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
    stage = velocity_distribution(model, 1.0, SETTINGS)
    got = pdf_v(model, 1.0, v)
    assert got.shape == v.shape
    np.testing.assert_allclose(got, stage.read(v), rtol=1e-14)
    # pdf_v broadcasts r of shape (2, 1) against v of shape (8,).
    assert pdf_v(model, np.array([[0.5], [1.0]]), v).shape == (2, v.size)
    # In the physical system pdf_v takes r in parsecs and v in km/s, and
    # returns a probability density per km/s.
    physical = model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    kms = _kms(physical)
    per_kms = pdf_v(physical, R_0, v * kms)
    np.testing.assert_allclose(
        np.asarray(getattr(per_kms, "value", per_kms)) * kms, got, rtol=1e-12
    )


def test_pdf_v_from_two_angle_nodes_agrees_with_the_default(families):
    # With n_velocity_angle_nodes = 2 the angle rule starts at two nodes per panel
    # and doubles the node count, at most VELOCITY_DOUBLINGS times, until the
    # change of the distribution is below the rule's tolerance.
    # The result agrees with the default rule's to rtol = 1e-6 and atol = 1e-6
    # times the peak.
    model = families["OM_I"]
    coarse = dataclasses.replace(
        model,
        settings=Settings(
            observables={
                "n_radii": 400,
                "n_jeans_nodes": 200,
                "n_projection_nodes": 200,
                "n_velocity_angle_nodes": 2,
            }
        ),
    )
    v = np.linspace(0.0, 1.4, 8)
    want = pdf_v(model, 1.0, v)
    np.testing.assert_allclose(
        pdf_v(coarse, 1.0, v),
        want,
        rtol=TIER_BOUNDS[0],
        atol=TIER_BOUNDS[0] * float(np.max(want)),
    )


def _kms(model):
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return float(getattr(one, "value", one))


def test_a_superposition_mixes_by_the_density_at_the_radius():
    # The members are the truncated isotropic Hernquist model and a copy whose
    # DF table keeps every other energy (and the last), an energy grid of its
    # own. Each member's velocity distribution is normalized on its own grid. The
    # superposition's pdf_v equals, to rtol = 1e-12, the sum of the members'
    # pdf_v weighted by weight times density at the radius, divided by the sum
    # of weight times density.
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
    with pytest.raises(InputError, match="pdf_v must be a Model or a LosTable"):
        pdf_v("plummer", 1.0, 0.5)
    with pytest.raises(InputError, match="velocity distribution is read at a radius"):
        velocity_distribution(plummer, 0.0, SETTINGS)
