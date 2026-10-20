"""Tests of the tangent rows of the second-moment tables in the anisotropy
parameters, on a fixed profile: the analytic factors against differences of
their definitions, and the refusals of models whose profile is not fixed."""

import dataclasses

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
    compute_df,
    solve,
    truncate,
)
from pianistpy.df.analytic import analytic_model
from pianistpy.observables._dispersion import (
    TANGENT_PARAMETERS,
    beta_sigma,
    beta_sigma_tangent,
    log_factor_tangent,
    radial_dispersion,
    radial_integral_reader,
)
from pianistpy.observables._projection import project
from pianistpy.observables._tables import edge_limit
from pianistpy.observables.stages import dispersion_profile, projected_dispersion
from pianistpy.particles.tests.test_stages import _lowered_hernquist

ANISOTROPIES = [
    Anisotropy(),
    Anisotropy(beta_c=0.3),
    Anisotropy(r_a=2.0),
    Anisotropy(r_a=2.0, kind="II"),
    Anisotropy(beta_c=-0.25, r_a=2.0),
    Anisotropy(beta_c=0.25, r_a=2.0, kind="II"),
]


def _moved(anisotropy, parameter, step):
    r"""The anisotropy with one parameter moved (the isotropic model becomes a
    constant-anisotropy one, whose factor it is at :math:`\beta_{\rm c} = 0`)."""
    return dataclasses.replace(
        anisotropy, **{parameter: getattr(anisotropy, parameter) + step}
    )


@pytest.mark.parametrize("anisotropy", ANISOTROPIES, ids=str)
def test_the_factor_tangents_are_the_derivatives_of_the_definitions(anisotropy):
    # ln g (Anisotropy.factor) and beta_sigma by central differences of step
    # 1e-5 (truncation 1e-10, rounding 1e-11 relative)
    r = np.array([0.05, 0.5, 1.0, 1.5])
    step = 1e-5
    for parameter in TANGENT_PARAMETERS[anisotropy.family]:
        plus, minus = (_moved(anisotropy, parameter, s) for s in (step, -step))
        d_ln_g = (np.log(plus.factor(r)) - np.log(minus.factor(r))) / (2.0 * step)
        d_beta = (beta_sigma(r, plus) - beta_sigma(r, minus)) / (2.0 * step)
        np.testing.assert_allclose(
            log_factor_tangent(r, anisotropy, parameter), d_ln_g, rtol=1e-8, atol=1e-9
        )
        np.testing.assert_allclose(
            beta_sigma_tangent(r, anisotropy, parameter), d_beta, rtol=1e-8, atol=1e-9
        )


def test_a_parameter_the_family_does_not_vary_in_is_refused():
    with pytest.raises(
        InputError, match=r"isotropic model's tangent rows are taken in \['beta_c'\]"
    ):
        log_factor_tangent(np.array([1.0]), Anisotropy(), "r_a")
    with pytest.raises(InputError, match="constant model's tangent rows"):
        beta_sigma_tangent(np.array([1.0]), Anisotropy(beta_c=0.3), "r_a")


def test_an_untruncated_model_is_refused_by_name():
    model = analytic_model("hernquist")
    with pytest.raises(InputError, match="it is untruncated"):
        dispersion_profile(model, Settings(quick=True), tangents=("beta_c",))


def test_a_truncated_model_whose_profile_follows_its_df_is_refused_by_name():
    # the lowered Hernquist model is truncated by its DF (the Taylor rule):
    # its profile moves with the anisotropy, so no tangent rows are taken
    with pytest.raises(InputError, match="not a Method II model .* 'taylor'"):
        dispersion_profile(
            _lowered_hernquist(), Settings(quick=True), tangents=("beta_c",)
        )


def test_without_the_keyword_no_tangent_rows_are_held_or_asked():
    model = _lowered_hernquist()
    settings = Settings(quick=True)
    profile = dispersion_profile(model, settings)
    assert profile.tangents == {}
    projected = projected_dispersion(model, profile, settings)
    assert projected.tangents == {}


QUICK = Settings(quick=True)
# The members of the likelihood's own kind: Method II children of a truncated
# parent, a COM Type I one and an OM Type I one.
MEMBERS = {"COM_I": dict(beta_c=0.25, r_a=5.0), "OM_I": dict(r_a=5.0)}
STEPS = {"beta_c": 1e-3, "r_a": 1e-2}


@pytest.fixture(scope="module")
def parent():
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    """A Method II model: a DF inverted on `parent`'s truncated profile."""
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


def _frozen(model, anisotropy, profile, projected):
    r""":math:`\sigma_r^2` on `profile`'s rows and :math:`\sigma_{\rm los}^2` on
    `projected`'s rows, by the same rules and nodes, under another anisotropy on
    the same profile and with the edge limit held."""
    pair = model.profile

    def density(r):
        return pair.density(r)[0]

    sigma_r2, _ = radial_dispersion(
        profile.r,
        pair.potential,
        density,
        pair.radius,
        anisotropy,
        float(pair.p_trunc[-1]),
        0.0,
        int(profile.certificate["n_jeans_nodes"]),
        knots=pair.p_trunc,
    )
    read = radial_integral_reader(
        profile.r,
        np.sqrt(sigma_r2) ** 2,
        pair.potential,
        lambda r: density(r) / anisotropy.factor(r),
        True,
        profile.edge_limit[0],
    )
    _, sigma_los2 = project(
        projected.r_2d,
        density,
        read,
        float(pair.r_table_max),
        anisotropy,
        int(QUICK.observables["n_projection_nodes"]),
    )
    return sigma_r2, sigma_los2


@pytest.mark.parametrize("family", MEMBERS)
def test_the_tangent_rows_are_the_derivatives_at_frozen_placement(parent, family):
    theta = MEMBERS[family]
    model = _child(parent, **theta)
    assert model.form.below_floor[0] == "edge_law"
    names = TANGENT_PARAMETERS[family]
    profile = dispersion_profile(model, QUICK, tangents=names)
    projected = projected_dispersion(model, profile, QUICK, tangents=names)
    # without the keyword, the same tables bit for bit
    plain = dispersion_profile(model, QUICK)
    plain_projected = projected_dispersion(model, plain, QUICK)
    assert plain.sigma_r.tobytes() == profile.sigma_r.tobytes()
    assert plain_projected.r_2d.tobytes() == projected.r_2d.tobytes()
    assert plain_projected.sigma_los.tobytes() == projected.sigma_los.tobytes()
    anisotropy = model.components[0].anisotropy
    for name in names:
        # the edge limit a re-inverted member reads does not move (2/7)
        # (an OM member moved in beta_c is the COM member of its type)
        moved = _child(
            parent, **dict(theta, **{name: theta.get(name, 0.0) + STEPS[name]})
        )
        assert edge_limit(moved) == pytest.approx(edge_limit(model), rel=1e-15)
        assert edge_limit(model) == pytest.approx(2.0 / 7.0, rel=1e-15)
        # central differences of steps h and h/2 at frozen placement, and
        # their Richardson extrapolation, against the tangent rows within the
        # extrapolation's own error times 3
        differences = []
        for step in (STEPS[name], 0.5 * STEPS[name]):
            plus, minus = (
                _frozen(model, _moved(anisotropy, name, s), profile, projected)
                for s in (step, -step)
            )
            differences.append(
                [(a - b) / (2.0 * step) for a, b in zip(plus, minus, strict=True)]
            )
        tangents = (profile.tangents[name][0], projected.tangents[name][0])
        for k, tangent in enumerate(tangents):
            coarse, fine = differences[0][k], differences[1][k]
            extrapolated = (4.0 * fine - coarse) / 3.0
            scale = np.max(np.abs(extrapolated))
            error = 3.0 * np.abs(extrapolated - fine) + 1e-9 * scale
            assert np.all(np.abs(tangent - extrapolated) <= error), (name, k)


def test_the_projection_asks_the_profile_for_its_tangent_rows(parent):
    model = _child(parent, r_a=5.0)
    profile = dispersion_profile(model, QUICK)
    with pytest.raises(InputError, match=r"holds no tangent rows in \['r_a'\]"):
        projected_dispersion(model, profile, QUICK, tangents=("r_a",))
    with pytest.raises(
        InputError, match=r"OM_I model's tangent rows are taken in \['beta_c', 'r_a'\]"
    ):
        dispersion_profile(model, QUICK, tangents=("q",))
