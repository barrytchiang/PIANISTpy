"""Tests of the tangent rows of the second-moment tables.

A tangent row is the derivative of a second-moment table in an anisotropy
parameter, ``beta_c`` or ``r_a``, with the model's density profile held fixed.
The tests compare the analytic derivatives of the anisotropy factor with
central differences of the factor. The tests compare the tangent rows of
`dispersion_profile` and `projected_dispersion` with Richardson-extrapolated
central differences. The tests check that tangent rows asked of an
untruncated model, of a model whose profile moves with the anisotropy, or in
a parameter outside the family's parameters raise InputError.
"""

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
    r"""Return `anisotropy` with the field `parameter` increased by `step`.

    An isotropic anisotropy moved in ``beta_c`` becomes a constant anisotropy.
    The factor of the constant anisotropy at :math:`\beta_{\rm c} = 0` equals
    the factor of the isotropic anisotropy.
    """
    return dataclasses.replace(
        anisotropy, **{parameter: getattr(anisotropy, parameter) + step}
    )


@pytest.mark.parametrize("anisotropy", ANISOTROPIES, ids=str)
def test_the_factor_tangents_are_the_derivatives_of_the_definitions(anisotropy):
    """Assert that the analytic tangents of ln g and beta_sigma are derivatives.

    For each parameter in TANGENT_PARAMETERS of the anisotropy's family, at 4
    radii, the test asserts that
    `log_factor_tangent` agrees with the central difference of ln g, where
    g is ``Anisotropy.factor``. The test asserts that `beta_sigma_tangent`
    agrees with the central difference of `beta_sigma`. Both comparisons use
    ``rtol=1e-8`` and ``atol=1e-9``, with a difference step of
    :math:`10^{-5}`.
    """
    # A central difference of step 1e-5 carries a truncation error of order
    # 1e-10 and a relative rounding error of order 1e-11.
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
    """Assert InputError for a tangent in a parameter outside the family's set.

    The test asserts that `log_factor_tangent` of an isotropic anisotropy in
    ``r_a`` raises InputError, and that `beta_sigma_tangent` of a constant
    anisotropy in ``r_a`` raises InputError.
    """
    with pytest.raises(
        InputError, match=r"family isotropic are taken in the parameters \['beta_c'\]"
    ):
        log_factor_tangent(np.array([1.0]), Anisotropy(), "r_a")
    with pytest.raises(InputError, match="family constant are taken"):
        beta_sigma_tangent(np.array([1.0]), Anisotropy(beta_c=0.3), "r_a")


def test_an_untruncated_model_is_refused_with_cause():
    """Assert InputError for tangent rows of an untruncated model.

    The test asserts that `dispersion_profile` of the untruncated Hernquist
    model with ``tangents=("beta_c",)`` raises InputError with a message that
    states that the model is untruncated.
    """
    model = analytic_model("hernquist")
    with pytest.raises(InputError, match="is untruncated"):
        dispersion_profile(model, Settings(quick=True), tangents=("beta_c",))


def test_a_truncated_model_whose_profile_follows_its_df_is_refused_with_cause():
    """Assert InputError for tangent rows of a model truncated by its own DF.

    The DF of the lowered Hernquist model below the floor of its table
    follows the Taylor rule, not the edge law of a Method II model. The
    profile of the lowered
    Hernquist model follows the model's DF, so the profile moves with the
    anisotropy. The test asserts that `dispersion_profile` with
    ``tangents=("beta_c",)`` raises InputError with a message that gives the
    rule "taylor".
    """
    with pytest.raises(InputError, match="not a Method II model .* 'taylor'"):
        dispersion_profile(
            _lowered_hernquist(), Settings(quick=True), tangents=("beta_c",)
        )


def test_without_the_keyword_no_tangent_rows_are_held_or_asked():
    """Assert that tables computed without `tangents` hold no tangent rows.

    The test asserts that the ``tangents`` dictionary is empty on the result
    of `dispersion_profile` and on the result of `projected_dispersion`.
    """
    model = _lowered_hernquist()
    settings = Settings(quick=True)
    profile = dispersion_profile(model, settings)
    assert profile.tangents == {}
    projected = projected_dispersion(model, profile, settings)
    assert projected.tangents == {}


QUICK = Settings(quick=True)
# MEMBERS holds the anisotropies of two Method II models on the profile of the
# fixture `parent`: a Cuddeford-Osipkov-Merritt (COM) Type I model with
# beta_c = 0.25 and r_a = 5, and an Osipkov-Merritt (OM) Type I model with
# r_a = 5. STEPS holds the step h of the central differences in each parameter.
MEMBERS = {"COM_I": dict(beta_c=0.25, r_a=5.0), "OM_I": dict(r_a=5.0)}
STEPS = {"beta_c": 1e-3, "r_a": 1e-2}


@pytest.fixture(scope="module")
def parent():
    """Return the Hernquist model truncated by ``Truncation(k_t=0.05)``, solved
    with the quick settings."""
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    """Return a Method II model, a DF inverted on `parent`'s truncated profile.

    The keyword arguments `anisotropy` set the `Anisotropy` of the model.
    """
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


def _frozen(model, anisotropy, profile, projected):
    r"""Return the second moments of `model` under `anisotropy`, at frozen rows.

    The result is :math:`\sigma_{\rm r}^2` on the radii of `profile` and
    :math:`\sigma_{\rm los}^2` on the projected radii of `projected`. The
    rules and the node counts are those of `profile` and `projected`. The
    density profile of `model` and the edge limit of `profile` are held fixed,
    and only the anisotropy changes.
    """
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
    """Assert that the tangent rows are the derivatives at frozen rows.

    The test asserts that the DF of the model below the floor of its table
    follows the edge law.
    The test asserts that the tables computed with `tangents` equal, bit for
    bit, the tables computed without `tangents`. For each parameter, the test
    asserts that the edge limit of the model is 2/7 and that the edge limit
    of a model re-inverted with the parameter moved equals 2/7, both to a
    relative difference of :math:`10^{-15}`. For each parameter, the test
    asserts that each tangent row lies within 3 times the Richardson error
    estimate, plus :math:`10^{-9}` times the largest extrapolated value, of
    the Richardson extrapolation of central differences at the steps h and
    h/2.
    """
    theta = MEMBERS[family]
    model = _child(parent, **theta)
    assert model.form.below_floor[0] == "edge_law"
    names = TANGENT_PARAMETERS[family]
    profile = dispersion_profile(model, QUICK, tangents=names)
    projected = projected_dispersion(model, profile, QUICK, tangents=names)
    # The tables without the keyword tangents must equal the tables with the
    # keyword tangents bit for bit.
    plain = dispersion_profile(model, QUICK)
    plain_projected = projected_dispersion(model, plain, QUICK)
    assert plain.sigma_r.tobytes() == profile.sigma_r.tobytes()
    assert plain_projected.r_2d.tobytes() == projected.r_2d.tobytes()
    assert plain_projected.sigma_los.tobytes() == projected.sigma_los.tobytes()
    anisotropy = model.components[0].anisotropy
    for name in names:
        # `_frozen` holds the edge limit of `profile` fixed. The edge limit of a
        # model re-inverted with the parameter moved must therefore equal the
        # edge limit 2/7 of the model. An OM model moved in beta_c is the COM
        # model of the same type.
        moved = _child(
            parent, **dict(theta, **{name: theta.get(name, 0.0) + STEPS[name]})
        )
        assert edge_limit(moved) == pytest.approx(edge_limit(model), rel=1e-15)
        assert edge_limit(model) == pytest.approx(2.0 / 7.0, rel=1e-15)
        # The test takes the central differences D(h) and D(h/2) at frozen
        # rows. The Richardson extrapolation of D(h) and D(h/2) is
        # (4 D(h/2) - D(h)) / 3. The error estimate of the extrapolation is
        # |extrapolation - D(h/2)|.
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
    """Assert InputError for tangent rows the profile does not hold or allow.

    The test asserts that `projected_dispersion` with ``tangents=("r_a",)``
    raises InputError when the profile from `dispersion_profile` holds no
    tangent rows. The test asserts that `dispersion_profile` of an OM Type I
    model with ``tangents=("q",)`` raises InputError, since the OM Type I
    tangent rows are taken in ``beta_c`` and ``r_a`` only.
    """
    model = _child(parent, r_a=5.0)
    profile = dispersion_profile(model, QUICK)
    with pytest.raises(InputError, match=r"holds no tangents in \['r_a'\]"):
        projected_dispersion(model, profile, QUICK, tangents=("r_a",))
    with pytest.raises(
        InputError, match=r"family OM_I are taken in the parameters \['beta_c', 'r_a'\]"
    ):
        dispersion_profile(model, QUICK, tangents=("q",))
