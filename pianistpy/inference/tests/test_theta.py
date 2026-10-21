r"""Tests of the log-likelihood as a function of the anisotropy parameters
(:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) of a free component: the value at
the component's own anisotropy, the gradient at a frozen placement of the
tabulated radii against central differences on the frozen radii, and the
exceptions for components that cannot be free."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from pianistpy.inference import Data, likelihood
from pianistpy.observables._dispersion import (
    radial_dispersion,
    radial_integral_reader,
)
from pianistpy.observables._projection import dispersion_table_reader, project
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import dispersion_profile, projected_dispersion

pytestmark = pytest.mark.slow

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
THETA = (-0.2, 1.5)
BETA_CONSTANT = 0.25
RADII = R_0 * np.geomspace(0.05, 0.5, 6)
APERTURE = Aperture.circle(0.3 * R_0)


def _physical(model):
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


@pytest.fixture(scope="module")
def case():
    """Return two Method II components on the truncated profile of the
    Hernquist model, one isotropic and one Cuddeford-Osipkov-Merritt (COM)
    of Type I, superposed; the truncated Hernquist parent; and data moved by
    3 percent from the isotropic component's own dispersion."""
    parent = truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.05),
    )
    models = [
        _physical(truncate(compute_df(Component(profile=parent.profile, anisotropy=a))))
        for a in (Anisotropy(), Anisotropy(beta_c=THETA[0], r_a=THETA[1]))
    ]
    model = superpose({models[0]: 0.4, models[1]: 0.6})
    own = projected_dispersion(
        models[0], dispersion_profile(models[0], SETTINGS), SETTINGS
    )
    v = float(
        models[0].units.factors(models[0].components[0].unit_profile).physical_v_kms
    )
    sigma = v * own.read(RADII / R_0)[1][0]
    data = [
        Data(
            "sigma_los",
            sigma * (1.0 + 0.03 * np.cos(np.arange(6))),
            r_2d=RADII,
            errors=0.02 * sigma,
            unit_system="physical",
        ),
        Data(
            "sigma_los_aperture",
            np.array([1.02 * sigma[3]]),
            apertures=(APERTURE,),
            errors=np.array([0.02 * sigma[3]]),
            unit_system="physical",
        ),
    ]
    return model, _physical(parent), data


def test_the_value_at_its_own_theta_is_the_fixed_likelihood(case):
    model, _, data = case
    fixed = likelihood(model, data, SETTINGS)
    function = likelihood(model, data, SETTINGS, free=(1,))
    np.testing.assert_array_equal(np.asarray(function.theta), [THETA])
    assert float(function()) == float(fixed())
    assert float(function(theta=[THETA])) == float(fixed())
    w = np.array([0.3, 0.7])
    np.testing.assert_array_equal(
        np.asarray(jax.grad(lambda x: function(weights=x))(w)),
        np.asarray(jax.grad(lambda x: fixed(weights=x))(w)),
    )


def test_the_gradient_in_theta_at_frozen_placement(case):
    # At an anisotropy other than the component's own, the component's DF is
    # inverted at the new anisotropy. The JAX gradient is compared with
    # Richardson-extrapolated central differences of the likelihood on the
    # tabulated radii placed at the new anisotropy, with steps 4h and 2h for
    # the step h of the tangent tables, and must agree within three times
    # the differences' own error estimate.
    model, _, data = case
    function = likelihood(model, data, SETTINGS, free=(1,))
    _check_against_frozen(function, np.array([THETA[0] + 0.03, THETA[1] * 1.1]))


def _check_against_frozen(function, theta):
    gradient = np.asarray(jax.grad(lambda t: function(theta=t))(theta[None, :]))[0]
    free = function.members[0]
    breaks, _, valid = free.forward(theta)
    assert valid == 1.0
    r_2d = free.tables(theta)[1].r_2d

    def frozen(at):
        c = dispersion_table_reader(r_2d, _frozen(free, theta, at))._spline.c
        coefficients = np.zeros((4, free.bucket))
        coefficients[:, : c.shape[1]] = c
        fixed = dataclasses.replace(
            function,
            readings=(lambda t: (jnp.asarray(breaks), jnp.asarray(coefficients), 1.0),),
        )
        return float(fixed(theta=theta[None, :]))

    for i in range(2):
        h = 0.001 * (1.0 if i == 0 else theta[1])
        e = np.zeros(2)
        e[i] = 1.0

        def difference(step, e=e):
            return (frozen(theta + step * e) - frozen(theta - step * e)) / (2.0 * step)

        d_4, d_2 = difference(4.0 * h), difference(2.0 * h)
        reference = d_2 + (d_2 - d_4) / 3.0
        error = abs(d_2 - d_4) / 3.0
        assert abs(gradient[i] - reference) <= 3.0 * error + 1e-9 * abs(reference)


def _frozen(free, theta, at):
    r"""Return the reference :math:`\sigma_{\rm los}^2` at the anisotropy
    `at` on the tabulated radii of the free component's table at `theta`,
    with the placement of the radii frozen: the radial quadrature rule on the
    table's radii and nodes, read as the table reads the radii with the
    table's edge limit held, and projected on the table's projected radii."""
    profile, projected = free.tables(theta)
    model = profile.model
    rows = np.asarray(profile.r)
    anisotropy = Anisotropy(
        beta_c=float(at[0]),
        r_a=float(at[1]),
        kind=model.components[0].anisotropy.kind,
    )
    truncated = model.profile

    def density(r):
        return truncated.density(r)[0]

    sigma_r2 = radial_dispersion(
        rows,
        truncated.potential,
        density,
        truncated.radius,
        anisotropy,
        float(truncated.p_trunc[-1]),
        0.0,
        int(profile.certificate["n_jeans_nodes"]),
        knots=truncated.p_trunc,
        panel_cache={},
    )[0]
    read = radial_integral_reader(
        rows,
        sigma_r2,
        truncated.potential,
        lambda r: density(r) / anisotropy.factor(r),
        True,
        profile.edge_limit[0],
    )
    return project(
        projected.r_2d,
        density,
        read,
        float(truncated.r_table_max),
        anisotropy,
        int(free.settings.observables["n_projection_nodes"]),
    )[1]


def test_at_beta_c_zero_the_gradient_in_beta_c(case):
    # At beta_c = 0 the anisotropy model is Osipkov-Merritt (OM), and the
    # anisotropy function g(r; beta_c, R_a) is smooth in beta_c across 0
    # whatever name the model carries. The derivative in beta_c at 0 is
    # therefore the COM I tangent at beta_c = 0, compared with frozen
    # central differences across 0.
    model, _, data = case
    function = likelihood(model, data, SETTINGS, free=(1,))
    _check_against_frozen(function, np.array([0.0, THETA[1]]))


@pytest.fixture(scope="module")
def parent():
    """Return the truncated Hernquist parent of the components of `case`,
    in internal units."""
    return truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.05),
    )


@pytest.mark.parametrize(
    "k, beta_c", [(0, 0.0), (1, BETA_CONSTANT)], ids=["isotropic", "constant"]
)
def test_an_isotropic_or_constant_member_is_free_in_beta_c(case, parent, k, beta_c):
    # One isotropic or constant-anisotropy component is free, and the other
    # component has the constant anisotropy BETA_CONSTANT or is isotropic.
    # The free component's R_a column stays at the component's own value,
    # infinity, with a zero gradient. The gradient in beta_c is compared with
    # Richardson-extrapolated central differences of the likelihood with the
    # component rebuilt at beta_c plus and minus a step, with steps 2h and h,
    # and must agree within three times the differences' own error estimate.
    _, _, data = case

    def build(b):
        members = [
            _physical(
                truncate(
                    compute_df(
                        Component(
                            profile=parent.profile, anisotropy=Anisotropy(beta_c=x)
                        )
                    )
                )
            )
            for x in ((b, BETA_CONSTANT) if k == 0 else (0.0, b))
        ]
        return superpose({members[0]: 0.4, members[1]: 0.6})

    function = likelihood(build(beta_c), data, SETTINGS, free=(k,))
    theta = np.asarray(function.theta)
    np.testing.assert_array_equal(theta, [[beta_c, np.inf]])
    gradient = np.asarray(jax.grad(lambda t: function(theta=t))(theta))[0]
    assert gradient[1] == 0.0

    def difference(step):
        return (
            float(likelihood(build(beta_c + step), data, SETTINGS)())
            - float(likelihood(build(beta_c - step), data, SETTINGS)())
        ) / (2.0 * step)

    d_2, d_1 = difference(2e-3), difference(1e-3)
    reference = d_1 + (d_1 - d_2) / 3.0
    error = abs(d_1 - d_2) / 3.0
    assert abs(gradient[0] - reference) <= 3.0 * error + 1e-9 * abs(reference)


def test_a_refused_theta_is_minus_infinity_with_a_zero_gradient(case):
    model, _, data = case
    function = likelihood(model, data, SETTINGS, free=(1,))
    for theta in ([[0.9, THETA[1]]], [[THETA[0], -1.0]]):
        assert float(function(theta=theta)) == -np.inf
        gradient = jax.grad(lambda t: function(theta=t))(jnp.asarray(theta))
        np.testing.assert_array_equal(np.asarray(gradient), 0.0)


def test_refusals_at_construction(case):
    model, parent, data = case
    with pytest.raises(InputError, match="truncated profile"):
        likelihood(parent, data, SETTINGS, free=(0,))
    with pytest.raises(InputError, match="distinct components"):
        likelihood(model, data, SETTINGS, free=(1, 1))
    with pytest.raises(InputError, match="no component's anisotropy is free"):
        likelihood(model, data, SETTINGS)(theta=[THETA])
    untruncated = _physical(
        truncate(
            compute_df(
                Component(DoublePowerLaw.hernquist(), Anisotropy(beta_c=-0.2, r_a=1.5))
            )
        )
    )
    with pytest.raises(InputError, match="not spatially finite"):
        likelihood(untruncated, data[0], SETTINGS, free=(0,))
