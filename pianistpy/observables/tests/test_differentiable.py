"""Tests of `Differentiable` on truncated Hernquist models of one component.

The anisotropy parameters are ``theta = (beta_c, R_a)``, in internal units.
Each method of `Differentiable` at the model's own theta returns the bytes of
the matching plain call. The JAX gradient of each method in theta agrees with a
Richardson-extrapolated central difference of the method's value.
`Differentiable` raises InputError for a superposition, for a model that is
not spatially finite and for a COM Type II model. The methods of
`Differentiable` raise InputError for a radius outside the model's tabulated
radii.
"""

import dataclasses

import jax
import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Aperture,
    Component,
    Differentiable,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
    sigma_los,
    sigma_los_aperture,
    sigma_r,
    sigma_t,
    superpose,
)
from pianistpy.df import compute_df, truncate

pytestmark = pytest.mark.slow

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
RADII = np.geomspace(0.05, 0.5, 6)
APERTURE = Aperture.circle(0.3)
# The central-difference step in theta is 1e-3 in internal units, the step of the
# example in docs/guide/differentiability.rst.
H = 1e-3
# Each entry maps a method name to a pair of callables: the method of a
# `Differentiable` d at theta t and radii x, and the plain call on a model m at
# radii x.
METHODS = {
    "sigma_r": (lambda d, t, x: d.sigma_r(t, x), sigma_r),
    "sigma_t": (lambda d, t, x: d.sigma_t(t, x), sigma_t),
    "sigma_los": (lambda d, t, x: d.sigma_los(t, x), sigma_los),
    "sigma_los_aperture": (
        lambda d, t, x: d.sigma_los_aperture(t, APERTURE),
        lambda m, x: sigma_los_aperture(m, APERTURE),
    ),
}


@pytest.fixture(scope="module")
def parent():
    """Return the isotropic Hernquist model truncated by
    ``Truncation(k_t=0.05)``.

    Both tested models are built on the truncated profile of the returned model.
    """
    return truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.05),
    )


def _member(parent, anisotropy):
    """Return the model with the density profile of `parent` and the
    anisotropy `anisotropy`, with SETTINGS."""
    model = truncate(
        compute_df(Component(profile=parent.profile, anisotropy=anisotropy))
    )
    return dataclasses.replace(model, settings=SETTINGS)


@pytest.fixture(scope="module", params=["constant", "COM_I"])
def member(parent, request):
    r"""Return a tested model, its `Differentiable` and the indices of theta to check.

    The parameter "constant" gives a constant-anisotropy model with
    :math:`\beta_{\rm c} = 0.25`, checked in theta index 0 (:math:`\beta_{\rm c}`).
    The parameter "COM_I" gives a Cuddeford-Osipkov-Merritt (COM) Type I model
    with :math:`\beta_{\rm c} = -0.2` and :math:`R_{\rm a} = 1.5`, checked in theta
    indices 0 and 1 (:math:`\beta_{\rm c}` and :math:`R_{\rm a}`). Both models
    carry SETTINGS.
    """
    if request.param == "constant":
        model, parameters = _member(parent, Anisotropy(beta_c=0.25)), (0,)
    else:
        model, parameters = _member(parent, Anisotropy(beta_c=-0.2, r_a=1.5)), (0, 1)
    return model, Differentiable(model), parameters


@pytest.mark.parametrize("method", list(METHODS))
def test_the_value_at_the_own_theta_is_the_plain_call(member, method):
    """Assert that each method at the model's own theta returns bytes identical
    to the float64 result of the plain call on the model, at the radii RADII."""
    model, d, _ = member
    call, plain = METHODS[method]
    value = np.asarray(call(d, d.theta, RADII))
    assert (
        value.tobytes() == np.asarray(plain(model, RADII), dtype=np.float64).tobytes()
    )


@pytest.mark.parametrize("method", list(METHODS))
def test_the_gradient_is_the_central_difference_of_the_plain_call(member, method):
    r"""Assert that each method's JAX gradient in theta matches a central difference.

    For each checked theta index i, d1 and d2 are central differences of the
    method's value at the radius RADII[3], with steps H and 2 H in theta[i]. The
    method rebuilds the model at each shifted theta. The test asserts
    ``abs(gradient[i] - (4 d1 - d2) / 3)``
    ``<= 3 abs(d1 - d2) + 1e-12 abs((4 d1 - d2) / 3)``. For the constant-anisotropy
    model, the test also asserts that the gradient's :math:`R_{\rm a}` component is
    exactly zero.
    """
    model, d, parameters = member
    call, _ = METHODS[method]
    theta = np.asarray(d.theta)
    radius = RADII[3]
    rule = np.asarray(jax.grad(lambda t: call(d, t, radius))(d.theta))

    def difference(i, step):
        e = np.zeros(2)
        e[i] = step
        up = float(call(d, theta + e, radius))
        down = float(call(d, theta - e, radius))
        return (up - down) / (2.0 * step)

    for i in parameters:
        d1, d2 = difference(i, H), difference(i, 2.0 * H)
        richardson = (4.0 * d1 - d2) / 3.0
        error = abs(d1 - d2)
        assert abs(rule[i] - richardson) <= 3.0 * error + 1e-12 * abs(richardson), (
            i,
            rule[i],
            richardson,
            error,
        )
    if model.components[0].anisotropy.family == "constant":
        # A constant-anisotropy model holds R_a fixed.
        assert rule[1] == 0.0


def test_a_model_whose_anisotropy_cannot_be_free_is_refused(parent, member):
    """Assert that `Differentiable` raises InputError for three models.

    The three models are a superposition of two components, a Hernquist model
    passed through `truncate` without a truncation (not spatially finite), and a
    COM Type II model.
    """
    model, _, _ = member
    with pytest.raises(InputError, match="one component"):
        Differentiable(superpose({model: 0.5, _member(parent, Anisotropy()): 0.5}))
    untruncated = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy()))
    with pytest.raises(InputError, match="not spatially finite"):
        Differentiable(truncate(untruncated))
    component = model.components[0]
    type_ii = dataclasses.replace(
        model,
        components=(
            dataclasses.replace(
                component, anisotropy=Anisotropy(beta_c=-0.1, r_a=40.0, kind="II")
            ),
        ),
    )
    with pytest.raises(InputError, match="COM_II.*not differentiable"):
        Differentiable(type_ii)


def test_a_radius_outside_the_rows_is_refused_before_it_is_traced(member):
    """Assert that the methods raise InputError for radii outside the model's
    tabulated radii.

    The radii are 1.01 times the profile's ``r_table_max`` for `sigma_los` and
    `sigma_r`, and 0.1 times the profile's innermost radius for `sigma_t`.
    """
    model, differentiable, _ = member
    theta = differentiable.theta
    edge = float(model.profile.r_table_max)
    with pytest.raises(InputError, match="beyond the outermost projected radius"):
        differentiable.sigma_los(theta, np.array([0.1, 1.01 * edge]))
    with pytest.raises(InputError, match="beyond the model's edge"):
        differentiable.sigma_r(theta, np.array([0.1, 1.01 * edge]))
    with pytest.raises(InputError, match="inside the innermost"):
        differentiable.sigma_t(theta, np.array([0.1 * float(model.profile.r[0])]))
