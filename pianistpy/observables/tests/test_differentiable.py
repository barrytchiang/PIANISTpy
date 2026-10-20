"""`Differentiable`: each method's value against the plain call at the model's
own theta (to the byte), its gradient against central differences of the
plain call on the model rebuilt at theta, and its refusals."""

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

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
RADII = np.geomspace(0.05, 0.5, 6)
APERTURE = Aperture.circle(0.3)
H = 1e-3  # as docs/guide/gradient.rst's example
# each method: (its call on the object at one place, the plain call there)
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
    """The truncated Hernquist model whose profile the members share."""
    return truncate(
        compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy())),
        Truncation(k_t=0.05),
    )


def _member(parent, anisotropy):
    model = truncate(
        compute_df(Component(profile=parent.profile, anisotropy=anisotropy))
    )
    return dataclasses.replace(model, settings=SETTINGS)


@pytest.fixture(scope="module", params=["constant", "COM_I"])
def member(parent, request):
    r"""A constant (:math:`\beta_{\rm c}`) and a Cuddeford Type I
    (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) member, its `Differentiable`, and the
    parameters each is differentiated in here."""
    if request.param == "constant":
        model, parameters = _member(parent, Anisotropy(beta_c=0.25)), (0,)
    else:
        model, parameters = _member(parent, Anisotropy(beta_c=-0.2, r_a=1.5)), (0, 1)
    return model, Differentiable(model), parameters


@pytest.mark.parametrize("method", list(METHODS))
def test_the_value_at_the_own_theta_is_the_plain_call(member, method):
    model, d, _ = member
    call, plain = METHODS[method]
    value = np.asarray(call(d, d.theta, RADII))
    assert (
        value.tobytes() == np.asarray(plain(model, RADII), dtype=np.float64).tobytes()
    )


@pytest.mark.parametrize("method", list(METHODS))
def test_the_gradient_is_the_central_difference_of_the_plain_call(member, method):
    # each parameter's derivative against central differences of the value at
    # theta +- step (the model rebuilt there: free placement), steps 2h and h,
    # Richardson; within three of the differences' own error estimate
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
        assert rule[1] == 0.0  # the held R_a column


def test_a_model_whose_anisotropy_cannot_be_free_is_refused(parent, member):
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
