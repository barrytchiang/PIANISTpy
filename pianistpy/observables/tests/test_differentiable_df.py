"""`Differentiable`'s DF path: losvd, pdf_v and aperture_losvd, each value
against the plain call at the model's own theta (to the byte), its gradient
against central differences of the plain call on the model rebuilt at theta,
and the refusals; the reading of an aperture's panels transposed."""

import dataclasses

import jax
import jax.numpy as jnp
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
    aperture_losvd,
    compute_df,
    losvd,
    pdf_v,
    solve,
    truncate,
)
from pianistpy.observables._panels import panels, read_panels, read_panels_vjp

QUICK = Settings(quick=True)
H = 1e-3  # relative to the parameter, as the skeleton's sigma_los test
RADII = np.array([0.3, 0.3, 1.0, 2.5])
SPEEDS = np.array([0.1, -0.35, 0.2, 0.05])
APERTURE = Aperture.circle(r_ape=1.0)
AP_SPEEDS = np.array([0.0, 0.15, -0.3, 0.6])


@pytest.fixture(scope="module")
def parent():
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


@pytest.fixture(scope="module")
def om(parent):
    r"""An Osipkov-Merritt Type I child (:math:`R_{\rm a}`) and its Differentiable."""
    model = _child(parent, r_a=5.0)
    return model, Differentiable(model)


@pytest.fixture(scope="module")
def com(parent):
    r"""A Cuddeford Type I child (:math:`\beta_{\rm c}` and :math:`R_{\rm a}`) and its
    Differentiable."""
    model = _child(parent, beta_c=-0.2, r_a=5.0)
    return model, Differentiable(model)


def _against_differences(function, theta, parameters):
    # the rule's gradient of <c, f(theta)> against central differences of the
    # plain call on the model rebuilt at theta +- step, steps 2h and h,
    # Richardson; within three of the differences' own error estimate
    rule = np.asarray(jax.grad(function)(jnp.asarray(theta)))
    for i in parameters:
        h = H * max(1.0, abs(float(theta[i])))

        def difference(step, i=i):
            e = np.zeros(2)
            e[i] = step
            return (float(function(theta + e)) - float(function(theta - e))) / (
                2.0 * step
            )

        d1, d2 = difference(h), difference(2.0 * h)
        richardson = (4.0 * d1 - d2) / 3.0
        error = abs(d1 - d2)
        assert abs(rule[i] - richardson) <= 3.0 * error + 1e-9 * abs(richardson), (
            i,
            rule[i],
            richardson,
            error,
        )
    return rule


def test_losvd_value_and_gradient(om):
    model, d = om
    value = np.asarray(d.losvd(d.theta, RADII, SPEEDS))
    assert value.tobytes() == np.asarray(losvd(model, RADII, SPEEDS)).tobytes()
    c = np.linspace(1.0, 2.0, RADII.size)
    theta = np.asarray(d.theta)
    rule = _against_differences(
        lambda t: jnp.sum(c * d.losvd(t, RADII, SPEEDS)), theta, (1,)
    )
    assert np.isnan(rule[0])  # an OM member's beta_c is not the DF path's


def test_aperture_losvd_value_and_gradient(om):
    model, d = om
    value = np.asarray(d.aperture_losvd(d.theta, APERTURE, AP_SPEEDS))
    plain = np.asarray(aperture_losvd(model, APERTURE, AP_SPEEDS))
    assert value.tobytes() == plain.tobytes()
    c = np.linspace(1.0, 2.0, AP_SPEEDS.size)
    _against_differences(
        lambda t: jnp.sum(c * d.aperture_losvd(t, APERTURE, AP_SPEEDS)),
        np.asarray(d.theta),
        (1,),
    )


def test_pdf_v_value_and_gradient(com):
    model, d = com
    r, v = np.array([0.3, 0.3, 1.5]), np.array([0.2, 0.6, 0.4])
    value = np.asarray(d.pdf_v(d.theta, r, v))
    assert value.tobytes() == np.asarray(pdf_v(model, r, v)).tobytes()
    c = np.array([1.0, -0.5, 2.0])
    _against_differences(
        lambda t: jnp.sum(c * d.pdf_v(t, r, v)), np.asarray(d.theta), (0, 1)
    )


@pytest.fixture(scope="module")
def iso(parent):
    """An isotropic child and its Differentiable."""
    model = _child(parent)
    return model, Differentiable(model)


def test_an_isotropic_member_is_refused_by_name_by_the_df_methods(iso):
    # the DF path takes no derivative in an isotropic child's beta_c: each DF
    # method refuses it by name, before any trace
    _, d = iso
    for call in (
        lambda: d.losvd(d.theta, RADII, SPEEDS),
        lambda: d.pdf_v(d.theta, RADII, SPEEDS),
        lambda: d.aperture_losvd(d.theta, APERTURE, AP_SPEEDS),
    ):
        with pytest.raises(InputError, match="not differentiable.*isotropic member"):
            call()


def test_a_member_the_df_rule_refuses_is_named_by_the_df_methods(om):
    # constant beta_c = 1/2 on the Hernquist profile: the center on its
    # critical branch; sigma_los is served, the DF path is refused by name
    model, _ = om
    component = model.components[0]
    critical = dataclasses.replace(
        model,
        components=(dataclasses.replace(component, anisotropy=Anisotropy(beta_c=0.5)),),
    )
    d = Differentiable(critical)
    for call in (
        lambda: d.losvd(d.theta, RADII, SPEEDS),
        lambda: d.pdf_v(d.theta, RADII, SPEEDS),
        lambda: d.aperture_losvd(d.theta, APERTURE, AP_SPEEDS),
    ):
        with pytest.raises(InputError, match="critical_analytic branch"):
            call()


def test_the_panels_reading_transpose_is_the_reading_transposed():
    # read_panels is linear in the values: <c, A y> against <A^T c, y>, with a
    # speed on an end two sub-panels share
    rng = np.random.default_rng(3)
    x, w = np.polynomial.legendre.leggauss(5)
    a, b = np.array([0.0, 0.4, 1.0]), np.array([0.4, 1.0, 1.7])
    nodes = np.concatenate(
        [0.5 * (lo + hi) + 0.5 * (hi - lo) * x for lo, hi in zip(a, b, strict=True)]
    )
    weights = np.concatenate([0.5 * (hi - lo) * w for lo, hi in zip(a, b, strict=True)])
    y = rng.normal(size=nodes.size)
    v = np.concatenate([rng.uniform(0.0, 1.7, 40), [0.4, 1.0, 1.7, 2.0]])
    c = rng.normal(size=v.size)
    forward = float(c @ read_panels(panels(nodes, weights, y), v))
    back = read_panels_vjp(panels(nodes, weights, y), v, c)
    assert float(back @ y) == pytest.approx(forward, rel=1e-12, abs=1e-12)
