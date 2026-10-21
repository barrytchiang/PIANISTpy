r"""Tests of the methods losvd, pdf_v and aperture_losvd of `Differentiable`,
which take their gradients through the DF.

theta is the anisotropy vector :math:`(\beta_{\rm c}, R_{\rm a})`. At the model's
own theta, each method's value equals the plain call's value byte for byte. Each
method's gradient agrees with central differences of the plain call on the model
rebuilt at theta. Further tests cover a zero cotangent, a partly zero cotangent,
the InputError for an isotropic member and for a member on a critical branch, and
the transpose of read_panels.
"""

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

pytestmark = pytest.mark.slow

QUICK = Settings(quick=True)
# The finite-difference step relative to |theta_i|, or absolute where
# |theta_i| < 1.
H = 1e-3
RADII = np.array([0.3, 0.3, 1.0, 2.5])
VELOCITIES = np.array([0.1, -0.35, 0.2, 0.05])
APERTURE = Aperture.circle(r_ape=1.0)
AP_VELOCITIES = np.array([0.0, 0.15, -0.3, 0.6])


@pytest.fixture(scope="module")
def parent():
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


@pytest.fixture(scope="module")
def om(parent):
    r"""Return an Osipkov-Merritt Type I child (:math:`R_{\rm a} = 5`) and its
    Differentiable."""
    model = _child(parent, r_a=5.0)
    return model, Differentiable(model)


@pytest.fixture(scope="module")
def com(parent):
    r"""Return a Cuddeford-Osipkov-Merritt (COM) Type I child
    (:math:`\beta_{\rm c} = -0.2`, :math:`R_{\rm a} = 5`) and its Differentiable."""
    model = _child(parent, beta_c=-0.2, r_a=5.0)
    return model, Differentiable(model)


def _against_differences(function, theta, parameters):
    # The custom VJP's gradient of the scalar function(theta) is compared, in
    # each parameter index of `parameters`, with the Richardson extrapolation of
    # central differences at steps h and 2h. The plain call evaluates the model
    # rebuilt at theta +- step. The gradient must lie within three times the
    # differences' error estimate |d1 - d2|, plus 1e-9 times the extrapolation.
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
    value = np.asarray(d.losvd(d.theta, RADII, VELOCITIES))
    assert value.tobytes() == np.asarray(losvd(model, RADII, VELOCITIES)).tobytes()
    c = np.linspace(1.0, 2.0, RADII.size)
    theta = np.asarray(d.theta)
    rule = _against_differences(
        lambda t: jnp.sum(c * d.losvd(t, RADII, VELOCITIES)), theta, (1,)
    )
    # The DF path takes no derivative in an OM member's beta_c, so the gradient
    # in beta_c is NaN.
    assert np.isnan(rule[0])


def test_aperture_losvd_value_and_gradient(om):
    model, d = om
    value = np.asarray(d.aperture_losvd(d.theta, APERTURE, AP_VELOCITIES))
    plain = np.asarray(aperture_losvd(model, APERTURE, AP_VELOCITIES))
    assert value.tobytes() == plain.tobytes()
    c = np.linspace(1.0, 2.0, AP_VELOCITIES.size)
    _against_differences(
        lambda t: jnp.sum(c * d.aperture_losvd(t, APERTURE, AP_VELOCITIES)),
        np.asarray(d.theta),
        (1,),
    )


def test_the_aperture_gradient_makes_its_panels_once_in_the_backward(om, monkeypatch):
    # The backward pass transposes the record of the panels that the forward
    # pass's stage made. aperture_losvd_panels runs once in the forward pass
    # (no record) and once in the backward pass (with a record). With the
    # record dropped, aperture_losvd_vjp makes the panels again, and the
    # gradient is the same bit for bit.
    import pianistpy.observables._aperture_losvd as panel_module
    import pianistpy.observables._differentiable as differentiable_module
    import pianistpy.observables.stages as stages_module

    _, d = om
    make = panel_module.aperture_losvd_panels
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("record") is not None)
        return make(*args, **kwargs)

    monkeypatch.setattr(stages_module, "aperture_losvd_panels", counted)
    monkeypatch.setattr(panel_module, "aperture_losvd_panels", counted)
    c = np.linspace(1.0, 2.0, AP_VELOCITIES.size)

    def function(t):
        return jnp.sum(c * d.aperture_losvd(t, APERTURE, AP_VELOCITIES))

    theta = jnp.asarray(d.theta)
    once = np.asarray(jax.grad(function)(theta))
    assert calls == [False, True]
    transpose = differentiable_module.aperture_losvd_vjp

    def recomputed(*args, record=None):
        return transpose(*args)

    monkeypatch.setattr(differentiable_module, "aperture_losvd_vjp", recomputed)
    calls.clear()
    again = np.asarray(jax.grad(function)(theta))
    assert calls == [False, True, True]
    assert again.tobytes() == once.tobytes()


def test_pdf_v_value_and_gradient(com):
    model, d = com
    r, v = np.array([0.3, 0.3, 1.5]), np.array([0.2, 0.6, 0.4])
    value = np.asarray(d.pdf_v(d.theta, r, v))
    assert value.tobytes() == np.asarray(pdf_v(model, r, v)).tobytes()
    c = np.array([1.0, -0.5, 2.0])
    _against_differences(
        lambda t: jnp.sum(c * d.pdf_v(t, r, v)), np.asarray(d.theta), (0, 1)
    )


def test_pdf_v_on_a_zero_cotangent_is_nan_in_a_parameter_the_path_does_not_take(om):
    # The DF path takes no derivative in an OM member's beta_c. For a zero
    # cotangent, pdf_v's gradient is NaN in beta_c, as losvd's is, and zero in
    # R_a.
    _, d = om
    r, v = np.array([0.3, 1.5]), np.array([0.2, 0.4])
    rule = np.asarray(
        jax.grad(lambda t: jnp.sum(0.0 * d.pdf_v(t, r, v)))(jnp.asarray(d.theta))
    )
    assert np.isnan(rule[0]) and rule[1] == 0.0


def _counted(monkeypatch, names):
    # Each function of _differentiable listed in `names` is patched to append
    # its name to the returned list at every call.
    from pianistpy.observables import _differentiable

    calls = []
    for name in names:
        function = getattr(_differentiable, name)
        monkeypatch.setattr(
            _differentiable,
            name,
            lambda *a, f=function, n=name, **k: calls.append(n) or f(*a, **k),
        )
    return calls


def test_a_zero_cotangent_is_a_zero_gradient_with_nothing_transposed(om, monkeypatch):
    # A zero cotangent calls none of losvd_read_vjp, losvd_rows_vjp,
    # stage_aperture_losvd and table_jacobian. The gradient for a zero cotangent
    # is NaN in an OM member's beta_c, which the DF path does not take, and
    # zero in R_a.
    _, d = om
    calls = _counted(
        monkeypatch,
        ("losvd_read_vjp", "losvd_rows_vjp", "stage_aperture_losvd", "table_jacobian"),
    )
    theta = jnp.asarray(d.theta)
    for function in (
        lambda t: jnp.sum(np.zeros(RADII.size) * d.losvd(t, RADII, VELOCITIES)),
        lambda t: jnp.sum(
            np.zeros(AP_VELOCITIES.size) * d.aperture_losvd(t, APERTURE, AP_VELOCITIES)
        ),
    ):
        g = np.asarray(jax.grad(function)(theta))
        assert np.isnan(g[0])
        assert g[1:].tobytes() == np.zeros(1).tobytes()
    assert calls == []
    # Velocities beyond every row's end give a nonzero cotangent that the read
    # carries to no row. losvd_read_vjp and losvd_rows_vjp run, and
    # table_jacobian does not.
    g = np.asarray(
        jax.grad(lambda t: jnp.sum(d.losvd(t, RADII, VELOCITIES + 100.0)))(theta)
    )
    assert np.isnan(g[0])
    assert g[1:].tobytes() == np.zeros(1).tobytes()
    assert calls == ["losvd_read_vjp", "losvd_rows_vjp"]


def test_a_partly_zero_cotangent_is_the_gradient_of_its_nonzero_part(om):
    # The gradient with zero cotangents on two (radius, velocity) pairs equals,
    # bit for bit, the gradient with the two zero-cotangent pairs left out.
    _, d = om
    c = np.array([1.0, 0.0, 2.0, 0.0])
    on = c != 0.0
    theta = jnp.asarray(d.theta)
    full = jax.grad(lambda t: jnp.sum(c * d.losvd(t, RADII, VELOCITIES)))(theta)
    part = jax.grad(lambda t: jnp.sum(c[on] * d.losvd(t, RADII[on], VELOCITIES[on])))(
        theta
    )
    assert np.asarray(full).tobytes() == np.asarray(part).tobytes()


def test_pdf_v_solves_no_radius_of_zero_cotangent(com, monkeypatch):
    # The backward pass calls velocity_distribution only at radii with a nonzero
    # cotangent. The gradient with a zero cotangent at one radius equals, bit
    # for bit, the gradient with the zero-cotangent radius left out. An
    # all-zero cotangent calls velocity_distribution at no radius and gives a zero
    # gradient.
    _, d = com
    r, v = np.array([0.3, 0.3, 1.5]), np.array([0.2, 0.6, 0.4])
    c = np.array([1.0, -0.5, 0.0])
    on = c != 0.0
    calls = _counted(monkeypatch, ("velocity_distribution",))
    theta = jnp.asarray(d.theta)
    full = jax.grad(lambda t: jnp.sum(c * d.pdf_v(t, r, v)))(theta)
    part = jax.grad(lambda t: jnp.sum(c[on] * d.pdf_v(t, r[on], v[on])))(theta)
    assert np.asarray(full).tobytes() == np.asarray(part).tobytes()
    assert len(calls) == 2
    zero = jax.grad(lambda t: jnp.sum(np.zeros(3) * d.pdf_v(t, r, v)))(theta)
    assert np.asarray(zero).tobytes() == np.zeros(2).tobytes()
    assert len(calls) == 2


@pytest.fixture(scope="module")
def iso(parent):
    """Return an isotropic child and its Differentiable."""
    model = _child(parent)
    return model, Differentiable(model)


def test_an_isotropic_member_is_refused_with_cause_by_the_df_methods(iso):
    # The DF path takes no derivative in an isotropic child's beta_c, because
    # the derivatives of the DF table from the two sides of beta_c = 0 do not
    # agree to 1e-6. Each DF method raises InputError before any trace, and the
    # message gives the Jeans path.
    _, d = iso
    for call in (
        lambda: d.losvd(d.theta, RADII, VELOCITIES),
        lambda: d.pdf_v(d.theta, RADII, VELOCITIES),
        lambda: d.aperture_losvd(d.theta, APERTURE, AP_VELOCITIES),
    ):
        with pytest.raises(
            InputError,
            match="not differentiable.*isotropic member.*do not agree.*Jeans path",
        ):
            call()


def test_a_member_the_df_rule_refuses_is_stated_by_the_df_methods(om):
    # A constant beta_c = 1/2 on the Hernquist profile puts the center on its
    # critical branch. Each DF method raises InputError whose message gives the
    # critical_analytic branch.
    model, _ = om
    component = model.components[0]
    critical = dataclasses.replace(
        model,
        components=(dataclasses.replace(component, anisotropy=Anisotropy(beta_c=0.5)),),
    )
    d = Differentiable(critical)
    for call in (
        lambda: d.losvd(d.theta, RADII, VELOCITIES),
        lambda: d.pdf_v(d.theta, RADII, VELOCITIES),
        lambda: d.aperture_losvd(d.theta, APERTURE, AP_VELOCITIES),
    ):
        with pytest.raises(InputError, match="critical_analytic branch"):
            call()


def test_the_panels_reading_transpose_is_the_reading_transposed():
    # read_panels is linear in the node values y, so <c, A y> equals
    # <A^T c, y> with A the reading and A^T c from read_panels_vjp. The velocities
    # include the ends 0.4 and 1.0 that two sub-panels share, the end 1.7 of
    # the last sub-panel and the velocity 2.0 outside every sub-panel.
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


def test_a_gradient_builds_the_route_with_the_df_once(om, monkeypatch):
    # The rows' transpose and the rows' explicit part read one ellipse route
    # with the DF, kept on the entry of theta. The first gradient builds the
    # ellipse route once. A second gradient at the same theta builds no route, and the
    # two gradients are equal bit for bit.
    from pianistpy.observables import _ellipse_model

    model, d = om
    d._df.entry(np.asarray(d.theta)).pop("rows_basis", None)
    built = []
    made = _ellipse_model.ellipse_route
    monkeypatch.setattr(
        _ellipse_model,
        "ellipse_route",
        lambda m, rules, with_df=True: built.append(with_df) or made(m, rules, with_df),
    )
    c = np.linspace(1.0, 2.0, RADII.size)

    def gradient():
        return np.asarray(
            jax.grad(lambda t: jnp.sum(c * d.losvd(t, RADII, VELOCITIES)))(d.theta)
        )

    first = gradient()
    assert built.count(True) == 1
    second = gradient()
    assert built.count(True) == 1
    assert second.tobytes() == first.tobytes()
