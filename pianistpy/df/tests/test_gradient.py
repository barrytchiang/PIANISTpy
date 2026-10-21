r"""Tests of `rows_jacobian` and `make_inversion_rows`.

`rows_jacobian` returns the derivatives of the inverted DF :math:`F(Q)` in
the anisotropy parameters :math:`\beta_{\rm c}` and :math:`R_{\rm a}`. The
reference is the exact derivative of the inversion's defining integral for
the Hernquist profile in model units, :math:`\rho = 2 R^{-1} (1 + R)^{-3}`
and :math:`P = 1 / (1 + R)`, so that :math:`R(P) = 1/P - 1` exactly. With
:math:`b = \beta_{\rm c}`:

.. math::

    F(Q) &= \frac{P_{\rm f}}{\Gamma(e + 1)} \int_0^Q h(P) \, (Q - P)^e \,
    {\rm d}P, \\
    h &= \frac{{\rm d}^k \rho_{\rm aux}}{{\rm d}P^k}, \quad
    k = \lceil 3/2 - b \rceil, \quad e = k - 5/2 + b, \\
    P_{\rm f} &= \frac{2^b}{(2 \pi)^{3/2} \, \Gamma(1 - b)}, \\
    \frac{\partial F}{\partial b} &= F \, [\ln 2 + \psi(1 - b) - \psi(e + 1)]
    + \frac{P_{\rm f}}{\Gamma(e + 1)} \int_0^Q \left[ \partial_b h \,
    (Q - P)^e + h \, (Q - P)^e \ln (Q - P) \right] {\rm d}P, \\
    \frac{\partial F}{\partial R_{\rm a}} &= \frac{P_{\rm f}}{\Gamma(e + 1)}
    \int_0^Q \partial_{R_{\rm a}} h \, (Q - P)^e \, {\rm d}P.

Here :math:`\rho_{\rm aux}` is the auxiliary density, :math:`\psi` the
digamma function, and :math:`\partial_b` and :math:`\partial_{R_{\rm a}}`
the partial derivatives in :math:`b` and :math:`R_{\rm a}`. The reference
evaluates the integrals in mpmath at 40 digits. Away from the corners of the
inversion's form in :math:`\beta_{\rm c}`, the derivative from
`rows_jacobian` must match the reference to a relative :math:`10^{-6}`.
"""

import math

import numpy as np
import pytest

from pianistpy import platform
from pianistpy.anisotropy import Anisotropy
from pianistpy.df.gradient import make_inversion_rows, rows_jacobian
from pianistpy.df.inputs import Component
from pianistpy.exceptions import InputError
from pianistpy.profiles import DoublePowerLaw
from pianistpy.settings import Settings

pytestmark = pytest.mark.slow

HERNQUIST = DoublePowerLaw(1.0, 4.0, 1.0)
# A transpose that is reproduced to the bit requires a fixed order of
# floating-point reductions, which a GPU does not guarantee. On a GPU the
# accuracy grid judges the derivative instead of this bitwise test.
CPU_ONLY = pytest.mark.skipif(
    platform() != "cpu",
    reason="deterministic recomputation, a property of a fixed reduction order; "
    "on a GPU the accuracy grid judges",
)
ENERGIES = (1.0 - 1e-4, 0.5, 1e-6)


def _exact(q, b, r_a, wrt, sign):
    import mpmath as mp
    import sympy as sp

    mp.mp.dps = 40
    p, ras = sp.symbols("P R_a", positive=True)
    bcs = sp.Symbol("beta_c", real=True)
    r = 1 / p - 1
    # The anisotropy factor is R^(2 beta_c) (1 + sign R^2 / R_a^2)^(1 - beta_c).
    # The COM Type I model has sign = 1. The OM models have beta_c = 0, with
    # sign = 1 for Type I and sign = -1 for Type II. The constant-anisotropy
    # model is evaluated with sign = 0.
    gfac = r ** (2 * bcs) * (1 + sign * r**2 / ras**2) ** (1 - bcs)
    k = math.ceil(1.5 - b)
    h = sp.diff(2 * r ** (-1) * (1 + r) ** (-3) * gfac, p, k)
    fh = sp.lambdify((p, bcs, ras), h, "mpmath")
    fd = sp.lambdify(
        (p, bcs, ras), sp.diff(h, {"beta_c": bcs, "r_a": ras}[wrt]), "mpmath"
    )
    bm, ram = mp.mpf(b), mp.mpf(r_a)
    e = k - mp.mpf(3) / 2 + bm - 1
    norm = (
        mp.mpf(2) ** bm
        / ((2 * mp.pi) ** mp.mpf(1.5) * mp.gamma(1 - bm))
        / mp.gamma(e + 1)
    )
    qq = mp.mpf(float(q))

    def integral(fn, log=False):
        w = (
            (lambda x: (qq - x) ** e * mp.log(qq - x))
            if log
            else (lambda x: (qq - x) ** e)
        )
        return mp.quad(lambda x: fn(x, bm, ram) * w(x), [0, qq / 2, qq])

    if wrt == "r_a":
        return float(norm * integral(fd))
    f = norm * integral(fh)
    return float(
        f * (mp.log(2) + mp.digamma(1 - bm) - mp.digamma(e + 1))
        + norm * (integral(fd) + integral(fh, log=True))
    )


def _nodes(q):
    return sorted({int(np.argmin(np.abs(q - t))) for t in ENERGIES})


def _inside(record):
    # Every difference quotient in beta_c must stay inside the region where
    # the inversion's form is the form beside b on the quotient's side. A
    # central quotient reaches b +- d and a one-sided quotient reaches b + 2 s d,
    # with d the step and s the side's sign. Where the steps were measured at
    # this call, the pilot step must also stay inside the region.
    reach = 1.0 if record["exponent"] == "central" else 2.0
    for d, p, w in zip(
        record["steps"], record["pilots"], record["regions"], strict=True
    ):
        assert 0.0 < d and reach * d <= w
        assert not record["measured"] or reach * p <= w


@pytest.fixture(scope="module")
def constant():
    c = Component(HERNQUIST, Anisotropy(beta_c=0.25))
    return rows_jacobian(c, Settings())


@pytest.fixture(scope="module")
def cuddeford():
    c = Component(HERNQUIST, Anisotropy(beta_c=-0.25, r_a=10.0))
    return rows_jacobian(c, Settings())


def test_constant_beta_against_the_exact_derivative(constant):
    q, _, jac, record = constant
    assert record["exponent"] == "central" and record["residual"] < 1e-6
    _inside(record)
    for i in _nodes(q):
        exact = _exact(q[i], 0.25, 1.0, "beta_c", 0)
        assert jac["beta_c"][i] == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("wrt", ["beta_c", "r_a"])
def test_cuddeford_against_the_exact_derivative(cuddeford, wrt):
    q, _, jac, record = cuddeford
    assert record["residual"] < 1e-6
    _inside(record)
    for i in _nodes(q):
        exact = _exact(q[i], -0.25, 10.0, wrt, 1)
        assert jac[wrt][i] == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("b0", [0.5, 0.0])
def test_a_corner_takes_both_sides(b0):
    # At the corners b0 = 0.5 and b0 = 0 the inversion's form changes, and
    # `rows_jacobian` differences one-sided on both sides of b0. F is smooth
    # across both corners, so the reference slope at b0 is the limit from
    # below, extrapolated linearly from the exact derivatives at b0 - 1e-3 and
    # b0 - 5e-4. The reference is not taken from above, because above
    # beta_c = 1/2 the reference's exponent e tends to -1, which the mpmath
    # quadrature does not resolve.
    c = Component(HERNQUIST, Anisotropy(beta_c=b0))
    q, _, jac, record = rows_jacobian(c, Settings())
    assert record["exponent"] == "one-sided" and record["sides"] == 2
    assert np.isfinite(record["residual"]) and np.isfinite(record["spread"])
    _inside(record)
    i = int(np.argmin(np.abs(q - 0.5)))
    left = [_exact(q[i], b0 - s, 1.0, "beta_c", 0) for s in (1e-3, 5e-4)]
    assert jac["beta_c"][i] == pytest.approx(2.0 * left[1] - left[0], rel=1e-5)


@pytest.mark.parametrize(("kind", "sign"), [("I", 1), ("II", -1)])
def test_osipkov_merritt_against_the_exact_derivative(kind, sign):
    c = Component(HERNQUIST, Anisotropy(r_a=10.0, kind=kind))
    q, _, jac, record = rows_jacobian(c, Settings())
    assert set(jac) == {"r_a"} and record == {}
    for i in _nodes(q):
        exact = _exact(q[i], 0.0, 10.0, "r_a", sign)
        assert jac["r_a"][i] == pytest.approx(exact, rel=1e-6)


def test_the_steps_travel_in_the_caller_s_store():
    # `rows_jacobian` reads the beta_c steps from the caller's store where the
    # store holds them. Without a store, `rows_jacobian` measures the steps at
    # each call, so that a result never depends on the calls made before it.
    steps: dict = {}
    first = rows_jacobian(
        Component(HERNQUIST, Anisotropy(beta_c=0.25)), Settings(), steps=steps
    )
    later = rows_jacobian(
        Component(HERNQUIST, Anisotropy(beta_c=0.3)), Settings(), steps=steps
    )
    assert first[3]["measured"] and not later[3]["measured"]
    assert later[3]["steps"] == first[3]["steps"]
    _inside(later[3])
    alone = rows_jacobian(Component(HERNQUIST, Anisotropy(beta_c=0.3)), Settings())
    again = rows_jacobian(Component(HERNQUIST, Anisotropy(beta_c=0.3)), Settings())
    assert alone[3]["measured"] and again[3]["measured"]
    assert alone[2]["beta_c"].tobytes() == again[2]["beta_c"].tobytes()


def test_cuddeford_type_two_is_refused():
    c = Component(HERNQUIST, Anisotropy(beta_c=-0.25, r_a=10.0, kind="II"))
    with pytest.raises(InputError, match="Type II"):
        rows_jacobian(c, Settings())


@CPU_ONLY
def test_the_vjp_is_the_transpose_and_the_forward_is_the_rows(constant):
    import jax

    q, f, jac, _ = constant
    c = Component(HERNQUIST, Anisotropy(beta_c=0.25))
    rows, params, q2 = make_inversion_rows(c, Settings(), q)
    theta = np.array([0.25])
    out, vjp = jax.vjp(rows, theta)
    assert params == ("beta_c",)
    np.testing.assert_array_equal(np.asarray(out), f)
    v = np.random.default_rng(1).normal(size=q.size)
    (g,) = vjp(v)
    assert float(g[0]) == pytest.approx(float(v @ jac["beta_c"]), rel=1e-12)
