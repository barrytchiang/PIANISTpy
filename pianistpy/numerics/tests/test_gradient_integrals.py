r"""The gradient contract on the forward integrals: a moment of the DF over the
anisotropy's kernel, on the package's Gauss-Legendre rule with the endpoint
map of the auxiliary density's quadrature, differentiated in
:math:`\beta_{\rm c}` and :math:`R_{\rm a}`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betaln

from pianistpy.numerics import gauss_legendre_unit

# rho(r, Psi) = g(r) I(Psi): g = r^(-2 beta) (1 + r^2 / R_a^2)^(beta - 1), the
# Cuddeford radial factor, and I = int_0^Psi F(Q) (2 (Psi - Q))^n dQ with
# n = 1/2 - beta, here for F = Q^A, whose closed form is
# I = 2^n Psi^(A + n + 1) B(A + 1, n + 1)
A, PSI, R = 1.5, 0.6, 1.3
SPLIT, STRETCH, NODES = 0.9, 2.0, 500  # the package's n_source_nodes


def _moment(beta, r_a):
    """The quadrature as `df.truncation`'s kernel forms it: [0, split Psi] in
    ln Q, and (split Psi, Psi] through Q = Psi - u^k, k = 1 / (1 + n) where the
    kernel is singular (n < 0) and 2 otherwise."""
    n = 0.5 - beta
    t, w = (jnp.asarray(a) for a in gauss_legendre_unit(NODES))
    split = SPLIT * PSI
    y0, y1 = jnp.log(1e-12), jnp.log(split)
    q1 = jnp.exp(y0 + (y1 - y0) * t)
    first = (y1 - y0) * jnp.dot(w, q1**A * (2.0 * (PSI - q1)) ** n * q1)
    k = jnp.where(n < 0.0, 1.0 / (1.0 + n), STRETCH)
    u_max = (PSI - split) ** (1.0 / k)
    u = u_max * t
    q2 = PSI - u**k
    # (2 u^k)^n k u^(k - 1): the map turns the kernel's u^(k n) into u^(k (n + 1) - 1)
    second = jnp.dot(w, q2**A * 2.0**n * k * u_max * u ** (k * (n + 1.0) - 1.0))
    g = R ** (-2.0 * beta) * (1.0 + (R / r_a) ** 2) ** (beta - 1.0)
    return g * (first + second)


def _closed(beta, r_a):
    n = 0.5 - beta
    g = R ** (-2.0 * beta) * (1.0 + (R / r_a) ** 2) ** (beta - 1.0)
    return g * 2.0**n * PSI ** (A + n + 1.0) * jnp.exp(betaln(A + 1.0, n + 1.0))


@pytest.mark.parametrize("beta", [0.25, 0.75])  # the kernel regular, and singular
def test_the_moment_differentiates_in_beta_c_and_r_a(beta):
    p = jnp.array([beta, 3.0])
    f = jax.jit(lambda q: _moment(q[0], q[1]))
    exact = np.asarray(jax.grad(lambda q: _closed(q[0], q[1]))(p))
    np.testing.assert_allclose(float(f(p)), float(_closed(beta, 3.0)), rtol=1e-12)
    # forward, one tangent per parameter, and reverse, one cotangent
    forward = np.array([float(jax.jvp(f, (p,), (jnp.eye(2)[j],))[1]) for j in range(2)])
    reverse = np.asarray(jax.vjp(f, p)[1](jnp.float64(1.0))[0])
    np.testing.assert_allclose(forward, exact, rtol=1e-10)
    np.testing.assert_allclose(reverse, exact, rtol=1e-10)
    for j in range(2):
        h = 1e-6 * float(p[j])
        step = jnp.eye(2)[j] * h
        central = (float(f(p + step)) - float(f(p - step))) / (2.0 * h)
        np.testing.assert_allclose(central, reverse[j], rtol=1e-6)
