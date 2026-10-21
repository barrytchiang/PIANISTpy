r"""Tests of the derivatives of a DF moment in the anisotropy parameters.

The anisotropy parameters are the central anisotropy :math:`\beta_{\rm c}`
and the anisotropy radius :math:`R_{\rm a}`.

The moment is the DF integrated against the anisotropy's kernel, by the
package's Gauss-Legendre rule with the change of variable that the quadrature
of the auxiliary density uses near the upper limit. The value, the forward
derivatives and the reverse derivatives of the moment are compared with the
closed form and with central differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betaln

from pianistpy.numerics import gauss_legendre_unit

# The moment is rho(r, Psi) = g(r) I(Psi). The radial factor of a COM Type I
# model is g = r^(-2 beta) (1 + r^2 / R_a^2)^(beta - 1). The energy integral is
# I = int_0^Psi F(Q) (2 (Psi - Q))^n dQ with n = 1/2 - beta. For the DF
# F = Q^A, the closed form of the energy integral is
# I = 2^n Psi^(A + n + 1) B(A + 1, n + 1), with B the beta function.
A, PSI, R = 1.5, 0.6, 1.3
SPLIT, STRETCH, NODES = 0.9, 2.0, 500  # NODES is the default n_source_nodes.


def _moment(beta, r_a):
    """Return the moment by the quadrature that `pianistpy.df.truncation` uses.

    The interval [0, split Psi] is integrated with nodes uniform in ln Q. The
    interval (split Psi, Psi] is integrated through Q = Psi - u^k, with
    k = 1 / (1 + n) where the kernel (Psi - Q)^n is singular (n < 0), and
    k = 2 otherwise.
    """
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
    # The kernel (2 u^k)^n times the Jacobian k u^(k - 1) gives the power
    # u^(k (n + 1) - 1), which is u^0 for k = 1 / (1 + n).
    second = jnp.dot(w, q2**A * 2.0**n * k * u_max * u ** (k * (n + 1.0) - 1.0))
    g = R ** (-2.0 * beta) * (1.0 + (R / r_a) ** 2) ** (beta - 1.0)
    return g * (first + second)


def _closed(beta, r_a):
    n = 0.5 - beta
    g = R ** (-2.0 * beta) * (1.0 + (R / r_a) ** 2) ** (beta - 1.0)
    return g * 2.0**n * PSI ** (A + n + 1.0) * jnp.exp(betaln(A + 1.0, n + 1.0))


@pytest.mark.parametrize("beta", [0.25, 0.75])  # Kernel regular, then singular.
def test_the_moment_differentiates_in_beta_c_and_r_a(beta):
    p = jnp.array([beta, 3.0])
    f = jax.jit(lambda q: _moment(q[0], q[1]))
    exact = np.asarray(jax.grad(lambda q: _closed(q[0], q[1]))(p))
    np.testing.assert_allclose(float(f(p)), float(_closed(beta, 3.0)), rtol=1e-12)
    # The forward derivatives use one tangent per parameter, and the reverse
    # derivatives use one cotangent.
    forward = np.array([float(jax.jvp(f, (p,), (jnp.eye(2)[j],))[1]) for j in range(2)])
    reverse = np.asarray(jax.vjp(f, p)[1](jnp.float64(1.0))[0])
    np.testing.assert_allclose(forward, exact, rtol=1e-10)
    np.testing.assert_allclose(reverse, exact, rtol=1e-10)
    for j in range(2):
        h = 1e-6 * float(p[j])
        step = jnp.eye(2)[j] * h
        central = (float(f(p + step)) - float(f(p - step))) / (2.0 * h)
        np.testing.assert_allclose(central, reverse[j], rtol=1e-6)
