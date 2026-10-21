"""Tests that the table interpolants differentiate in their tables.

The PCHIP interpolant and the cubic Hermite interpolant of
`pianistpy.numerics.pchip` are differentiated in the table's values, slopes and
nodes. The derivative of `hermite_inverse` is checked by the implicit function
rule.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy.numerics import hermite, hermite_inverse
from pianistpy.numerics.pchip import hermite_eval, interp1d, pchip_slopes

X_NODES = jnp.array([0.0, 0.4, 1.1, 1.7, 2.5, 3.0])
F_NODES = X_NODES + 0.3 * jnp.sin(X_NODES)  # F_NODES increases with X_NODES.
SLOPES = 1.0 + 0.3 * jnp.cos(X_NODES)
# POINTS holds a point on a node, two points between nodes, and one point
# beyond each end node.
POINTS = jnp.array([1.1, 0.73, 2.21, -0.3, 3.4])


def _central(f, p, h):
    out = []
    for j in range(p.size):
        step = jnp.zeros(p.size).at[j].set(h)
        out.append((np.asarray(f(p + step)) - np.asarray(f(p - step))) / (2.0 * h))
    return np.stack(out, axis=-1)


@pytest.mark.parametrize("wrt", ["values", "nodes"])
def test_the_pchip_interpolant_differentiates_in_its_table(wrt):
    # interp1d with order "cubic" forms the PCHIP slopes from the table and
    # evaluates the cubic Hermite interpolant with those slopes. The forward
    # derivatives (jacfwd) and the reverse derivatives (jacrev) are compared
    # with each other and with central differences of step 1e-6.
    def f(p):
        xp, fp = (p, F_NODES) if wrt == "nodes" else (X_NODES, p)
        return interp1d(POINTS, xp, fp, fp[0], fp[-1])

    p = X_NODES if wrt == "nodes" else F_NODES
    forward, reverse = np.asarray(jax.jacfwd(f)(p)), np.asarray(jax.jacrev(f)(p))
    np.testing.assert_allclose(forward, reverse, rtol=1e-13, atol=1e-15)
    inside = np.asarray((POINTS >= X_NODES[0]) & (POINTS <= X_NODES[-1]))
    central = _central(jax.jit(f), p, 1e-6)
    np.testing.assert_allclose(central[inside], reverse[inside], rtol=1e-6, atol=1e-9)
    # The derivatives of the PCHIP slopes alone are compared with central
    # differences of the PCHIP slopes.
    slopes = jax.jacfwd(
        lambda q: pchip_slopes(*((q, F_NODES) if wrt == "nodes" else (X_NODES, q)))
    )
    np.testing.assert_allclose(
        _central(
            jax.jit(
                lambda q: pchip_slopes(
                    *((q, F_NODES) if wrt == "nodes" else (X_NODES, q))
                )
            ),
            p,
            1e-6,
        ),
        np.asarray(slopes(p)),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize("wrt", ["values", "slopes", "nodes"])
def test_the_hermite_interpolant_differentiates_in_its_table(wrt):
    # hermite_eval is the jax.numpy form of `numerics.hermite` and must give
    # the same values as `numerics.hermite` between the nodes.
    def f(p):
        args = {
            "values": (X_NODES, p, SLOPES),
            "slopes": (X_NODES, F_NODES, p),
            "nodes": (p, F_NODES, SLOPES),
        }[wrt]
        return hermite_eval(POINTS, *args, 0.0, 0.0)

    inside = np.asarray((POINTS >= X_NODES[0]) & (POINTS <= X_NODES[-1]))
    numpy_form, _ = hermite(
        np.asarray(POINTS), *(np.asarray(a) for a in (X_NODES, F_NODES, SLOPES))
    )
    np.testing.assert_allclose(
        np.asarray(f({"values": F_NODES, "slopes": SLOPES, "nodes": X_NODES}[wrt]))[
            inside
        ],
        numpy_form[inside],
        rtol=1e-15,
    )
    p = {"values": F_NODES, "slopes": SLOPES, "nodes": X_NODES}[wrt]
    forward, reverse = np.asarray(jax.jacfwd(f)(p)), np.asarray(jax.jacrev(f)(p))
    np.testing.assert_allclose(forward, reverse, rtol=1e-13, atol=1e-15)
    central = _central(jax.jit(f), p, 1e-6)
    np.testing.assert_allclose(central[inside], reverse[inside], rtol=1e-6, atol=1e-9)


def test_the_numpy_hermite_is_not_traced():
    # `numerics.hermite` and `hermite_inverse` run in NumPy and cannot be
    # traced by JAX. A gradient goes through `pchip.hermite_eval` instead, and
    # the gradient of the inverse through the implicit function rule.
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(lambda v: hermite(1.0, np.asarray(X_NODES), v, np.asarray(SLOPES))[0])(
            F_NODES
        )


@pytest.mark.parametrize("target", [0.9, 1.75, 2.6])
def test_the_hermite_inverse_by_the_implicit_rule(target):
    # The root x(theta) solves H(x; theta) = target, so at the root
    # dx/dtheta = -(dH/dtheta) / (dH/dx). The implicit derivative is compared
    # with central differences of `hermite_inverse`.
    table = np.asarray(F_NODES)
    x, _ = hermite_inverse(target, np.asarray(X_NODES), table, np.asarray(SLOPES))

    def h(xq, values):
        return hermite_eval(xq, X_NODES, values, SLOPES, 0.0, 0.0)

    dh_dx = jax.grad(h, argnums=0)(jnp.float64(x), F_NODES)
    dh_dv = jax.grad(h, argnums=1)(jnp.float64(x), F_NODES)
    implicit = -np.asarray(dh_dv) / float(dh_dx)
    step = 1e-6
    central = np.array(
        [
            (
                hermite_inverse(
                    target, np.asarray(X_NODES), table + step * e, np.asarray(SLOPES)
                )[0]
                - hermite_inverse(
                    target, np.asarray(X_NODES), table - step * e, np.asarray(SLOPES)
                )[0]
            )
            / (2.0 * step)
            for e in np.eye(table.size)
        ]
    )
    np.testing.assert_allclose(central, implicit, rtol=1e-6, atol=1e-9)
