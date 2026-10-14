"""The gradient contract on the table interpolants: PCHIP and the cubic Hermite
differentiated in the table's values and nodes, and the Hermite's inverse by
the implicit rule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy.numerics import hermite, hermite_inverse
from pianistpy.numerics.pchip import hermite_eval, interp1d, pchip_slopes

X_NODES = jnp.array([0.0, 0.4, 1.1, 1.7, 2.5, 3.0])
F_NODES = X_NODES + 0.3 * jnp.sin(X_NODES)  # increasing
SLOPES = 1.0 + 0.3 * jnp.cos(X_NODES)
# at a node, between nodes, and beyond each end
POINTS = jnp.array([1.1, 0.73, 2.21, -0.3, 3.4])


def _central(f, p, h):
    out = []
    for j in range(p.size):
        step = jnp.zeros(p.size).at[j].set(h)
        out.append((np.asarray(f(p + step)) - np.asarray(f(p - step))) / (2.0 * h))
    return np.stack(out, axis=-1)


@pytest.mark.parametrize("wrt", ["values", "nodes"])
def test_the_pchip_interpolant_differentiates_in_its_table(wrt):
    # interp1d's cubic: the PCHIP slopes formed from the table and the Hermite
    # through them; forward (jacfwd) and reverse (jacrev) against central
    # differences at 1e-6
    def f(p):
        xp, fp = (p, F_NODES) if wrt == "nodes" else (X_NODES, p)
        return interp1d(POINTS, xp, fp, fp[0], fp[-1])

    p = X_NODES if wrt == "nodes" else F_NODES
    forward, reverse = np.asarray(jax.jacfwd(f)(p)), np.asarray(jax.jacrev(f)(p))
    np.testing.assert_allclose(forward, reverse, rtol=1e-13, atol=1e-15)
    inside = np.asarray((POINTS >= X_NODES[0]) & (POINTS <= X_NODES[-1]))
    central = _central(jax.jit(f), p, 1e-6)
    np.testing.assert_allclose(central[inside], reverse[inside], rtol=1e-6, atol=1e-9)
    # the slopes alone, the interpolant's table
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
    # hermite_eval, the jax.numpy form of `numerics.hermite`: the same values
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
    # `numerics.hermite` and `hermite_inverse` are NumPy: a gradient goes through
    # `pchip.hermite_eval` and the implicit rule below
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(lambda v: hermite(1.0, np.asarray(X_NODES), v, np.asarray(SLOPES))[0])(
            F_NODES
        )


@pytest.mark.parametrize("target", [0.9, 1.75, 2.6])
def test_the_hermite_inverse_by_the_implicit_rule(target):
    # x(theta) solves H(x; theta) = target: dx/dtheta = -(dH/dtheta) / (dH/dx) at
    # the root, against central differences of `hermite_inverse` itself
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
