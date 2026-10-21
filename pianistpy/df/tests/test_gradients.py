"""Tests of derivatives through a diffrax solve stopped by an event.

The derivatives from JAX are compared with closed forms and with central
finite differences.
"""

import math

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix
import pytest

# The stopping problem integrates y' = -k y and z' = y from (y, z) = (y0, 0)
# with Dopri5 and stops at the event y = c. The problem has the shape of the
# solve that continues the truncated model to its edge. The stopping time is
# s* = ln(y0 / c) / k and the state there is z* = (y0 - c) / k.
K, Y0, C = 0.7, 2.0, 0.1


def _stop(p, adjoint):
    k, y0 = p[0], p[1]
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: jnp.stack([-k * y[0], y[0]])),
        diffrax.Dopri5(),
        0.0,
        100.0,
        1e-3,
        jnp.stack([y0, 0.0]),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-14),
        saveat=diffrax.SaveAt(t1=True),
        event=diffrax.Event(
            cond_fn=lambda t, y, args, **kwargs: y[0] - C,
            root_finder=optimistix.Newton(rtol=1e-13, atol=1e-15),
        ),
        max_steps=100_000,
        adjoint=adjoint,
    )
    return jnp.stack([sol.ts[-1], sol.ys[-1, 1]])


# The exact Jacobian d(s*, z*) / d(k, y0): rows s* and z*, columns k and y0.
EXACT = np.array(
    [[-math.log(Y0 / C) / K**2, 1.0 / (K * Y0)], [-(Y0 - C) / K**2, 1.0 / K]]
)
P = jnp.array([K, Y0])


@pytest.mark.parametrize(
    ("mode", "adjoint"),
    [
        ("reverse", diffrax.RecursiveCheckpointAdjoint()),  # diffrax's default
        ("forward", diffrax.ForwardMode()),
    ],
)
def test_the_stopping_state_differentiates_through_the_event(mode, adjoint):
    # The derivatives of the event's time s* and of the state z* at s* pass
    # through the event's root find. The JAX Jacobian must match the closed
    # form to a relative 1e-9. Central differences with a relative step of
    # 1e-6 must match the JAX Jacobian to a relative 1e-6.
    jac = (jax.jacrev if mode == "reverse" else jax.jacfwd)(lambda p: _stop(p, adjoint))
    jac = np.asarray(jac(P))
    np.testing.assert_allclose(jac, EXACT, rtol=1e-9)
    plain = jax.jit(lambda p: _stop(p, diffrax.RecursiveCheckpointAdjoint()))
    for j in range(2):
        h = 1e-6 * float(P[j])
        step = jnp.zeros(2).at[j].set(h)
        fd = (np.asarray(plain(P + step)) - np.asarray(plain(P - step))) / (2.0 * h)
        np.testing.assert_allclose(fd, jac[:, j], rtol=1e-6)


def test_forward_mode_needs_the_forward_adjoint():
    # The default adjoint of diffrax is a custom VJP, and forward-mode
    # differentiation through a custom VJP raises TypeError. A forward-mode
    # derivative therefore requires the ForwardMode adjoint.
    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jacfwd(lambda p: _stop(p, diffrax.RecursiveCheckpointAdjoint()))(P)
