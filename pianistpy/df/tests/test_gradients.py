"""Tests of the gradient contract: derivatives through the package's solves and
integrals against closed forms and finite differences."""

import math

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix
import pytest

# the stopping problem: y' = -k y, z' = y from (y0, 0), stopped by an Event at
# y = c, a Dopri5 solve of the shape of the truncation's edge continuation;
# s* = ln(y0 / c) / k and z* = (y0 - c) / k
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


# d(s*, z*) / d(k, y0)
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
    # the event's time and the state there, through the root find: 1.05e-10 of
    # the closed form measured (the solve's own s* is 3e-13), and central
    # differences at 1e-6 relative agree to 1e-6
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
    # diffrax's default adjoint is a custom VJP: forward mode through it raises,
    # so a forward derivative names ForwardMode
    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jacfwd(lambda p: _stop(p, diffrax.RecursiveCheckpointAdjoint()))(P)
