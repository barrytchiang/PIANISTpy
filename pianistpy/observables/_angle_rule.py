r"""The angle integral of the LOSVD table's ellipse route, as a JAX kernel.

The LOSVD table behind `losvd`, `aperture_losvd` and `LosTable` calls
`angle_rule` (through `EllipseRoute`) for every model whose DF carries the
factor :math:`(L^2)^{-\beta_{\rm c}}` of the angular momentum :math:`L`.
`angle_rule` evaluates, at each node of the velocity plane,

.. math::

    \Lambda = \int_0^\pi P(\phi)^{-\beta_{\rm c}}\,{\rm d}\phi, \qquad
    P(\phi) = (c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi.

In the integral, :math:`\beta_{\rm c}` is the central anisotropy. The angle :math:`\phi`
parametrizes one level set (an ellipse) of the energy variable on the
velocity plane. :math:`P` is proportional to :math:`L^2` on that ellipse,
with a factor independent of :math:`\phi`. The coefficients :math:`c_1`,
:math:`c_2` and :math:`c_3` belong to the node, and the route forms them.

The rule splits :math:`[0, \pi]` at :math:`\pi/2`. The half next to
:math:`\phi = 0` is mapped by :math:`\phi = \lambda_0\sinh(\mu x)`, with
:math:`x \in [0, 1]`, :math:`\mu = {\rm arsinh}(\pi/(2\lambda_0))` and
:math:`\lambda_0` the width of the minimum of :math:`P` at :math:`\phi = 0`
(the argument `lam`). The half next to :math:`\phi = \pi` is mapped the same
way in :math:`\pi - \phi`, with the width :math:`\lambda_\pi` of the minimum
of :math:`P` at :math:`\phi = \pi` (the argument `lpi`). Each half takes
`n` Gauss-Legendre nodes in :math:`x`.

The route passes, for the nodes of one count `n`, eight float64 arrays of
one length: :math:`c_1`, :math:`c_2`, :math:`c_3`,
:math:`\kappa = c_1 c_2 + c_3 - c_2^2`, :math:`d^2 = (c_1 - c_2)^2`,
:math:`e_2 = c_2^2 - c_3`, `lam` and `lpi`. The route clips `lam` and `lpi`
to :math:`[10^{-15}, \pi]`. `angle_rule` returns :math:`\Lambda` at each
node, an array of the same length. The DF, the quadrature weights and the
prefactor multiply :math:`\Lambda` in the route after `angle_rule` returns,
so the DF is not an argument of `angle_rule`.

The route does not call `angle_rule` in two cases. A model without the
factor :math:`(L^2)^{-\beta_{\rm c}}` (isotropic, Osipkov-Merritt, or
:math:`\beta_{\rm c} = 0`) has no angle integral. At
:math:`\beta_{\rm c} = \pm 1/2` the route takes :math:`\Lambda` in closed
form (`lam_half`, `lam_mhalf`). `angle_rule` still evaluates
:math:`\beta_{\rm c} = \pm 1/2`, and the tests compare that value with the
closed form.

The kernel `_kernel` performs the operations of the numpy rule
`lam_rule` in the same order on each pair of a node and an angle node. The
kernel sums over the angle nodes with ``jnp.sum``, and `lam_rule` sums with
``np.einsum``. The two rules therefore agree to rounding, not bit for bit.
The two rules share the Gauss-Legendre nodes (`gauss_legendre_unit`) and the
square-root forms of the power (`power_kind`).

The node arrays are padded to a power of two, at least `BUCKET_MIN`, with
values at which :math:`P = 1`, and the padding is cut off on return. The
kernel takes :math:`\beta_{\rm c}` and the Gauss-Legendre nodes as traced
arrays. One compilation therefore serves every model with the same count
`n`, the same padded length and the same power form.

The rule's node count is chosen in `_angle.angle_count`, to the tolerance
:math:`10^{-8}` (``ANGLE_TOL``). Notes Section T:6.4 and Eq. (T:6.2) state
the route.
"""

__all__ = ["BUCKET_MIN", "angle_rule", "power_kind"]

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics.quadrature import gauss_legendre_unit
from ..typing import Array

# The smallest padded length of the node arrays of one call of angle_rule.
BUCKET_MIN = 256
# The padding values of c1, c2, c3, kap, d2, e2, lam and lpi, in that order.
# At the padding values, P = 1 on both halves of the angle interval.
_PAD = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
# The values of beta_c at which P^(-beta_c) is taken by square roots, mapped to
# the form's name.
_KINDS = {-0.5: "sqrt", 0.5: "rsqrt", -0.25: "qrt", 0.25: "rqrt"}


def power_kind(beta_c: float) -> str:
    r"""Return the name of the form that evaluates :math:`P^{-\beta_{\rm c}}`.

    The name is ``"sqrt"``, ``"rsqrt"``, ``"qrt"`` or ``"rqrt"`` where
    :math:`\beta_{\rm c}` is :math:`-1/2`, :math:`1/2`, :math:`-1/4` or
    :math:`1/4`, and ``"pow"`` for every other :math:`\beta_{\rm c}`."""
    return _KINDS.get(float(beta_c), "pow")


def _kernel(b, c1, c2, c3, kap, d2, e2, lam, lpi, xp, wp, kind):
    r"""Return :math:`\Lambda` at each node, the sum of the rule's two halves.

    The argument `b` is :math:`\beta_{\rm c}`. The arguments `c1` to `lpi` are
    the node arrays of the module docstring. The arguments `xp` and `wp` are
    the Gauss-Legendre nodes and weights on :math:`[0, 1]`. The argument
    `kind` is the form of the power (`power_kind`). `_rows_theta` also calls
    `_kernel` under JAX's differentiation, with ``kind="pow"``."""

    def powm(p):
        if kind == "sqrt":
            return jnp.sqrt(p)
        if kind == "rsqrt":
            return 1.0 / jnp.sqrt(p)
        if kind == "qrt":
            return jnp.sqrt(jnp.sqrt(p))
        if kind == "rqrt":
            return 1.0 / jnp.sqrt(jnp.sqrt(p))
        return jnp.exp(-b * jnp.log(p))

    total: Array = 0.0
    for lm, to_pi in ((lam, False), (lpi, True)):
        mu = jnp.arcsinh(0.5 * jnp.pi / lm)
        ex = jnp.exp(mu[:, None] * xp[None, :])
        ih = 0.5 / ex
        half = 0.5 * ex
        psi = (half - ih) * lm[:, None]
        wpsi = (half + ih) * (lm * mu)[:, None]
        if to_pi:
            # On the half next to pi, phi = pi - psi, and
            # P = (c1 + c2 cos psi)^2 + c3 sin^2 psi.
            p = jnp.cos(psi) * c2[:, None] + c1[:, None]
            p = p * p + c3[:, None] * jnp.sin(psi) ** 2
        else:
            # On the half next to 0, phi = psi, and with h = 1 - cos phi,
            # P = d^2 + h (2 kappa + (c2^2 - c3) h). The code forms h as
            # 2 sin^2(phi / 2).
            h = jnp.sin(0.5 * psi)
            h = h * h * 2.0
            p = (h * e2[:, None] + 2.0 * kap[:, None]) * h + d2[:, None]
        p = jnp.maximum(p, 1e-300)
        total = total + jnp.sum(wpsi * powm(p) * wp[None, :], axis=1)
    return total


_jit_kernel = jax.jit(_kernel, static_argnames=("kind",))


def angle_rule(
    beta_c: float,
    c1: Array,
    c2: Array,
    c3: Array,
    kap: Array,
    d2: Array,
    e2: Array,
    lam: Array,
    lpi: Array,
    n: int,
) -> Array:
    r"""Return :math:`\Lambda` at each node by the angle rule at `n` nodes on
    each half of :math:`[0, \pi]`.

    Parameters
    ----------
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`, the exponent of the factor
        :math:`(L^2)^{-\beta_{\rm c}}`.
    c1, c2, c3, kap, d2, e2, lam, lpi : array
        The node arrays :math:`c_1`, :math:`c_2`, :math:`c_3`,
        :math:`\kappa`, :math:`d^2`, :math:`e_2`, :math:`\lambda_0` and
        :math:`\lambda_\pi` of the module docstring, all of one shape.
    n : int
        Number of Gauss-Legendre nodes on each half of :math:`[0, \pi]`.

    Returns
    -------
    array
        :math:`\Lambda` at each node, with the shape of `c1`.
    """
    shape = np.shape(c1)
    m = int(np.prod(shape))
    size = max(BUCKET_MIN, 1 << max(m - 1, 0).bit_length())
    args = []
    for a_, pad in zip((c1, c2, c3, kap, d2, e2, lam, lpi), _PAD, strict=True):
        buf = np.full(size, pad)
        buf[:m] = np.ravel(a_)
        args.append(buf)
    xp, wp = gauss_legendre_unit(int(n))
    out = _jit_kernel(float(beta_c), *args, xp, wp, kind=power_kind(beta_c))
    return np.asarray(out)[:m].reshape(shape)
