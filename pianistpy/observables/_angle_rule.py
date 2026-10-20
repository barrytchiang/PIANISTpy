r"""The angle rule of the ellipse route's rows as a JAX kernel.

:math:`\Lambda = \int_0^\pi P^{-\beta_c}\,{\rm d}\phi`,
:math:`P = (c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi`, by the split sinh-mapped
rule at `n` Gauss-Legendre nodes a half: each half of :math:`[0, \pi]` is
dense toward its own minimum of :math:`P`, of width `lam` at 0 and `lpi` at
:math:`\pi`, :math:`\phi = \lambda\sinh(\mu x)` on :math:`[0, \pi/2]`,
:math:`\mu = {\rm arsinh}(\pi / 2\lambda)`.

The interface (the LOSVD stage's rows). The rows pass, for the nodes
of one count group, flat float64 arrays of one length: :math:`c_1`,
:math:`c_2`, :math:`c_3`, :math:`\kappa = c_1 c_2 + c_3 - c_2^2`,
:math:`d^2 = (c_1 - c_2)^2`, :math:`e_2 = c_2^2 - c_3`, and the two widths
`lam` and `lpi` (each clipped to :math:`[10^{-15}, \pi]`), all formed by the
rows exactly as for the numpy rule; with :math:`\beta_c` and the count `n`.
They take back :math:`\Lambda` at each node, an array of the same length.
Nothing else enters: the DF, the weights and the prefactor multiply
:math:`\Lambda` in the rows, after this call, so F is not an argument here.

Where it is not called: a model without the L factor (isotropic, Osipkov-Merritt,
:math:`\beta_c = 0`) has no angle integral, and at :math:`\beta_c = \pm 1/2`
the rows take :math:`\Lambda` in closed form; the kernel computes the rule
there too, as the rule's control.

The kernel holds the numpy rule's operations in the same order on each
(node, angle node) element; the sum over the angle nodes is ``jnp.sum`` where
the numpy rule has ``einsum``, so the two agree at rounding, not to the bit
(measured over the battery: the rows' D within :math:`9.1 \times 10^{-16}` of
the numpy rows). What the two share is written once: the Gauss-Legendre nodes
(`gauss_legendre_unit`) and the power's closed forms (`power_kind`). Fixed
shapes: the node arrays are padded to a power of two, at least `BUCKET_MIN`,
with values that keep :math:`P` positive, and cut off on return;
:math:`\beta_c` and the Gauss nodes are traced, so one compilation per (count,
padded length, power form) serves every model.
"""

__all__ = ["BUCKET_MIN", "angle_rule", "power_kind"]

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics.quadrature import gauss_legendre_unit
from ..typing import Array

# The least padded length of a call.
BUCKET_MIN = 256
# The values the padding takes, for c1 c2 c3 kap d2 e2 lam lpi: P = 1 there.
_PAD = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
# The exponents whose power is taken by square roots.
_KINDS = {-0.5: "sqrt", 0.5: "rsqrt", -0.25: "qrt", 0.25: "rqrt"}


def power_kind(beta_c: float) -> str:
    r"""Return the form of :math:`P^{-\beta_c}`: square roots where
    :math:`\beta_c` is a half or a quarter, else ``"pow"``."""
    return _KINDS.get(float(beta_c), "pow")


def _kernel(b, c1, c2, c3, kap, d2, e2, lam, lpi, xp, wp, kind):
    """Lambda at each node: the two halves of the rule, summed."""

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
            # phi = pi - psi: P = (c1 + c2 cos psi)^2 + c3 sin^2 psi
            p = jnp.cos(psi) * c2[:, None] + c1[:, None]
            p = p * p + c3[:, None] * jnp.sin(psi) ** 2
        else:
            # P = d^2 + h (2 kappa + (c2^2 - c3) h), h = 1 - cos phi
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
    r"""Return :math:`\Lambda` at each node by the angle rule at `n` nodes a
    half.

    Parameters
    ----------
    beta_c : float
        The anisotropy exponent of the L factor.
    c1, c2, c3, kap, d2, e2, lam, lpi : array
        The rows' node arrays of one shape (the module's docstring).
    n : int
        Gauss-Legendre nodes on each half of :math:`[0, \pi]`.

    Returns
    -------
    array
        :math:`\Lambda`, of the shape of `c1`.
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
