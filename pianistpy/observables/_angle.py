r"""The angle integral of the LOSVD table's ellipse route, in numpy.

The LOSVD table behind `losvd`, `aperture_losvd` and `LosTable` reads the
functions of `_angle` (through `EllipseRoute`) for every model whose DF
carries the factor :math:`(L^2)^{-\beta_{\rm c}}` of the angular momentum
:math:`L`. The integral is

.. math::

    \Lambda = \int_0^\pi P(\phi)^{-\beta_{\rm c}}\,{\rm d}\phi, \qquad
    P(\phi) = (c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi,

with :math:`\beta_{\rm c}` the central anisotropy, :math:`\phi` the angle
that parametrizes one level set (an ellipse) of the energy variable on the
velocity plane, and :math:`c_1`, :math:`c_2`, :math:`c_3` the coefficients
of one node, which the route forms.

`angle_count` chooses each node's number of
Gauss-Legendre nodes so that the rule's error is below
:math:`10^{-8}` (``ANGLE_TOL``). `lam_half` and `lam_mhalf` give
:math:`\Lambda` in closed form at :math:`\beta_{\rm c} = 1/2` and
:math:`\beta_{\rm c} = -1/2`. `lam_rule` is the numpy form of the split
sinh-mapped rule, of which `_angle_rule.angle_rule` is the JAX form.
Notes Section T:6.4 and Eq. (T:6.2) state the route.
"""

__all__ = [
    "angle_count",
    "angle_table",
    "lam_half",
    "lam_mhalf",
    "lam_rule",
    "power_of",
]

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import ellipkm1

from ..exceptions import AccuracyError
from ..typing import Array
from ._ellipse_rules import ANGLE_TOL, gl01

# The grid of the need table in log10 a_+ and in log10 a_-, where a_+ and a_- are
# the imaginary parts of the zeros of P on the lines Re phi = 0 and Re phi = pi.
# The grid starts at -16, steps by 0.05 and has 371 points on each axis, so the
# grid ends at 2.5.
_U0, _DU, _NU = -16.0, 0.05, 371
# The number of steps of Bulirsch's cel iteration, the same for every element.
# The iteration converges quadratically. With 14 steps, E(k^2) agrees with
# scipy's to 1e-15 for k_c down to 1e-300, and the third-kind term agrees to
# 1e-15 for k_c down to 1e-150. Both agreements are below the angle rule's
# tolerance 1e-8 (ANGLE_TOL, a hundredth of 1e-6).
CEL_STEPS = 14
# The need table of angle_table, held once a process under the key "grid".
_TABLE: dict[str, Any] = {}


def _roots(c1: Array, c2: Array, c3: Array, e: Array) -> tuple[Array, Array]:
    """Return :math:`d_+ = z_+ - 1` and :math:`d_- = -1 - z_-` for the two
    roots :math:`z_+` and :math:`z_-` of the quadratic
    :math:`P(z) = (c_1^2 + c_3) - 2 c_1 c_2 z - e z^2`.

    In the quadratic, :math:`z = \\cos\\phi`, and the caller passes
    :math:`e = c_3 - c_2^2`. The function forms :math:`z_-` first, then
    :math:`d_+ = (c_1 - c_2)^2 / (e(1 - z_-))` and
    :math:`d_- = (c_1 + c_2)^2 / (e(2 + d_+))`. Neither :math:`d_+` nor
    :math:`d_-` is therefore the difference of a root and :math:`\\pm 1`."""
    d = np.sqrt(c1 * c1 * c2 * c2 + e * (c1 * c1 + c3))
    zm = -(c1 * c2 + d) / e
    dp = (c1 - c2) ** 2 / (e * (1.0 - zm))
    return dp, (c1 + c2) ** 2 / (e * (2.0 + dp))


def _need(ap: Array, am: Array, lam: Array, lpi: Array) -> Array:
    """Return the need: the smallest real node count n at which
    ``rho**(-2 * n) <= ANGLE_TOL`` holds on both halves of the angle rule.

    The arguments `ap` and `am` are :math:`a_+` and :math:`a_-`, the
    imaginary parts of the zeros of P on the lines :math:`{\\rm Re}\\,\\phi = 0`
    and :math:`{\\rm Re}\\,\\phi = \\pi`. The arguments `lam` and `lpi` are the
    widths of the sinh maps of the two halves. On each half, the zeros of P
    with real parts :math:`-\\pi`, 0, :math:`\\pi` and :math:`2\\pi` are mapped
    to the Gauss-Legendre variable on :math:`[-1, 1]` by five branches of the
    inverse sinh map. The value rho is the smallest Bernstein-ellipse
    parameter over all the mapped zeros of both halves.
    Gauss-Legendre's error at n nodes falls as ``rho**(-2 * n)``. A need that
    is not finite is returned as infinity."""
    rmin = np.full(np.shape(lam), np.inf)
    for lm, sing in (
        (lam, (1j * ap, np.pi + 1j * am, -np.pi + 1j * am, 2 * np.pi + 1j * ap)),
        (lpi, (1j * am, np.pi + 1j * ap, -np.pi + 1j * ap, 2 * np.pi + 1j * am)),
    ):
        mu = np.arcsinh(0.5 * np.pi / lm)
        for z in sing:
            w0 = np.arcsinh(z / lm)
            for w in (
                w0,
                1j * np.pi - w0,
                w0 + 2j * np.pi,
                w0 - 2j * np.pi,
                1j * np.pi - w0 - 2j * np.pi,
            ):
                t = 2.0 * (w / mu) - 1.0
                r = np.abs(t + np.sqrt(t - 1.0) * np.sqrt(t + 1.0))
                rmin = np.minimum(rmin, np.maximum(r, 1.0 / np.maximum(r, 1e-300)))
    need = np.log(1.0 / ANGLE_TOL) / (2.0 * np.log(rmin))
    return np.where(np.isfinite(need), need, np.inf)


def angle_table() -> Array:
    """Return the need table: the need (`_need`) on the grid of
    :math:`\\log_{10} a_+` and :math:`\\log_{10} a_-`, built once a process.

    The table's entry (i, j) is the need at :math:`a_+ = 10^{u_i}` and
    :math:`a_- = \\max(10^{u_i}, 10^{u_j})`, where :math:`u_i` is the grid's
    i-th point. The widths `lam` and `lpi` at each entry are the widths the
    route forms, written in :math:`d_\\pm = \\cosh a_\\pm - 1`, and clipped to
    :math:`[10^{-15}, \\pi]` as the route clips them."""
    if "grid" not in _TABLE:
        a = 10.0 ** (_U0 + _DU * np.arange(_NU))
        ap = np.broadcast_to(a[:, None], (_NU, _NU))
        am = np.maximum(ap, a[None, :])
        dp, dm = 2.0 * np.sinh(0.5 * ap) ** 2, 2.0 * np.sinh(0.5 * am) ** 2
        cc = 0.5 * (dm - dp)
        with np.errstate(all="ignore"):
            lam = np.clip(np.sqrt(dp * (2.0 + dm) / (1.0 + cc)), 1e-15, np.pi)
            lpi = np.clip(
                np.where(
                    1.0 - cc > 0.0,
                    np.sqrt((2.0 + dp) * dm / np.maximum(1.0 - cc, 1e-300)),
                    np.pi,
                ),
                1e-15,
                np.pi,
            )
            _TABLE["grid"] = _need(ap, am, lam, lpi)
    return _TABLE["grid"]


def angle_count(
    c1: Array, c2: Array, c3: Array, e: Array, counts: tuple[int, ...]
) -> Array:
    """Return each node's index into `counts`.

    A node's need is the largest need (`angle_table`) at the four corners of
    the grid cell that holds the node's :math:`(a_+, a_-)`. The index is the
    position of the smallest count in `counts` that is at least the node's
    need. The index is ``len(counts)`` where every count in `counts` is below
    the node's need. At the index ``len(counts)``, `EllipseRoute` takes its
    full rule of ``nphi // 2`` nodes on each half. A node whose :math:`a_+`
    or :math:`a_-` lies below the grid's first point :math:`10^{-16}` or above
    its last point :math:`10^{2.5}` also takes the index ``len(counts)``. A
    root of P at an end of the angle interval gives :math:`a_+ = 0` or
    :math:`a_- = 0`, which lies below the grid's first point.

    Raises
    ------
    AccuracyError
        If :math:`a_+` or :math:`a_-` is NaN at any node."""
    g = angle_table()
    with np.errstate(all="ignore"):
        dp, dm = _roots(c1, c2, c3, e)
        x = (
            np.log10(np.maximum(np.log1p(dp + np.sqrt(dp * (dp + 2.0))), 1e-300)) - _U0
        ) / _DU
        y = (
            np.log10(np.maximum(np.log1p(dm + np.sqrt(dm * (dm + 2.0))), 1e-300)) - _U0
        ) / _DU
    if np.isnan(x).any() or np.isnan(y).any():
        raise AccuracyError(
            f"{int(np.isnan(x).sum() + np.isnan(y).sum())} of the roots a_+ and a_- "
            "of the angle integral are not numbers (NaN), so the number of angle "
            "nodes at each node cannot be chosen."
        )
    off = (x < 0.0) | (y < 0.0) | (x > _NU - 1) | (y > _NU - 1)
    # A node on the grid has floor(x) at most _NU - 1. The minimum with _NU - 2
    # puts a node at the grid's last point into the last cell, whose upper
    # corner is that point.
    i = np.minimum(np.floor(np.where(off, 0.0, x)), _NU - 2).astype(np.intp)
    j = np.minimum(np.floor(np.where(off, 0.0, y)), _NU - 2).astype(np.intp)
    need = np.maximum(
        np.maximum(g[i, j], g[i + 1, j]), np.maximum(g[i, j + 1], g[i + 1, j + 1])
    )
    need = np.where(off, np.inf, need)
    return np.searchsorted(np.asarray(counts, dtype=np.float64), need)


def lam_half(c1: Array, c3: Array, e: Array, dm: Array) -> Array:
    r"""Return :math:`\Lambda` at :math:`\beta_{\rm c} = 1/2` in closed form.

    The closed form is :math:`\Lambda = 2K(m)/\sqrt{T}`, with :math:`K` the
    complete elliptic integral of the first kind of parameter :math:`m`,
    :math:`T = c_1^2 + c_3 + e + 2\sqrt{c_3(c_1^2 + e)}` and
    :math:`1 - m = (d_m/T)^2`. The argument `e` is :math:`c_3 - c_2^2` and the
    argument `dm` is :math:`d_m = c_1^2 - c_2^2`. The caller forms `e` and
    `dm`, and `lam_half` does not form them from :math:`c_1`, :math:`c_2` and
    :math:`c_3`. `scipy.special.ellipkm1` evaluates :math:`K` from
    :math:`1 - m`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = c1 * c1 + c3 + e + 2.0 * np.sqrt(c3 * (c1 * c1 + e))
        return 2.0 * ellipkm1((dm / t) ** 2) / np.sqrt(t)


def cel(kc: Array, p: Array, a: Array, b: Array) -> Array:
    r"""Return Bulirsch's general complete elliptic integral for `p` > 0.

    The integral is

    .. math::

        {\rm cel}(k_c, p, a, b) = \int_0^{\pi/2}
        \frac{a\cos^2\theta + b\sin^2\theta}
        {(\cos^2\theta + p\sin^2\theta)
        \sqrt{\cos^2\theta + k_c^2\sin^2\theta}}\,{\rm d}\theta.

    The function evaluates the integral by Bulirsch's quadratically
    convergent iteration (Bulirsch 1969, Numer. Math. 13, 305). Every element
    takes `CEL_STEPS` steps of the same arithmetic. An element's value
    therefore does not depend on the other elements of the call. At
    :math:`k_c = 0` the iteration does not converge, and the caller
    `lam_mhalf` does not use the value of `cel` at :math:`k_c = 0`.
    """
    shape = np.broadcast_shapes(np.shape(kc), np.shape(p), np.shape(a), np.shape(b))

    def full(x: Any) -> Array:
        return np.broadcast_to(np.asarray(x, dtype=np.float64), shape).copy()

    qc = full(np.abs(np.asarray(kc, dtype=np.float64)))
    p = full(np.sqrt(np.asarray(p, dtype=np.float64)))
    a = full(a)
    b = full(b)
    b /= p
    e, em = qc.copy(), np.ones(shape)
    f, g = np.empty(shape), np.empty(shape)
    for _ in range(CEL_STEPS):
        np.copyto(f, a)
        a += b / p
        np.divide(e, p, out=g)
        b += f * g
        b *= 2.0
        p += g
        em += qc
        np.sqrt(e, out=qc)
        qc *= 2.0
        np.multiply(qc, em, out=e)
    return 0.5 * np.pi * (b + a * em) / (em * (em + p))


def lam_mhalf(c1: Array, c2: Array, e: Array, dm: Array) -> Array:
    r"""Return :math:`\Lambda` at :math:`\beta_{\rm c} = -1/2` in closed form.

    The closed form is

    .. math::

        \Lambda = 2\sqrt{M}\,\{E(k^2) + (1 - k^2)[\Pi(n|k^2) - K(k^2)]\},

    with :math:`E`, :math:`K` and :math:`\Pi` the complete elliptic integrals
    of the second, first and third kinds. The quantities are
    :math:`B = \max(c_1^2, c_2^2)`, :math:`M = B + e`,
    :math:`k_c^2 = 1 - k^2 = |d_m|/M` and :math:`n = \min(c_1^2, c_2^2)/B`.
    The argument `e` is :math:`c_3 - c_2^2` and the argument `dm` is
    :math:`d_m = c_1^2 - c_2^2`, both formed by the caller.

    Bulirsch's `cel` gives both terms. The first term is
    :math:`E = {\rm cel}(k_c, 1, 1, k_c^2)`, and :math:`E = 1` at
    :math:`k_c = 0`. The second term is
    :math:`k_c^2(\Pi - K) = {\rm cel}(k_c, 1 - n, 0, k_c^2 n)`, with
    :math:`1 - n = |d_m|/B`. At :math:`d_m = 0` and :math:`n > 0` the second
    term takes its limit :math:`{\rm arsinh}\sqrt{h}/\sqrt{h(1+h)}`, with
    :math:`h = e/B`. At :math:`n = 0` the second term is zero.
    """
    with np.errstate(all="ignore"):
        c1s, c2s = c1 * c1, c2 * c2
        b_ = np.maximum(c1s, c2s)
        m_ = b_ + e
        adm = np.abs(dm)
        k1 = adm / m_
        n = np.where(b_ > 0.0, np.minimum(c1s, c2s) / np.where(b_ > 0.0, b_, 1.0), 0.0)
        ok = (n > 0.0) & (adm > 0.0)
        h = e / np.where(b_ > 0.0, b_, 1.0)
        kc = np.sqrt(k1)
        e_k = np.where(k1 > 0.0, cel(kc, 1.0, 1.0, k1), 1.0)
        pi_k = cel(
            np.where(ok, kc, 1.0),
            np.where(ok, adm / np.where(b_ > 0.0, b_, 1.0), 1.0),
            0.0,
            np.where(ok, k1 * n, 0.0),
        )
        lim = np.arcsinh(np.sqrt(h)) / np.sqrt(h * (1.0 + h))
        term = np.where(ok, pi_k, np.where(n > 0.0, lim, 0.0))
        return 2.0 * np.sqrt(m_) * (e_k + term)


class _Power:
    r"""The map :math:`P \mapsto P^{-\beta_{\rm c}}` of the numpy rule.

    The map uses square roots where :math:`\beta_{\rm c}` is :math:`\pm 1/4`
    or :math:`\pm 1/2`, and ``exp(-beta_c * log(P))`` for every other
    :math:`\beta_{\rm c}`. The map is a class rather than a closure so that an
    `EllipseRoute` that holds the map can be pickled."""

    def __init__(self, beta_c: float) -> None:
        self.b = beta_c

    def __call__(self, p: Array) -> Array:
        b = self.b
        if b == -0.5:
            return np.sqrt(p)
        if b == 0.5:
            return 1.0 / np.sqrt(p)
        if b == -0.25:
            return np.sqrt(np.sqrt(p))
        if b == 0.25:
            return 1.0 / np.sqrt(np.sqrt(p))
        return np.exp(-b * np.log(p))


def power_of(beta_c: float) -> Callable[[Array], Array]:
    r"""Return the map :math:`P \mapsto P^{-\beta_{\rm c}}` (`_Power`).

    The map uses square roots where :math:`\beta_{\rm c}` is :math:`\pm 1/4`
    or :math:`\pm 1/2`."""
    return _Power(beta_c)


def lam_rule(
    powm: Callable[[Array], Array],
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
    r"""Return :math:`\Lambda` by the split sinh-mapped angle rule at `n`
    Gauss-Legendre nodes on each half of :math:`[0, \pi]`.

    The half next to :math:`\phi = 0` is mapped by
    :math:`\phi = \lambda_0\sinh(\mu x)`, with :math:`x \in [0, 1]`,
    :math:`\mu = {\rm arsinh}(\pi/(2\lambda_0))` and :math:`\lambda_0` the
    width `lam` of the minimum of :math:`P` at :math:`\phi = 0`. The half next
    to :math:`\phi = \pi` is mapped the same way in :math:`\pi - \phi`, with
    the width `lpi` of the minimum of :math:`P` at :math:`\phi = \pi`. The
    argument `powm` is the map :math:`P \mapsto P^{-\beta_{\rm c}}`
    (`power_of`). The arguments `c1` to `lpi` are the node arrays of
    `_angle_rule`'s module docstring. `_angle_rule.angle_rule` is the JAX
    form of `lam_rule`.
    """
    xp, wp = gl01(n)
    total: Any = 0.0
    for lm, to_pi in ((lam, False), (lpi, True)):
        mu = np.arcsinh(0.5 * np.pi / lm)
        ex = np.exp(mu[..., None] * xp)
        ih = 0.5 / ex
        half = 0.5 * ex
        psi = (half - ih) * lm[..., None]
        wpsi = (half + ih) * (lm * mu)[..., None]
        del ex, ih, half
        if to_pi:
            # On the half next to pi, phi = pi - psi, and
            # P = (c1 + c2 cos psi)^2 + c3 sin^2 psi.
            p = np.cos(psi) * c2[..., None]
            p += c1[..., None]
            np.multiply(p, p, out=p)
            p += c3[..., None] * np.sin(psi, out=psi) ** 2
        else:
            # On the half next to 0, phi = psi, and with h = 1 - cos phi,
            # P = d^2 + h (2 kappa + (c2^2 - c3) h). The code forms h as
            # 2 sin^2(phi / 2).
            h = np.sin(0.5 * psi, out=psi)
            np.multiply(h, h, out=h)
            h *= 2.0
            p = h * e2[..., None]
            p += 2.0 * kap[..., None]
            p *= h
            p += d2[..., None]
            del h
        del psi
        np.maximum(p, 1e-300, out=p)
        total = total + np.einsum("...k,...k,k->...", wpsi, powm(p), wp)
        del p, wpsi
    return total
