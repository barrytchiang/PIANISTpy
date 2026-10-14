r"""The angle integral of the ellipse route,
:math:`\Lambda = \int_0^\pi P^{-\beta_c}\,{\rm d}\phi`,
:math:`P = (c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi`."""

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
from scipy.special import ellipe, ellipkm1, elliprj

from ..typing import Array
from ._ellipse_rules import ANGLE_TOL, gl01

# The grid of the angle rule's need in (log10 a_+, log10 a_-): its start, step
# and points.
_U0, _DU, _NU = -16.0, 0.05, 371
_TABLE: dict[str, Any] = {}


def _roots(c1: Array, c2: Array, c3: Array, e: Array) -> tuple[Array, Array]:
    """z_+ - 1 and -1 - z_- of P(z) = (c1^2 + c3) - 2 c1 c2 z - e z^2, each
    without a subtraction of near-equal numbers."""
    d = np.sqrt(c1 * c1 * c2 * c2 + e * (c1 * c1 + c3))
    zm = -(c1 * c2 + d) / e
    dp = (c1 - c2) ** 2 / (e * (1.0 - zm))
    return dp, (c1 + c2) ** 2 / (e * (2.0 + dp))


def _need(ap: Array, am: Array, lam: Array, lpi: Array) -> Array:
    """The least n (real) with rho^(-2n) <= ANGLE_TOL on both halves, rho over
    every image of P's zeros."""
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
    """Return the angle rule's need on the (log10 a_+, log10 a_-) grid, built
    once a process."""
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
    """Return each node's index into `counts`, and ``len(counts)`` where no
    listed count reaches ANGLE_TOL: the largest need of the four corners of
    the node's cell."""
    g = angle_table()
    with np.errstate(all="ignore"):
        dp, dm = _roots(c1, c2, c3, e)
        x = (
            np.log10(np.maximum(np.log1p(dp + np.sqrt(dp * (dp + 2.0))), 1e-300)) - _U0
        ) / _DU
        y = (
            np.log10(np.maximum(np.log1p(dm + np.sqrt(dm * (dm + 2.0))), 1e-300)) - _U0
        ) / _DU
    i = np.clip(np.floor(np.nan_to_num(x, nan=0.0)), 0, _NU - 2).astype(np.intp)
    j = np.clip(np.floor(np.nan_to_num(y, nan=0.0)), 0, _NU - 2).astype(np.intp)
    need = np.maximum(
        np.maximum(g[i, j], g[i + 1, j]), np.maximum(g[i, j + 1], g[i + 1, j + 1])
    )
    return np.searchsorted(np.asarray(counts, dtype=np.float64), need)


def lam_half(c1: Array, c3: Array, e: Array, dm: Array) -> Array:
    r"""Return :math:`\Lambda` at :math:`\beta_c = 1/2` in closed form.

    :math:`2 K(m) / \sqrt{T}`, :math:`T = c_1^2 + c_3 + e + 2\sqrt{c_3(c_1^2
    + e)}`, :math:`1 - m = (d_m / T)^2`, with :math:`e = c_3 - c_2^2` and
    :math:`d_m = c_1^2 - c_2^2` passed in, never formed by subtraction.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = c1 * c1 + c3 + e + 2.0 * np.sqrt(c3 * (c1 * c1 + e))
        return 2.0 * ellipkm1((dm / t) ** 2) / np.sqrt(t)


def lam_mhalf(c1: Array, c2: Array, e: Array, dm: Array) -> Array:
    r"""Return :math:`\Lambda` at :math:`\beta_c = -1/2` in closed form.

    :math:`2\sqrt{M}\,[E(k^2) + (1 - k^2)(\Pi(n|k^2) - K(k^2))]` with
    :math:`B = \max(c_1^2, c_2^2)`, :math:`M = B + e`,
    :math:`1 - k^2 = |d_m|/M`, :math:`n = \min(c_1^2, c_2^2)/B` and
    :math:`\Pi - K = (n/3) R_J(0, 1 - k^2, 1, 1 - n)`; at :math:`d_m = 0` the
    term's limit :math:`{\rm arsinh}\sqrt{h}/\sqrt{h(1+h)}`, :math:`h = e/B`.
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
        rj = elliprj(
            0.0,
            np.where(ok, k1, 1.0),
            1.0,
            np.where(ok, adm / np.where(b_ > 0.0, b_, 1.0), 1.0),
        )
        lim = np.arcsinh(np.sqrt(h)) / np.sqrt(h * (1.0 + h))
        term = np.where(ok, k1 * (n / 3.0) * rj, np.where(n > 0.0, lim, 0.0))
        return 2.0 * np.sqrt(m_) * (ellipe(1.0 - k1) + term)


class _Power:
    r""":math:`P \mapsto P^{-\beta_c}`, by square roots where :math:`\beta_c`
    is a quarter or a half; a class, so that a route carrying it pickles."""

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
    r"""Return :math:`P \mapsto P^{-\beta_c}`, by square roots where
    :math:`\beta_c` is a quarter or a half."""
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
    Gauss nodes a half.

    Each half of :math:`[0, \pi]` is dense toward its own minimum of
    :math:`P`, of width `lam` at 0 and `lpi` at :math:`\pi`:
    :math:`\phi = \lambda\sinh(\mu x)` on :math:`[0, \pi/2]`.
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
            # phi = pi - psi: P = (c1 + c2 cos psi)^2 + c3 sin^2 psi
            p = np.cos(psi) * c2[..., None]
            p += c1[..., None]
            np.multiply(p, p, out=p)
            p += c3[..., None] * np.sin(psi, out=psi) ** 2
        else:
            # P = d^2 + h (2 kappa + (c2^2 - c3) h), h = 1 - cos phi
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
