r"""The inner amplitude and the central deficit of a truncated model.

The truncated solve (`truncated.py`) calls `center_amplitude` for two
results, which the fill of the truncated model's inner rows (`fill.py`)
reads: the amplitude :math:`C` of the inner density series that the rows
near the center use, and the central deficit. A
phase-space truncation at the energy :math:`\mathcal{K}_{\rm t}` removes the
orbits with energies below :math:`\mathcal{K}_{\rm t}` and subtracts the
constant :math:`F(\mathcal{K}_{\rm t})` from the DF above
:math:`\mathcal{K}_{\rm t}`. The auxiliary density that the truncation removes
near the center is the central deficit :math:`D(u)`, a smooth function of the
depth :math:`u = P_0 - \Phi` below the central potential :math:`P_0`.

On a cusp with :math:`0 < \gamma` and :math:`2 \beta_{\rm c} \le \gamma`, the
inner amplitude is the profile's own amplitude :math:`C`, and
`lowered_center` solves the deficit at all orders as a fixed point on a grid
uniform in :math:`\ln r`. On a core (:math:`\gamma \le 0`), or where
:math:`2 \beta_{\rm c} > \gamma`, the inner amplitude is :math:`C` times the
median, over the solve's rows at depths between :math:`10^{-4}` and
:math:`3 \times 10^{-4}` below :math:`P_0`, of the ratio of the solve's
density to the inner density series, and no deficit is carried.

The module also holds the inner series of the double-power-law density
:math:`\rho = C r^{-\gamma} (1 + r^{\alpha})^{e}`, with
:math:`e = (\gamma - \beta) / \alpha`, :math:`\alpha` the transition slope,
:math:`\beta` the outer slope and :math:`\gamma` the inner slope: the
density's closed form, and the enclosed mass and the depth summed term by
term. The Chebyshev interpolant of the deficit and the reach of the inner
depth series serve the bound :math:`10^{-7}`, one tenth of :math:`10^{-6}`.

References
----------
Notes Section T:3.4, Notes Section T:1.2.2.
"""

__all__ = [
    "CenterAmplitude",
    "anisotropy_g",
    "central_deficit",
    "center_amplitude",
    "ceiling_depth",
    "inner_density_series",
    "inner_depth_series",
    "inner_mass_series",
    "join_depth",
    "lowered_center",
    "representable_end",
]

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..anisotropy import Anisotropy
from ..exceptions import AccuracyError, InputError
from ..validation import TIER_BOUNDS
from .poisson import cuddeford_prefactor, density_cut
from .truncation import INNER_CEILING, K_FLOOR


@dataclass(frozen=True)
class CenterAmplitude:
    r"""The inner amplitude and the central deficit of a truncated model.

    Parameters
    ----------
    c_trunc : float
        The amplitude :math:`C` of the inner density series that the rows
        near the center use, in internal units.
    delta : float
        The first-order deficit :math:`\delta_1`, the first-order solution
        for the missing auxiliary density :math:`\delta(r)` of
        `lowered_center`, in internal units. Zero where no deficit is
        carried: an untruncated model, a core with :math:`\gamma \le 0`, or
        a model with :math:`2 \beta_{\rm c} > \gamma`.
    lowered : dict or None
        The dict that `lowered_center` returns, with the truncated central
        potential :math:`P_0` added under the key ``"P0"``. None where no
        deficit is carried.
    deficit_degree : int, optional
        The degree of the Chebyshev interpolant of the deficit :math:`D(u)`:
        the first degree of `DEFICIT_DEGREES` whose largest residual at the
        midpoints between the interpolant's nodes is at most :math:`10^{-7}`
        of the largest absolute deficit at the nodes. Default: ``0``, which
        means no interpolant was fitted.
    """

    c_trunc: float
    delta: float
    lowered: dict[str, Any] | None
    deficit_degree: int = 0


# The degrees of the Chebyshev interpolant of the deficit D(u), tried in
# increasing order. The first degree whose largest residual at the midpoints
# between its nodes is at most 1e-7 of the largest absolute deficit at its
# nodes is kept, where 1e-7 is one tenth of the bound 1e-6. A deficit that no
# listed degree fits raises AccuracyError.
DEFICIT_DEGREES = (8, 12, 16, 24, 32, 48, 64)


def inner_density_series(r, c, gamma, alpha, beta):
    r"""Return the double-power-law density in closed form.

    The density is :math:`\rho(r) = C r^{-\gamma} (1 + r^{\alpha})^{e}`
    with :math:`e = (\gamma - \beta) / \alpha`, at the user-specified 3D
    radii ``r`` in internal units. The argument ``c`` is the amplitude
    :math:`C`, and ``gamma``, ``alpha`` and ``beta`` are :math:`\gamma`,
    :math:`\alpha` and :math:`\beta`.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 2.
    """
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the enclosed mass diverges")
    r = np.asarray(r, dtype=np.float64)
    e = (gamma - beta) / alpha
    return c * (r ** (-gamma)) * (1.0 + r**alpha) ** e


def inner_mass_series(r, c, gamma, alpha, beta):
    r"""Return the enclosed mass of the first four terms of the inner density series.

    The density :math:`C r^{-\gamma} (1 + r^{\alpha})^{e}`, with
    :math:`e = (\gamma - \beta) / \alpha`, is expanded as
    :math:`C \sum_k b_k r^{k \alpha - \gamma}`, with :math:`b_k` the binomial
    coefficient of :math:`e` over :math:`k`. The enclosed mass of the terms
    :math:`k = 0` to 3 is
    :math:`m(r) = C \sum_k b_k r^{3 - \gamma + k \alpha} / (3 - \gamma + k \alpha)`,
    without the factor :math:`4 \pi`, in internal units.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 3.
    """
    if not gamma < 3.0:
        raise InputError(f"gamma = {gamma:.4f} >= 3: the mass diverges at the origin")
    r = np.asarray(r, dtype=np.float64)
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (3.0 - g + k * a) / (3.0 - g + k * a)
    return c * acc


def inner_depth_series(r, c, gamma, alpha, beta):
    r"""Return the depth of the first four terms of the inner density series.

    The depth below the central potential :math:`P_0` is
    :math:`u(r) = P_0 - \Phi(r)`, the integral of :math:`m(r') / r'^2` from
    0 to :math:`r`, with :math:`m` the enclosed mass of `inner_mass_series`:
    :math:`u(r) = C \sum_k b_k r^{2 - \gamma + k \alpha} /
    [(2 - \gamma + k \alpha) (3 - \gamma + k \alpha)]` for :math:`k = 0` to
    3, in internal units.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 2.
    """
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the depth diverges at the origin")
    r = np.asarray(r, dtype=np.float64)
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (2.0 - g + k * a) / (
            (2.0 - g + k * a) * (3.0 - g + k * a)
        )
    return c * acc


def join_depth(c, gamma, alpha, beta, tol):
    r"""Return the largest depth at which the four-term depth series meets ``tol``.

    On 5801 radii uniform in :math:`\log r` from :math:`10^{-14}` to
    :math:`10^{0.5}`, the absolute value of the fifth term of the depth
    series of `inner_depth_series` is divided by the sum of the first four
    terms, where that sum is positive. The largest radius at which the ratio
    is at most ``tol`` is the join.

    Returns
    -------
    u_join, r_join, bound : float
        The four-term depth at the join, the join radius, and the ratio of
        the fifth term to the four-term depth there.

    Raises
    ------
    AccuracyError
        If the ratio of the fifth term to the four-term depth exceeds ``tol``
        at every radius of the grid.
    """
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    r = np.logspace(-14.0, 0.5, 5801)
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (2.0 - g + k * a) / (
            (2.0 - g + k * a) * (3.0 - g + k * a)
        )
    bk *= (e - 3) / 4
    nxt = np.abs(bk) * r ** (2.0 - g + 4 * a) / ((2.0 - g + 4 * a) * (3.0 - g + 4 * a))
    bound = np.where(acc > 0.0, nxt / np.maximum(acc, 1e-300), np.inf)
    ok = np.nonzero(np.isfinite(bound) & (bound <= tol))[0]
    if ok.size == 0:
        raise AccuracyError(
            f"the inner series reaches no depth at bound {tol:.1e} for "
            f"(gamma, alpha, beta) = ({g}, {a}, {float(beta)})"
        )
    k = int(ok.max())
    return float(c) * float(acc[k]), float(r[k]), float(bound[k])


def ceiling_depth(c, gamma, alpha, beta, tol):
    r"""Return the largest depth at which the four-term mass series meets ``tol``.

    On 5801 radii uniform in :math:`\log r` from :math:`10^{-14}` to
    :math:`10^{0.5}`, the absolute value of the fifth term of the mass
    series of `inner_mass_series` is divided by the sum of the first four
    terms, where that sum is positive. The largest radius at which the ratio
    is at most ``tol`` is the ceiling radius. The truncated model's enclosed
    mass is read from the series inside the ceiling radius.

    Returns
    -------
    u_ceil, r_ceil, bound : float
        The four-term depth (`inner_depth_series`) at the ceiling radius, the
        ceiling radius, and the ratio of the fifth term to the four-term mass
        there.

    Raises
    ------
    AccuracyError
        If the ratio of the fifth term to the four-term mass exceeds ``tol``
        at every radius of the grid.
    """
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    r = np.logspace(-14.0, 0.5, 5801)
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (3.0 - g + k * a) / (3.0 - g + k * a)
    bk *= (e - 3) / 4
    nxt = np.abs(bk) * r ** (3.0 - g + 4 * a) / (3.0 - g + 4 * a)
    bound = np.where(acc > 0.0, nxt / np.maximum(acc, 1e-300), np.inf)
    ok = np.nonzero(np.isfinite(bound) & (bound <= tol))[0]
    if ok.size == 0:
        raise AccuracyError(
            f"the inner mass series reaches no depth at bound {tol:.1e} for "
            f"(gamma, alpha, beta) = ({g}, {a}, {float(beta)})"
        )
    k = int(ok.max())
    u = float(inner_depth_series(np.array([r[k]]), c, g, a, float(beta))[0])
    return u, float(r[k]), float(bound[k])


def central_deficit(q, df, k_t, p_0, bc, f_at_k_t=None):
    r"""Return the first five Taylor coefficients of the central deficit at ``p_0``.

    The central deficit :math:`D(u)` is the auxiliary density that a truncation
    at the energy ``k_t`` removes at the depth u below the central potential
    ``p_0``:

    .. math::

        D(u) = C \int_0^{\mathcal{K}_{\rm t}} F(Q)\,
               [2 (P_0 - u - Q)]^n \, {\rm d}Q
               + C F(\mathcal{K}_{\rm t})
               \int_{\mathcal{K}_{\rm t}}^{P_0 - u} [2 (P_0 - u - Q)]^n \, {\rm d}Q,

    with :math:`n = 1/2 - \beta_{\rm c}`, C the Cuddeford prefactor
    (`cuddeford_prefactor`), F the untruncated DF,
    :math:`\mathcal{K}_{\rm t}` the truncation energy ``k_t`` and :math:`P_0`
    the central potential ``p_0``. The first integral is the contribution of
    the orbits with energies below :math:`\mathcal{K}_{\rm t}`. The second
    integral is the contribution of the constant :math:`F(\mathcal{K}_{\rm t})`
    that the truncation subtracts from the DF above
    :math:`\mathcal{K}_{\rm t}`, integrated in closed form. The table's rows
    used are those with a positive energy :math:`Q`, a finite and positive
    :math:`F`, and :math:`Q \le 4 \mathcal{K}_{\rm t}`. The first integral
    uses 400 Gauss-Legendre nodes in :math:`\ln Q`, from the first energy of
    the rows used to :math:`\mathcal{K}_{\rm t}`, through a PCHIP
    interpolant of :math:`\ln F` against :math:`\ln Q` on the rows used. The
    first integral omits the energies below the first energy of the rows
    used.

    Parameters
    ----------
    q : array
        The energies of the untruncated DF table, in internal units.
    df : array
        The values of the untruncated DF table at ``q``, in internal units.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal units.
    p_0 : float
        The central potential at which the series is formed, in internal
        units.
    bc : float
        The central anisotropy :math:`\beta_{\rm c}`.
    f_at_k_t : float or None, optional
        The value :math:`F(\mathcal{K}_{\rm t})`. Default: ``None``, which
        means :math:`F(\mathcal{K}_{\rm t})` is read from the PCHIP
        interpolant of the table.

    Returns
    -------
    d_0, d_1, d_2, d_3, d_4 : float
        The coefficients :math:`D_k` of
        :math:`D(u) = \sum_k (-1)^k D_k u^k`, so that :math:`D_0` is the
        deficit at :math:`u = 0` and :math:`-D_1` is the derivative
        :math:`{\rm d}D/{\rm d}u` at :math:`u = 0`.

    Raises
    ------
    InputError
        If the table holds fewer than 2 rows with a positive energy, a
        finite and positive DF, and :math:`Q \le 4 \mathcal{K}_{\rm t}`.
    """
    return _deficit_reader(q, df, k_t, bc, f_at_k_t)(p_0)


def _deficit_reader(q, df, k_t, bc, f_at_k_t=None):
    r"""Return `central_deficit` as a function ``at(p_0, terms=5)``.

    The PCHIP interpolant of the table, the Gauss-Legendre nodes and weights
    and the value :math:`F(\mathcal{K}_{\rm t})` do not depend on the central
    potential :math:`P_0`, so `_deficit_reader` forms the interpolant, the
    nodes, the weights and the value :math:`F(\mathcal{K}_{\rm t})` once.
    The returned function ``at`` gives the first ``terms`` coefficients
    :math:`D_k` of the central deficit at the central potential ``p_0``.
    The argument ``terms`` defaults to 5.

    Raises
    ------
    InputError
        If the table holds fewer than 2 rows with a positive energy, a
        finite and positive DF, and :math:`Q \le 4 \mathcal{K}_{\rm t}`.
    """
    qd = np.asarray(q, dtype=np.float64)
    fd = np.asarray(df, dtype=np.float64)
    ok = (qd > 0.0) & np.isfinite(fd) & (fd > 0.0) & (qd <= 4.0 * float(k_t))
    if int(ok.sum()) < 2:
        raise InputError(
            f"the DF table holds {int(ok.sum())} positive rows below 4 K_t = "
            f"{4.0 * float(k_t):.3e}: the deficit has no table to read"
        )
    log_f = PchipInterpolator(np.log(qd[ok]), np.log(fd[ok]))
    x, w = np.polynomial.legendre.leggauss(400)
    la, lb = float(np.log(qd[ok][0])), float(np.log(float(k_t)))
    lq = 0.5 * (lb - la) * x + 0.5 * (lb + la)
    qn = np.exp(lq)
    n = 0.5 - float(bc)
    wq = 0.5 * (lb - la) * w * np.exp(log_f(lq)) * qn
    pref = float(cuddeford_prefactor(float(bc)))
    fz = float(np.exp(log_f(lb))) if f_at_k_t is None else float(f_at_k_t)

    def at(p_0, terms=5):
        ph = float(p_0) - float(k_t)
        # The coefficient D_k is (-1)^k / k! times the k-th u-derivative at
        # u = 0. The kernel (2 (P_0 - u - Q))^n contributes
        # 2^k binom(n, k) (2 (P_0 - Q))^(n - k) to D_k. The closed-form term
        # (P_0 - K_t - u)^(n + 1) contributes
        # binom(n + 1, k) (P_0 - K_t)^(n + 1 - k) to D_k.
        out = []
        b_n = b_n1 = 1.0
        for k in range(terms):
            if k > 0:
                b_n *= (n - (k - 1)) / k
                b_n1 *= (n + 1.0 - (k - 1)) / k
            d_k = pref * float(
                np.sum(wq * 2.0**k * b_n * (2.0 * (float(p_0) - qn)) ** (n - k))
            )
            x_k = pref * fz * 2.0**n * b_n1 * ph ** (n + 1.0 - k) / (n + 1.0)
            out.append(d_k + x_k)
        return tuple(out)

    return at


def anisotropy_g(r, bc, kind, ra):
    r"""Return the factor g(R) of the augmented density at the radii ``r``.

    The factor is `Anisotropy.factor` of the anisotropy with central
    anisotropy :math:`\beta_{\rm c}` = ``bc``, anisotropy radius
    :math:`R_{\rm a}` = ``ra`` and Type ``kind`` ("I" or "II"). A ``kind``
    of None means no anisotropy radius: :math:`R_{\rm a}` is infinite and
    ``ra`` is not read. The radii ``r`` are in internal units.
    """
    aniso = Anisotropy(
        beta_c=float(bc),
        r_a=float(ra) if kind is not None else math.inf,
        kind=kind if kind is not None else "I",
    )
    return aniso.factor(np.asarray(r, dtype=np.float64))


def _series(r, c, g, a, b, shift, terms=12):
    """Return the untruncated inner mass or depth series to ``terms`` terms.

    The series is the mass for ``shift`` 3 and the depth for ``shift`` 2.
    The mass is c SUM_k b_k r^x / x with x = 3 - g + k a. The depth is
    c SUM_k b_k r^x / (x (x + 1)) with x = 2 - g + k a. Here b_k is the
    binomial coefficient of e = (g - b) / a over k. The mass carries no
    factor 4 pi.
    """
    e = (g - b) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(terms):
        if k > 0:
            bk *= (e - (k - 1)) / k
        w = (3.0 - g + k * a) if shift == 3 else (2.0 - g + k * a) * (3.0 - g + k * a)
        acc = acc + bk * r ** (shift - g + k * a) / w
    return c * acc


def _series_step(r, lg, c, g, a, b, shift, terms=12):
    """Return the difference _series(r e^lg) - _series(r), summed term by term.

    Each term's difference c b_k r^x (e^(x lg) - 1) / w is formed through
    expm1, so the two nearly equal sums _series(r e^lg) and _series(r) are
    never subtracted from each other.
    """
    e = (g - b) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(terms):
        if k > 0:
            bk *= (e - (k - 1)) / k
        x = shift - g + k * a
        w = (3.0 - g + k * a) if shift == 3 else (2.0 - g + k * a) * (3.0 - g + k * a)
        acc = acc + bk * r**x * np.expm1(x * lg) / w
    return c * acc


def _log_aux_step(r, lg, g, a, b, bc, kind, ra):
    """Return ln G(r e^lg) - ln G(r) for the untruncated auxiliary density G.

    The untruncated auxiliary density is G(x) = x^(2 bc) A(x) rho_N(x), with
    rho_N the untruncated density and A(x) = (1 + (x / ra)^2)^(1 - bc) for
    Type I, (1 - (x / ra)^2)^(1 - bc) for Type II, and 1 for a ``kind``
    without an anisotropy radius. Every term of the difference is formed
    through log1p and expm1, so no two nearly equal logarithms are
    subtracted.
    """
    ra2 = (r / float(ra)) ** 2 if kind in ("I", "II") else 0.0
    out = (2.0 * float(bc) - g) * lg + ((g - b) / a) * np.log1p(
        r**a * np.expm1(a * lg) / (1.0 + r**a)
    )
    if kind == "I":
        out = out + (1.0 - float(bc)) * np.log1p(ra2 * np.expm1(2.0 * lg) / (1.0 + ra2))
    elif kind == "II":
        out = out + (1.0 - float(bc)) * np.log1p(
            -ra2 * np.expm1(2.0 * lg) / (1.0 - ra2)
        )
    return out


def _cum_power(t, h, w):
    """Return the running integral of h r^(w - 1) dr from t[0] to each node of t = ln r.

    The integrand is h e^(w t) dt in t = ln r. The power e^(w t) is exact;
    h is the cubic spline through the nodes in t. Each interval between two
    consecutive nodes takes a 5-point Gauss-Legendre rule. The first entry of
    the result is 0.
    """
    cs = CubicSpline(t, h)
    x, wt = np.polynomial.legendre.leggauss(5)
    a, b = t[:-1], t[1:]
    tm = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (b + a)[:, None]
    pan = np.sum(0.5 * (b - a)[:, None] * wt[None, :] * cs(tm) * np.exp(w * tm), axis=1)
    return np.concatenate([[0.0], np.cumsum(pan)])


def _map_bracketed(r, uh, start, c, g, a, b, iters=200):
    """Return the radius r' in (0, r] where the untruncated depth equals u_hat.

    `lowered_center` calls `_map_bracketed` for the nodes whose linear start
    r - du r^2 / M lies outside (0, r]. The root r' is found by Newton steps
    on the depth series, with a bisection step wherever a Newton step leaves
    the current bracket. The iteration stops after 200 steps, or when, at
    every node, the change of r' in one step is at most 4 times the machine
    epsilon times r'.

    Raises
    ------
    AccuracyError
        If u_hat is not positive, or if the untruncated depth at r is below
        u_hat. In both cases the interval (0, r] holds no root r'.
    """
    lo, hi = np.zeros_like(r), r.copy()
    f_hi = _series(hi, c, g, a, b, 2) - uh
    bad = ~((uh > 0.0) & (f_hi >= 0.0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AccuracyError(
            f"the lowered center's map has no root in (0, r] at r = {r[i]:.6e}: "
            f"u_hat = {uh[i]:.6e}, the depth at r less u_hat {f_hi[i]:+.6e}, the "
            f"linear step's start {start[i]:+.6e}"
        )
    x = 0.5 * hi
    for _ in range(iters):
        f = _series(x, c, g, a, b, 2) - uh
        lo, hi = np.where(f < 0.0, x, lo), np.where(f >= 0.0, x, hi)
        step = x - f * x**2 / _series(x, c, g, a, b, 3)
        inside = np.isfinite(step) & (step > lo) & (step < hi)
        new = np.where(inside, step, 0.5 * (lo + hi))
        done = np.abs(new - x) <= 4.0 * np.finfo(np.float64).eps * new
        x = new
        if bool(np.all(done)):
            break
    return x


def _response(r, t, omf, w3, lg, uh, c, g, a, b, bc, d_of_u, kind, ra):
    """Return the response of the fixed-point update to a uniform unit change of delta.

    The response is J 1 = s K 1, with J the Jacobian of the update and 1
    the vector of ones over the nodes. The factor K 1 is the change of the
    lowered model's depth shift du caused by a unit change of delta at every
    node, through the deficit's mass dm. The factor s is the derivative of
    the updated delta with respect to du: s = -D'(u_hat) + G(r') Lambda'(l)
    r' / M(r'), with G the untruncated auxiliary density, M the untruncated
    mass series, r' = r e^l, and Lambda'(l) = d ln G(r e^l) / dl.
    `lowered_center` states the largest value of the response in its
    AccuracyError message.
    """
    k1 = _cum_power(t, _cum_power(t, 1.0 / omf, w3) / r**w3, w3 - 1.0)
    rp = r * np.exp(lg)
    g_p = c * rp ** (-g) / anisotropy_g(rp, bc, kind, ra)
    g_p = g_p * (1.0 + rp**a) ** ((g - b) / a)
    dlam = 2.0 * bc - g + (g - b) * rp**a / (1.0 + rp**a)
    if kind in ("I", "II"):
        x2 = (rp / float(ra)) ** 2 * (1.0 if kind == "I" else -1.0)
        dlam = dlam + (1.0 - bc) * 2.0 * x2 / (1.0 + x2)
    s = -np.asarray(d_of_u.deriv()(uh), dtype=np.float64) + g_p * dlam * rp / _series(
        rp, c, g, a, b, 3
    )
    return s * k1


def lowered_center(
    r_lo, r_hi, c, g, a, b, bc, d_of_u, kind, ra, n_per_decade=200, tol=1e-15
):
    r"""Return the lowered model's central deficit at all orders.

    The profile is c r^(-g) (1 + r^a)^((g - b) / a) with g, a, b the slopes
    gamma, alpha, beta, and the central anisotropy is ``bc``
    (:math:`\beta_{\rm c}`), with :math:`\gamma \ge 2 \beta_{\rm c}`.
    ``d_of_u`` is the central deficit D(u) as a Chebyshev series in the
    depth u, with a ``deriv`` method.

    At a fixed depth u, the truncated auxiliary density is G_N(u) - D(u),
    with G_N the untruncated auxiliary density as a function of depth. At a
    fixed radius r, the lowered model is shallower than the untruncated
    model by du(r), the Poisson depth of the missing density
    delta(r) g(r), with g the anisotropy factor (`anisotropy_g`):
    dm(r) = INT_0^r delta g r'^2 dr' and du(r) = INT_0^r dm / r'^2 dr'. The
    missing auxiliary density at r is

        delta(r) = D(u_hat) - [G(r') - G(r)],

    with u_hat = u_N(r) - du(r), u_N the untruncated depth, G(x) =
    x^(2 bc) A(x) rho_N(x) the untruncated auxiliary density as a function
    of radius, and r' the radius at which the untruncated depth equals
    u_hat. The equation for delta is iterated as a fixed point (Picard
    iteration) on a grid uniform in ln r from ``r_lo`` to ``r_hi``, with
    ``n_per_decade`` nodes per decade, starting from delta_1 at every node.
    The first-order solution is delta_1 = D(0) / (1 + kappa), with
    kappa = (g - 2 bc) (3 - g) / ((3 - 2 bc) (2 - 2 bc)). The second-order
    solution delta_1 + delta_2 r^p, with p = g - 2 bc, is a closed form used
    below ``r_lo``.

    The radius r' = r e^l is solved for l from U_N(r') - U_N(r) = -du(r),
    by Newton steps with dU_N/dl = M_N(r') / r', where

        U_N(r') - U_N(r) = c SUM_k b_k r^(e_k) expm1(e_k l) / (e_k (e_k + 1)),

    e_k = 2 - g + k a, and b_k is the binomial coefficient of (g - b) / a
    over k. The difference of G is G(r') - G(r) = G(r) expm1(Lambda), with

    .. math::

        \Lambda = (2 \beta_{\rm c} - \gamma)\, l
            + \frac{\gamma - \beta}{\alpha}\,
            {\rm log1p}\left[\frac{r^{\alpha}\, {\rm expm1}(\alpha l)}
            {1 + r^{\alpha}}\right]
            + (1 - \beta_{\rm c})\,
            {\rm log1p}\left[\frac{s\, (r / R_{\rm a})^2\, {\rm expm1}(2 l)}
            {1 + s\, (r / R_{\rm a})^2}\right],

    where :math:`R_{\rm a}` is the anisotropy radius ``ra``, and the last
    term of :math:`\Lambda` enters for ``kind`` "I" (s = +1) and "II"
    (s = -1) only. Neither difference subtracts two nearly equal numbers. The Newton
    steps start from the linear step r - du r^2 / M_N(r); at a node where
    the linear step lies outside (0, r], `_map_bracketed` solves for r'
    instead.

    The iteration stops when the largest change of delta in one iteration,
    divided by delta_1, is below ``tol``, or after 200 iterations. An
    iteration that ends without meeting ``tol`` is at its rounding floor when
    the last change at every node is at most the machine epsilon
    :math:`2^{-52}` times |G(r')| + |G(r)| at the node.

    Returns
    -------
    lowered : dict
        ``at``: a function of radii in (0, ``r_hi``] returning
        (delta, du, dm), with dm carrying no factor 4 pi; the closed form
        below ``r_lo`` and cubic splines in ln r of delta, ln du and ln dm
        above ``r_lo``. ``u_hat_of_R``: a cubic spline of u_hat in ln r.
        ``delta_1``, ``delta_2`` and ``p``: the closed-form coefficients.
        ``iters``: the number of iterations. ``err``: the last largest
        change divided by delta_1. ``floor_stopped``: True where the
        iteration ended at its rounding floor. ``floor_node_r`` and
        ``floor_size``: the radius of the node with the largest last change,
        and the rounding bound at the node with the largest last change
        divided by delta_1, both NaN where ``floor_stopped`` is False. ``R_lo``,
        ``R_hi``: the grid's ends.
        ``bc``: the central anisotropy.

    Raises
    ------
    AccuracyError
        If an iteration leaves the domain of the map: the interval (0, r]
        holds no root r', the updated delta is not finite, or dm / r^(3 - 2 bc)
        is not finite because r^(3 - 2 bc) underflows or dm overflows at the
        grid's inner nodes. The message states the largest response of the update to a
        uniform change of delta (`_response`) and the change of delta at the
        last iteration on which the map was defined.
    InputError
        If ``at`` is called at a radius beyond ``r_hi``.
    """
    lo, hi = float(r_lo), float(r_hi)
    n = int(np.ceil(np.log10(hi / lo) * n_per_decade)) + 1
    t = np.linspace(np.log(lo), np.log(hi), n)
    r = np.exp(t)
    p = g - 2.0 * bc
    sx = p / (2.0 - g)

    def amp(m):
        return (2.0 - g) * (3.0 - g) / ((2.0 - 2.0 * bc + m) * (3.0 - 2.0 * bc + m))

    def rho_n(x):
        return c * x ** (-g) * (1.0 + x**a) ** ((g - b) / a)

    d1 = float(d_of_u(np.zeros(1))[0]) / (1.0 + sx * amp(0.0))
    d2 = -sx * (sx + 1.0) * amp(0.0) ** 2 * d1**2 / (2.0 * c * (1.0 + sx * amp(p)))
    g_r = anisotropy_g(r, bc, kind, ra)
    omf = r ** (-2.0 * bc) / g_r
    u_n, m_n = _series(r, c, g, a, b, 2), _series(r, c, g, a, b, 3)
    g_n = rho_n(r) / g_r
    w3 = 3.0 - 2.0 * bc
    dm0 = d1 * lo**w3 / w3 + d2 * lo ** (w3 + p) / (w3 + p)
    du0 = d1 * lo ** (w3 - 1.0) / (w3 * (w3 - 1.0)) + d2 * lo ** (w3 - 1.0 + p) / (
        (w3 + p) * (w3 - 1.0 + p)
    )
    delta = np.full(n, d1)
    err = np.inf
    last = None

    def refuse(cause, it):
        if last is None:
            raise AccuracyError(
                f"the lowered center's map is not defined at delta_1: {cause}"
            )
        j1 = _response(r, t, omf, w3, *last[1:], c, g, a, b, bc, d_of_u, kind, ra)
        i = int(np.argmax(np.abs(j1)))
        raise AccuracyError(
            "the lowered center's fixed point is not reached from delta_1: at "
            f"iteration {last[0]} the update's response to a uniform change of "
            f"delta is {j1[i]:+.3e} at r = {r[i]:.6e} and its change "
            f"{err:.3e} of delta_1; at iteration {it + 1} {cause}"
        )

    for it in range(200):
        dm = dm0 + _cum_power(t, delta / omf, w3)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            dm_r = dm / r**w3
        if not np.all(np.isfinite(dm_r)):
            # The quotient dm / r^w3 is not finite where r^w3 underflows or dm
            # overflows at the grid's inner nodes, so the Poisson depth's
            # integrand cannot be formed at the grid's inner nodes.
            i = int(np.flatnonzero(~np.isfinite(dm_r))[0])
            refuse(
                f"dm / r^{w3:g} is not finite at r = {r[i]:.6e} (dm = {dm[i]:.3e}, "
                f"r^{w3:g} = {r[i] ** w3:.3e})",
                it,
            )
        du = du0 + _cum_power(t, dm_r, w3 - 1.0)
        uh = u_n - du
        # Six Newton steps in l solve U_N(r e^l) - U_N(r) = -du for the radius
        # r' = r e^l, starting from the linear step log1p(-du r / M_N(r)).
        start = r - du * r**2 / m_n
        with np.errstate(invalid="ignore", divide="ignore"):
            lg = np.log1p(-du * r / m_n)
            for _ in range(6):
                rp = r * np.exp(lg)
                lg = lg - (_series_step(r, lg, c, g, a, b, 2) + du) * rp / _series(
                    rp, c, g, a, b, 3
                )
        cause = None
        off = ~(start > 0.0)
        if off.any():
            # At the nodes in `off` the linear step lies outside (0, r], so the root
            # r' of the depth series at u_hat is found by safeguarded Newton
            # steps inside the bracket (0, r].
            lg = lg.copy()
            try:
                lg[off] = np.log(
                    _map_bracketed(r[off], uh[off], start[off], c, g, a, b) / r[off]
                )
            except AccuracyError as ex:
                cause = str(ex)
        if cause is None:
            step = g_n * np.expm1(_log_aux_step(r, lg, g, a, b, bc, kind, ra))
            new = np.asarray(d_of_u(uh), dtype=np.float64) - step
            if not np.all(np.isfinite(new)):
                i = int(np.flatnonzero(~np.isfinite(new))[0])
                cause = f"the update is not finite at r = {r[i]:.6e}"
        if cause is not None:
            refuse(cause, it)
        last = (it + 1, lg, uh)
        change = np.abs(new - delta)
        err = float(np.max(change) / d1)
        delta = new
        iters = it + 1
        if err < tol:
            break
    # The update forms G(r') - G(r) = g_n expm1(Lambda), whose rounding error
    # is of order eps (|G(r')| + |G(r)|). An iteration that misses tol is
    # therefore at its rounding floor when the last change at every node is
    # at most eps (|G(r')| + |G(r)|) at the node.
    floor_stopped = False
    floor_node_r = floor_size = float("nan")
    if not err < tol:
        bound = np.finfo(np.float64).eps * (np.abs(g_n + step) + np.abs(g_n))
        if bool(np.all(change <= bound)):
            floor_stopped = True
            i = int(np.argmax(change))
            floor_node_r = float(r[i])
            floor_size = float(bound[i] / d1)
    dm = dm0 + _cum_power(t, delta / omf, w3)
    du = du0 + _cum_power(t, dm / r**w3, w3 - 1.0)
    s_d, s_u, s_m = (
        CubicSpline(t, delta),
        CubicSpline(t, np.log(du)),
        CubicSpline(t, np.log(dm)),
    )

    def at(rq):
        rq = np.asarray(rq, dtype=np.float64)
        if rq.size and float(np.max(rq)) > hi:
            raise InputError(
                f"the lowered center is asked at r up to {float(np.max(rq)):.3e}, "
                f"solved up to {hi:.3e}"
            )
        tq = np.log(np.maximum(rq, lo))
        low = rq < lo
        return (
            np.where(low, d1 + d2 * rq**p, s_d(tq)),
            np.where(
                low,
                d1 * rq ** (w3 - 1.0) / (w3 * (w3 - 1.0))
                + d2 * rq ** (w3 - 1.0 + p) / ((w3 + p) * (w3 - 1.0 + p)),
                np.exp(s_u(tq)),
            ),
            np.where(
                low, d1 * rq**w3 / w3 + d2 * rq ** (w3 + p) / (w3 + p), np.exp(s_m(tq))
            ),
        )

    return dict(
        at=at,
        delta_1=d1,
        delta_2=d2,
        p=p,
        iters=iters,
        err=err,
        floor_stopped=floor_stopped,
        floor_node_r=floor_node_r,
        floor_size=floor_size,
        R_lo=lo,
        R_hi=hi,
        bc=float(bc),
        u_hat_of_R=CubicSpline(t, u_n - du),
    )


def representable_end(w3: float, delta_1: float) -> float:
    r"""Return the smallest radius at which the deficit's mass is a normal double.

    At the returned radius :math:`r`, both :math:`r^{w_3}` and the
    deficit's mass :math:`|\delta_1| r^{w_3} / w_3` are at least the
    smallest normal double, with :math:`\delta_1` the first-order deficit
    ``delta_1`` and :math:`w_3 = 3 - 2 \beta_{\rm c}` the argument ``w3``.
    Where :math:`\delta_1` is 0, only :math:`r^{w_3}` is tested.
    `center_amplitude` starts the lowered center's grid at a radius no
    smaller than the returned radius. Inside the grid's first radius,
    `lowered_center` uses the closed form :math:`\delta_1 + \delta_2 r^p`,
    with :math:`\delta_2` the second-order coefficient and
    :math:`p = \gamma - 2 \beta_{\rm c}`.
    """
    tiny = float(np.finfo(np.float64).tiny)
    scale = w3 / abs(float(delta_1)) if float(delta_1) != 0.0 else 1.0
    return float((tiny * max(1.0, scale)) ** (1.0 / float(w3)))


def center_amplitude(
    r: Any,
    phi: Any,
    rho: Any,
    k_t: float,
    p_0: float,
    pot_center: float,
    q: Any,
    df: Any,
    f_at_k_t: float,
    bc: float,
    profile: tuple[float, float, float, float],
    factor: tuple[str | None, float],
    n_per_decade: int = 200,
) -> CenterAmplitude:
    r"""Return the inner amplitude and the central deficit of a truncated model.

    The window is the set of the solve's rows with a finite, positive
    density and a depth :math:`P_0 - \Phi` between INNER_CEILING
    (:math:`10^{-4}`) and 3 INNER_CEILING below the truncated central
    potential :math:`P_0`. An untruncated model (``k_t`` equal to 0) keeps
    the profile's amplitude :math:`C` and carries no deficit. On a core
    (:math:`\gamma \le 0`), or where :math:`2 \beta_{\rm c} > \gamma`, the
    inner amplitude is :math:`C` times the median over the window of the
    ratio of the solve's density to `inner_density_series`, with the median
    clipped to [0, 1], and no deficit is carried. On a cusp
    (:math:`\gamma > 0`) with :math:`2 \beta_{\rm c} \le \gamma`, the inner
    amplitude is :math:`C`, and `lowered_center` solves the deficit at all
    orders.

    Parameters
    ----------
    r : array
        The radii of the truncated Poisson solve's rows, in internal units.
    phi : array
        The potential Phi on the solve's rows, in internal units.
    rho : array
        The density on the solve's rows, in internal units.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal units.
    p_0 : float
        The truncated model's central potential, ``pot_center - k_t``.
    pot_center : float
        The untruncated model's central potential.
    q : array
        The energies of the untruncated DF table, for the deficit.
    df : array
        The values of the untruncated DF table on ``q``.
    f_at_k_t : float
        The value :math:`F(\mathcal{K}_{\rm t})` of the untruncated DF, the
        constant that the truncation subtracts.
    bc : float
        The central anisotropy :math:`\beta_{\rm c}`.
    profile : tuple of float
        The amplitude :math:`C` and the slopes :math:`\alpha`, :math:`\beta`
        and :math:`\gamma` of the source density, in the order
        :math:`(C, \alpha, \beta, \gamma)`.
    factor : tuple
        The Type of the anisotropy ("I", "II" or None for no anisotropy
        radius) and the anisotropy radius :math:`R_{\rm a}`.
    n_per_decade : int, optional
        The number of nodes per decade of the lowered center's grid in r.
        Default: ``200``.

    Returns
    -------
    center : CenterAmplitude
        The inner amplitude, the first-order deficit :math:`\delta_1`, the
        result of `lowered_center` and the degree of the deficit's
        interpolant.

    Raises
    ------
    InputError
        If the DF table holds fewer than 2 rows with a positive energy, a
        finite and positive DF, and :math:`Q \le 4 \mathcal{K}_{\rm t}`, or
        if the lowered center's grid ends inside the window's largest radius.
    AccuracyError
        If the window holds fewer than 5 rows. The message states the
        density's cut (`density_cut`), the radius below which
        :math:`C r^{-\gamma}` exceeds the largest double. Also if no degree
        of `DEFICIT_DEGREES` fits the deficit to :math:`10^{-7}` of the
        largest absolute deficit at the interpolant's nodes, if the
        inner depth series meets :math:`10^{-7}` at no radius
        (`join_depth`), if the fixed point of `lowered_center` leaves the
        map's domain, or if the fixed point's last change is not below
        :math:`10^{-12}` times :math:`\delta_1` and the iteration did not end
        at its rounding floor.

    Notes
    -----
    The deficit is solved on every cusp with
    :math:`2 \beta_{\rm c} \le \gamma`, the case
    :math:`2 \beta_{\rm c} = \gamma` included. At
    :math:`2 \beta_{\rm c} = \gamma`, the missing density
    :math:`\delta\, r^{-2 \beta_{\rm c}}` has the cusp's own power of r, so
    the deficit lowers the inner amplitude instead of fading toward the
    center. The deficit :math:`D(u) = D_0 - D_1 u + \dots` also changes with
    the depth :math:`u`, so the window's median reads :math:`C` less the
    deficit at the window's depth, not at the center. At
    :math:`2 \beta_{\rm c} = \gamma`, the lowered model's depth shift
    :math:`\delta R^{2 - 2 \beta_{\rm c}} /
    [(3 - 2 \beta_{\rm c}) (2 - 2 \beta_{\rm c})]` and the depth :math:`u`
    both scale as :math:`R^{2 - \gamma}`, so the depth shift is of the order
    of the term :math:`D_1 u`. A subtraction of the deficit at fixed radius
    therefore misses terms of the deficit's own order, and the fixed point
    of `lowered_center` includes those terms.

    The deficit's interpolant is a Chebyshev series of :math:`D(u)` on
    :math:`[0, u_{\rm top}]`, with :math:`u_{\rm top}` the larger of 30
    INNER_CEILING and 3 times the join depth of `join_depth` at the bound
    :math:`10^{-7}`. The radius at a depth is read from the leading term of
    `inner_depth_series`. The lowered center's grid starts at the larger of
    two radii: :math:`10^{-3}` times the radius at the depth K_FLOOR, and
    `representable_end`. Inside the grid's first radius, the closed form
    :math:`\delta_1 + \delta_2 r^p` holds. The grid ends at the smaller of
    0.5 and twice the radius at the depth :math:`u_{\rm top} / 2`.
    """
    c, a_p, b_p, g_p = (float(v) for v in profile)
    if float(k_t) == 0.0:
        return CenterAmplitude(c, 0.0, None)
    ceil = INNER_CEILING
    r_np, psi_np, rho_np = (np.array(v) for v in (r, phi, rho))
    lo_psi, hi_psi = p_0 - 3.0 * ceil, p_0 - ceil
    win = (
        (psi_np >= lo_psi)
        & (psi_np <= hi_psi)
        & np.isfinite(rho_np)
        & (rho_np > 0.0)
        & np.isfinite(psi_np)
    )
    if int(np.sum(win)) < 5:
        rows = (psi_np >= lo_psi) & (psi_np <= hi_psi) & np.isfinite(psi_np)
        reach = (
            f"its rows at R {float(np.min(r_np[rows])):.3e} to "
            f"{float(np.max(r_np[rows])):.3e}"
            if rows.any()
            else "no row of the solve"
        )
        raise AccuracyError(
            f"the center's window (depth {ceil:.0e} to {3.0 * ceil:.0e} below P_0) "
            f"holds {int(np.sum(win))} rows with a positive density, fewer than 5: "
            f"{reach}, against the density's cut R = {density_cut(c, g_p):.3e} "
            f"for (alpha, beta, gamma) = ({a_p:g}, {b_p:g}, {g_p:g})"
        )
    r_win, rho_win = r_np[win], rho_np[win]
    rho_exp = inner_density_series(r_win, c, g_p, a_p, b_p)
    ratios = rho_win / np.maximum(rho_exp, 1e-300)
    c_med = c * float(np.clip(np.median(ratios), 0.0, 1.0))
    if not (g_p > 0.0 and 2.0 * float(bc) <= g_p):
        return CenterAmplitude(c_med, 0.0, None)
    d_exact = central_deficit(q, df, k_t, pot_center, bc, f_at_k_t=f_at_k_t)[0]
    b_c = float(bc)
    kappa = (g_p - 2.0 * b_c) * (3.0 - g_p) / ((3.0 - 2.0 * b_c) * (2.0 - 2.0 * b_c))
    delta_1 = d_exact / (1.0 + kappa)
    delta = delta_1
    u_join = join_depth(c, g_p, a_p, b_p, TIER_BOUNDS[0] / 10.0)[0]
    u_top = max(30.0 * ceil, 3.0 * float(u_join))

    reader = _deficit_reader(q, df, k_t, bc, f_at_k_t)

    def deficit_at(v):
        return reader(float(pot_center) - float(v), terms=1)[0]

    for degree in DEFICIT_DEGREES:
        xk = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / float(degree + 1))
        uk = 0.5 * u_top * (1.0 + xk)
        dk = np.array([deficit_at(v) for v in uk])
        cheb = np.polynomial.chebyshev.Chebyshev.fit(
            uk, dk, degree, domain=[0.0, u_top]
        )
        uchk = 0.5 * (uk[:-1] + uk[1:])
        dchk = np.array([deficit_at(v) for v in uchk])
        res = float(
            np.max(np.abs(cheb(uchk) - dchk)) / max(float(np.max(np.abs(dk))), 1e-300)
        )
        if res <= 0.1 * TIER_BOUNDS[0]:
            break
    else:
        raise AccuracyError(
            f"the deficit's interpolant misses it by {res:.2e} of its scale "
            f"over [0, {u_top:.3e}] at degree {degree}, the last of the ladder "
            "(limit 0.0000001)"
        )

    def r_of(u):
        return (float(u) * (3.0 - g_p) * (2.0 - g_p) / c) ** (1.0 / (2.0 - g_p))

    lc = lowered_center(
        max(1e-3 * r_of(K_FLOOR), representable_end(3.0 - 2.0 * b_c, delta_1)),
        min(2.0 * r_of(0.5 * u_top), 0.5),
        c,
        g_p,
        a_p,
        b_p,
        b_c,
        cheb,
        factor[0],
        factor[1],
        n_per_decade=int(n_per_decade),
    )
    if not (lc["err"] < 1e-12 or lc["floor_stopped"]):
        raise AccuracyError(
            f"the lowered center's fixed point did not converge: last change "
            f"{lc['err']:.2e} of delta_1 after {lc['iters']} iterations, above the "
            "rounding bound of its canceling difference"
        )
    lc["P0"] = float(p_0)
    lowered = lc
    if float(np.max(r_win)) > lc["R_hi"]:
        raise InputError(
            f"the lowered center is solved up to r = {lc['R_hi']:.3e}, inside the "
            f"window's last row {float(np.max(r_win)):.3e}"
        )
    return CenterAmplitude(c, delta, lowered, deficit_degree=degree)
