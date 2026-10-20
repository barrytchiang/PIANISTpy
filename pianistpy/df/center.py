r"""The center of a truncated model.

The inner amplitude of the density and the lowered model's central deficit,
from the Poisson solve's rows and the untruncated DF. A phase-space
truncation at :math:`\mathcal{K}_{\rm t}` removes the orbits below it and the
constant ``F(K_t)`` above it; the aux density they carry at the center is the
deficit D(u), smooth in the depth ``u = P_0 - Phi``. On a cusp (gamma > 2 beta)
the inner amplitude is the profile's own and the rows near the center carry
the deficit, solved at all orders as a fixed point on a log grid in r; on a
core, or where ``2 beta >= gamma``, the amplitude is measured from a window
of the solve's rows.

The inner series of the double power law rho = C r^-gamma (1 + r^alpha)^e,
e = (gamma - beta) / alpha, are here too: the density's closed form, the
enclosed mass and the depth, term by term.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
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
from ..validation import GOAL
from .poisson import cuddeford_prefactor, density_cut
from .truncation import INNER_CEILING, K_FLOOR


@dataclass(frozen=True)
class CenterAmplitude:
    """The inner amplitude and the central deficit of a truncated model.

    Parameters
    ----------
    c_trunc : float
        The amplitude C of the inner series the rows near the center use.
    delta : float
        The deficit's limit at r -> 0 at fixed radius (0 where none is
        carried: untruncated, cored, 2 beta >= gamma, or a window too thin).
    lowered : dict or None
        The deficit at all orders (`lowered_center`), with ``P0``; None where
        it was not solved.
    deficit_degree : int
        The degree of the deficit's interpolant (`DEFICIT_DEGREES`, the first
        whose residual is within the limit); 0 where no deficit was fit.
    """

    c_trunc: float
    delta: float
    lowered: dict[str, Any] | None
    deficit_degree: int = 0


# the degrees of the deficit's interpolant, tried in turn until its residual
# at the midpoints between the fit's nodes is within the limit; past the last
# the center is refused
DEFICIT_DEGREES = (8, 12, 16, 24, 32, 48, 64)


def inner_density_series(r, c, gamma, alpha, beta):
    """Return rho = C r^-gamma (1 + r^alpha)^e, the closed form."""
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the enclosed mass diverges")
    r = np.asarray(r, dtype=np.float64)
    e = (gamma - beta) / alpha
    return c * (r ** (-gamma)) * (1.0 + r**alpha) ** e


def inner_mass_series(r, c, gamma, alpha, beta):
    """Return the enclosed mass (no 4 pi) of the density's first four terms."""
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
    """Return the depth ``u = P_0 - Phi(r)`` of the density's first four terms."""
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
    """Return (u_join, r_join, bound): the deepest depth at which the depth
    series' next term, relative to the depth, is within ``tol``.

    Raises
    ------
    AccuracyError
        Where the series reaches no depth within ``tol``.
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
    """Return (u_ceil, r_ceil, bound): the deepest depth at which the mass
    series' next term, relative to the series, is within ``tol``; the
    mass's inner zone reads the series to there.

    Raises
    ------
    AccuracyError
        Where the series reaches no depth within ``tol``.
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
    r"""Return (D0, D1, D2, D3, D4): the aux density the truncation removes at
    depth u below :math:`P_0` as its Taylor series, D = sum_k (-1)^k D_k u^k,
    the D_k from the k-th u-derivative of the same integral; D0 and D1 the
    value and the slope at :math:`P_0`.

    The orbits below :math:`\mathcal{K}_{\rm t}`,
    ``C INT_0^{K_t} F(Q) (2 (P_0 - Q))^n dQ``, n = 1/2 - bc, over the
    untruncated DF (Gauss-Legendre, 400 nodes in ln Q, through a log-log PCHIP
    of the table), plus the lowered constant ``F(K_t)`` on ``K_t < Q < P_0``
    in closed form.
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
    ph = float(p_0) - float(k_t)
    # (-1)^k / k! d^k/du^k at u = 0: of (2 (P_0 - u - Q))^n, 2^k binom(n, k)
    # (2 (P_0 - Q))^(n - k); of (P_0 - K_t - u)^(n + 1), binom(n + 1, k)
    # (P_0 - K_t)^(n + 1 - k)
    out = []
    b_n = b_n1 = 1.0
    for k in range(5):
        if k > 0:
            b_n *= (n - (k - 1)) / k
            b_n1 *= (n + 1.0 - (k - 1)) / k
        d_k = pref * float(
            np.sum(wq * 2.0**k * b_n * (2.0 * (float(p_0) - qn)) ** (n - k))
        )
        x_k = pref * fz * 2.0**n * b_n1 * ph ** (n + 1.0 - k) / (n + 1.0)
        out.append(d_k + x_k)
    return tuple(out)


def anisotropy_g(r, bc, kind, ra):
    r"""Return g(R) of the augmented density, `Anisotropy.factor` (the
    package's one g), for the anisotropy given as :math:`\beta_{\rm c}`,
    ``kind`` ("I", "II", or None for no anisotropy radius) and
    :math:`R_{\rm a}`."""
    aniso = Anisotropy(
        beta_c=float(bc),
        r_a=float(ra) if kind is not None else math.inf,
        kind=kind if kind is not None else "I",
    )
    return aniso.factor(np.asarray(r, dtype=np.float64))


def _series(r, c, g, a, b, shift, terms=12):
    """The untruncated mass (shift 3) or depth (shift 2), term by term."""
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
    """_series(r e^lg) - _series(r), term by term through expm1, so the two
    nearly equal sums are never formed."""
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
    """ln G(r e^lg) - ln G(r) of the untruncated aux density
    G(x) = x^(2 bc) A(x) rho_N(x), through log1p and expm1."""
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
    """INT h r^(w - 1) dr along t = ln r from t[0]: the power exact, h a cubic
    spline, 5-point Gauss-Legendre per panel."""
    cs = CubicSpline(t, h)
    x, wt = np.polynomial.legendre.leggauss(5)
    a, b = t[:-1], t[1:]
    tm = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (b + a)[:, None]
    pan = np.sum(0.5 * (b - a)[:, None] * wt[None, :] * cs(tm) * np.exp(w * tm), axis=1)
    return np.concatenate([[0.0], np.cumsum(pan)])


def _map_bracketed(r, uh, start, c, g, a, b, iters=200):
    """Return r' in (0, r] with the untruncated depth series equal to u_hat,
    by Newton steps kept inside the bracket (bisection where a step leaves
    it), for the nodes whose linear start r - du r^2 / M left (0, r].

    Raises
    ------
    AccuracyError
        Where the bracket holds no sign change of the residual (``u_hat <= 0``
        or the depth at r below u_hat): no r' in (0, r].
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
    """The update's response to a uniform change of delta, J 1 = s K 1: K 1
    the change of du (through dm), s = d new / d du = -D'(u_hat)
    + G(r') Lambda'(l) r' / M(r'), Lambda' = d ln G(r e^l) / dl."""
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
    r"""Return the lowered model's central deficit at all orders, any
    (gamma, alpha, beta, :math:`\beta_{\rm c}`) with
    :math:`\gamma > 2 \beta_{\rm c}`.

    At fixed depth the truncated aux density is g(u) - D(u), g the
    untruncated aux density against depth; at fixed radius the lowered model
    is shallower by du, the Poisson depth of the missing density
    delta(r) r^(-2 bc) / factor:
    delta(r) = D(u_hat) - [g(u_hat) - g(u_N)], du'' + 2 du' / r = that density,
    a contraction iterated on a log grid in r from ``r_lo`` to ``r_hi``.
    Its first order is delta_1 = D(0) / (1 + kappa), its second the closed
    form delta_2 r^p, p = gamma - 2 bc, used below the grid.

    The map is solved in the step from r, exactly: with G(x) = x^(2 bc)
    A(x) rho_N(x) (A the anisotropy factor) and r' = r e^l the radius where
    the untruncated depth is u_hat, U_N(r') - U_N(r) = -du(r), where

        U_N(r') - U_N(r) = c SUM_k b_k r^(e_k) expm1(e_k l) / (e_k (e_k + 1)),
        e_k = 2 - gamma + k alpha, b_k the binomial coefficients of the
        series in r^alpha,

    solved for l by Newton (dU_N/dl = M_N(r') / r'), and

        G(r') - G(r) = G(r) expm1(Lambda),
        Lambda = (2 bc - gamma) l
            + ((gamma - beta) / alpha) log1p(r^alpha expm1(alpha l) / (1 + r^alpha))
            + ``(1 - bc) log1p(+-(r / R_a)^2 expm1(2 l) / (1 +- (r / R_a)^2))``,

    the last term for ``kind`` "I" (+) and "II" (-) only. No difference of
    two nearly equal terms is formed.

    Returns a dict with ``at(r) -> (delta, du, dm)`` (dm = INT delta r^(2 - 2 bc)
    dr, no 4 pi), ``u_hat_of_R``, ``delta_1``, ``delta_2``, ``p``, ``iters``,
    ``err``, ``floor_stopped``, ``floor_node_r``, ``floor_size``,
    ``R_lo``, ``R_hi`` and ``bc``. Where the change does not fall below
    ``tol`` in 200 iterations, the iteration is accepted when, at every
    node, its last change is within eps times |G(r')| + |G(r)| there:
    ``floor_stopped`` says it was, ``floor_node_r`` is the
    node of the largest last change and ``floor_size`` that node's bound, in
    units of delta_1.

    The map r -> r' starts from the linear step r - du r^2 / M; where that
    start leaves (0, r] it is solved in the bracket instead
    (`_map_bracketed`). Where (0, r] holds no root, an iterate is not
    finite, or the Poisson depth's integrand dm / r^(3 - 2 bc) is not (r^(3 -
    2 bc) underflowing at the grid's inner nodes), the iteration has left the
    map's domain: an AccuracyError names
    the update's response to a uniform change of delta (`_response`) and the
    change at the last iterate where the map was defined. This is a stated
    limit, not a failure of the step: at :math:`\beta_{\rm c} = -1/2` on
    (alpha, beta) = (2, 4) at :math:`\mathcal{K}_{\rm t} = 0.3`, from gamma
    1.735 on, no iteration from delta_1 (Picard, or Newton on the residual,
    full or backtracked) reaches a fixed point, and from gamma 1.78 the map has
    no root at delta_1 itself.
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
            # r^w3 underflows (or dm overflows) at the grid's inner nodes: the
            # Poisson depth's integrand is not formed there
            i = int(np.flatnonzero(~np.isfinite(dm_r))[0])
            refuse(
                f"dm / r^{w3:g} is not finite at r = {r[i]:.6e} (dm = {dm[i]:.3e}, "
                f"r^{w3:g} = {r[i] ** w3:.3e})",
                it,
            )
        du = du0 + _cum_power(t, dm_r, w3 - 1.0)
        uh = u_n - du
        # r' = r e^l, Newton on U_N(r') - U_N(r) = -du in l
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
            # the linear start left (0, r]: the root r' of the depth series at
            # u_hat, in (0, r] where du > 0, by Newton kept in its bracket
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
    # where the test is not met, the iteration is at its rounding floor when
    # at every node its last change is within eps times |G(r')| + |G(r)|
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
    """Return the smallest radius at which the lowered center's Poisson
    depth is formed in normal float64 numbers: r^w3 and the mass of the
    deficit there, dm ~ |delta_1| r^w3 / w3, both at least the smallest
    normal double. Below it the closed form delta_1 + delta_2 r^p holds.
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
    r"""Return the inner amplitude and the central deficit of a truncated
    model from the Poisson solve's rows (r, Phi, rho).

    Parameters
    ----------
    k_t, p_0, pot_center : float
        The truncation energy, the truncated model's central potential
        ``P_0 = pot_center - k_t`` and the untruncated one.
    q, df : array
        The untruncated DF's table, for the deficit.
    f_at_k_t : float
        ``F(K_t)``, the lowered constant.
    profile : tuple
        (C, alpha, beta, gamma) of the source density.
    factor : tuple
        The anisotropy factor's kind ("I", "II" or None) and :math:`R_{\rm a}`.
    n_per_decade : int
        Rows per decade of the lowered center's grid in r.

    Notes
    -----
    The deficit is solved on every cusp with ``2 beta_c <= gamma``, the
    marginal case ``2 beta_c = gamma`` included. There the deficit
    ``D r^(-2 beta_c)`` has the cusp's own power, so it lowers the amplitude
    instead of fading inward, and D = D0 - D1 u still changes with the depth:
    the window's median reads C - D at the window's depth, not at the center.
    The lowered model's depth shift,
    ``D R^(2 - 2 beta_c) / ((3 - 2 beta_c) (2 - 2 beta_c))``, is then of
    order R, the order of D1 u, so a subtraction at fixed radius is not the
    deficit there; the fixed point (`lowered_center`) is.

    The lowered center's grid starts at the larger of its depth-set end
    (:math:`10^{-3}` of the radius at depth K_FLOOR) and `representable_end`,
    the smallest radius at which ``r^(3 - 2 beta_c)`` and the deficit's mass
    are normal doubles; below it the closed form holds.

    Raises
    ------
    InputError
        Where the DF table holds no rows to read the deficit from, or the
        lowered center's grid does not reach the window.
    AccuracyError
        Where the window holds fewer than 5 rows with a positive density (the
        rows below the density's cut, `density_cut`), the deficit's
        interpolant misses it by more than a tenth of the goal at the last
        degree of `DEFICIT_DEGREES`, the series
        reaches no depth within it, or the lowered center's fixed point does
        not converge.
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
    u_join = join_depth(c, g_p, a_p, b_p, GOAL / 10.0)[0]
    u_top = max(30.0 * ceil, 3.0 * float(u_join))

    def deficit_at(v):
        return central_deficit(q, df, k_t, float(pot_center) - float(v), bc, f_at_k_t)[
            0
        ]

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
        if res <= 0.1 * GOAL:
            break
    else:
        raise AccuracyError(
            f"the deficit's interpolant misses it by {res:.2e} of its scale "
            f"over [0, {u_top:.3e}] at degree {degree}, the last of the ladder "
            "(limit a tenth of the goal)"
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
