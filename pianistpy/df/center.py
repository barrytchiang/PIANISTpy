"""The center of a truncated model.

The inner amplitude of the density and the lowered model's central deficit,
from the Poisson solve's rows and the untruncated DF. A phase-space
truncation at K_t removes the orbits below K_t and the constant F(K_t) above
it; the aux density they carry at the center is the deficit D(u), smooth in
the depth u = P_0 - Phi. On a cusp (gamma > 2 beta) the inner amplitude is
the profile's own and the rows near the center carry the deficit, solved at
all orders as a fixed point on a log grid in r; on a core, or where
2 beta >= gamma, the amplitude is measured from a window of the solve's rows.

The inner series of the double power law rho = C r^-gamma (1 + r^alpha)^e,
e = (gamma - beta) / alpha, are here too: the density's closed form, the
enclosed mass and the depth, term by term.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
"""

__all__ = [
    "CenterAmplitude",
    "anisotropy_factor",
    "central_deficit",
    "center_amplitude",
    "ceiling_depth",
    "inner_density_series",
    "inner_depth_series",
    "inner_mass_series",
    "join_depth",
    "lowered_center",
]

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..exceptions import AccuracyError, InputError
from ..validation import GOAL
from .poisson import cuddeford_prefactor
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
    """

    c_trunc: float
    delta: float
    lowered: dict[str, Any] | None


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
    """Return the depth u = P_0 - Phi(r) of the density's first four terms."""
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
    """Return (D0, D1): the aux density the truncation removes at depth
    P_0 and its slope, D = D0 - D1 u near there.

    The orbits below K_t, C INT_0^{K_t} F(Q) (2 (P_0 - Q))^n dQ, n = 1/2 - bc,
    over the untruncated DF (Gauss-Legendre, 400 nodes in ln Q, through a
    log-log PCHIP of the table), plus the lowered constant F(K_t) on
    K_t < Q < P_0 in closed form.
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
    d0 = pref * float(np.sum(wq * (2.0 * (float(p_0) - qn)) ** n))
    d1 = pref * float(np.sum(wq * 2.0 * n * (2.0 * (float(p_0) - qn)) ** (n - 1.0)))
    fz = float(np.exp(log_f(lb))) if f_at_k_t is None else float(f_at_k_t)
    ph = float(p_0) - float(k_t)
    x0 = pref * fz * 2.0**n * ph ** (n + 1.0) / (n + 1.0)
    x1 = pref * fz * 2.0**n * ph**n
    return d0 + x0, d1 + x1


def anisotropy_factor(r, bc, kind, ra):
    """Return the aux density's factor beyond r^(2 bc):
    (1 + r^2 / R_a^2)^(1 - bc) for ``kind`` "I", (1 - r^2 / R_a^2)^(1 - bc)
    for "II", 1 for None."""
    r = np.asarray(r, dtype=np.float64)
    if kind == "I":
        return (1.0 + (r / float(ra)) ** 2) ** (1.0 - float(bc))
    if kind == "II":
        return np.maximum(1.0 - (r / float(ra)) ** 2, 1e-300) ** (1.0 - float(bc))
    return np.ones_like(r)


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


def _cum_power(t, h, w):
    """INT h r^(w - 1) dr along t = ln r from t[0]: the power exact, h a cubic
    spline, 5-point Gauss-Legendre per panel."""
    cs = CubicSpline(t, h)
    x, wt = np.polynomial.legendre.leggauss(5)
    a, b = t[:-1], t[1:]
    tm = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (b + a)[:, None]
    pan = np.sum(0.5 * (b - a)[:, None] * wt[None, :] * cs(tm) * np.exp(w * tm), axis=1)
    return np.concatenate([[0.0], np.cumsum(pan)])


def lowered_center(
    r_lo, r_hi, c, g, a, b, bc, d_of_u, kind, ra, n_per_decade=200, tol=1e-15
):
    """Return the lowered model's central deficit at all orders, any
    (gamma, alpha, beta, beta_c) with gamma > 2 beta_c.

    At fixed depth the truncated aux density is g(u) - D(u), g the
    untruncated aux density against depth; at fixed radius the lowered model
    is shallower by du, the Poisson depth of the missing density
    delta(r) r^(-2 bc) / factor:
    delta(r) = D(u_hat) - [g(u_hat) - g(u_N)], du'' + 2 du' / r = that density,
    a contraction iterated on a log grid in r from ``r_lo`` to ``r_hi``.
    Its first order is delta_1 = D(0) / (1 + kappa), its second the closed
    form delta_2 r^p, p = gamma - 2 bc, used below the grid.

    Returns a dict with ``at(r) -> (delta, du, dm)`` (dm = INT delta r^(2 - 2 bc)
    dr, no 4 pi), ``u_hat_of_R``, ``delta_1``, ``delta_2``, ``p``, ``iters``,
    ``err``, ``floor_stopped``, ``floor_node_r``, ``floor_size``,
    ``R_lo``, ``R_hi`` and ``bc``. Where the change does not fall below
    ``tol`` in 200 iterations, the iteration is accepted when, at every
    node, its last change is within the rounding bound of the cancelling
    difference rho_aux(r') - rho_aux(r) there, eps times the sum of the two
    terms' magnitudes: ``floor_stopped`` says it was, ``floor_node_r`` is the
    node of the largest last change and ``floor_size`` that node's bound, in
    units of delta_1.
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
    omf = anisotropy_factor(r, bc, kind, ra)
    u_n, m_n = _series(r, c, g, a, b, 2), _series(r, c, g, a, b, 3)
    g_n = r ** (2.0 * bc) * omf * rho_n(r)
    w3 = 3.0 - 2.0 * bc
    dm0 = d1 * lo**w3 / w3 + d2 * lo ** (w3 + p) / (w3 + p)
    du0 = d1 * lo ** (w3 - 1.0) / (w3 * (w3 - 1.0)) + d2 * lo ** (w3 - 1.0 + p) / (
        (w3 + p) * (w3 - 1.0 + p)
    )
    delta = np.full(n, d1)
    err = np.inf
    for it in range(200):
        dm = dm0 + _cum_power(t, delta / omf, w3)
        du = du0 + _cum_power(t, dm / r**w3, w3 - 1.0)
        uh = u_n - du
        rp = r - du * r**2 / m_n
        for _ in range(6):
            rp = rp - (_series(rp, c, g, a, b, 2) - uh) * rp**2 / _series(
                rp, c, g, a, b, 3
            )
        term = rp ** (2.0 * bc) * anisotropy_factor(rp, bc, kind, ra) * rho_n(rp)
        new = np.asarray(d_of_u(uh), dtype=np.float64) - (term - g_n)
        change = np.abs(new - delta)
        err = float(np.max(change) / d1)
        delta = new
        iters = it + 1
        if err < tol:
            break
    # where the test is not met, the iteration has reached the rounding of
    # the cancelling difference term - g_n when at every node its last change
    # is within that difference's bound, eps times the terms' magnitudes
    floor_stopped = False
    floor_node_r = floor_size = float("nan")
    if not err < tol:
        bound = np.finfo(np.float64).eps * (np.abs(term) + np.abs(g_n))
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
    """Return the inner amplitude and the central deficit of a truncated
    model from the Poisson solve's rows (r, Phi, rho).

    Parameters
    ----------
    k_t, p_0, pot_center : float
        The truncation energy, the truncated model's central potential
        P_0 = pot_center - k_t and the untruncated one.
    q, df : array
        The untruncated DF's table, for the deficit.
    f_at_k_t : float
        F(K_t), the lowered constant.
    profile : tuple
        (C, alpha, beta, gamma) of the source density.
    factor : tuple
        The anisotropy factor's kind ("I", "II" or None) and R_a.
    n_per_decade : int
        Rows per decade of the lowered center's grid in r.

    Raises
    ------
    InputError
        Where the DF table holds no rows to read the deficit from, or the
        lowered center's grid does not reach the window.
    AccuracyError
        Where the deficit's interpolant misses it by more than a tenth of the
        goal, the series reaches no depth within it, or the lowered center's
        fixed point does not converge.
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
        return CenterAmplitude(c, 0.0, None)
    r_win, rho_win = r_np[win], rho_np[win]
    rho_exp = inner_density_series(r_win, c, g_p, a_p, b_p)
    ratios = rho_win / np.maximum(rho_exp, 1e-300)
    c_med = c * float(np.clip(np.median(ratios), 0.0, 1.0))
    if not (g_p > 0.0 and 2.0 * float(bc) < g_p):
        return CenterAmplitude(c_med, 0.0, None)
    d_exact, _ = central_deficit(q, df, k_t, pot_center, bc, f_at_k_t=f_at_k_t)
    b_c = float(bc)
    kappa = (g_p - 2.0 * b_c) * (3.0 - g_p) / ((3.0 - 2.0 * b_c) * (2.0 - 2.0 * b_c))
    delta_1 = d_exact / (1.0 + kappa)
    delta = delta_1
    u_join = join_depth(c, g_p, a_p, b_p, GOAL / 10.0)[0]
    u_top = max(30.0 * ceil, 3.0 * float(u_join))
    xk = np.cos(np.pi * (np.arange(9) + 0.5) / 9.0)
    uk = 0.5 * u_top * (1.0 + xk)
    dk = np.array(
        [
            central_deficit(q, df, k_t, float(pot_center) - float(v), bc, f_at_k_t)[0]
            for v in uk
        ]
    )
    cheb = np.polynomial.chebyshev.Chebyshev.fit(uk, dk, 8, domain=[0.0, u_top])
    uchk = np.array([0.1, 0.5, 0.9]) * u_top
    dchk = np.array(
        [
            central_deficit(q, df, k_t, float(pot_center) - float(v), bc, f_at_k_t)[0]
            for v in uchk
        ]
    )
    res = float(
        np.max(np.abs(cheb(uchk) - dchk)) / max(float(np.max(np.abs(dk))), 1e-300)
    )
    if not res <= 0.1 * GOAL:
        raise AccuracyError(
            f"the deficit's interpolant misses it by {res:.2e} of its scale "
            f"over [0, {u_top:.3e}] (limit a tenth of the goal)"
        )

    def r_of(u):
        return (float(u) * (3.0 - g_p) * (2.0 - g_p) / c) ** (1.0 / (2.0 - g_p))

    lc = lowered_center(
        1e-3 * r_of(K_FLOOR),
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
            "rounding bound of its cancelling difference"
        )
    lc["P0"] = float(p_0)
    lowered = lc
    if float(np.max(r_win)) > lc["R_hi"]:
        raise InputError(
            f"the lowered center is solved up to r = {lc['R_hi']:.3e}, inside the "
            f"window's last row {float(np.max(r_win)):.3e}"
        )
    return CenterAmplitude(c, delta, lowered)
