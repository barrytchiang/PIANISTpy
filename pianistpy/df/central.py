"""The DF's center in closed form, from the model alone.

Near the central potential :math:`P_0` the auxiliary density is a series in
``u = P_0 - P``; its least non-integer power s gives the inversion the DF's
singular term at the center, and the next one the term after it. The branch
(regular, critical, unphysical), the form (``"power"``, ``"log"``,
``"constant_plus_power"``, ``"analytic"``, ``"none"``), the exponent and the
closed-form amplitude follow; no number of the table is read.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.

References
----------
Section T:2.2.2 [regular], Section T:2.2.3 [critical].
"""

__all__ = [
    "TOL_EXACT_BRANCH",
    "central_form",
    "center_form",
    "rho_aux_series",
    "top_limit",
]

from math import factorial
from typing import Any

import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import gammaln, gammasgn

from ..anisotropy import Anisotropy
from ..profiles import DoublePowerLaw

# the tolerance of the branch ladder: gamma against 2 beta_c, integer powers
TOL_EXACT_BRANCH = 1e-10


def _gamma_ratio(a, b):
    """Gamma(a) / Gamma(b), from the log of each Gamma with its sign: never the
    quotient of two Gammas, which is inf / inf past an argument of 171.6
    although the ratio itself is an ordinary number."""
    sign = gammasgn(a) * gammasgn(b)
    return float(sign * np.exp(gammaln(a) - gammaln(b)))


_MODEL_TYPES = {
    "isotropic": "constant",
    "constant": "constant",
    "OM_I": "OM_I",
    "OM_II": "OM_II",
    "COM_I": "COM_I",
    "COM_II": "COM_II",
}


def rho_aux_series(alpha, beta_r, gamma, bc, m_typ, Ra, C, x_max=None, tol=None):
    """Return rho_aux near :math:`P_0` as its series in ``u = P_0 - P``,
    [(s, C_s), ...] sorted by s, equal powers merged.

    With rho = C R^-gamma (1 + R^alpha)^e, e = (gamma - beta)/alpha, the depth
    u = c_u R^(2-gamma) (1 + D(R)) is inverted for R = t (1 + S(t)),
    t = (u / c_u)^(1/(2-gamma)), and rho_aux = C R^q (1 + R^alpha)^e A(R),
    q = 2 bc - gamma (constant, COM) or -gamma (isotropic, OM), A the
    anisotropy's factor. Every term up to t^(q + x_max) is exact (by default
    x_max is three times the larger of alpha and 2).
    """
    a, b, g = float(alpha), float(beta_r), float(gamma)
    tol = TOL_EXACT_BRANCH if tol is None else float(tol)
    bce = float(bc) if m_typ in ("constant", "COM_I", "COM_II") else 0.0
    e = (g - b) / a
    cu = float(C) / ((2.0 - g) * (3.0 - g))
    xm = 3.0 * max(a, 2.0) if x_max is None else float(x_max)
    eps_x = 1e-9

    def _key(x):
        return round(x, 9)

    def mul(P, Qs):
        out: dict[float, float] = {}
        for x1, c1 in P.items():
            for x2, c2 in Qs.items():
                x = x1 + x2
                if x <= xm + eps_x:
                    out[_key(x)] = out.get(_key(x), 0.0) + c1 * c2
        return out

    def pow1p(S, p):
        # (1 + S)^p for S a series of positive powers, the binomial series
        out, term, coef = {0.0: 1.0}, {0.0: 1.0}, 1.0
        xmin = min((x for x, c in S.items() if c != 0.0), default=None)
        if xmin is None:
            return out
        for k in range(1, int(np.floor(xm / xmin + eps_x)) + 1):
            coef *= (p - (k - 1)) / k
            term = mul(term, S)
            for x, c in term.items():
                out[x] = out.get(x, 0.0) + coef * c
        return out

    K = int(np.floor(xm / a + eps_x)) + 1
    bk = [1.0]
    for k in range(1, K + 1):
        bk.append(bk[-1] * (e - (k - 1)) / k)
    dk = {
        _key(k * a): bk[k]
        * (2.0 - g)
        * (3.0 - g)
        / ((2.0 - g + k * a) * (3.0 - g + k * a))
        for k in range(1, K + 1)
        if k * a <= xm + eps_x
    }
    # R = t (1 + S): S = (1 + D(t (1 + S)))^(-1/(2-gamma)) - 1, one order of
    # t^alpha a pass
    S: dict[float, float] = {}
    for _ in range(K + 1):
        Dt: dict[float, float] = {}
        for x, d in dk.items():
            for xx, cc in mul({x: d}, pow1p(S, x)).items():
                Dt[xx] = Dt.get(xx, 0.0) + cc
        S = {x: c for x, c in pow1p(Dt, -1.0 / (2.0 - g)).items() if x > 0.0}
    q = (2.0 * bce - g) if m_typ in ("constant", "COM_I", "COM_II") else -g
    T = pow1p(S, q)
    Ra_t = {_key(a): 1.0}
    T = mul(T, pow1p(mul(Ra_t, pow1p(S, a)), e))
    if m_typ in ("OM_I", "OM_II", "COM_I", "COM_II"):
        sgn = 1.0 if m_typ in ("OM_I", "COM_I") else -1.0
        mexp = 1.0 if m_typ in ("OM_I", "OM_II") else (1.0 - bce)
        R2 = mul({2.0: sgn / float(Ra) ** 2}, pow1p(S, 2.0))
        T = mul(T, pow1p(R2, mexp))
    out: dict[float, float] = {}
    for x, c in T.items():
        s = (q + x) / (2.0 - g)
        s_key = next((kk for kk in out if abs(kk - s) <= tol), s)
        out[s_key] = out.get(s_key, 0.0) + float(C) * cu ** (-s) * c
    return sorted((s, c) for s, c in out.items() if c != 0.0)


def central_form(alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=None):
    """Return the center's form and its next singular term, in closed form.

    The keys: branch, form, s, kappa, kappa_sing, amplitude, log_coef, sign,
    local, coef_rho, line, and next_form, next_s, next_kappa, next_coef,
    next_coef_rho, next_line.
    """
    res = _central_form_leading(alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=tol)
    res.update(_central_next_term(res, alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=tol))
    return res


def center_form(profile: DoublePowerLaw, anisotropy: Anisotropy) -> dict[str, Any]:
    r"""Return `central_form` of a component: the profile's own amplitude, the
    anisotropy's model type, OM entering with :math:`\beta_{\rm c} = 0`."""
    m_typ = _MODEL_TYPES[anisotropy.family]
    bc = 0.0 if m_typ in ("OM_I", "OM_II") else float(anisotropy.beta_c)
    return central_form(
        float(profile.rho_alpha),
        float(profile.rho_beta),
        float(profile.rho_gamma),
        bc,
        m_typ,
        float(anisotropy.r_a),
        float(profile.rho_norm),
    )


def top_limit(
    profile: DoublePowerLaw,
    anisotropy: Anisotropy,
    orders: tuple[int, ...] | None = None,
) -> float:
    """Return the least depth ``u = P_0 - P`` at which the exact derivatives of
    the given orders of rho_aux are finite in float64, from the model alone.

    Each term C_s u^s of `rho_aux_series` gives d^N rho_aux/dP^N =
    (-1)^N C_s s (s-1) ... (s-N+1) u^(s-N), which exceeds the largest float64
    below u = (|C_s s ... (s-N+1)| / DBL_MAX)^(1/(N-s)) where s < N; the limit
    is the largest over the terms and the orders, 0 where none grows. By
    default the orders are the inversion's, j - 1 to j + 2
    (`aux_derivatives`).
    """
    if orders is None:
        from .inversion import order_pair

        beta = 0.0 if anisotropy.model == "om" else float(anisotropy.beta_c)
        j = order_pair(beta).order_j
        orders = tuple(range(max(j - 1, 0), j + 3))
    m_typ = _MODEL_TYPES[anisotropy.family]
    bc = 0.0 if m_typ in ("OM_I", "OM_II") else float(anisotropy.beta_c)
    series = rho_aux_series(
        float(profile.rho_alpha),
        float(profile.rho_beta),
        float(profile.rho_gamma),
        bc,
        m_typ,
        float(anisotropy.r_a),
        float(profile.rho_norm),
    )
    log_max = np.log(np.finfo(np.float64).max)
    u_min = 0.0
    for s, c in series:
        for n in orders:
            falling = float(np.prod([s - i for i in range(n)])) if n > 0 else 1.0
            if not (s < n and falling != 0.0 and c != 0.0):
                continue
            log_u = (np.log(abs(c * falling)) - log_max) / (n - s)
            u_min = max(u_min, float(np.exp(log_u)))
    return u_min


def _central_next_term(res, alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=None):
    tol = TOL_EXACT_BRANCH if tol is None else float(tol)
    nan = float("nan")
    none = dict(
        next_form="none", next_s=nan, next_kappa=nan, next_coef=nan, next_coef_rho=nan
    )
    if res["form"] in ("none", "analytic"):
        return dict(
            none,
            next_line=f"no next term: the form {res['form']!r} has no singular term",
        )
    bce = float(bc) if m_typ in ("constant", "COM_I", "COM_II") else 0.0
    nu = 1.5 - bce
    Np = 2.0**bce / ((2.0 * np.pi) ** 1.5 * gamma_function(1.0 - bce))
    g = float(gamma)
    xm = 3.0 * max(float(alpha), 2.0)
    ser = rho_aux_series(alpha, beta_r, gamma, bc, m_typ, Ra, C, x_max=xm, tol=tol)
    q = (2.0 * bce - g) if m_typ in ("constant", "COM_I", "COM_II") else -g
    s_hi = (q + xm) / (2.0 - g)
    s1 = float(res["s"])
    lead = [c for s, c in ser if abs(s - s1) <= tol]
    if not lead or not abs(lead[0] / float(res["coef_rho"]) - 1.0) <= 1e-9:
        return dict(
            none,
            next_line=(
                f"no next term: the series' coefficient at s = {s1:.6f} "
                f"({lead[0] if lead else 'absent'}) does not reproduce the "
                f"leading form's {float(res['coef_rho']):.10e}"
            ),
        )
    # a non-negative integer power is analytic; a negative integer one singular
    cand = [
        (s, c)
        for s, c in ser
        if s > s1 + tol
        and not (abs(s - round(s)) <= tol and round(s) >= 0)
        and s < s_hi - tol
    ]
    if not cand:
        return dict(
            none,
            next_line=(
                f"no next term: no non-integer power of rho_aux above "
                f"s = {s1:.6f} up to {s_hi:.4f}"
            ),
        )
    s2, c2 = cand[0]
    k2 = s2 - nu
    if abs(k2 - round(k2)) <= tol and round(k2) >= 0:
        kk = int(round(k2))
        L2 = float(Np * c2 * (-1.0) ** kk / (factorial(kk) * gamma_function(-s2)))
        return dict(
            next_form="log_power",
            next_s=s2,
            next_kappa=float(kk),
            next_coef=L2,
            next_coef_rho=c2,
            next_line=f"next: s2 = {s2:.6f} = nu + {kk}, L2 = {L2:.10e}",
        )
    A2 = float(Np * c2 * _gamma_ratio(nu - s2, -s2))
    return dict(
        next_form="power",
        next_s=s2,
        next_kappa=k2,
        next_coef=A2,
        next_coef_rho=c2,
        next_line=f"next: s2 = {s2:.6f}, A2 = {A2:.10e}",
    )


def _central_form_leading(alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=None):
    """The center's leading form: the least non-integer power s of rho_aux
    gives F ~ N' C_s Gamma(nu - s)/Gamma(-s) u^(s - nu), or N' C_s/Gamma(-nu)
    (-ln u) at s = nu, N' = 2^bc / ((2 pi)^(3/2) Gamma(1 - bc)),
    nu = 3/2 - bc; off the critical branch (gamma > 2 bc) rho_aux ~
    C (u/c_u)^s0, s0 = (2 bc - gamma)/(2 - gamma)."""
    tol = TOL_EXACT_BRANCH if tol is None else float(tol)
    a, br_, g = float(alpha), float(beta_r), float(gamma)
    bce = float(bc) if m_typ in ("constant", "COM_I", "COM_II") else 0.0
    nu = 1.5 - bce
    n = nu - np.floor(nu + tol)
    local = abs(n) <= tol
    Np = 2.0**bce / ((2.0 * np.pi) ** 1.5 * gamma_function(1.0 - bce))
    cu = float(C) / ((2.0 - g) * (3.0 - g))
    nan = float("nan")

    def _isint(x):
        return abs(x - round(x)) <= tol

    def _amp(Cs, s):
        return float(Np * Cs * _gamma_ratio(nu - s, -s))

    if g > 2.0 * bce + tol and g < 2.0:
        s0 = (2.0 * bce - g) / (2.0 - g)
        Cs = float(C) * cu ** (-s0)
        A = _amp(Cs, s0)
        kap = s0 - nu
        return dict(
            branch="regular",
            form="power",
            s=s0,
            kappa=kap,
            kappa_sing=kap,
            amplitude=A,
            log_coef=nan,
            sign=float(np.sign(A)),
            local=local,
            coef_rho=Cs,
            line=f"regular: s = {s0:.6f}, kappa = {kap:.6f}, A = {A:.10e}",
        )
    if not (abs(g - 2.0 * bce) <= tol and g < 2.0):
        return dict(
            branch="unphysical_center",
            form="none",
            s=nan,
            kappa=nan,
            kappa_sing=nan,
            amplitude=nan,
            log_coef=nan,
            sign=0.0,
            local=local,
            coef_rho=nan,
            line=f"gamma = {g:.4f} < 2 bc = {2.0 * bce:.4f}: no consistent DF",
        )
    delta = a / (2.0 - g)
    terms: dict[float, float] = {}
    if not _isint(delta):
        terms[delta] = float(C) * (g - br_) / a * cu ** (-delta)
    if m_typ in ("COM_I", "COM_II"):
        sc = 2.0 / (2.0 - g)
        if not _isint(sc):
            cc = (
                float(C)
                * (1.0 - bce)
                / float(Ra) ** 2
                * cu ** (-sc)
                * (1.0 if m_typ == "COM_I" else -1.0)
            )
            key = next((k for k in terms if abs(k - sc) <= tol), sc)
            terms[key] = terms.get(key, 0.0) + cc
    terms = {k: v for k, v in terms.items() if v != 0.0}
    if not terms:
        return dict(
            branch="critical_analytic",
            form="analytic",
            s=nan,
            kappa=0.0,
            kappa_sing=nan,
            amplitude=nan,
            log_coef=nan,
            sign=0.0,
            local=local,
            coef_rho=nan,
            line="gamma = 2 bc: rho_aux is analytic in u at P0; F is finite",
        )
    s = min(terms)
    Cs = terms[s]
    kap = s - nu
    if abs(kap) <= tol:
        L = float(Np * Cs / gamma_function(-nu))
        return dict(
            branch="critical_log",
            form="log",
            s=s,
            kappa=0.0,
            kappa_sing=0.0,
            amplitude=nan,
            log_coef=L,
            sign=float(np.sign(L)),
            local=local,
            coef_rho=Cs,
            line=f"critical, s = nu = {s:.6f}: F ~ L (-ln u), L = {L:.10e}",
        )
    A = _amp(Cs, s)
    if kap < 0.0:
        return dict(
            branch="critical_divergent",
            form="power",
            s=s,
            kappa=kap,
            kappa_sing=kap,
            amplitude=A,
            log_coef=nan,
            sign=float(np.sign(A)),
            local=local,
            coef_rho=Cs,
            line=f"critical, s = {s:.6f} < nu: kappa = {kap:.6f}, A = {A:.10e}",
        )
    if local:
        return dict(
            branch="critical_vanishing",
            form="power",
            s=s,
            kappa=kap,
            kappa_sing=kap,
            amplitude=A,
            log_coef=nan,
            sign=float(np.sign(A)),
            local=local,
            coef_rho=Cs,
            line=f"critical, s = {s:.6f} > nu, n = 0: kappa = {kap:.6f}, A = {A:.10e}",
        )
    return dict(
        branch="critical_finite",
        form="constant_plus_power",
        s=s,
        kappa=0.0,
        kappa_sing=kap,
        amplitude=A,
        log_coef=nan,
        sign=0.0,
        local=local,
        coef_rho=Cs,
        line=f"critical, s = {s:.6f} > nu: F -> F0 + A u^kappa, kappa = {kap:.6f}",
    )
