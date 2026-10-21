r"""The central form of the DF in closed form, from the model's parameters alone.

The inversion calls `center_form` for the singular behavior of the DF at
the central potential :math:`P_0`, and `top_limit` and `ceiling_limit` for
the depths at which the DF table ends. Near :math:`P_0`, the auxiliary
density is a series :math:`\varrho_{\rm aux} = \sum_s C_s u^s` in the depth
:math:`u = P_0 - P` (`rho_aux_series`). The least power :math:`s` of the
series that is not a non-negative integer gives the leading singular term
of the DF :math:`F` at the center, :math:`F \simeq A u^{\kappa}` with
amplitude :math:`A`, :math:`\kappa = s - \nu` and
:math:`\nu = 3/2 - \beta_{\rm c}`, or a term in :math:`\ln u` where
:math:`s = \nu`. The next power of the series above :math:`s` that is not
a non-negative integer gives the DF's next singular term. `center_form`
returns the branch (regular, critical or unphysical), the form
(``"power"``, ``"log"``, ``"constant_plus_power"``, ``"analytic"`` or
``"none"``), the exponent and the closed-form amplitude. No value of the DF
table is read. Two powers of the series are taken as equal where the two powers differ
by at most `TOL_EXACT_BRANCH` (:math:`10^{-10}`), and :math:`\gamma` and
:math:`2 \beta_{\rm c}` are taken as equal where :math:`\gamma` and
:math:`2 \beta_{\rm c}` differ by at most `TOL_EXACT_BRANCH`.

References
----------
Notes Section T:2.2.1, Notes Section T:2.2.2, Notes Section T:2.2.3,
Notes Section T:2.5.1.
"""

__all__ = [
    "LOG10_DF_CEILING",
    "PATCHED_BRANCHES",
    "TOL_EXACT_BRANCH",
    "ceiling_limit",
    "central_form",
    "center_form",
    "rho_aux_series",
    "top_limit",
]

from functools import lru_cache
from math import factorial
from typing import Any

import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import gammaln, gammasgn

from ..anisotropy import Anisotropy
from ..profiles import DoublePowerLaw

# The largest difference at which two numbers of the branch choice count as
# equal: gamma against 2 beta_c, a power s against the nearest integer, and
# two powers of the series against each other.
TOL_EXACT_BRANCH = 1e-10
# The base-10 logarithm of the largest DF value a table holds. The DF
# values of a table stay at or below 10^LOG10_DF_CEILING (`ceiling_limit`),
# 8 decades below the largest double (about 1.8e308), so that the
# inversion's arithmetic on the DF stays within the range of doubles. The
# inversion (`invert.py`) and the table checks (`checks.py`) also read
# LOG10_DF_CEILING.
LOG10_DF_CEILING = 300.0
# The central branches whose closed form the inversion carries above the
# table's last row, through the anchor and the top patch (`top.py`).
PATCHED_BRANCHES = ("regular", "critical_divergent")


def _gamma_ratio(a, b):
    r"""Return :math:`\Gamma(a) / \Gamma(b)` from the logarithms and signs.

    The quotient of the two Gamma values is never formed: :math:`\Gamma(x)`
    exceeds the largest double for :math:`x` above 171.6, while the ratio
    :math:`\Gamma(a) / \Gamma(b)` can still be a finite double.
    """
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
    r"""Return the series of the auxiliary density in the depth :math:`u = P_0 - P`.

    The density is :math:`\rho = C R^{-\gamma} (1 + R^{\alpha})^e` with
    :math:`e = (\gamma - \beta_r) / \alpha`, where :math:`\beta_r` is the
    outer slope ``beta_r``. The depth
    :math:`u = c_u R^{2 - \gamma} (1 + D(R))`, with
    :math:`c_u = C / ((2 - \gamma)(3 - \gamma))` and :math:`D` a power
    series in :math:`R^{\alpha}`, is inverted for
    :math:`R = t (1 + S(t))`, with :math:`t = (u / c_u)^{1 / (2 - \gamma)}`
    and :math:`S` a power series in :math:`t`. The auxiliary density is
    :math:`\varrho_{\rm aux} = C R^q (1 + R^{\alpha})^e A(R)`, with
    :math:`q = 2 \beta_{\rm c} - \gamma` for ``m_typ`` "constant", "COM_I"
    or "COM_II", :math:`q = -\gamma` for ``m_typ`` "OM_I" or "OM_II", and
    :math:`A` the anisotropy factor of anisotropy radius ``Ra`` and model
    type ``m_typ``. The coefficient of every power :math:`t^{q + x}` with
    :math:`x` at most `x_max` is exact.

    Parameters
    ----------
    x_max : float or None, optional
        The highest power :math:`x` of :math:`t^{q + x}` that the series
        keeps. Default: ``None``, which means three times the larger of
        :math:`\alpha` and 2.
    tol : float or None, optional
        The largest difference at which two powers :math:`s` of the series
        are merged. Default: ``None``, which means `TOL_EXACT_BRANCH`.

    Returns
    -------
    series : list of tuple of float
        The pairs :math:`(s, C_s)` of
        :math:`\varrho_{\rm aux} = \sum_s C_s u^s`, sorted by :math:`s`,
        with equal powers merged and zero coefficients dropped.
    """
    return list(
        _rho_aux_series(
            float(alpha),
            float(beta_r),
            float(gamma),
            float(bc),
            m_typ,
            float(Ra),
            float(C),
            None if x_max is None else float(x_max),
            None if tol is None else float(tol),
        )
    )


@lru_cache(maxsize=256)
def _rho_aux_series(alpha, beta_r, gamma, bc, m_typ, Ra, C, x_max, tol):
    """Return `rho_aux_series` as a tuple, cached by its arguments.

    The series depends on its arguments alone, so the cache forms the series
    once per distinct set of arguments.
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
        # The binomial series of (1 + S)^p, for S a series of positive powers
        # only, truncated at the power xm.
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
    # The inverse R = t (1 + S) satisfies
    # S = (1 + D(t (1 + S)))^(-1 / (2 - gamma)) - 1. Each pass of the fixed
    # point iteration makes S exact to one more order of t^alpha.
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
    return tuple(sorted((s, c) for s, c in out.items() if c != 0.0))


def central_form(alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=None):
    """Return the leading and the next singular term of the DF at the center.

    The leading term comes from `_central_form_leading`, the next term from
    `_central_next_term`.

    Returns
    -------
    form : dict
        The leading term's keys ``branch``, ``form``, ``s``, ``kappa``,
        ``kappa_sing``, ``amplitude``, ``log_coef``, ``sign``, ``local``,
        ``coef_rho`` and ``line``, and the next term's keys ``next_form``,
        ``next_s``, ``next_kappa``, ``next_coef``, ``next_coef_rho`` and
        ``next_line``.
    """
    res = _central_form_leading(alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=tol)
    res.update(_central_next_term(res, alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=tol))
    return res


def center_form(profile: DoublePowerLaw, anisotropy: Anisotropy) -> dict[str, Any]:
    r"""Return the central form of the DF (`central_form`) of one component.

    The component has the profile ``profile`` and the anisotropy
    ``anisotropy``. The density amplitude :math:`C` is the profile's own
    ``rho_norm``. The model type follows from the anisotropy's family, and
    an OM model enters with :math:`\beta_{\rm c} = 0`.
    """
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
    r"""Return the least depth with finite derivatives of the auxiliary density.

    The depth is :math:`u = P_0 - P`, in internal units, computed from the
    model's parameters alone. Each term :math:`C_s u^s` of `rho_aux_series`
    gives the derivative term
    :math:`d^N \varrho_{\rm aux} / dP^N = (-1)^N C_s\, s (s - 1) \cdots
    (s - N + 1)\, u^{s - N}`. For :math:`s < N`, the derivative term exceeds
    the largest double :math:`D_{\rm max}` at depths below
    :math:`u = (|C_s\, s \cdots (s - N + 1)| / D_{\rm max})^{1 / (N - s)}`.
    The returned depth is the largest such depth over the terms and the
    orders :math:`N`, and 0 where no derivative term grows toward the
    center.

    Parameters
    ----------
    orders : tuple of int or None, optional
        The derivative orders :math:`N`. Default: ``None``, which means the
        orders :math:`j - 1` to :math:`j + 2`, with :math:`j` the
        inversion's derivative order (`order_pair`) and :math:`j - 1`
        raised to 0 where :math:`j = 0`.
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


def ceiling_limit(profile: DoublePowerLaw, anisotropy: Anisotropy) -> float:
    r"""Return the least depth where the DF's central form is at most :math:`10^{300}`.

    The central closed form is :math:`F = A u^{\kappa}` (`center_form`) in the
    depth :math:`u = P_0 - P`. The returned depth is
    :math:`u = (10^{L} / A)^{1 / \kappa}`, with :math:`L` the constant
    `LOG10_DF_CEILING` (300), computed from the model's parameters alone.
    The returned depth is 0 in four cases, in each of which the table
    carries no closed form above its last row: the anisotropy is Type II;
    the central branch is not in `PATCHED_BRANCHES`; the central form is
    not ``"power"``; or not both :math:`\kappa < 0` and :math:`A > 0`.
    """
    form = center_form(profile, anisotropy)
    kappa, amplitude = float(form["kappa"]), float(form["amplitude"])
    if (
        anisotropy.kind == "II"
        or form["branch"] not in PATCHED_BRANCHES
        or form["form"] != "power"
        or not (kappa < 0.0 and amplitude > 0.0)
    ):
        return 0.0
    return float(10.0 ** ((LOG10_DF_CEILING - np.log10(amplitude)) / kappa))


def _central_next_term(res, alpha, beta_r, gamma, bc, m_typ, Ra, C, tol=None):
    r"""Return the singular term of the central DF next after the leading term ``res``.

    The next term comes from the least power :math:`s_2` of `rho_aux_series`
    above the leading power :math:`s` that is not a non-negative integer.
    Here :math:`\nu = 3/2 - \beta_{\rm c}`,
    :math:`N' = 2^{\beta_{\rm c}} / ((2 \pi)^{3/2} \Gamma(1 - \beta_{\rm c}))`,
    :math:`C_{s_2}` is the series' coefficient at :math:`s_2`, and
    :math:`\beta_{\rm c}` is taken as 0 for an OM model. Where
    :math:`s_2 - \nu` is a non-negative integer :math:`k`, the next term is
    :math:`u^k` times a logarithm of :math:`u`, with coefficient
    :math:`L_2 = N' C_{s_2} (-1)^k / (k!\, \Gamma(-s_2))` and ``next_form``
    "log_power". Otherwise the next term is :math:`A_2 u^{s_2 - \nu}`, with
    :math:`A_2 = N' C_{s_2} \Gamma(\nu - s_2) / \Gamma(-s_2)` and
    ``next_form`` "power". The keys ``next_form``, ``next_s``,
    ``next_kappa``, ``next_coef`` and ``next_coef_rho`` are "none" and NaN
    where no next term exists, in three cases: the leading form is "none" or
    "analytic"; the series has no power at :math:`s`, or the series'
    coefficient at :math:`s` differs from the leading term's ``coef_rho`` by
    more than :math:`10^{-9}` relative; or no power that is not a
    non-negative integer lies between :math:`s` and the series' highest
    exact power. ``next_line`` states the result in words.
    """
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
    # A non-negative integer power of u is analytic and has no singular part
    # in the DF; a negative integer power is singular and stays a candidate.
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
    r"""Return the leading singular term of the DF at the center.

    The least power :math:`s` of :math:`\varrho_{\rm aux}` that is not a
    non-negative integer, with coefficient :math:`C_s`, gives
    :math:`F \simeq N' C_s \Gamma(\nu - s) / \Gamma(-s)\, u^{s - \nu}`, or
    :math:`F \simeq N' C_s / \Gamma(-\nu)\, (-\ln u)` at :math:`s = \nu`,
    with :math:`N' = 2^{\beta_{\rm c}} / ((2 \pi)^{3/2}
    \Gamma(1 - \beta_{\rm c}))` and :math:`\nu = 3/2 - \beta_{\rm c}`, and
    with :math:`\beta_{\rm c}` taken as 0 for an OM model. On the regular
    branch, :math:`2 \beta_{\rm c} < \gamma < 2`, the leading term of
    :math:`\varrho_{\rm aux}` is :math:`C (u / c_u)^{s_0}` with
    :math:`s_0 = (2 \beta_{\rm c} - \gamma) / (2 - \gamma)` and
    :math:`c_u = C / ((2 - \gamma)(3 - \gamma))`. Where
    :math:`\gamma = 2 \beta_{\rm c}` (the critical branch), the leading
    powers come from the terms in :math:`R^{\alpha}` and, for a COM model,
    in :math:`R^2 / R_{\rm a}^2`. Where :math:`\gamma < 2 \beta_{\rm c}`, or
    :math:`\gamma \ge 2`, the branch is "unphysical_center" with no form.

    Returns
    -------
    form : dict
        The keys ``branch``, ``form``, ``s``, ``kappa``, ``kappa_sing``,
        ``amplitude``, ``log_coef``, ``sign``, ``local``, ``coef_rho`` and
        ``line``. ``local`` is True where :math:`\nu` is an integer, so that
        the inversion is local.
    """
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
