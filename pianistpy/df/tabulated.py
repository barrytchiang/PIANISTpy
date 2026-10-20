r"""The inversion's sources on a tabulated pair: a component on the
truncated profile of another model (Method II of Chiang+(2026d)), with its own
anisotropy.

The component's auxiliary density is the profile's density times its
factor, :math:`\varrho_{\rm aux} = g(R)\,\hat\varrho`, both read on the
profile's potential. Its derivatives are not taken from a spline of the
rows: the profile's density is the integral of its model's DF,

.. math::

    \hat\varrho(P) = 4\pi\sqrt{2}\int_0^P F(K)\,(P - K)^{1/2}\,dK,

so its derivatives are integrals of the DF and its derivatives against
:math:`(P - K)^{-1/2}` (`parent_sources`), and the factor's derivatives
follow by the chain rule on :math:`R(P)` from the rows. The rows of the
inversion and the pilot that places the grid follow (`child_rows`,
`pilot_feature`).
"""

__all__ = [
    "child_model",
    "child_rows",
    "factor_sources",
    "floor_record",
    "parent_density",
    "parent_sources",
    "pilot_feature",
    "refusals",
    "tabulated_df",
    "tabulated_grid",
]

import dataclasses
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator

from ..anisotropy.factor import inverse_factor
from ..exceptions import AccuracyError, InputError, ResultError, UnphysicalModelError
from ..profiles import DoublePowerLaw
from ..runtime import timed
from ..settings import Settings
from ..validation import HARD_LIMIT, Check, ValidationReport, announce
from .central import center_form
from .checks import measure_table
from .derivatives import aux_derivatives
from .edge_law import edge_expansion_at_node, edge_law
from .form import inversion_form
from .grid import (
    CEILING_GAP,
    K_FLOOR,
    TABLE_FLOOR,
    GridZones,
    feature_zones,
    hybrid_grid,
    standard_zones,
)
from .inputs import Component
from .inversion import inversion_prefactor, order_pair
from .pair import potential_density_pair
from .profile_checks import PROFILE_CHECK_KEYS, profile_checks
from .report import (
    NO_SLOPE_COLUMN_CHILD,
    slope_column_not_judged,
    stated_forms,
    validate_df,
)
from .rows import complete_rows, floor_source, inversion_kernel, kernel_rules
from .stages import (
    AsymptoticForm,
    DfTable,
    EnergyGrid,
    InversionRecord,
    InversionRows,
)
from .top import anchor

# 4 pi sqrt 2 and its half: rho_hat = 4 pi sqrt2 INT F (P - K)^(1/2) dK, and
# rho_hat' = 2 pi sqrt2 INT F (P - K)^(-1/2) dK
_A = 4.0 * np.pi * np.sqrt(2.0)
_PREF = 2.0 * np.pi * np.sqrt(2.0)
# the quadrature of the DF integrals: 600 Gauss-Legendre nodes (converged:
# 2400 nodes move the critical branch's rows by < 1e-3 of their residual)
_N_GAUSS = 600
_TINY = 1e-300
# the branches of the center whose DF diverges at p_0: the only ones a top
# band is judged against
_PATCHED = ("regular", "critical_divergent")


def parent_sources(
    q: np.ndarray,
    p_top: float,
    order: int,
    parent_df: dict[str, Any],
    parent_rows: dict[str, Any],
    parent_r_a: float | None,
    family: str,
    beta_c: float,
    r_a: float,
) -> dict[str, Any]:
    r"""Return the child's auxiliary density and its derivatives on the
    energies q, from the parent's DF and rows.

    Parameters
    ----------
    q : array
        The child's energies (its grid).
    p_top : float
        The top of the window: the parent's first row's potential. The
        sources are made on ``K_min <= q < p_top`` (``K_min < q`` without
        the parent's Taylor edge), K_min the floor of the parent's table.
    order : int
        The highest derivative made, 1 to 3.
    parent_df : dict
        The parent's table: ``K``, ``logF`` (log10 F) and ``dFdK``, and
        ``taylor``, the edge's (F_1, F_2, K_switch) or None.
    parent_rows : dict
        The parent's rows: ``R``, ``Psi``, ``rho``, ``M`` (4 pi times the
        enclosed mass), ``u`` (``P_0 - Psi``, exact near the center) and ``p_0``
        (the parent's central potential, 1 - Z_t).
    parent_r_a : float or None
        The parent's :math:`R_{\rm a}` where it is Osipkov-Merritt Type I;
        None where it is isotropic.
    family : str
        The child's anisotropy family.
    beta_c, r_a : float
        The child's central anisotropy and anisotropy radius.

    Returns
    -------
    sources : dict
        ``window`` (the nodes made), ``rho`` (the parent's density there),
        ``aux`` (order to rho_aux^(order) for orders 1 to `order`), the
        factor ``g``, its slope in P ``g_p`` (g_R R'), ``r`` and ``r_p``
        (R and R' there), ``h`` and ``h_1`` (the OM parent's h and h'; None
        for an isotropic parent).

    Raises
    ------
    InputError
        If the order is above 3; if the order is 3 and the parent states no
        Taylor edge; if the parent's u is not ``P_0 - Psi`` to the rounding of
        Psi.
    """
    density = parent_density(q, p_top, order, parent_df, parent_rows, parent_r_a)
    return factor_sources(density, family, beta_c, r_a)


def parent_density(
    q: np.ndarray,
    p_top: float,
    order: int,
    parent_df: dict[str, Any],
    parent_rows: dict[str, Any],
    parent_r_a: float | None,
) -> dict[str, Any]:
    r"""Return the parent's part of `parent_sources` on the energies q: the
    density's derivatives rho_hat^(k) (an OM I parent's divided by its h),
    R(P) and its derivatives, and the window; none depends on the child.

    Parameters and Raises as `parent_sources`.
    """
    if not 1 <= order <= 3:
        raise InputError(
            f"Method II derivatives of order {order}: orders up to 3 are supported."
        )
    taylor = parent_df.get("taylor")
    if order == 3 and taylor is None:
        raise InputError(
            "Order-3 Method II derivatives need the Taylor edge of the parent "
            "(F_1, F_2, K_switch): the parent is not a truncated Method I model."
        )
    p0h = float(parent_rows["p_0"])
    om_parent = parent_r_a is not None

    # the parent's DF read against the logit of its own axis, x = ln(K / u)
    k_all = np.asarray(parent_df["K"], dtype=np.float64)
    lg_all = np.asarray(parent_df["logF"], dtype=np.float64)
    okt = (k_all > 0) & np.isfinite(lg_all)
    k_t = k_all[okt]

    def xc(k: Any, uk: Any = None) -> Any:
        k = np.maximum(np.asarray(k, dtype=np.float64), _TINY)
        uk = (p0h - k) if uk is None else uk
        return np.log(k) - np.log(np.maximum(uk, _TINY))

    def dxdlnk(k: Any, uk: Any = None) -> Any:
        k = np.asarray(k, dtype=np.float64)
        uk = (p0h - k) if uk is None else uk
        return 1.0 + k / np.maximum(uk, _TINY)

    l_f = PchipInterpolator(xc(k_t), lg_all[okt] * np.log(10.0))
    k_min, k_max = float(k_t[0]), float(k_t[-1])
    d_all = np.asarray(parent_df["dFdK"], dtype=np.float64)
    okd = np.isfinite(d_all) & (k_all > 0)
    k_a, d_a = k_all[okd], d_all[okd]
    f_a = np.exp(l_f(xc(np.clip(k_a, k_min, k_max))))
    s_a = d_a * k_a / np.maximum(f_a, _TINY)
    l_s = PchipInterpolator(xc(k_a), s_a)
    x_lo, x_hi = float(xc(k_min)), float(xc(k_max))
    x3_lo, x3_hi = float(xc(k_a[0])), float(xc(k_a[-1]))
    if taylor is not None:
        f_1, f_2, k_sw = (float(v) for v in taylor)

    def f_of(k: Any, uk: Any = None) -> Any:
        k = np.asarray(k, dtype=np.float64)
        out = np.exp(l_f(np.clip(xc(k, uk), x_lo, x_hi)))
        out[k < k_min] = 0.0
        if taylor is not None:
            t = k < k_sw
            out[t] = np.maximum(k[t], 0.0) * (f_1 + 0.5 * f_2 * k[t])
        return out

    def df_of(k: Any, uk: Any = None) -> Any:
        k = np.asarray(k, dtype=np.float64)
        kc = np.clip(k, k_min, k_max)
        x = np.clip(xc(k, uk), x_lo, x_hi)
        out = np.exp(l_f(x)) * l_s(np.clip(x, x3_lo, x3_hi)) / kc
        out[k < k_min] = 0.0
        if taylor is not None:
            t = k < k_sw
            out[t] = np.where(k[t] >= 0.0, f_1 + f_2 * k[t], 0.0)
        return out

    p = np.asarray(q, dtype=np.float64)
    ok = ((p >= k_min) if taylor is not None else (p > k_min)) & (p < float(p_top))
    xg, wg = np.polynomial.legendre.leggauss(_N_GAUSS)
    tg, wtg = 0.5 * (xg + 1.0), 0.5 * wg

    def integral(fun: Any) -> Any:
        # INT_0^P fun(K) (P - K)^(-1/2) dK on the nodes u_K = u_P e^(v^2): the
        # DF's peak at K -> P_0 and the kernel's singularity at K -> P resolved
        out = np.zeros_like(p)
        gap = p0h - p[ok]
        big_v = np.sqrt(np.log(p0h / gap))
        v = big_v[:, None] * tg[None, :]
        y = v * v
        uk = gap[:, None] * np.exp(y)
        fk = fun(p0h - uk, uk)
        kj = (
            2.0
            * big_v[:, None]
            * np.sqrt(gap[:, None])
            * (v / np.sqrt(np.expm1(y)))
            * np.exp(y)
        )
        out[ok] = np.sum(wtg[None, :] * fk * kj, axis=1)
        return out

    rho1 = _PREF * integral(f_of)
    rho2 = _PREF * integral(df_of)
    rho3: Any = None
    if order == 3:
        l_sd = l_s.derivative()

        def d2f_of(k: Any, uk: Any = None) -> Any:
            k = np.asarray(k, dtype=np.float64)
            kc = np.clip(k, k_min, k_max)
            x = np.clip(xc(k, uk), x_lo, x_hi)
            x3 = np.clip(x, x3_lo, x3_hi)
            s = l_s(x3)
            dsdlnk = l_sd(x3) * dxdlnk(
                kc, None if uk is None else np.maximum(uk, p0h - k_max)
            )
            out = np.exp(l_f(x)) * (s * s - s + dsdlnk) / kc**2
            out[k < k_min] = 0.0
            if taylor is not None:
                t = k < k_sw
                out[t] = np.where(k[t] >= 0.0, f_2, 0.0)
            return out

        with np.errstate(divide="ignore"):
            rho3 = np.where(
                ok, _PREF * (f_1 * np.where(ok, p, 1.0) ** -0.5 + integral(d2f_of)), 0.0
            )
    if taylor is not None:
        # below the switch the parent is its Taylor form: rho_hat' .. rho_hat'''
        # in closed form
        me = ok & (p < k_sw)
        rho1 = np.where(
            me, _A * ((2.0 / 3.0) * f_1 * p**1.5 + (4.0 / 15.0) * f_2 * p**2.5), rho1
        )
        rho2 = np.where(me, _A * (f_1 * p**0.5 + (2.0 / 3.0) * f_2 * p**1.5), rho2)
        if order == 3:
            with np.errstate(divide="ignore"):
                rho3 = np.where(
                    me,
                    _A * (0.5 * f_1 * np.where(me, p, 1.0) ** -0.5 + f_2 * p**0.5),
                    rho3,
                )

    # R(P), rho_hat(P), M(P) from the parent's rows, read against the logit
    # ln(Psi / u) of its central coordinate
    r_t = np.asarray(parent_rows["R"], dtype=np.float64)
    rho_t = np.asarray(parent_rows["rho"], dtype=np.float64)
    psi_t = np.asarray(parent_rows["Psi"], dtype=np.float64)
    m_t = np.asarray(parent_rows["M"], dtype=np.float64)
    u_t = np.asarray(parent_rows["u"], dtype=np.float64)
    if u_t.size != psi_t.size:
        raise InputError("The parent's rows carry no central coordinate u.")
    ok_t = (psi_t > 0) & (rho_t > 0) & (m_t > 0) & (r_t > 0)
    off = float(np.max(np.abs((p0h - psi_t) - u_t)))
    if not off <= 4.0 * np.finfo(np.float64).eps * p0h:
        raise InputError(
            f"The parent's u is not P_0 - Psi to the rounding of Psi (max difference "
            f"{off:.3e}, P_0 = {p0h!r}): not the central coordinate of its rows."
        )
    ok_t = ok_t & (u_t > 0)
    lam = np.log(psi_t[ok_t]) - np.log(u_t[ok_t])
    o = np.argsort(lam)
    lam_t = lam[o]
    uq = p0h - p
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_q = np.clip(
            np.log(np.where(ok, p, 1.0)) - np.log(np.where(ok & (uq > 0), uq, 1.0)),
            lam_t[0],
            lam_t[-1],
        )
    rr = np.exp(PchipInterpolator(lam_t, np.log(r_t[ok_t][o]))(lam_q))
    rho0 = np.exp(PchipInterpolator(lam_t, np.log(rho_t[ok_t][o]))(lam_q))
    mm = np.exp(PchipInterpolator(lam_t, np.log(m_t[ok_t][o]))(lam_q))
    rp = -4.0 * np.pi * rr**2 / mm
    mp = 4.0 * np.pi * rr**2 * rho0 * rp
    rpp = -4.0 * np.pi * (2.0 * rr * rp * mm - rr**2 * mp) / mm**2
    h = h_1 = None
    if om_parent:
        # an OM I parent: the integrals are its augmented density h rho_hat,
        # h = 1 + R^2 / r_a^2; rho_hat^(k) by Leibniz with h's derivatives in P
        ra_p = float(parent_r_a)  # type: ignore[arg-type]
        h, h_r, h_rr, *h_rrr = inverse_factor(rr, "OM_I", 0.0, ra_p, max(order, 2))
        h_1 = h_r * rp
        h_2 = h_rr * rp**2 + h_r * rpp
        rho1 = (rho1 - h_1 * rho0) / h
        rho2 = (rho2 - 2.0 * h_1 * rho1 - h_2 * rho0) / h
        if order == 3:
            mpp_h = (
                4.0
                * np.pi
                * (2.0 * rr * rp * rho0 * rp + rr**2 * rho1 * rp + rr**2 * rho0 * rpp)
            )
            rppp_h = (
                -4.0
                * np.pi
                * (
                    ((2.0 * rp**2 + 2.0 * rr * rpp) * mm - rr**2 * mpp_h) / mm**2
                    - 2.0 * (2.0 * rr * rp * mm - rr**2 * mp) * mp / mm**3
                )
            )
            h_3 = h_rrr[0] * rp**3 + 3.0 * h_rr * rp * rpp + h_r * rppp_h
            rho3 = (rho3 - 3.0 * h_1 * rho2 - 3.0 * h_2 * rho1 - h_3 * rho0) / h
    return dict(
        order=order,
        window=ok,
        rho=(rho0, rho1, rho2, rho3),
        r=rr,
        r_p=rp,
        r_pp=rpp,
        m=mm,
        m_p=mp,
        h=h,
        h_1=h_1,
        p0_hat=p0h,
        taylor=None if taylor is None else (f_1, f_2, k_sw),
    )


def factor_sources(
    density: dict[str, Any],
    family: str,
    beta_c: float,
    r_a: float,
    factor: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    r"""Return `parent_sources` from the parent's part (`parent_density`) and
    the child's factor ``g = R^(2 beta_c) u^s``, ``u = 1 +- R^2 / r_a^2``. The
    sources are linear in g: with ``factor`` (g and its R-derivatives to the
    density's order, on its R) they are made with it in place of g."""
    order = density["order"]
    rho0, rho1, rho2, rho3 = density["rho"]
    rr, rp, rpp = density["r"], density["r_p"], density["r_pp"]
    mm, mp = density["m"], density["m_p"]
    b, ra = float(beta_c), float(r_a)
    if factor is None:
        factor = inverse_factor(rr, family, b, ra, max(order, 2))
    g, g_r, g_rr = factor[:3]
    aux = {
        1: rho1 * g + rho0 * g_r * rp,
        2: rho2 * g + 2.0 * rho1 * g_r * rp + rho0 * (g_rr * rp**2 + g_r * rpp),
    }
    if order == 3:
        g_rrr = factor[3]
        a1 = 2.0 * rr * rp
        a2 = 2.0 * rp**2 + 2.0 * rr * rpp
        mpp = (
            4.0
            * np.pi
            * (2.0 * rr * rp * rho0 * rp + rr**2 * rho1 * rp + rr**2 * rho0 * rpp)
        )
        rppp = (
            -4.0
            * np.pi
            * (
                (a2 * mm - rr**2 * mpp) / mm**2
                - 2.0 * (a1 * mm - rr**2 * mp) * mp / mm**3
            )
        )
        g1 = g_r * rp
        g2 = g_rr * rp**2 + g_r * rpp
        g3 = g_rrr * rp**3 + 3.0 * g_rr * rp * rpp + g_r * rppp
        aux[3] = rho3 * g + 3.0 * rho2 * g1 + 3.0 * rho1 * g2 + rho0 * g3
    return dict(
        window=density["window"],
        rho=rho0,
        aux=aux,
        g=g,
        g_p=g_r * rp,
        r=rr,
        r_p=rp,
        h=density["h"],
        h_1=density["h_1"],
        p0_hat=density["p0_hat"],
        taylor=density["taylor"],
    )


def floor_record(
    q: np.ndarray, sources: dict[str, Any], anisotropy: Any
) -> dict[str, Any]:
    r"""Return the sources at the lowest node of the window, for the
    inversion's boundary term, and with the parent's Taylor edge the child's
    edge expansion there (`edge_law.edge_expansion_at_node`, the factor and
    its slope at the floor node, an OM I parent's divided out) and the edge
    law it inverts to (`edge_law.edge_law`).

    Returns
    -------
    floor : dict
        ``P`` (the node), ``aux`` (order to rho_aux^(order) there, order 0
        included), and with the edge ``a_q`` (the powers' coefficients) and
        ``law`` (``p``, ``C0``, ``C1``).
    """
    p = np.asarray(q, dtype=np.float64)
    ok = sources["window"]
    i0 = int(np.argmin(np.where(ok, p, np.inf)))
    g = sources["g"]
    out: dict[str, Any] = dict(
        P=float(p[i0]),
        aux={
            0: float(sources["rho"][i0] * g[i0]),
            **{k: float(v[i0]) for k, v in sources["aux"].items()},
        },
    )
    if sources["taylor"] is None:
        return out
    f_1, f_2, k_sw = sources["taylor"]
    h, h_1 = sources["h"], sources["h_1"]
    expansion = edge_expansion_at_node(
        f_1,
        f_2,
        k_sw,
        float(p[i0]),
        float(g[i0]),
        float(sources["g_p"][i0]),
        anisotropy,
        None if h is None else (float(h[i0]), float(h_1[i0])),
    )
    law_p, c_0, c_1 = edge_law(expansion)
    out["a_q"] = dict(expansion.coefficients)
    out["law"] = dict(p=float(law_p), C0=float(c_0), C1=float(c_1))
    return out


def _pair(anisotropy: Any) -> tuple[Any, float, int, int, bool, bool]:
    r"""The child's pair: (pair, :math:`\beta_{\rm c}` of the kernel, j, the
    source's order, half-integer, switched); Osipkov-Merritt inverts with the
    isotropic pair."""
    bc = 0.0 if anisotropy.family.startswith("OM") else float(anisotropy.beta_c)
    pair = order_pair(bc)
    j = int(pair.order_j)
    half = pair.kernel_exponent == 0.0
    switched = bool(pair.order_j == pair.order_m + 1)
    return pair, bc, j, (j if half else j + 1), half, switched


def child_rows(
    q: np.ndarray,
    parent_df: dict[str, Any],
    parent_rows: dict[str, Any],
    parent_r_a: float | None,
    anisotropy: Any,
    settings: Settings,
    unit_profile: DoublePowerLaw | None = None,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Return the child's inversion rows on the energies q.

    The kernel reads the parent sources (`parent_sources`) on every node;
    the boundary term at the floor is the sources' own value there, and the
    exact tail below the floor follows from the edge expansion
    (`floor_record`, `rows.method2_floor_tail`).

    With ``unit_profile`` the rows are the pilot's: as the working pipeline
    makes them, the boundary terms are then the untruncated analytic
    profile's rho_aux^(j) and rho_aux^(j-1) at the floor, and the rows from
    ten floors up take the derivative-outside form where rho_aux^(j) is
    unbounded at P = 0. With ``sources`` (`parent_sources`' on q) they are
    read in place of the parent's, the anisotropy giving the kernel and the
    floor's order.

    Returns
    -------
    rows : dict
        ``f`` (the completed rows), ``f_kernel`` (the kernel's), ``floor``
        (`floor_record`), ``tail`` (the floor tail's input, or None),
        ``sources`` and the kernel's inputs ``dm_j``, ``dm_jm1``,
        ``outer_from``.

    Raises
    ------
    InputError
        If a node lies outside the parent's table, where the sources are
        not made.
    """
    pair, bc, j, order, half, switched = _pair(anisotropy)
    p_top = float(np.asarray(parent_rows["Psi"], dtype=np.float64)[0])
    s = sources
    if s is None:
        s = parent_sources(
            q,
            p_top,
            order,
            parent_df,
            parent_rows,
            parent_r_a,
            anisotropy.family,
            float(anisotropy.beta_c),
            float(anisotropy.r_a),
        )
    if not np.all(s["window"]):
        raise InputError(
            f"{int(np.sum(~s['window']))} of the child's energies lie outside the "
            "parent's table, where its sources are not made."
        )
    fl = floor_record(q, s, anisotropy)
    g1 = s["aux"][order]
    g0 = g1 if half else s["aux"][j]
    outer_from = 0.0
    floor = None
    if unit_profile is None:
        dm_j = fl["aux"][j]
        dm_jm1 = fl["aux"][j - 1] if switched else 0.0
    else:
        # the pilot's rows are not compact: the boundary terms, the outer end
        # and the completion below the floor are the untruncated analytic
        # pair's, as an untruncated component's inversion makes them
        q_min = float(np.min(q))
        grid = EnergyGrid(
            k=np.asarray(q, dtype=np.float64)[None, :],
            r_of_k=np.full((1, np.size(q)), np.nan),
            k_floor=(q_min,),
            k_ceiling=(float(np.max(q)),),
        )
        analytic = potential_density_pair((Component(unit_profile, anisotropy),))
        form, (frec,) = inversion_form(analytic, grid)
        exact = aux_derivatives(analytic, grid, form)
        at_floor = dict(zip(exact.orders, exact.at_floor[0], strict=True))
        dm_j = float(at_floor[j])
        dm_jm1 = float(at_floor[j - 1]) if switched else 0.0
        outer_from = float(frec.outer_from)
        floor = floor_source(
            exact,
            frec,
            unit_profile,
            anisotropy,
            q_min,
            int(settings.inversion["n_floor_nodes"]),
        )
    f_k = np.asarray(
        inversion_kernel(
            jnp.asarray(q, dtype=jnp.float64),
            jnp.asarray(g1, dtype=jnp.float64),
            beta_c=bc,
            is_om=anisotropy.family.startswith("OM"),
            rules=kernel_rules(settings),
            dm_j=float(dm_j),
            p_0=p_top,
            g0=jnp.asarray(g0, dtype=jnp.float64),
            dm_jm1=float(dm_jm1),
            switched=switched,
            outer_from=outer_from,
        ),
        dtype=np.float64,
    )
    tail = None
    if "a_q" in fl and (dm_jm1 if switched else dm_j) != 0.0 and pair.order_n != 0.0:
        tail = dict(
            a_q=fl["a_q"],
            m_f=int(pair.order_m),
            n_f=float(pair.order_n),
            pref=inversion_prefactor(pair.order_n, bc),
        )
    f, _ = complete_rows(q, f_k, None, floor=floor, m2_tail=tail)
    return dict(
        f=f,
        f_kernel=f_k,
        floor=fl,
        tail=tail,
        sources=s,
        dm_j=float(dm_j),
        dm_jm1=float(dm_jm1),
        outer_from=outer_from,
    )


def pilot_feature(
    parent_df: dict[str, Any],
    parent_rows: dict[str, Any],
    parent_r_a: float | None,
    anisotropy: Any,
    settings: Settings,
    unit_profile: DoublePowerLaw,
) -> dict[str, Any]:
    """Return the feature an OM Type II or COM child's grid is placed
    around: the pilot's rows on the coarse grid scaled to the parent's first
    potential :math:`P_0`, and the minimum of the DF below the midpoint of the
    preflight zones' B/BC and BC/C boundaries (on :math:`P_0 = 1`), refined by a
    parabola through its neighbors (`invert.pilot_feature`'s rule)."""
    p0 = float(np.asarray(parent_rows["Psi"], dtype=np.float64)[0])
    extent = float(settings.inversion["grid_tail_extent"])
    low = np.logspace(np.log10(K_FLOOR), np.log10(0.05 * p0), 200)
    # the log parts may meet the linear part an ulp off its ends; each seam is
    # the linear part's end node exactly, so the two are one node
    low[-1] = 0.05 * p0
    mid = np.linspace(0.05 * p0, 0.95 * p0, 600)
    high = p0 - low[::-1]
    high[0] = mid[-1]
    q = np.unique(np.concatenate([low, mid, high]))
    f = child_rows(
        q, parent_df, parent_rows, parent_r_a, anisotropy, settings, unit_profile
    )["f"]
    df, qq = np.asarray(f)[2:], q[2:]
    cutoff = 0.5 * (5.0 / 8.0 + (1.0 - extent))
    pos = df[(qq < cutoff) & (df > 0)]
    df_floor = float(np.median(pos)) * 1e-3 if len(pos) >= 3 else 0.0
    safe = (qq < cutoff) & (df > df_floor)
    df_s, q_s = df[safe], qq[safe]
    if len(df_s) < 3:
        q_feature = 0.0
    else:
        i = max(1, min(int(np.argmin(df_s)), len(df_s) - 2))
        qa, qb, qc = float(q_s[i - 1]), float(q_s[i]), float(q_s[i + 1])
        fa, fb, fc = df_s[i - 1], df_s[i], df_s[i + 1]
        var = max(abs(fc - fa), abs(fb - fa), abs(fc - fb), 1e-300)
        denom = 2.0 * ((fb - fa) * (qc - qb) - (fc - fb) * (qb - qa))
        if abs(denom) > 0.01 * var * (qc - qa):
            q_feature = float(np.clip(qb - 0.5 * (fc - fa) * (qc - qa) / denom, qa, qc))
        else:
            q_feature = qb
        if q_feature < K_FLOOR * 1e3 or q_feature >= cutoff:
            q_feature = 0.0
    return dict(q_feature=q_feature)


def tabulated_grid(
    parent_rows: dict[str, Any],
    anisotropy: Any,
    settings: Settings,
    unit_profile: DoublePowerLaw,
    feature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, GridZones]:
    r"""Return the energies a component on a tabulated pair is inverted on,
    and the zones: the hybrid grid on :math:`P_0` = the profile's first
    potential, from the larger of the floor and its last potential, compact
    (no table extension below, no interior-point nodes). An OM Type I
    model's zones are placed around :math:`\Psi(R_{\rm a})` of the
    untruncated analytic potential, an OM Type II or COM model's around
    its pilot's feature."""
    psi = np.asarray(parent_rows["Psi"], dtype=np.float64)
    p0 = float(psi[0])
    group = settings.inversion
    extent = float(group["grid_tail_extent"])
    family = anisotropy.family
    if family == "OM_I":
        psi_ra = float(unit_profile.p(np.array([float(anisotropy.r_a)]))[0])
        zones = feature_zones(p0, extent, K_FLOOR, psi_ra)
    elif family in ("OM_II", "COM_I", "COM_II"):
        if feature is None:
            raise ValueError(f"An {family} model's grid needs its pilot's feature.")
        zones = feature_zones(p0, extent, K_FLOOR, float(feature["q_feature"]))
    else:
        zones = standard_zones(p0, extent, K_FLOOR)
    q = hybrid_grid(
        zones,
        int(group["n_grid_rows"]),
        int(group["n_grid_tail_rows"]),
        float(psi[-1]),
        K_FLOOR,
        TABLE_FLOOR,
        int(group["table_nodes_per_decade"]),
        CEILING_GAP,
    )
    return q, zones


def critical_band_series(
    c: float,
    beta_r: float,
    d_0: float,
    d_1: float,
    order: int = 5,
    r_a: float = math.inf,
    higher: tuple[float, ...] = (),
) -> np.ndarray:
    r"""Return the coefficients :math:`a_0 .. a_{\rm order}` of
    :math:`d\varrho_{\rm aux}/d\hat\Psi = \sum a_k u^k` near the center of a
    gamma = 1, alpha = 1 profile :math:`\rho = C R^{-1} (1 + R)^{1 - \beta_r}`
    truncated in phase space, the inversion of the critical branch
    :math:`\beta_{\rm c} = 1/2`: :math:`\varrho_{\rm aux} = R\hat\varrho`
    (constant beta), or :math:`R\hat\varrho\,(1 + R^2/R_{\rm a}^2)^{1/2}`
    (COM Type I, finite `r_a`: :math:`\varrho/g` with g of Eq. B43 at
    :math:`\beta_{\rm c} = 1/2`), the center that describes the table at
    :math:`\beta_{\rm c} = +1/2`.

    Plain power-series arithmetic in R, then the reversion R(u). The
    truncation enters as the deficit at a given depth, :math:`\hat\varrho(u)
    = g(u) - \Delta_0 + \Delta_1 u - \Delta_2 u^2 + \dots` with g the
    untruncated density against depth (`higher` holds :math:`\Delta_2,
    \Delta_3, \dots`; none, the deficit is linear), and the lowered model's
    own radius at that depth by a fixed point (at :math:`\Delta = 0`: NFW 4,
    -10/3, 14/3; Hernquist 6, -4, 4). The COM factor is the binomial series
    in :math:`R^2/R_{\rm a}^2`; it multiplies the auxiliary density only
    (the mass and the depth are the profile's), and is 1 at infinite `r_a`,
    where the constant series is returned as before.
    """
    n = int(order) + 4

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.convolve(a, b)[:n]

    def integ(a: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], a[:-1] / np.arange(1, n)])[:n]

    binom = np.ones(n)
    for k in range(1, n):
        binom[k] = binom[k - 1] * (1.0 - float(beta_r) - (k - 1)) / k
    rho_r = float(c) * binom
    r_ser = np.zeros(n)
    r_ser[1] = 1.0

    def compose(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        pw = np.zeros(n)
        pw[0] = 1.0
        for k in range(n):
            out = out + a[k] * pw
            pw = mul(pw, x)
        return out

    def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        qq = np.zeros(n)
        for k in range(n):
            qq[k] = (a[k] - np.dot(qq[:k], b[1 : k + 1][::-1])) / b[0]
        return qq

    def revert(b: np.ndarray) -> np.ndarray:
        ru = np.zeros(n)
        ru[1] = 1.0 / b[1]
        for _ in range(n):
            acc = np.zeros(n)
            pw = mul(ru, ru)
            for k in range(2, n):
                acc = acc + b[k] * pw
                pw = mul(pw, ru)
            new = -acc / b[1]
            new[1] += 1.0 / b[1]
            ru = new
        return ru

    m_n = integ(mul(rho_r, r_ser))
    u_n = integ(np.concatenate([m_n[2:], [0.0, 0.0]]))
    r_n = revert(u_n)
    g = compose(rho_r, r_n)
    uh = u_n.copy()
    for _ in range(n):
        s = np.concatenate([compose(r_n, uh)[1:], [0.0]])
        rrh = (
            divide(compose(g, uh), s) - float(d_0) * r_ser + float(d_1) * mul(r_ser, uh)
        )
        pw = mul(r_ser, uh)
        for k, d_k in enumerate(higher, start=2):
            pw = mul(pw, uh)
            rrh = rrh - (-1.0) ** k * float(d_k) * pw
        mh = integ(mul(rrh, r_ser))
        uh = integ(np.concatenate([mh[2:], [0.0, 0.0]]))
    aux = rrh
    if math.isfinite(float(r_a)):
        h_ser = np.zeros(n)
        bj = 1.0
        for j in range(0, (n + 1) // 2):
            if j > 0:
                bj *= (0.5 - (j - 1)) / j
            h_ser[2 * j] = bj / float(r_a) ** (2 * j)
        aux = mul(rrh, h_ser)
    num = np.concatenate([(aux * np.arange(n))[1:], [0.0]])
    den = -np.concatenate([mh[2:], [0.0, 0.0]])
    out = compose(divide(num, den), revert(uh))
    return out[: int(order) + 1]


def critical_band(
    k: np.ndarray,
    f: np.ndarray,
    p_0: float,
    extent: float,
    unit_profile: DoublePowerLaw,
    deficit: tuple[float, ...],
    r_a: float = math.inf,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the rows with the critical branch's closed-form top band, and
    its record.

    At :math:`\beta_{\rm c} = 1/2` on a gamma = 1, alpha = 1 parent the inversion is a
    derivative whose terms are both of order 1/u: the rows carry the parent's
    accuracy times 1/(4u) toward the center. Above an anchor the rows are the
    closed form :math:`F = A \sum_{k \le 4} a_k u^k`,
    :math:`A = 1/(2\pi^2)`, exact in the truncation through the parent's
    central deficit to its cubic term, :math:`\Delta_0` to :math:`\Delta_3`
    of `deficit` (`critical_band_series`). The anchor is analytic: the
    smaller of the series' own range at the goal tier and the grid's top
    zone, ``extent`` :math:`P_0`; the range reads the first two neglected
    coefficients and the deficit's own truncation, the move of the series
    under its next term :math:`\Delta_4` (zero where `deficit` holds only
    the linear pair, which then reads as linear). The
    rows validate it: the median signed deviation of the computed rows from
    the closed form over the outermost decade inside the anchor must be
    within the goal tier, on at least 10 rows. Where it is not, the anchor
    steps inward one decade at a time to the outermost decade that is;
    where none is, no band is applied and the record says so. The
    record's ``u`` and ``K`` are the anchor, its ``validation_*`` that
    decade's. A finite `r_a` is COM Type I's (`critical_band_series`).
    """
    from ..validation import GOAL

    d_k = [float(v) for v in deficit] + [0.0] * max(0, 5 - len(deficit))
    d_0, d_1 = d_k[0], d_k[1]
    big_a = 1.0 / (2.0 * np.pi**2)

    def series_with(higher: tuple[float, ...]) -> np.ndarray:
        return critical_band_series(
            float(unit_profile.rho_norm),
            float(unit_profile.rho_beta),
            d_0,
            d_1,
            order=6,
            r_a=r_a,
            higher=higher,
        )

    ak = series_with((d_k[2], d_k[3]))
    # the deficit's own truncation: the series' move under its next term
    d_ak = series_with((d_k[2], d_k[3], d_k[4])) - ak
    a0, c1, c2 = float(ak[0]), float(ak[1] / ak[0]), float(ak[2] / ak[0])

    def band(uu: Any) -> Any:
        return big_a * (ak[0] + uu * (ak[1] + uu * (ak[2] + uu * (ak[3] + uu * ak[4]))))

    def rem(uu: float) -> float:
        trunc = abs(ak[5]) * uu**5 + abs(ak[6]) * uu**6
        return float(
            (trunc + abs(np.polynomial.polynomial.polyval(uu, d_ak))) / abs(ak[0])
        )

    cap = float(extent) * float(p_0)
    if rem(cap) <= GOAL:
        u_series, u_a = float("inf"), cap
    else:
        lo, hi = 0.0, cap
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if rem(mid) <= GOAL:
                lo = mid
            else:
                hi = mid
        u_series = u_a = lo
    kk = np.asarray(k, dtype=np.float64)
    ff = np.asarray(f, dtype=np.float64)
    uu = float(p_0) - kk
    # the outermost decade on which the rows meet the closed form at the goal,
    # stepping inward from the series' anchor one decade at a time
    while True:
        val = (uu <= u_a) & (uu >= 0.1 * u_a) & np.isfinite(ff) & (ff > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = ff[val] / band(uu[val]) - 1.0
        med = float(np.median(sd)) if sd.size else float("nan")
        if sd.size < 10 or abs(med) <= GOAL:
            break
        u_a = 0.1 * u_a
    inside = (uu > 0.0) & (uu <= u_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(ff[inside] / band(uu[inside]) - 1.0)
    rec: dict[str, Any] = dict(
        u=float(u_a),
        K=float(p_0 - u_a),
        u_series=float(u_series),
        A=float(big_a),
        Ftop=float(big_a * a0),
        c1=float(c1),
        c2=float(c2),
        a3=float(ak[3]),
        a4=float(ak[4]),
        D0=d_0,
        D1=d_1,
        D2=d_k[2],
        D3=d_k[3],
        D4=d_k[4],
        validation_rows=int(sd.size),
        validation_median=med,
        validation_scatter=(
            float(np.median(np.abs(sd - med))) if sd.size else float("nan")
        ),
    )
    if not (sd.size >= 10 and abs(med) <= GOAL):
        return ff, dict(applied=False, rows=0, **rec)
    out = ff.copy()
    out[inside] = band(uu[inside])
    return out, dict(
        applied=True,
        rows=int(inside.sum()),
        max_raw_dev=(float(np.nanmax(dev)) if dev.size else 0.0),
        **rec,
    )


# the parents whose DF integral is a known density: an isotropic parent's is
# its density, an OM Type I parent's its augmented density (1 + R^2 / r_a^2)
# rho_hat; a constant-beta or COM parent's F = L^(-2 beta_c) f(Q)
# needs its own identity
_PARENTS = ("isotropic", "OM_I")
# the highest order of the sources (the derivatives of the factor are taken
# to the third)
_MAX_ORDER = 3


def refusals(
    parent_rows: dict[str, Any],
    parent_anisotropy: Any,
    r_t: float,
    anisotropy: Any,
    gamma: float | None = None,
) -> None:
    r"""Refuse a component on a tabulated pair that has no DF here, by name.

    The central validity is checked here, before any stage runs, as
    `truncate` checks it for a Method I model: the profile's inner slope
    ``gamma`` (its source's) below :math:`2\beta_{\rm c}(0)`, with
    :math:`\beta_{\rm c}(0) = 0` for the Osipkov-Merritt models.

    Raises
    ------
    UnphysicalModelError
        CENTRAL VALIDITY: :math:`\gamma < 2\beta_{\rm c}(0)`; the DF is negative at the
        center (An & Evans 2006).
    InputError
        If the parent's anisotropy is neither isotropic nor Osipkov-Merritt
        Type I; if the child needs sources of an order above 3; if a Type II
        child's :math:`R_{\rm a}` does not lie beyond the parent's edge.
    """
    beta_0 = 0.0 if anisotropy.family.startswith("OM") else float(anisotropy.beta_c)
    if gamma is not None and gamma < 2.0 * beta_0 - 1e-12:
        raise UnphysicalModelError(
            f"CENTRAL VALIDITY: gamma = {gamma:.2f} < 2 beta_c(0) = "
            f"{2.0 * beta_0:.2f}: "
            "the DF is negative at the center (An & Evans 2006); no truncation "
            "of the parent removes this. Use a steeper cusp or a smaller central "
            "beta_c."
        )
    family = parent_anisotropy.family
    if family not in _PARENTS:
        raise InputError(
            f"The parent's anisotropy is {family}: a component on a tabulated pair "
            "takes an isotropic or Osipkov-Merritt Type I parent, whose DF "
            "integrates to a known density."
        )
    _, bc, _, order, _, _ = _pair(anisotropy)
    if order > _MAX_ORDER:
        raise InputError(
            f"beta_c = {float(anisotropy.beta_c):+.4f} needs sources of order {order}; "
            f"orders up to {_MAX_ORDER} are supported, which covers beta_c >= -1 "
            "and beta_c = -3/2."
        )
    r_a = float(anisotropy.r_a)
    if not np.isfinite(r_a):
        return
    if anisotropy.kind == "II" and not float(r_t) < r_a:
        raise InputError(
            f"A Type II component needs its parent's edge inside r_a: R_t = "
            f"{float(r_t):.4e} is not inside r_a = {r_a:g}."
        )


def _parent_inputs(profile: Any) -> tuple[dict[str, Any], dict[str, Any], float | None]:
    """The parent's DF and rows as the sources read them, from the profile.

    F' is rebuilt from the table's slope, F' = s ln 10 F / K (the run's own
    column is not kept: a rounding against it), and the rows' mass is 4 pi
    times the profile's.
    """
    if np.size(profile.df_k) == 0:
        raise InputError(
            "The truncated profile carries no DF of its model: a component on it "
            "needs that DF (a profile made by `truncate` carries it)."
        )
    k = np.asarray(profile.df_k, dtype=np.float64)[0]
    log_f = np.asarray(profile.df_log_f, dtype=np.float64)[0]
    slope = np.asarray(profile.df_slope, dtype=np.float64)[0]
    with np.errstate(all="ignore"):
        d_f = slope * np.log(10.0) * 10.0**log_f / k
    parent_df = dict(
        K=k,
        logF=log_f,
        dFdK=d_f,
        taylor=tuple(profile.df_taylor[0]) if profile.df_taylor else None,
    )
    parent_rows = dict(
        R=np.asarray(profile.r, dtype=np.float64),
        Psi=np.asarray(profile.p_trunc, dtype=np.float64),
        rho=np.asarray(profile.rho_trunc, dtype=np.float64)[0],
        M=4.0 * np.pi * np.asarray(profile.mass_trunc, dtype=np.float64),
        u=np.asarray(profile.u_trunc, dtype=np.float64),
        p_0=1.0 - float(profile.k_t),
    )
    parent = profile.source_anisotropy
    r_a = float(parent.r_a) if parent.family == "OM_I" else None
    return parent_df, parent_rows, r_a


def _table(k: np.ndarray, f: np.ndarray) -> DfTable:
    """The child's table: log10 |F| on every row, signed where F is
    negative; no slope column (the inversion of a tabulated pair makes no
    dF/dK)."""
    with np.errstate(divide="ignore"):
        log_df = np.log10(np.abs(f))
    finite = np.flatnonzero(np.isfinite(log_df))
    extra: dict[str, Any] = {}
    if np.any(f < 0.0):
        extra["sign"] = np.sign(f).astype(np.int8)[None, :]
    return DfTable(
        k=k[None, :],
        log_df=log_df[None, :],
        log_df_slope=np.full(k.shape, np.nan)[None, :],
        i_first=(int(finite[0]),),
        i_last=(int(finite[-1]),),
        slopes_exact=(False,),
        **extra,
    )


def tabulated_df(
    component: Any, settings: Settings
) -> tuple[
    EnergyGrid,
    DfTable,
    AsymptoticForm,
    float,
    ValidationReport,
    InversionRecord,
    InversionRows,
]:
    """Return the DF of a component on a tabulated pair: the truncated
    profile of a model of one component, with its own anisotropy.

    The stages of the analytic inversion on the tabulated pair: the grid on
    the profile's first potential (around the pilot's feature where the
    family has one), the sources from the integral of the profile's model's
    DF, the kernel's rows with the exact tail below the floor, the critical
    branch's closed-form band where it applies, the table, the forms (the
    center's of the model, the edge law below the floor) and the report.
    The DF is compact: its support ends at the profile's edge.

    Returns
    -------
    grid, table, form, k_t_min, report, inversion, rows
        As `invert.inversion_df` returns them.

    Raises
    ------
    InputError
        Where the component has no DF on this pair (`refusals`), or the
        profile carries no DF of its model.
    ResultError
        Where a band of the rows at the top is defective: the repair of an
        untruncated table is not made on a tabulated pair.
    """
    profile, aniso = component.profile, component.anisotropy
    unit = profile.source
    with timed("potential_density_pair"):
        parent_df, parent_rows, parent_r_a = _parent_inputs(profile)
        # a truncation with no edge is not a truncated model: its rows end at
        # the solve's outer radius above the floor, and the child's grid would
        # end there with nothing below it
        if float(profile.k_t) > 0.0 and not profile.spatially_finite:
            raise UnphysicalModelError(
                "PARENT NOT SPATIALLY FINITE: the parent's truncation at K_t = "
                f"{float(profile.k_t)!r} has no edge; its rows end at R = "
                f"{float(profile.r[-1]):.6e} with Phi = "
                f"{float(profile.p_trunc[-1]):.6e}, above the floor, so it is not a "
                "truncated model and no component is built on it."
            )
        refusals(
            parent_rows,
            profile.source_anisotropy,
            float(profile.r_t),
            aniso,
            gamma=float(unit.rho_gamma),
        )
    with timed("energy_grid"):
        feature = None
        if aniso.family in ("OM_II", "COM_I", "COM_II"):
            feature = pilot_feature(
                parent_df, parent_rows, parent_r_a, aniso, settings, unit
            )
        q, zones = tabulated_grid(parent_rows, aniso, settings, unit, feature)
    with timed("invert_rows"):
        made = child_rows(q, parent_df, parent_rows, parent_r_a, aniso, settings)
    with timed("asymptotic_form"):
        f_all = np.asarray(made["f"], dtype=np.float64)
        if not np.all(np.isfinite(f_all)):
            bad = ~np.isfinite(f_all)
            err = AccuracyError(
                f"{int(bad.sum())} rows of the child's inversion are not finite, "
                f"the first at K = {float(q[bad][0]):.6e}: refused"
            )
            err.result = made
            raise err
        k, f = q, f_all
        p0_hat = float(parent_rows["p_0"])

        # the critical branch's closed-form band
        critical: dict[str, Any] = {}
        deficit = profile.central_deficit
        # COM Type I's center is on the same branch, beta(0) = beta_c
        if (
            aniso.family in ("constant", "COM_I")
            and order_pair(float(aniso.beta_c)) == order_pair(0.5)
            and float(unit.rho_gamma) == 1.0
            and float(unit.rho_alpha) == 1.0
            and deficit
        ):
            extent = float(settings.inversion["grid_tail_extent"])
            f_raw = f
            com = aniso.family == "COM_I"
            f, rec = critical_band(
                k,
                f_raw,
                p0_hat,
                extent,
                unit,
                deficit[0],
                r_a=float(aniso.r_a) if com else math.inf,
            )
            applied = rec.pop("applied")
            rows_in = rec.pop("rows")
            critical = dict(
                critical_band=(
                    "closed-form series band (beta_c = 1/2, gamma = 1, alpha = 1"
                    + (", COM Type I" if com else "")
                    + "), analytic anchor"
                    if applied
                    else "NOT APPLIED: validation failed, the rows are as computed"
                ),
                critical_band_rows=rows_in,
                **{f"critical_band_{key}": value for key, value in rec.items()},
            )

        # the top: only a defective band can call for the center's form; the
        # critical band, made first, is the closed form where it applies
        center = center_form(unit, aniso)
        kappa = float(center["kappa_sing"]) if center["branch"] in _PATCHED else 0.0
        at = anchor(k, f, p0_hat, kappa, center, math.nan, None, below_ceiling=False)
        if at is not None:
            raise ResultError(
                "The rows of a component on a tabulated pair carry a defective band "
                f"at the top (from K = {at.k_anchor!r}); the repair of an untruncated "
                "table is not made here."
            )

        # the bands: no top band (no repair); the boundary term at the floor is
        # the sources' own value there, so the floor is carried exactly
        floor_exact = True
        floor_k = 0.0 if floor_exact else 100.0 * K_FLOOR
        fin = np.isfinite(f)
        bands = dict(
            patch_anchor_K=math.inf,
            floor_band_K=floor_k,
            floor_term_exact=floor_exact,
            top_band_raw_negatives=0,
            floor_band_negatives=int(np.sum(fin & (k <= floor_k) & (f < 0.0))),
            interior_negatives=int(np.sum(fin & (k > floor_k) & (f < 0.0))),
        )

        table = _table(k, f)
        law = made["floor"].get("law")
        form = AsymptoticForm(
            p_0=p0_hat,
            central_label=(str(center["branch"]),),
            central_exponent=(float(center["kappa_sing"]),),
            central_amplitude=(float(center["amplitude"]),),
            central_form=(str(center["form"]),),
            central_log_coefficient=(float(center.get("log_coef", math.nan)),),
            central_next_form=(str(center["next_form"]),),
            central_next_exponent=(float(center["next_kappa"]),),
            central_next_coefficient=(float(center["next_coef"]),),
            central_anchor=(math.nan,),
            central_anchor_amplitude=(math.nan,),
            central_regular=((math.nan, math.nan),),
            below_floor=("edge_law",) if law else ("zero",),
            below_floor_coefficients=(
                (float(law["p"]), float(law["C0"]), float(law["C1"])),
            )
            if law
            else ((),),
            tail_exponent=(0.0,),
            tail_amplitude=(0.0,),
            tail_log_exponent=(0.0,),
            tail_log_scale=(1.0,),
        )
    with timed("validate_df"):
        k_t_min, report, _ = validate_df(
            potential_density_pair((component,)), table, form
        )
    # the table has no slope column: its slope check is not judged, and says so
    report = slope_column_not_judged(report, table, NO_SLOPE_COLUMN_CHILD)

    # r(K) from the profile's rows, and the record
    order = np.argsort(parent_rows["Psi"])
    pos = parent_rows["Psi"][order] > 0.0
    r_of_k = np.exp(
        np.interp(
            np.log(k),
            np.log(parent_rows["Psi"][order][pos]),
            np.log(parent_rows["R"][order][pos]),
            left=np.nan,
            right=np.nan,
        )
    )
    grid = EnergyGrid(
        k=k[None, :],
        r_of_k=r_of_k[None, :],
        k_floor=(float(k[0]),),
        k_ceiling=(float(k[-1]),),
    )
    analytic = potential_density_pair((Component(unit, aniso),))
    _, (frec,) = inversion_form(analytic, grid)
    inversion = InversionRecord(
        form=(str(frec.form),),
        order_j=(int(frec.order_j),),
        order_n=(float(frec.kernel_exponent),),
        floor_m=(int(frec.order_m),),
        floor_n=(float(frec.order_n),),
        # the documented conventions: NaN where there is no interior point,
        # (P_0, 0) and (0, R_outer) where no row was replaced
        k_interior=(math.nan,),
        interior_exponent=(math.nan,),
        outer_power=(float(frec.outer_power),),
        outer_end=(str(frec.outer_end),),
        high_k_override_k=(p0_hat,),
        high_k_override_r=(0.0,),
        high_k_override_rows=(0,),
        high_k_override_rule=("method2_bands",),
        low_k_override_k=(0.0,),
        low_k_override_r=(float(profile.r_t),),
        method2_bands=(bands,),
        critical_band=(critical,) if critical else (),
    )
    rows = InversionRows(
        k=np.asarray(q, dtype=np.float64)[None, :],
        f=f_all[None, :],
        ddf=np.full((1, q.size), np.nan),
        data_top=(float(parent_rows["Psi"][0]),),
    )
    return grid, table, form, k_t_min, report, inversion, rows


# the checks of a model on a tabulated pair: the table's top form, the edge
# law's join to the table at its floor, and the profile's own checks
_MODEL_KEYS = ("top_form", "edge_join", *PROFILE_CHECK_KEYS)


def child_model(df: Any) -> Any:
    """Return the model of a DF on a tabulated pair: at the edge of the
    profile it was inverted on, the truncation of that profile's model.

    The model's profile is the rows it was inverted on, with the parent's
    truncation energy, the model's own anisotropy and DF; the rows carry no
    central deficit of their own (the deficit is the parent's).

    Raises
    ------
    UnphysicalModelError
        Where the DF is negative inside its support: the anisotropy cannot be
        supported by this truncated density. The exception carries `df`.
    AccuracyError
        Where one of the model's checks lies beyond the hard limit, unless the
        settings ask for the model anyway (``write_anyway``).
    """
    from .core import model_identity
    from .results import Model

    component = df.components[0]
    parent = component.profile
    table, form = df.table, df.form
    if np.size(table.sign) and np.any(np.asarray(table.sign) < 0):
        f = (
            np.asarray(table.sign, dtype=np.float64)[0]
            * 10.0 ** np.asarray(table.log_df, dtype=np.float64)[0]
        )
        k = np.asarray(table.k, dtype=np.float64)[0]
        neg = np.flatnonzero(f < 0.0)
        raise UnphysicalModelError(
            f"The DF is negative on {neg.size} energies inside its support "
            f"(F_min = {float(f[neg].min()):.3e}, highest K with F < 0 = "
            f"{float(k[neg[-1]]):.4e}): "
            "the anisotropy cannot be supported by this truncated density.",
            result=df,
        )
    identity = model_identity(df.components, None, df.settings)
    profile = dataclasses.replace(
        parent,
        source_anisotropy=component.anisotropy,
        model=identity,
        central_deficit=(),
        df_k=np.asarray(table.k, dtype=np.float64),
        df_log_f=np.asarray(table.log_df, dtype=np.float64),
        df_slope=np.asarray(table.log_df_slope, dtype=np.float64),
        df_taylor=(),
    )
    lo, hi = int(table.i_first[0]), int(table.i_last[0]) + 1
    top, _, _, law = stated_forms(form, 0)
    measured = measure_table(
        np.asarray(table.k, dtype=np.float64)[0, lo:hi],
        np.asarray(table.log_df, dtype=np.float64)[0, lo:hi],
        None,
        float(form.p_0),
        top,
        None,
        None,
        None,
        law,
    )
    checks = {
        "top_form": Check(abs(measured.kappa_residual), measured.k_top),
        "edge_join": Check(abs(measured.edge_rel_at_floor)),
        **profile_checks(
            profile, table, form, component.anisotropy, df.settings.truncation
        ),
    }
    # the parent's tier is the DF's, as an untruncated model's is
    report = ValidationReport(
        _MODEL_KEYS, {key: checks[key] for key in _MODEL_KEYS}, df.validation.level
    )
    # the model's own checks, judged as truncate judges its own: refused beyond
    # the hard limit, unless the settings ask for the model anyway, in which
    # case the report carries the verdict and the one channel says so
    beyond = "; ".join(
        f"the {key} misses it by {check.error:.3e}"
        + ("" if check.where is None else f" at {check.where:.6g}")
        for key, check in report.checks.items()
        if check.level >= 3
    )
    model = Model(
        components=df.components,
        truncation=None,
        settings=df.settings,
        identity=identity,
        parent=df.identity,
        table=table,
        form=form,
        profile=dataclasses.replace(profile, model_level=report.level),
        validation=report,
        timing={},
        r_trust={},
        inversion=df.inversion,
    )
    if beyond and not df.settings.write_anyway:
        raise AccuracyError(
            f"The truncated model's DF does not meet its own checks: {beyond}, "
            f"beyond the hard limit {HARD_LIMIT:g}.",
            result=model,
        )
    announce(
        f"The truncated model of {component.name!r}",
        report,
        "; ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at {check.where:.6g}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        + (
            f" (returned on request, write_anyway: beyond the hard limit "
            f"{HARD_LIMIT:g})"
            if beyond
            else ""
        ),
    )
    return model
