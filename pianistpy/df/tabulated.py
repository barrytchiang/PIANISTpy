r"""The distribution function (DF) of a component on a tabulated pair (Method II).

`compute_df` calls `tabulated_df` when a component's profile is a
`TruncatedProfile`, and `truncate` without a truncation calls
`child_model` on the DF that `tabulated_df` produced. The tabulated pair
is the truncated profile of one model, the parent. The component, the
child, carries its own anisotropy and is inverted on the parent's
potential and density: Method II of Chiang+(2026d), Appendix B2.2.

The child's auxiliary density is
:math:`\varrho_{\rm aux}(P) = g(R)\,\hat\varrho(P)`, with :math:`P` the
parent's relative potential, :math:`\hat\varrho` the parent's density,
:math:`R(P)` the radius at which the parent's potential equals :math:`P`,
and :math:`g` the child's anisotropy factor. The parent's density is the
integral of the parent's DF :math:`F` over the energy :math:`K`,

.. math::

    \hat\varrho(P) = 4\pi\sqrt{2}\int_0^P F(K)\,(P - K)^{1/2}\,dK.

The derivatives of :math:`\hat\varrho` with respect to :math:`P` are
therefore integrals of :math:`F` and of the derivatives of :math:`F`
against :math:`(P - K)^{-1/2}` (`parent_density`). The derivatives of
:math:`g` with respect to :math:`P` follow by the chain rule from
:math:`R(P)`, which is interpolated in the parent's tabulated profile
(`factor_sources`). The child's DF is checked as a truncated model's DF
is: a check above :math:`10^{-2}` raises `AccuracyError` unless the
settings set ``write_anyway=True``.

References
----------
Notes Section T:2.6.
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
from ..validation import Check, ValidationReport, announce
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

# The prefactors of the parent's density integral and of its first P
# derivative: rho_hat(P) = _A * INT_0^P F(K) (P - K)^(1/2) dK, and
# rho_hat'(P) = _PREF * INT_0^P F(K) (P - K)^(-1/2) dK, with _PREF = _A / 2.
_A = 4.0 * np.pi * np.sqrt(2.0)
_PREF = 2.0 * np.pi * np.sqrt(2.0)
# The number of Gauss-Legendre nodes of each DF integral in `parent_density`.
# The number is fixed by a convergence test in the node count.
_N_GAUSS = 600
# The lower bound on a divisor and on the argument of a logarithm, so that a
# zero divisor or a zero argument gives a finite result.
_TINY = 1e-300
# The central branches for which `tabulated_df` passes the center's singular
# exponent to `anchor`; for every other branch, `tabulated_df` passes 0. The
# tuple equals PATCHED_BRANCHES of `central.py`: the branches whose closed
# form an untruncated inversion carries above the table's last row.
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
    r"""Return the child's auxiliary density and its P derivatives on energies `q`.

    The function computes the parent's part with `parent_density` and
    applies the child's anisotropy factor with `factor_sources`.

    Parameters
    ----------
    q : array, shape (n,)
        The child's energy grid, in internal units.
    p_top : float
        The parent's potential at the parent's first tabulated radius, in
        internal units. With a parent's Taylor edge, the sources are
        computed on the nodes with :math:`K_{\min} \le q < p_{\rm top}`;
        without a Taylor edge, on the nodes with
        :math:`K_{\min} < q < p_{\rm top}`. :math:`K_{\min}` is the lowest
        positive energy with a finite DF in the parent's DF table.
    order : int
        The highest P derivative of the auxiliary density computed, from 1
        to 3.
    parent_df : dict
        The parent's DF table: ``K`` (the energies), ``logF`` (the base-10
        logarithm of the DF), ``dFdK`` (the derivative of the DF with
        respect to K) and ``taylor``. ``taylor`` is the parent's Taylor edge
        ``(F_1, F_2, K_switch)``, below whose ``K_switch`` the parent's DF
        is ``F_1 K + F_2 K^2 / 2``, or None for a parent without the edge.
    parent_rows : dict
        The parent's tabulated profile: ``R`` (the radii), ``Psi`` (the
        potential), ``rho`` (the density), ``M`` (4 pi times the enclosed
        mass), ``u`` (the depth ``p_0 - Psi``) and ``p_0`` (the parent's
        central potential, ``1 - K_t`` with ``K_t`` the parent's truncation
        energy).
    parent_r_a : float or None
        The parent's anisotropy radius :math:`R_{\rm a}` for an
        Osipkov-Merritt Type I parent; None for an isotropic parent.
    family : str
        The child's anisotropy family.
    beta_c : float
        The child's central anisotropy :math:`\beta_{\rm c}`.
    r_a : float
        The child's anisotropy radius :math:`R_{\rm a}`, in internal units.

    Returns
    -------
    sources : dict
        ``window``: boolean array, True on the nodes of `q` where the
        sources are computed. ``rho``: the parent's density on `q`.
        ``aux``: a dict from each order k, 1 to `order`, to the k-th P
        derivative of the auxiliary density. ``g``: the child's anisotropy
        factor. ``g_p``: the P derivative of ``g``. ``r`` and ``r_p``: the
        radius R(P) and its P derivative. ``h`` and ``h_1``: an
        Osipkov-Merritt Type I parent's factor ``1 + R^2 / R_a^2`` and the
        factor's P derivative, None for an isotropic parent. ``p0_hat``: the
        parent's central potential. ``taylor``: the parent's Taylor edge or
        None.

    Raises
    ------
    InputError
        If `order` is not 1, 2 or 3.
    InputError
        If `order` is 3 and the parent's DF table carries no Taylor edge.
    InputError
        If the parent's ``u`` column and ``Psi`` column differ in length.
    InputError
        If the parent's ``u`` differs from ``p_0 - Psi`` by more than four
        times the double-precision machine epsilon times ``p_0``.
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
    r"""Return the part of `parent_sources` that depends on the parent alone.

    The part holds the P derivatives of the parent's density, the radius
    R(P) with the radius's P derivatives, and the window of nodes where the
    sources are computed. For an Osipkov-Merritt Type I parent, the density
    derivatives are those of the parent's density itself, not of the
    parent's augmented density.

    The parameters of `parent_density` and the errors that `parent_density`
    raises are the parameters and the errors of `parent_sources`.

    Returns
    -------
    density : dict
        ``order``: `order`. ``window``: as in `parent_sources`. ``rho``: the
        tuple of the parent's density and its first, second and third P
        derivatives on `q`; the third derivative is None unless `order` is
        3. ``r``,
        ``r_p`` and ``r_pp``: R(P) and its first and second P derivatives.
        ``m`` and ``m_p``: the column ``M`` of `parent_rows` interpolated at
        R(P), and its P derivative. ``h``, ``h_1``, ``p0_hat`` and
        ``taylor``: as in `parent_sources`.
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

    # The parent's DF is interpolated in the logit x = ln(K / (p_0 - K)),
    # which stretches both ends of the energy range, K near 0 and K near p_0.
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
        # The integral INT_0^P fun(K) (P - K)^(-1/2) dK, substituted as
        # p_0 - K = (p_0 - P) exp(v^2) with Gauss-Legendre nodes in v. The
        # substitution removes the singularity of (P - K)^(-1/2) at K = P, and
        # the substituted nodes are spaced logarithmically in the depth
        # p_0 - K, which resolves the growth of the DF toward K = p_0.
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
        # Below K_switch the parent's DF is the Taylor form F_1 K + F_2 K^2 / 2,
        # so the first, second and third P derivatives of rho_hat have closed
        # forms on the nodes below K_switch.
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

    # R(P), rho_hat(P) and M(P) are interpolated in the parent's tabulated
    # profile against the logit lambda = ln(Psi / u), with u = p_0 - Psi.
    # Poisson's equation gives dR/dP = -4 pi R^2 / M, with M the column of
    # 4 pi times the enclosed mass.
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
        # For an Osipkov-Merritt Type I parent, the DF integrals give the P
        # derivatives of the augmented density h rho_hat, with
        # h = 1 + R^2 / r_a^2. The P derivatives of rho_hat follow from the
        # Leibniz rule with the P derivatives of h.
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
    r"""Return `parent_sources` from the parent's part and the child's factor.

    The child's auxiliary density is ``g(R) rho_hat(P)``, with ``g`` the
    child's anisotropy factor of `inverse_factor`. The P derivatives of the
    auxiliary density follow from the Leibniz rule, with the P derivatives
    of ``g`` from the chain rule on R(P). A given `factor` replaces the
    factor that `inverse_factor` computes from `family`, `beta_c` and
    `r_a`.

    Parameters
    ----------
    density : dict
        The parent's part, as `parent_density` returns it.
    family : str
        The child's anisotropy family.
    beta_c : float
        The child's central anisotropy :math:`\beta_{\rm c}`.
    r_a : float
        The child's anisotropy radius :math:`R_{\rm a}`, in internal units.
    factor : tuple of array, optional
        The factor ``g`` and its first, second and, for order 3, third R
        derivatives on the radii ``density["r"]``. Default: ``None``, which
        means the factor of `inverse_factor` for `family`, `beta_c` and
        `r_a`.

    Returns
    -------
    sources : dict
        The sources, with the keys `parent_sources` describes.
    """
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
    r"""Return the sources at the lowest node of the window, the floor node.

    The inversion's boundary term reads the auxiliary density's derivatives
    at the floor node. When the parent carries a Taylor edge, the record
    also holds the child's edge expansion at the floor node
    (`edge_expansion_at_node`, given the child's factor, the factor's P
    derivative and, for an Osipkov-Merritt Type I parent, the parent's
    ``h`` and ``h_1``) and the edge law of the child's DF that the
    expansion inverts to (`edge_law`).

    Parameters
    ----------
    q : array, shape (n,)
        The child's energy grid, in internal units.
    sources : dict
        The sources on `q`, as `parent_sources` returns them.
    anisotropy : Anisotropy
        The child's anisotropy.

    Returns
    -------
    floor : dict
        ``P``: the floor node's energy. ``aux``: a dict from each order k,
        0 to the sources' highest order, to the k-th P derivative of the
        auxiliary density at the floor node. With a Taylor edge, also
        ``a_q`` (the coefficient of each power of the edge expansion) and
        ``law`` (the edge law's exponent ``p`` and coefficients ``C0`` and
        ``C1``).
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
    r"""Return the child's order pair and the inversion's quantities from the pair.

    An Osipkov-Merritt child is inverted with the isotropic order pair,
    :math:`\beta_{\rm c} = 0`.

    Returns
    -------
    pair : OrderPair
        The order pair of `order_pair`.
    beta_c : float
        The :math:`\beta_{\rm c}` the kernel takes: 0 for an
        Osipkov-Merritt child, the child's :math:`\beta_{\rm c}` otherwise.
    j : int
        The pair's ``order_j``.
    order : int
        The highest P derivative of the auxiliary density the inversion
        reads: ``j`` when the pair's kernel exponent is zero, ``j + 1``
        otherwise.
    half : bool
        True when the pair's kernel exponent is zero, which holds at a
        half-integer :math:`\beta_{\rm c}`.
    switched : bool
        True when the pair's ``order_j`` equals ``order_m + 1``.
    """
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
    r"""Return the child's inversion rows on energies `q`.

    The inversion kernel reads the sources of `parent_sources` on every node
    of `q`. Without `unit_profile`, the boundary term at the floor node is
    the sources' value at the floor node (`floor_record`). When the parent
    carries a Taylor edge, the exact tail of the DF below the floor node
    follows from the child's edge expansion (`method2_floor_tail`).

    Parameters
    ----------
    q : array, shape (n,)
        The child's energy grid, in internal units.
    parent_df : dict
        The parent's DF table, as in `parent_sources`.
    parent_rows : dict
        The parent's tabulated profile, as in `parent_sources`.
    parent_r_a : float or None
        The parent's anisotropy radius, as in `parent_sources`.
    anisotropy : Anisotropy
        The child's anisotropy, which sets the kernel and the sources' order.
    settings : Settings
        The numerical settings; the kernel's quadrature rules and the
        number of floor nodes are read from the inversion group.
    unit_profile : DoublePowerLaw, optional
        The parent's untruncated analytic profile. Default: ``None``, which
        means the child's own rows, with the boundary terms read from the
        sources at the floor node and no completion below the floor node
        from `floor_source`. When given, the rows are the
        pilot rows of `pilot_feature`: the boundary terms at the floor node,
        the energy ``outer_from`` above which the rows take the general
        form, and the completion below the floor node are taken from the
        untruncated analytic pair, as for an untruncated component.
    sources : dict, optional
        Sources on `q`, as `parent_sources` returns them. Default: ``None``,
        which means the sources computed from the parent's DF table and
        tabulated profile.

    Returns
    -------
    rows : dict
        ``f``: the completed rows of the child's DF on `q`. ``f_kernel``:
        the kernel's rows before the completion. ``floor``: the record of
        `floor_record`. ``tail``: the input of the exact tail below the
        floor node, or None. ``sources``: the sources used. ``dm_j``,
        ``dm_jm1`` and ``outer_from``: the kernel's boundary inputs.

    Raises
    ------
    InputError
        If a node of `q` lies outside the window where the sources are
        computed.
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
        # For the pilot rows, the boundary terms at the floor node, the energy
        # outer_from from which the rows take the general form and the
        # completion below the floor node are taken from the untruncated
        # analytic pair of unit_profile, as the inversion of an untruncated
        # component takes them.
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
    r"""Return the energy around which an OM Type II or COM child's grid is placed.

    The pilot rows are the child's rows of `child_rows` with `unit_profile`,
    on a coarse grid scaled to the parent's potential :math:`P_0` at the
    parent's first tabulated radius. The feature is the minimum of the
    pilot DF below a cutoff energy, refined by the vertex of the parabola
    through the minimum's node and its two neighboring nodes. Only nodes
    whose pilot DF exceeds :math:`10^{-3}` times the median of the positive
    pilot DF values below the cutoff enter the minimum. The cutoff is not
    scaled by :math:`P_0`: the cutoff is the midpoint between :math:`5/8`
    and ``1 - grid_tail_extent``, the boundaries of the standard zones B
    and BC and of the standard zones BC and C for :math:`P_0 = 1`. The rule
    follows the rule of `invert.pilot_feature` for an untruncated
    component.

    Parameters
    ----------
    parent_df : dict
        The parent's DF table, as in `parent_sources`.
    parent_rows : dict
        The parent's tabulated profile, as in `parent_sources`.
    parent_r_a : float or None
        The parent's anisotropy radius, as in `parent_sources`.
    anisotropy : Anisotropy
        The child's anisotropy.
    settings : Settings
        The numerical settings; ``grid_tail_extent`` is read from the
        inversion group.
    unit_profile : DoublePowerLaw
        The parent's untruncated analytic profile.

    Returns
    -------
    feature : dict
        ``q_feature``: the feature's energy, or 0 where fewer than three
        nodes below the cutoff enter the minimum, or where the
        feature lies below ``1000 K_FLOOR`` or at or above the cutoff. A
        zero ``q_feature`` gives the standard zones.
    """
    p0 = float(np.asarray(parent_rows["Psi"], dtype=np.float64)[0])
    extent = float(settings.inversion["grid_tail_extent"])
    low = np.logspace(np.log10(K_FLOOR), np.log10(0.05 * p0), 200)
    # The logarithmic parts of the grid can end one unit in the last place
    # away from the linear part's end nodes. Each seam is set to the linear
    # part's end node exactly, so that `np.unique` merges the two nodes at
    # each seam.
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
    r"""Return the energy grid of a component on a tabulated pair, and its zones.

    The grid is the hybrid grid of `hybrid_grid` on :math:`P_0`, the
    parent's potential at the parent's first tabulated radius. The grid
    starts at the larger of ``K_FLOOR`` and the parent's potential at the
    parent's last tabulated radius, and has no nodes below that start. An
    OM Type I child's zones are placed around :math:`P(R_{\rm a})`, the
    untruncated analytic potential at the child's anisotropy radius. An OM
    Type II or COM child's zones are placed around the energy of
    `pilot_feature`. Every other child takes the standard zones.

    Parameters
    ----------
    parent_rows : dict
        The parent's tabulated profile, as in `parent_sources`.
    anisotropy : Anisotropy
        The child's anisotropy.
    settings : Settings
        The numerical settings; the grid's node counts and
        ``grid_tail_extent`` are read from the inversion group.
    unit_profile : DoublePowerLaw
        The parent's untruncated analytic profile.
    feature : dict, optional
        The record of `pilot_feature`. Default: ``None``, which is valid only
        for a child that is neither OM Type II nor COM.

    Returns
    -------
    q : array
        The energy grid, increasing, in internal units.
    zones : GridZones
        The zones the grid is built on.

    Raises
    ------
    ValueError
        If the child is OM Type II or COM and `feature` is None.
    """
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
    r"""Return the central power series of the auxiliary density's P derivative.

    The series is
    :math:`d\varrho_{\rm aux}/dP = \sum_{k=0}^{\rm order} a_k u^k`, with
    :math:`u = P_0 - P` the depth below the central potential :math:`P_0`.
    The profile has :math:`\gamma = 1` and :math:`\alpha = 1`, density
    :math:`\rho = C R^{-1} (1 + R)^{1 - \beta_r}`, and is truncated in phase
    space. The series serves the critical branch
    :math:`\beta_{\rm c} = 1/2`, where the DF is proportional to
    :math:`d\varrho_{\rm aux}/dP`. The auxiliary density is
    :math:`\varrho_{\rm aux} = R\hat\varrho` for constant anisotropy, and
    :math:`R\hat\varrho\,(1 + R^2/R_{\rm a}^2)^{1/2}` for COM Type I with a
    finite `r_a`.

    The function uses power-series arithmetic in R and then reverts the
    series of :math:`u(R)` to :math:`R(u)`. The truncation enters as the
    central deficit,
    :math:`\hat\varrho(u) = \varrho_{\rm u}(u) - \Delta_0 + \Delta_1 u
    - \Delta_2 u^2 + \dots`, with :math:`\varrho_{\rm u}(u)` the untruncated
    density as a function of depth. The radius of the truncated model at a
    given depth follows from a fixed-point iteration. The COM factor
    :math:`(1 + R^2/R_{\rm a}^2)^{1/2}` enters as its binomial series in
    :math:`R^2/R_{\rm a}^2` and multiplies the auxiliary density only; the
    mass and the depth are the profile's mass and depth. At an infinite `r_a`, the
    COM factor is 1 and the series is the constant-anisotropy series.

    Parameters
    ----------
    c : float
        The density normalization :math:`C`, in internal units.
    beta_r : float
        The profile's outer slope :math:`\beta_r`.
    d_0 : float
        The deficit's constant term :math:`\Delta_0`.
    d_1 : float
        The deficit's linear coefficient :math:`\Delta_1`.
    order : int, optional
        The highest power of :math:`u` returned. Default: ``5``.
    r_a : float, optional
        The COM Type I anisotropy radius :math:`R_{\rm a}`, in internal
        units. Default: ``math.inf``, which means constant anisotropy.
    higher : tuple of float, optional
        The deficit's higher coefficients :math:`\Delta_2, \Delta_3, \dots`.
        Default: ``()``, which means a deficit linear in :math:`u`.

    Returns
    -------
    a : array, shape (order + 1,)
        The coefficients :math:`a_0` to :math:`a_{\rm order}`.
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
    r"""Return the rows with the critical branch's closed-form top band.

    At :math:`\beta_{\rm c} = 1/2` on a parent with :math:`\gamma = 1` and
    :math:`\alpha = 1`, the inversion is a derivative whose two terms both
    grow as :math:`1/u` toward the center, with :math:`u = P_0 - K` the
    depth. The computed rows therefore carry the parent's relative error
    multiplied by :math:`1/(4u)`. Above an anchor depth, the rows are
    replaced by the closed form
    :math:`F = A \sum_{k=0}^{4} a_k u^k` with :math:`A = 1/(2\pi^2)` and
    the coefficients :math:`a_k` of `critical_band_series`. The closed form
    is exact in the parent's central deficit to the cubic term,
    :math:`\Delta_0` to :math:`\Delta_3` of `deficit`.

    The anchor depth is the smaller of two depths. The first depth is the
    series' range: the largest depth at which the series' estimated relative error
    is at most :math:`10^{-6}`. The estimate is the sum of the first two
    neglected terms and of the change of the series when the deficit's
    next coefficient :math:`\Delta_4` is added; :math:`\Delta_4` is zero
    when `deficit` holds fewer than five coefficients. The second depth is
    the grid's top zone, ``extent`` times :math:`P_0`.

    The computed rows then validate the anchor. Over the decade of depth
    from 0.1 times the anchor depth to the anchor depth, the median signed
    relative deviation of the
    computed rows from the closed form must be at most :math:`10^{-6}` in
    magnitude, on at least 10 rows. When the median exceeds
    :math:`10^{-6}`, the anchor moves inward by one decade, until a decade
    passes or a decade holds fewer than 10 rows. When no decade passes, the
    rows are returned unchanged and the record says that no band was
    applied.

    Parameters
    ----------
    k : array, shape (n,)
        The energy grid, in internal units.
    f : array, shape (n,)
        The computed rows of the DF on `k`.
    p_0 : float
        The parent's central potential :math:`P_0`, in internal units.
    extent : float
        The fraction of :math:`P_0` that the grid's top zone spans.
    unit_profile : DoublePowerLaw
        The parent's untruncated analytic profile.
    deficit : tuple of float
        The parent's central deficit coefficients :math:`\Delta_0,
        \Delta_1, \dots`.
    r_a : float, optional
        The anisotropy radius of a COM Type I child, in internal units.
        Default: ``math.inf``, which means a constant-anisotropy child.

    Returns
    -------
    f : array, shape (n,)
        The rows, with the closed form on the nodes above the anchor where
        the band is applied.
    record : dict
        The band's record, with the keys ``applied`` (whether the band is
        applied), ``rows`` (the number of rows replaced), ``u`` and ``K``
        (the anchor's depth and energy),
        ``u_series`` (the series' range, infinite when the range exceeds the
        top zone), ``A``, ``Ftop`` (the closed form at :math:`u = 0`),
        ``c1`` and ``c2`` (:math:`a_1/a_0` and :math:`a_2/a_0`), ``a3``,
        ``a4``, ``D0`` to ``D4`` (the deficit coefficients used),
        ``validation_rows``, ``validation_median`` and
        ``validation_scatter`` (the validating decade's row count, median
        signed deviation, and median absolute deviation from the median
        signed deviation),
        and, when the band is applied, ``max_raw_dev`` (the largest
        relative deviation of a replaced row from the closed form).
    """
    from ..validation import TIER_BOUNDS

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
    # The error of truncating the deficit at Delta_3 is estimated by the change
    # of the series when Delta_4 is added.
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
    if rem(cap) <= TIER_BOUNDS[0]:
        u_series, u_a = float("inf"), cap
    else:
        lo, hi = 0.0, cap
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if rem(mid) <= TIER_BOUNDS[0]:
                lo = mid
            else:
                hi = mid
        u_series = u_a = lo
    kk = np.asarray(k, dtype=np.float64)
    ff = np.asarray(f, dtype=np.float64)
    uu = float(p_0) - kk
    # The anchor moves inward from the series' anchor one decade of depth at a
    # time, until the median signed deviation of the rows from the closed form
    # over depths from 0.1 times the anchor depth to the anchor depth is at
    # most 1e-6, the bound TIER_BOUNDS[0].
    while True:
        val = (uu <= u_a) & (uu >= 0.1 * u_a) & np.isfinite(ff) & (ff > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = ff[val] / band(uu[val]) - 1.0
        med = float(np.median(sd)) if sd.size else float("nan")
        if sd.size < 10 or abs(med) <= TIER_BOUNDS[0]:
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
    if not (sd.size >= 10 and abs(med) <= TIER_BOUNDS[0]):
        return ff, dict(applied=False, rows=0, **rec)
    out = ff.copy()
    out[inside] = band(uu[inside])
    return out, dict(
        applied=True,
        rows=int(inside.sum()),
        max_raw_dev=(float(np.nanmax(dev)) if dev.size else 0.0),
        **rec,
    )


# The parent anisotropies whose DF integrates to a known density. An
# isotropic parent's DF integrates to the parent's density, and an OM Type I
# parent's DF to the augmented density (1 + R^2 / r_a^2) rho_hat. A
# constant-beta or COM parent's DF, F = L^(-2 beta_c) f(Q), would need an
# identity of its own, which the module does not implement.
_PARENTS = ("isotropic", "OM_I")
# The highest P derivative of the auxiliary density that `parent_density`
# computes.
_MAX_ORDER = 3


def refusals(
    parent_rows: dict[str, Any],
    parent_anisotropy: Any,
    r_t: float,
    anisotropy: Any,
    gamma: float | None = None,
) -> None:
    r"""Raise an error for a component on a tabulated pair that has no DF.

    The checks run before any stage of `tabulated_df`. The central validity
    check equals the check that `truncate` runs for a Method I model: the
    inner slope
    :math:`\gamma` of the parent's untruncated profile must be at least
    :math:`2\beta_{\rm c}(0)`, the child's central anisotropy, with
    :math:`\beta_{\rm c}(0) = 0` for an Osipkov-Merritt child
    (Chiang+(2026d), Eq. (12)).

    Parameters
    ----------
    parent_rows : dict
        The parent's tabulated profile, as in `parent_sources`.
    parent_anisotropy : Anisotropy
        The parent's anisotropy.
    r_t : float
        The parent's truncation radius :math:`R_{\rm t}`, in internal units.
    anisotropy : Anisotropy
        The child's anisotropy.
    gamma : float, optional
        The inner slope :math:`\gamma` of the parent's untruncated profile.
        Default: ``None``, which means the central validity check is
        skipped.

    Raises
    ------
    UnphysicalModelError
        If :math:`\gamma < 2\beta_{\rm c}(0)` by more than
        :math:`10^{-12}`: the DF is then negative at the center
        (An & Evans 2006).
    InputError
        If the parent's anisotropy is neither isotropic nor Osipkov-Merritt
        Type I.
    InputError
        If the child's inversion needs P derivatives of the auxiliary
        density above the third.
    InputError
        If the child is of Type II with a finite :math:`R_{\rm a}` and the
        parent's truncation radius :math:`R_{\rm t}` is not below the
        child's :math:`R_{\rm a}`.
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
    """Return the parent's DF table, tabulated profile and anisotropy radius.

    The three values are the arguments `parent_df`, `parent_rows` and
    `parent_r_a` of `parent_sources`, read from the truncated profile. The
    DF's derivative is rebuilt from the table's logarithmic slope
    ``s = d log10 F / d ln K`` as
    ``dF/dK = s ln(10) F / K``, because the profile does not store dF/dK.
    The column ``M`` is 4 pi times the profile's enclosed mass.

    Parameters
    ----------
    profile : TruncatedProfile
        The parent's truncated profile.

    Returns
    -------
    parent_df : dict
        The parent's DF table, as in `parent_sources`.
    parent_rows : dict
        The parent's tabulated profile, as in `parent_sources`.
    parent_r_a : float or None
        The parent's anisotropy radius for an Osipkov-Merritt Type I parent;
        None for every other parent.

    Raises
    ------
    InputError
        If the profile carries no DF table of the parent model.
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
    """Return the child's DF table from the rows `f` on energies `k`.

    The table holds the base-10 logarithm of ``|f|`` on every row, and a
    sign column when any row of `f` is negative. The slope column is NaN,
    because the inversion on a tabulated pair does not compute dF/dK.
    """
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
    """Return the DF of a component on a tabulated pair.

    The tabulated pair is the truncated profile of a one-component model,
    the parent; the component carries its own anisotropy. The stages run in
    order: the energy grid on the parent's potential at the parent's first
    tabulated radius (`tabulated_grid`, placed around the energy of
    `pilot_feature` for an OM Type II or COM component); the sources from
    the integral of the parent's DF (`parent_sources`); the kernel's rows
    with the exact tail below the floor node (`child_rows`); the critical
    branch's closed-form band where the band applies (`critical_band`); the
    table; the asymptotic forms, which are the center's form of the
    component on the parent's untruncated profile and the edge law below
    the floor node; and the validation report. The DF's support ends at
    the parent's edge.

    Parameters
    ----------
    component : Component
        The component, whose profile is a `TruncatedProfile`.
    settings : Settings
        The numerical settings.

    Returns
    -------
    grid, table, form, k_t_min, report, inversion, rows
        The stages in the order and with the types `inversion_df` returns
        for an untruncated component.

    Raises
    ------
    InputError
        If `refusals` raises `InputError` for the component, or if the
        parent's profile carries no DF table of the parent model.
    UnphysicalModelError
        If `refusals` raises `UnphysicalModelError` for the component.
    UnphysicalModelError
        If the parent is truncated at a positive energy but has no edge.
    AccuracyError
        If a row of the child's inversion is not finite.
    ResultError
        If a band of rows at the top of the table is defective; the repair
        that an untruncated table receives is not applied on a tabulated
        pair.
    """
    profile, aniso = component.profile, component.anisotropy
    unit = profile.source
    with timed("potential_density_pair"):
        parent_df, parent_rows, parent_r_a = _parent_inputs(profile)
        # A parent truncation without an edge is not a truncated model. The
        # parent's tabulated profile then ends at the outer radius of the
        # parent's Poisson solve, where the potential is above K_FLOOR, and the
        # child's grid would have no nodes below the potential at that outer
        # radius.
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

        # The critical branch's closed-form band (`critical_band`).
        critical: dict[str, Any] = {}
        deficit = profile.central_deficit
        # A COM Type I component has beta(0) = beta_c, so a COM Type I
        # component with beta_c = 1/2 is on the critical branch too.
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

        # At the top of the table, `anchor` returns an anchor only for a
        # defective band of rows. Where the critical band applies, the rows at
        # the top of the table are already the closed form of `critical_band`.
        center = center_form(unit, aniso)
        kappa = float(center["kappa_sing"]) if center["branch"] in _PATCHED else 0.0
        at = anchor(k, f, p0_hat, kappa, center, math.nan, None, below_ceiling=False)
        if at is not None:
            raise ResultError(
                "The rows of a component on a tabulated pair carry a defective band "
                f"at the top (from K = {at.k_anchor!r}); the repair of an untruncated "
                "table is not made here."
            )

        # The record of the bands. No top band is repaired. The boundary term
        # at the floor node is the sources' value at the floor node, so the
        # floor term is exact and the floor band ends at K = 0.
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
    # The child's table has no slope column, so the report marks the slope
    # check as not judged and states the reason.
    report = slope_column_not_judged(report, table, NO_SLOPE_COLUMN_CHILD)

    # The grid's radii r(K), at which the parent's potential equals K, are
    # interpolated in the parent's tabulated profile; r(K) is NaN outside the
    # potentials of the parent's tabulated profile.
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
        # The record's conventions: NaN where the profile has no interior
        # point; (P_0, 0) for the high-K override and (0, R_t) for the low-K
        # override, because no row is replaced.
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


# The checks of a model on a tabulated pair: the table's top form, the join
# of the edge law to the table at the table's floor, and the profile's checks.
_MODEL_KEYS = ("top_form", "edge_join", *PROFILE_CHECK_KEYS)


def child_model(df: Any) -> Any:
    """Return the model of a DF on a tabulated pair.

    The model is truncated at the edge of the parent's truncated profile,
    on which the DF was inverted. The model's profile is the parent's
    tabulated profile, with the parent's truncation energy, the child's
    anisotropy and the child's DF table. The model's profile carries no
    central deficit, because the central deficit belongs to the parent.

    Parameters
    ----------
    df : DistributionFunction
        The DF of `tabulated_df`, of one component on a tabulated pair.

    Returns
    -------
    model : Model
        The model, with a validation report of the table's top form, the
        edge law's join and the profile's checks.

    Raises
    ------
    UnphysicalModelError
        If the DF is negative on an energy inside the DF's support: the
        child's anisotropy cannot be supported by the parent's truncated
        density. The exception carries `df` as its result.
    AccuracyError
        If a check of the model's report is above :math:`10^{-2}` and the
        settings of `df` do not set ``write_anyway=True``. The exception
        carries the model as its result.
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
    # The report's inherited level is the DF's validation level, as for an
    # untruncated model.
    report = ValidationReport(
        _MODEL_KEYS, {key: checks[key] for key in _MODEL_KEYS}, df.validation.level
    )
    # A check above 1e-2 raises AccuracyError, as `truncate_model` does for a
    # Method I model, unless the settings set write_anyway. With write_anyway,
    # the model is returned, its report records the checks above 1e-2, and
    # `announce` states that the model is returned on request.
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
            f"beyond 0.01.",
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
        + (" (returned on request, write_anyway: beyond 0.01)" if beyond else ""),
    )
    return model
