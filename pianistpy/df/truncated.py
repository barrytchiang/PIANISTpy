r"""The model of one component from its untruncated distribution function (DF).

`truncate` calls `truncate_model` when given a truncation, and
`untruncated_model` when given none for a DF inverted from an analytic
profile. `solve_target` calls `truncation_chain` for each trial truncation
energy. The truncation is Method I of Chiang+(2026d), Section 3.1: the
truncated DF at the truncation energy :math:`\mathcal{K}_{\rm t}` is

.. math::

    F_{\rm t}(K) = F(K + \mathcal{K}_{\rm t}) - F(\mathcal{K}_{\rm t}),

with :math:`F` the untruncated DF and :math:`K` the energy of the truncated
model, and the truncated model's potential solves Poisson's equation with
the density that :math:`F_{\rm t}` gives. The stages run in order: the base
table, which is the DF's rows continued above the rows' anchor by the
center's form (`base_table`); the gates on the truncation energy; the
truncated DF and the auxiliary density that the truncated DF gives
(`truncated_source`); the Poisson solve; the density on the solve's radii
and at the center; the fills; and the validation of the profile's radii.
Every number derives from the DF, the truncation and the settings.
`truncate_model` and `untruncated_model` raise `AccuracyError` when a check
of the model's validation report exceeds :math:`10^{-2}` (the band
"refused"), the round trip of the model's density rebuilt from the model's
DF included, unless the settings set ``write_anyway=True``.

References
----------
Notes Section T:3.
"""

__all__ = [
    "base_table",
    "log10_r_max",
    "truncate_model",
    "truncated_table",
    "truncation_chain",
]

import dataclasses
import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..exceptions import AccuracyError, InputError, UnphysicalModelError
from ..numerics.pchip import interp1d
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..profiles.series import inner_depth, series_join
from ..validation import TIER_BOUNDS, Check, ValidationReport, announce
from .center import center_amplitude
from .central import center_form, top_limit
from .checks import table_checks
from .core import model_identity
from .derivatives import ExactDerivs, host_reachable_radius
from .edge import density_edge_exponent
from .fill import fill_profile, profile_rows
from .form import tail_exponent_analytic
from .gates import truncation_gates
from .inputs import Truncation
from .inversion import order_pair
from .poisson import density_cut, profile_density, run_poisson_ode, source_slopes
from .profile_checks import PROFILE_CHECK_KEYS, _model_type, profile_checks
from .results import DistributionFunction, Model
from .stages import AsymptoticForm, DfTable
from .truncation import (
    CEILING_GAP,
    K_FLOOR,
    inner_zone_depth,
    row_bin_edges,
    truncated_source,
)
from .validate import validate_rows

# The number of nodes per decade of depth on which `center_amplitude`
# resolves the center of the truncated model.
_CENTER_NODES_PER_DECADE = 200
# The bisection of `log10_r_max` for the outer end of the radial grid: the
# upper bound in log10 R, the number of bisection steps, and the target
# potential as a fraction of K_FLOOR.
_R_MAX_BOUND = 60.0
_R_MAX_STEPS = 200
_R_MAX_DEPTH = 0.5
# The fewest and the most energies that `_extend` adds above a table's last
# row.
_EXTENSION_MIN = 500
_EXTENSION_MAX = 50000


def log10_r_max(profile: DoublePowerLaw, configured: float) -> float:
    """Return the outer end of the radial grid, in log10 R.

    When the profile's potential at the configured end is at most
    ``0.5 K_FLOOR``, the configured end is returned. Otherwise the function
    bisects in log10 R, up to 60, for the radius at which the potential
    falls to ``0.5 K_FLOOR``, and returns that radius's log10 rounded up to
    a multiple of 0.1.

    Parameters
    ----------
    profile : DoublePowerLaw
        The untruncated analytic profile.
    configured : float
        The configured outer end, ``log10_r_max`` of the truncation
        settings, in log10 of internal units of length.

    Returns
    -------
    log10_r_max : float
        The outer end of the radial grid, in log10 of internal units of
        length.
    """
    lo, hi = float(configured), _R_MAX_BOUND
    target = _R_MAX_DEPTH * K_FLOOR

    def p(log_r: float) -> float:
        return float(np.asarray(profile.p(np.array([10.0**log_r])))[0])

    if p(lo) <= target:
        return lo
    for _ in range(_R_MAX_STEPS):
        mid = 0.5 * (lo + hi)
        if p(mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.ceil(hi * 10.0) / 10.0)


def _extend(
    q: np.ndarray,
    f: np.ndarray,
    p_0: float,
    a_anchor: float,
    u_anchor: float,
    kappa: float,
    b: float,
    d: float,
    floor: float = K_FLOOR,
    *,
    ode_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the table continued above the table's last energy by the
    center's form.

    The center's form is
    :math:`a_{\rm anchor} (u / u_{\rm anchor})^{\kappa} + b + d\,u`, with
    :math:`u = P_0 - K` the depth below the central potential :math:`P_0`
    (`p_0`). The candidate energies are spaced logarithmically in the depth
    :math:`u`, from the depth `floor` up to the larger of the last energy's
    depth and the depth where the Poisson solve's inner zone ends
    (`inner_zone_depth`), with between 500 and 50000 candidates. The
    candidates that lie strictly between the table's last energy and the
    ceiling ``p_0 - floor`` are added. The table is returned unchanged when
    the ceiling is not above the table's last energy plus ``K_FLOOR``, or
    when no candidate lies strictly between the table's last energy and the
    ceiling.

    Parameters
    ----------
    q : array, shape (n,)
        The table's energies, increasing.
    f : array, shape (n,)
        The table's DF values.
    p_0 : float
        The central potential, in internal units.
    a_anchor : float
        The form's amplitude at the anchor depth.
    u_anchor : float
        The anchor depth.
    kappa : float
        The form's exponent.
    b : float
        The form's constant term.
    d : float
        The form's linear coefficient in ``u``.
    floor : float, optional
        The least depth of the table. Default: ``K_FLOOR``. A caller passes
        a larger depth when the exact DF at a half-integer
        :math:`\beta_{\rm c}` is not finite in double precision at smaller
        depths (`_table_top`).
    ode_rtol : float
        The relative tolerance of the Poisson solve, from the truncation
        settings.

    Returns
    -------
    q, f : array
        The table with the added energies and the form's values on them.

    Raises
    ------
    AccuracyError
        If a value of the form on an added energy is NaN or minus infinity.
    """
    ceiling = p_0 - floor
    q_last = float(q[-1])
    if not ceiling > q_last + K_FLOOR:
        return q, f
    step = max(float(np.diff(q[-min(10, len(q) - 1) :])[0]), 1e-15)
    n = min(max(_EXTENSION_MIN, int((ceiling - q_last) / step)), _EXTENSION_MAX)
    u_lo = max(p_0 - ceiling, floor)
    u_hi = max(p_0 - q_last, inner_zone_depth(p_0, ode_rtol))
    q_ext = p_0 - np.logspace(np.log10(u_lo), np.log10(u_hi), n)[::-1]
    q_ext = q_ext[(q_ext > q_last) & (q_ext < ceiling)]
    if q_ext.size == 0:
        return q, f
    f_ext = (
        a_anchor * (np.maximum(p_0 - q_ext, 1e-300) / max(u_anchor, 1e-300)) ** kappa
        + b
        + d * np.maximum(p_0 - q_ext, 0.0)
    )
    # A value of the form beyond the range of a double is +inf, which
    # `truncated_source` reads as the table's top band. A NaN or -inf value
    # raises AccuracyError.
    bad = np.isnan(f_ext) | np.isneginf(f_ext)
    if bad.any():
        err = AccuracyError(
            f"{int(bad.sum())} values of the center's form above the table are NaN "
            f"or -inf, the first at K = {float(q_ext[bad][0]):.6e}: refused"
        )
        err.result = dict(k=q_ext, f=f_ext)
        raise err
    return np.concatenate([q, q_ext]), np.concatenate([f, f_ext])


def base_table(df: DistributionFunction) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the base table, the untruncated DF table that the truncation reads.

    When the DF's central anchor energy is finite, the base table holds the
    DF's rows up to the anchor energy, and the center's form above the
    anchor energy, extended to the ceiling by `_extend`; otherwise the base
    table is the DF's rows. At a half-integer constant
    :math:`\beta_{\rm c}`, the exact closed-form DF replaces the table's DF
    at every energy where the exact derivatives are evaluated (depth above
    `_top_limit`) and the closed-form DF is positive.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF, with its rows.

    Returns
    -------
    q : array
        The base table's energies, in internal units.
    f : array
        The base table's DF values.

    Raises
    ------
    InputError
        If the DF carries no rows, because the DF was not inverted.
    """
    rows = df.rows
    if rows is None:
        raise InputError("The DF carries no rows as made; it was not inverted.")
    q = np.asarray(rows.k[0], dtype=np.float64)
    f = np.asarray(rows.f[0], dtype=np.float64).copy()
    k_anchor = float(df.form.central_anchor[0])
    if math.isfinite(k_anchor):
        above = q > k_anchor
        f[above] = df.form.central_df(q[above])
        b, d = df.form.central_regular[0]
        q, f = _extend(
            q,
            f,
            1.0,
            float(df.form.central_anchor_amplitude[0]),
            1.0 - k_anchor,
            float(df.form.central_exponent[0]),
            float(b),
            float(d),
            max(K_FLOOR, _table_top(df.components[0])),
            ode_rtol=float(df.settings.truncation["ode_rtol"]),
        )
    component = df.components[0]
    aniso = component.anisotropy
    order = _half_integer_order(aniso)
    if order is not None:
        derivs = ExactDerivs(
            component.unit_profile,
            beta_c=float(aniso.beta_c),
            model="constant",
            max_order=order + 1,
        )
        exact = np.zeros_like(q)
        # The exact DF is used only at depths where the exact derivatives of
        # the auxiliary density are finite in double precision.
        u_min = _top_limit(component, (0, 1, order))
        inside = (q > 0.0) & (q < 1.0) & (1.0 - q > u_min)
        exact[inside] = np.asarray(
            derivs.derivs_at_p(q[inside], order)[order], dtype=np.float64
        ) / (2.0 * np.pi**2)
        f = np.where(exact > 0.0, exact, f)
    return q, f


def central_series_refusal(df: DistributionFunction) -> None:
    r"""Raise an error for a `DoublePowerLaw` model whose table ends only at
    the depth of `host_reachable_radius`.

    `refusals` runs this check after the central validity check and before
    the central sign check; `untruncated_model` runs this check first.
    Without the check, the DF of such a model takes a spurious global
    minimum near the center, and the untruncated model's check would report
    that minimum instead of the cause.

    The top of a table is set by `top_limit` and by the depth at
    `host_reachable_radius`, the smallest radius whose square double
    precision holds; the center's closed form continues the table above
    the top. When the leading exponent of the center's series vanishes,
    at :math:`\gamma = 2\beta_{\rm c}(0)`, `top_limit` is 0 and gives no
    closed form. When, in addition, the depth at `host_reachable_radius`
    exceeds ``CEILING_GAP``, which happens for :math:`\gamma` near 2, the
    depth at `host_reachable_radius` alone ends the table, with no closed
    form to join above the table.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF of one component.

    Raises
    ------
    UnphysicalModelError
        If the profile is a `DoublePowerLaw`, `top_limit` is 0, and the
        depth at `host_reachable_radius` exceeds ``CEILING_GAP``. The
        message states :math:`\gamma`, :math:`2\beta_{\rm c}(0)` and the
        depth at `host_reachable_radius`; the exception carries `df` as its
        result.
    """
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    if not isinstance(profile, DoublePowerLaw) or top_limit(profile, aniso) > 0.0:
        return
    r_floor = host_reachable_radius()
    u_floor = float(profile.depth_inner(np.array([r_floor]), terms=8)[0][0])
    if not u_floor > CEILING_GAP:
        return
    beta_0 = 0.0 if _model_type(aniso).startswith("OM") else float(aniso.beta_c)
    raise UnphysicalModelError(
        f"CENTRAL SERIES: gamma = {float(profile.rho_gamma):g} = 2 beta_c(0) = "
        f"{2.0 * beta_0:g}: the leading exponent of the center's series vanishes "
        "(a degenerate central series), and the depth u = "
        f"{u_floor:.6e} at R = {r_floor:.6e}, the smallest radius whose square "
        "host float64 holds, alone would end the table, with no closed form of "
        "the center to join above it. Use a smaller central beta_c or a "
        "shallower cusp.",
        result=df,
    )


def refusals(df: DistributionFunction, k_t: float) -> None:
    r"""Raise an error for a DF that no truncation can turn into a model.

    The checks run before any stage of the truncation, in three steps.
    First, the central validity check: the inner slope :math:`\gamma` must
    be at least :math:`2\beta_{\rm c}(0)`, with
    :math:`\beta_{\rm c}(0) = 0` for the isotropic and Osipkov-Merritt
    models (the cusp slope-central anisotropy theorem, An & Evans 2006,
    Chiang+(2026d), Eq. (12)). A DF that fails the check is negative at the
    center for every :math:`\mathcal{K}_{\rm t}`. Second,
    `central_series_refusal`. Third, the central sign check: the central DF
    must not tend to minus infinity, to zero from below, or to a negative
    constant.

    The bound of a Type II model, :math:`\mathcal{K}_{\rm t}` above
    :math:`P(R_{\rm a})`, is checked by the DOMAIN gate of
    `truncation_gates`, not here.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF of one component.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal
        units. None of the three checks reads `k_t`.

    Raises
    ------
    UnphysicalModelError
        If :math:`\gamma < 2\beta_{\rm c}(0)` by more than
        :math:`10^{-12}`.
    UnphysicalModelError
        If `central_series_refusal` raises.
    UnphysicalModelError
        If the coefficient of the center's closed form is negative, so
        that the central DF tends to minus infinity or to zero from below.
    UnphysicalModelError
        If the center's form is a constant plus a power, or analytic, and
        the constant term ``F0`` is negative at the row nearest the top
        with a depth of at least ``CEILING_GAP``. For a constant plus a
        power, ``F0 = F - A u^kappa`` with ``A`` the center's amplitude and
        ``kappa`` the center's singular exponent; for an analytic center,
        ``F0 = F``.

    Each exception carries `df` as its result.
    """
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    beta_0 = 0.0 if _model_type(aniso).startswith("OM") else float(aniso.beta_c)
    gamma = float(profile.rho_gamma)
    if gamma < 2.0 * beta_0 - 1e-12:
        raise UnphysicalModelError(
            f"CENTRAL VALIDITY: gamma = {gamma:.2f} < 2 beta_c(0) = "
            f"{2.0 * beta_0:.2f}: "
            "the DF is negative at the center (An & Evans 2006); no K_t removes "
            "this. Use a steeper cusp or a smaller central beta_c.",
            result=df,
        )
    central_series_refusal(df)
    center = center_form(profile, aniso)
    if float(center["sign"]) < 0.0:
        coefficient = (
            center["log_coef"] if center["form"] == "log" else center["amplitude"]
        )
        raise UnphysicalModelError(
            "CENTRAL SIGN: the central DF tends to minus infinity or to zero from "
            f"below: {center['branch']}, coefficient {float(coefficient):.6e} < 0.",
            result=df,
        )
    rows = df.rows
    if rows is not None and center["form"] in ("constant_plus_power", "analytic"):
        q = np.asarray(rows.k[0], dtype=np.float64)
        f = np.asarray(rows.f[0], dtype=np.float64)
        u = 1.0 - q
        ok = np.isfinite(f) & (u >= CEILING_GAP * (1.0 - 1e-9))
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmin(u[ok])])
            f_0 = float(f[j])
            if center["form"] == "constant_plus_power":
                f_0 -= float(center["amplitude"]) * u[j] ** float(center["kappa_sing"])
            if f_0 < 0.0:
                raise UnphysicalModelError(
                    "CENTRAL SIGN: the central DF tends to a negative constant: "
                    f"{center['branch']}, F0 = {f_0:.6e} < 0.",
                    result=df,
                )


def _domain_edge_potential(profile: Any, aniso: Any) -> float:
    r"""Return the untruncated potential at the edge of the anisotropy's domain.

    The value is :math:`P(R_{\rm a})` for a Type II model, and 0 for a
    model whose anisotropy's domain has no edge.
    """
    edge = aniso.domain_edge
    if not math.isfinite(edge):
        return 0.0
    return float(np.asarray(profile.p(np.array([edge])))[0])


def truncation_chain(df: DistributionFunction, k_t: float) -> dict[str, Any]:
    r"""Return every stage of the truncated model at the truncation energy `k_t`.

    The record holds the output of each stage and the inputs each stage
    was given, so that `truncate_model` and `solve_target` read the same
    stages.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF of one component, with its rows.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal
        units.

    Returns
    -------
    chain : dict
        ``gates_in`` and ``gates``, ``source_in`` and ``source``, ``ode_in``
        and ``solve``, ``rho`` (the density on the solve's radii),
        ``center_in`` and ``center``, ``filled``, ``rows`` (the profile's
        radii), ``validated`` and ``log10_r_max``.

    Raises
    ------
    InputError
        If the DF carries no rows, because the DF was not inverted.
    AccuracyError
        If `base_table` raises `AccuracyError` (a NaN or minus infinity
        value of the center's form in `_extend`).
    UnphysicalModelError
        If `refusals` raises.
    UnphysicalModelError
        If a gate of `truncation_gates` fires.
    UnphysicalModelError
        If the Poisson solve raises `UnphysicalModelError`.

    Each `UnphysicalModelError` carries `df` as its result.
    """
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    settings = df.settings
    rows = df.rows
    if rows is None:
        raise InputError("The DF carries no rows as made; it was not inverted.")
    q_sign = np.asarray(rows.k[0], dtype=np.float64)
    f_sign = np.asarray(rows.f[0], dtype=np.float64)
    ddf = np.asarray(rows.ddf[0], dtype=np.float64)
    refusals(df, k_t)
    q, f = base_table(df)

    # The gates read both the DF's rows and the base table.
    gates_in: dict[str, Any] = dict(
        q_sign=q_sign,
        f_sign=f_sign,
        q=q,
        f=f,
        k_t=float(k_t),
        a_analytic=float(tail_exponent_analytic(profile, aniso)),
        k_edge=_domain_edge_potential(profile, aniso),
    )
    gates = truncation_gates(**gates_in)
    if gates.fired:
        why = ""
        if "DOMAIN" in gates.fired:
            why = (
                f" (DOMAIN: K_t must lie above P(R_a) = {gates_in['k_edge']!r}, the "
                f"potential at the anisotropy's domain edge R_a = "
                f"{float(aniso.r_a):g}, so that R_t < R_a; Merritt 1985)"
            )
        raise UnphysicalModelError(
            f"The truncation at K_t = {k_t!r} is refused: "
            f"{'; '.join(gates.fired)}{why}.",
            result=df,
        )

    # The truncated DF and the auxiliary density that the truncated DF gives.
    model_type = _model_type(aniso)
    is_om = model_type.startswith("OM")
    beta_c = 0.0 if is_om else float(aniso.beta_c)
    ra = float(aniso.r_a) if math.isfinite(aniso.r_a) else 1.0
    fin = np.isfinite(f_sign)
    source_in: dict[str, Any] = dict(
        q_table=q,
        df_table=f,
        ddf_dq=ddf,
        df_has_neg=bool(np.any(f_sign[fin] < 0.0)),
        k_t=float(k_t),
        p_center=1.0,
        beta_c=beta_c,
        model_type=model_type,
    )
    r_max = log10_r_max(profile, float(settings.truncation["log10_r_max"]))
    trunc_settings = dict(settings.truncation, log10_r_max=r_max)
    source = truncated_source(**source_in, settings=trunc_settings)

    # The Poisson solve.
    p_0 = 1.0 - float(k_t)
    alpha, beta, gamma = profile.rho_alpha, profile.rho_beta, profile.rho_gamma
    is_nfw = (alpha, beta, gamma) == (1.0, 3.0, 1.0)

    if is_nfw:

        def potential(r: Any) -> Any:
            return jnp.log1p(r) / r

    else:

        def potential(r: Any) -> Any:
            return jnp.asarray(profile.p(np.asarray(r, dtype=np.float64)))

    p_s = float(df.inversion.k_interior[0]) if df.inversion is not None else math.nan
    ode_in: dict[str, Any] = dict(
        q=np.asarray(source.q, dtype=np.float64),
        p_rhs=np.asarray(source.rho_aux, dtype=np.float64),
        p_0=p_0,
        ra=ra,
        bc=beta_c,
        model_type=model_type,
        gamma_inner=float(gamma),
        p_s=p_s,
        sentinel_q=source.sentinel_q,
    )
    try:
        solve = run_poisson_ode(
            ode_in["q"],
            ode_in["p_rhs"],
            p_0,
            ra,
            beta_c,
            model_type,
            float(gamma),
            potential,
            trunc_settings,
            p_s=p_s,
            sentinel_q=source.sentinel_q,
            edge_continuation=float(k_t) > 0.0,
        )
    except UnphysicalModelError as exc:
        raise UnphysicalModelError(
            f"The truncation at K_t = {k_t!r} is refused: {exc}.", result=df
        ) from exc

    # The density on the solve's radii, and the center of the truncated model.
    m = source_slopes(ode_in["q"], ode_in["p_rhs"], p_0, p_s, "cubic")
    rho = profile_density(
        solve.r,
        solve.phi,
        ode_in["q"],
        ode_in["p_rhs"],
        ra,
        beta_c,
        model_type,
        p_0=p_0,
        m=m,
        order="cubic",
        r_cut=density_cut(float(profile.rho_norm), float(gamma)),
    )
    kind = aniso.kind if math.isfinite(aniso.r_a) else None
    center_in: dict[str, Any] = dict(
        k_t=float(k_t),
        p_0=p_0,
        pot_center=1.0,
        q=q,
        df=f,
        f_at_k_t=float(source.df_at_k_t),
        bc=beta_c,
        profile=(float(profile.rho_norm), float(alpha), float(beta), float(gamma)),
        factor=(kind, ra),
        n_per_decade=_CENTER_NODES_PER_DECADE,
        goal=TIER_BOUNDS[0],
    )
    center = center_amplitude(
        solve.r,
        solve.phi,
        rho,
        center_in["k_t"],
        p_0,
        1.0,
        q,
        f,
        center_in["f_at_k_t"],
        beta_c,
        center_in["profile"],
        center_in["factor"],
        n_per_decade=_CENTER_NODES_PER_DECADE,
    )

    # The fills, the profile's radii and the validation of the radii.
    taylor = source.taylor
    edge = None if taylor is None else (*taylor, beta_c, model_type, ra)
    ode_record = dict(ode_in, R_MAX_EXP=r_max)
    filled = fill_profile(
        solve,
        rho,
        p_0,
        center,
        center_in["profile"],
        beta_c,
        center_in["factor"],
        edge,
        ode_record,
        trunc_settings,
    )
    made_rows = profile_rows(
        filled,
        solve,
        p_0,
        center,
        center_in["profile"],
        beta_c,
        float(k_t),
        1.0,
        q,
        f,
        center_in["f_at_k_t"],
    )
    validated = validate_rows(
        made_rows,
        solve,
        float(k_t),
        1.0,
        DoublePowerLaw(alpha, beta, gamma),
        trunc_settings,
    )
    return dict(
        gates_in=gates_in,
        gates=gates,
        source_in=source_in,
        source=source,
        ode_in=ode_record,
        solve=solve,
        rho=rho,
        center_in=center_in,
        center=center,
        filled=filled,
        rows=made_rows,
        validated=validated,
        log10_r_max=r_max,
    )


def _half_integer_order(aniso: Any) -> int | None:
    r"""Return the order of the closed-form DF at a half-integer central anisotropy.

    At a half-integer constant :math:`\beta_{\rm c}`, the kernel exponent of
    `order_pair` is exactly zero and the DF is the derivative of order
    :math:`m = 3/2 - \beta_{\rm c}` of the auxiliary density: :math:`m = 1`
    at :math:`\beta_{\rm c} = 1/2` and :math:`m = 2` at
    :math:`\beta_{\rm c} = -1/2`. The package identifies the half-integer
    case by the zero kernel exponent alone.

    Returns
    -------
    order : int or None
        The order :math:`m`; None for an anisotropy with a finite
        anisotropy radius or a kernel exponent other than zero.
    """
    pair = order_pair(float(aniso.beta_c))
    if math.isfinite(aniso.r_a) or pair.kernel_exponent != 0.0:
        return None
    return pair.order_m


def _top_limit(component: Any, orders: tuple[int, ...]) -> float:
    """Return `top_limit` for the exact derivatives of the given orders.

    The value is the least depth at which the exact derivatives of the
    auxiliary density are finite in double precision, for a component
    whose profile is a `DoublePowerLaw`; the value is 0 for every other
    profile.
    """
    profile = component.unit_profile
    if not isinstance(profile, DoublePowerLaw):
        return 0.0
    return top_limit(profile, component.anisotropy, orders)


def _table_top(component: Any) -> float:
    r"""Return the depth below which the half-integer closed-form DF is not finite.

    The depth is `_top_limit` for the derivative orders the exact DF at a
    half-integer constant :math:`\beta_{\rm c}` reads; the depth is 0 for
    every other anisotropy. When the returned depth exceeds ``K_FLOOR``,
    `base_table` ends the center's form at the returned depth and
    `truncated_table` drops the energies whose depth below :math:`P_0` is
    at or below the returned depth.
    """
    order = _half_integer_order(component.anisotropy)
    return 0.0 if order is None else _top_limit(component, (0, 1, order))


def _exact(component: Any, q: np.ndarray, order: int, derivative: bool) -> np.ndarray:
    r"""Return the half-integer closed-form DF, or the DF's K derivative.

    At a half-integer constant :math:`\beta_{\rm c}`, the DF is
    ``rho_aux^(m) / (2 pi^2)`` and the DF's K derivative is
    ``rho_aux^(m+1) / (2 pi^2)``, with ``rho_aux^(m)`` the m-th P
    derivative of the auxiliary density and ``m`` = `order`.

    Parameters
    ----------
    component : Component
        The component.
    q : array, shape (n,)
        The energies, in internal units.
    order : int
        The order ``m`` of `_half_integer_order`.
    derivative : bool
        True to return the K derivative of the DF; False to return the DF.

    Returns
    -------
    values : array, shape (n,)
        The DF or the DF's K derivative on `q`. Outside ``0 < q < 1``, and
        at depths ``1 - q`` at or below `_top_limit`, the DF is 0 and the
        derivative is NaN. A derivative that is not finite is NaN.
    """
    aniso = component.anisotropy
    n = order + 1 if derivative else order
    derivs = ExactDerivs(
        component.unit_profile,
        beta_c=float(aniso.beta_c),
        model="constant",
        max_order=order + 1,
    )
    out = np.full(q.size, np.nan) if derivative else np.zeros_like(q)
    # The exact derivatives are evaluated only at depths where they are
    # finite in double precision.
    u_min = _top_limit(component, (0, 1, n))
    inside = (q > 0.0) & (q < 1.0) & (1.0 - q > u_min)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.asarray(derivs.derivs_at_p(q[inside], n)[2 if derivative else n])
    values = values.astype(np.float64) / (2.0 * np.pi**2)
    out[inside] = (
        np.where(np.isfinite(values), values, np.nan) if derivative else values
    )
    return out


def _exact_taylor(component: Any, k_t: float, k: np.ndarray, order: int) -> np.ndarray:
    r"""Return the shifted half-integer closed-form DF as a Taylor sum.

    The shifted DF ``F_u(K + K_t) - F_u(K_t)`` is evaluated as its Taylor
    sum at :math:`\mathcal{K}_{\rm t}` to the cubic term,
    ``SUM_{j=1}^{3} F_u^(j)(K_t) K^j / j!``, with
    ``F_u^(j) = rho_aux^(m+j) / (2 pi^2)`` and ``F_u`` the untruncated
    closed-form DF. Every term is positive, so the sum has no cancellation,
    whereas the direct difference cancels at small K. `truncated_table`
    uses the sum at the energies at or below the switch energy of the
    truncated DF's Taylor form (the third entry of the truncated source's
    ``taylor``).

    Parameters
    ----------
    component : Component
        The component.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal
        units.
    k : array, shape (n,)
        The energies of the truncated DF, in internal units.
    order : int
        The order ``m`` of `_half_integer_order`.

    Returns
    -------
    f : array, shape (n,)
        The Taylor sum on `k`.
    """
    aniso = component.anisotropy
    derivs = ExactDerivs(
        component.unit_profile,
        beta_c=float(aniso.beta_c),
        model="constant",
        max_order=order + 3,
    )
    at = np.array([k_t])
    out = np.zeros_like(k)
    for j in (3, 2, 1):
        d_j = float(np.asarray(derivs.derivs_at_p(at, order + j)[2])[0])
        out = (out + d_j / (2.0 * np.pi**2) / math.factorial(j)) * k
    return out


def truncated_table(
    df: DistributionFunction, k_t: float, chain: dict[str, Any]
) -> dict[str, Any]:
    r"""Return the truncated DF's table on the energies of the truncated source.

    The table's DF is the truncated source's DF, ``F(K + K_t) - F(K_t)``,
    at the energies K at or below the central potential
    :math:`P_0 = 1 - \mathcal{K}_{\rm t}` and, where `_table_top` exceeds
    ``K_FLOOR``, at depths :math:`P_0 - K` above `_table_top`. For a model
    that is not Type II and whose center's branch is ``"regular"`` or
    ``"critical_divergent"``, the center's form fitted at :math:`P_0`
    (`top_patch`) replaces the DF above the anchor energy that `anchor`
    returns, and `_extend` continues the table to the ceiling. At a
    half-integer constant :math:`\beta_{\rm c}`, the exact shifted closed
    form replaces the DF at every energy where the shifted closed form is
    positive and :math:`K + \mathcal{K}_{\rm t} < 1`. At and below the
    switch energy of the truncated DF's Taylor form, the shifted closed
    form is its Taylor sum at :math:`\mathcal{K}_{\rm t}`
    (`_exact_taylor`), because the direct difference cancels at those
    energies.

    The table's dF/dK is the untruncated DF's derivative at
    :math:`K + \mathcal{K}_{\rm t}`, taken in five steps, each step
    overriding the previous steps on the step's range:

    1. Between the lowest and the highest usable untruncated row, the
       logarithmic slope :math:`d \ln F / d \ln Q` of the untruncated rows,
       read by a not-a-knot cubic spline in :math:`\ln Q` on each run of at
       least 4 consecutive usable rows (runs split at the interior point)
       and by a PCHIP interpolant elsewhere, times F at
       :math:`Q = K + \mathcal{K}_{\rm t}` from the truncated source's
       lookup (or, without a lookup, from a PCHIP interpolant of
       :math:`\ln F` of the usable rows), divided by
       :math:`K + \mathcal{K}_{\rm t}`. Outside the energy range of the
       usable untruncated rows, dF/dK is 0. With fewer than 4 usable rows,
       steps 1 to 5 are skipped and dF/dK is NaN at every energy.
    2. For :math:`\mathcal{K}_{\rm t} > 0`, at K from 0 to the splice
       energy of the edge, the Lagrange interpolation of the edge's nodes,
       where the interpolated value is finite and positive.
    3. Where :math:`K + \mathcal{K}_{\rm t}` lies within a factor 3 of a
       sign change of the untruncated rows' dF/dQ at an untruncated energy
       of at least :math:`10^{-3}`, a four-node Lagrange interpolation of
       the untruncated rows' dF/dQ.
    4. At a half-integer constant :math:`\beta_{\rm c}`, the closed form,
       where the closed form is finite.
    5. At and above the anchor energy of the truncated table's top patch,
       or, without such a patch, of the untruncated DF's top patch at
       :math:`P_0 = 1`, the derivative of the patch's form.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF, with its rows.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal
        units.
    chain : dict
        The stages of `truncation_chain` at `k_t`.

    Returns
    -------
    table : dict
        ``k`` (the energies), ``f`` (the DF), ``log_f`` (the base-10
        logarithm of the DF, minus infinity where the DF is not positive),
        ``ddf`` (dF/dK) and ``override`` (the record of the top patch, or
        None where no patch replaced a row).
    """
    from scipy.interpolate import CubicSpline, PchipInterpolator

    from .top import anchor, top_patch
    from .truncation import _lagrange4

    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    rows = df.rows
    assert rows is not None
    source = chain["source"]
    k = np.asarray(source.q, dtype=np.float64)
    f = np.asarray(source.df, dtype=np.float64).copy()
    p_0 = 1.0 - float(k_t)
    keep = k <= p_0
    u_top = _table_top(component)
    if u_top > K_FLOOR:
        keep &= p_0 - k > u_top
    k, f = k[keep], f[keep]

    # The top patch at the central potential P_0 = 1 - K_t, for a model that
    # is not Type II and whose center's branch is "regular" or
    # "critical_divergent".
    center = center_form(profile, aniso)
    patched = aniso.kind != "II" and center["branch"] in (
        "regular",
        "critical_divergent",
    )
    data_top = float(rows.data_top[0])
    reached = data_top >= 1.0 - K_FLOOR
    override: dict[str, Any] | None = None
    if patched:
        kappa = float(center["kappa_sing"])
        at = anchor(k, f, p_0, kappa, center, data_top, None, below_ceiling=not reached)
        if at is not None:
            amplitude = float(center["amplitude"])
            closed = (
                center["form"] == "power"
                and math.isfinite(amplitude)
                and abs(float(center["kappa_sing"]) - kappa) <= 1e-12
            )
            below = k <= at.k_anchor
            patch = top_patch(
                k[below], f[below], p_0, kappa, amplitude if closed else None, k[~below]
            )
            if patch is not None:
                u = p_0 - k[~below]
                ratio = np.maximum(u / max(patch.u_anchor, 1e-300), 1e-300)
                f[~below] = patch.a_anchor * ratio**kappa + patch.b + patch.d * u
                k, f = _extend(
                    k,
                    f,
                    p_0,
                    patch.a_anchor,
                    patch.u_anchor,
                    kappa,
                    patch.b,
                    patch.d,
                    max(K_FLOOR, u_top),
                    ode_rtol=float(df.settings.truncation["ode_rtol"]),
                )
                two_terms = math.isfinite(patch.fit_window)
                override = dict(
                    K_anchor=at.k_anchor,
                    P0=p_0,
                    rows=int(np.sum(~below)),
                    rule=at.rule,
                    kappa=kappa,
                    slope_mismatch=at.slope_mismatch,
                    B=patch.b if two_terms else 0.0,
                    D=patch.d if two_terms else 0.0,
                )
    order = _half_integer_order(aniso)
    if order is not None:
        f_kt = float(_exact(component, np.array([float(k_t)]), order, False)[0])
        shifted = _exact(component, k + float(k_t), order, False) - f_kt
        ok = (shifted > 0.0) & (k + float(k_t) < 1.0)
        f = np.where(ok, shifted, f)
        if source.taylor is not None:
            # Below the switch of the Taylor form, the difference
            # F_u(K + K_t) - F_u(K_t) cancels, so the closed form's Taylor sum
            # at K_t replaces the difference there.
            low = k <= float(source.taylor[2])
            f = np.where(low, _exact_taylor(component, float(k_t), k, order), f)
    with np.errstate(divide="ignore"):
        log_f = np.where(f > 0.0, np.log10(np.maximum(f, 1e-320)), -np.inf)

    # The table's dF/dK is the untruncated rows' derivative at K + K_t.
    q_s, f_s = base_table(df)
    d_raw = np.asarray(rows.ddf[0], dtype=np.float64)
    q_s, f_s = q_s[: d_raw.size], f_s[: d_raw.size]
    ok = np.isfinite(d_raw) & np.isfinite(f_s) & (f_s > 0) & (q_s >= K_FLOOR)
    ok &= np.concatenate([[True], np.diff(q_s) > 0.0])
    ddf = np.full(k.shape, np.nan)
    if int(ok.sum()) >= 4:
        s_r = d_raw[ok] * q_s[ok] / f_s[ok]
        p_s_lin = PchipInterpolator(np.log(q_s[ok]), s_r)
        p_f = PchipInterpolator(np.log(q_s[ok]), np.log(f_s[ok]))
        qq = k + float(k_t)
        inside = (qq >= q_s[ok][0]) & (qq <= q_s[ok][-1])
        lq = np.log(np.clip(qq, q_s[ok][0], q_s[ok][-1]))
        q_d, x_d = q_s[ok], np.log(q_s[ok])
        x_q = np.log(np.maximum(qq, 1e-300))
        s_d = np.array(p_s_lin(lq), dtype=np.float64)
        runs = np.cumsum(np.r_[0, np.diff(np.flatnonzero(ok)) != 1])
        p_int = (
            float(df.inversion.k_interior[0]) if df.inversion is not None else math.nan
        )
        if np.isfinite(p_int):
            runs = runs * 2 + (q_d > p_int)
        for run in np.unique(runs):
            idx = np.flatnonzero(runs == run)
            if idx.size < 4:
                continue
            sel = inside & (qq >= q_d[idx[0]]) & (qq <= q_d[idx[-1]])
            s_d[sel] = CubicSpline(x_d[idx], s_r[idx], bc_type="not-a-knot")(x_q[sel])
        look = source.lookup
        if look is not None:
            f_q = np.array(
                interp1d(
                    jnp.array(np.where(inside, qq, q_d[0])),
                    look["xp"],
                    look["fp"],
                    0.0,
                    look["right"],
                    order="cubic",
                    logx=True,
                    logf=look["logf"],
                    m=None if look["m"] is None else jnp.asarray(look["m"]),
                ),
                dtype=np.float64,
            )
        else:
            f_q = np.exp(p_f(lq))
        ddf = np.where(inside, f_q * s_d / np.maximum(qq, 1e-300), 0.0)
        # The rows at or below the splice of the edge.
        nodes = source.edge_nodes
        if float(k_t) > 0.0 and nodes is not None and source.taylor_check is not None:
            x_e, y_e = nodes
            k_spl = float(source.taylor_check["K_splice"])
            kk = qq - float(k_t)
            on = np.isfinite(kk) & (kk >= 0.0) & (kk <= k_spl) & (qq <= x_e[-1])
            if on.any():
                fe = np.asarray(_lagrange4(x_e, y_e, qq[on]), dtype=np.float64)
                good = np.isfinite(fe) & (fe > 0)
                ddf[np.flatnonzero(on)[good]] = fe[good]
        # The rows whose K + K_t lies within a factor 3 of a sign change of
        # the untruncated rows' dF/dQ at an untruncated energy of at least 1e-3.
        q_z, d_z = q_s[ok], d_raw[ok]
        cross = np.flatnonzero(np.sign(d_z[1:]) * np.sign(d_z[:-1]) < 0.0)
        cross = cross[q_z[cross] >= 1e-3]
        if cross.size:
            near = np.zeros(qq.size, dtype=bool)
            for i in cross.tolist():
                q_0 = 0.5 * (q_z[i] + q_z[i + 1])
                near |= (qq >= q_0 / 3.0) & (qq <= 3.0 * q_0)
            if near.any():
                ddf[near] = np.asarray(_lagrange4(q_z, d_z, qq[near]), dtype=np.float64)
        # The closed form at a half-integer constant beta_c.
        if order is not None:
            d_x = _exact(component, np.where(inside, qq, q_d[0]), order, True)
            good = inside & np.isfinite(d_x)
            ddf[good] = d_x[good]
        # The derivative of the last top patch's form, at and above the
        # patch's anchor. The last patch is the truncated table's patch, or
        # else the untruncated DF's patch at P_0 = 1.
        last = override
        if last is None and math.isfinite(float(df.form.central_anchor[0])):
            b, d = df.form.central_regular[0]
            last = dict(
                K_anchor=float(df.form.central_anchor[0]),
                P0=1.0,
                rows=1,
                kappa=float(df.form.central_exponent[0]),
                B=float(b) if float(d) != 0.0 else 0.0,
                D=float(d),
            )
        if last is not None and np.isfinite(last["kappa"]) and int(last["rows"]) > 0:
            u_f = float(last["P0"]) - k
            top = (
                (k >= float(last["K_anchor"]))
                & (u_f > 0.0)
                & np.isfinite(f)
                & (f > 0.0)
            )
            if top.any():
                ddf[top] = -(
                    float(last["kappa"])
                    * (f[top] - last["B"] - last["D"] * u_f[top])
                    / u_f[top]
                    + last["D"]
                )
    return dict(k=k, f=f, log_f=log_f, ddf=ddf, override=override)


# The keys of the center's form under the names that `table_checks` reads.
_TOP_KEYS = (
    ("top_kappa", "kappa"),
    ("top_kappa_branch", "branch"),
    ("top_form", "form"),
    ("top_singular_exponent", "kappa_sing"),
    ("top_rho_power", "s"),
    ("top_amplitude", "amplitude"),
    ("top_log_coefficient", "log_coef"),
    ("top_next_form", "next_form"),
    ("top_next_exponent", "next_kappa"),
    ("top_next_coefficient", "next_coef"),
    ("top_next_rho_power", "next_s"),
)
# The checks of a truncated Method I model's report.
_MODEL_KEYS = ("top_form", "slope_column", "edge_join", "df_clip", *PROFILE_CHECK_KEYS)


def _edge_exponent_of(gates: Any, k_t: float, beta_c: float) -> tuple[float, ...]:
    r"""Return the exponent of the truncated model's density at the edge.

    Near the edge the truncated DF behaves as :math:`F \propto K^p`, and
    `density_edge_exponent` turns the DF's exponent :math:`p` into the
    density's exponent. Above the lowest admissible truncation energy
    ``k_t_min``, :math:`p = 1`. At ``k_t_min``, :math:`p = 2` when the DF's
    global minimum sets ``k_t_min``, and :math:`p = 1` when the DF's zero
    crossing or the anisotropy's domain edge sets ``k_t_min``.

    Parameters
    ----------
    gates : TruncationGates
        The gates of `truncation_gates`.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal
        units.
    beta_c : float
        The central anisotropy :math:`\beta_{\rm c}`, 0 for an
        Osipkov-Merritt model.

    Returns
    -------
    exponent : tuple of float
        The density's exponent at the edge, as a one-element tuple.

    Raises
    ------
    InputError
        If `k_t` is below ``k_t_min``, or equals ``k_t_min`` while a
        feature other than the global minimum, the zero crossing or the
        domain edge sets ``k_t_min``.
    """
    gate = float(gates.k_t_min)
    if k_t > gate or (k_t == gate and gates.set_by == "domain_edge"):
        p = 1.0
    elif k_t == gate and gates.set_by in ("K_0", "Q_min"):
        p = 2.0 if gates.set_by == "Q_min" else 1.0
    else:
        raise InputError(
            f"The edge exponent is not stated: k_t = {k_t!r} is not above the lowest "
            f"admissible truncation energy {gate!r}."
        )
    return (density_edge_exponent(p, beta_c),)


def truncate_model(df: DistributionFunction, truncation: Truncation) -> Model:
    """Return the truncated model of a DF at a truncation.

    Parameters
    ----------
    df : DistributionFunction
        The result of `compute_df`, with its rows.
    truncation : Truncation
        The truncation: a truncation energy ``k_t``, or a target mass or a
        target radius, for which `solve_target` finds the truncation
        energy.

    Returns
    -------
    model : Model
        The truncated model, with its profile, DF table, asymptotic forms
        and validation report.

    Raises
    ------
    UnphysicalModelError
        If `truncation_chain` raises `UnphysicalModelError`; the exception
        carries `df` as its result.
    TargetRefusal
        If no admissible model reaches the target mass or target radius.
    AccuracyError
        If a check of the model's validation report lies above
        :math:`10^{-2}` (the band "refused"), the round trip included, and
        the settings of `df` do not set ``write_anyway=True``; the exception
        carries the model as its result.
    UnphysicalModelError
        If the truncation energy is not above 0, or the Poisson solve
        reaches the solve's outer radius with the potential above
        ``K_FLOOR``, so that the truncation has no edge; the exception
        carries the model as its result.
    """
    if truncation.k_t is None:
        # A target mass or a target radius: the search finds the truncation
        # energy that reaches the target.
        from .target import _solve_target

        kind = "mass" if truncation.mass_t is not None else "radius"
        value = truncation.mass_t if kind == "mass" else truncation.r_t
        assert value is not None
        k_t, _, chain = _solve_target(
            df, kind, value, concentration=truncation.concentration
        )
    else:
        k_t, chain = float(truncation.k_t), None
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    # The target search returns the truncation chain at the root, which is
    # reused here; with a given k_t, the chain is computed here.
    if chain is None:
        chain = truncation_chain(df, k_t)
    made = truncated_table(df, k_t, chain)
    source, solve, validated = chain["source"], chain["solve"], chain["validated"]
    p_0 = 1.0 - k_t
    beta_c = chain["source_in"]["beta_c"]

    # The truncated DF's table and the table's checks.
    k, log_f, ddf = made["k"], made["log_f"], made["ddf"]
    with np.errstate(all="ignore"):
        slope = k * ddf * 10.0 ** (-log_f) / np.log(10.0)
    slope = np.where(np.isfinite(ddf) & np.isfinite(slope), slope, np.nan)
    finite = np.flatnonzero(np.isfinite(log_f))
    support = slope[finite[0] : finite[-1] + 1]
    table = DfTable(
        k=k[None, :],
        log_df=log_f[None, :],
        log_df_slope=slope[None, :],
        i_first=(int(finite[0]),),
        i_last=(int(finite[-1]),),
        slopes_exact=(bool(np.all(np.isfinite(support)) and ddf[finite[-1]] != 0.0),),
    )
    center = center_form(profile, aniso)
    top = {name: center[key] for name, key in _TOP_KEYS if key in center}
    checks = table_checks(k, log_f, ddf, p_0, top, edge_taylor=source.taylor)
    form = AsymptoticForm(
        p_0=p_0,
        central_label=(str(center["branch"]),),
        central_exponent=(float(center["kappa_sing"]),),
        central_amplitude=(float(center["amplitude"]),),
        tail_exponent=(0.0,),
        tail_amplitude=(0.0,),
        tail_log_exponent=(0.0,),
        central_form=(str(center["form"]),),
        central_log_coefficient=(float(center.get("log_coef", math.nan)),),
        central_next_form=(str(center["next_form"]),),
        central_next_exponent=(float(center["next_kappa"]),),
        central_next_coefficient=(float(center["next_coef"]),),
        tail_log_scale=(1.0,),
        below_floor=("taylor",) if source.taylor is not None else ("zero",),
        below_floor_coefficients=(
            tuple(float(v) for v in source.taylor) if source.taylor else (),
        ),
        central_anchor=(math.nan,),
        central_anchor_amplitude=(math.nan,),
        central_regular=((math.nan, math.nan),),
    )

    # The profile keeps the validated radii where the potential is at or
    # above K_FLOOR and the density is positive, and keeps the last validated
    # radius where the edge radius validated.r_t is finite.
    finite_model = k_t > 0.0 and bool(solve.event_hit)
    r_t = float(validated.r_t) if finite_model else math.inf
    mask = (validated.phi >= K_FLOOR) & (validated.rho > 0)
    if math.isfinite(validated.r_t):
        mask[-1] = True
    r, phi = validated.r[mask], validated.phi[mask]
    rho, mass = validated.rho[mask], validated.mass[mask]
    i = np.clip(np.searchsorted(validated.u_r, r), 0, validated.u_r.size - 1)
    u = validated.u[i] if np.array_equal(validated.u_r[i], r) else np.zeros(0)
    origin = validated.origin[mask] if validated.origin.size else np.zeros(0, np.int8)
    deficit = chain["rows"].central_deficit
    identity = model_identity(df.components, truncation, df.settings)
    truncated = TruncatedProfile(
        r=r,
        p_trunc=phi,
        rho_trunc=rho[None, :],
        mass_trunc=mass / (4.0 * np.pi),
        k_t=k_t,
        r_t=r_t,
        mass_t=float(mass[-1]) / (4.0 * np.pi) if finite_model else math.inf,
        spatially_finite=finite_model,
        source=profile,
        source_anisotropy=aniso,
        model=identity,
        edge_exponent=_edge_exponent_of(chain["gates"], k_t, beta_c)
        if finite_model
        else (),
        u_trunc=u,
        central_deficit=() if deficit is None else (tuple(float(v) for v in deficit),),
        origin=origin,
        df_k=np.asarray(table.k, dtype=np.float64),
        df_log_f=np.asarray(table.log_df, dtype=np.float64),
        df_slope=np.asarray(table.log_df_slope, dtype=np.float64),
        df_taylor=tuple(
            (float(c[0]), float(c[1]), float(c[2]))
            for rule, c in zip(
                form.below_floor, form.below_floor_coefficients, strict=True
            )
            if rule == "taylor"
        ),
    )

    # The validation report.
    common = {
        "top_form": Check(
            error=abs(float(checks.kappa_residual)), where=float(checks.k_top)
        ),
        "slope_column": Check(error=abs(float(checks.slope_column_excess)), where=None),
    }
    diag = source.taylor_check or {}
    common["edge_join"] = Check(
        error=max(
            abs(float(diag.get("splice_rel_diff", math.nan))),
            abs(float(diag.get("switch_rel_diff", math.nan))),
        ),
        where=None,
    )
    # The error of the df_clip check is the magnitude of the most negative
    # difference F(K + K_t) - F(K_t) that `truncated_source` set to zero as
    # rounding, divided by |F(K_t)|.
    clip_error, clip_at = source.clip
    common["df_clip"] = Check(
        error=float(clip_error), where=None if math.isnan(clip_at) else float(clip_at)
    )
    trunc_settings = dict(df.settings.truncation, log10_r_max=chain["log10_r_max"])
    common.update(profile_checks(truncated, table, form, aniso, trunc_settings))
    parents_level = df.validation.level if df.table.sign.size == 0 else 0
    report = ValidationReport(
        _MODEL_KEYS, {key: common[key] for key in _MODEL_KEYS}, parents_level
    )
    # The inversion record holds the top patch of the truncated table.
    over = made["override"]
    if over is not None:
        k_c = float(over["K_anchor"])
        r_c = 0.0
        if k_c < float(np.max(phi)):
            order = np.argsort(phi)
            r_c = float(np.interp(k_c, phi[order], r[order]))
        high = (k_c, r_c, int(over["rows"]), str(over["rule"]))
    else:
        high = (p_0, 0.0, 0, "none")
    inversion = None
    if df.inversion is not None:
        inversion = dataclasses.replace(
            df.inversion,
            high_k_override_k=(high[0],),
            high_k_override_r=(high[1],),
            high_k_override_rows=(high[2],),
            high_k_override_rule=(high[3],),
            low_k_override_k=(0.0,),
            low_k_override_r=(r_t,),
            deficit_degree=(int(chain["center"].deficit_degree),),
        )
    model = Model(
        components=df.components,
        truncation=truncation,
        settings=df.settings,
        identity=identity,
        parent=df.identity,
        table=table,
        form=form,
        profile=dataclasses.replace(truncated, model_level=report.level),
        validation=report,
        timing={},
        r_trust={},
        inversion=inversion,
    )
    # A model whose report is in the band "refused" (a check beyond 1e-2, or a
    # parent in that band) raises AccuracyError, unless the settings set
    # write_anyway. With write_anyway, the model is returned, the model's
    # report records the band, and `announce` states that the model is
    # returned on request.
    round_trip = common["round_trip"]
    if report.level >= 3 and not df.settings.write_anyway:
        raise AccuracyError(
            _refused_message("The truncated model", report, round_trip),
            result=model,
        )
    announce(
        f"The truncated model of {df.components[0].name!r}",
        report,
        "; ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at {check.where:.6g}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        + (
            " (returned on request, write_anyway: beyond 0.01)"
            if report.level >= 3
            else ""
        ),
    )
    if not finite_model:
        # The truncation energy is not above 0, or the Poisson solve reached
        # the solve's outer radius with the potential above K_FLOOR; either
        # way the truncation has no edge and the model is not a truncated
        # model. The logarithmic slope of the potential at the last
        # radius is -M / (r Phi), with M the enclosed mass.
        r_end, phi_end = float(r[-1]), float(phi[-1])
        slope_end = -float(mass[-1]) / (4.0 * np.pi) / (r_end * phi_end)
        raise UnphysicalModelError(
            f"NO EDGE: the solve reaches R = {r_end:.6e} with Phi = {phi_end:.6e} "
            f"and d ln Phi / d ln r = {slope_end:.4g}, above the floor: the "
            f"truncation at K_t = {k_t!r} has no edge within the solve's radius, so "
            "it is not a truncated model.",
            result=model,
        )
    return model


# The checks of an untruncated model. The join of the outer radii and the
# offset of the potential at large radii are not judged, because the profile
# is the closed form on every radius. The profile's checks
# (PROFILE_CHECK_KEYS) follow the two unjudged checks. The DF's report
# supplies the level that the model's report inherits.
_UNTRUNCATED_KEYS = ("outer_join", "eps", *PROFILE_CHECK_KEYS)
_CLOSED_FORM_OUTER_JOIN = (
    "the profile is the closed form on every row: no rows of a solve join it"
)
_CLOSED_FORM_EPS = (
    "the potential is the closed form: it has no offset at large radii to measure"
)


def untruncated_model(df: DistributionFunction) -> Model:
    r"""Return the untruncated model of an analytic DF that is a valid equilibrium.

    The model's profile is the analytic pair the DF was inverted from: the
    closed-form density, potential and mass on radii of the settings'
    radial grid, with the central depth from the profile's inner series
    inside the series' join radius. The model's table and forms are the
    DF's. The untruncated model is the case ``K_t = 0``. The model exists
    when the DF's lowest admissible truncation energy is zero
    (`DistributionFunction.is_physical`: no zero crossing, no global
    minimum above zero and no domain edge) and the profile's total mass is
    finite.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF of one component on a `DoublePowerLaw` profile.

    Returns
    -------
    model : Model
        The untruncated model, with its profile, the DF's table and forms,
        and a validation report.

    Raises
    ------
    UnphysicalModelError
        If `central_series_refusal` raises; `central_series_refusal` runs
        first.
    UnphysicalModelError
        If the DF is not physical as it stands. The message states the
        feature that sets the lowest admissible truncation energy, and the
        lowest admissible truncation energy.
    UnphysicalModelError
        If the profile's total mass diverges, at an outer slope
        :math:`\beta \le 3`.
    AccuracyError
        If a check of the model's validation report lies above
        :math:`10^{-2}` (the band "refused"), the round trip included, and
        the settings of `df` do not set ``write_anyway=True``; the exception
        carries the model as its result.

    Each `UnphysicalModelError` carries `df` as its result.
    """
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    central_series_refusal(df)
    if not df.is_physical:
        from .pair import potential_density_pair
        from .report import validate_df

        _, _, gates = validate_df(
            potential_density_pair(df.components), df.table, df.form
        )
        feature = {
            "K_0": f"the DF crosses zero at Q = {gates.k_0!r}",
            "Q_min": f"the DF has a global minimum at Q = {gates.q_min!r}",
            "domain_edge": "its support ends at the anisotropy's domain edge",
        }.get(gates.set_by, "it admits no truncation energy at zero")
        raise UnphysicalModelError(
            f"UNTRUNCATED: the DF is not a valid equilibrium as it stands: "
            f"{feature}, so its lowest admissible truncation energy is k_t_min = "
            f"{df.k_t_min!r} (set by {gates.set_by}); truncate at K_t >= k_t_min.",
            result=df,
        )
    mass_t = float(profile.mass_total)
    if not math.isfinite(mass_t):
        raise UnphysicalModelError(
            f"UNTRUNCATED MASS: the profile's total mass diverges (outer slope "
            f"beta = {float(profile.rho_beta):g} <= 3); give a truncation.",
            result=df,
        )
    settings = df.settings.truncation
    lo = float(settings["log10_r_min"])
    hi = log10_r_max(profile, float(settings["log10_r_max"]))
    per = int(settings["profile_rows_per_decade"])
    _, r_join, _ = series_join(profile)

    def rows_at(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(profile.p(r), dtype=np.float64)
        u = np.where(r <= r_join, np.asarray(inner_depth(profile, r)), 1.0 - p)
        return p, u

    # The radii are geometric in R. In addition, each sub-interval of the bins
    # of `row_bin_edges` (at P_0 = 1), in the potential and in the depth,
    # receives one radius at the sub-interval's center in log10, placed by log-log
    # interpolation of the geometric radii.
    r = np.logspace(lo, hi, int(round(per * (hi - lo))) + 1)
    p, u = rows_at(r)
    placed = [r]
    bins = row_bin_edges(1.0, settings)
    base = np.log10(bins.psi_edges[0])
    for value, side_top in ((p, bins.psi_top), (u, bins.u_top)):
        ok = np.isfinite(value) & (value >= K_FLOOR)
        x, y = np.log10(value[ok]), np.log10(r[ok])
        order = np.argsort(x)
        x, y = x[order], y[order]
        top = min(float(x[-1]), np.log10(side_top))
        k = np.arange(np.floor((x[0] - base) * per), np.ceil((top - base) * per))
        mid = base + (k + 0.5) / per
        mid = mid[(mid > x[0]) & (mid < top)]
        placed.append(10.0 ** np.interp(mid, x, y))
    r = np.unique(np.concatenate(placed))
    p, u = rows_at(r)
    # The profile keeps the radii where the potential is at or above K_FLOOR,
    # as a truncated model's profile does.
    r, p, u = r[p >= K_FLOOR], p[p >= K_FLOOR], u[p >= K_FLOOR]
    identity = model_identity(df.components, None, df.settings)
    model_profile = TruncatedProfile(
        r=r,
        p_trunc=p,
        rho_trunc=np.asarray(profile.rho(r), dtype=np.float64)[None, :],
        mass_trunc=np.asarray(profile.mass(r), dtype=np.float64),
        k_t=0.0,
        r_t=math.inf,
        mass_t=mass_t,
        spatially_finite=False,
        source=profile,
        source_anisotropy=aniso,
        model=identity,
        edge_exponent=(),
        u_trunc=u,
        df_k=np.asarray(df.table.k, dtype=np.float64),
        df_log_f=np.asarray(df.table.log_df, dtype=np.float64),
        df_slope=np.asarray(df.table.log_df_slope, dtype=np.float64),
    )
    checks = {
        "outer_join": Check(math.nan, reason=_CLOSED_FORM_OUTER_JOIN),
        "eps": Check(math.nan, reason=_CLOSED_FORM_EPS),
        **profile_checks(
            model_profile, df.table, df.form, aniso, dict(settings, log10_r_max=hi)
        ),
    }
    report = ValidationReport(
        _UNTRUNCATED_KEYS,
        {key: checks[key] for key in _UNTRUNCATED_KEYS},
        df.validation.level,
    )
    model = Model(
        components=df.components,
        truncation=None,
        settings=df.settings,
        identity=identity,
        parent=df.identity,
        table=df.table,
        form=df.form,
        profile=dataclasses.replace(model_profile, model_level=report.level),
        validation=report,
        timing={},
        r_trust={"silent": math.inf, "recorded": math.inf},
        inversion=df.inversion,
    )
    # A model whose report is in the band "refused" raises AccuracyError,
    # unless the settings set write_anyway.
    round_trip = checks["round_trip"]
    if report.level >= 3 and not df.settings.write_anyway:
        raise AccuracyError(
            _refused_message("The untruncated model", report, round_trip),
            result=model,
        )
    announce(
        f"The untruncated model of {component.name!r}",
        report,
        "; ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at {check.where:.6g}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        + (
            " (returned on request, write_anyway: beyond 0.01)"
            if report.level >= 3
            else ""
        ),
    )
    return model


def _refused_message(what: str, report: ValidationReport, round_trip: Check) -> str:
    """Return the AccuracyError message of a model in the band "refused".

    The message states the round trip first where the round trip is beyond
    0.01, then every other check beyond 0.01, and the parents' band where only
    the parents are in the band "refused".
    """
    parts = []
    if round_trip.level >= 3:
        parts.append(
            f"the model's DF does not rebuild its density: the round trip misses it "
            f"by {round_trip.error:.3e} at R = {round_trip.where}"
        )
    parts += [
        f"{key} {check.error:.3e}"
        + ("" if check.where is None else f" at {check.where:.6g}")
        for key, check in report.checks.items()
        if check.level >= 3 and key != "round_trip"
    ]
    if not parts:
        parts.append("the result it is built from is in the band 'refused'")
    return (
        f"{what} is beyond 0.01 (tier 'refused'): {'; '.join(parts)}. "
        "Settings(write_anyway=True) returns the model with its tier."
    )
