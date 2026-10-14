"""The truncated model of one component, from its untruncated DF.

The stages in order: the DF's rows as made, continued above their anchor by
the center's form (the base table); the gates on the rows; the truncated DF
and the auxiliary density it gives (`truncated_source`); the Poisson solve;
the density on the solve's rows and the center; the fills; the profile's rows
and their validation. Every number comes from the DF, the truncation and the
settings; nothing is read from a run.
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
from ..validation import GOAL, HARD_LIMIT, Check, ValidationReport, announce
from .center import center_amplitude
from .central import center_form
from .checks import table_checks
from .core import model_identity
from .derivatives import ExactDerivs
from .edge import density_edge_exponent
from .fill import fill_profile, profile_rows
from .form import tail_exponent_analytic
from .gates import truncation_gates
from .inputs import Truncation
from .poisson import profile_density, run_poisson_ode, source_slopes
from .profile_checks import PROFILE_CHECK_KEYS, _model_type, profile_checks
from .results import DistributionFunction, Model
from .stages import AsymptoticForm, DfTable
from .truncation import (
    CEILING_GAP,
    INNER_DEPTH,
    K_FLOOR,
    row_bin_edges,
    truncated_source,
)
from .validate import validate_rows

# the lowered center's nodes per decade
_CENTRE_NODES_PER_DECADE = 200
# the outer end of the radial grid: the bisection's upper bound in log10 R,
# its steps, and the potential it reaches, a half of the floor
_R_MAX_BOUND = 60.0
_R_MAX_STEPS = 200
_R_MAX_DEPTH = 0.5
# the extension of a patched table to its ceiling: its fewest and most energies
_EXTENSION_MIN = 500
_EXTENSION_MAX = 50000


def log10_r_max(profile: DoublePowerLaw, configured: float) -> float:
    """Return the radial grid's outer end: the configured one, or where the
    potential falls to half the floor if it lies above that there, found by
    bisection in log10 R and rounded up to a tenth."""
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return the table continued by its top patch's form from its last row to
    p_0 less the floor: at least 500 energies spaced in log(p_0 - K), those
    strictly between the last row and that ceiling."""
    ceiling = p_0 - K_FLOOR
    q_last = float(q[-1])
    if not ceiling > q_last + K_FLOOR:
        return q, f
    step = max(float(np.diff(q[-min(10, len(q) - 1) :])[0]), 1e-15)
    n = min(max(_EXTENSION_MIN, int((ceiling - q_last) / step)), _EXTENSION_MAX)
    u_lo = max(p_0 - ceiling, K_FLOOR)
    u_hi = max(p_0 - q_last, INNER_DEPTH)
    q_ext = p_0 - np.logspace(np.log10(u_lo), np.log10(u_hi), n)[::-1]
    q_ext = q_ext[(q_ext > q_last) & (q_ext < ceiling)]
    if q_ext.size == 0:
        return q, f
    f_ext = (
        a_anchor * (np.maximum(p_0 - q_ext, 1e-300) / max(u_anchor, 1e-300)) ** kappa
        + b
        + d * np.maximum(p_0 - q_ext, 0.0)
    )
    f_ext = np.where(np.isfinite(f_ext), f_ext, 0.0)
    return np.concatenate([q, q_ext]), np.concatenate([f, f_ext])


def base_table(df: DistributionFunction) -> tuple[np.ndarray, np.ndarray]:
    """Return the table the truncation reads: the rows as made up to the
    anchor, the center's form above it, and at beta_c = +-1/2 the exact DF
    wherever it is positive."""
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
        )
    component = df.components[0]
    aniso = component.anisotropy
    if not math.isfinite(aniso.r_a) and abs(abs(float(aniso.beta_c)) - 0.5) < 1e-9:
        order = 1 if float(aniso.beta_c) > 0.0 else 2
        derivs = ExactDerivs(
            component.unit_profile,
            beta_c=float(aniso.beta_c),
            model="constant",
            max_order=order + 1,
        )
        exact = np.zeros_like(q)
        inside = (q > 0.0) & (q < 1.0)
        exact[inside] = np.asarray(
            derivs.derivs_at_p(q[inside], order)[order], dtype=np.float64
        ) / (2.0 * np.pi**2)
        f = np.where(exact > 0.0, exact, f)
    return q, f


def refusals(df: DistributionFunction, k_t: float) -> None:
    """Refuse a truncation no model can carry, before any stage runs.

    - CENTRAL VALIDITY: gamma < 2 beta_c(0), with beta_c(0) = 0 for the
      isotropic and Osipkov-Merritt models: the DF is negative at the center
      (the cusp-slope-central-anisotropy theorem, An & Evans 2006); no K_t
      removes it.
    - CENTRAL SIGN: the central DF tends to minus infinity or to zero from
      below (a negative closed-form coefficient of a divergent center), or to
      a negative constant (F0 = F - A u^kappa' at the row nearest the top).

    A Type II model's bound, K_t above P(R_a), is the truncation gates'
    DOMAIN (the anisotropy's domain edge), not a refusal of its own.

    Raises
    ------
    UnphysicalModelError
        Naming the condition; it carries the DF as its result.
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
    """The untruncated potential at the anisotropy's domain edge: P(R_a) for
    a Type II model, 0 where the domain has no edge."""
    edge = aniso.domain_edge
    if not math.isfinite(edge):
        return 0.0
    return float(np.asarray(profile.p(np.array([edge])))[0])


def truncation_chain(df: DistributionFunction, k_t: float) -> dict[str, Any]:
    """Return every stage of the truncated model at the truncation energy
    k_t, with the inputs each stage was given.

    Raises
    ------
    UnphysicalModelError
        If a gate fires; it carries the DF as its result.
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

    # the gates, on the rows as made and on the base table
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

    # the truncated DF and the auxiliary density
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

    # the Poisson solve
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

    # the density on the solve's rows and the center
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
        n_per_decade=_CENTRE_NODES_PER_DECADE,
        goal=GOAL,
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
        n_per_decade=_CENTRE_NODES_PER_DECADE,
    )

    # the fills, the rows and their validation
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
    """The derivative order of the closed-form DF at beta_c = +-1/2 (1 for
    +1/2, 2 for -1/2); None for any other anisotropy."""
    if math.isfinite(aniso.r_a) or abs(abs(float(aniso.beta_c)) - 0.5) >= 1e-9:
        return None
    return 1 if float(aniso.beta_c) > 0.0 else 2


def _exact(component: Any, q: np.ndarray, order: int, derivative: bool) -> np.ndarray:
    """The closed-form DF at beta_c = +-1/2, rho_aux^(m) / (2 pi^2), or its
    K derivative, rho_aux^(m+1) / (2 pi^2); 0 (the DF) or NaN (the derivative)
    outside 0 < q < 1 and where it is not finite."""
    aniso = component.anisotropy
    n = order + 1 if derivative else order
    derivs = ExactDerivs(
        component.unit_profile,
        beta_c=float(aniso.beta_c),
        model="constant",
        max_order=order + 1,
    )
    out = np.full(q.size, np.nan) if derivative else np.zeros_like(q)
    inside = (q > 0.0) & (q < 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.asarray(derivs.derivs_at_p(q[inside], n)[2 if derivative else n])
    values = values.astype(np.float64) / (2.0 * np.pi**2)
    out[inside] = (
        np.where(np.isfinite(values), values, np.nan) if derivative else values
    )
    return out


def truncated_table(
    df: DistributionFunction, k_t: float, chain: dict[str, Any]
) -> dict[str, Any]:
    """Return the truncated DF's own table: K, F, log10 F and dF/dK on the
    source's grid, with the top patch the table carries above its anchor.

    F is the truncated source's, F(K + K_t) - F(K_t). Above the anchor the
    center's form fitted to it at P_0 = 1 - K_t replaces it, where the
    untruncated rows took a form; at beta_c = +-1/2 the exact shifted closed
    form replaces every row. dF/dK is the untruncated rows' derivative read at
    K + K_t: d ln F / d ln K by not-a-knot splines on each unbroken run of the
    rows (split at the interior point) times F from the source's own lookup;
    the edge's Lagrange interpolation at or below the splice; a four-node
    Lagrange within a factor 3 of a zero of F'; the closed form at
    beta_c = +-1/2; the patch's own derivative at and above its anchor.
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
    k, f = k[keep], f[keep]

    # the top patch at P_0 = 1 - K_t, where the untruncated rows took a form
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
                    k, f, p_0, patch.a_anchor, patch.u_anchor, kappa, patch.b, patch.d
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
    with np.errstate(divide="ignore"):
        log_f = np.where(f > 0.0, np.log10(np.maximum(f, 1e-320)), -np.inf)

    # dF/dK: the untruncated rows' derivative read at K + K_t
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
        # the edge rows, at or below the splice
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
        # the rows near a zero of F'
        q_z, d_z = q_s[ok], d_raw[ok]
        cross = np.flatnonzero(np.sign(d_z[1:]) * np.sign(d_z[:-1]) < 0.0)
        cross = cross[q_z[cross] >= 1e-3]
        if cross.size:
            near = np.zeros(qq.size, dtype=bool)
            for i in cross:
                q_0 = 0.5 * (q_z[i] + q_z[i + 1])
                near |= (qq >= q_0 / 3.0) & (qq <= 3.0 * q_0)
            if near.any():
                ddf[near] = np.asarray(_lagrange4(q_z, d_z, qq[near]), dtype=np.float64)
        # the closed form at beta_c = +-1/2
        if order is not None:
            d_x = _exact(component, np.where(inside, qq, q_d[0]), order, True)
            good = inside & np.isfinite(d_x)
            ddf[good] = d_x[good]
        # the patch's own derivative at and above its anchor (the last patch)
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


# the center's form under the table checks' names
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
# the checks of a truncated Method I model's report
_MODEL_KEYS = ("top_form", "slope_column", "edge_join", *PROFILE_CHECK_KEYS)


def _edge_exponent_of(gates: Any, k_t: float, beta_c: float) -> tuple[float, ...]:
    """The exponent of the density at the edge: F ~ K^p below K_t's Taylor
    form, p = 1 above the lowest admissible truncation energy; at it, 2 where
    the last minimum sets it, 1 where the last zero crossing does."""
    gate = float(gates.k_t_min)
    if k_t > gate:
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
    """Return the truncated model of a DF at a truncation energy.

    Parameters
    ----------
    df : DistributionFunction
        Result of `compute_df`, with its rows as made.
    truncation : Truncation
        The truncation: its energy k_t, or a target mass or radius, whose
        energy is solved for (`target.solve_target`).

    Returns
    -------
    model : Model

    Raises
    ------
    UnphysicalModelError
        If a gate fires; it carries the DF as its result.
    TargetRefusal
        If no admissible model reaches a target mass or radius.
    """
    if truncation.k_t is None:
        # a target mass or radius: the truncation energy that reaches it
        from .target import solve_target

        kind = "mass" if truncation.mass_t is not None else "radius"
        value = truncation.mass_t if kind == "mass" else truncation.r_t
        assert value is not None
        k_t, _ = solve_target(df, kind, value, concentration=truncation.concentration)
    else:
        k_t = float(truncation.k_t)
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    chain = truncation_chain(df, k_t)
    made = truncated_table(df, k_t, chain)
    source, solve, validated = chain["source"], chain["solve"], chain["validated"]
    p_0 = 1.0 - k_t
    beta_c = chain["source_in"]["beta_c"]

    # the table and its checks
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

    # the profile: the validated rows at or above the floor where rho > 0
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
        central_deficit=()
        if deficit is None
        else ((float(deficit[0]), float(deficit[1])),),
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

    # the report
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
    trunc_settings = dict(df.settings.truncation, log10_r_max=chain["log10_r_max"])
    common.update(profile_checks(truncated, table, form, aniso, trunc_settings))
    parents_level = df.validation.level if df.table.sign.size == 0 else 0
    report = ValidationReport(
        _MODEL_KEYS, {key: common[key] for key in _MODEL_KEYS}, parents_level
    )
    # the model's own DF against its own density: refused beyond the hard limit,
    # unless the settings ask for the model anyway, in which case the report
    # carries the verdict and the one channel says so
    round_trip = common["round_trip"]
    if round_trip.error > HARD_LIMIT and not df.settings.write_anyway:
        raise AccuracyError(
            f"The truncated model's DF does not rebuild its density: the round trip "
            f"misses it by {round_trip.error:.3e} at R = {round_trip.where}, beyond "
            f"the hard limit {HARD_LIMIT:g}."
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
            f" (returned on request, write_anyway: beyond the hard limit "
            f"{HARD_LIMIT:g})"
            if round_trip.error > HARD_LIMIT
            else ""
        ),
    )

    # the inversion record: the top patch of the truncated table
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
        )
    return Model(
        components=df.components,
        truncation=truncation,
        settings=df.settings,
        identity=identity,
        parent=df.identity,
        table=table,
        form=form,
        profile=truncated,
        validation=report,
        timing={},
        r_trust={},
        inversion=inversion,
    )


# the checks of an untruncated model: the join of its outer rows and the
# offset of its potential at large radii (both zero on the closed form its
# profile is), and the profile's own checks; its DF's report is the parent's
_UNTRUNCATED_KEYS = ("outer_join", "eps", *PROFILE_CHECK_KEYS)


def untruncated_model(df: DistributionFunction) -> Model:
    """Return the untruncated model of an analytic DF where it is a valid
    equilibrium.

    The model is the DF's own: its profile is the analytic pair it was
    inverted from (the closed-form density, potential and mass on rows of
    the settings' radial grid, the central depth from its series where that
    holds), its table and forms the DF's. It is the case K_t = 0 of the
    truncation's lower limit, so it exists where the DF admits no truncation
    energy above zero (`DistributionFunction.is_physical`: no zero crossing,
    no global minimum above zero, no domain edge) and the total mass is
    finite.

    Raises
    ------
    UnphysicalModelError
        Naming the condition (the feature that sets the DF's lowest
        admissible truncation energy, and that energy; or the mass), with
        the DF as its result.
    """
    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
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

    # rows geometric in R, and in each sub-interval of the row-count bins
    # (`truncation.row_bin_edges`, P_0 = 1) of the potential and of the depth,
    # a row at its center, placed by log-log interpolation of the radial rows
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
    # the rows at or above the floor, as on a truncated model
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
        "outer_join": Check(0.0),
        "eps": Check(0.0),
        **profile_checks(
            model_profile, df.table, df.form, aniso, dict(settings, log10_r_max=hi)
        ),
    }
    report = ValidationReport(
        _UNTRUNCATED_KEYS,
        {key: checks[key] for key in _UNTRUNCATED_KEYS},
        df.validation.level,
    )
    # the model's own DF against its own density: refused beyond the hard limit,
    # unless the settings ask for the model anyway
    round_trip = checks["round_trip"]
    if round_trip.error > HARD_LIMIT and not df.settings.write_anyway:
        raise AccuracyError(
            f"The untruncated model's DF does not rebuild its density: the round trip "
            f"misses it by {round_trip.error:.3e} at R = {round_trip.where}, beyond "
            f"the hard limit {HARD_LIMIT:g}."
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
            f" (returned on request, write_anyway: beyond the hard limit "
            f"{HARD_LIMIT:g})"
            if round_trip.error > HARD_LIMIT
            else ""
        ),
    )
    return Model(
        components=df.components,
        truncation=None,
        settings=df.settings,
        identity=identity,
        parent=df.identity,
        table=df.table,
        form=df.form,
        profile=model_profile,
        validation=report,
        timing={},
        r_trust={"goal": math.inf, "ceiling": math.inf},
        inversion=df.inversion,
    )
