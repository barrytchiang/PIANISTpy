"""The untruncated DF of one component, from its inputs alone.

The stages in order: the energy grid, the inversion's form, the exact
derivatives of rho_aux, the kernel's rows, their completion below the floor,
the anchor and the form above it, the nodes below the written floor and the
tail, then the table, the asymptotic form and its report. Every number comes
from the component and the settings; nothing is read from a run.

The families served are the Method I ones: isotropic, constant beta,
Osipkov-Merritt and Cuddeford-Osipkov-Merritt (COM), each of Type I and II.
The grid of an OM Type II or a COM model is placed around the feature its
pilot inversion finds (`pilot_feature`). A Type II DF is returned as the
inversion gives it, signed where it is negative; the refusal is the
truncation's.
"""

__all__ = ["SERVED", "inversion_df", "pilot_feature"]

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..exceptions import AccuracyError, InputError, ResultError
from ..runtime import log, timed
from ..settings import Settings
from ..validation import ValidationReport
from .central import center_form, top_limit
from .derivatives import ExactDerivs, aux_derivatives, host_reachable_radius
from .form import inversion_form as choose_form
from .form import tail_exponent_analytic
from .grid import K_FLOOR, OUTPUT_FLOOR_FACTOR, inversion_grid
from .inputs import Component
from .inversion import order_pair
from .pair import potential_density_pair
from .report import NO_SLOPE_COLUMN, slope_column_not_judged, validate_df
from .rows import (
    below_floor,
    complete_rows,
    floor_source,
    inversion_kernel,
    kernel_rules,
)
from .stages import (
    AsymptoticForm,
    DfTable,
    EnergyGrid,
    InversionRecord,
    InversionRows,
)
from .top import anchor, top_patch

# the anisotropy families the inversion serves from inputs alone
SERVED = ("isotropic", "constant", "OM_I", "OM_II", "COM_I", "COM_II")
# the families whose grid is placed around their pilot inversion's feature
_PILOTED = ("OM_II", "COM_I", "COM_II")
# the branches of the center whose DF diverges at p_0: the only ones patched
_PATCHED = ("regular", "critical_divergent")
_NAN = float("nan")
# the auxiliary density the inversion integrates beyond a Type II model's R_a,
# u = 1 - R^2 / R_a^2: Merritt's polynomial for OM, analytic through R_a, and
# the sign-preserving continuation for COM (`derivatives`)
_CONTINUATION = {
    "OM_II": "u = 1 - R^2/R_a^2, Merritt's polynomial",
    "COM_II": "sgn(u) |u|^(1 - beta_c), u = 1 - R^2/R_a^2",
}
# the pilot inversion (Pass 0): its coarse grid, 200 log nodes from the
# floor to 0.05, 600 linear ones to 0.95 and 200 log ones to the top; the
# rows it drops first; the B/BC and BC/C boundaries of the preflight zones,
# whose midpoint cuts its rows; the DF floor relative to their median
_PILOT_N = (200, 600, 200)
_PILOT_DROP = 2
_PILOT_FLOOR = 1e-3


def _rows(
    component: Component,
    settings: Settings,
    pair: Any,
    q: np.ndarray,
    sources: Any = None,
) -> tuple[Any, ...]:
    """The inversion's rows on the energies q: the form, the exact sources,
    the kernel's rows and their completion below the floor. ``sources``
    (pair, grid, form) -> AuxDerivatives replaces `aux_derivatives`: the
    gradient's rule applies the same operator to other rows (`gradient`)."""
    profile, aniso = component.unit_profile, component.anisotropy
    family = aniso.family
    grid = EnergyGrid(
        k=q[None, :],
        r_of_k=np.full((1, q.size), np.nan),
        k_floor=(float(q[0]),),
        k_ceiling=(float(q[-1]),),
    )
    form, (frec,) = choose_form(pair, grid)
    aux = (aux_derivatives if sources is None else sources)(pair, grid, form)
    is_om = family.startswith("OM")
    beta_c = 0.0 if is_om else float(aniso.beta_c)
    pairs = order_pair(beta_c)
    by_order = {order: aux.values[0, i] for i, order in enumerate(aux.orders)}
    at_floor = dict(zip(aux.orders, aux.at_floor[0], strict=True))
    j = int(form.order_j[0])
    order = j if pairs.order_n == 0.0 else j + 1
    g1, g2 = by_order[order], by_order[order + 1]
    if not np.all(np.isfinite(g2)):
        raise ResultError(
            f"The source's slope rho_aux^({order + 1}) is not finite on the grid."
        )
    common: dict[str, Any] = dict(
        beta_c=beta_c,
        is_om=is_om,
        rules=kernel_rules(settings),
        dm_j=float(at_floor[j]),
        p_0=1.0,
        g0=jnp.asarray(by_order[j], dtype=jnp.float64),
        dm_jm1=float(at_floor[j - 1]) if j - 1 in at_floor else 0.0,
        switched=bool(pairs.order_j == pairs.order_m + 1),
        p_s=float(frec.k_interior),
        s=float(frec.interior_exponent),
        outer_from=float(frec.outer_from),
    )
    qj = jnp.asarray(q, dtype=jnp.float64)
    g1j = jnp.asarray(g1, dtype=jnp.float64)
    g2j = jnp.asarray(g2, dtype=jnp.float64)
    half = pairs.kernel_exponent == 0.0
    # the exact slopes of the source where the source is the exact derivative
    # (unswitched rows, not a half-integer closed form); else the kernel's PCHIP
    # slopes of the table
    slope = (
        None
        if common["switched"] or half
        else jnp.asarray(np.where(g1 != 0.0, g2, 0.0))
    )
    f_k = np.asarray(
        inversion_kernel(qj, g1j, g1_slope=slope, **common), dtype=np.float64
    )
    ddf_k = np.asarray(
        inversion_kernel(qj, g2j if half else g1j, deriv=True, g2=g2j, **common),
        dtype=np.float64,
    )
    floor = floor_source(
        aux,
        frec,
        profile,
        aniso,
        float(np.min(q)),
        int(settings.inversion["n_floor_nodes"]),
    )
    f, ddf = complete_rows(q, f_k, ddf_k, edge=None, floor=floor, m2_tail=None)
    return f, ddf, frec, aux, floor


def _below_limit(q: np.ndarray, component: Component) -> np.ndarray:
    """The energies q whose depth u = 1 - q exceeds the shallower of
    `top_limit` and the depth at `host_reachable_radius`, where that lies
    above the grid's top: the table's top is then the first node above it
    and the center's closed form covers the energies above; q unchanged
    otherwise."""
    profile = component.unit_profile
    u_ask = top_limit(profile, component.anisotropy)
    r_floor = host_reachable_radius()
    u_floor = float(profile.depth_inner(np.array([r_floor]), terms=8)[0][0])
    u_min = max(u_ask, u_floor)
    if not u_min > 1.0 - float(q[-1]):
        return q
    kept = q[1.0 - q > u_min]
    if u_floor > u_ask:
        log.info(
            "The table of %r would reach the depth u = %.6e, below the depth "
            "u = %.6e at R = %.6e, the smallest radius whose square host "
            "float64 holds: its table ends at u = %.6e, the center's closed "
            "form above it.",
            component.name,
            max(u_ask, 1.0 - float(q[-1])),
            u_floor,
            r_floor,
            1.0 - float(kept[-1]),
        )
    else:
        log.info(
            "The exact derivatives of %r exceed float64 below the depth u = %.6e "
            "(the model's inner series): its table ends at u = %.6e, the center's "
            "closed form above it.",
            component.name,
            u_ask,
            1.0 - float(kept[-1]),
        )
    return kept


def pilot_feature(component: Component, settings: Settings) -> dict[str, Any]:
    """Return the feature an OM Type II or COM model's grid is placed
    around: Pass 0, the inversion's rows on a coarse grid and the
    minimum of the DF below the midpoint of the preflight zones' B/BC and
    BC/C boundaries, refined by a parabola through its neighbors.

    Parameters
    ----------
    component : Component
        The component.
    settings : Settings
        The numerical settings.

    Returns
    -------
    feature : dict
        ``q_feature`` (0 where no minimum is found: the standard zones).
    """
    extent = float(settings.inversion["grid_tail_extent"])
    n_lo, n_mid, n_hi = _PILOT_N
    low = np.logspace(np.log10(K_FLOOR), np.log10(0.05), n_lo)
    # the log part ends one ulp below 0.05; the seam is the linear part's
    # first node exactly, so the two are one node
    low[-1] = 0.05
    q = np.unique(
        np.concatenate([low, np.linspace(0.05, 0.95, n_mid), 1.0 - low[::-1]])
    )
    q = _below_limit(q, component)
    f = _rows(component, settings, potential_density_pair((component,)), q)[0]
    df, qq = np.asarray(f)[_PILOT_DROP:], q[_PILOT_DROP:]
    cutoff = 0.5 * (5.0 / 8.0 + (1.0 - extent))
    pos = df[(qq < cutoff) & (df > 0)]
    df_floor = float(np.median(pos)) * _PILOT_FLOOR if len(pos) >= 3 else 0.0
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


def _tabulate(
    function: Callable[[np.ndarray, np.ndarray], Any], q: np.ndarray
) -> np.ndarray:
    """A function's values F on the grid's energies q, called with q and their
    depth ``P_0 - q`` (``P_0 = 1``; exact in float64 for ``q >= P_0 / 2``),
    refused where not finite."""
    f = np.asarray(function(q.copy(), 1.0 - q), dtype=np.float64)
    if f.shape != q.shape:
        raise InputError(
            f"The function returned shape {f.shape} for the {q.size} energies of "
            "the grid: it takes the arrays K and P_0 - K and returns F at each."
        )
    bad = ~np.isfinite(f)
    if np.any(bad):
        raise InputError(
            f"The function's DF is not finite at K = {float(q[bad][0]):.17g}, the "
            f"first of {int(np.count_nonzero(bad))} such energies on the grid."
        )
    return f


def _nodes_below_floor(
    component: Component, settings: Settings, pair: Any, q: np.ndarray
) -> np.ndarray:
    """The energies of the nodes below the written floor that `compute_df`
    places for the same component and settings on the grid q (the
    completion's own log grid and its derived bottom, `below_floor`); empty
    where it places none."""
    grid = EnergyGrid(
        k=q[None, :],
        r_of_k=np.full((1, q.size), np.nan),
        k_floor=(float(q[0]),),
        k_ceiling=(float(q[-1]),),
    )
    form, (frec,) = choose_form(pair, grid)
    aux = aux_derivatives(pair, grid, form)
    floor = floor_source(
        aux,
        frec,
        component.unit_profile,
        component.anisotropy,
        float(np.min(q)),
        int(settings.inversion["n_floor_nodes"]),
    )
    if floor is None:
        return np.empty(0)
    k_written = float(np.min(q[q >= OUTPUT_FLOOR_FACTOR * K_FLOOR]))
    made = below_floor(aux, floor, k_written)
    if made.get("not_positive"):
        return np.empty(0)
    return np.asarray(made["K"], dtype=np.float64)


def inversion_df(
    component: Component,
    settings: Settings,
    function: Callable[[np.ndarray, np.ndarray], Any] | None = None,
) -> tuple[
    EnergyGrid,
    DfTable,
    AsymptoticForm,
    float,
    ValidationReport,
    InversionRecord | None,
    InversionRows | None,
]:
    """Return the untruncated DF of one component, from its inputs alone.

    Parameters
    ----------
    component : Component
        The component.
    settings : Settings
        The numerical settings.
    function : callable, optional
        A DF given as a function f(k, depth) of the energy and its depth
        ``P_0 - k``: its values on the grid, and on the nodes below the written
        floor the inversion places, take the place of the inversion's rows
        and its completion, and everything after them runs unchanged. The
        table then has no exact slopes, and there is no inversion record and
        no rows.

    Returns
    -------
    grid, table, form, k_t_min, report, inversion, rows
        The energy grid (with r(K) from the analytic potential), the table,
        the asymptotic form, the lowest admissible truncation energy, the
        report of `validate_df`, the inversion's record and its rows as made
        on its whole grid (None for a function).

    Raises
    ------
    NotImplementedError
        If the component's anisotropy family is not served yet.
    InputError
        If the function is not finite somewhere on the grid.
    """
    family = component.anisotropy.family
    if family not in SERVED:
        raise NotImplementedError(
            f"The inversion from inputs serves the families {', '.join(SERVED)}; "
            f"the component {component.name!r} is {family}."
        )
    profile, aniso = component.unit_profile, component.anisotropy
    with timed("potential_density_pair"):
        pair = potential_density_pair((component,))

    # the grid, around the pilot's feature where the family has one, and the rows
    with timed("energy_grid"):
        feature = pilot_feature(component, settings) if family in _PILOTED else None
        q, _ = inversion_grid(profile, aniso, settings, feature=feature)
        q = _below_limit(q, component)
    below = None
    if function is not None:
        # the function's values in place of the inversion's rows, and on the
        # nodes below the written floor the inversion places
        with timed("tabulate_function"):
            k_below = _nodes_below_floor(component, settings, pair, q)
            k_all = np.unique(np.concatenate([k_below, q]))
            f_all = _tabulate(function, k_all)
            f = f_all[np.searchsorted(k_all, q)]
            if k_below.size:
                below = dict(K=k_below, F=f_all[np.searchsorted(k_all, k_below)])
            ddf, frec, aux, floor = None, None, None, None
    else:
        with timed("invert_rows"):
            f, ddf, frec, aux, floor = _rows(component, settings, pair, q)

    with timed("asymptotic_form"):
        pieces = _table_pieces(component, settings, q, f, ddf, aux, floor, below)
        at, patch, data_top = pieces["at"], pieces["patch"], pieces["data_top"]
        table, as_form = pieces["table"], pieces["form"]
    with timed("validate_df"):
        k_t_min, report, _ = validate_df(pair, table, as_form)
    if function is not None:
        # a function states no dF/dK column: nothing to judge, and said so
        report = slope_column_not_judged(report, table, NO_SLOPE_COLUMN)

    # r(K) from the analytic potential, and the inversion's record
    radii = ExactDerivs(profile, max_order=1)
    k_t = np.asarray(table.k[0], dtype=np.float64)
    r_of_k = np.asarray(radii.r_of_p(k_t), dtype=np.float64)
    out_grid = EnergyGrid(
        k=k_t[None, :],
        r_of_k=r_of_k[None, :],
        k_floor=(float(k_t[0]),),
        k_ceiling=(float(k_t[-1]),),
    )
    if frec is None:
        return out_grid, table, as_form, k_t_min, report, None, None
    if at is not None and patch is not None:
        r_c = float(radii.r_of_p(np.array([at.k_anchor]))[0])
        high = (at.k_anchor, r_c, at.rows, at.rule)
    else:
        high = (1.0, 0.0, 0, "none")
    inversion = InversionRecord(
        form=(str(frec.form),),
        order_j=(int(frec.order_j),),
        order_n=(float(frec.kernel_exponent),),
        floor_m=(int(frec.order_m),),
        floor_n=(float(frec.order_n),),
        k_interior=(float(frec.k_interior),),
        interior_exponent=(float(frec.interior_exponent),),
        outer_power=(float(frec.outer_power),),
        outer_end=(str(frec.outer_end),),
        high_k_override_k=(float(high[0]),),
        high_k_override_r=(float(high[1]),),
        high_k_override_rows=(int(high[2]),),
        high_k_override_rule=(str(high[3]),),
        low_k_override_k=(0.0,),
        low_k_override_r=(math.inf,),
        method2_bands=(),
        critical_band=(),
        continuation=(_CONTINUATION.get(family, "none"),),
    )
    rows = InversionRows(
        k=np.asarray(q, dtype=np.float64)[None, :],
        f=np.asarray(f, dtype=np.float64)[None, :],
        ddf=np.asarray(ddf, dtype=np.float64)[None, :],
        data_top=(data_top,),
    )
    return out_grid, table, as_form, k_t_min, report, inversion, rows


def _table_pieces(
    component: Component,
    settings: Settings,
    q: np.ndarray,
    f: np.ndarray,
    ddf: np.ndarray | None,
    aux: Any,
    floor: dict[str, Any] | None,
    below: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The table and the forms beyond it from the rows: the anchor and the
    form above it, the nodes below the written floor and the tail, then the
    table and the asymptotic form; each piece returned with them (the
    gradient's rule reads them, `gradient`). ``below`` gives the nodes below
    the written floor as made elsewhere (a function's ``K`` and ``F``, no
    slopes), in place of the completion's."""
    profile, aniso = component.unit_profile, component.anisotropy
    # the anchor and the form above it: none for Type II, whose rows carry no
    # form above them (3o)
    center = center_form(profile, aniso)
    kappa = float(center["kappa_sing"]) if center["branch"] in _PATCHED else 0.0
    r_min = 10.0 ** float(settings.truncation["log10_r_min"])
    data_top = float(np.asarray(profile.p(r_min)))
    at = None if aniso.kind == "II" else anchor(q, f, 1.0, kappa, center, data_top, ddf)
    patch = None
    if at is not None:
        amplitude = float(center["amplitude"])
        closed = (
            center["form"] == "power"
            and math.isfinite(amplitude)
            and abs(float(center["kappa_sing"]) - kappa) <= 1e-12
        )
        below_anchor = q <= at.k_anchor
        patch = top_patch(
            q[below_anchor],
            f[below_anchor],
            1.0,
            kappa,
            amplitude if closed else None,
            q[~below_anchor],
        )

    # the nodes below the written floor and the tail
    k_written = float(np.min(q[q >= OUTPUT_FLOOR_FACTOR * K_FLOOR]))
    if floor is not None:
        made = below_floor(aux, floor, k_written)
        below = None if made.get("not_positive") else made
    a = tail_exponent_analytic(profile, aniso)
    tail = None
    if a > 0.0:
        if below is not None and below.get("dFdK") is None:
            # no slope below the floor (a function's): the tail's power through
            # its lowest node, as through the written floor below
            kb0, fb0 = float(below["K"][0]), float(below["F"][0])
            tail = dict(A=fb0 * kb0**a, a=a, k=0.0, E0=1.0)
        elif below is not None:
            kb0, fb0, db0 = (
                float(below["K"][0]),
                float(below["F"][0]),
                float(below["dFdK"][0]),
            )
            lnb = float(np.log(1.0 / kb0))
            k_log = float((kb0 * db0 / fb0 + a) * lnb)
            if not np.isfinite(k_log):
                err = AccuracyError(
                    f"the tail's logarithmic exponent below the floor is {k_log!r} "
                    f"(K {kb0:.6e}, F {fb0:.6e}, dF/dK {db0:.6e}): refused"
                )
                err.result = below
                raise err
            tail = dict(A=fb0 * kb0**a * lnb**k_log, a=a, k=k_log, E0=1.0)
        else:
            fw = float(f[np.flatnonzero(q == k_written)[0]])
            tail = dict(A=fw * k_written**a, a=a, k=0.0, E0=1.0)
        if not tail["A"] > 0.0:
            tail = None

    # the support: a Type II DF is returned above P(R_a) only, with nothing
    # below its first row there (its rows below are the continuation's)
    edge = aniso.domain_edge
    k_edge = (
        float(np.asarray(profile.p(np.array([edge])))[0])
        if math.isfinite(edge)
        else 0.0
    )
    if k_edge > 0.0:
        below, tail = None, None
    table = _table(q, f, ddf, patch, below, k_edge)
    as_form = _form(center, tail, patch)
    return dict(
        center=center,
        kappa=kappa,
        data_top=data_top,
        at=at,
        patch=patch,
        k_written=k_written,
        below=below,
        tail=tail,
        k_edge=k_edge,
        table=table,
        form=as_form,
    )


def _table(
    q: np.ndarray,
    f: np.ndarray,
    ddf: np.ndarray | None,
    patch: Any,
    below: dict[str, Any] | None,
    k_edge: float = 0.0,
) -> DfTable:
    """The inversion's own table: the rows as made, to the anchor, joined
    below to the nodes below the written floor. Where the anisotropy's
    domain has an edge (a Type II model, ``k_edge = P(R_a) > 0``), the DF is
    returned over its support K > k_edge only: the rows at or below it stay
    on the grid as NaN and the table's first row is the first above it."""
    anchor_k = patch.k_anchor if patch is not None else np.inf
    k_w = float(np.max(below["K"])) if below is not None else -np.inf
    outside = q <= k_edge
    window = (q <= anchor_k) & (q >= k_w * (1.0 - 1e-12))
    bad = window & ~outside & ~(np.isfinite(f) & (f != 0.0))
    if bad.any():
        err = AccuracyError(
            f"{int(bad.sum())} rows of the inversion inside the table are not finite "
            f"or zero, the first at K = {float(q[bad][0]):.6e}: refused"
        )
        err.result = dict(k=q, f=f)
        raise err
    keep = outside | window
    q, f, outside = q[keep], f[keep], outside[keep]
    inside = ~outside
    with np.errstate(divide="ignore", invalid="ignore"):
        log_df = np.where(inside, np.log10(np.abs(f)), np.nan)
        slope = (
            np.full(q.shape, np.nan)
            if ddf is None
            else np.where(
                inside, q * np.asarray(ddf)[keep] / (f * np.log(10.0)), np.nan
            )
        )
    extra: dict[str, Any] = {}
    if np.any(f[inside] < 0.0):
        extra["sign"] = np.where(inside, np.sign(f), 0).astype(np.int8)[None, :]
    exact = bool(np.all(np.isfinite(slope[inside])))
    if below is not None:
        kb = np.asarray(below["K"], dtype=np.float64)
        fb = np.asarray(below["F"], dtype=np.float64)
        order = np.argsort(kb)
        kb, fb = kb[order], fb[order]
        db = below.get("dFdK")
        sb = (
            np.full(kb.shape, np.nan)
            if db is None
            else kb * np.asarray(db, dtype=np.float64)[order] / (fb * np.log(10.0))
        )
        exact = exact and bool(np.all(np.isfinite(sb)))
        extra.update(
            k_below=kb[None, :],
            log_df_below=np.log10(fb)[None, :],
            log_df_slope_below=sb[None, :],
            n_below=(int(kb.size),),
        )
    return DfTable(
        k=q[None, :],
        log_df=log_df[None, :],
        log_df_slope=slope[None, :],
        i_first=(int(np.flatnonzero(inside)[0]),),
        i_last=(int(q.size) - 1,),
        slopes_exact=(exact,),
        **extra,
    )


def _form(
    center: dict[str, Any], tail: dict[str, float] | None, patch: Any
) -> AsymptoticForm:
    """The forms beyond the table: the center's closed form, with the anchor
    and the regular part where the form takes over above the rows, and below
    the floor the tail where the DF has one, else zero."""
    fields: dict[str, Any] = dict(
        p_0=1.0,
        central_label=(str(center["branch"]),),
        central_exponent=(float(center["kappa_sing"]),),
        central_amplitude=(float(center["amplitude"]),),
        central_form=(str(center["form"]),),
        central_log_coefficient=(float(center.get("log_coef", _NAN)),),
        central_next_form=(str(center["next_form"]),),
        central_next_exponent=(float(center["next_kappa"]),),
        central_next_coefficient=(float(center["next_coef"]),),
        central_anchor=(_NAN,),
        central_anchor_amplitude=(_NAN,),
        central_regular=((_NAN, _NAN),),
        below_floor=("zero",),
        below_floor_coefficients=((),),
        tail_exponent=(0.0,),
        tail_amplitude=(0.0,),
        tail_log_exponent=(0.0,),
        tail_log_scale=(1.0,),
    )
    if tail is not None:
        fields.update(
            below_floor=("tail",),
            tail_exponent=(float(tail["a"]),),
            tail_amplitude=(float(tail["A"]),),
            tail_log_exponent=(float(tail["k"]),),
            tail_log_scale=(float(tail["E0"]),),
        )
    if patch is not None:
        fields.update(
            central_anchor=(patch.k_anchor,),
            central_anchor_amplitude=(patch.a_anchor,),
            central_regular=((patch.b, patch.d),),
        )
    return AsymptoticForm(**fields)
