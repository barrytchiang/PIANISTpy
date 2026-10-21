"""Untruncated DF of one component, inverted from the component's inputs alone.

`inversion_df` inverts the potential-density pair of one component into a
DF table, the closed forms beyond the table's two ends and the DF's
validation report. The inversion runs in this order: the energy grid, the
choice of the inversion's form, the exact derivatives of the auxiliary
density rho_aux, the kernel's rows, the completion of the rows below the
floor, the anchor and the closed form above the anchor, the nodes below
the written floor and the tail, and finally the table, the asymptotic form
and the report. Every number comes from the component and the settings.

`SERVED` lists the families the inversion serves: isotropic, constant
anisotropy, Osipkov-Merritt (OM) and Cuddeford-Osipkov-Merritt (COM), with
OM and COM each of Type I and Type II. The grid of an OM Type II or a COM
model is placed around the feature that a pilot inversion of the model
finds (`pilot_feature`). A DF that is negative somewhere is returned with
its sign. The gates of a truncation energy decide whether a truncation of
a DF with negative values is admissible.
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
from .central import (
    LOG10_DF_CEILING,
    PATCHED_BRANCHES,
    ceiling_limit,
    center_form,
    top_limit,
)
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

# The anisotropy families that `inversion_df` inverts from the inputs alone.
SERVED = ("isotropic", "constant", "OM_I", "OM_II", "COM_I", "COM_II")
# The families whose grid is placed around the feature of a pilot inversion.
_PILOTED = ("OM_II", "COM_I", "COM_II")
# The branches of the center's closed form whose DF diverges at P_0. For a
# branch in _PATCHED the closed form above the anchor uses the singular
# exponent kappa_sing; for every other branch the closed form above the
# anchor uses the exponent 0.
_PATCHED = PATCHED_BRANCHES
_NAN = float("nan")
# The record of the auxiliary density that the inversion integrates beyond a
# Type II model's anisotropy radius R_a, with u = 1 - R^2 / R_a^2. For OM Type
# II the auxiliary density is Merritt's polynomial in u, analytic through R_a.
# For COM Type II the auxiliary density is the sign-preserving continuation
# sgn(u) |u|^(1 - beta_c) (`derivatives`).
_CONTINUATION = {
    "OM_II": "u = 1 - R^2/R_a^2, Merritt's polynomial",
    "COM_II": "sgn(u) |u|^(1 - beta_c), u = 1 - R^2/R_a^2",
}
# The pilot inversion's coarse grid has 200 nodes log-spaced in Q from K_FLOOR
# to 0.05, 600 linear nodes from 0.05 to 0.95, and 200 nodes from 0.95 to
# 1 - K_FLOOR log-spaced in 1 - Q. The pilot drops its first _PILOT_DROP rows
# before the search for the DF's minimum. The search skips every DF value at
# or below _PILOT_FLOOR times the median of the positive DF values below the
# search's upper energy.
_PILOT_N = (200, 600, 200)
_PILOT_DROP = 2
_PILOT_FLOOR = 1e-3


def _rows(
    component: Component,
    settings: Settings,
    pair: Any,
    q: np.ndarray,
    sources: Any = None,
    deriv: bool = True,
) -> tuple[Any, ...]:
    """Return the inversion's rows on the energies q.

    The rows are the kernel's values of F and dF/dQ on q, completed below
    the floor (`complete_rows`). The function chooses the inversion's form,
    computes the exact derivatives of rho_aux, calls the kernel and adds
    the completion.

    Parameters
    ----------
    component : Component
        The component to invert.
    settings : Settings
        Numerical settings; the kernel's rules and ``n_floor_nodes`` of the
        ``inversion`` group are read.
    pair : PotentialDensityPair
        The component's potential-density pair.
    q : numpy.ndarray
        Energies of the rows, increasing, in internal units.
    sources : callable, optional
        A function (pair, grid, form) -> AuxDerivatives that replaces
        `aux_derivatives`. The gradient's rule (`gradient`) passes such a
        function to apply the inversion's operator to other source rows.
        Default: ``None``,
        which means `aux_derivatives`.
    deriv : bool, optional
        Default: ``True``, which means the rows' dF/dQ is computed. With
        ``False``, dF/dQ is not computed and None stands in its place. F
        does not depend on `deriv`, because the kernel computes F and dF/dQ
        in separate calls.

    Returns
    -------
    f : numpy.ndarray
        F on q.
    ddf : numpy.ndarray or None
        dF/dQ on q, or None where `deriv` is false.
    frec : object
        The record of the inversion's form for the component.
    aux : AuxDerivatives
        The derivatives of rho_aux on q.
    floor : dict or None
        The source of the completion below the floor (`floor_source`).

    Raises
    ------
    ResultError
        If the source's slope rho_aux^(order + 1) is not finite on q, with
        order the derivative order j of the inversion's form, plus 1 where
        the kernel exponent n is not 0.
    """
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
    # The kernel reads the exact slope rho_aux^(order + 1) of its source g1
    # unless the derivative order j is switched to m + 1 or the kernel
    # exponent is 0 (beta_c a half-integer). In those two cases the slope is
    # None and the kernel uses PCHIP slopes of g1.
    slope = (
        None
        if common["switched"] or half
        else jnp.asarray(np.where(g1 != 0.0, g2, 0.0))
    )
    f_k = np.asarray(
        inversion_kernel(qj, g1j, g1_slope=slope, **common), dtype=np.float64
    )
    ddf_k = (
        np.asarray(
            inversion_kernel(qj, g2j if half else g1j, deriv=True, g2=g2j, **common),
            dtype=np.float64,
        )
        if deriv
        else None
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
    """Return the energies q without the energies above the center's depth limit.

    Three depths u = 1 - q limit the table at the center: `top_limit`, the
    depth below which the exact derivatives of rho_aux exceed float64; the
    depth at `host_reachable_radius`, the smallest radius whose square
    float64 holds; and `ceiling_limit`, the depth below which the center's
    closed form exceeds the DF's ceiling. The limit u_min is the largest
    of the three depths. Where u_min exceeds the depth 1 - q[-1] of the last
    energy, the call keeps only the energies whose depth exceeds u_min,
    logs which depth set the cut, and the center's closed form covers the
    energies above the kept ones. Otherwise the call returns q unchanged.
    """
    profile = component.unit_profile
    u_ask = top_limit(profile, component.anisotropy)
    r_floor = host_reachable_radius()
    u_floor = float(profile.depth_inner(np.array([r_floor]), terms=8)[0][0])
    u_ceiling = ceiling_limit(profile, component.anisotropy)
    u_min = max(u_ask, u_floor, u_ceiling)
    if not u_min > 1.0 - float(q[-1]):
        return q
    kept = q[1.0 - q > u_min]
    if u_ceiling > max(u_ask, u_floor):
        log.info(
            "The DF of %r exceeds the package's ceiling 10^%g below the depth "
            "u = %.6e (the center's closed form): its table ends at u = %.6e, "
            "the closed form above it.",
            component.name,
            LOG10_DF_CEILING,
            u_ceiling,
            1.0 - float(kept[-1]),
        )
    elif u_floor > u_ask:
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
    """Return the energy an OM Type II or COM model's grid is placed around.

    The pilot inversion computes the DF on a coarse grid (`_PILOT_N`),
    without dF/dQ. The search covers the energies below a cutoff, the
    midpoint of 5/8 and 1 - ``grid_tail_extent`` (the zone B/BC and zone
    BC/C boundaries of `grid.standard_zones`). The search skips the first
    ``_PILOT_DROP`` nodes and every DF value at or below ``_PILOT_FLOOR``
    times the median of the positive DF values below the cutoff. The
    feature is the DF's minimum over the searched nodes, refined to the
    vertex of the parabola through the minimum node and its two neighbors
    and clipped to the two neighbors' energies. With :math:`(Q_a, F_a)`,
    :math:`(Q_b, F_b)` and :math:`(Q_c, F_c)` the three nodes, the feature
    is the minimum node itself where
    :math:`|2[(F_b - F_a)(Q_c - Q_b) - (F_c - F_b)(Q_b - Q_a)]|` is at most
    0.01 times the largest difference of two of the three DF values times
    :math:`Q_c - Q_a`.

    Parameters
    ----------
    component : Component
        The component to invert.
    settings : Settings
        Numerical settings; ``grid_tail_extent`` of the ``inversion``
        group is read.

    Returns
    -------
    feature : dict
        The key ``q_feature``, the feature's energy in internal units. The
        value is 0, which selects the standard zones, where fewer than 3
        nodes are searched, where the feature lies below :math:`10^{3}` times
        ``K_FLOOR``, or where the feature is at or above the cutoff.
    """
    extent = float(settings.inversion["grid_tail_extent"])
    n_lo, n_mid, n_hi = _PILOT_N
    low = np.logspace(np.log10(K_FLOOR), np.log10(0.05), n_lo)
    # np.logspace can end one ulp away from 0.05. Setting the last log node to
    # exactly 0.05 makes the last log node equal to the first linear node,
    # and np.unique then keeps one node at 0.05.
    low[-1] = 0.05
    q = np.unique(
        np.concatenate([low, np.linspace(0.05, 0.95, n_mid), 1.0 - low[::-1]])
    )
    q = _below_limit(q, component)
    f = _rows(
        component, settings, potential_density_pair((component,)), q, deriv=False
    )[0]
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
    """Return a user-specified DF function's values F on the energies q.

    The function is called with q and the depth :math:`P_0 - q`, where the
    central potential :math:`P_0` is 1. The depth 1 - q is exact in float64
    for q at or above 1/2.

    Raises
    ------
    InputError
        If the returned array's shape differs from the shape of q.
    InputError
        If a returned value is not finite.
    """
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
    """Return the energies of the inversion's nodes below the written floor.

    The nodes are the nodes below the written floor that `_table_pieces`
    places for the component and the settings on the grid q: the log grid
    of the completion below the floor, down to the bottom that
    `below_floor` derives. The array is empty where the inversion
    places no node below the written floor, because the form has no
    completion source or the completion is not positive.
    """
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
    """Return the untruncated DF of one component from the component's inputs.

    Parameters
    ----------
    component : Component
        The component to invert.
    settings : Settings
        Numerical settings.
    function : callable, optional
        A user-specified DF as a function f(k, depth) of the energy k and
        the depth :math:`P_0 - k` below the central potential :math:`P_0`.
        The function's values on the grid and on the nodes below the
        written floor that the inversion places replace the inversion's rows
        and their completion; every later step runs unchanged. The table of
        a function's DF has no exact slopes. Default: ``None``, which means
        the DF is
        inverted from the component.

    Returns
    -------
    grid : EnergyGrid
        The table's energies, with r(K) from the analytic potential.
    table : DfTable
        The DF table.
    form : AsymptoticForm
        The closed forms beyond the table's two ends.
    k_t_min : float
        The lowest admissible truncation energy (`validate_df`).
    report : ValidationReport
        The DF's report (`validate_df`).
    inversion : InversionRecord or None
        The record of the inversion; None where `function` is given.
    rows : InversionRows or None
        The inversion's rows on its whole grid; None where `function` is
        given.

    Raises
    ------
    NotImplementedError
        If the component's anisotropy family is not in `SERVED`.
    InputError
        If the function's values have the wrong shape or are not finite
        somewhere on the grid.
    ResultError
        If the slope of the inversion's source, the derivative of rho_aux
        one order above the source, is not finite on the grid.
    AccuracyError
        If the table cannot be formed: a row inside the table is not
        finite or zero, a row exceeds the DF's ceiling where no closed form
        covers the center, or the tail's logarithmic exponent is not finite.
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

    # The grid of a piloted family is placed around the pilot's feature.
    with timed("energy_grid"):
        feature = pilot_feature(component, settings) if family in _PILOTED else None
        q, _ = inversion_grid(profile, aniso, settings, feature=feature)
        q = _below_limit(q, component)
    below = None
    if function is not None:
        # The function is tabulated on the grid and on the nodes below the
        # written floor that the inversion would place, so that the function's
        # table has the nodes of the table that the inversion would build.
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
        # A DF from a function has no dF/dK column, so the report states the
        # slope column as not judged.
        report = slope_column_not_judged(report, table, NO_SLOPE_COLUMN)

    # The radius r(K) of each table energy comes from the analytic potential.
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
    """Return the table and the forms beyond the table, built from the rows.

    The function builds, in order, the anchor and the closed form above the
    anchor, the nodes below the written floor and the tail, then the table
    and the asymptotic form. The returned dict holds every intermediate
    piece, because the gradient's rule (`gradient`) reads the intermediate
    pieces. The argument
    ``below`` gives nodes below the written floor made elsewhere (a
    function's ``K`` and ``F``, without slopes); where `floor` is not None,
    the completion's nodes replace ``below``.

    Raises
    ------
    AccuracyError
        If the tail's logarithmic exponent below the floor is not finite.
    AccuracyError
        If `_table` raises AccuracyError.
    """
    profile, aniso = component.unit_profile, component.anisotropy
    # A Type II model has no anchor and no closed form above its rows.
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

    # The tail exponent a is the analytic exponent (`tail_exponent_analytic`),
    # and a tail is built only where a > 0. The tail's amplitude, and its
    # logarithmic exponent where the lowest node has a slope, are matched to
    # the lowest node below the floor, or to the written floor where there is
    # no node below the floor.
    k_written = float(np.min(q[q >= OUTPUT_FLOOR_FACTOR * K_FLOOR]))
    if floor is not None:
        made = below_floor(aux, floor, k_written)
        below = None if made.get("not_positive") else made
    a = tail_exponent_analytic(profile, aniso)
    tail = None
    if a > 0.0:
        if below is not None and below.get("dFdK") is None:
            # Nodes below the floor from a function have no slope, so the tail
            # is the pure power through the lowest node, with no logarithm.
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

    # A Type II DF is returned only above P(R_a): the rows below P(R_a) come
    # from the continuation of the auxiliary density, so the table gets no
    # nodes below the floor and no tail.
    edge = aniso.domain_edge
    k_edge = (
        float(np.asarray(profile.p(np.array([edge])))[0])
        if math.isfinite(edge)
        else 0.0
    )
    if k_edge > 0.0:
        below, tail = None, None
    covered = ceiling_limit(profile, aniso) > 0.0
    table = _table(q, f, ddf, patch, below, k_edge, covered)
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
    covered: bool = True,
) -> DfTable:
    """Return the inversion's DF table.

    The table holds the rows as made, from the energy of the highest node
    below the written floor (from the first row where there is no node
    below the written floor) up to
    the anchor (up to the last row where there is no anchor), and stores
    the nodes below the written floor separately. Where the anisotropy's
    domain has an edge (a Type II model, with ``k_edge = P(R_a) > 0``),
    the DF covers only the support K > k_edge: the rows at or below
    k_edge stay in the table as NaN, and
    the table's first valid row is the first row above k_edge. A table
    with a negative row carries a sign column.

    Raises
    ------
    AccuracyError
        If ``covered`` is false (no closed form of the center covers the
        energies above a cut, `ceiling_limit`) and a row inside the table
        exceeds the DF's ceiling 10^LOG10_DF_CEILING in absolute value.
    AccuracyError
        If a row inside the table is not finite or is zero.
    """
    anchor_k = patch.k_anchor if patch is not None else np.inf
    k_w = float(np.max(below["K"])) if below is not None else -np.inf
    outside = q <= k_edge
    window = (q <= anchor_k) & (q >= k_w * (1.0 - 1e-12))
    if not covered:
        over = window & ~outside & (np.abs(f) > 10.0**LOG10_DF_CEILING)
        if over.any():
            first = int(np.flatnonzero(over)[0])
            err = AccuracyError(
                f"{int(over.sum())} rows of the inversion inside the table exceed the "
                f"DF's ceiling 10^{LOG10_DF_CEILING:g}, the first row {first} at K = "
                f"{float(q[first]):.6e}, and no closed form of the center covers the "
                "energies above: refused"
            )
            err.result = dict(k=q, f=f)
            raise err
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
    """Return the asymptotic form beyond the table's two ends.

    Above the table, the form is the center's closed form. Where `patch` is
    not None, the form also holds the patch's anchor energy, anchor
    amplitude and regular part (b, d).
    Below the floor, the form is the tail where the DF has one, and zero
    otherwise.
    """
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
