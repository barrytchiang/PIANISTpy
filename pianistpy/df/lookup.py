r"""The DF of one component at any energy, from its table and its forms.

`log10_df` returns the base-10 logarithm of a component's distribution
function (DF) :math:`F` at given relative energies :math:`\mathcal{K}`.
Between the table's first and last energies, :math:`\log_{10}F` is the cubic
Hermite interpolant in :math:`\ln\mathcal{K}` through the table's nodes.
Above the table's last energy, :math:`\log_{10}F` follows the center's form
in the depth :math:`u = P_0 - \mathcal{K}`, with :math:`P_0` the central
relative potential. Below the table's first energy, :math:`\log_{10}F`
follows the rule of `AsymptoticForm.below_floor`.

The lookup is split into two functions. `prepare_lookup` runs once per table
on the host. `prepare_lookup` returns the arrays that the evaluation reads
(`LookupArrays`: the nodes padded to a multiple of `NODE_BUCKET`, and every
number of the forms in one array) and the kinds of the forms
(`LookupKinds`), which are the only static part of the lookup.
`evaluate_lookup` is a function of the `LookupArrays` and of the energies,
under NumPy or under JAX. In a JAX kernel the table's arrays are then traced
arguments, so two models whose tables have the same padded length and the
same kinds of forms share one compiled function. `log10_df` calls
`prepare_lookup` and then `evaluate_lookup`.

References
----------
Notes Section T:2.5.4.
"""

__all__ = [
    "LookupArrays",
    "LookupKinds",
    "below_floor_mass",
    "evaluate_lookup",
    "log10_df",
    "prepare_lookup",
]

import contextlib
import dataclasses
import math
from typing import NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline

from ..exceptions import ResultError
from ..typing import Array, ArrayLike
from .asymptotics import tail_mass_below
from .stages import AsymptoticForm, DfTable

# Lower bound on the argument of a logarithm on the branch that a `where`
# keeps only for a positive argument. VALUE_MIN keeps the logarithm on the
# discarded branch finite.
VALUE_MIN = 1e-300
# The nodes of a table are padded to a multiple of NODE_BUCKET so that tables
# of different lengths share one compiled function. The interpolant never
# uses a padded node, since `_cells` clips the cell index to the last real cell.
NODE_BUCKET = 16384
# The fields of an AsymptoticForm that the check for one entry per component
# skips. The field p_0 is one number for all components. The fields
# central_anchor, central_anchor_amplitude and central_regular may be empty,
# and an empty central_anchor means that no component has an anchor.
_NOT_READ = ("p_0", "central_anchor", "central_anchor_amplitude", "central_regular")

# The index of each number of the forms in `LookupArrays.params`. N_PARAMS is
# the length of `LookupArrays.params`.
(
    K_0,
    K_TOP,
    P_0,
    U_TOP,
    LOG_DF_TOP,
    CENTRAL_EXPONENT,
    CENTRAL_AMPLITUDE,
    LOG_AMPLITUDE,
    R_MINUS_1,
    LOG_COEFFICIENT,
    NEXT_EXPONENT,
    NEXT_COEFFICIENT,
    NEXT_TOP,
    NEXT_TOP_RELATIVE,
    TAIL_AMPLITUDE,
    TAIL_LOG_AMPLITUDE,
    TAIL_EXPONENT,
    TAIL_LOG_EXPONENT,
    TAIL_LOG_SCALE,
    FLOOR_0,
    FLOOR_1,
    FLOOR_2,
    ANCHOR_AMPLITUDE,
    U_ANCHOR,
    REGULAR_B,
    REGULAR_D,
    N_PARAMS,
) = range(27)
# The names of the coefficients of each rule below the table, in the order of
# `AsymptoticForm.below_floor_coefficients`. The rules "tail" and "zero" take
# no coefficients.
FLOOR_COEFFICIENTS = {
    "taylor": ("F_1", "F_2", "k_switch"),
    "edge_law": ("p", "C_0", "C_1"),
}


class LookupArrays(NamedTuple):
    r"""The arrays of one component's lookup, padded to a multiple of `NODE_BUCKET`.

    Parameters
    ----------
    k : array, shape (m,)
        Relative energies :math:`\mathcal{K}_i` of the nodes, in internal
        units. The entries from index `n` on are padding, set to infinity.
    log_df : array, shape (m,)
        Base-10 logarithm of the DF at the nodes. The entries from index `n`
        on are padding, set to zero.
    slope : array, shape (m,)
        Slope :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K}` at the
        nodes. The entries from index `n` on are padding, set to zero.
    width : array, shape (m,)
        Width :math:`\ln(\mathcal{K}_{i+1}/\mathcal{K}_i)` of each cell,
        computed as ``log1p`` of the relative difference of the two nodes.
        The entries from index ``n - 1`` on are padding, set to zero.
    n : array of int, shape ()
        Number of real (unpadded) nodes.
    params : array, shape (N_PARAMS,)
        Every number of the forms beyond the table, at the indices that the
        module-level constants (`K_0`, `P_0`, `TAIL_EXPONENT` and the others)
        name.
    """

    k: Array
    log_df: Array
    slope: Array
    width: Array
    n: Array
    params: Array


class LookupKinds(NamedTuple):
    """The kinds of one component's forms: the static part of the lookup.

    Parameters
    ----------
    central : str
        Form of the DF above the table, from `AsymptoticForm.central_form`.
    next_term : str
        Next singular term above the table, from
        `AsymptoticForm.central_next_form`.
    below_floor : str
        Rule of the DF below the table, from `AsymptoticForm.below_floor`.
    anchored : bool
        True where the table ends at an anchor energy and the anchored form
        of `AsymptoticForm.central_df` continues the table above the anchor
        energy.
        Default: ``False``.
    """

    central: str
    next_term: str
    below_floor: str
    anchored: bool = False


def prepare_lookup(
    table: DfTable, form: AsymptoticForm, component: int = 0
) -> tuple[LookupArrays, LookupKinds]:
    """Return the arrays and the kinds that one component's lookup reads.

    Parameters
    ----------
    table : DfTable
        The DF on its energy grid.
    form : AsymptoticForm
        The forms beyond the two ends of the table, and `p_0`, the energy at
        which the support of the DF ends.
    component : int, optional
        Index of the component. Default: ``0``, which means the first
        component.

    Returns
    -------
    arrays : LookupArrays
        The nodes, padded to a multiple of `NODE_BUCKET`, and every number of
        the forms.
    kinds : LookupKinds
        The kinds of the forms.

    Raises
    ------
    ResultError
        If the table is signed (some DF of the table is negative).
    ResultError
        If a field of `form` holds no entry for the component, or a form is
        of a kind that the lookup does not evaluate.
    ResultError
        If the rule below the table has fewer coefficients than the rule
        needs, the rule ``"tail"`` has an amplitude that is not positive,
        or the tail's logarithmic factor ends below the table's first
        energy.
    ResultError
        If the form's anchor energy is not the table's last energy.
    """
    nodes = _nodes(table, component)
    n = int(nodes["k"].size)
    m = NODE_BUCKET * max(1, math.ceil(n / NODE_BUCKET))
    pad = m - n
    k = np.concatenate([nodes["k"], np.full(pad, np.inf)])
    log_df = np.concatenate([nodes["log_df"], np.zeros(pad)])
    slope = np.concatenate([nodes["slope"], np.zeros(pad)])
    width = np.concatenate([nodes["width"], np.zeros(pad + 1)])
    missing = [
        field.name
        for field in dataclasses.fields(form)
        if field.name not in _NOT_READ and len(getattr(form, field.name)) <= component
    ]
    if missing:
        raise ResultError(
            f"The form states nothing for component {component} in {missing}."
        )
    kinds = LookupKinds(
        central=form.central_form[component],
        next_term=form.central_next_form[component],
        below_floor=form.below_floor[component],
    )
    if kinds.central not in (
        "power",
        "constant_plus_power",
        "log",
        "analytic",
    ):
        raise ResultError(f"The form states no center {kinds.central!r}.")
    if kinds.next_term not in ("none", "power", "log_power"):
        raise ResultError(f"The form states no next term {kinds.next_term!r}.")
    if kinds.below_floor not in ("tail", "taylor", "edge_law", "zero"):
        raise ResultError(
            f"The form states no rule {kinds.below_floor!r} below the table."
        )
    coefficients = form.below_floor_coefficients[component]
    needed = FLOOR_COEFFICIENTS.get(kinds.below_floor, ())
    if len(coefficients) < len(needed):
        raise ResultError(
            f"The rule {kinds.below_floor!r} below the table needs the coefficients "
            f"{needed}; the form states {len(coefficients)}."
        )
    tail_amplitude = float(form.tail_amplitude[component])
    if kinds.below_floor == "tail":
        if not tail_amplitude > 0.0:
            raise ResultError(
                "The rule 'tail' below the table needs a positive amplitude; the "
                f"form states {tail_amplitude!r}."
            )
        scale = float(form.tail_log_scale[component])
        if form.tail_log_exponent[component] != 0.0 and not nodes["k"][0] <= scale:
            raise ResultError(
                f"The tail's logarithmic factor ln(E_0 / K) ends at E_0 = {scale!r}, "
                f"below the table's first energy {float(nodes['k'][0])!r}."
            )
    params = np.zeros(N_PARAMS)
    params[K_0], params[K_TOP] = nodes["k"][0], nodes["k"][-1]
    params[P_0] = form.p_0
    params[U_TOP] = float(form.p_0) - float(nodes["k"][-1])
    params[LOG_DF_TOP] = nodes["log_df"][-1]
    s = float(form.central_exponent[component])
    amplitude = float(form.central_amplitude[component])
    params[CENTRAL_EXPONENT], params[CENTRAL_AMPLITUDE] = s, amplitude
    params[LOG_COEFFICIENT] = form.central_log_coefficient[component]
    params[NEXT_EXPONENT] = form.central_next_exponent[component]
    params[NEXT_COEFFICIENT] = form.central_next_coefficient[component]
    u_top = params[U_TOP]
    with np.errstate(all="ignore"):
        # `_above` evaluates a power A u^s with A > 0 in units of the power
        # itself, through the ratio r = F_top / (A u_top^s) of the DF at the
        # table's last node to the power at that node. r - 1 and log10 A are
        # formed here on the host. log10 A exists for A > 0 only, so for
        # A <= 0 the entry LOG_AMPLITUDE stores zero.
        log_amplitude = np.log10(amplitude) if amplitude > 0.0 else 0.0
        params[LOG_AMPLITUDE] = log_amplitude
        params[R_MINUS_1] = np.expm1(
            np.log(10.0) * (params[LOG_DF_TOP] - log_amplitude - s * np.log10(u_top))
        )
        g_top = (
            float(_next_term(kinds.next_term, params, np.array(u_top), np))
            if kinds.next_term != "none" and kinds.central != "analytic"
            else 0.0
        )
        params[NEXT_TOP] = g_top
        params[NEXT_TOP_RELATIVE] = g_top * 10.0 ** (
            -(log_amplitude + s * np.log10(u_top))
        )
    params[TAIL_AMPLITUDE] = tail_amplitude
    # The logarithm of the tail's amplitude is read only under the rule "tail",
    # and this function raises ResultError for a rule "tail" whose amplitude is
    # not positive.
    params[TAIL_LOG_AMPLITUDE] = (
        math.log10(tail_amplitude) if kinds.below_floor == "tail" else 0.0
    )
    params[TAIL_EXPONENT] = form.tail_exponent[component]
    params[TAIL_LOG_EXPONENT] = form.tail_log_exponent[component]
    params[TAIL_LOG_SCALE] = form.tail_log_scale[component]
    for i in range(len(needed)):
        params[FLOOR_0 + i] = coefficients[i]
    anchor = (
        float(form.central_anchor[component])
        if len(form.central_anchor) > component
        else math.nan
    )
    if math.isfinite(anchor):
        # The table ends at the anchor energy, and above the anchor energy the
        # anchored form A u_a^kappa (u / u_a)^kappa + B + D u is the only form
        # the lookup evaluates, with kappa the central exponent and u_a the
        # anchor's depth p_0 - anchor.
        if anchor != float(nodes["k"][-1]):
            raise ResultError(
                f"The form's anchor {anchor!r} is not the table's last energy "
                f"{float(nodes['k'][-1])!r}."
            )
        kinds = kinds._replace(anchored=True)
        params[ANCHOR_AMPLITUDE] = form.central_anchor_amplitude[component]
        params[U_ANCHOR] = float(form.p_0) - anchor
        params[REGULAR_B], params[REGULAR_D] = form.central_regular[component]
    arrays = LookupArrays(
        k=k, log_df=log_df, slope=slope, width=width, n=np.array(n), params=params
    )
    return arrays, kinds


def evaluate_lookup(
    arrays: LookupArrays,
    kinds: LookupKinds,
    k: ArrayLike,
    xp=np,
    u: ArrayLike | None = None,
) -> Array:
    r"""Return the base-10 logarithm of the DF from a prepared lookup.

    A caller gives each point either by its relative energy
    :math:`\mathcal{K}` or by its depth :math:`u = P_0 - \mathcal{K}`, and
    gives whichever of the two quantities the caller holds exactly. Where
    `u` is given and finite, the energy is taken as :math:`P_0 - u`; in a
    cell whose lower node satisfies :math:`\mathcal{K}_i \ge P_0/2`, the
    offset :math:`\mathcal{K} - \mathcal{K}_i` is formed as :math:`u_i - u`
    with :math:`u_i = P_0 - \mathcal{K}_i`, a difference that is exact in
    double precision in such a cell; and the form above the table is
    evaluated at `u` itself, including where :math:`P_0 - u` rounds to
    :math:`P_0`. Where `u` is not finite, or `u` is not given, the point is
    given by `k` alone.

    Parameters
    ----------
    arrays : LookupArrays
        The prepared arrays (`prepare_lookup`). Under JAX, `arrays` may be
        traced.
    kinds : LookupKinds
        The kinds of the forms (`prepare_lookup`); static under JAX.
    k : array-like
        Relative energies :math:`\mathcal{K}`, in internal units. Under JAX,
        `k` may be traced.
    xp : module, optional
        The array module, ``numpy`` or ``jax.numpy`` with 64-bit floats;
        static under JAX. Default: ``numpy``.
    u : array-like, optional
        Depths :math:`u = P_0 - \mathcal{K}`, in internal units, broadcast
        against `k`; NaN where a point is given by `k`. Under JAX, `u` may be
        traced. Default: ``None``, which means that every point is given by
        `k`.

    Returns
    -------
    log10_df : array
        Base-10 logarithm of the DF at each point. Minus infinity where the
        DF is zero: at a depth :math:`u` at or below zero (an energy at or
        above :math:`P_0`), at and below zero energy, below the table under
        the rule ``"zero"``, and wherever a form beyond the table evaluates
        to a value at or below zero.
    """
    k = xp.asarray(k, dtype=xp.float64)
    params = arrays.params
    k_0, k_top, p_0 = params[K_0], params[K_TOP], params[P_0]
    with np.errstate(all="ignore") if xp is np else contextlib.nullcontext():
        if u is None:
            depth = p_0 - k
        else:
            u = xp.asarray(u, dtype=xp.float64)
            held = xp.isfinite(u)
            k = xp.where(held, p_0 - u, k)
            depth = xp.where(held, u, p_0 - k)
        if xp is np:
            # Under NumPy, `_by_part` evaluates each part of the lookup only on
            # the points that the part selects, with the same arithmetic as
            # the JAX branch below, so that the forms beyond the table are not
            # evaluated at the points inside the table.
            return _by_part(arrays, kinds, k, depth, u)
        inside = _cells(arrays, k, xp, u)
        above = _above(arrays, kinds, depth, xp)
        below = _below(arrays, kinds, k, xp)
        # A point lies above the table where k > k_top and its depth is
        # positive. Where the depth is formed from k, a positive depth means
        # k < p_0. A depth given in `u` that is below half the spacing of
        # doubles at p_0 stays above the table, although the point's energy
        # p_0 - u rounds to p_0.
        return xp.where(
            (k >= k_0) & (k <= k_top),
            inside,
            xp.where(
                (k > k_top) & (depth > 0.0),
                above,
                xp.where((k > 0.0) & (k < k_0), below, -xp.inf),
            ),
        )


def _by_part(
    arrays: LookupArrays,
    kinds: LookupKinds,
    k: Array,
    depth: Array,
    u: Array | None,
) -> Array:
    """Return `evaluate_lookup`'s result under NumPy, one part at a time.

    The cubic Hermite interpolant is evaluated only at the points inside the
    table, the form above the table only at the points above the table (by
    their depth), and the rule below the table only at the points below the
    table; every other point is minus infinity. Each part uses the same
    arithmetic as the JAX branch of `evaluate_lookup`.
    """
    params = arrays.params
    k, depth = np.broadcast_arrays(k, depth)
    if u is not None:
        u = np.broadcast_to(u, k.shape)
    out = np.full(k.shape, -np.inf)
    inside = (k >= params[K_0]) & (k <= params[K_TOP])
    out[inside] = _cells(arrays, k[inside], np, None if u is None else u[inside])
    above = ~inside & (k > params[K_TOP]) & (depth > 0.0)
    if above.any():
        out[above] = _above(arrays, kinds, depth[above], np)
    below = ~inside & ~above & (k > 0.0) & (k < params[K_0])
    if below.any():
        out[below] = _below(arrays, kinds, k[below], np)
    return out


def log10_df(
    table: DfTable,
    form: AsymptoticForm,
    k: ArrayLike,
    component: int = 0,
    xp=np,
) -> Array:
    r"""Return the base-10 logarithm of one component's DF at given energies.

    Parameters
    ----------
    table : DfTable
        The DF on its energy grid.
    form : AsymptoticForm
        The forms beyond the two ends of the table. The center's forms are
        written in the depth :math:`u = P_0 - \mathcal{K}`, with :math:`P_0`
        the field `form.p_0`.
    k : array-like
        Relative energies :math:`\mathcal{K}`, in internal units.
    component : int, optional
        Index of the component. Default: ``0``, which means the first
        component.
    xp : module, optional
        The array module, ``numpy`` or ``jax.numpy`` with 64-bit floats.
        Default: ``numpy``.

    Returns
    -------
    log10_df : array, shape of `k`
        Base-10 logarithm of the DF at the energies `k`. Minus infinity
        where the DF is zero: at and above the energy `form.p_0`, at and
        below zero energy, below the table under the rule ``"zero"``, and
        wherever a form beyond the table evaluates to a value at or below
        zero.

    Raises
    ------
    ResultError
        If the table is signed (some DF of the table is negative): only a
        non-negative DF has a logarithm.
    ResultError
        If `prepare_lookup` rejects the table's forms (see
        `prepare_lookup`).

    Notes
    -----
    In the cell :math:`[\mathcal{K}_i, \mathcal{K}_{i+1}]`, the position
    :math:`t` is formed from differences of energies,

    .. math::

        t = \ln\!\left(1 + \frac{\mathcal{K} - \mathcal{K}_i}{\mathcal{K}_i}
            \right) \Big/ \ln\!\left(1 + \frac{\mathcal{K}_{i+1} -
            \mathcal{K}_i}{\mathcal{K}_i}\right).

    The difference :math:`\mathcal{K} - \mathcal{K}_i` is exact where the
    cell spans less than a factor of two in energy, so the position
    :math:`t` never differences a logarithm of the energy against a stored
    logarithm. The slopes at the nodes are the table's exact slopes
    :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K} =
    \mathcal{K}F'/(F\ln 10)` where `DfTable.slopes_exact` is True for the
    component; otherwise the slopes are those of the not-a-knot cubic spline
    through the nodes.

    References
    ----------
    Notes Section T:2.5.4.
    """
    arrays, kinds = prepare_lookup(table, form, component)
    if xp is not np:
        arrays = LookupArrays(*(xp.asarray(a) for a in arrays))
    return evaluate_lookup(arrays, kinds, k, xp)


def _nodes(table: DfTable, component: int) -> dict:
    """Return the nodes of a component's support, joined to its nodes below the table.

    The result is a dict of NumPy arrays: ``k``, ``log_df``, ``slope`` and
    the cell widths ``width``. Where `DfTable.slopes_exact` is False for the
    component, ``slope`` is taken from the not-a-knot cubic spline of
    ``log_df`` in ln k.

    Raises
    ------
    ResultError
        If the table is signed (some DF of the table is negative).
    """
    if table.sign.size:
        raise ResultError(
            "The table is signed: a DF that is negative somewhere has no logarithm."
        )
    support = slice(table.i_first[component], table.i_last[component] + 1)
    k = table.k[component, support]
    log_df = table.log_df[component, support]
    slope = table.log_df_slope[component, support]
    n_below = table.nodes_below(component)
    if n_below:
        # The last node below the table may lie on the table's first node,
        # where the join of the nodes below the table to the table is measured.
        # Such a node duplicates the table's first node, so only the nodes
        # below the table's first energy by more than a relative 1e-12 are
        # joined.
        kb = table.k_below[component, :n_below]
        keep = kb < k[0] * (1.0 - 1e-12)
        k = np.concatenate([kb[keep], k])
        log_df = np.concatenate([table.log_df_below[component, :n_below][keep], log_df])
        slope = np.concatenate(
            [table.log_df_slope_below[component, :n_below][keep], slope]
        )
    if not table.slopes_exact[component]:
        slope = CubicSpline(np.log(k), log_df, bc_type="not-a-knot")(np.log(k), 1)
    return dict(
        k=k,
        log_df=log_df,
        slope=np.asarray(slope, dtype=np.float64),
        width=np.log1p((k[1:] - k[:-1]) / k[:-1]),
    )


def _cells(arrays: LookupArrays, k: Array, xp, u: Array | None = None) -> Array:
    r"""Return the cubic Hermite interpolant of log10 F in ln K at the energies `k`.

    The position in the cell is formed from the nodes themselves (see
    `log10_df`). Energies outside the table are clipped to the table's first
    or last node, so the interpolant returns the end value there. Where the
    depth `u` is finite and the cell's lower node satisfies
    :math:`\mathcal{K}_i \ge P_0/2`, the offset
    :math:`\mathcal{K} - \mathcal{K}_i` is formed as :math:`u_i - u` with
    :math:`u_i = P_0 - \mathcal{K}_i`.
    """
    k_nodes, y, m, width = arrays.k, arrays.log_df, arrays.slope, arrays.width
    n = arrays.n
    k = xp.clip(k, k_nodes[0], k_nodes[n - 1])
    i = xp.clip(xp.searchsorted(k_nodes, k, side="right") - 1, 0, n - 2)
    k_i = k_nodes[i]
    h = width[i]
    offset = k - k_i
    if u is not None:
        p_0 = arrays.params[P_0]
        offset = xp.where((k_i >= 0.5 * p_0) & xp.isfinite(u), (p_0 - k_i) - u, offset)
    t = xp.log1p(offset / k_i) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
        + (t3 - 2.0 * t2 + t) * h * m[i]
        + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
        + (t3 - t2) * h * m[i + 1]
    )


def _next_term(kind: str, params: Array, u: Array, xp) -> Array:
    """Return the center's next singular term at the depths `u`.

    The term is ``c u**e`` for the kind ``"power"`` and ``c u**e (-ln u)``
    for the kind ``"log_power"``, with the exponent ``e`` and the
    coefficient ``c`` read from `params` at the indices `NEXT_EXPONENT` and
    `NEXT_COEFFICIENT`.
    """
    exponent, coefficient = params[NEXT_EXPONENT], params[NEXT_COEFFICIENT]
    u = xp.asarray(u, dtype=xp.float64)
    if kind == "power":
        return coefficient * u**exponent
    return coefficient * u**exponent * (-xp.log(u))


def _above(arrays: LookupArrays, kinds: LookupKinds, u: Array, xp) -> Array:
    r"""Return log10 F above the table, at depths :math:`u` in :math:`(0, u_{\rm top})`.

    The depth is :math:`u = P_0 - \mathcal{K}`, and :math:`u_{\rm top}` is
    the depth of the table's last node. Every form matches the table's
    value :math:`F_{\rm top}` at the last node. A power :math:`A u^s` with
    :math:`A > 0` is evaluated relative to the power itself, so that a steep
    center, whose :math:`F` exceeds the range of a double, does not
    overflow. With :math:`r = F_{\rm top}/(A u_{\rm top}^s)`, the DF is
    :math:`F = A u^s[1 + (r - 1)(u_{\rm top}/u)^s]` for :math:`s < 0` and
    :math:`F = A u^s[1 + (r - 1)(u/u_{\rm top})^{1-s}]` for
    :math:`s \ge 0`. In both expressions, the remainder
    :math:`F - A u^s` is constant for :math:`s < 0` and proportional to
    :math:`u` for :math:`s \ge 0`. With a next singular term :math:`G_2`, the
    two singular terms :math:`A u^s` and :math:`G_2` are evaluated exactly,
    and the remainder :math:`F - A u^s - G_2` is constant for :math:`s < 0`
    and proportional to :math:`u` for :math:`s \ge 0`. The
    signs of :math:`A` and :math:`s` are
    numbers of the arrays, not static kinds, so a `where` selects the branch
    for each sign. Where `kinds.anchored` is True, the anchored form of
    `AsymptoticForm.central_df` is evaluated instead of the form that
    `kinds.central` identifies.
    """
    params = arrays.params
    log_df_top, u_top = params[LOG_DF_TOP], params[U_TOP]
    if kinds.anchored:
        u = xp.where(u > 0.0, u, u_top)
        ratio = xp.maximum(u / params[U_ANCHOR], 1e-300)
        df = (
            params[ANCHOR_AMPLITUDE] * ratio ** params[CENTRAL_EXPONENT]
            + params[REGULAR_B]
            + params[REGULAR_D] * u
        )
        return xp.where(df > 0.0, xp.log10(xp.maximum(df, VALUE_MIN)), -xp.inf)
    # A depth u <= 0 lies beyond the support, where `evaluate_lookup` returns
    # -inf. At such a depth the form is evaluated at u_top instead, where the
    # form is finite for every exponent s; a fixed small floor on u would not
    # keep the form finite for a steep negative exponent. The branch that a
    # `where` discards, and its gradient, thus stay finite under NumPy and JAX.
    u = xp.where(u > 0.0, u, u_top)
    s, amplitude = params[CENTRAL_EXPONENT], params[CENTRAL_AMPLITUDE]
    has_next = kinds.next_term != "none" and kinds.central != "analytic"
    g2 = _next_term(kinds.next_term, params, u, xp) if has_next else None
    g_top = params[NEXT_TOP]
    df_top = 10.0**log_df_top
    d2 = (g2 - g_top) if has_next else 0.0
    if kinds.central == "power":
        # Branch for A > 0: the power evaluated relative to itself.
        log_amplitude = params[LOG_AMPLITUDE]
        factor = xp.where(s < 0.0, (u_top / u) ** s, (u / u_top) ** (1.0 - s))
        correction = params[R_MINUS_1] * factor
        if has_next:
            g_rel = g2 * 10.0 ** (-(log_amplitude + s * xp.log10(u)))
            correction = correction + g_rel - params[NEXT_TOP_RELATIVE] * factor
        v = 1.0 + correction
        relative = xp.where(
            v > 0.0,
            log_amplitude + s * xp.log10(u) + xp.log10(xp.maximum(v, VALUE_MIN)),
            -xp.inf,
        )
        # Branch for A <= 0: the power evaluated in F itself, with the
        # remainder F - A u^s matched to F at the table's last node.
        linear_part = xp.where(s > 0.0, u / u_top, 1.0)
        if has_next:
            df = xp.where(
                s > 0.0,
                amplitude * u**s
                + g2
                + (df_top - amplitude * u_top**s - g_top) * (u / u_top),
                amplitude * u**s + (df_top - amplitude * u_top**s) * linear_part + d2,
            )
        else:
            df = amplitude * u**s + (df_top - amplitude * u_top**s) * linear_part
        absolute = xp.where(df > 0.0, xp.log10(xp.maximum(df, VALUE_MIN)), -xp.inf)
        return xp.where(amplitude > 0.0, relative, absolute)
    if kinds.central == "constant_plus_power":
        df = df_top + amplitude * (u**s - u_top**s) + d2
    elif kinds.central == "log":
        df = df_top + params[LOG_COEFFICIENT] * xp.log(u_top / u) + d2
    else:
        df = xp.full(u.shape, df_top)
    return xp.where(df > 0.0, xp.log10(xp.maximum(df, VALUE_MIN)), -xp.inf)


def _below(arrays: LookupArrays, kinds: LookupKinds, k: Array, xp) -> Array:
    r"""Return log10 F below the table, at energies above zero.

    The energies satisfy :math:`0 < \mathcal{K} < \mathcal{K}_0`, with
    :math:`\mathcal{K}_0` the table's first energy. The rules are
    ``"tail"``, :math:`F = A\mathcal{K}^{-a}[\ln(E_0/\mathcal{K})]^{-b}`;
    ``"taylor"``, :math:`F = F_1\mathcal{K} + F_2\mathcal{K}^2/2`;
    ``"edge_law"``, :math:`F = \mathcal{K}^p(C_0 + C_1\mathcal{K})`; and
    ``"zero"``, :math:`F = 0`. Here :math:`A`, :math:`a`, :math:`b` and
    :math:`E_0` are the tail's amplitude, exponent, logarithmic exponent
    and logarithmic scale, and :math:`F_1`, :math:`F_2`, :math:`p`,
    :math:`C_0` and :math:`C_1` are the rule's coefficients. At the points
    that a `where` discards, the tail's argument is moved inside
    :math:`(0, E_0)` and the argument of the other rules' powers is moved
    above zero, so that the discarded branch stays finite. The tail is one
    expression under NumPy and JAX. `prepare_lookup` raises ResultError for
    a tail whose amplitude is not positive and for a tail with
    :math:`b \ne 0` whose logarithmic factor ends below
    :math:`\mathcal{K}_0`, so for :math:`b \ne 0` every kept point
    :math:`\mathcal{K}` lies in :math:`(0, E_0)`.
    """
    params = arrays.params
    rule = kinds.below_floor
    if rule == "zero":
        return xp.full(k.shape, -xp.inf)
    if rule == "tail":
        exponent = params[TAIL_EXPONENT]
        log_exponent, scale = params[TAIL_LOG_EXPONENT], params[TAIL_LOG_SCALE]
        k_pos = xp.where(k > 0.0, k, 0.5 * params[K_0])
        k_log = xp.where(k_pos < scale, k_pos, 0.5 * scale)
        return (
            params[TAIL_LOG_AMPLITUDE]
            - exponent * xp.log10(k_pos)
            - log_exponent * xp.log10(xp.log(scale / k_log))
        )
    # An energy K <= 0 lies beyond the support, where `evaluate_lookup` returns
    # -inf; the rule is evaluated at K_0 / 2 instead, inside the rule's range.
    k = xp.where(k > 0.0, k, 0.5 * params[K_0])
    if rule == "taylor":
        v = params[FLOOR_0] * k + 0.5 * params[FLOOR_1] * k * k
    else:
        p, c_0, c_1 = params[FLOOR_0], params[FLOOR_1], params[FLOOR_2]
        v = k**p * (c_0 + c_1 * k)
    return xp.where(v > 0.0, xp.log10(xp.maximum(v, VALUE_MIN)), -xp.inf)


def below_floor_mass(arrays: LookupArrays, kinds: LookupKinds, k: ArrayLike) -> Array:
    r"""Return the integral of a DF from zero energy to energies below its table.

    The integrand is the rule below the table that `_below` evaluates, with
    the coefficients stored in `arrays.params`. The integral
    :math:`\int_0^{\mathcal{K}} F\,\mathrm{d}\mathcal{K}'` is
    :math:`F_1\mathcal{K}^2/2 + F_2\mathcal{K}^3/6` for the rule
    ``"taylor"``,
    :math:`C_0\mathcal{K}^{p+1}/(p+1) + C_1\mathcal{K}^{p+2}/(p+2)` for the
    rule ``"edge_law"``, the result of `tail_mass_below` for the rule
    ``"tail"``, and zero for the rule ``"zero"``.

    Parameters
    ----------
    arrays : LookupArrays
        The arrays of a lookup prepared by `prepare_lookup`, as NumPy arrays.
    kinds : LookupKinds
        The kinds of the forms of the lookup that `arrays` belongs to.
    k : array-like
        Upper limits :math:`\mathcal{K}` of the integral, at or below the
        table's first energy, in internal units.

    Returns
    -------
    mass : array
        The integral at each upper limit in `k`, in internal units, with the
        shape of `k` and at least one dimension.
    """
    params = np.asarray(arrays.params)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    rule = kinds.below_floor
    if rule == "tail":
        return tail_mass_below(
            float(params[TAIL_AMPLITUDE]),
            float(params[TAIL_EXPONENT]),
            k,
            float(params[TAIL_LOG_EXPONENT]),
            float(params[TAIL_LOG_SCALE]),
        )
    if rule == "taylor":
        f_1, f_2 = float(params[FLOOR_0]), float(params[FLOOR_1])
        return f_1 * k * k / 2.0 + f_2 * k * k * k / 6.0
    if rule == "edge_law":
        p, c_0, c_1 = (float(params[FLOOR_0 + i]) for i in range(3))
        return c_0 * k ** (p + 1.0) / (p + 1.0) + c_1 * k ** (p + 2.0) / (p + 2.0)
    return np.zeros(k.shape)
