"""The DF of a component at any energy, from its table and its forms.

Inside the table the base-10 logarithm of the DF is the cubic Hermite in
ln K through the nodes. Above the table's last energy it continues by the
center's form, in u = P_0 - K; below its first energy by the rule of
`AsymptoticForm.below_floor`.

The lookup is two functions. `prepare_lookup` runs once per table on the host
and returns the arrays the evaluation reads (`LookupArrays`: the nodes padded
to a fixed bucket, and every number of the forms in one array) and the kinds
of the forms (`LookupKinds`), which are the only static part. `evaluate_lookup`
is a function of those arrays and of the energies under either array module,
so that in a kernel the table is a traced argument: two models whose tables
have the same bucket and the same kinds share one compiled function.
`log10_df` composes the two.
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

# Floor of the argument of a logarithm whose sign a `where` has tested.
VALUE_MIN = 1e-300
# The nodes are padded to a multiple of this count (the battery's tables hold
# 8,334 to 12,411 nodes: one bucket); the padding is never read, since the
# cell index stops at the last real cell.
NODE_BUCKET = 16384
# the fields of the form the every-field check leaves out: p_0 is one number,
# and a result written before the anchor existed holds its patched rows to the
# ceiling, so its empty anchor fields read as no anchor
_NOT_READ = ("p_0", "central_anchor", "central_anchor_amplitude", "central_regular")

# The positions of the forms' numbers in `LookupArrays.params`.
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
# The coefficients of each rule below the table, in the order the form states
# them (`AsymptoticForm.below_floor_coefficients`).
FLOOR_COEFFICIENTS = {
    "taylor": ("F_1", "F_2", "k_switch"),
    "edge_law": ("p", "C_0", "C_1"),
}


class LookupArrays(NamedTuple):
    """The arrays of a component's lookup, padded to `NODE_BUCKET`.

    Parameters
    ----------
    k, log_df, slope : array, shape (m,)
        The nodes, log10 F and d log10 F / d ln K at the nodes; the entries
        beyond `n` are padding (k infinite, the others zero).
    width : array, shape (m,)
        ln(K_i+1 / K_i) formed as log1p of the difference; the entries from
        n - 1 on are padding.
    n : array of int, shape ()
        The number of nodes.
    params : array, shape (N_PARAMS,)
        Every number of the forms, at the positions this module names.
    """

    k: Array
    log_df: Array
    slope: Array
    width: Array
    n: Array
    params: Array


class LookupKinds(NamedTuple):
    """The kinds of a component's forms: the static part of the lookup."""

    central: str
    next_term: str
    below_floor: str
    anchored: bool = False


def prepare_lookup(
    table: DfTable, form: AsymptoticForm, component: int = 0
) -> tuple[LookupArrays, LookupKinds]:
    """Return the arrays and the kinds a component's lookup reads.

    Parameters
    ----------
    table : DfTable
        The DF on its energy grid.
    form : AsymptoticForm
        The forms beyond the two ends of the table, and `p_0`, where the
        support ends.
    component : int, optional
        The component.

    Raises
    ------
    ResultError
        If the table is signed, or a form is of a kind the lookup does not
        know.
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
        # the power relative to itself, formed on the host: A > 0 only
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
    # read under the rule "tail" only, where the amplitude is positive
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
        # the table ends at the anchor and the form continues it: the one path
        # above the table, A u_a^kappa (u / u_a)^kappa + B + D u
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
    r"""Return log10 F at the energies `k` from a prepared lookup.

    Under JAX, `arrays`, `k` and `u` may be traced; `kinds` and `xp` are
    static.

    The lookup has two entries, by the energy and by the depth
    :math:`u = P_0 - \mathcal{K}`; a caller asks by the one whose argument
    it holds exactly. Where `u` is given and finite, the energy is
    :math:`P_0 - u`, the cell position is formed from :math:`u_i - u` with
    :math:`u_i = P_0 - \mathcal{K}_i` (exact where
    :math:`\mathcal{K}_i \ge P_0/2`) and the form above the table is read at
    `u` itself, also where :math:`P_0 - u` rounds to :math:`P_0` [3-206];
    where it is NaN, or not given, by `k`, unchanged.
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
        inside = _cells(arrays, k, xp, u)
        above = _above(arrays, kinds, depth, xp)
        below = _below(arrays, kinds, k, xp)
        # above the table by the depth: p_0 - k > 0 is k < p_0 where the depth
        # is formed from k, and a held depth below half an ulp of p_0 stays
        # above the table although its energy rounds to p_0 [3-206]
        return xp.where(
            (k >= k_0) & (k <= k_top),
            inside,
            xp.where(
                (k > k_top) & (depth > 0.0),
                above,
                xp.where((k > 0.0) & (k < k_0), below, -xp.inf),
            ),
        )


def log10_df(
    table: DfTable,
    form: AsymptoticForm,
    k: ArrayLike,
    component: int = 0,
    xp=np,
) -> Array:
    r"""Return the base-10 logarithm of a component's DF at given energies.

    Parameters
    ----------
    table : DfTable
        The DF on its energy grid.
    form : AsymptoticForm
        The forms beyond the two ends of the table; the center's are written
        in :math:`u = P_0 - \mathcal{K}`, :math:`P_0` being `form.p_0`.
    k : array-like
        Energies.
    component : int, optional
        The component.
    xp : module, optional
        The array module, ``numpy`` or ``jax.numpy`` (with 64-bit floats).

    Returns
    -------
    log10_df : array, shape of `k`
        Minus infinity where the DF is zero: at and beyond `form.p_0`, at and
        below zero, and below the table where the rule is ``"zero"``.

    Raises
    ------
    ResultError
        If the table is signed (some DF is negative): only a non-negative
        DF has a logarithm.

    Notes
    -----
    In the cell :math:`[\mathcal{K}_i, \mathcal{K}_{i+1}]` the position is
    formed from the nodes themselves,

    .. math::

        t = \ln\!\left(1 + \frac{\mathcal{K} - \mathcal{K}_i}{\mathcal{K}_i}
            \right) \Big/ \ln\!\left(1 + \frac{\mathcal{K}_{i+1} -
            \mathcal{K}_i}{\mathcal{K}_i}\right),

    whose differences are exact where the cell is narrower than a factor of
    two, so no logarithm of the energy is differenced against a stored one.
    The slopes at the nodes are the table's exact ones,
    :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K} =
    \mathcal{K}F'/(F\ln 10)`, where `DfTable.slopes_exact` says the table
    holds them; otherwise those of the not-a-knot cubic spline through the
    nodes.
    """
    arrays, kinds = prepare_lookup(table, form, component)
    if xp is not np:
        arrays = LookupArrays(*(xp.asarray(a) for a in arrays))
    return evaluate_lookup(arrays, kinds, k, xp)


def _nodes(table: DfTable, component: int) -> dict:
    """The nodes of a component's support with the nodes below the table, in NumPy."""
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
        # a last node on the table's first row (where the join is measured)
        # is the table's own row: only the nodes strictly below are read
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
    """The cubic Hermite in ln K, the position from the nodes; the end values
    beyond. Where the depth `u` is finite and the cell's K_i >= P_0 / 2 the
    offset K - K_i is u_i - u."""
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
    """The center's next singular term, c u^e or c u^e (-ln u)."""
    exponent, coefficient = params[NEXT_EXPONENT], params[NEXT_COEFFICIENT]
    u = xp.asarray(u, dtype=xp.float64)
    if kind == "power":
        return coefficient * u**exponent
    return coefficient * u**exponent * (-xp.log(u))


def _above(arrays: LookupArrays, kinds: LookupKinds, u: Array, xp) -> Array:
    r"""log10 F above the table, at :math:`u = P_0 - \mathcal{K}` in (0, u_top).

    The form is anchored at the top node. A power :math:`A u^s` with
    :math:`A > 0` is evaluated relative to itself, so that a steep top whose F
    exceeds the range of a double does not overflow: with
    :math:`r = F_\mathrm{top}/(A u_\mathrm{top}^s)`,
    :math:`F = A u^s[1 + (r - 1)(u_\mathrm{top}/u)^s]` for :math:`s < 0`
    and :math:`F = A u^s[1 + (r - 1)(u/u_\mathrm{top})^{1-s}]` for
    :math:`s > 0`. With a next singular term :math:`G_2` both singular terms
    are kept exact and the rest is held at its top value. The signs of A and
    s choose their branch by `where`: both are numbers of the arrays.
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
    # u <= 0 lies beyond the support, where `evaluate_lookup` returns -inf;
    # there the form is read at the table's top, where it is finite for every
    # exponent (no floor is: s reaches -31 in the battery), so that the side a
    # `where` discards and its gradient stay finite under both array modules
    u = xp.where(u > 0.0, u, u_top)
    s, amplitude = params[CENTRAL_EXPONENT], params[CENTRAL_AMPLITUDE]
    has_next = kinds.next_term != "none" and kinds.central != "analytic"
    g2 = _next_term(kinds.next_term, params, u, xp) if has_next else None
    g_top = params[NEXT_TOP]
    df_top = 10.0**log_df_top
    d2 = (g2 - g_top) if has_next else 0.0
    if kinds.central == "power":
        # A > 0: the power relative to itself
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
        # A <= 0: the power in F itself, the rest held at its top value
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
    r"""log10 F below the table, at :math:`0 < \mathcal{K} < \mathcal{K}_0`.

    The rules: ``"tail"`` :math:`A\mathcal{K}^{-a}[\ln(E_0/\mathcal{K})]^{-b}`,
    ``"taylor"`` :math:`F_1\mathcal{K} + F_2\mathcal{K}^2/2`, ``"edge_law"``
    :math:`\mathcal{K}^p(C_0 + C_1\mathcal{K})`, ``"zero"``. Under JAX the
    tail's argument is held inside :math:`(0, E_0)` and the powers' above
    zero on the side a `where` discards. The tail is one expression under
    both array modules; `prepare_lookup` has refused a tail whose amplitude
    is not positive, and one whose logarithmic factor ends below the table's
    first energy, so on the side kept K lies in (0, E_0).
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
    # K <= 0 lies beyond the support: the rule is read inside its own range
    k = xp.where(k > 0.0, k, 0.5 * params[K_0])
    if rule == "taylor":
        v = params[FLOOR_0] * k + 0.5 * params[FLOOR_1] * k * k
    else:
        p, c_0, c_1 = params[FLOOR_0], params[FLOOR_1], params[FLOOR_2]
        v = k**p * (c_0 + c_1 * k)
    return xp.where(v > 0.0, xp.log10(xp.maximum(v, VALUE_MIN)), -xp.inf)


def below_floor_mass(arrays: LookupArrays, kinds: LookupKinds, k: ArrayLike) -> Array:
    r"""Return the integral of a DF from zero to energies below its table.

    The integral of the rule `_below` evaluates, from the same numbers:
    :math:`F_1\mathcal{K}^2/2 + F_2\mathcal{K}^3/6` for ``"taylor"``,
    :math:`C_0\mathcal{K}^{p+1}/(p+1) + C_1\mathcal{K}^{p+2}/(p+2)` for
    ``"edge_law"``, `tail_mass_below` for ``"tail"`` and zero for ``"zero"``.

    Parameters
    ----------
    arrays, kinds
        A prepared lookup (`prepare_lookup`), in NumPy.
    k : array-like
        Upper limits of the integral, at or below the table's first energy.

    Returns
    -------
    mass : array, shape of `k` with at least one dimension
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
