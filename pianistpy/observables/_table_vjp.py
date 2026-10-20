r"""The transpose of the DF's lookup: a cotangent on F at energies, carried to
the numbers of the DF's table and forms that the lookup reads.

The lookup (`df.lookup`) reads :math:`\log_{10}F` by a cubic Hermite in
:math:`\ln\mathcal{K}` through the table's nodes, by the center's form above
the table and by the rule below it; F is :math:`10^{(\cdot)}`. Its local
derivatives are taken by JAX on the lookup itself (`evaluate_lookup` under
``jax.numpy``, the arrays traced), so the transpose reads the same
expressions as the forward. Three maps lie between the lookup's arrays and
the table's own numbers, and are transposed here on the host:

- the nodes are the table's support with the nodes below it, padded;
- where the table holds no exact slopes, the slopes are those of the
  not-a-knot cubic spline through the nodes in :math:`\ln\mathcal{K}`, a
  linear map of the values (`spline_slopes_vjp`);
- some numbers of the forms are formed from others and from the top node's
  value (`_derived_params`, held against `prepare_lookup`'s own).

The energies (the nodes, the form's :math:`P_0`, the anchor) are the grid,
held fixed.
"""

__all__ = [
    "FORM_LEAVES",
    "DfCotangent",
    "aperture_losvd_vjp",
    "lookup_vjp",
    "losvd_read_vjp",
    "losvd_rows_vjp",
    "speed_distribution_vjp",
    "spline_slopes_vjp",
]

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from ..df import Model
from ..df import lookup as lk
from ..df.lookup import LookupArrays, LookupKinds, evaluate_lookup, prepare_lookup
from ..exceptions import InputError, ResultError
from ..runtime import get_workers
from ..typing import Array, ArrayLike
from ._kept import WORKER
from ._losvd_table import new_pool
from ._speed import angle_integral, angle_pieces, graded_rule, speed_model

# The numeric fields of a component's form that the lookup reads, besides the
# grid: one number each, but the floor's coefficients and the regular part,
# which are tuples.
FORM_LEAVES = (
    "central_exponent",
    "central_amplitude",
    "central_log_coefficient",
    "central_next_exponent",
    "central_next_coefficient",
    "tail_amplitude",
    "tail_exponent",
    "tail_log_exponent",
    "tail_log_scale",
    "below_floor_coefficients",
    "central_anchor_amplitude",
    "central_regular",
)
# The relative agreement asked of the derived numbers of the forms, formed
# here under JAX, with `prepare_lookup`'s, formed on the host.
_DERIVED_TOLERANCE = 1e-13
# The parameters of each family the DF path takes derivatives in: the DF
# table's rule's (`df.gradient.PARAMETERS`) for a child, so not beta_c of an
# Osipkov-Merritt model, which the Jeans path's tangent rows take
# (`TANGENT_PARAMETERS`), nor of an isotropic one (its child's table carries
# no derivative from beta_c = 0, nor its rows, which read G with no L factor).
DF_PARAMETERS = {
    "isotropic": (),
    "constant": ("beta_c",),
    "OM_I": ("r_a",),
    "OM_II": ("r_a",),
    "COM_I": ("beta_c", "r_a"),
}
# Energies whose F is read at once under JAX, which bounds the memory.
_CHUNK = 262_144


@dataclass(frozen=True)
class DfCotangent:
    """A cotangent on the numbers of a component's DF table and forms.

    Parameters
    ----------
    log_df, log_df_slope : array
        On `DfTable.log_df` and `DfTable.log_df_slope`, of their shape; zero
        off the component's support, and on the slopes where the table holds
        none (`slopes_exact` false: the spline's slopes are carried to the
        values).
    log_df_below, log_df_slope_below : array
        On the nodes below the table, of the shape of `DfTable.log_df_below`.
    form : dict
        On each of `FORM_LEAVES` of the component: a float, or a tuple of
        floats for the floor's coefficients and the regular part.
    """

    log_df: Array
    log_df_slope: Array
    log_df_below: Array
    log_df_slope_below: Array
    form: dict[str, Any] = field(default_factory=dict)


def spline_slopes_vjp(x: ArrayLike, cotangent: ArrayLike) -> Array:
    r"""Return the transpose of the not-a-knot cubic spline's slopes at its
    knots, as a map of the values, applied to a cotangent on the slopes.

    Parameters
    ----------
    x : array-like, shape (n,)
        The knots, increasing; n of at least 4.
    cotangent : array-like, shape (n,)
        A cotangent on the slopes :math:`s = S y` that
        ``CubicSpline(x, y, bc_type="not-a-knot")`` holds at the knots.

    Returns
    -------
    cotangent_y : array, shape (n,)
        :math:`S^{\mathsf T} c`.

    Notes
    -----
    SciPy forms the slopes as the solution of :math:`A s = b`, with the
    tridiagonal :math:`A` a function of the knots alone and :math:`b` linear
    in the chords :math:`(y_{i+1} - y_i)/\Delta x_i`. So :math:`S^{\mathsf T}
    c = B^{\mathsf T} A^{-\mathsf T} c`: one banded solve with the band
    transposed, and the chords' transpose. The rows are SciPy's own (its
    ``CubicSpline.__init__``, not-a-knot at both ends, n > 3).
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(cotangent, dtype=np.float64)
    n = x.size
    if n < 4:
        raise ResultError(f"The spline's transpose needs at least 4 knots; got {n}.")
    dx = np.diff(x)
    ab = np.zeros((3, n))
    ab[1, 1:-1] = 2.0 * (dx[:-1] + dx[1:])
    ab[0, 2:] = dx[:-1]
    ab[-1, :-2] = dx[1:]
    ab[1, 0] = dx[1]
    ab[0, 1] = x[2] - x[0]
    ab[1, -1] = dx[-2]
    ab[-1, -2] = x[-1] - x[-3]
    # the band of the transpose: its upper diagonal is A's lower, and back
    ab_t = np.zeros((3, n))
    ab_t[1] = ab[1]
    ab_t[0, 1:] = ab[2, :-1]
    ab_t[2, :-1] = ab[0, 1:]
    z = solve_banded((1, 1), ab_t, c, check_finite=False)
    c_chord = np.zeros(n - 1)
    # interior rows: b_i = 3 (dx_i s_{i-1} + dx_{i-1} s_i), s the chords
    c_chord[:-1] += 3.0 * dx[1:] * z[1:-1]
    c_chord[1:] += 3.0 * dx[:-1] * z[1:-1]
    d = x[2] - x[0]
    c_chord[0] += (dx[0] + 2.0 * d) * dx[1] / d * z[0]
    c_chord[1] += dx[0] ** 2 / d * z[0]
    d = x[-1] - x[-3]
    c_chord[-2] += dx[-1] ** 2 / d * z[-1]
    c_chord[-1] += (2.0 * d + dx[-1]) * dx[-2] / d * z[-1]
    c_y = np.zeros(n)
    c_y[1:] += c_chord / dx
    c_y[:-1] -= c_chord / dx
    return c_y


def _form_numbers(form: Any, component: int) -> dict[str, Any]:
    """The `FORM_LEAVES` of a component, as floats and tuples of floats."""
    out: dict[str, Any] = {}
    for name in FORM_LEAVES:
        values = getattr(form, name)
        if len(values) <= component:
            continue
        value = values[component]
        out[name] = (
            tuple(float(v) for v in value) if isinstance(value, tuple) else float(value)
        )
    return out


def _derived_params(
    arrays: LookupArrays, kinds: LookupKinds, form: Any, component: int
) -> Callable[[Any, dict[str, Any]], Any]:
    """The lookup's numbers of the forms as a function of the top node's
    value and the form's leaves, under JAX: `prepare_lookup`'s formulas, the
    grid's numbers held at the host's, checked against the host's."""
    host = np.asarray(arrays.params)
    rule = kinds.below_floor
    n_floor = len(lk.FLOOR_COEFFICIENTS.get(rule, ()))
    has_next = kinds.next_term != "none" and kinds.central != "analytic"
    u_top = float(host[lk.U_TOP])
    positive = float(host[lk.CENTRAL_AMPLITUDE]) > 0.0

    def params_of(y_top: Any, leaves: dict[str, Any]) -> Any:
        p = jnp.asarray(host)
        s = leaves["central_exponent"]
        amplitude = leaves["central_amplitude"]
        log_amplitude = jnp.log10(amplitude) if positive else 0.0
        p = p.at[lk.LOG_DF_TOP].set(y_top)
        p = p.at[lk.CENTRAL_EXPONENT].set(s)
        p = p.at[lk.CENTRAL_AMPLITUDE].set(amplitude)
        p = p.at[lk.LOG_COEFFICIENT].set(leaves["central_log_coefficient"])
        p = p.at[lk.NEXT_EXPONENT].set(leaves["central_next_exponent"])
        p = p.at[lk.NEXT_COEFFICIENT].set(leaves["central_next_coefficient"])
        p = p.at[lk.LOG_AMPLITUDE].set(log_amplitude)
        p = p.at[lk.R_MINUS_1].set(
            jnp.expm1(math.log(10.0) * (y_top - log_amplitude - s * math.log10(u_top)))
        )
        g_top = (
            lk._next_term(kinds.next_term, p, jnp.asarray(u_top), jnp)
            if has_next
            else 0.0
        )
        p = p.at[lk.NEXT_TOP].set(g_top)
        p = p.at[lk.NEXT_TOP_RELATIVE].set(
            g_top * 10.0 ** (-(log_amplitude + s * math.log10(u_top)))
        )
        tail = leaves["tail_amplitude"]
        p = p.at[lk.TAIL_AMPLITUDE].set(tail)
        p = p.at[lk.TAIL_LOG_AMPLITUDE].set(jnp.log10(tail) if rule == "tail" else 0.0)
        p = p.at[lk.TAIL_EXPONENT].set(leaves["tail_exponent"])
        p = p.at[lk.TAIL_LOG_EXPONENT].set(leaves["tail_log_exponent"])
        p = p.at[lk.TAIL_LOG_SCALE].set(leaves["tail_log_scale"])
        for i in range(n_floor):
            p = p.at[lk.FLOOR_0 + i].set(leaves["below_floor_coefficients"][i])
        if kinds.anchored:
            p = p.at[lk.ANCHOR_AMPLITUDE].set(leaves["central_anchor_amplitude"])
            p = p.at[lk.REGULAR_B].set(leaves["central_regular"][0])
            p = p.at[lk.REGULAR_D].set(leaves["central_regular"][1])
        return p

    formed = np.asarray(
        params_of(float(host[lk.LOG_DF_TOP]), _form_numbers(form, component)),
        dtype=np.float64,
    )
    with np.errstate(invalid="ignore"):
        scale = np.maximum(np.abs(host), np.abs(formed))
        off = np.abs(formed - host) > _DERIVED_TOLERANCE * scale
    off &= ~(np.isnan(formed) & np.isnan(host))
    if np.any(off):
        raise ResultError(
            "The lookup's numbers of the forms, formed again for the transpose, "
            f"differ from prepare_lookup's at {np.flatnonzero(off).tolist()}."
        )
    return params_of


def _lookup_parts(model: Model) -> tuple[Any, Any, Any, Any]:
    """The prepared lookup of a model's one component, the function of its
    forms' numbers (`_derived_params`), its arrays under JAX and the reader
    of F under JAX: ``read(y, m, params, k, k_c)``, F as the observables
    read it (zero at and below `k_c` and where the lookup is minus
    infinity)."""
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    params_of = _derived_params(arrays, kinds, model.form, 0)
    leaves, read = _lookup_reader(arrays, kinds)
    return arrays, params_of, leaves, read


def _lookup_reader(arrays: LookupArrays, kinds: LookupKinds) -> tuple[Any, Any]:
    """The prepared lookup's arrays under JAX and its reader of F under JAX
    (`_lookup_parts`), from the lookup alone."""
    k_nodes, width, count = (
        jnp.asarray(arrays.k),
        jnp.asarray(arrays.width),
        jnp.asarray(arrays.n),
    )

    def read(
        y: Any, m: Any, params: Any, k: Any, k_c: float, shift: float = 0.0
    ) -> Any:
        a = LookupArrays(k_nodes, y, m, width, count, params)
        log_f = evaluate_lookup(a, kinds, k, jnp)
        keep = (k > k_c) & jnp.isfinite(log_f)
        return jnp.where(keep, 10.0 ** jnp.where(keep, log_f - shift, 0.0), 0.0)

    leaves = tuple(jnp.asarray(a) for a in (arrays.log_df, arrays.slope, arrays.params))
    return leaves, read


def _table_cotangent(
    model: Model, arrays: LookupArrays, params_of: Any, c_y: Any, c_m: Any, c_p: Any
) -> DfCotangent:
    """Carry a cotangent on the lookup's arrays to the table and the forms."""
    table, form = model.table, model.form
    n = int(arrays.n)
    c_y = np.array(c_y, dtype=np.float64)[:n]
    c_m = np.array(c_m, dtype=np.float64)[:n]
    # the forms' numbers: to the top node's value and the form's leaves
    top = jnp.asarray(float(arrays.params[lk.LOG_DF_TOP]))
    _, back = jax.vjp(params_of, top, _form_numbers(form, 0))
    d_top, d_leaves = back(jnp.asarray(c_p, dtype=jnp.float64))
    c_y[n - 1] += float(d_top)
    form_cotangent = {
        name: (
            tuple(float(v) for v in value) if isinstance(value, tuple) else float(value)
        )
        for name, value in d_leaves.items()
    }
    # the slopes: the table's own, or the spline's of the values
    if not table.slopes_exact[0]:
        c_y += spline_slopes_vjp(np.log(np.asarray(arrays.k)[:n]), c_m)
        c_m = np.zeros(n)
    # the nodes: those below the table the lookup keeps, then the support
    first, last = table.i_first[0], table.i_last[0]
    n_kept = n - (last + 1 - first)
    out = dict(
        log_df=np.zeros(table.log_df.shape),
        log_df_slope=np.zeros(table.log_df_slope.shape),
        log_df_below=np.zeros(table.log_df_below.shape),
        log_df_slope_below=np.zeros(table.log_df_slope_below.shape),
    )
    out["log_df"][0, first : last + 1] = c_y[n_kept:]
    out["log_df_slope"][0, first : last + 1] = c_m[n_kept:]
    if n_kept:
        kb = table.k_below[0, : table.nodes_below(0)]
        kept = np.flatnonzero(kb < table.k[0, first] * (1.0 - 1e-12))
        out["log_df_below"][0, kept] = c_y[:n_kept]
        out["log_df_slope_below"][0, kept] = c_m[:n_kept]
    return DfCotangent(**out, form=form_cotangent)


def lookup_vjp(
    model: Model, k: ArrayLike, cotangent: ArrayLike, k_c: float = 0.0
) -> DfCotangent:
    r"""Return the cotangent on a component's DF table and forms of a
    cotangent on F at energies.

    Parameters
    ----------
    model : Model
        A model of one component.
    k : array-like
        Energies, where F is read as the observables read it:
        :math:`10^{\log_{10}F}` by the lookup, zero at and below `k_c` and
        where the lookup is minus infinity.
    cotangent : array-like, shape of `k`
        A cotangent on F at `k`.
    k_c : float, optional
        The floor of the support (the lookup's first energy where the DF is
        zero below it).

    Returns
    -------
    cotangent : DfCotangent
    """
    arrays, params_of, leaves, read = _lookup_parts(model)
    sums = _read_sums(leaves, read, k, cotangent, k_c)
    return _table_cotangent(model, arrays, params_of, *sums)


def _read_sums(
    leaves: Any,
    read: Any,
    k: ArrayLike,
    cotangent: ArrayLike,
    k_c: float = 0.0,
    shift: float = 0.0,
) -> list[Array]:
    """The cotangent on the lookup's arrays (its leaves) of a cotangent on F
    at energies, summed over chunks of the energies (`lookup_vjp`); F read
    as :math:`10^{\\log_{10}F - c}` with the shift c inside the exponent, so
    a table whose F passes the float64 range reads a finite F."""
    k = np.asarray(k, dtype=np.float64).ravel()
    c = np.asarray(cotangent, dtype=np.float64).ravel()
    sums = [np.zeros(np.shape(a)) for a in leaves]
    for start in range(0, k.size, _CHUNK):
        kk = jnp.asarray(k[start : start + _CHUNK])
        _, back = jax.vjp(lambda y, m, p, kk=kk: read(y, m, p, kk, k_c, shift), *leaves)
        parts = back(jnp.asarray(c[start : start + _CHUNK]))
        for i, part in enumerate(parts):
            sums[i] += np.asarray(part)
    return sums


def _df_parameters(model: Model, parameters: Any) -> tuple[str, ...]:
    """The anisotropy parameters of the explicit part, refused where the DF
    path does not take them: COM Type II (the DF table's rule refuses it:
    its interior point moves with both parameters) and a parameter the
    family does not vary in."""
    names = tuple(parameters)
    family = model.components[0].anisotropy.family
    if names and family == "COM_II":
        raise InputError(
            "The DF path takes no derivative in the anisotropy of a COM Type II "
            "model: the DF table's rule refuses it; its second moments take "
            "theirs by the Jeans path (dispersion_profile's tangents)."
        )
    for name in names:
        if name not in DF_PARAMETERS[family]:
            raise InputError(
                f"A {family} model's derivatives are taken in "
                f"{list(DF_PARAMETERS[family])}; got {name!r}."
            )
    return names


def speed_distribution_vjp(
    result: Any, cotangent: ArrayLike, parameters: Any = ()
) -> tuple[DfCotangent, dict[str, float]]:
    r"""Return the transpose of a speed distribution: a cotangent on its
    values, carried to the DF's table and forms and to the anisotropy.

    Parameters
    ----------
    result : SpeedDistribution
        The distribution (`speed_distribution`), its rule held: the speeds
        `v_nodes`, the angle rule its record checked, and every piece of
        the angle integral where it lies.
    cotangent : array-like, shape of ``result.pdf``
        A cotangent on :math:`p(V\,|\,r)` at `v_nodes`.
    parameters : sequence of {"beta_c", "r_a"}, optional
        The anisotropy parameters of the explicit part: the derivative at a
        fixed DF table and at fixed speeds.

    Returns
    -------
    table : DfCotangent
        The cotangent on the DF's table and forms (`lookup_vjp`'s).
    explicit : dict of str to float
        The cotangent on each parameter named.

    Raises
    ------
    ResultError
        If the distribution recomputed by its rule is not the one given, bit
        for bit.
    InputError
        For a parameter the DF path does not take (`_df_parameters`).

    Notes
    -----
    The distribution is recomputed on the host by `angle_integral` with the
    record's rule and held equal to the one given, bit for bit, so the rule
    below is the forward's. Its derivative is taken by JAX on a twin of the
    angle integral: the same pieces in t (`angle_pieces`, each edge by what
    sets it), the same graded rule mapped onto each piece, the same
    substitution :math:`1 - t = w^{1/(1-b)}` on the last piece where
    :math:`b > 0`, F read by the lookup traced in its arrays. The edges move
    with :math:`\xi = r^2/R_{\rm a}^2` and the substitution with
    :math:`b = \beta_{\rm c}`, so the explicit part holds the edges'
    motion: where F is not zero at the support's edge (a floor rule
    ``"zero"``), that is the boundary term, from the rule's own nodes.
    """
    from .stages import _speed_point

    model = result.model
    names = _df_parameters(model, parameters)
    sm = speed_model(model)
    r = float(result.r)
    psi, xi, scale = _speed_point(model, sm, r)
    v = np.asarray(result.v_nodes, dtype=np.float64)
    n_nodes = int(result.record["n_angle_nodes"])
    n_panels = int(result.record["n_angle_panels"])
    again = (
        scale
        * v ** (2.0 - 2.0 * sm.b)
        * angle_integral(sm, psi, xi, v, n_nodes, n_panels)
    )
    if again.tobytes() != np.asarray(result.pdf).tobytes():
        raise ResultError(
            "The speed distribution recomputed by its record's rule is not the "
            "one given: its rule is not the forward's, so it has no transpose here."
        )
    # the pieces of every speed, each edge by what sets it
    codes = {"0": 0, "1": 1, "root": 2, "join": 3}
    speed_of, lo_by, hi_by, substituted = [], [], [], []
    for j, vj in enumerate(v):
        _, _, edges, by = angle_pieces(sm, psi, xi, float(vj))
        last = len(edges) - 2
        for i in range(len(edges) - 1):
            if not edges[i + 1] > edges[i]:
                continue
            speed_of.append(j)
            lo_by.append(codes[by[i]])
            hi_by.append(codes[by[i + 1]])
            substituted.append(sm.b > 0.0 and i == last and edges[i + 1] == 1.0)
    unit_t, unit_w = (jnp.asarray(a) for a in graded_rule(0.0, 1.0, n_panels, n_nodes))
    speed_of_a = jnp.asarray(speed_of, dtype=jnp.int32)
    lo_by_a, hi_by_a = jnp.asarray(lo_by), jnp.asarray(hi_by)
    substituted_a = jnp.asarray(substituted)
    arrays, params_of, leaves, read = _lookup_parts(model)
    rho = 4.0 * np.pi * r ** (-2.0 * sm.b) / scale
    v_j = jnp.asarray(v)
    sign, k_c, k_w = sm.sign, sm.k_c, sm.k_w

    def edge(by: Any, c0: Any, c2: Any) -> Any:
        # the square root is taken only where an edge is set by it, inside
        # (0, 1) where the host found it, so that the side a `where` discards
        # carries no infinite slope
        safe = jnp.where(c2 != 0.0, c2, 1.0)
        arg = jnp.where(by == 2, (k_c - c0) / safe, (k_w - c0) / safe)
        moving = (by >= 2) & (arg > 0.0) & (arg < 1.0)
        root = jnp.sqrt(jnp.where(moving, arg, 0.25))
        return jnp.where(by == 0, 0.0, jnp.where(by == 1, 1.0, root))

    def pdf(y: Any, m: Any, p: Any, b: Any, x: Any) -> Any:
        vv = v_j[speed_of_a]
        c0 = psi - 0.5 * vv * vv * (1.0 + sign * x)
        c2 = 0.5 * sign * x * vv * vv
        a = edge(lo_by_a, c0, c2)
        e = edge(hi_by_a, c0, c2)
        # the plain pieces: the graded rule on [a, e]
        t = a[:, None] + (e - a)[:, None] * unit_t[None, :]
        w = (e - a)[:, None] * unit_w[None, :]
        f_plain = read(y, m, p, c0[:, None] + c2[:, None] * t * t, k_c)
        # a node the map rounds onto t = 1 carries no weight's worth (the
        # forward's own nodes stay inside): it is left out, so that neither
        # (1 - t^2)^(-b) nor its slope is infinite there
        one = 1.0 - t * t
        inner = one > 0.0
        factor = jnp.where(inner, jnp.where(inner, one, 1.0) ** (-b), 0.0)
        plain = jnp.sum(w * factor * f_plain, axis=1)
        # the substituted last piece: 1 - t = w^p on [0, (1 - a)^(1 - b)]
        if bool(np.any(substituted)):
            power = 1.0 / (1.0 - b)
            top = jnp.where(substituted_a, 1.0 - a, 1.0) ** (1.0 - b)
            ws = top[:, None] * unit_t[None, :]
            om = ws**power
            ts = 1.0 - om
            f_sub = read(y, m, p, c0[:, None] + c2[:, None] * ts * ts, k_c)
            sub = power * jnp.sum(
                top[:, None] * unit_w[None, :] * (2.0 - om) ** (-b) * f_sub, axis=1
            )
            total = jnp.where(substituted_a, sub, plain)
        else:
            total = plain
        integral = jax.ops.segment_sum(total, speed_of_a, num_segments=v.size)
        return 4.0 * jnp.pi * r ** (-2.0 * b) / rho * v_j ** (2.0 - 2.0 * b) * integral

    b0, x0 = jnp.asarray(sm.b), jnp.asarray(xi)
    value, back = jax.vjp(pdf, *leaves, b0, x0)
    scale_value = float(np.max(np.abs(result.pdf)))
    if not np.allclose(
        np.asarray(value), result.pdf, rtol=0.0, atol=1e-9 * scale_value
    ):
        raise ResultError(
            "The twin of the angle integral does not reproduce the speed "
            "distribution to 1e-9 of its largest value."
        )
    c_y, c_m, c_p, c_b, c_x = back(jnp.asarray(cotangent, dtype=jnp.float64))
    explicit: dict[str, float] = {}
    for name in names:
        if name == "beta_c":
            explicit[name] = float(c_b)
        else:
            explicit[name] = float(c_x) * (-2.0 * xi / sm.r_a)
    return _table_cotangent(model, arrays, params_of, c_y, c_m, c_p), explicit


# The relative agreement asked of G's twin under JAX with the route's own G.
_G_TOLERANCE = 1e-12


def _cumulative_order(arrays: LookupArrays, kinds: LookupKinds, q: float) -> int:
    """The Gauss-Legendre order whose panels `cumulative_df` returns at q
    (its doubling, run again on the host); 0 at or below the first energy,
    where the value is the rule's closed form."""
    from ._ellipse_model import (
        CUMULATIVE_FIRST_ORDER,
        CUMULATIVE_LAST_ORDER,
        CUMULATIVE_TOLERANCE,
        _cumulative_at_order,
    )

    if q <= float(arrays.params[lk.K_0]):
        return 0
    order = CUMULATIVE_FIRST_ORDER
    value = _cumulative_at_order(arrays, kinds, q, order)
    while order < CUMULATIVE_LAST_ORDER:
        order *= 2
        new = _cumulative_at_order(arrays, kinds, q, order)
        if abs(new - value) <= CUMULATIVE_TOLERANCE * abs(new):
            return order
        value = new
    return order


def _below_mass(params: Any, kinds: LookupKinds, k: float) -> Any:
    """`below_floor_mass` under JAX, in the lookup's numbers; refused for the
    rule "tail", whose closed form is not written here."""
    rule = kinds.below_floor
    if rule == "taylor":
        return params[lk.FLOOR_0] * k * k / 2.0 + params[lk.FLOOR_1] * k**3 / 6.0
    if rule == "edge_law":
        p, c_0, c_1 = params[lk.FLOOR_0], params[lk.FLOOR_1], params[lk.FLOOR_2]
        return c_0 * k ** (p + 1.0) / (p + 1.0) + c_1 * k ** (p + 2.0) / (p + 2.0)
    if rule == "zero":
        return 0.0
    raise ResultError(
        "The transpose of G below a table whose rule is 'tail' is not written: "
        "its closed form (tail_mass_below) has no twin under JAX here."
    )


def _cumulative_twin(
    arrays: LookupArrays, kinds: LookupKinds, a: LookupArrays, q: float, order: int
) -> Any:
    """`cumulative_df` at q under JAX, at the order the host's doubling ended
    on: the same panels and nodes, F read by the traced lookup `a`."""
    k_0 = float(arrays.params[lk.K_0])
    if order == 0:
        return _below_mass(a.params, kinds, q)
    n = int(arrays.n)
    k_nodes = np.asarray(arrays.k[:n], dtype=np.float64)
    half = 0.5 * q
    nodes = np.unique(
        np.concatenate(
            [
                [k_0],
                k_nodes[(k_nodes > k_0) & (k_nodes < q)],
                [half] if k_0 < half < q else [],
                [q],
            ]
        )
    )
    lo, hi = nodes[:-1], nodes[1:]
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    xg, wg = np.polynomial.legendre.leggauss(order)
    low = hi <= half
    total = _below_mass(a.params, kinds, k_0)
    if np.any(low):
        ua, ub = np.log(lo[low]), np.log(hi[low])
        um, uw = 0.5 * (ua + ub), 0.5 * (ub - ua)
        u = um[:, None] + uw[:, None] * xg[None, :]
        k = np.clip(np.exp(u), k_0, q)
        ln_f = evaluate_lookup(a, kinds, jnp.asarray(k), jnp) * math.log(10.0)
        total = total + jnp.sum(uw[:, None] * wg[None, :] * jnp.exp(ln_f + u))
    if np.any(~low):
        ah, bh = lo[~low], hi[~low]
        ta = np.log(q - ah)
        with np.errstate(divide="ignore"):
            tb = np.log(np.maximum(q - bh, 0.0))
        tb = np.where(np.isfinite(tb) & (tb > ta - 60.0), tb, ta - 60.0)
        hm, hw = 0.5 * (ta + tb), 0.5 * (ta - tb)
        t = hm[:, None] + hw[:, None] * xg[None, :]
        k = np.clip(q - np.exp(t), k_0, q)
        ln_f = evaluate_lookup(a, kinds, jnp.asarray(k), jnp) * math.log(10.0)
        total = total + jnp.sum(hw[:, None] * wg[None, :] * jnp.exp(ln_f + t))
    return total


def _cum_twin(g: Any, h: float, base: Any) -> Any:
    """`EllipseDF._cum` under JAX."""
    c = jnp.zeros(g.shape[0])
    c = c.at[0].set(base)
    c = c.at[1:].set(base + jnp.cumsum(0.5 * h * (g[1:] + g[:-1])))
    c = c.at[2::2].set(
        base + jnp.cumsum(h / 3.0 * (g[0:-2:2] + 4.0 * g[1:-1:2] + g[2::2]))
    )
    return c.at[1:-1:2].set(
        0.5 * (c[0:-2:2] + c[2::2]) + (h / 4.0) * (g[0:-2:2] - g[2::2])
    )


def _g_twin(model: Model, edf: Any, shift: float) -> Callable[..., Any]:
    r"""`EllipseDF.G` under JAX, a function of the lookup's arrays (y, m,
    params) and of energies: its three branches on their own grids, their
    bases from the lookup's cumulative integral at the order the host's
    doubling ended on, and the reading between nodes, as the route made
    them; F is the route's, :math:`10^{\log_{10}F - c}`."""
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    q_low = 1e-40 * edf.Q0
    orders = (
        _cumulative_order(arrays, kinds, edf.Q0),
        _cumulative_order(arrays, kinds, q_low),
    )
    fixed = (jnp.asarray(arrays.k), jnp.asarray(arrays.width), jnp.asarray(arrays.n))
    lq, lb = edf._lq, edf._lb
    yu = edf._yu[::-1]  # the branch's own order, from Q_mid toward Q_end
    scale = edf.scale

    def g_of(y: Any, m: Any, params: Any, q: Any, held: Any = None) -> Any:
        a = LookupArrays(fixed[0], y, m, fixed[1], fixed[2], params)

        def f(x: Any) -> Any:
            log_f = evaluate_lookup(a, kinds, x, jnp)
            ok = jnp.isfinite(log_f)
            return jnp.where(ok, 10.0 ** (jnp.where(ok, log_f, 0.0) - shift), 0.0)

        base = _cumulative_twin(arrays, kinds, a, edf.Q0, orders[0]) * scale
        gl = _cum_twin(f(jnp.exp(lq)) * np.exp(lq), lq[1] - lq[0], base)
        gu = _cum_twin(f(edf.Qend - np.exp(yu)) * np.exp(yu), yu[0] - yu[1], gl[-1])
        gb = _cum_twin(
            f(jnp.exp(lb)) * np.exp(lb),
            lb[1] - lb[0],
            _cumulative_twin(arrays, kinds, a, q_low, orders[1]) * scale,
        )
        lgl = jnp.log(jnp.maximum(gl, 1e-300))
        lgu = jnp.log(jnp.maximum(gu, 1e-300))[::-1]
        lgb = jnp.log(jnp.maximum(gb, 1e-300))
        sb = (lgb[1] - lgb[0]) / (lb[1] - lb[0])
        q = jnp.asarray(q)
        in_a = (q > 0.0) & (q < edf.Q0)
        lqa = jnp.log(jnp.where(in_a, q, edf.Q0))
        g_a = jnp.exp(
            jnp.where(
                lqa >= lb[0],
                jnp.interp(lqa, lb, lgb),
                lgb[0] + sb * (lqa - lb[0]),
            )
        )
        in_b = (q >= edf.Q0) & (q <= edf.Qmid)
        g_b = jnp.exp(jnp.interp(jnp.log(jnp.where(in_b, q, edf.Q0)), lq, lgl))
        in_c = q > edf.Qmid
        below = edf.Qend - jnp.where(in_c, q, edf.Qmid)
        if held is not None and edf.depth_end:
            held = jnp.broadcast_to(jnp.asarray(held, dtype=jnp.float64), q.shape)
            below = jnp.where(in_c & jnp.isfinite(held), held, below)
        depth = jnp.log(jnp.maximum(below, 1e-13 * edf.Qend))
        g_c = jnp.exp(jnp.interp(depth, edf._yu, lgu))
        return jnp.where(in_a, g_a, jnp.where(in_b, g_b, jnp.where(in_c, g_c, 0.0)))

    return g_of


def _add(a: DfCotangent, b: DfCotangent) -> DfCotangent:
    """The sum of two cotangents on one table."""
    form = dict(a.form)
    for name, value in b.form.items():
        if name in form:
            form[name] = (
                tuple(x + y for x, y in zip(form[name], value, strict=True))
                if isinstance(value, tuple)
                else form[name] + value
            )
        else:
            form[name] = value
    return DfCotangent(
        log_df=a.log_df + b.log_df,
        log_df_slope=a.log_df_slope + b.log_df_slope,
        log_df_below=a.log_df_below + b.log_df_below,
        log_df_slope_below=a.log_df_slope_below + b.log_df_slope_below,
        form=form,
    )


def _row_levels(
    route: Any, r2d: float, v_nodes: Array, v_weights: Array, rule: Any
) -> tuple[Array, Array, Array, list[tuple[int, int, Array]]]:
    """Rerun a row's own rule (`EllipseRoute.row`) with its trace: the row,
    the distinct speeds, the inverse map to the row's nodes and each level."""
    trace: list[tuple[int, int, Array]] = []
    d = route.row(r2d, v_nodes, rule, w=v_weights, trace=trace)
    va, inv = np.unique(
        np.abs(np.asarray(v_nodes, dtype=np.float64)), return_inverse=True
    )
    return d, va, inv, trace


def _row_support(
    route: Any, r2d: float, va: Array
) -> tuple[Array, Array, Array, Array]:
    """The speeds' supports in t, split as the row splits them."""
    jv, tl, th = route.supports(r2d, va)
    return route.l0_split(r2d, va, jv, tl, th)


def _row_by_levels(
    route: Any,
    r2d: float,
    va: Array,
    rule: Any,
    trace: list[tuple[int, int, Array]],
    collect: list[tuple[Any, ...]] | None = None,
    support: tuple[Array, Array, Array, Array] | None = None,
) -> Array:
    """D on the distinct speeds `va`, each from the level of the trace where
    it was last computed, the rule's own nodes at that level; on the
    `support` in t given (`_row_support`, the t-nodes held), or the route's
    own."""
    level = np.full(len(va), -1)
    for i, (_, _, mask) in enumerate(trace):
        level[mask] = i
    jv, tl, th, cut = _row_support(route, r2d, va) if support is None else support
    out = np.zeros(len(va))
    for i in np.unique(level):
        speeds = level == i
        sel = speeds[jv]
        if not sel.any():
            continue
        n_p, n_lay, _ = trace[i]
        d = route._row_at(
            r2d, va, jv[sel], tl[sel], th[sel], rule, n_p, n_lay, cut[sel], collect
        )
        out[speeds] = d[speeds]
    return out


def _row_transpose(
    route: Any,
    i: int,
    r2d: float,
    v_nodes: Array,
    v_weights: Array,
    losvd: Array,
    c: Array,
) -> tuple[list[tuple[Array, Array]], list[tuple[Array, Array]]]:
    """One row's transpose: the row rerun and held to the stored one, then
    the energies of every read of F and of G with the cotangent each takes."""
    rule, _ = route.rule(r2d)
    vp, wp = route.panels(r2d)[:2]
    if vp.tobytes() != v_nodes.tobytes() or wp.tobytes() != v_weights.tobytes():
        raise ResultError(f"Row {i}: its panels recomputed are not the stored ones.")
    d, va, inv, trace = _row_levels(route, r2d, vp, wp, rule)
    norm = 2.0 * float(np.dot(wp, d))
    if (d / norm).tobytes() != losvd.tobytes():
        raise ResultError(
            f"Row {i} recomputed by its own rule is not the row stored: its rule "
            "is not the forward's, so it has no transpose here."
        )
    collect: list[tuple[Any, ...]] = []
    by_speed = _row_by_levels(route, r2d, va, rule, trace, collect)
    if by_speed[inv].tobytes() != d.tobytes():
        raise ResultError(f"Row {i} recomputed level by level is not the row.")
    c_d = (c - 2.0 * wp * float(c @ (d / norm))) / norm
    c_speed = np.bincount(inv, weights=c_d, minlength=len(va))
    f_reads: list[tuple[Array, Array]] = []
    g_reads: list[tuple[Array, Array]] = []
    for kind, speed, k, w in collect:
        if kind == "P":
            continue
        into = f_reads if kind == "F" else g_reads
        into.append((np.asarray(k).ravel(), (c_speed[speed][:, None] * w).ravel()))
    return f_reads, g_reads


def _row_cotangent(
    route: Any, lookup: tuple[LookupArrays, LookupKinds], shift: float, *args: Any
) -> tuple[list[Array], list[tuple[Array, Array]]]:
    """One row's transpose (`_row_transpose`) with its reads of F carried
    onto the lookup's arrays, the zero weights dropped and F read with the
    route's shift inside the exponent: the row returns arrays the size of the
    table's nodes, never its reads; its reads of G as they are."""
    f_reads, g_reads = _row_transpose(route, *args)
    leaves, read = _lookup_reader(*lookup)
    k = np.concatenate([k for k, _ in f_reads]) if f_reads else np.zeros(0)
    c = np.concatenate([w for _, w in f_reads]) if f_reads else np.zeros(0)
    keep = c != 0.0
    return _read_sums(leaves, read, k[keep], c[keep], 0.0, shift), g_reads


def _row_cotangent_worker(
    args: tuple[Any, ...],
) -> tuple[list[Array], list[tuple[Array, Array]]]:
    """`_row_cotangent` in a pool worker, on the worker's route with the
    kept ranges of the table."""
    route = WORKER["route"]
    route.vrad_kept, route.vk_kept = args[6]
    return _row_cotangent(route, args[7], args[8], *args[:6])


def losvd_rows_vjp(
    model: Model, table: Any, settings: Any, cotangent: Any
) -> DfCotangent:
    r"""Return the transpose of a LOSVD table's rows: cotangents on the rows,
    carried to the DF's table and forms.

    Parameters
    ----------
    model : Model
        The model of one component the table belongs to.
    table : LosvdTable
        Its table (`losvd_table`), each row's rule held.
    settings : Settings
        The settings the table was made with.
    cotangent : sequence of array or None
        One per row: a cotangent on the row's normalized values
        ``row.losvd[0]`` at its nodes, or None where there is none.

    Returns
    -------
    table : DfCotangent

    Raises
    ------
    ResultError
        Where a row recomputed by its own rule is not the row stored, bit for
        bit.

    Notes
    -----
    Each row with a nonzero cotangent is recomputed by its rule
    (`EllipseRoute.row`, its trace kept) and held equal to the stored row, bit
    for bit. Its normalization's transpose is :math:`c_D = (c - 2w\,(c\cdot\hat
    D))/N`; each distinct speed is then computed again at the level its value
    came from, with the energies and weights of every read of F collected
    (`EllipseRoute.plane`), and the cotangent on F at those energies goes to
    the lookup's transpose within the row (`lookup_vjp`'s, onto the lookup's
    arrays): a row returns arrays the size of the table's nodes, never its
    reads, and the rows' arrays are summed. The route reads F as
    :math:`10^{\log_{10}F - c}`, the shift c held (it cancels in a
    normalized row). Held with it: the rows' radii, the kept ranges and the
    layouts, the panels and layers of every level, the angle counts, and the
    map toward K = 0 with the edge law's exponent as the route took it.

    The rows are transposed in a pool of `get_workers` processes, each row by
    the same function as in one process (`_row_cotangent`), and their arrays
    summed in row order: the result is the same to the bit for any number of
    workers.
    """
    from ._ellipse_model import DfReading, ellipse_route
    from ._losvd_stages import ellipse_rules

    route = ellipse_route(model, ellipse_rules(settings))
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    reading = route._df().F
    if not isinstance(reading, DfReading):
        raise ResultError("The route reads its DF by no lookup: no transpose here.")
    shift = reading.shift
    arrays, params_of, leaves, _ = _lookup_parts(model)
    lookup = prepare_lookup(model.table, model.form, 0)
    args = [
        (
            i,
            float(row.r_2d),
            row.v_nodes,
            row.v_weights,
            np.asarray(row.losvd[0]),
            np.asarray(cotangent[i], dtype=np.float64).ravel(),
            (route.vrad_kept, route.vk_kept),
            lookup,
            shift,
        )
        for i, row in enumerate(table.rows)
        # a row whose cotangent is zero adds nothing: every read of it carries a
        # zero weight, which the join drops
        if cotangent[i] is not None and np.any(np.asarray(cotangent[i]))
    ]
    n_workers = min(get_workers(), len(args))
    if n_workers > 1:
        # the rows in the pool, each by the same function; summed in row order
        with new_pool(n_workers, route) as pool:
            parts = pool.map(_row_cotangent_worker, args, chunksize=1)
    else:
        parts = [_row_cotangent(route, a[7], a[8], *a[:6]) for a in args]
    # each row's cotangent on the lookup's arrays, summed in row order
    sums = [np.zeros(np.shape(a)) for a in leaves]
    for row_sums, _ in parts:
        for i, part in enumerate(row_sums):
            sums[i] += part
    g_energies = [k for _, g in parts for k, _ in g]
    g_weights = [w for _, g in parts for _, w in g]
    out = _table_cotangent(model, arrays, params_of, *sums)
    if g_energies:
        # the rows read G: its twin's transpose, held to the route's G first
        edf = route._df()
        q = np.concatenate(g_energies)
        c_g = np.concatenate(g_weights)
        g_of = _g_twin(model, edf, shift)
        arrays, params_of, leaves, _ = _lookup_parts(model)
        value, back = jax.vjp(lambda y, m, p: g_of(y, m, p, q), *leaves)
        host = edf.G(q)
        scale_g = float(np.max(np.abs(host))) if host.size else 0.0
        if not np.allclose(
            np.asarray(value), host, rtol=_G_TOLERANCE, atol=_G_TOLERANCE * scale_g
        ):
            raise ResultError(
                "G's twin does not reproduce the route's G to "
                f"{_G_TOLERANCE:.0e}: its transpose is not that of the forward."
            )
        c_y, c_m, c_p = back(jnp.asarray(c_g))
        out = _add(out, _table_cotangent(model, arrays, params_of, c_y, c_m, c_p))
    return out


def _row_weights(stencil: Any, i: int, p: Array, xi: Array) -> tuple[Array, Array]:
    """The weights of `PanelTable._row_at`: for each position, the indices
    into the table's flat values and the weight of each, so that its value
    is the sum of the products. A panel the row does not have weighs
    nothing."""
    m = stencil.lattice.rules.vp_nodes
    half0 = int(stencil.off[i]) + (int(stencil.off[i + 1]) - int(stencil.off[i])) // 2
    units = stencil.lattice.panel_units(int(stencil._layout[i]))
    nsub = np.concatenate([[0], np.cumsum([len(u) - 1 for u in units])])
    index = np.zeros((len(xi), m), dtype=np.int64)
    weight = np.zeros((len(xi), m))
    for pp in np.unique(p):
        if pp >= int(stencil.npan[i]):
            continue
        sel = np.flatnonzero(p == pp)
        e = units[pp]
        k = np.clip(np.searchsorted(e, xi[sel], side="right") - 1, 0, len(e) - 2)
        t = (xi[sel] - e[k]) / (e[k + 1] - e[k])
        base = half0 + m * int(nsub[pp])
        dd = t[:, None] - stencil._xg[None, :]
        hit = dd == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            c = stencil._bw[None, :] / dd
            w = c / np.sum(c, axis=1)[:, None]
        anyhit = hit.any(axis=1)
        w[anyhit] = hit[anyhit].astype(np.float64)
        index[sel] = (base + m * k)[:, None] + np.arange(m)[None, :]
        weight[sel] = w
    return index, weight


def _stencil_read_vjp(
    stencil: Any,
    route: Any,
    r_2d: Array,
    v: Array,
    cotangent: Array,
    curves: dict[str, Array] | None = None,
) -> list[Array]:
    """The transpose of `PanelTable.at_pairs` (with the breaks `curves` at
    the radii, where it was given them), onto each row's normalized values
    on ``V >= 0``."""
    c = np.asarray(cotangent, dtype=np.float64).ravel()
    r2, brk, npan, vend, rows, lag = stencil._prepare(
        route, np.asarray(r_2d, dtype=np.float64).ravel(), curves
    )
    a = np.abs(np.asarray(v, dtype=np.float64).ravel())
    flat = np.zeros(stencil.D.shape)
    live = np.flatnonzero(a < vend)
    if live.size:
        p = np.minimum((brk[live, 1:] <= a[live, None]).sum(axis=1), npan[live] - 1)
        lo, hi = brk[live, p], brk[live, p + 1]
        xi = np.clip((a[live] - lo) / (hi - lo), 0.0, 1.0)
        for col in range(rows.shape[1]):
            for i in np.unique(rows[live, col]):
                if i < 0:
                    continue
                sel = np.flatnonzero(rows[live, col] == i)
                factor = (
                    c[live[sel]]
                    * lag[live[sel], col]
                    * stencil._width[int(i), p[sel]]
                    / (hi[sel] - lo[sel])
                )
                index, weight = _row_weights(stencil, int(i), p[sel], xi[sel])
                np.add.at(flat, index, factor[:, None] * weight)
    out = []
    for i in range(len(stencil.R_2D)):
        start, end = int(stencil.off[i]), int(stencil.off[i + 1])
        out.append(flat[start + (end - start) // 2 : end].copy())
    return out


def losvd_read_vjp(
    part: Any, r_2d: ArrayLike, v: ArrayLike, cotangent: ArrayLike
) -> list[Array]:
    """Return the transpose of one component's reading of its table at the
    pairs ``(r_2d[j], v[j])`` (`observables.core._read_losvd`, in internal
    units): a cotangent on the values read, carried to each row's normalized
    values ``row.losvd[0]``.

    The reading is linear in the rows: the stencil's Lagrange weights
    between rows, the panel widths' ratio and each sub-panel's barycentric
    reading through its own nodes, all set by the radii, the speeds and the
    table's breaks. Its transpose scatters the cotangent back with the same
    weights. Rows no pair reads take zeros.
    """
    from ._ellipse_model import ellipse_route
    from ._losvd_stages import ellipse_rules, table_stencil

    model, table = part.model, part.table
    route = ellipse_route(model, ellipse_rules(model.settings), with_df=False)
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    return _stencil_read_vjp(
        table_stencil(route, table),
        route,
        np.asarray(r_2d, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(cotangent, dtype=np.float64),
    )


def aperture_losvd_vjp(
    model: Model,
    table: Any,
    projected: Any,
    aperture: Any,
    settings: Any,
    result: Any,
    cotangent: ArrayLike,
) -> list[Array]:
    """Return the transpose of an aperture's distribution (`aperture_losvd`)
    onto the table's rows: a cotangent on ``result.losvd[0]`` at its nodes,
    carried to each row's normalized values ``row.losvd[0]``.

    The aperture's panels are recomputed as the stage made them, each value
    with the radii it read the table at, their weights and the breaks there
    (`aperture_losvd_panels`, its record), and held equal to the result,
    bit for bit. Each value is linear in the table's rows through those
    reads, so the transpose is that of the reading (`_stencil_read_vjp`)
    with each node's cotangent spread on its own radii by its weights.

    Raises
    ------
    ResultError
        Where the panels recomputed are not the result given, bit for bit.
    """
    from ._aperture_losvd import aperture_losvd_panels
    from ._ellipse_model import ellipse_route
    from ._losvd_stages import ellipse_rules, table_stencil
    from ._projection import surface_density_reader
    from .stages import INNER_DISK_TOLERANCE

    observables = settings.observables
    route = ellipse_route(model, ellipse_rules(settings), with_df=False)
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    stencil = table_stencil(route, table)
    surface = surface_density_reader(
        projected.r_2d,
        projected.surface_density[0],
        float(model.profile.r_table_max),
        model.spatially_finite,
    )
    record: list[tuple[Array, Array, dict[str, Array]]] = []
    v, _, d, *_ = aperture_losvd_panels(
        surface,
        stencil,
        route,
        aperture,
        int(observables["n_aperture_nodes"]),
        int(observables["aperture_nodes_per_decade"]),
        INNER_DISK_TOLERANCE,
        record=record,
    )
    if (
        np.asarray(d).tobytes() != np.asarray(result.losvd[0]).tobytes()
        or np.asarray(v).tobytes() != np.asarray(result.v_nodes).tobytes()
    ):
        raise ResultError(
            "The aperture's distribution recomputed is not the one given: its "
            "rule is not the forward's, so it has no transpose here."
        )
    c = np.asarray(cotangent, dtype=np.float64).ravel()
    radii, speeds, weights = [], [], []
    keys = tuple(record[0][2]) if record else ()
    curves: dict[str, list[Array]] = {k: [] for k in keys}
    for j, (rn, wn, cv) in enumerate(record):
        if c[j] == 0.0:
            continue
        radii.append(np.asarray(rn))
        speeds.append(np.full(len(rn), float(v[j])))
        weights.append(c[j] * np.asarray(wn))
        for k in keys:
            curves[k].append(np.broadcast_to(np.asarray(cv[k]), (len(rn),)))
    if not radii:
        return [np.zeros(row.losvd[0].shape) for row in table.rows]
    return _stencil_read_vjp(
        stencil,
        route,
        np.concatenate(radii),
        np.concatenate(speeds),
        np.concatenate(weights),
        {k: np.concatenate(x) for k, x in curves.items()},
    )
