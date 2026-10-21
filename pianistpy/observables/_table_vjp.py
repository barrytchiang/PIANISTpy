r"""The transpose of the DF lookup and of the observables that read the DF.

`Differentiable.losvd`, `Differentiable.aperture_losvd` and
`Differentiable.pdf_v` use the module `_table_vjp` through the private
module `_differentiable`. Each function of `_table_vjp` carries a cotangent on the
values of the DF :math:`F` at energies back to the numbers of the DF table
and of the DF's forms that the DF lookup reads.

The DF lookup (`df.lookup`) reads :math:`\log_{10}F` at an energy
:math:`\mathcal{K}` in three ranges. Inside the DF table, the DF lookup reads a
cubic Hermite interpolant in :math:`\ln\mathcal{K}` through the nodes of the DF
table. Above the DF table, the DF lookup reads the central form of the DF.
Below the DF table, the DF lookup reads the floor rule of the DF. The value of
the DF is :math:`F = 10^{\log_{10}F}`.

JAX differentiates the DF lookup itself (`evaluate_lookup` with ``jax.numpy``
on traced arrays), so the transpose reads the expressions of the forward
lookup. Three maps lie between the arrays of the DF lookup and the numbers of
the DF table, and `_table_vjp` transposes the three maps on the host:

- the nodes of the DF lookup are the nodes of the DF table's support, preceded
  by the nodes below the DF table that the DF lookup keeps, and padded;
- where the DF table holds no exact slopes, the slopes of the DF lookup are
  the slopes of the not-a-knot cubic spline through the nodes in
  :math:`\ln\mathcal{K}`, a linear map of the node values
  (`spline_slopes_vjp`);
- some numbers of the DF lookup's forms are formed from other numbers of the
  forms and from the value at the top node (`_derived_params`).

The transpose holds every energy fixed: the nodes, the energy :math:`P_0` of
the central form, and the anchor energy.

References: Notes Section T:2.5.4. Notes Section T:7.1.
"""

__all__ = [
    "FORM_LEAVES",
    "DfCotangent",
    "aperture_losvd_vjp",
    "lookup_vjp",
    "losvd_read_vjp",
    "losvd_rows_vjp",
    "velocity_distribution_vjp",
    "spline_slopes_vjp",
]

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from ..df import Model
from ..df import lookup as lk
from ..df.lookup import LookupArrays, LookupKinds, evaluate_lookup, prepare_lookup
from ..exceptions import InputError, ResultError
from ..runtime import get_cores
from ..typing import Array, ArrayLike
from ._kept import WORKER
from ._losvd_table import new_pool
from ._velocity import angle_integral, angle_pieces, graded_rule, velocity_model

# The numeric fields of a component's form that the DF lookup reads, apart from
# the energies. Each field holds one number per component, except the floor's
# coefficients and the regular part, which hold one tuple per component.
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
# The largest relative difference allowed between the derived numbers of the
# forms computed under JAX (`_derived_params`) and the derived numbers that
# prepare_lookup computes on the host. Both computations evaluate the formulas
# of prepare_lookup in float64. The value 1e-13 is above the float64 rounding
# unit, 2.2e-16, and far below the 1e-6 bound that the gradient serves.
_DERIVED_TOLERANCE = 1e-13
# The anisotropy parameters of each family that the DF path takes derivatives
# in. DF_PARAMETERS equals df.gradient.PARAMETERS except for the isotropic
# family. The DF path takes no beta_c of an Osipkov-Merritt (OM) model, and the
# tangent rows of the Jeans path (TANGENT_PARAMETERS) carry the beta_c of an OM
# model. The DF path takes no parameter of an isotropic model, because the DF
# table of a model at beta_c = 0 carries no derivative in beta_c, and the LOSVD
# rows of an isotropic model read G with no L factor.
DF_PARAMETERS = {
    "isotropic": (),
    "constant": ("beta_c",),
    "OM_I": ("r_a",),
    "OM_II": ("r_a",),
    "COM_I": ("beta_c", "r_a"),
}
# The number of energies at which one JAX call reads F. The chunk size bounds
# the memory of the transpose.
_CHUNK = 262_144


@dataclass(frozen=True)
class DfCotangent:
    """A cotangent on the numbers of a component's DF table and forms.

    Parameters
    ----------
    log_df : array
        Cotangent on `DfTable.log_df`, with the shape of `DfTable.log_df`.
        The cotangent is zero outside the component's support.
    log_df_slope : array
        Cotangent on `DfTable.log_df_slope`, with the shape of
        `DfTable.log_df_slope`. The cotangent is zero outside the component's
        support. The cotangent is also zero where the DF table holds no exact
        slopes (``slopes_exact`` False), because the transpose then carries
        the cotangent on the spline's slopes to `log_df`.
    log_df_below : array
        Cotangent on the values at the nodes below the DF table, with the
        shape of `DfTable.log_df_below`.
    log_df_slope_below : array
        Cotangent on the slopes at the nodes below the DF table, with the
        shape of `DfTable.log_df_slope_below`.
    form : dict
        Cotangent on each field of `FORM_LEAVES` of the component. Each value
        is a float, or a tuple of floats for the floor's coefficients and for
        the regular part.
    """

    log_df: Array
    log_df_slope: Array
    log_df_below: Array
    log_df_slope_below: Array
    form: dict[str, Any] = field(default_factory=dict)


def spline_slopes_vjp(x: ArrayLike, cotangent: ArrayLike) -> Array:
    r"""Return the transpose of the map from the values of a not-a-knot cubic
    spline to the spline's slopes at the knots, applied to a cotangent on the
    slopes.

    Parameters
    ----------
    x : array-like, shape (n,)
        Knots of the spline, increasing. The number of knots n is at least 4.
    cotangent : array-like, shape (n,)
        Cotangent :math:`c` on the slopes :math:`s = S y` that
        ``CubicSpline(x, y, bc_type="not-a-knot")`` holds at the knots, with
        :math:`y` the values at the knots and :math:`S` the linear map from
        values to slopes.

    Returns
    -------
    cotangent_y : array, shape (n,)
        Cotangent :math:`S^{\mathsf T} c` on the values :math:`y`.

    Raises
    ------
    ResultError
        If fewer than 4 knots are given.

    Notes
    -----
    SciPy forms the slopes as the solution of :math:`A s = b`. The matrix
    :math:`A` is tridiagonal and depends on the knots alone. The vector
    :math:`b = B d` is linear in the chords
    :math:`d_i = (y_{i+1} - y_i)/\Delta x_i`, with :math:`\Delta x_i` the
    spacing of the knots. The transpose is therefore
    :math:`S^{\mathsf T} c = D^{\mathsf T} B^{\mathsf T} A^{-\mathsf T} c`,
    with :math:`D` the map from values to chords. The function computes the
    transpose with one banded solve with :math:`A^{\mathsf T}`, then the
    transposes of :math:`B` and :math:`D`. The rows of :math:`A` and
    :math:`B` are the rows of SciPy's ``CubicSpline.__init__`` for the
    not-a-knot condition at both ends and n > 3.
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(cotangent, dtype=np.float64)
    n = x.size
    if n < 4:
        raise ResultError(
            "The transpose of the cubic spline needs at least 4 knots; the spline "
            f"given has {n} knots."
        )
    dx = np.diff(x)
    ab = np.zeros((3, n))
    ab[1, 1:-1] = 2.0 * (dx[:-1] + dx[1:])
    ab[0, 2:] = dx[:-1]
    ab[-1, :-2] = dx[1:]
    ab[1, 0] = dx[1]
    ab[0, 1] = x[2] - x[0]
    ab[1, -1] = dx[-2]
    ab[-1, -2] = x[-1] - x[-3]
    # The band of A transposed: the upper diagonal of A^T is the lower
    # diagonal of A, and the lower diagonal of A^T is the upper diagonal of A.
    ab_t = np.zeros((3, n))
    ab_t[1] = ab[1]
    ab_t[0, 1:] = ab[2, :-1]
    ab_t[2, :-1] = ab[0, 1:]
    z = solve_banded((1, 1), ab_t, c, check_finite=False)
    c_chord = np.zeros(n - 1)
    # Interior rows of b: b_i = 3 (dx_i d_{i-1} + dx_{i-1} d_i), with d the
    # chords. The first and the last rows of b follow the not-a-knot condition.
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
    """Return the fields of `FORM_LEAVES` of one component as floats and
    tuples of floats, keyed by field name.

    A field that holds no entry for the component is left out."""
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
    """Return the parameter vector of the DF lookup as a JAX function of the
    value at the top node and of the fields of `FORM_LEAVES`.

    The function evaluates the formulas of `prepare_lookup`. The entries of
    the parameter vector that the function does not set, among them every
    entry that depends on the energies, keep the values that
    `prepare_lookup` computed on the host. Before returning the function,
    `_derived_params` evaluates the function at the host's inputs and compares
    the result with the host's parameter vector.

    Raises ResultError when an entry differs from the host's entry by more
    than `_DERIVED_TOLERANCE` relative to the larger of the two magnitudes."""
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
            "The DF lookup's parameters, formed again for the transpose, differ "
            "from the parameters of prepare_lookup at the indices "
            f"{np.flatnonzero(off).tolist()}."
        )
    return params_of


def _lookup_parts(model: Model) -> tuple[Any, Any, Any, Any]:
    """Return four parts of the DF lookup of a one-component model.

    The four parts are: the prepared lookup arrays; the parameter function
    of `_derived_params`; the lookup's values, slopes and parameters as JAX
    arrays; and the JAX reader ``read(y, m, params, k, k_c)`` of F. The reader
    returns F as the observables read F: zero at and below the energy `k_c`,
    and zero where the lookup gives minus infinity."""
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    params_of = _derived_params(arrays, kinds, model.form, 0)
    leaves, read = _lookup_reader(arrays, kinds)
    return arrays, params_of, leaves, read


def _lookup_reader(arrays: LookupArrays, kinds: LookupKinds) -> tuple[Any, Any]:
    """Return the lookup's values, slopes and parameters as JAX arrays and the
    JAX reader of F, built from the prepared lookup alone.

    The reader is the reader of `_lookup_parts`, with one more argument: the
    shift c, which reads F as :math:`10^{\\log_{10}F - c}`."""
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
    """Return the `DfCotangent` of a cotangent on the lookup's arrays.

    The cotangents `c_y`, `c_m` and `c_p` sit on the lookup's node values,
    node slopes and parameter vector. The function carries the cotangent on
    the parameter vector to the value at the top node and to the fields of
    `FORM_LEAVES`. The function carries the cotangent on the spline's slopes
    to the node values when the DF table holds no exact slopes. The function
    then places the node cotangents on the DF table's support and on the kept
    nodes below the DF table."""
    table, form = model.table, model.form
    n = int(arrays.n)
    c_y = np.array(c_y, dtype=np.float64)[:n]
    c_m = np.array(c_m, dtype=np.float64)[:n]
    # The cotangent on the parameter vector goes to the value at the top node
    # and to the fields of FORM_LEAVES.
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
    # The slopes are the DF table's exact slopes, or else the spline's slopes,
    # which are a linear map of the node values.
    if not table.slopes_exact[0]:
        c_y += spline_slopes_vjp(np.log(np.asarray(arrays.k)[:n]), c_m)
        c_m = np.zeros(n)
    # The lookup's nodes are the kept nodes below the DF table, followed by the
    # nodes of the DF table's support.
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
        Model of one component.
    k : array-like
        Energies at which F is read as the observables read F. The DF lookup
        gives :math:`F = 10^{\log_{10}F}`. F is zero at and below `k_c`, and
        zero where the DF lookup gives minus infinity.
    cotangent : array-like, shape of `k`
        Cotangent on F at the energies `k`.
    k_c : float, optional
        Lower edge of the support: F is zero at and below `k_c`. Default:
        ``0.0``, which means F is zero at and below zero energy.

    Returns
    -------
    cotangent : DfCotangent
        Cotangent on the component's DF table and forms.

    Raises
    ------
    ResultError
        If the derived numbers of the forms computed under JAX differ from
        the derived numbers of `prepare_lookup` (`_derived_params`).
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
    """Return the cotangent on the lookup's values, slopes and parameters
    (`leaves`) of a cotangent on F at energies.

    The function sums the cotangent over chunks of `_CHUNK` energies. The
    reader `read` gives F as :math:`10^{\\log_{10}F - c}`, with the shift c
    (`shift`) inside the exponent. With the shift, a DF table whose F exceeds
    the float64 range still gives a finite F."""
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
    """Return the anisotropy parameters of the explicit part as a tuple.

    Raises InputError for a nonempty `parameters` on a COM Type II model,
    because the DF path takes no derivative of a COM Type II model. Raises
    InputError for a parameter outside the family's entry of
    `DF_PARAMETERS`."""
    names = tuple(parameters)
    family = model.components[0].anisotropy.family
    if names and family == "COM_II":
        raise InputError(
            "The DF path takes no derivative in the anisotropy of a COM Type II "
            "model, because the derivative of the DF table is not computed for a "
            "COM Type II model. The second moments of a COM Type II model take the "
            "derivative in the anisotropy by the Jeans path (the tangents of "
            "dispersion_profile)."
        )
    for name in names:
        if name not in DF_PARAMETERS[family]:
            raise InputError(
                f"The derivatives of a model of the anisotropy family {family} are "
                f"taken in the parameters {list(DF_PARAMETERS[family])}; the "
                f"parameter given is {name!r}."
            )
    return names


def velocity_distribution_vjp(
    result: Any, cotangent: ArrayLike, parameters: Any = ()
) -> tuple[DfCotangent, dict[str, float]]:
    r"""Return the transpose of a velocity distribution: a cotangent on the
    distribution's values, carried to the DF table and forms and to the
    anisotropy parameters.

    Parameters
    ----------
    result : VelocityDistribution
        Velocity distribution from `velocity_distribution`, at the radius
        ``result.r`` and the velocities ``result.v_nodes``. The record of
        `result` holds the angle rule (the numbers of angle nodes and angle
        panels) that computed the distribution.
    cotangent : array-like, shape of ``result.pdf``
        Cotangent on the velocity distribution :math:`p(v\,|\,r)` at the velocities
        ``result.v_nodes``.
    parameters : sequence of {"beta_c", "r_a"}, optional
        Anisotropy parameters of the explicit part. The explicit part is the
        derivative at a fixed DF table and at fixed velocities. Default: ``()``,
        which means no explicit part.

    Returns
    -------
    table : DfCotangent
        Cotangent on the DF table and forms, in the form that `lookup_vjp`
        returns.
    explicit : dict of str to float
        Explicit part of the cotangent on each parameter in `parameters`.

    Raises
    ------
    ResultError
        If the distribution recomputed with the angle rule of the record
        differs from ``result.pdf`` in any bit.
    ResultError
        If the JAX copy of the angle integral differs from ``result.pdf`` by
        more than :math:`10^{-9}` of the largest absolute value of
        ``result.pdf``.
    ResultError
        If the derived numbers of the forms computed under JAX differ from
        the derived numbers of `prepare_lookup` (`_derived_params`).
    InputError
        If `parameters` lists a parameter that the DF path does not take
        (`_df_parameters`).

    Notes
    -----
    The function first recomputes the distribution on the host with
    `angle_integral` and the angle rule of the record. The recomputed
    distribution must equal ``result.pdf`` bit for bit, so the angle rule
    below is the angle rule of the forward computation.

    JAX then differentiates a copy of the angle integral. The JAX copy uses
    the pieces in the angle variable t of `angle_pieces`, with each edge of a
    piece set by the condition that sets the edge on the host. The JAX copy
    maps the graded rule onto each piece, as the host does. On the last piece
    of a velocity with :math:`b > 0`, the JAX copy uses the substitution
    :math:`1 - t = w^{1/(1-b)}`, as the host does. The JAX copy reads F by
    the DF lookup with the lookup's arrays as JAX inputs.

    The edges of the pieces move with :math:`\xi = r^2/R_{\rm a}^2`, and the
    substitution moves with :math:`b = \beta_{\rm c}`. The explicit part
    therefore holds the motion of the edges. Where F is not zero at the edge
    of the support (the floor rule ``"zero"``), the motion of the edges gives
    the boundary term, evaluated on the nodes of the angle rule.
    """
    from .stages import _velocity_point

    model = result.model
    names = _df_parameters(model, parameters)
    sm = velocity_model(model)
    r = float(result.r)
    psi, xi, scale = _velocity_point(model, sm, r)
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
            "The velocity distribution recomputed by the rule in the distribution's "
            "record differs from the velocity distribution given, so the velocity "
            "distribution given was not made by the forward computation's rule and "
            "has no transpose."
        )
    # The pieces of every velocity, with each edge coded by the condition that
    # sets the edge: 0 and 1 for the fixed ends, 2 for a root of
    # c0 + c2 t^2 = k_c, and 3 for a join where c0 + c2 t^2 = k_w.
    codes = {"0": 0, "1": 1, "root": 2, "join": 3}
    velocity_of, lo_by, hi_by, substituted = [], [], [], []
    for j, vj in enumerate(v):
        _, _, edges, by = angle_pieces(sm, psi, xi, float(vj))
        last = len(edges) - 2
        for i in range(len(edges) - 1):
            if not edges[i + 1] > edges[i]:
                continue
            velocity_of.append(j)
            lo_by.append(codes[by[i]])
            hi_by.append(codes[by[i + 1]])
            substituted.append(sm.b > 0.0 and i == last and edges[i + 1] == 1.0)
    unit_t, unit_w = (jnp.asarray(a) for a in graded_rule(0.0, 1.0, n_panels, n_nodes))
    velocity_of_a = jnp.asarray(velocity_of, dtype=jnp.int32)
    lo_by_a, hi_by_a = jnp.asarray(lo_by), jnp.asarray(hi_by)
    substituted_a = jnp.asarray(substituted)
    arrays, params_of, leaves, read = _lookup_parts(model)
    rho = 4.0 * np.pi * r ** (-2.0 * sm.b) / scale
    v_j = jnp.asarray(v)
    sign, k_c, k_w = sm.sign, sm.k_c, sm.k_w

    def edge(by: Any, c0: Any, c2: Any) -> Any:
        # The square root is taken only for an edge set by a root or a join
        # whose argument lies inside (0, 1), as on the host. Every other
        # entry takes the square root of 0.25, so the branch that `where`
        # discards carries no infinite slope.
        safe = jnp.where(c2 != 0.0, c2, 1.0)
        arg = jnp.where(by == 2, (k_c - c0) / safe, (k_w - c0) / safe)
        moving = (by >= 2) & (arg > 0.0) & (arg < 1.0)
        root = jnp.sqrt(jnp.where(moving, arg, 0.25))
        return jnp.where(by == 0, 0.0, jnp.where(by == 1, 1.0, root))

    def pdf(y: Any, m: Any, p: Any, b: Any, x: Any) -> Any:
        vv = v_j[velocity_of_a]
        c0 = psi - 0.5 * vv * vv * (1.0 + sign * x)
        c2 = 0.5 * sign * x * vv * vv
        a = edge(lo_by_a, c0, c2)
        e = edge(hi_by_a, c0, c2)
        # A piece without substitution takes the graded rule mapped onto the
        # piece's interval [a, e].
        t = a[:, None] + (e - a)[:, None] * unit_t[None, :]
        w = (e - a)[:, None] * unit_w[None, :]
        f_plain = read(y, m, p, c0[:, None] + c2[:, None] * t * t, k_c)
        # A node that the map rounds onto t = 1 gets a zero factor. The nodes
        # of the forward computation lie inside t < 1. Dropping a node at
        # t = 1 keeps both (1 - t^2)^(-b) and the slope of (1 - t^2)^(-b)
        # finite.
        one = 1.0 - t * t
        inner = one > 0.0
        factor = jnp.where(inner, jnp.where(inner, one, 1.0) ** (-b), 0.0)
        plain = jnp.sum(w * factor * f_plain, axis=1)
        # The substituted last piece: 1 - t = w^p with p = 1/(1 - b), for w on
        # [0, (1 - a)^(1 - b)].
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
        integral = jax.ops.segment_sum(total, velocity_of_a, num_segments=v.size)
        return 4.0 * jnp.pi * r ** (-2.0 * b) / rho * v_j ** (2.0 - 2.0 * b) * integral

    b0, x0 = jnp.asarray(sm.b), jnp.asarray(xi)
    value, back = jax.vjp(pdf, *leaves, b0, x0)
    scale_value = float(np.max(np.abs(result.pdf)))
    if not np.allclose(
        np.asarray(value), result.pdf, rtol=0.0, atol=1e-9 * scale_value
    ):
        raise ResultError(
            "The twin of the angle integral does not reproduce the velocity "
            "distribution to 1e-9 of the velocity distribution's largest value."
        )
    c_y, c_m, c_p, c_b, c_x = back(jnp.asarray(cotangent, dtype=jnp.float64))
    explicit: dict[str, float] = {}
    for name in names:
        if name == "beta_c":
            explicit[name] = float(c_b)
        else:
            explicit[name] = float(c_x) * (-2.0 * xi / sm.r_a)
    return _table_cotangent(model, arrays, params_of, c_y, c_m, c_p), explicit


# The largest difference allowed between G computed by the JAX copy
# (`_g_twin`) and G computed by the route, relative to the route's G and to
# the largest absolute value of the route's G. The value 1e-12 is above the
# float64 rounding unit, 2.2e-16, and far below the 1e-6 bound that the
# gradient serves.
_G_TOLERANCE = 1e-12


def _cumulative_order(arrays: LookupArrays, kinds: LookupKinds, q: float) -> int:
    """Return the Gauss-Legendre order at which `cumulative_df` converges at
    the energy q, by rerunning the doubling of `cumulative_df` on the host.

    The order is 0 at or below the first energy of the lookup, where
    `cumulative_df` returns the closed form of the floor rule. The function
    raises the error of `unconverged_cumulative` when the doubling reaches
    `CUMULATIVE_LAST_ORDER` without converging, as `cumulative_df` does."""
    from ._ellipse_model import (
        CUMULATIVE_FIRST_ORDER,
        CUMULATIVE_LAST_ORDER,
        CUMULATIVE_TOLERANCE,
        _cumulative_at_order,
        unconverged_cumulative,
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
        last, value = value, new
    raise unconverged_cumulative(q, value, last)


def _below_mass(params: Any, kinds: LookupKinds, k: float) -> Any:
    """Return `below_floor_mass` at the energy k as a JAX expression in the
    lookup's parameter vector.

    Raises ResultError for the floor rule "tail", because `_table_vjp` has no
    JAX version of the closed form of the floor rule "tail"."""
    rule = kinds.below_floor
    if rule == "taylor":
        return params[lk.FLOOR_0] * k * k / 2.0 + params[lk.FLOOR_1] * k**3 / 6.0
    if rule == "edge_law":
        p, c_0, c_1 = params[lk.FLOOR_0], params[lk.FLOOR_1], params[lk.FLOOR_2]
        return c_0 * k ** (p + 1.0) / (p + 1.0) + c_1 * k ** (p + 2.0) / (p + 2.0)
    if rule == "zero":
        return 0.0
    raise ResultError(
        "The transpose of G below a DF table with the floor rule 'tail' is not "
        "implemented, because the closed form of the floor rule 'tail' "
        "(tail_mass_below) has no JAX version."
    )


def _cumulative_twin(
    arrays: LookupArrays, kinds: LookupKinds, a: LookupArrays, q: float, order: int
) -> Any:
    """Return `cumulative_df` at the energy q as a JAX expression, at the
    Gauss-Legendre order `order` where the host's doubling converged.

    The JAX expression uses the panels and nodes of `cumulative_df` and reads
    F by the lookup `a`, whose arrays are JAX inputs."""
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
    """Return `EllipseDF._cum` as a JAX expression: the cumulative integral of
    the samples g at spacing h, starting from `base`."""
    c = jnp.zeros(g.shape[0])
    c = c.at[0].set(base)
    c = c.at[1:].set(base + jnp.cumsum(0.5 * h * (g[1:] + g[:-1])))
    c = c.at[2::2].set(
        base + jnp.cumsum(h / 3.0 * (g[0:-2:2] + 4.0 * g[1:-1:2] + g[2::2]))
    )
    return c.at[1:-1:2].set(
        0.5 * (c[0:-2:2] + c[2::2]) + (h / 4.0) * (g[0:-2:2] - g[2::2])
    )


def _g_twin(
    model: Model,
    edf: Any,
    shift: float,
    lookup: tuple[LookupArrays, LookupKinds] | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    r"""Return `EllipseDF.G` as a JAX function of the lookup's arrays
    ``(y, m, params)`` and of energies.

    The JAX copy builds the three branches of G on the grids of the route.
    The base of each branch is the lookup's cumulative integral of F at the
    Gauss-Legendre order where the host's doubling converged. The JAX copy
    reads G between grid nodes as the route reads G. F is the route's F,
    :math:`10^{\log_{10}F - c}`, with the shift c (`shift`). The argument
    `lookup` is the model's prepared lookup when the caller holds the
    prepared lookup, and None otherwise.

    The return value is a pair of functions ``(tables, read)``, and G at the
    energies q is ``read(tables(y, m, params), q, held)``. The function
    ``tables`` builds the branches from the lookup's arrays alone. The
    function ``read`` reads the branches at the energies. Branches built once
    serve every reading at one set of lookup arrays."""
    arrays, kinds = (
        prepare_lookup(model.table, model.form, 0) if lookup is None else lookup
    )
    q_low = 1e-40 * edf.Q0
    orders = (
        _cumulative_order(arrays, kinds, edf.Q0),
        _cumulative_order(arrays, kinds, q_low),
    )
    fixed = (jnp.asarray(arrays.k), jnp.asarray(arrays.width), jnp.asarray(arrays.n))
    lq, lb = edf._lq, edf._lb
    yu = edf._yu[::-1]  # The upper branch's order, from Q_mid toward Q_end.
    scale = edf.scale

    def tables(y: Any, m: Any, params: Any) -> tuple[Any, Any, Any, Any]:
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
        return lgl, lgu, lgb, sb

    def read(branches: Any, q: Any, held: Any = None) -> Any:
        lgl, lgu, lgb, sb = branches
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

    return tables, read


def _add(a: DfCotangent, b: DfCotangent) -> DfCotangent:
    """Return the sum of two cotangents on one DF table and forms."""
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
    """Rerun the rule of one row (`EllipseRoute.row`) and record the row's
    trace of levels.

    Returns the row, the distinct absolute velocities, the map from the row's
    nodes to the distinct velocities, and the trace with one entry per level."""
    trace: list[tuple[int, int, Array]] = []
    d = route.row(r2d, v_nodes, rule, w=v_weights, trace=trace)
    va, inv = np.unique(
        np.abs(np.asarray(v_nodes, dtype=np.float64)), return_inverse=True
    )
    return d, va, inv, trace


def _row_reads(
    route: Any, r2d: float, v_nodes: Array, v_weights: Array, rule: Any
) -> tuple[Array, Array, Array, list[tuple[Any, ...]]]:
    """Run the rule of one row (`EllipseRoute.row`) once and collect the
    row's readings of F and G.

    For each distinct velocity, the collected readings are the readings of the level
    that gives the velocity's value, in the format that `_row_by_levels`
    collects. Returns the row, the distinct absolute velocities, the map from the
    row's nodes to the distinct velocities, and the collected readings."""
    collect: list[tuple[Any, ...]] = []
    d = route.row(r2d, v_nodes, rule, w=v_weights, collect=collect)
    va, inv = np.unique(
        np.abs(np.asarray(v_nodes, dtype=np.float64)), return_inverse=True
    )
    return d, va, inv, collect


def _row_support(
    route: Any, r2d: float, va: Array
) -> tuple[Array, Array, Array, Array]:
    """Return the supports in t of the velocities `va`, split as the row's rule
    splits the supports (`EllipseRoute.l0_split`)."""
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
    """Return the row values D at the distinct velocities `va`, each velocity
    computed again at the last level of `trace` that computed the velocity.

    Each velocity uses the nodes of the rule at the velocity's level. The supports
    in t are `support` when given (from `_row_support`, with the t-nodes held
    fixed), and the route's supports otherwise. With `collect`, the readings
    of F and G are appended to `collect`."""
    level = np.full(len(va), -1)
    for i, (_, _, mask) in enumerate(trace):
        level[mask] = i
    jv, tl, th, cut = _row_support(route, r2d, va) if support is None else support
    out = np.zeros(len(va))
    for i in np.unique(level):
        velocities = level == i
        sel = velocities[jv]
        if not sel.any():
            continue
        n_p, n_lay, _ = trace[i]
        d = route._row_at(
            r2d, va, jv[sel], tl[sel], th[sel], rule, n_p, n_lay, cut[sel], collect
        )
        out[velocities] = d[velocities]
    return out


def _row_transpose(
    route: Any,
    i: int,
    r2d: float,
    v_nodes: Array,
    v_weights: Array,
    losvd: Array,
    c: Array,
    check: bool = False,
) -> tuple[list[tuple[Array, Array]], list[tuple[Array, Array]]]:
    """Return the transpose of row `i`: the energies of every reading of F
    and of G, each with the cotangent that the reading takes.

    Without `check`, the function runs the row's rule once and collects the
    readings (`_row_reads`). With `check`, the function reruns the row and
    requires the normalized row to equal the stored row `losvd` bit for bit.
    The function then collects the readings level by level
    (`_row_by_levels`) and requires the row built level by level to equal the
    rerun row bit for bit.

    Raises ResultError when the recomputed panels differ from the stored
    panels, and, with `check`, when either bit-for-bit comparison fails."""
    rule, _ = route.rule(r2d)
    vp, wp = route.panels(r2d)[:2]
    if vp.tobytes() != v_nodes.tobytes() or wp.tobytes() != v_weights.tobytes():
        raise ResultError(
            f"At the radius index {i}, the velocity panels recomputed differ from the "
            "stored velocity panels."
        )
    if not check:
        d, va, inv, collect = _row_reads(route, r2d, vp, wp, rule)
        norm = 2.0 * float(np.dot(wp, d))
    else:
        d, va, inv, trace = _row_levels(route, r2d, vp, wp, rule)
        norm = 2.0 * float(np.dot(wp, d))
        if (d / norm).tobytes() != losvd.tobytes():
            raise ResultError(
                "The line-of-sight velocity distribution at the radius index "
                f"{i}, recomputed by the distribution's own rule, differs from the "
                "stored distribution, so the stored distribution was not made by "
                "the forward computation's rule and has no transpose."
            )
        collect = []
        by_velocity = _row_by_levels(route, r2d, va, rule, trace, collect)
        if by_velocity[inv].tobytes() != d.tobytes():
            raise ResultError(
                f"At the radius index {i}, the line-of-sight velocity distribution "
                "recomputed level by level differs from the distribution recomputed "
                "by the distribution's own rule."
            )
    c_d = (c - 2.0 * wp * float(c @ (d / norm))) / norm
    c_velocity = np.bincount(inv, weights=c_d, minlength=len(va))
    f_reads: list[tuple[Array, Array]] = []
    g_reads: list[tuple[Array, Array]] = []
    for kind, velocity, k, w in collect:
        if kind == "P":
            continue
        into = f_reads if kind == "F" else g_reads
        into.append(
            (np.asarray(k).ravel(), (c_velocity[velocity][:, None] * w).ravel())
        )
    return f_reads, g_reads


def _row_cotangent(
    route: Any, lookup: tuple[LookupArrays, LookupKinds], shift: float, *args: Any
) -> tuple[list[Array], list[tuple[Array, Array]]]:
    """Return the transpose of one row (`_row_transpose`) with the readings
    of F already carried onto the lookup's arrays.

    The function drops the readings of F with zero cotangent and reads F with
    the route's shift inside the exponent. The result for F is a list of
    arrays the size of the lookup's arrays, never the list of readings. The
    readings of G are returned unchanged."""
    f_reads, g_reads = _row_transpose(route, *args)
    leaves, read = _lookup_reader(*lookup)
    k = np.concatenate([k for k, _ in f_reads]) if f_reads else np.zeros(0)
    c = np.concatenate([w for _, w in f_reads]) if f_reads else np.zeros(0)
    keep = c != 0.0
    return _read_sums(leaves, read, k[keep], c[keep], 0.0, shift), g_reads


def _row_cotangent_worker(
    args: tuple[Any, ...],
) -> tuple[list[Array], list[tuple[Array, Array]]]:
    """Run `_row_cotangent` in a pool process, on the process's route, after
    setting the route's kept ranges to the table's kept ranges."""
    route = WORKER["route"]
    route.vrad_kept, route.vk_kept = args[7]
    return _row_cotangent(route, args[8], args[9], *args[:7])


class RowsBasis:
    """The parts of a model that the transpose of the LOSVD rows
    (`losvd_rows_vjp`) and the explicit part of the rows
    (`losvd_rows_explicit`) read, built once per model.

    A `RowsBasis` holds the model's route with the DF (`ellipse_route`, with
    the DF's G on three grids). The prepared lookup (`lookup`) and the
    lookup's JAX parts (`parts`, from `_lookup_parts`) are built at the first
    read.

    Each use takes a new route (`route`): a new `EllipseRoute` on the pieces
    that `ellipse_route` built, with the table's kept ranges and with empty
    caches, as `ellipse_route` returns a route. The uses only read the
    pieces. The DF's memo of the map toward K = 0 (`EllipseDF.zero_map`)
    holds, at each key, the value that the map's expression gives at the
    key, so sharing the memo does not change a result.
    """

    def __init__(self, model: Model, settings: Any) -> None:
        from ._ellipse_model import ellipse_route
        from ._losvd_stages import ellipse_rules

        self._model = model
        self._made = ellipse_route(model, ellipse_rules(settings))

    @cached_property
    def lookup(self) -> tuple[LookupArrays, LookupKinds]:
        """The prepared lookup of the model's DF (`prepare_lookup`)."""
        return prepare_lookup(self._model.table, self._model.form, 0)

    @cached_property
    def parts(self) -> tuple[Any, Any, Any, Any]:
        """The four parts of the model's DF lookup (`_lookup_parts`)."""
        return _lookup_parts(self._model)

    def route(self, table: Any) -> Any:
        """Return a new route on the stored pieces, with the kept ranges of
        `table`."""
        made = self._made
        route = type(made)(made.psi, made.R_t, made.model, made.df, made.rules)
        route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
        return route


def losvd_rows_vjp(
    model: Model,
    table: Any,
    settings: Any,
    cotangent: Any,
    basis: RowsBasis | None = None,
) -> DfCotangent:
    r"""Return the transpose of the rows of a LOSVD table: cotangents on the
    rows, carried to the DF table and forms.

    Parameters
    ----------
    model : Model
        Model of one component that owns `table`.
    table : LosvdTable
        LOSVD table of `model` (`losvd_table`). The transpose reruns the rule
        of each row.
    settings : Settings
        Settings with which `table` was built.
    cotangent : sequence of array or None
        One entry per row: a cotangent on the row's normalized values
        ``row.losvd[0]`` at the row's nodes, or None for no cotangent.
    basis : RowsBasis, optional
        Route and lookup of `model`, when the caller holds a `RowsBasis`.
        Default: ``None``, which means the function builds a new `RowsBasis`.

    Returns
    -------
    table : DfCotangent
        Cotangent on the DF table and forms of `model`.

    Raises
    ------
    ResultError
        If the route reads the DF without a lookup.
    ResultError
        If the panels of a row, recomputed, differ from the stored panels.
    ResultError
        With ``settings.check_rows``, if a row recomputed by the row's rule
        differs from the stored row in any bit, or if the row computed again
        level by level differs from the recomputed row in any bit.
    ResultError
        If the JAX copy of G (`_g_twin`) differs from the route's G by more
        than `_G_TOLERANCE`.
    ResultError
        If the derived numbers of the forms computed under JAX differ from
        the derived numbers of `prepare_lookup` (`_derived_params`).

    Notes
    -----
    The function recomputes each row with a nonzero cotangent once by the
    row's rule (`EllipseRoute.row`). For each distinct velocity, the function
    collects the energies and weights of every reading of F
    (`EllipseRoute.plane`) at the level that gives the velocity's value. The
    transpose of the row's normalization is
    :math:`c_D = (c - 2w\,(c\cdot\hat D))/N`, with :math:`c` the cotangent on
    the normalized row :math:`\hat D = D/N`, :math:`w` the weights of the
    row's nodes and :math:`N = 2\,w \cdot D` the normalization. The cotangent
    on F at the collected energies goes through the transpose of the lookup
    (as in `lookup_vjp`) inside the row's computation. Each row therefore
    returns arrays the size of the lookup's arrays, never the list of
    readings, and the function sums the arrays of the rows.

    The route reads F as :math:`10^{\log_{10}F - c_{\rm s}}`, with the shift
    :math:`c_{\rm s}` held fixed. The shift cancels in a normalized row. The
    transpose also holds fixed: the radii of the rows, the kept ranges, the
    layouts, the panels and layers of every level, the numbers of angle
    nodes, and the map toward K = 0 with the exponent of the edge law as the
    route computed the map.

    With ``settings.check_rows`` (`Settings`), each recomputed row must equal
    the stored row bit for bit. Each distinct velocity is then computed again at
    the level that gives the velocity's value. The row built from the velocities
    computed again must equal the recomputed row bit for bit. The readings
    are then the readings of the levels computed again (`_row_by_levels`).

    The function transposes the rows in a pool of at most `get_cores`
    processes, or in the calling process when only one process is used. Each
    process runs the function that one process would run (`_row_cotangent`),
    and the function sums the arrays of the rows in row order. The result is
    therefore identical bit for bit for any number of processes.
    """
    from ._ellipse_model import DfReading

    if basis is None:
        basis = RowsBasis(model, settings)
    route = basis.route(table)
    reading = route._df().F
    if not isinstance(reading, DfReading):
        raise ResultError(
            "The line-of-sight integrator's DF is not a DfReading lookup, so the "
            "derivative of the line-of-sight velocity distributions cannot be "
            "computed."
        )
    shift = reading.shift
    arrays, params_of, leaves, _ = basis.parts
    lookup = basis.lookup
    args = [
        (
            i,
            float(row.r_2d),
            row.v_nodes,
            row.v_weights,
            np.asarray(row.losvd[0]),
            np.asarray(cotangent[i], dtype=np.float64).ravel(),
            settings.check_rows,
            (route.vrad_kept, route.vk_kept),
            lookup,
            shift,
        )
        for i, row in enumerate(table.rows)
        # A row whose cotangent is zero adds nothing: every reading of the row
        # carries a zero weight, and `_row_cotangent` drops zero weights.
        if cotangent[i] is not None and np.any(np.asarray(cotangent[i]))
    ]
    n_workers = min(get_cores(), len(args))
    if n_workers > 1:
        # The pool runs `_row_cotangent` on each row, and `pool.map` returns
        # the results in row order.
        with new_pool(n_workers, route) as pool:
            parts = pool.map(_row_cotangent_worker, args, chunksize=1)
    else:
        parts = [_row_cotangent(route, a[8], a[9], *a[:7]) for a in args]
    # The cotangents of the rows on the lookup's arrays, summed in row order.
    sums = [np.zeros(np.shape(a)) for a in leaves]
    for row_sums, _ in parts:
        for i, part in enumerate(row_sums):
            sums[i] += part
    g_energies = [k for _, g in parts for k, _ in g]
    g_weights = [w for _, g in parts for _, w in g]
    out = _table_cotangent(model, arrays, params_of, *sums)
    if g_energies:
        # The rows read G. The transpose of G runs through the JAX copy of G,
        # after a check that the JAX copy reproduces the route's G.
        edf = route._df()
        q = np.concatenate(g_energies)
        c_g = np.concatenate(g_weights)
        g_tables, g_read = _g_twin(model, edf, shift, basis.lookup)
        value, back = jax.vjp(lambda y, m, p: g_read(g_tables(y, m, p), q), *leaves)
        host = edf.G(q)
        scale_g = float(np.max(np.abs(host))) if host.size else 0.0
        if not np.allclose(
            np.asarray(value), host, rtol=_G_TOLERANCE, atol=_G_TOLERANCE * scale_g
        ):
            raise ResultError(
                "The twin of G does not reproduce the DF's G to "
                f"{_G_TOLERANCE:.0e}, so the transpose of the twin is not the "
                "transpose of the forward computation."
            )
        c_y, c_m, c_p = back(jnp.asarray(c_g))
        out = _add(out, _table_cotangent(model, arrays, params_of, c_y, c_m, c_p))
    return out


def _row_weights(stencil: Any, i: int, p: Array, xi: Array) -> tuple[Array, Array]:
    """Return the reading weights of `PanelTable._row_at` for row `i`.

    For each position, the function returns the indices into the table's
    flat values and one weight per index. The value at a position is the sum
    of the weights times the flat values at the indices. A position in a
    panel that row `i` does not have gets zero weights."""
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
    """Return the transpose of `PanelTable.at_pairs` onto the normalized
    values of each row on ``V >= 0``.

    The argument `curves` holds the breaks at the radii when the forward
    reading received the breaks, and is None otherwise."""
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
    reader: tuple[Any, Any], r_2d: ArrayLike, v: ArrayLike, cotangent: ArrayLike
) -> list[Array]:
    """Return the transpose of one component's reading of the LOSVD table at
    the pairs ``(r_2d[j], v[j])``: a cotangent on the values read, carried
    to the normalized values ``row.losvd[0]`` of each row.

    The forward reading is `observables.core._read_losvd`, in internal
    units. The forward reading is linear in the rows. The weights of the
    forward reading are the stencil's Lagrange weights between rows, the
    ratio of panel widths, and the barycentric weights of each sub-panel
    through the sub-panel's nodes. The radii, the velocities and the table's
    breaks set every weight. The transpose scatters the cotangent back onto
    the rows with the weights of the forward reading. A row that no pair
    reads gets zeros. The argument `reader` is the pair (route, stencil) of
    the forward reading, kept on the table (`observables.core._reader_of`),
    and the caller passes the pair in.
    """
    route, stencil = reader
    return _stencil_read_vjp(
        stencil,
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
    record: list[tuple[Array, Array, dict[str, Array]]] | None = None,
) -> list[Array]:
    """Return the transpose of an aperture's line-of-sight velocity
    distribution (`aperture_losvd`) onto the rows of the LOSVD table: a
    cotangent on ``result.losvd[0]`` at the result's velocity nodes, carried
    to the normalized values ``row.losvd[0]`` of each row.

    The stage `aperture_losvd_panels` reads each value of the result from
    the table at a set of projected radii, with one weight per radius and
    with the breaks at the radii, and the stage's record holds the radii,
    the weights and the breaks of each value. When `record` is given, the
    record comes from the stage run that made `result`, so the record matches
    `result` by construction. When `record` is None, the function recomputes
    the panels as the stage computed the panels, on the extrema that the
    table's apertures kept, and requires the recomputed distribution to equal
    `result` bit for bit. Each value of the result is linear in the table's
    rows through the recorded readings. The transpose is therefore the
    transpose of the table reading (`_stencil_read_vjp`), with the cotangent
    of each velocity node spread over the node's radii by the node's
    weights.

    Raises
    ------
    ResultError
        If the recomputed distribution or the recomputed velocity nodes
        differ from `result` in any bit.
    ResultError
        If the given `record` does not hold exactly one entry per velocity
        node of `result`.
    """
    from ._aperture_losvd import aperture_losvd_panels
    from ._ellipse_model import ellipse_route
    from ._losvd_stages import ellipse_rules, table_stencil
    from ._projection import surface_density_reader
    from .stages import INNER_DISK_TOLERANCE

    observables = settings.observables
    rules = ellipse_rules(settings)
    route = ellipse_route(model, rules, with_df=False)
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    # The extrema that earlier apertures on the table computed, kept on the
    # table.
    route._fext = table.extrema.setdefault((model.identity, rules), {})
    stencil = table_stencil(route, table)
    if record is not None:
        v = np.asarray(result.v_nodes)
        if len(record) != len(v):
            raise ResultError(
                "The aperture record given does not hold one entry per velocity "
                "node of the result, so the record is not the result's record, and "
                "the result has no transpose with the record given."
            )
    else:
        surface = surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        )
        record = []
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
                "The aperture distribution recomputed differs from the aperture "
                "distribution given, so the distribution given was not made by the "
                "forward computation's rule and has no transpose."
            )
    c = np.asarray(cotangent, dtype=np.float64).ravel()
    radii, velocities, weights = [], [], []
    keys = tuple(record[0][2]) if record else ()
    curves: dict[str, list[Array]] = {k: [] for k in keys}
    for j, (rn, wn, cv) in enumerate(record):
        if c[j] == 0.0:
            continue
        radii.append(np.asarray(rn))
        velocities.append(np.full(len(rn), float(v[j])))
        weights.append(c[j] * np.asarray(wn))
        for k in keys:
            curves[k].append(np.broadcast_to(np.asarray(cv[k]), (len(rn),)))
    if not radii:
        return [np.zeros(row.losvd[0].shape) for row in table.rows]
    return _stencil_read_vjp(
        stencil,
        route,
        np.concatenate(radii),
        np.concatenate(velocities),
        np.concatenate(weights),
        {k: np.concatenate(x) for k, x in curves.items()},
    )
