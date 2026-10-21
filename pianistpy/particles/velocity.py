r"""The velocity table of a one-component model: the inverse cumulative distribution
of the scaled velocity on rows of constant potential.

The particle sampler (`Sampler`) calls `build_velocity_table` to build the table
that turns a uniform random number into a velocity at a given radius. On the row
of relative potential Psi, the probability density of the scaled velocity
x = v / (2 Psi)^{1/2} is

    ``p(x | Psi) proportional to x^(2 - 2 beta_c) F(Psi (1 - x^2))``,
    ``0 <= x <= x_e``,

where :math:`\beta_{\rm c}` is the component's constant anisotropy, F(E) is the
distribution function at relative energy E, and x_e is the edge of the velocity
support:
x_e = 1 where F continues down to E = 0, and x_e = (1 - K_0 / Psi)^{1/2} where
F vanishes below its lowest tabulated energy K_0. The density p(x | Psi)
depends on the radius only through Psi, so one row of the table serves every
radius at the row's potential (Notes Section T:5.2.1).

The rows are uniform in lambda_p = ln(Psi / (p_0 - Psi)), where p_0 is the
central value of the relative potential, and one extra row lies beyond each end
of the profile's range of Psi. On each row the table holds
s = ln(x / (x_e - x)) at nodes uniform in lambda_u = ln(u / (1 - u)), where u is
the cumulative probability of x, together with the slope ds / dlambda_u along
the row and the slope ds / dlambda_p across rows at fixed u. The row count
starts at sampling["n_rows"] (512) and is refined from n rows to 2 n - 3 rows,
every existing row kept, until the table's own error across rows is below
sampling["row_error_goal"] (:math:`10^{-6}` under the default settings,
:math:`10^{-4}` under the quick settings) or the next refinement would pass
sampling["n_rows_max"] (4075 rows).

Each row is built from the probability masses of the density on panels of the
scaled velocity: Gauss-Legendre panels in ln x near x = 0 and in ln(x_e - x) near
the edge, with closed forms below the smallest refined velocity and above the
closest refined distance from the edge. The cumulative masses at the panel
edges are the row's construction nodes. The inverse of the construction nodes'
cubic Hermite interpolant gives a first s at each table node, and a safeguarded
Newton iteration on the exact cumulative mass then moves each table node until
the node's residual in lambda_u is below NODE_TOLERANCE (:math:`10^{-13}`).
"""

__all__ = ["VELOCITY_TABLE_KEYS", "build_velocity_table"]

import functools
import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..df.asymptotics import (
    _scaled_upper_gamma_small_order,
    scaled_upper_gamma,
    tail_mass_below,
)
from ..df.lookup import (
    FLOOR_0,
    FLOOR_1,
    K_0,
    LookupArrays,
    LookupKinds,
    _below,
    below_floor_mass,
    evaluate_lookup,
    prepare_lookup,
)
from ..exceptions import AccuracyError, ResultError, UnphysicalModelError
from ..numerics import hermite, hermite_inverse
from ..typing import Array
from ..validation import TIER_BOUNDS, Check, ValidationReport

# The Newton polish of a row stops moving a table node when one of three
# conditions holds: the node's absolute residual in lambda_u (the node's target
# lambda_u minus the lambda_u of the exact cumulative mass at the node's s) is
# below NODE_TOLERANCE; the node's Newton step in s is at most 4 times the
# spacing of doubles at max(|s|, 1); or two successive polish passes both leave
# the node's absolute residual at or above the smallest absolute residual the
# node has reached. A residual of 1e-13 in lambda_u is at most 2.5e-14 in the
# cumulative probability u, because du / dlambda_u = u (1 - u) <= 1/4, which is
# more than seven orders of magnitude below the 1e-6 bound of the table's error
# across rows.
NODE_TOLERANCE = 1e-13
# Largest number of polish passes over one row on the host, or over one chunk
# of rows in the row kernels. A table node still moving after
# POLISH_PASSES_MAX passes keeps the s of the node's smallest absolute residual,
# and the build reports the largest such residual as the check
# "polish_residual".
POLISH_PASSES_MAX = 60
# Largest number of iterations, and the stopping tolerance, of the Newton
# iteration that places a table node beyond a row's last construction node on a
# DF tail with a logarithmic factor. The iteration stops when the difference
# between the logarithm of the tail's mass and the logarithm of the target mass,
# or the iteration's step relative to max(1, y), is below
# TAIL_NEWTON_TOLERANCE. A difference of 1e-12 in the logarithm of the tail's
# mass is at most 1e-12 in the cumulative probability, six orders of magnitude
# below the 1e-6 bound of the table's error across rows. A node whose iteration
# reaches TAIL_NEWTON_ITERATIONS keeps the last iterate. The iteration's
# variable is y = ln(E_1 / E), where E is the energy at the node and E_1 is the
# energy at the closest refined distance from the support's edge.
TAIL_NEWTON_ITERATIONS = 80
TAIL_NEWTON_TOLERANCE = 1e-12
# The upper refinement of a row starts at the last interior node of the
# velocity grid whose scaled velocity lies below x_e (1 - EDGE_MARGIN), so that no
# interior interval of the grid ends within a relative distance of 1e-4 of the
# support's edge x_e, where the panels in ln(x_e - x) take over.
EDGE_MARGIN = 1e-4
# A row with fewer than ROW_NODES_MIN construction nodes, or with fewer than
# ROW_NODES_MIN finite slopes across rows, is refused as unresolved, because the
# row's cubic Hermite interpolant and the interpolant's inverse are then built
# on too few nodes to represent the row's distribution.
ROW_NODES_MIN = 8
# Smallest argument passed to a logarithm. The value 1e-300 lies above the
# smallest normal double (about 2.2e-308), so the logarithm of a vanishing
# scaled velocity stays finite.
LOG_ARGUMENT_MIN = 1e-300
# The names of the two checks of a velocity table: "row_error", the table's own
# error across rows (the larger of the error in the cumulative probability u
# and the error relative to the scaled velocity x), and "polish_residual", the
# largest absolute residual in lambda_u that the polish leaves at a node.
VELOCITY_TABLE_KEYS = ("row_error", "polish_residual")

_log = logging.getLogger("pianistpy")


# ---- JAX functions of the velocity table. Each function takes the DF's lookup
# arrays and the model's numbers as traced arguments, so every model with the
# equal array shapes and equal lookup kinds reuses one compilation. ----


@functools.partial(jax.jit, static_argnums=(3,))
def _log10_df_kernel(e, u, arrays, kinds):
    """Return log10 F(E) at energies `e` of any shape. Where the depth
    `u` = p_0 - E is finite, the lookup reads F by the depth `u` instead of by
    the energy `e` (the depth argument of `evaluate_lookup`)."""
    return evaluate_lookup(arrays, kinds, e, xp=jnp, u=u)


@functools.partial(jax.jit, static_argnums=(3,))
def _kappa_kernel(e, u, arrays, kinds):
    """Return kappa = d ln F / d ln E at energies `e` of shape (n,), with the
    depth `u` used where the depth is finite, as `_kappa_at` defines."""
    return _kappa_at(e, u, arrays, kinds)


def _kappa_at(e, u, arrays, kinds):
    """Return kappa = d ln F / d ln E at energies `e` of shape (n,).

    Where the depth u = p_0 - E is finite, kappa = -E d ln F / du through the
    lookup's depth argument, because an energy E formed near p_0 changes only
    in steps of the last bit of E, while the depth u carries every digit.
    Where the depth is not finite, kappa is differentiated in E. At the DF
    table's lowest energy K_0, where the DF continues below K_0, kappa comes
    from the DF's rule below the table, whose derivative at K_0 is exact,
    instead of from the table, whose derivative at K_0 is one-sided. The
    bottom row of a velocity table whose lowest potential equals K_0 evaluates
    kappa at E = K_0 at every node, so the choice at K_0 sets the bottom row."""
    k_0 = arrays.params[K_0]

    def one(z, d):
        held = jnp.isfinite(d)
        by_k = jax.grad(lambda q: evaluate_lookup(arrays, kinds, q, xp=jnp))(z) * z
        if kinds.below_floor != "zero":
            at_k_0 = jax.grad(lambda q: _below(arrays, kinds, q, jnp))(z) * z
            by_k = jnp.where(z == k_0, at_k_0, by_k)
        by_u = -z * jax.grad(lambda v: evaluate_lookup(arrays, kinds, z, xp=jnp, u=v))(
            jnp.where(held, d, 1.0)
        )
        return jnp.where(held, by_u, by_k) * jnp.log(10.0)

    return jax.vmap(one)(e, u)


def _on_bucket(kernel, e, arrays, kinds, u=None):
    """Return `kernel` evaluated at host energies `e`, and at depths `u`
    where `u` is given, as a NumPy array of the shape of `e`. The inputs are
    padded to the next power of two in length, so `kernel` compiles once per
    power of two instead of once per input length."""
    e = np.asarray(e, dtype=np.float64)
    flat = e.ravel()
    if flat.size == 0:
        return np.zeros(e.shape)
    m = 1 << (flat.size - 1).bit_length()
    padded = [jnp.asarray(np.concatenate([flat, np.full(m - flat.size, flat[0])]))]
    if u is not None:
        u_flat = np.asarray(u, dtype=np.float64).ravel()
        padded.append(
            jnp.asarray(np.concatenate([u_flat, np.full(m - flat.size, np.nan)]))
        )
    return np.asarray(kernel(*padded, arrays, kinds))[: flat.size].reshape(e.shape)


# ---- JAX functions that build a chunk of rows at once. The construction,
# polish and slopes of every row run on fixed array shapes and repeat the
# operations of `_Rows.row`, the host reference. ----

# ROWS_PER_CHUNK is the number of rows passed to the row kernels at once.
# COUNT_BUCKET is the multiple to which each chunk's padded per-row counts are
# rounded up, so each rounded size compiles once.
ROWS_PER_CHUNK = 64
COUNT_BUCKET = 64
# KERNEL_ROWS = True builds the rows in the JAX row kernels; KERNEL_ROWS = False
# builds the rows one at a time by `_Rows.row` on the host, the reference that
# the row kernels reproduce.
KERNEL_ROWS = True


class _RowConfig(NamedTuple):
    """The static arguments of the row kernels: the lookup's kinds, whether
    the DF continues below its table, the name of the DF's rule below its
    table, and whether table nodes beyond a row's last construction node are
    placed by the tail's Newton iteration."""

    kinds: LookupKinds
    continues: bool
    rule: str
    tail_newton: bool


class _RowNumbers(NamedTuple):
    r"""The model's numbers that the row kernels read as traced arguments: the
    central potential p_0, the power :math:`2 - 2\beta_{\rm c}` of x in the density, the
    energy at the support's edge, the effective exponent of the DF as E -> 0,
    the exponent and the scale of the tail's logarithmic factor, the velocity
    grid, the panel edges of the lower refinement, the Gauss-Legendre nodes and
    weights, and the target lambda_u of the table nodes."""

    p_0: Array
    velocity_power: Array
    e_edge: Array
    a_eff: Array
    tail_log_exponent: Array
    tail_log_scale: Array
    x_nodes: Array
    sub_lo_edges: Array
    gl_x: Array
    gl_w: Array
    lam_t: Array


def _k_logpdf(psi, w, xe, x, d, arrays, cfg, nums):
    r"""Return :math:`\log_{10}[x^{2 - 2\beta_{\rm c}} F(E)]` at scaled velocities
    `x` on the row of potential `psi`, as `_Rows.logpdf` does on the host.

    Where the distance `d` = x_e - x is finite, the energy is formed as
    E = E_edge + Psi d (2 x_e - d), which keeps every digit of E near the
    edge; elsewhere E = Psi (1 - x^2). Where `d` is not finite and
    :math:`E \ge p_0/2`, F is read by the depth `w` + Psi x^2, where `w` = p_0 - Psi
    is the row's depth."""
    e = jnp.maximum(psi * (1.0 - x * x), 0.0)
    fd = jnp.isfinite(d)
    dd = jnp.where(fd, d, 0.0)
    e = jnp.where(fd, nums.e_edge + psi * dd * (2.0 * xe - dd), e)
    u = jnp.where(~fd & (e >= 0.5 * nums.p_0), w + psi * x * x, jnp.nan)
    return evaluate_lookup(arrays, cfg.kinds, e, xp=jnp, u=u) + nums.velocity_power * (
        jnp.log10(jnp.maximum(x, LOG_ARGUMENT_MIN))
    )


def _k_kappa(e, u, arrays, cfg):
    """Return kappa = d ln F / d ln E at energies `e` of any shape, with the
    depth `u` used where the depth is finite (`_kappa_at`)."""
    return _kappa_at(e.ravel(), u.ravel(), arrays, cfg.kinds).reshape(e.shape)


def _k_gl_mass(psi, w, xe, s0, s1, valid, by_d, arrays, cfg, nums):
    """Return the Gauss-Legendre masses of the density on the panels between
    `s0` and `s1`, where s is ln x, or ln d = ln(x_e - x) when `by_d` is true; a
    panel whose `valid` entry is false gets mass 0. Every panel's integrand is
    divided by the largest integrand value over all panels before the power of
    ten is taken, so the powers of ten stay finite."""
    mid, half = 0.5 * (s0 + s1), 0.5 * (s1 - s0)
    e = jnp.exp(mid[:, None] + half[:, None] * nums.gl_x[None, :])
    x = xe - e if by_d else e
    d = e if by_d else jnp.full(e.shape, jnp.nan)
    levels = _k_logpdf(psi, w, xe, x, d, arrays, cfg, nums)
    fin = jnp.isfinite(levels) & valid[:, None]
    peak = jnp.max(jnp.where(fin, levels, -jnp.inf))
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    vals = jnp.where(fin, 10.0 ** (levels - peak), 0.0) * e
    return jnp.where(valid, jnp.abs(half) * (vals @ nums.gl_w) * 10.0**peak, 0.0)


def _k_compact(mask, *arrays):
    """Return the count of true entries of `mask`, and each array of `arrays`
    reordered so that the entries where `mask` is true come first, in their
    original order."""
    n = mask.size
    order = jnp.argsort(jnp.where(mask, jnp.arange(n), n + jnp.arange(n)))
    return jnp.sum(mask).astype(jnp.int32), [a[order] for a in arrays]


def _k_floor_mass(e, arrays, cfg):
    """Return the integral of F from 0 to `e` under the DF's "taylor" rule or
    edge-law rule below its table, as `below_floor_mass` does on the host."""
    params = arrays.params
    if cfg.rule == "taylor":
        return params[FLOOR_0] * e * e / 2.0 + params[FLOOR_1] * e * e * e / 6.0
    p, c_0, c_1 = params[FLOOR_0], params[FLOOR_0 + 1], params[FLOOR_0 + 2]
    return c_0 * e ** (p + 1.0) / (p + 1.0) + c_1 * e ** (p + 2.0) / (p + 2.0)


def _k_hermite(xq, xn, fn, mn, n):
    """Return the value and the derivative at `xq` of the cubic Hermite
    interpolant through nodes `xn`, values `fn` and slopes `mn`, as `hermite`
    does, on a row whose first `n` nodes are real and whose remaining nodes are
    padding. Below the first real node and above the last real node the
    interpolant continues linearly with the end node's slope."""
    i = jnp.clip(jnp.searchsorted(xn, xq, side="right") - 1, 0, n - 2)
    h = xn[i + 1] - xn[i]
    t = (xq - xn[i]) / h
    t2, t3 = t * t, t * t * t
    val = (
        (2 * t3 - 3 * t2 + 1) * fn[i]
        + (t3 - 2 * t2 + t) * h * mn[i]
        + (-2 * t3 + 3 * t2) * fn[i + 1]
        + (t3 - t2) * h * mn[i + 1]
    )
    der = (
        (6 * t2 - 6 * t) / h * fn[i]
        + (3 * t2 - 4 * t + 1) * mn[i]
        + (-6 * t2 + 6 * t) / h * fn[i + 1]
        + (3 * t2 - 2 * t) * mn[i + 1]
    )
    lo, hi = xq < xn[0], xq > xn[n - 1]
    val = jnp.where(
        lo,
        fn[0] + mn[0] * (xq - xn[0]),
        jnp.where(hi, fn[n - 1] + mn[n - 1] * (xq - xn[n - 1]), val),
    )
    der = jnp.where(lo, mn[0], jnp.where(hi, mn[n - 1], der))
    return val, der


def _k_hermite_inverse(xn, fn, mn, n, target):
    """Return the point where the cubic Hermite interpolant of a padded row
    equals `target`, as `hermite_inverse` does: 60 bisection halvings of the
    bracketing interval, then two Newton steps, each Newton step taken only
    where the step's length is below the bracket's width plus :math:`10^{-12}`."""
    i = jnp.clip(jnp.searchsorted(fn, target, side="right") - 1, 0, n - 2)
    lo = jnp.where(target < fn[0], xn[0] + (target - fn[0]) / mn[0] - 1.0, xn[i])
    hi = jnp.where(
        target > fn[n - 1],
        xn[n - 1] + (target - fn[n - 1]) / mn[n - 1] + 1.0,
        xn[i + 1],
    )

    def halve(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        below = _k_hermite(mid, xn, fn, mn, n)[0] < target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 60, halve, (lo, hi))
    xq = 0.5 * (lo + hi)
    for _ in range(2):
        val, der = _k_hermite(xq, xn, fn, mn, n)
        step = (val - target) / jnp.maximum(der, 1e-300)
        xq = jnp.where(jnp.abs(step) < (hi - lo) + 1e-12, xq - step, xq)
    return xq


def _k_pair_masses(psi, w, xe, xa, da, xq, dq, arrays, cfg, nums, kappa_weight=False):
    """Return the signed mass of the density from each construction node
    (`xa`, at distance `da` from the edge) to the node's query point (`xq`, at
    distance `dq`), one Gauss-Legendre panel per pair, in ln x below x_e / 2
    and in ln(x_e - x) above, as `_Rows.pair_masses` does. Without
    `kappa_weight` (the polish), each pair's integrand is divided by the
    pair's own largest integrand value; with `kappa_weight` (the slopes across
    rows), the integrand carries the weight kappa(E) and every pair is divided
    by the row's largest integrand value."""
    up = xq >= 0.5 * xe
    a = jnp.where(up, jnp.log(da), jnp.log(xa))
    b = jnp.where(up, jnp.log(dq), jnp.log(xq))
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    e = jnp.exp(mid[:, None] + half[:, None] * nums.gl_x[None, :])
    xp = jnp.where(up[:, None], xe - e, e)
    d_up = jnp.where(up[:, None], e, jnp.nan)
    levels = _k_logpdf(psi, w, xe, xp, d_up, arrays, cfg, nums)
    fin = jnp.isfinite(levels)
    if kappa_weight:
        peak = jnp.max(jnp.where(fin, levels, -jnp.inf))
        peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
        fd = jnp.isfinite(d_up)
        dd = jnp.where(fd, d_up, 0.0)
        energies = jnp.where(
            fd,
            nums.e_edge + psi * dd * (2.0 * xe - dd),
            jnp.maximum(psi * (1.0 - xp * xp), 0.0),
        )
        depths = jnp.where(
            ~fd & (energies >= 0.5 * nums.p_0), w + psi * xp * xp, jnp.nan
        )
        weight = e * _k_kappa(energies, depths, arrays, cfg)
        values = jnp.where(fin, 10.0 ** (levels - peak), 0.0) * weight
        integral = half * (values @ nums.gl_w) * 10.0**peak
    else:
        peak = jnp.max(jnp.where(fin, levels, -jnp.inf), axis=1)
        peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
        values = jnp.where(fin, 10.0 ** (levels - peak[:, None]), 0.0) * e
        integral = half * (values @ nums.gl_w) * 10.0**peak
    return jnp.where(up, -integral, integral)


def _k_place_beyond(psi, xe, tot, d1, c_above, s_t, s_last, q_t, nums):
    """Return `s_t` with every table node beyond the row's last construction
    node `s_last` placed on the DF's logarithmic tail, as
    `_Rows._place_beyond` does, with the nodes selected by a mask instead of
    an index list."""
    a_eff, b, e_0 = nums.a_eff, nums.tail_log_exponent, nums.tail_log_scale
    e_1 = psi * d1 * (2.0 * xe - d1)
    m_1 = tail_mass_below(1.0, a_eff, e_1, b, e_0, jnp)
    s_1 = 1.0 - b
    target = q_t * tot / c_above * m_1
    ok = (s_t >= s_last) & (target > 0.0) & (target < m_1)
    goal = jnp.where(ok, target, m_1)
    y = jnp.where(ok, jnp.log(m_1 / goal) / (1.0 - a_eff), 0.0)

    def more(st):
        return (st[0] < TAIL_NEWTON_ITERATIONS) & jnp.any(st[2])

    def step(st):
        it, y, live = st
        e = e_1 * jnp.exp(-y)
        g = jnp.log(tail_mass_below(1.0, a_eff, e, b, e_0, jnp)) - jnp.log(goal)
        xx = (1.0 - a_eff) * jnp.log(e_0 / e)
        gamma = _scaled_upper_gamma_small_order(s_1, xx, jnp)
        dg = -(1.0 - a_eff) * xx ** (s_1 - 1.0) / gamma
        move = jnp.where(live, -g / dg, 0.0)
        y = jnp.maximum(0.0, y + move)
        live = live & ~(
            (jnp.abs(g) < TAIL_NEWTON_TOLERANCE)
            | (jnp.abs(move) < TAIL_NEWTON_TOLERANCE * jnp.maximum(1.0, y))
        )
        return it + 1, y, live

    y = jax.lax.while_loop(more, step, (jnp.int32(0), y, ok))[1]
    e = e_1 * jnp.exp(-y)
    d = (e / psi) / (xe + jnp.sqrt(jnp.maximum(xe * xe - e / psi, 0.0)))
    put = ok & (d > 0.0) & (d < xe)
    d_safe = jnp.where(put, d, 1.0)
    return jnp.where(put, jnp.log((xe - d_safe) / d_safe), s_t)


def _k_construct_row(
    psi, w, xe, k_hi, d1, n_h, d_hi, ratio, i_lo, i_hi, i_kind, i_own, i_valid, arrays,
    cfg, nums,
):  # fmt: skip
    """Return the construction of one row of the layout, as `_Rows.row` does
    before the polish: the construction nodes padded beyond their count, the
    row's total mass, the row's mass above the closest refined distance from
    the edge, the number of construction nodes after the monotonicity filter,
    and whether the kept construction nodes fail to increase in both s and
    lambda_u."""
    x_nodes, sub_lo = nums.x_nodes, nums.sub_lo_edges
    n_x = x_nodes.size
    args = (arrays, cfg, nums)
    s_lo = jnp.log(sub_lo)
    m_lo = _k_gl_mass(
        psi, w, xe, s_lo[:-1], s_lo[1:], s_lo[1:] > -jnp.inf, False, *args
    )
    n_hi = d_hi.size
    s_hi = jnp.log(d_hi)
    m_hi = _k_gl_mass(
        psi, w, xe, s_hi[:-1], s_hi[1:], jnp.arange(n_hi - 1) < n_h - 1, True, *args
    )
    up = i_kind == 1
    s0 = jnp.where(up, jnp.log(xe - i_lo), jnp.log(i_lo))
    s1 = jnp.where(up, jnp.log(xe - i_hi), jnp.log(i_hi))
    m_int = _k_gl_mass(psi, w, xe, s0, s1, i_valid & ~up, False, *args) + _k_gl_mass(
        psi, w, xe, s0, s1, i_valid & up, True, *args
    )
    inc = jnp.zeros(n_x - 1).at[i_own].add(jnp.where(i_valid, m_int, 0.0))
    k = jnp.arange(n_x - 1)
    inc = jnp.where((k >= 1) & (k < k_hi), inc, 0.0)
    x0 = sub_lo[0]
    c_below = (
        x0
        * 10.0 ** _k_logpdf(psi, w, xe, x0[None], jnp.full(1, jnp.nan), *args)[0]
        / (nums.velocity_power + 1.0)
    )
    if not cfg.continues:
        dg = 0.5 * d1 * (1.0 + nums.gl_x)
        c_above = (
            0.5
            * d1
            * jnp.sum(nums.gl_w * 10.0 ** _k_logpdf(psi, w, xe, xe - dg, dg, *args))
        )
    elif cfg.rule != "tail":
        c_above = _k_floor_mass(psi * d1 * (2.0 * xe - d1), arrays, cfg) / (2.0 * psi)
    else:
        edge = 10.0 ** _k_logpdf(psi, w, xe, (xe - d1)[None], d1[None], *args)[0]
        c_above = edge * d1 / (1.0 - nums.a_eff) * ratio
    c_lo = c_below + jnp.cumsum(m_lo)
    up_inc = jnp.cumsum(inc)
    c_low_last = c_lo[-1] + up_inc[-1]
    cc_hi = c_above + jnp.cumsum(m_hi[::-1])[::-1]
    tot = c_low_last + cc_hi[0]
    down_inc = jnp.cumsum(inc[::-1])[::-1]
    q_lo = (cc_hi[0] + up_inc[-1]) + jnp.concatenate(
        [jnp.cumsum(m_lo[::-1])[::-1][1:], jnp.zeros(1)]
    )
    # Cumulative masses at the interior grid nodes x_nodes[2] to x_nodes[k_hi]
    # and at the upper panel edges 1 to n_h - 1.
    i_b = jnp.arange(n_x - 2)
    u_b = c_lo[-1] + up_inc[i_b + 1]
    q_b = cc_hi[0] + jnp.where(
        i_b + 2 <= n_x - 2, down_inc[jnp.minimum(i_b + 2, n_x - 2)], 0.0
    )
    j_c = jnp.arange(1, n_hi)
    u_c = c_low_last + jnp.cumsum(m_hi)[j_c - 1]
    q_c = jnp.where(j_c == n_h - 1, c_above, cc_hi[jnp.minimum(j_c, n_hi - 2)])
    x_all = jnp.concatenate([sub_lo[1:], x_nodes[2:], xe - d_hi[1:]])
    d_all = jnp.concatenate([xe - sub_lo[1:], xe - x_nodes[2:], d_hi[1:]])
    u = jnp.concatenate([c_lo, u_b, u_c]) / tot
    q = jnp.concatenate([q_lo, q_b, q_c]) / tot
    valid = jnp.concatenate(
        [jnp.ones(c_lo.size, bool), i_b + 2 <= k_hi, j_c <= n_h - 1]
    )
    ok = valid & (u > 0.0) & (q > 0.0) & (x_all > 0.0) & (d_all > 0.0)
    n_1, (x_all, d_all, u, q) = _k_compact(ok, x_all, d_all, u, q)
    idx = jnp.arange(u.size)
    lam = jnp.log(u) - jnp.log(q)
    prog = jnp.concatenate(
        [
            jnp.ones(1, bool),
            (jnp.diff(u) >= 0.0) & (jnp.diff(q) <= 0.0) & (jnp.diff(lam) > 0.0),
        ]
    ) & (idx < n_1)
    n_2, (x_all, d_all, u, q) = _k_compact(prog, x_all, d_all, u, q)
    d_or_nan = jnp.where(x_all >= 0.5 * xe, d_all, jnp.nan)
    p_n = 10.0 ** (_k_logpdf(psi, w, xe, x_all, d_or_nan, *args) - jnp.log10(tot))
    s_n = jnp.log(x_all) - jnp.log(d_all)
    lam_n = jnp.log(u) - jnp.log(q)
    m_n = p_n * x_all * d_all / (xe * u * q)
    last = m_n[n_2 - 1]
    m_n = m_n.at[n_2 - 1].set(
        jnp.where(jnp.isfinite(last) & (last > 0.0), last, 1.0 - nums.a_eff)
    )
    keep = (idx < n_2) & jnp.isfinite(m_n) & (m_n > 0.0)
    n, (s_n, lam_n, m_n, xk, dk, uk, qk) = _k_compact(
        keep, s_n, lam_n, m_n, x_all, d_all, u, q
    )
    real = idx < n
    pair = real[1:] & real[:-1]
    monotone = jnp.all(
        jnp.where(pair, (jnp.diff(s_n) > 0.0) & (jnp.diff(lam_n) > 0.0), True)
    )
    s_n, lam_n = jnp.where(real, s_n, jnp.inf), jnp.where(real, lam_n, jnp.inf)
    m_n, xk, dk, uk, qk = (jnp.where(real, a, 1.0) for a in (m_n, xk, dk, uk, qk))
    nodes = (s_n, lam_n, m_n, xk, dk, uk, qk, n)
    return nodes, tot, c_above, n_2, ~monotone


def _k_polish_start(psi, xe, tot, nodes, nums):
    """Return the polish's starting state on one row, with each table node's
    first s from the inverse of the row's cubic Hermite interpolant, and a
    mask of the table nodes whose s lies strictly between the first and the
    last construction node. A table node outside the construction nodes is finished
    from the start."""
    s_n, lam_n, m_n, _, _, _, _, n = nodes
    s_t = _k_hermite_inverse(s_n, lam_n, m_n, n, nums.lam_t)
    inside = (s_t > s_n[0]) & (s_t < s_n[n - 1])
    state = (
        s_t,
        jnp.full(s_t.shape, s_n[0]),
        jnp.full(s_t.shape, s_n[n - 1]),
        ~inside,
        jnp.where(inside, jnp.inf, 0.0),
        s_t,
        jnp.zeros(s_t.shape, jnp.int32),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
    )
    return state, inside


def _k_polish_pass(state, psi, w, xe, tot, nodes, arrays, cfg, nums):
    """Return the polish state after one pass of the safeguarded Newton
    iteration of `_Rows.row` on one row.

    Every unfinished table node evaluates the node's residual in lambda_u. A
    table node finishes when the node's absolute residual is below
    NODE_TOLERANCE, when two successive passes both leave the node's absolute
    residual at or above the node's smallest absolute residual so far, or when
    the node's step is at most 4 times the spacing of doubles at max(|s|, 1).
    An unfinished node takes the Newton step where the Newton step lands inside
    the node's bracket; otherwise the node moves to the bracket's midpoint
    where both bracket ends have moved from the row's end nodes; otherwise the
    node takes the Newton step clipped to the width of the node's construction
    interval."""
    s_n, _, _, xk, dk, uk, qk, n = nodes
    lam_t = nums.lam_t
    u_t, q_t = 1.0 / (1.0 + jnp.exp(-lam_t)), 1.0 / (1.0 + jnp.exp(lam_t))
    s_t, lo_b, hi_b, done, best_r, best_s, strikes, it, bad, passes = state
    s_0, s_1 = s_n[0], s_n[n - 1]
    live = ~done
    x_q, d_q = xe / (1.0 + jnp.exp(-s_t)), xe / (1.0 + jnp.exp(s_t))
    i0 = jnp.clip(jnp.searchsorted(s_n, s_t, side="right") - 1, 0, n - 2)
    kn = jnp.where((s_t - s_n[i0]) <= (s_n[i0 + 1] - s_t), i0, i0 + 1)
    m = _k_pair_masses(psi, w, xe, xk[kn], dk[kn], x_q, d_q, arrays, cfg, nums) / tot
    r = lam_t - (jnp.log(uk[kn] + m) - jnp.log(qk[kn] - m))
    d_q_or_nan = jnp.where(x_q >= 0.5 * xe, d_q, jnp.nan)
    p_q = 10.0 ** (
        _k_logpdf(psi, w, xe, x_q, d_q_or_nan, arrays, cfg, nums) - jnp.log10(tot)
    )
    ms_q = xe * u_t * q_t / (p_q * x_q * d_q)
    bad = jnp.maximum(bad, jnp.sum(live & ~jnp.isfinite(r)).astype(jnp.int32))
    ar = jnp.abs(r)
    better = live & (ar < best_r)
    strikes = jnp.where(live, jnp.where(better, 0, strikes + 1), strikes)
    best_s, best_r = jnp.where(better, s_t, best_s), jnp.where(better, ar, best_r)
    fin = (ar < NODE_TOLERANCE) | (strikes >= 2)
    lo_b = jnp.where(live & ~fin & (r > 0.0), jnp.maximum(lo_b, s_t), lo_b)
    hi_b = jnp.where(live & ~fin & (r < 0.0), jnp.minimum(hi_b, s_t), hi_b)
    valid = jnp.isfinite(ms_q) & (ms_q > 0.0)
    w_q = s_n[i0 + 1] - s_n[i0]
    newton = s_t + jnp.where(valid, r * ms_q, 0.0)
    both = (lo_b > s_0) & (hi_b < s_1)
    new = jnp.where(
        valid & (newton > lo_b) & (newton < hi_b),
        newton,
        jnp.where(
            both,
            0.5 * (lo_b + hi_b),
            s_t + jnp.where(valid, jnp.clip(r * ms_q, -w_q, w_q), 0.0),
        ),
    )
    new = jnp.clip(new, s_0, s_1)
    fin = fin | (
        jnp.abs(new - s_t)
        <= 4.0 * np.finfo(np.float64).eps * jnp.maximum(jnp.abs(s_t), 1.0)
    )
    passes = passes + jnp.any(live).astype(jnp.int32)
    return (
        jnp.where(live & ~fin, new, s_t),
        lo_b,
        hi_b,
        done | (live & fin),
        best_r,
        best_s,
        strikes,
        it + 1,
        bad,
        passes,
    )


@functools.partial(jax.jit, static_argnums=(14,))
def _rows_construct_polish(
    psi, w, xe, k_hi, d1, n_h, d_hi, ratio, i_lo, i_hi, i_kind, i_own, i_valid, arrays,
    cfg, nums,
):  # fmt: skip
    """Return the construction and the polish of every row of a chunk. Polish
    passes repeat while any table node of the chunk is unfinished, up to
    POLISH_PASSES_MAX passes, and each table node inside the construction
    nodes keeps the s of the node's smallest absolute residual."""
    nodes, tot, c_above, n_2, not_monotone = jax.vmap(
        functools.partial(_k_construct_row, arrays=arrays, cfg=cfg, nums=nums)
    )(psi, w, xe, k_hi, d1, n_h, d_hi, ratio, i_lo, i_hi, i_kind, i_own, i_valid)
    start, inside = jax.vmap(functools.partial(_k_polish_start, nums=nums))(
        psi, xe, tot, nodes
    )

    def more(state):
        return (state[7][0] < POLISH_PASSES_MAX) & jnp.any(~state[3])

    def one_pass(state):
        return jax.vmap(
            functools.partial(_k_polish_pass, arrays=arrays, cfg=cfg, nums=nums)
        )(state, psi, w, xe, tot, nodes)

    state = jax.lax.while_loop(more, one_pass, start)
    s_t = jnp.where(inside, state[5], state[0])
    return dict(
        nodes=nodes, tot=tot, c_above=c_above, n_2=n_2, not_monotone=not_monotone,
        s_t=s_t, inside=inside, best_r=state[4], bad=state[8], passes=state[9],
    )  # fmt: skip


def _k_finish_row(psi, w, xe, tot, d1, c_above, s_t, nodes, arrays, cfg, nums):
    """Return one row after the polish, as `_Rows.row` does after the polish:
    the stack (s, ds / dlambda_u, ds / dlambda_p) at the row's table nodes, and
    the count of finite slopes across rows. Table nodes beyond the last
    construction node are first placed on a logarithmic tail where the
    configuration asks for the tail's Newton iteration; each non-finite slope
    across rows is replaced by linear interpolation in node index between the
    finite slopes."""
    s_n, lam_n, m_n, _, _, _, _, n = nodes
    lam_t = nums.lam_t
    u_t, q_t = 1.0 / (1.0 + jnp.exp(-lam_t)), 1.0 / (1.0 + jnp.exp(lam_t))
    if cfg.tail_newton:
        s_t = _k_place_beyond(psi, xe, tot, d1, c_above, s_t, s_n[n - 1], q_t, nums)
    args = (arrays, cfg, nums)
    x_t, d_t = xe / (1.0 + jnp.exp(-s_t)), xe / (1.0 + jnp.exp(s_t))
    d_t_or_nan = jnp.where(x_t >= 0.5 * xe, d_t, jnp.nan)
    p_t = 10.0 ** (_k_logpdf(psi, w, xe, x_t, d_t_or_nan, *args) - jnp.log10(tot))
    ms_t = xe * u_t * q_t / (p_t * x_t * d_t)
    der_t = _k_hermite(s_t, s_n, lam_n, m_n, n)[1]
    ms_t = jnp.where(
        jnp.isfinite(ms_t) & (ms_t > 0.0), ms_t, 1.0 / jnp.maximum(der_t, 1e-300)
    )
    vp = nums.velocity_power
    pdf_t = p_t * tot
    okp = jnp.isfinite(pdf_t) & (pdf_t > 0.0)
    dxe = nums.e_edge / (2.0 * psi * psi * xe)
    g_k = _k_pair_masses(
        psi, w, xe, x_t[:-1], d_t[:-1], x_t[1:], d_t[1:], *args, kappa_weight=True
    )
    g_k = jnp.where(jnp.isfinite(g_k), g_k, 0.0)
    e_ends = jnp.stack(
        [psi * (1.0 - x_t[0] ** 2), nums.e_edge + psi * d_t[-1] * (2.0 * xe - d_t[-1])]
    )
    u_ends = jnp.stack(
        [
            jnp.where(e_ends[0] >= 0.5 * nums.p_0, w + psi * x_t[0] ** 2, jnp.nan),
            jnp.nan,
        ]
    )
    k_end = _k_kappa(e_ends, u_ends, arrays, cfg)
    pe: Array | float
    if cfg.continues:
        pe = 0.0
    else:
        pdf_e = 10.0 ** _k_logpdf(psi, w, xe, xe[None], jnp.zeros(1), *args)[0]
        pe = jnp.where(jnp.isfinite(pdf_e), pdf_e, 0.0)
    k_b = jnp.where(okp[0], k_end[0] * pdf_t[0] * x_t[0] / (vp + 1.0), 0.0) + (
        jnp.concatenate([jnp.zeros(1), jnp.cumsum(g_k)])
    )
    k_a = k_end[1] * q_t[-1] * tot + jnp.concatenate(
        [jnp.cumsum(g_k[::-1])[::-1], jnp.zeros(1)]
    )
    jm = lam_t.size // 2
    k_total = k_b[jm] + k_a[jm]
    t_du = jnp.where(
        lam_t <= 0.0,
        (k_b - u_t * k_total) / psi - u_t * pe * dxe,
        -((k_a - q_t * k_total) / psi + u_t * pe * dxe),
    )
    rs_t = (psi * w / nums.p_0) * (-(t_du / pdf_t) * xe / (x_t * d_t) - dxe / d_t)
    good = okp & jnp.isfinite(rs_t)
    n_good, (where_good, values) = _k_compact(
        good, jnp.arange(rs_t.size).astype(jnp.float64), rs_t
    )
    idx = jnp.arange(rs_t.size)
    where_good = jnp.where(idx < n_good, where_good, 1e300)
    values = jnp.where(idx < n_good, values, values[jnp.maximum(n_good - 1, 0)])
    rs_t = jnp.where(
        good, rs_t, jnp.interp(idx.astype(jnp.float64), where_good, values)
    )
    return jnp.stack([s_t, ms_t, rs_t]), n_good


@functools.partial(jax.jit, static_argnums=(9,))
def _rows_finish(psi, w, xe, tot, d1, c_above, s_t, nodes, arrays, cfg, nums):
    """Return `_k_finish_row` applied to every row of a chunk."""
    return jax.vmap(
        functools.partial(_k_finish_row, arrays=arrays, cfg=cfg, nums=nums)
    )(psi, w, xe, tot, d1, c_above, s_t, nodes)


class _Rows:
    """The quantities that every row of one model's velocity table needs, fixed
    when the table is built, and the host functions that build one row."""

    def __init__(self, model, settings) -> None:
        sampling = settings.sampling
        component = 0
        anisotropy = model.components[component].anisotropy
        form = model.form
        self.p_0 = float(form.p_0)
        # The DF is read through the package's lookup. The lookup arrays are
        # held once on the device, as traced arguments of every kernel, and once
        # on the host.
        self.host_arrays, self.kinds = prepare_lookup(model.table, form, component)
        self.arrays = LookupArrays(*(jnp.asarray(a) for a in self.host_arrays))
        # The DF's lowest energy K_0, counting the nodes below the written
        # table.
        self.q_min = float(self.host_arrays.params[K_0])
        self.n_below = model.table.nodes_below(component)
        self.q_min_written = float(
            model.table.k[component, model.table.i_first[component]]
        )
        self.tail_amplitude = float(form.tail_amplitude[component])
        rule = form.below_floor[component]
        self.rule = rule
        self.tail_log_exponent = float(form.tail_log_exponent[component])
        self.tail_log_scale = float(form.tail_log_scale[component])
        # floor_s is the power of the DF as K -> 0 below the DF's table. For a
        # tail and for an edge law, floor_s is the exponent of the asymptotic
        # form. For the "taylor" rule F = F_1 K + F_2 K^2 / 2, floor_s is the
        # power of the leading term: 1 where F_1 is not zero, and 2 where F_1 is
        # zero. Under the "zero" rule the support ends at the table's lowest
        # energy, and floor_s does not enter the density.
        if rule == "taylor":
            floor_s = 1.0 if form.below_floor_coefficients[component][0] != 0.0 else 2.0
        else:
            floor_s = form.exponent_at_zero(component)
        self.floor_s = floor_s
        self.continues = rule != "zero"
        if self.continues and not floor_s > -1.0:
            raise UnphysicalModelError(
                f"Below the DF's table, the DF goes as K^{floor_s:g} under the "
                f"{rule!r} rule, so the DF's mass is not finite at K = 0."
            )
        beta_eff = float(anisotropy.beta_c)
        self.velocity_power = 2.0 - 2.0 * beta_eff
        self.p_map = 1.0 + floor_s if (self.continues and floor_s < 0.0) else 1.0
        self.n_intervals = int(sampling["n_velocity_intervals"])
        self.n_nodes = int(sampling["n_nodes"])
        self.lnu0 = math.log(sampling["tail_mass_min"])
        self.panels_per_decade = int(sampling["panels_per_decade"])
        self.edge_refined = float(sampling["edge_refined"])
        self.x_nodes = (
            np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, self.n_intervals + 1)) ** 1.5
        )
        low = float(sampling["velocity_low_refined"])
        self.sub_lo_edges = np.exp(
            np.linspace(
                math.log(low),
                math.log(self.x_nodes[1]),
                _panel_count(
                    self.panels_per_decade, self.x_nodes[1] / low, "lower end"
                ),
            )
        )
        self.gl_x, self.gl_w = np.polynomial.legendre.leggauss(
            int(sampling["n_gauss_legendre"])
        )

    def log10_df(self, e, u=None):
        """Return log10 F at host energies `e`, read by the depth `u` where
        `u` is finite."""
        if u is None:
            u = np.full(np.shape(e), np.nan)
        return _on_bucket(_log10_df_kernel, e, self.arrays, self.kinds, u)

    def kappa(self, e, u=None):
        """Return kappa = d ln F / d ln E at host energies `e`, read by the
        depth `u` where `u` is finite."""
        if u is None:
            u = np.full(np.shape(e), np.nan)
        return _on_bucket(_kappa_kernel, e, self.arrays, self.kinds, u)

    # ---- The density of the scaled velocity, evaluated on the host. ----

    def support_edge(self, psi):
        """Return the support's edge x_e on the row of potential `psi`: 1 where
        F continues below its table, else (1 - K_0 / Psi)^{1/2}, the scaled
        velocity at which the energy Psi (1 - x^2) reaches the DF's lowest energy
        K_0."""
        return 1.0 if self.continues else math.sqrt(max(1.0 - self.q_min / psi, 0.0))

    def hi_edges(self, psi, x_from):
        """Return the panel edges of the upper refinement on the row of
        potential `psi`, as decreasing distances d = x_e - x from
        x_e - `x_from` down to the closest refined distance, and the support's
        edge x_e. The closest refined distance is the smaller of
        sampling["edge_refined"] x_e and, where F continues below its table,
        the distance at which the energy reaches K_0."""
        xe = self.support_edge(psi)
        q = self.q_min
        d_pl = (
            (q / psi) / (xe + math.sqrt(max(1.0 - q / psi, 0.0)))
            if self.continues
            else self.edge_refined * xe
        )
        d0, d1 = xe - x_from, min(self.edge_refined * xe, max(d_pl, 1e-300))
        count = _panel_count(
            self.panels_per_decade, d0 / d1, f"edge at Psi = {psi:.6e}"
        )
        return np.exp(np.linspace(math.log(d0), math.log(d1), count)), xe

    def energy(self, psi, x, d=None, xe=None):
        """Return the energy E = Psi (1 - x^2) at scaled velocities `x`. Where the
        distance `d` = x_e - x is given and finite, E is formed as
        E = E_edge + Psi d (2 x_e - d), which keeps every digit of E near the
        edge; E_edge is 0 where F continues below its table and K_0
        otherwise."""
        e = np.maximum(psi * (1.0 - x * x), 0.0)
        if d is not None:
            d = np.asarray(d, dtype=np.float64)
            e_edge = 0.0 if self.continues else float(self.q_min)
            fin = np.isfinite(d)
            dd = np.where(fin, d, 0.0)
            e = np.where(fin, e_edge + psi * dd * (2.0 * xe - dd), e)
        return e

    def depth(self, psi, w, x, d, e):
        r"""Return the depth u = p_0 - E where the depth is known exactly, and
        NaN elsewhere.

        Where E is formed from x (no finite distance `d`) and
        ``E >= p_0 / 2``, the depth is u = `w` + Psi x^2, where `w` = p_0 - Psi
        is the row's depth as `build_velocity_table` passes the depth; both terms
        are exact because ``Psi >= E >= p_0 / 2``. The energy formed as
        Psi (1 - x^2) loses the depth's digits there: on the top rows the last
        bit of E is :math:`10^{-10}` of the depth, so F read by E would change
        in steps of the last bit of E instead of continuously with x."""
        x = np.asarray(x, dtype=np.float64)
        by_x = (
            np.ones(np.shape(e), dtype=bool)
            if d is None
            else ~np.isfinite(np.asarray(d, dtype=np.float64))
        )
        with np.errstate(invalid="ignore"):
            return np.where(by_x & (e >= 0.5 * self.p_0), w + psi * x * x, np.nan)

    def logpdf(self, psi, w, x, d=None, xe=None):
        """Return log10 of ``x^(2 - 2 beta_c) F(E)`` at host scaled velocities `x`
        on the row of potential `psi`, with F read by the depth where `depth`
        gives an exact depth."""
        e = self.energy(psi, x, d, xe)
        return self.log10_df(e, self.depth(psi, w, x, d, e)) + self.velocity_power * (
            np.log10(np.maximum(x, 1e-300))
        )

    def panel_masses_d(self, psi, w, d_edges, xe):
        """Return the mass of the density on each panel between consecutive
        edges `d_edges`, given as distances from the support's edge x_e, by
        Gauss-Legendre quadrature in ln d."""
        s0, s1 = np.log(d_edges[:-1]), np.log(d_edges[1:])
        mid, half = 0.5 * (s0 + s1), 0.5 * (s1 - s0)
        sp = mid[:, None] + half[:, None] * self.gl_x[None, :]
        dp = np.exp(sp)
        levels = self.logpdf(psi, w, (xe - dp).ravel(), dp.ravel(), xe).reshape(
            dp.shape
        )
        peak = (
            np.max(levels[np.isfinite(levels)]) if np.any(np.isfinite(levels)) else 0.0
        )
        vals = np.where(np.isfinite(levels), 10.0 ** (levels - peak), 0.0) * dp
        return np.abs(half) * (vals @ self.gl_w) * 10.0**peak

    def panel_masses(self, psi, w, edges, xe=None):
        """Return the mass of the density on each panel between consecutive
        scaled velocities `edges`, by Gauss-Legendre quadrature in ln x, or in
        ln(x_e - x) when `xe` is given."""
        if xe is not None:
            s0, s1 = np.log(xe - edges[:-1]), np.log(xe - edges[1:])
        else:
            s0, s1 = np.log(edges[:-1]), np.log(edges[1:])
        mid, half = 0.5 * (s0 + s1), 0.5 * (s1 - s0)
        sp = mid[:, None] + half[:, None] * self.gl_x[None, :]
        xp = (xe - np.exp(sp)) if xe is not None else np.exp(sp)
        levels = self.logpdf(
            psi, w, xp.ravel(), np.exp(sp).ravel() if xe is not None else None, xe
        ).reshape(xp.shape)
        jac = np.exp(sp) if xe is not None else xp
        peak = (
            np.max(levels[np.isfinite(levels)]) if np.any(np.isfinite(levels)) else 0.0
        )
        vals = np.where(np.isfinite(levels), 10.0 ** (levels - peak), 0.0) * jac
        return np.abs(half) * (vals @ self.gl_w) * 10.0**peak

    def interior_masses(self, psi, w, k_hi, xe):
        """Return the mass of the density on each interior interval of the
        velocity grid, x_nodes[1] to x_nodes[k_hi]. Each interval is split into
        sub-panels at sampling["panels_per_decade"] panels per decade of the
        interval's local variable: ln x below x_e / 2 and ln(x_e - x) above."""
        xn = self.x_nodes[1 : k_hi + 1]
        up = xn[:-1] >= 0.5 * xe
        out = np.zeros(xn.size - 1)
        for kind in (False, True):
            idx = np.flatnonzero(up == kind)
            if idx.size == 0:
                continue
            a, b = (xe - xn[idx], xe - xn[idx + 1]) if kind else (xn[idx], xn[idx + 1])
            nsub = np.maximum(
                np.ceil(self.panels_per_decade * np.abs(np.log10(b / a))).astype(int), 1
            )
            la = np.array([math.log(v) for v in a])
            lb = np.array([math.log(v) for v in b])
            owner = np.repeat(np.arange(idx.size), nsub)
            k = (
                np.arange(int(nsub.sum())) - np.repeat(np.cumsum(nsub) - nsub, nsub) + 1
            ).astype(np.float64)
            e = np.exp(k * ((lb - la) / nsub)[owner] + la[owner])
            e[np.cumsum(nsub) - 1] = b
            edges = np.concatenate([[a[0]], e])
            m = self.panel_masses(
                psi, w, (xe - edges) if kind else edges, xe if kind else None
            )
            np.add.at(out, idx[owner], m)
        return out

    def pair_masses(self, psi, w, xa, xq, dq, xe, da=None, xshift=0.0, kappa_w=None):
        """Return the signed mass of the density from each construction node
        `xa` to the node's query point `xq`, one Gauss-Legendre panel per pair,
        in ln x below x_e / 2 and in ln(x_e - x) above. With `kappa_w` = c the
        integrand carries the weight c + kappa(E); with `xshift` the integrand
        carries the extra factor x^xshift."""
        up = xq >= 0.5 * xe
        a = np.where(up, np.log((xe - xa) if da is None else da), np.log(xa))
        b = np.where(up, np.log(dq), np.log(xq))
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        e = np.exp(mid[:, None] + half[:, None] * self.gl_x[None, :])
        xp = np.where(up[:, None], xe - e, e)
        d_up = np.where(up[:, None], e, np.nan).ravel()
        levels = self.logpdf(psi, w, xp.ravel(), d_up, xe).reshape(xp.shape)
        peak = (
            np.max(levels[np.isfinite(levels)]) if np.any(np.isfinite(levels)) else 0.0
        )
        weight = e if xshift == 0.0 else e * xp**xshift
        if kappa_w is not None:
            energies = self.energy(psi, xp.ravel(), d_up, xe)
            depths = self.depth(psi, w, xp.ravel(), d_up, energies)
            kappa = self.kappa(energies, depths).reshape(xp.shape)
            weight = weight * (float(kappa_w) + kappa)
        values = np.where(np.isfinite(levels), 10.0 ** (levels - peak), 0.0) * weight
        integral = half * (values @ self.gl_w) * 10.0**peak
        return np.where(up, -integral, integral)

    def floor_mass_below(self, e):
        """Return the integral of F from 0 to `e` under the DF's rule below
        the DF's table."""
        return below_floor_mass(self.host_arrays, self.kinds, e)

    # ---- One row of the velocity table, built on the host. ----

    def row(self, psi, w):
        """Return one row of the velocity table at potential `psi` and depth
        `w` = p_0 - Psi: the stack (s, ds / dlambda_u, ds / dlambda_p) at the
        row's table nodes, the largest absolute residual in lambda_u that the
        polish leaves at a table node inside the construction nodes, the row's
        total mass, and the support's edge x_e."""
        vp = self.velocity_power
        x_nodes = self.x_nodes
        xe0 = self.support_edge(psi)
        below_edge = np.flatnonzero(x_nodes < xe0 * (1.0 - EDGE_MARGIN))
        if below_edge.size < 2:
            raise AccuracyError(
                f"The velocity support of the velocity table's row at Psi = {psi:.6e} "
                f"ends at x_e = {xe0:.3e}, inside the first interval of the "
                "velocity grid, where the lower refinement of the grid ends."
            )
        k_hi = int(below_edge[-1])
        k_hi = min(k_hi, x_nodes.size - 2)
        sub_hi_d, xe = self.hi_edges(psi, x_nodes[k_hi])
        # The masses of the velocity grid's interior intervals, x_nodes[1] to
        # x_nodes[k_hi].
        inc = self.interior_masses(psi, w, k_hi, xe) if k_hi > 1 else np.zeros(0)
        m_lo = self.panel_masses(psi, w, self.sub_lo_edges)
        m_hi = self.panel_masses_d(psi, w, sub_hi_d, xe)
        x0 = self.sub_lo_edges[0]
        c_below = x0 * 10.0 ** self.logpdf(psi, w, np.array([x0]))[0] / (vp + 1.0)
        a_eff = -self.floor_s if self.continues else 0.0
        d1 = float(sub_hi_d[-1])
        if not self.continues:
            dg = 0.5 * d1 * (1.0 + self.gl_x)
            c_above = (
                0.5
                * d1
                * float(
                    np.sum(self.gl_w * 10.0 ** self.logpdf(psi, w, xe - dg, dg, xe))
                )
            )
        elif self.rule != "tail":
            e_1 = psi * d1 * (2.0 * xe - d1)
            c_above = float(self.floor_mass_below(e_1)[0]) / (2.0 * psi)
        else:
            e_1 = psi * d1 * (2.0 * xe - d1)
            ratio = (
                float(
                    tail_mass_below(
                        1.0, a_eff, e_1, self.tail_log_exponent, self.tail_log_scale
                    )[0]
                )
                * (1.0 - a_eff)
                / (
                    e_1 ** (1.0 - a_eff)
                    * np.log(self.tail_log_scale / e_1) ** (-self.tail_log_exponent)
                )
                if self.tail_log_exponent != 0.0
                else 1.0
            )
            c_above = (
                10.0 ** self.logpdf(psi, w, np.array([xe - d1]), np.array([d1]), xe)[0]
                * d1
                / (1.0 - a_eff)
                * ratio
            )
        c_lo = c_below + np.cumsum(m_lo)
        cc_hi = c_above + np.cumsum(m_hi[::-1])[::-1]
        x_low = np.concatenate([self.sub_lo_edges[1:], x_nodes[2 : k_hi + 1]])
        c_low = np.concatenate([c_lo, c_lo[-1] + np.cumsum(inc)])
        cc_int = cc_hi[0] + np.cumsum(inc[::-1])[::-1]
        tot = c_low[-1] + cc_hi[0]
        x_all = np.concatenate([x_low, xe - sub_hi_d[1:]])
        d_all = np.concatenate([xe - x_low, sub_hi_d[1:]])
        q_lo = cc_int[0] + np.concatenate([np.cumsum(m_lo[::-1])[::-1][1:], [0.0]])
        u_hi = c_low[-1] + np.cumsum(m_hi)
        u = np.concatenate([c_low, u_hi]) / tot
        q = np.concatenate([q_lo, cc_int[1:], cc_hi, [c_above]]) / tot
        ok = (u > 0.0) & (q > 0.0) & (x_all > 0.0) & (d_all > 0.0)
        x_all, d_all, u, q = x_all[ok], d_all[ok], u[ok], q[ok]
        prog = np.concatenate(
            [
                [True],
                (np.diff(u) >= 0.0)
                & (np.diff(q) <= 0.0)
                & (np.diff(np.log(u) - np.log(q)) > 0.0),
            ]
        )
        x_all, d_all, u, q = x_all[prog], d_all[prog], u[prog], q[prog]
        if x_all.size < ROW_NODES_MIN:
            raise AccuracyError(
                f"The velocity table's row at Psi = {psi:.6e} carries no resolvable "
                f"mass on the row's {x_all.size} nodes."
            )
        d_or_nan = np.where(x_all >= 0.5 * xe, d_all, np.nan)
        p_n = 10.0 ** (self.logpdf(psi, w, x_all, d_or_nan, xe) - np.log10(tot))
        s_n = np.log(x_all) - np.log(d_all)
        lam_n = np.log(u) - np.log(q)
        m_n = p_n * x_all * d_all / (xe * u * q)
        m_n[-1] = m_n[-1] if (np.isfinite(m_n[-1]) and m_n[-1] > 0.0) else 1.0 - a_eff
        keep = np.isfinite(m_n) & (m_n > 0.0)
        s_n, lam_n, m_n = s_n[keep], lam_n[keep], m_n[keep]
        if not (np.all(np.diff(s_n) > 0.0) and np.all(np.diff(lam_n) > 0.0)):
            raise ResultError(
                "The cumulative distribution on the velocity table's row at "
                f"Psi = {psi:.6e} is not monotone in (s, lambda_u)."
            )
        lam_t = np.linspace(self.lnu0, -self.lnu0, self.n_nodes)
        s_t, _ = hermite_inverse(lam_t, s_n, lam_n, m_n)
        xk, dk, uk, qk = x_all[keep], d_all[keep], u[keep], q[keep]
        u_t, q_t = 1.0 / (1.0 + np.exp(-lam_t)), 1.0 / (1.0 + np.exp(lam_t))
        inside = (s_t > s_n[0]) & (s_t < s_n[-1])
        res = 0.0
        lo_b, hi_b = np.full(s_t.shape, s_n[0]), np.full(s_t.shape, s_n[-1])
        done = ~inside
        best_r = np.where(inside, np.inf, 0.0)
        best_s = s_t.copy()
        strikes = np.zeros(s_t.shape, dtype=int)
        for _ in range(POLISH_PASSES_MAX):
            act = np.flatnonzero(~done)
            if act.size == 0:
                break
            n_pad = max(16, 1 << int(np.ceil(np.log2(act.size))))
            ix = np.concatenate([act, np.full(n_pad - act.size, act[0])])
            sa = s_t[ix]
            x_q, d_q = xe / (1.0 + np.exp(-sa)), xe / (1.0 + np.exp(sa))
            i0 = np.clip(np.searchsorted(s_n, sa, side="right") - 1, 0, s_n.size - 2)
            kn = np.where((sa - s_n[i0]) <= (s_n[i0 + 1] - sa), i0, i0 + 1)
            with np.errstate(all="ignore"):
                m = self.pair_masses(psi, w, xk[kn], x_q, d_q, xe, da=dk[kn]) / tot
                r = lam_t[ix] - (np.log(uk[kn] + m) - np.log(qk[kn] - m))
                d_q_or_nan = np.where(x_q >= 0.5 * xe, d_q, np.nan)
                p_q = 10.0 ** (self.logpdf(psi, w, x_q, d_q_or_nan, xe) - np.log10(tot))
                ms_q = xe * u_t[ix] * q_t[ix] / (p_q * x_q * d_q)
            r, sa, i0, ms_q = (
                r[: act.size],
                sa[: act.size],
                i0[: act.size],
                ms_q[: act.size],
            )
            if not np.all(np.isfinite(r)):
                raise ResultError(
                    "The residual of the exact cumulative distribution on the velocity "
                    f"table's row at Psi = {psi:.6e} is not finite at "
                    f"{int(np.sum(~np.isfinite(r)))} node(s)."
                )
            better = np.abs(r) < best_r[act]
            strikes[act] = np.where(better, 0, strikes[act] + 1)
            best_s[act] = np.where(better, sa, best_s[act])
            best_r[act] = np.where(better, np.abs(r), best_r[act])
            fin = (np.abs(r) < NODE_TOLERANCE) | (strikes[act] >= 2)
            lo_b[act] = np.where(~fin & (r > 0.0), np.maximum(lo_b[act], sa), lo_b[act])
            hi_b[act] = np.where(~fin & (r < 0.0), np.minimum(hi_b[act], sa), hi_b[act])
            valid = np.isfinite(ms_q) & (ms_q > 0.0)
            w_q = s_n[i0 + 1] - s_n[i0]
            newton = sa + np.where(valid, r * ms_q, 0.0)
            lo_a, hi_a = lo_b[act], hi_b[act]
            both = (lo_a > s_n[0]) & (hi_a < s_n[-1])
            new = np.where(
                valid & (newton > lo_a) & (newton < hi_a),
                newton,
                np.where(
                    both,
                    0.5 * (lo_a + hi_a),
                    sa + np.where(valid, np.clip(r * ms_q, -w_q, w_q), 0.0),
                ),
            )
            new = np.clip(new, s_n[0], s_n[-1])
            fin = fin | (
                np.abs(new - sa)
                <= 4.0 * np.finfo(np.float64).eps * np.maximum(np.abs(sa), 1.0)
            )
            done[act] = fin
            s_t[act] = np.where(fin, s_t[act], new)
        s_t = np.where(inside, best_s, s_t)
        beyond = np.flatnonzero(s_t >= s_n[-1])
        if self.tail_amplitude > 0.0 and self.tail_log_exponent != 0.0 and beyond.size:
            self._place_beyond(psi, s_t, beyond, q_t, tot, c_above, d1, xe, a_eff)
        if np.any(inside):
            res = float(np.max(np.where(inside, best_r, 0.0)))
        u_t, q_t = 1.0 / (1.0 + np.exp(-lam_t)), 1.0 / (1.0 + np.exp(lam_t))
        x_t, d_t = xe / (1.0 + np.exp(-s_t)), xe / (1.0 + np.exp(s_t))
        d_t_or_nan = np.where(x_t >= 0.5 * xe, d_t, np.nan)
        p_t = 10.0 ** (self.logpdf(psi, w, x_t, d_t_or_nan, xe) - np.log10(tot))
        ms_t = xe * u_t * q_t / (p_t * x_t * d_t)
        _, der_t = hermite(s_t, s_n, lam_n, m_n)
        ms_t = np.where(
            np.isfinite(ms_t) & (ms_t > 0.0), ms_t, 1.0 / np.maximum(der_t, 1e-300)
        )
        rs_t = self._across_rows(psi, w, x_t, d_t, p_t, tot, xe, u_t, q_t, lam_t)
        return np.stack([s_t, ms_t, rs_t]), res, tot, xe

    def _place_beyond(self, psi, s_t, beyond, q_t, tot, c_above, d1, xe, a_eff):
        """Place in `s_t` each table node whose index is in `beyond`, a node
        beyond the row's last construction node, on the DF's tail with a
        logarithmic factor of exponent tail_log_exponent and scale
        tail_log_scale. A Newton iteration in y = ln(E_1 / E), where E_1 is the
        energy at the closest refined distance `d1` from the edge, matches the
        logarithm of the tail's mass below E to the logarithm of the node's
        target mass."""
        b, e0 = self.tail_log_exponent, self.tail_log_scale
        e_1 = psi * d1 * (2.0 * xe - d1)
        m_1 = float(tail_mass_below(1.0, a_eff, e_1, b, e0)[0])
        s_1 = 1.0 - b
        target = q_t[beyond] * tot / c_above * m_1
        ok = (target > 0.0) & (target < m_1)
        y = np.where(ok, np.log(m_1 / np.where(ok, target, m_1)) / (1.0 - a_eff), 0.0)
        live = ok.copy()
        for _ in range(TAIL_NEWTON_ITERATIONS):
            if not np.any(live):
                break
            e = e_1 * np.exp(-y)
            g = np.log(tail_mass_below(1.0, a_eff, e, b, e0)) - np.log(
                np.where(ok, target, m_1)
            )
            xx = (1.0 - a_eff) * np.log(e0 / e)
            dg = -(1.0 - a_eff) * xx ** (s_1 - 1.0) / scaled_upper_gamma(s_1, xx)
            step = np.where(live, -g / dg, 0.0)
            y = np.maximum(0.0, y + step)
            live &= ~(
                (np.abs(g) < TAIL_NEWTON_TOLERANCE)
                | (np.abs(step) < TAIL_NEWTON_TOLERANCE * np.maximum(1.0, y))
            )
        e = e_1 * np.exp(-y)
        d = (e / psi) / (xe + np.sqrt(np.maximum(xe * xe - e / psi, 0.0)))
        put = ok & (d > 0.0) & (d < xe)
        s_t[beyond[put]] = np.log((xe - d[put]) / d[put])

    def _across_rows(self, psi, w, x_t, d_t, p_t, tot, xe, u_t, q_t, lam_t):
        """Return ds / dlambda_p at fixed cumulative probability u at the row's
        table nodes, from the derivative of the density in Psi. Each non-finite
        slope is replaced by linear interpolation in node index between the
        finite slopes."""
        vp = self.velocity_power
        pdf_t = p_t * tot
        okp = np.isfinite(pdf_t) & (pdf_t > 0.0)
        dxe = (0.0 if self.continues else float(self.q_min)) / (2.0 * psi * psi * xe)
        with np.errstate(all="ignore"):
            g_k = self.pair_masses(
                psi, w, x_t[:-1], x_t[1:], d_t[1:], xe, da=d_t[:-1], kappa_w=0.0
            )
            e_ends = np.array(
                [
                    psi * (1.0 - x_t[0] ** 2),
                    float(
                        self.energy(psi, np.array([x_t[-1]]), np.array([d_t[-1]]), xe)[
                            0
                        ]
                    ),
                ]
            )
            u_ends = np.array(
                [
                    float(self.depth(psi, w, np.array([x_t[0]]), None, e_ends[:1])[0]),
                    np.nan,
                ]
            )
            k_end = self.kappa(e_ends, u_ends)
            pdf_e = (
                10.0
                ** float(self.logpdf(psi, w, np.array([xe]), np.array([0.0]), xe)[0])
                if dxe != 0.0
                else 0.0
            )
        g_k = np.where(np.isfinite(g_k), g_k, 0.0)
        k_b = (
            float(k_end[0]) * pdf_t[0] * x_t[0] / (vp + 1.0) if okp[0] else 0.0
        ) + np.concatenate([[0.0], np.cumsum(g_k)])
        k_a = float(k_end[1]) * q_t[-1] * tot + np.concatenate(
            [np.cumsum(g_k[::-1])[::-1], [0.0]]
        )
        jm = lam_t.size // 2
        k_total = k_b[jm] + k_a[jm]
        pe = pdf_e if np.isfinite(pdf_e) else 0.0
        with np.errstate(all="ignore"):
            t_du = np.where(
                lam_t <= 0.0,
                (k_b - u_t * k_total) / psi - u_t * pe * dxe,
                -((k_a - q_t * k_total) / psi + u_t * pe * dxe),
            )
            rs_t = (psi * w / self.p_0) * (
                -(t_du / pdf_t) * xe / (x_t * d_t) - dxe / d_t
            )
        good = okp & np.isfinite(rs_t)
        if good.sum() < ROW_NODES_MIN:
            raise ResultError(
                "The slope across rows on the velocity table's row at "
                f"Psi = {psi:.6e} is finite at only {int(good.sum())} node(s)."
            )
        return np.where(
            good,
            rs_t,
            np.interp(np.arange(rs_t.size), np.flatnonzero(good), rs_t[good]),
        )

    # ---- The rows of the velocity table, built in the JAX row kernels. ----

    def kernel_inputs(self):
        """Return the static configuration (`_RowConfig`) and the traced
        numbers (`_RowNumbers`) of the row kernels for the model of `_Rows`."""
        b = self.tail_log_exponent
        cfg = _RowConfig(
            kinds=self.kinds,
            continues=self.continues,
            rule=self.rule,
            tail_newton=(
                self.rule == "tail"
                and self.tail_amplitude > 0.0
                and b != 0.0
                and -1.0 < 1.0 - b < 0.0
            ),
        )
        nums = _RowNumbers(
            p_0=jnp.float64(self.p_0),
            velocity_power=jnp.float64(self.velocity_power),
            e_edge=jnp.float64(0.0 if self.continues else self.q_min),
            a_eff=jnp.float64(-self.floor_s if self.continues else 0.0),
            tail_log_exponent=jnp.float64(b),
            tail_log_scale=jnp.float64(self.tail_log_scale),
            x_nodes=jnp.asarray(self.x_nodes),
            sub_lo_edges=jnp.asarray(self.sub_lo_edges),
            gl_x=jnp.asarray(self.gl_x),
            gl_w=jnp.asarray(self.gl_w),
            lam_t=jnp.asarray(np.linspace(self.lnu0, -self.lnu0, self.n_nodes)),
        )
        return cfg, nums

    def layout(self, psis):
        """Return a chunk's layout on the host, raising the errors that
        `_Rows.row` raises. For each row the layout holds x_e, the index k_hi of
        the last interior grid node, the closest refined distance d1 and the
        tail's remainder ratio; the upper panel edges as distances from x_e;
        and the interior panels' x edges, each panel's owning interval and each
        panel's kind (1 where the interval lies above x_e / 2). Each count is
        padded up to a multiple of COUNT_BUCKET."""
        n = psis.size
        xe, k_hi, d1, ratio, d_his, panels = (
            np.zeros(n), np.zeros(n, np.int32), np.zeros(n), np.ones(n), [], []
        )  # fmt: skip
        a_eff = -self.floor_s if self.continues else 0.0
        for j, psi in enumerate(float(p) for p in psis):
            xe0 = self.support_edge(psi)
            below_edge = np.flatnonzero(self.x_nodes < xe0 * (1.0 - EDGE_MARGIN))
            if below_edge.size < 2:
                raise AccuracyError(
                    "The velocity support of the velocity table's row at "
                    f"Psi = {psi:.6e} "
                    f"ends at x_e = {xe0:.3e}, inside the first interval of the "
                    "velocity grid, where the lower refinement of the grid ends."
                )
            k_hi[j] = min(int(below_edge[-1]), self.x_nodes.size - 2)
            d_hi, xe[j] = self.hi_edges(psi, self.x_nodes[k_hi[j]])
            d1[j] = d_hi[-1]
            d_his.append(d_hi)
            panels.append(self.interior_edges(xe[j], int(k_hi[j])))
        if self.rule == "tail" and self.tail_log_exponent != 0.0:
            # tail_mass_below works element by element, so one call gives the
            # remainder ratio of every row of the chunk.
            e_1 = np.asarray(psis, dtype=np.float64) * d1 * (2.0 * xe - d1)
            b, e_0 = self.tail_log_exponent, self.tail_log_scale
            ratio = (
                tail_mass_below(1.0, a_eff, e_1, b, e_0)
                * (1.0 - a_eff)
                / (e_1 ** (1.0 - a_eff) * np.log(e_0 / e_1) ** (-b))
            )

        def padded(rows, size, fill, dtype=np.float64):
            out = np.full((n, size), fill, dtype=dtype)
            for j, row in enumerate(rows):
                out[j, : row.size] = row
            return out

        n_h = np.array([d.size for d in d_his], dtype=np.int32)
        n_p = max([1] + [p[0].size for p in panels])
        size_h = COUNT_BUCKET * math.ceil(int(np.max(n_h)) / COUNT_BUCKET)
        size_p = COUNT_BUCKET * math.ceil(n_p / COUNT_BUCKET)
        return dict(
            xe=xe, k_hi=k_hi, d1=d1, n_h=n_h, d_hi=padded(d_his, size_h, 1.0),
            ratio=ratio,
            i_lo=padded([p[0] for p in panels], size_p, 0.5),
            i_hi=padded([p[1] for p in panels], size_p, 0.5),
            i_kind=padded([p[2] for p in panels], size_p, 0, np.int32),
            i_own=padded([p[3] for p in panels], size_p, 0, np.int32),
            i_valid=padded(
                [np.ones(p[0].size, bool) for p in panels], size_p, False, bool
            ),
        )  # fmt: skip

    def interior_edges(self, xe, k_hi):
        """Return the panels of the interior intervals as `interior_masses`
        forms the panels: each panel's lower and upper x edges, each panel's
        kind (1 where the panel's interval lies above x_e / 2) and the index of
        the interval that owns each panel."""
        xn = self.x_nodes[1 : k_hi + 1]
        if xn.size < 2:
            return (np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0, int))
        up = xn[:-1] >= 0.5 * xe
        lo, hi, kinds, owners = [], [], [], []
        for kind in (False, True):
            idx = np.flatnonzero(up == kind)
            if idx.size == 0:
                continue
            a, b = (xe - xn[idx], xe - xn[idx + 1]) if kind else (xn[idx], xn[idx + 1])
            nsub = np.maximum(
                np.ceil(self.panels_per_decade * np.abs(np.log10(b / a))).astype(int), 1
            )
            la, lb = np.log(a), np.log(b)  # Equals math.log in interior_masses.
            owner = np.repeat(np.arange(idx.size), nsub)
            k = (
                np.arange(int(nsub.sum())) - np.repeat(np.cumsum(nsub) - nsub, nsub) + 1
            ).astype(np.float64)
            e = np.exp(k * ((lb - la) / nsub)[owner] + la[owner])
            e[np.cumsum(nsub) - 1] = b
            edges = np.concatenate([[a[0]], e])
            x_edges = (xe - edges) if kind else edges
            lo.append(x_edges[:-1])
            hi.append(x_edges[1:])
            kinds.append(np.full(owner.size, int(kind)))
            owners.append(idx[owner] + 1)
        return tuple(np.concatenate(v) for v in (lo, hi, kinds, owners))

    def build_kernels(self, psis, ws):
        """Return the output of `build`, computed by the row kernels on
        ROWS_PER_CHUNK rows at a time."""
        n = int(psis.size)
        table = np.zeros((n, 3, self.n_nodes))
        totals, edges = np.zeros(n), np.ones(n)
        residual, residual_psi = 0.0, float(psis[0])
        cfg, nums = self.kernel_inputs()
        lam_t = np.linspace(self.lnu0, -self.lnu0, self.n_nodes)
        q_t = 1.0 / (1.0 + np.exp(lam_t))
        a_eff = -self.floor_s if self.continues else 0.0
        for c0 in range(0, n, ROWS_PER_CHUNK):
            block = psis[c0 : c0 + ROWS_PER_CHUNK]
            real = block.size
            pad = np.concatenate([block, np.full(ROWS_PER_CHUNK - real, block[-1])])
            w_block = ws[c0 : c0 + ROWS_PER_CHUNK]
            w_pad = np.concatenate(
                [w_block, np.full(ROWS_PER_CHUNK - real, w_block[-1])]
            )
            lay = self.layout(pad)
            got = _rows_construct_polish(
                jnp.asarray(pad), jnp.asarray(w_pad),
                *(jnp.asarray(lay[k]) for k in (
                    "xe", "k_hi", "d1", "n_h", "d_hi", "ratio", "i_lo", "i_hi",
                    "i_kind", "i_own", "i_valid",
                )),
                self.arrays, cfg, nums,
            )  # fmt: skip
            nodes = got["nodes"]
            host = {
                k: np.asarray(got[k])
                for k in ("tot", "c_above", "n_2", "not_monotone", "s_t", "inside",
                          "best_r", "bad")
            }  # fmt: skip
            s_t = host["s_t"].copy()
            s_n, count = np.asarray(nodes[0]), np.asarray(nodes[7])
            # The refusals of the whole chunk are found at once. The first
            # failing row raises the error of the row's first failed check, the
            # error that `_Rows.row` raises for the failing row.
            few = host["n_2"][:real] < ROW_NODES_MIN
            failed = few | host["not_monotone"][:real] | (host["bad"][:real] > 0)
            if failed.any():
                j = int(np.argmax(failed))
                psi = float(pad[j])
                if few[j]:
                    raise AccuracyError(
                        f"The velocity table's row at Psi = {psi:.6e} carries no "
                        f"resolvable mass on the row's {int(host['n_2'][j])} nodes."
                    )
                if host["not_monotone"][j]:
                    raise ResultError(
                        "The cumulative distribution on the velocity table's row at "
                        f"Psi = {psi:.6e} is not monotone in (s, lambda_u)."
                    )
                raise ResultError(
                    "The residual of the exact cumulative distribution on the velocity "
                    f"table's row at Psi = {psi:.6e} is not finite at "
                    f"{int(host['bad'][j])} node(s)."
                )
            if (
                not cfg.tail_newton
                and self.tail_amplitude > 0.0
                and self.tail_log_exponent != 0.0
            ):
                for j in range(real):
                    beyond = np.flatnonzero(s_t[j] >= s_n[j][count[j] - 1])
                    if beyond.size:
                        self._place_beyond(
                            float(pad[j]), s_t[j], beyond, q_t,
                            float(host["tot"][j]), float(host["c_above"][j]),
                            float(lay["d1"][j]), float(lay["xe"][j]), a_eff,
                        )  # fmt: skip
            rows, n_good = _rows_finish(
                jnp.asarray(pad), jnp.asarray(w_pad), jnp.asarray(lay["xe"]),
                got["tot"], jnp.asarray(lay["d1"]), got["c_above"], jnp.asarray(s_t),
                nodes,
                self.arrays, cfg, nums,
            )  # fmt: skip
            rows, n_good = np.asarray(rows)[:real], np.asarray(n_good)[:real]
            thin = n_good < ROW_NODES_MIN
            failed = thin | ~(
                np.all(np.isfinite(rows), axis=(1, 2))
                & np.all(np.diff(rows[:, 0], axis=1) > 0.0, axis=1)
                & np.all(rows[:, 1] > 0.0, axis=1)
            )
            if failed.any():
                j = int(np.argmax(failed))
                psi = float(pad[j])
                if thin[j]:
                    raise ResultError(
                        "The slope across rows on the velocity table's row at "
                        f"Psi = {psi:.6e} is finite at only {int(n_good[j])} node(s)."
                    )
                raise ResultError(
                    "The values of s on the velocity table's row at "
                    f"Psi = {psi:.6e} are not all finite and increasing."
                )
            res = np.max(
                np.where(host["inside"][:real], host["best_r"][:real], 0.0), axis=1
            )
            table[c0 : c0 + real], totals[c0 : c0 + real] = rows, host["tot"][:real]
            edges[c0 : c0 + real] = lay["xe"][:real]
            j = int(np.argmax(res))  # First row with the chunk's largest residual.
            if res[j] > residual:
                residual, residual_psi = float(res[j]), float(pad[j])
        return table, totals, edges, (residual, residual_psi)

    # ---- The rows of the velocity table. ----

    def build(self, psis, ws):
        """Return the rows of the velocity table at potentials `psis` and depths
        `ws` = p_0 - Psi: the table of shape (rows, 3, nodes), the rows' total
        masses, the rows' support edges, and the largest absolute residual of
        the polish with the potential of the row where the residual occurs. The
        row kernels build the rows when KERNEL_ROWS is true (the default);
        otherwise `_Rows.row` builds the rows one at a time on the host, the
        reference."""
        if KERNEL_ROWS:
            return self.build_kernels(psis, ws)
        n = int(psis.size)
        table = np.zeros((n, 3, self.n_nodes))
        totals, edges = np.zeros(n), np.ones(n)
        residual, residual_psi = 0.0, float(psis[0])
        for j in range(n):
            psi = float(psis[j])
            row, res, tot, xe = self.row(psi, float(ws[j]))
            if not (
                np.all(np.isfinite(row))
                and np.all(np.diff(row[0]) > 0.0)
                and np.all(row[1] > 0.0)
            ):
                raise ResultError(
                    "The values of s on the velocity table's row at "
                    f"Psi = {psi:.6e} are not all finite and increasing."
                )
            table[j], totals[j], edges[j] = row, tot, xe
            if res > residual:
                residual, residual_psi = res, psi
        return table, totals, edges, (residual, residual_psi)


def _panel_count(per_decade, ratio, where):
    """Return the number of panels for a span of `ratio` at `per_decade`
    panels per decade, at least one panel. A `ratio` that is not above 1
    raises ResultError."""
    if not ratio > 1.0:
        raise ResultError(
            f"The panels of the velocity integral at the {where} span a ratio of outer "
            f"end to inner end of {ratio!r}, which is not above 1."
        )
    return int(per_decade * math.log10(ratio)) + 1


def _odd_row_error(table, dlam, lnu0, stride=1, in_velocity=False):
    r"""Return the table's own error across rows and the index of the row
    where the error is largest.

    Every interior row is predicted from the two rows `stride` rows away on
    each side by the draw's cubic Hermite interpolant across rows, and the
    prediction is compared with the row itself. The miss is measured in the
    cumulative probability u, or relative to the scaled velocity x when
    `in_velocity` is true. Only table nodes with :math:`|\lambda_u| \le 53 \ln 2` count,
    the cumulative probabilities that a uniform double reaches."""
    n_nodes = table.shape[2]
    lam_t = np.linspace(lnu0, -lnu0, n_nodes)
    reach = np.abs(lam_t) <= 53.0 * math.log(2.0)
    s, ms, rs = table[:, 0, :], table[:, 1, :], table[:, 2, :]
    k2 = 2 * int(stride)
    pred = 0.5 * (s[:-k2] + s[k2:]) + (k2 * dlam / 8.0) * (rs[:-k2] - rs[k2:])
    if in_velocity:
        miss = np.abs(pred - s[stride:-stride]) / (1.0 + np.exp(s[stride:-stride]))
    else:
        w = 1.0 / ((1.0 + np.exp(-lam_t)) * (1.0 + np.exp(lam_t)))
        miss = np.abs(pred - s[stride:-stride]) * w[None, :] / ms[stride:-stride]
    by_row = np.max(np.where(reach[None, :], miss, 0.0), axis=1)
    worst = int(np.argmax(by_row))
    return float(by_row[worst]), worst + int(stride)


def build_velocity_table(model, settings):
    """Return the arrays of the velocity table of a one-component model.

    Returns
    -------
    fields : dict
        The fields of `VelocityTable`, as arrays of one component. The field
        `validation` holds the two checks listed in `VELOCITY_TABLE_KEYS`, each
        located at the potential Psi of the check's row: "row_error", the
        table's own error across rows where the refinement stopped (at
        ``sampling["row_error_goal"]`` or at ``sampling["n_rows_max"]``), and
        "polish_residual", the largest absolute residual in lambda_u that the
        polish leaves at a table node.

    Raises
    ------
    AccuracyError
        If either check is above :math:`10^{-2}` and the settings do not set
        ``write_anyway=True``.
    AccuracyError
        If more than :math:`10^{-6}` of the model's mass lies at potentials
        below the lowest row.
    AccuracyError
        If a row's velocity support ends inside the velocity grid's first
        interval, or a row keeps fewer than ROW_NODES_MIN construction nodes.
    ResultError
        If the profile holds fewer than two potentials between 0 and p_0, or
        a row fails its own consistency checks.
    UnphysicalModelError
        If the DF below its table has a mass that is not finite at E = 0.
    """
    sampling = settings.sampling
    rows = _Rows(model, settings)
    p_0 = rows.p_0
    psi_profile = np.asarray(model.profile.p_trunc, dtype=np.float64)
    psi_pos = psi_profile[(psi_profile > 0.0) & (psi_profile < p_0)]
    if psi_pos.size < 2:
        raise ResultError(
            "The rows of the velocity table span the potentials strictly between 0 "
            f"and p_0 = {p_0!r}; the profile holds {psi_pos.size} potential(s) in "
            "that interval, and the velocity table needs at least 2."
        )
    if rows.continues or rows.n_below > 0:
        psi_lo = float(np.min(psi_pos))
    else:
        psi_lo = max(
            float(np.min(psi_pos)), rows.q_min_written / math.sqrt(TIER_BOUNDS[0])
        )
    mass = np.asarray(model.profile.mass_trunc, dtype=np.float64)
    below_rows = (psi_profile > 0.0) & (psi_profile < psi_lo)
    mass_beyond = float(
        1.0
        - (np.min(mass[below_rows]) if below_rows.any() else mass.max()) / mass.max()
    )
    if mass_beyond > TIER_BOUNDS[0]:
        raise AccuracyError(
            f"A fraction {mass_beyond:.3e} of the model's mass lies where the "
            "relative potential Psi is below the potential of the velocity table's "
            f"lowest row, {psi_lo:.4e}; the fraction must be at most 0.000001."
        )
    psi_hi = float(np.max(psi_pos))
    # The top row's depth comes from the profile's column u_trunc, which holds
    # p_0 - Psi exactly. The difference p_0 - Psi formed by subtraction loses
    # its digits near the top: on a cored model the largest potential below p_0
    # lies one unit in the last place below p_0. Where u_trunc does not cover
    # the top row, the depth is p_0 - Psi.
    u_col = np.asarray(model.profile.u_trunc, dtype=np.float64)
    i_hi = int(np.flatnonzero(psi_profile == psi_hi)[0])
    u_hi = (
        float(u_col[i_hi]) if i_hi < u_col.size and u_col[i_hi] > 0.0 else p_0 - psi_hi
    )
    lam_lo = float(np.log(psi_lo / (p_0 - psi_lo)))
    lam_hi = float(np.log(psi_hi / u_hi))
    n_rows = int(sampling["n_rows"])
    dlam = float((lam_hi - lam_lo) / (n_rows - 3))
    lam_rows = (lam_lo - dlam) + dlam * np.arange(n_rows)
    # Each row's Psi and depth p_0 - Psi both come from lambda_p, because the
    # depth formed by subtraction near the top keeps only the depth's last
    # bits.
    psi_rows = p_0 / (1.0 + np.exp(-lam_rows))
    w_rows = p_0 / (1.0 + np.exp(lam_rows))
    table, totals, edges, residual = rows.build(psi_rows, w_rows)
    previous = math.nan
    while True:
        row_error = max(
            _odd_row_error(table, dlam, rows.lnu0),
            _odd_row_error(table, dlam, rows.lnu0, in_velocity=True),
        )
        goal = float(sampling["row_error_goal"])
        if row_error[0] <= goal or 2 * n_rows - 3 > int(sampling["n_rows_max"]):
            break
        previous = row_error[0]
        n_new = 2 * n_rows - 3
        dlam = float((lam_hi - lam_lo) / (n_new - 3))
        lam_new = (lam_lo - dlam) + dlam * np.arange(n_new)
        psi_new = p_0 / (1.0 + np.exp(-lam_new))
        w_new = p_0 / (1.0 + np.exp(lam_new))
        keep = 2 * np.arange(1, n_rows - 1) - 1
        fresh = np.setdiff1d(np.arange(n_new), keep)
        t_f, tot_f, xe_f, res_f = rows.build(psi_new[fresh], w_new[fresh])
        table_new = np.zeros((n_new, 3, rows.n_nodes))
        totals_new, edges_new = np.zeros(n_new), np.ones(n_new)
        table_new[keep], totals_new[keep], edges_new[keep] = (
            table[1:-1],
            totals[1:-1],
            edges[1:-1],
        )
        table_new[fresh], totals_new[fresh], edges_new[fresh] = t_f, tot_f, xe_f
        table, totals, edges = table_new, totals_new, edges_new
        residual = max(residual, res_f)
        n_rows, lam_rows, psi_rows = n_new, lam_new, psi_new
    checks = {
        "row_error": Check(error=row_error[0], where=float(psi_rows[row_error[1]])),
        "polish_residual": Check(error=residual[0], where=residual[1]),
    }
    for key, check in checks.items():
        # A check above 0.01 refuses the table unless the settings set
        # write_anyway. With write_anyway, the table's validation report carries
        # the tier "refused" and the sampler announces the tier.
        if check.level == 3 and not settings.write_anyway:
            # The message states the observed error and, for an Osipkov-Merritt
            # or Cuddeford-Osipkov-Merritt (COM) Type I model, the anisotropy's
            # inputs, because the Notes section of `Sampler` describes the rows
            # of Osipkov-Merritt and COM Type I models reaching n_rows_max.
            capped = key == "row_error" and 2 * n_rows - 3 > int(sampling["n_rows_max"])
            anisotropy = model.components[0].anisotropy
            raise AccuracyError(
                f"The velocity table's {key.replace('_', ' ')} is {check.error:.2e} at "
                f"Psi = {check.where:.6e}, above 0.01"
                + (
                    f", and the table's rows reached n_rows_max ({n_rows}) before "
                    "the table's error across rows reached sampling.row_error_goal"
                    if capped
                    else ""
                )
                + (
                    "; the model's anisotropy is "
                    f"{anisotropy.family.replace('_', ' Type ')} with "
                    f"beta_c = {anisotropy.beta_c:g} and r_a = {anisotropy.r_a:g} (the "
                    "Notes of Sampler describe the rows of such models)."
                    if capped and anisotropy.family in ("OM_I", "COM_I")
                    else "."
                )
            )
    # Where the rows stopped at n_rows_max with the error across rows above
    # row_error_goal, the cause is not a check, because the validation report
    # holds only checks. The table's log line and the sampler's announcement
    # state the cause.
    row_cap = (
        f"the velocity table's rows reached n_rows_max ({n_rows})"
        + (
            " with the table's error across rows still falling, by a factor of "
            f"{previous / row_error[0]:.1f} over the last doubling of the rows"
            if math.isfinite(previous)
            else ""
        )
        if row_error[0] > goal
        else ""
    )
    if row_cap:
        _log.info(
            "The velocity table's error across rows is %.2e: %s.", row_error[0], row_cap
        )
    return dict(
        lambda_p=lam_rows,
        dlambda_p=dlam,
        lambda_u=np.linspace(rows.lnu0, -rows.lnu0, rows.n_nodes),
        s=table[None, :, 0],
        ds_dlambda_u=table[None, :, 1],
        ds_dlambda_p=table[None, :, 2],
        x_edge=edges[None, :],
        p_0=p_0,
        e_edge=(0.0 if rows.continues else rows.q_min,),
        validation=ValidationReport(keys=VELOCITY_TABLE_KEYS, checks=checks),
        row_cap=row_cap,
    )
