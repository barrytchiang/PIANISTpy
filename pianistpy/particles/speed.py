"""The speed table: the inverse distribution of the scaled speed, by potential.

The density of the scaled speed x on the row of potential Psi is

    ``p(x | Psi) ~ x^(2 - 2 beta_c) F(Psi (1 - x^2))``,

a function of Psi alone, so one table serves every radius (Section T:5.2.1
[speed-table]). Its rows are uniform in lambda_p = ln(Psi / (p_0 -
Psi)) over the profile's range of Psi, with one guard row beyond each end;
on each row the inverse s(lambda_u), s = ln(x / (x_e - x)) and lambda_u =
ln(u / (1 - u)), is tabulated at nodes uniform in lambda_u with its exact
slope, and the slope across rows at fixed u is a third channel. The rows are
refined (n to 2 n - 3, every row kept) until the table's own error across rows
is under the goal.

The construction of a row: the mass of the density on the panels of the
velocity grid, refined at both ends by Gauss-Legendre panels in the logarithm
of x and of x_e - x, with the closed forms beyond; the cumulative masses from
each end at the panels' edges are the construction nodes; their cubic Hermite
is inverted at the table's nodes, and each node is then polished by a
safeguarded Newton on the true residual.
"""

__all__ = ["SPEED_TABLE_KEYS", "build_speed_table"]

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
from ..validation import GOAL, HARD_LIMIT, Check, ValidationReport

# A node of a row is polished until its residual in lambda_u is under this,
# its step is below the spacing of doubles, or two passes fail to improve it.
NODE_TOLERANCE = 1e-13
# Most passes of the polish of a row: a guard. With the lookup asked by the
# depth near the top, the 33 test models take at most 5.
POLISH_PASSES_MAX = 60
# Most iterations, and the tolerance, of the Newton that places a node beyond
# the last construction node on a tail with a logarithmic factor. On the two
# members of the subset with such a tail, 1 or 2 points of some rows reach the
# 80 iterations at |g| up to 1.8e-11: the resolution of the tail's mass there
# (scaled_upper_gamma at the tail's order s = 1 - b near -0.01, below an
# argument of 30, reads 1e-11 relative), which 1e-12 lies under. In the
# cumulative probability that is 2e-11.
TAIL_NEWTON_ITERATIONS = 80
TAIL_NEWTON_TOLERANCE = 1e-12
# The upper refinement starts at the last interior node this far, relative,
# below the edge of the support. At 1e-3 or 1e-5 instead, the tables of the 33
# members move by at most 1.5e-7 in s, inside their own error across rows.
EDGE_MARGIN = 1e-4
# Fewest nodes a row keeps, and fewest finite slopes across rows, below which
# the row is refused: the cubic Hermite and its inversion need a stencil. The
# 33 members keep at least 309 nodes a row and all 2048 slopes.
ROW_NODES_MIN = 8
# Floor of an argument of a logarithm.
LOG_ARGUMENT_MIN = 1e-300
# The checks of a speed table: the table's own error across rows and the
# largest residual its polish leaves at a node, both in the cumulative
# probability.
SPEED_TABLE_KEYS = ("row_error", "polish_residual")

_log = logging.getLogger("pianistpy")


# ---- the kernels: module functions, the tables and the model's numbers traced,
# so that every model of the same shapes and kinds shares one compilation ----


@functools.partial(jax.jit, static_argnums=(3,))
def _log10_df_kernel(e, u, arrays, kinds):
    """log10 F at energies of any shape, asked by the depth `u` where it is
    finite (`evaluate_lookup`'s second entry)."""
    return evaluate_lookup(arrays, kinds, e, xp=jnp, u=u)


@functools.partial(jax.jit, static_argnums=(3,))
def _kappa_kernel(e, u, arrays, kinds):
    """d ln F / d ln K at energies of shape (n,), by the depth `u` where it
    is finite (`_kappa_at`)."""
    return _kappa_at(e, u, arrays, kinds)


def _kappa_at(e, u, arrays, kinds):
    """d ln F / d ln K at energies of shape (n,): where the depth u = p_0 - K
    is held (finite), -K d ln F / d u through the lookup's depth entry, since
    K formed near p_0 moves in steps of its last bit; elsewhere by K. At the
    table's first energy K_0 itself, where the DF continues below it, the
    rule below the table's: the lookup reads K_0 from the table, whose
    derivative there is one-sided, while the rule's is exact (the bottom row
    of a table whose lowest potential is K_0 reads it at every node)."""
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
    """A kernel at host energies (and depths, where given), padded to a power
    of two (one compilation per bucket), on the host."""
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


# ---- the rows in kernels: a chunk of rows at once, every row's construction,
# polish and slopes on fixed shapes (the same operations as `_Rows.row`) ----

# Rows given to the row kernels at once, and the multiple the chunk's padded
# counts are rounded up to: one compiled shape per bucket.
ROWS_PER_CHUNK = 64
COUNT_BUCKET = 64
# The rows are built by the kernels (True) or by `_Rows.row` on the host, one
# row at a time (False): the reference the kernels are held to.
KERNEL_ROWS = True


class _RowConfig(NamedTuple):
    """The static part of the row kernels: the lookup's kinds and the rule's
    branches."""

    kinds: LookupKinds
    continues: bool
    rule: str
    tail_newton: bool


class _RowNumbers(NamedTuple):
    """The model's numbers the row kernels read, traced."""

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
    """`_Rows.logpdf` in a kernel: E from d where d is finite, F asked by the
    depth on the side formed from x near the top, w the row's depth p_0 - Psi."""
    e = jnp.maximum(psi * (1.0 - x * x), 0.0)
    fd = jnp.isfinite(d)
    dd = jnp.where(fd, d, 0.0)
    e = jnp.where(fd, nums.e_edge + psi * dd * (2.0 * xe - dd), e)
    u = jnp.where(~fd & (e >= 0.5 * nums.p_0), w + psi * x * x, jnp.nan)
    return evaluate_lookup(arrays, cfg.kinds, e, xp=jnp, u=u) + nums.velocity_power * (
        jnp.log10(jnp.maximum(x, LOG_ARGUMENT_MIN))
    )


def _k_kappa(e, u, arrays, cfg):
    """d ln F / d ln K of the lookup at energies of any shape, by the depth u
    where it is finite."""
    return _kappa_at(e.ravel(), u.ravel(), arrays, cfg.kinds).reshape(e.shape)


def _k_gl_mass(psi, w, xe, s0, s1, valid, by_d, arrays, cfg, nums):
    """GL masses of panels between s0 and s1 in ln x, or in ln d = ln(x_e - x)
    with by_d; the group shifted by its own largest term, as one host call."""
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
    """The entries under mask moved to the front in order; -> (count, arrays)."""
    n = mask.size
    order = jnp.argsort(jnp.where(mask, jnp.arange(n), n + jnp.arange(n)))
    return jnp.sum(mask).astype(jnp.int32), [a[order] for a in arrays]


def _k_floor_mass(e, arrays, cfg):
    """`below_floor_mass` for the Taylor form and the edge law in a kernel."""
    params = arrays.params
    if cfg.rule == "taylor":
        return params[FLOOR_0] * e * e / 2.0 + params[FLOOR_1] * e * e * e / 6.0
    p, c_0, c_1 = params[FLOOR_0], params[FLOOR_0 + 1], params[FLOOR_0 + 2]
    return c_0 * e ** (p + 1.0) / (p + 1.0) + c_1 * e ** (p + 2.0) / (p + 2.0)


def _k_hermite(xq, xn, fn, mn, n):
    """`hermite` on one row whose nodes are padded beyond n: value, derivative."""
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
    """`hermite_inverse` on one row padded beyond n: the bracketed bisection,
    60 halvings, then two Newton steps kept inside the bracket."""
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
    """`_Rows.pair_masses` in a kernel: each pair's panel shifted by its own
    largest term (the polish), or the row's (the slopes, with the weight
    kappa(E))."""
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
    """`_Rows._place_beyond` in a kernel, every node of the row under a mask."""
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
    """`_Rows.row` up to the polish, on one row of the layout: the construction
    nodes padded beyond their count, the row's total and remainder, and the
    refusal flags (too few nodes, not monotone)."""
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
    # the interior nodes x_nodes[2 .. k_hi], the upper edges 1 .. n_h - 1
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
    """The polish's start on one row: the Hermite inversion and the state."""
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
    """One pass of `_Rows.row`'s safeguarded Newton on one row: every node
    evaluated, the live ones updated."""
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
    """A chunk's rows: the construction of each and the polish of all of them,
    the passes running while any node of the chunk moves."""
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
    """`_Rows.row` after the polish on one row: the nodes beyond the last
    construction node on a logarithmic tail, the slope along the row and the
    across-row slope. -> (stack[s, ms, rs], the count of finite slopes)."""
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
    """A chunk's rows after the polish (`_k_finish_row` on each)."""
    return jax.vmap(
        functools.partial(_k_finish_row, arrays=arrays, cfg=cfg, nums=nums)
    )(psi, w, xe, tot, d1, c_above, s_t, nodes)


class _Rows:
    """Everything a row of the speed table needs, fixed for a model."""

    def __init__(self, model, settings) -> None:
        sampling = settings.sampling
        component = 0
        anisotropy = model.components[component].anisotropy
        form = model.form
        self.p_0 = float(form.p_0)
        # the DF is the package's lookup; its arrays are traced arguments of
        # every kernel, held once on the device and once on the host
        self.host_arrays, self.kinds = prepare_lookup(model.table, form, component)
        self.arrays = LookupArrays(*(jnp.asarray(a) for a in self.host_arrays))
        # the DF's first energy, the nodes below the written table included
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
        # the DF's power as K -> 0 below its table: the contract's for a tail and
        # an edge law; for "taylor", F_1 K + F_2 K^2 / 2, the power of its
        # leading term, 1 where F_1 is not zero and 2 where it is (the
        # stationary edge); none for "zero", where the support ends at the
        # table's first energy
        if rule == "taylor":
            floor_s = 1.0 if form.below_floor_coefficients[component][0] != 0.0 else 2.0
        else:
            floor_s = form.exponent_at_zero(component)
        self.floor_s = floor_s
        self.continues = rule != "zero"
        if self.continues and not floor_s > -1.0:
            raise UnphysicalModelError(
                f"Below its table the DF goes as K^{floor_s:g} ({rule!r}): its "
                "mass is not finite at K = 0."
            )
        beta_eff = float(anisotropy.beta_c)
        self.velocity_power = 2.0 - 2.0 * beta_eff
        self.p_map = 1.0 + floor_s if (self.continues and floor_s < 0.0) else 1.0
        self.n_intervals = int(sampling["n_speed_intervals"])
        self.n_nodes = int(sampling["n_nodes"])
        self.lnu0 = math.log(sampling["tail_mass_min"])
        self.panels_per_decade = int(sampling["panels_per_decade"])
        self.edge_refined = float(sampling["edge_refined"])
        self.x_nodes = (
            np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, self.n_intervals + 1)) ** 1.5
        )
        low = float(sampling["speed_low_refined"])
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
        """log10 F at host energies, by the depth `u` where it is finite."""
        if u is None:
            u = np.full(np.shape(e), np.nan)
        return _on_bucket(_log10_df_kernel, e, self.arrays, self.kinds, u)

    def kappa(self, e, u=None):
        """d ln F / d ln K at host energies, by the depth `u` where it is finite."""
        if u is None:
            u = np.full(np.shape(e), np.nan)
        return _on_bucket(_kappa_kernel, e, self.arrays, self.kinds, u)

    # ---- the density on the host ----

    def support_edge(self, psi):
        """x_e: 1 where F continues below its table, else where E reaches its floor."""
        return 1.0 if self.continues else math.sqrt(max(1.0 - self.q_min / psi, 0.0))

    def hi_edges(self, psi, x_from):
        """Distances from the edge, decreasing, of the upper refinement's panels."""
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
        """E = Psi (1 - x^2), formed from the distance d = x_e - x where it is given."""
        e = np.maximum(psi * (1.0 - x * x), 0.0)
        if d is not None:
            d = np.asarray(d, dtype=np.float64)
            e_edge = 0.0 if self.continues else float(self.q_min)
            fin = np.isfinite(d)
            dd = np.where(fin, d, 0.0)
            e = np.where(fin, e_edge + psi * dd * (2.0 * xe - dd), e)
        return e

    def depth(self, psi, w, x, d, e):
        r"""The depth u = p_0 - E where it is held exactly, NaN elsewhere: on the
        side formed from x (no finite d) with ``E >= p_0 / 2``, u = w + Psi x^2,
        both terms exact (``Psi >= E >= p_0 / 2``), w = p_0 - Psi the row's depth
        as `build_speed_table` gives it. E formed as
        Psi (1 - x^2) loses it there: on the top rows its last bit is :math:`10^{-10}`
        of u, so F would move in steps of E rather than with x, and the polish's
        Newton slope would be off by a factor near 2."""
        x = np.asarray(x, dtype=np.float64)
        by_x = (
            np.ones(np.shape(e), dtype=bool)
            if d is None
            else ~np.isfinite(np.asarray(d, dtype=np.float64))
        )
        with np.errstate(invalid="ignore"):
            return np.where(by_x & (e >= 0.5 * self.p_0), w + psi * x * x, np.nan)

    def logpdf(self, psi, w, x, d=None, xe=None):
        """log10 of ``x^(2 - 2 beta_c) F(E)`` at host points, F asked by the depth
        where it is held."""
        e = self.energy(psi, x, d, xe)
        return self.log10_df(e, self.depth(psi, w, x, d, e)) + self.velocity_power * (
            np.log10(np.maximum(x, 1e-300))
        )

    def panel_masses_d(self, psi, w, d_edges, xe):
        """Mass on each panel between edges given as distances from the edge."""
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
        """Mass on each panel between edges: GL in ln x, or in ln(x_e - x) with xe."""
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
        """Mass on each interior interval of the velocity grid, in sub-panels."""
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
        """Signed mass from xa to xq, one GL panel per pair; with kappa_w = c the
        integrand carries the weight c + kappa(E)."""
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
        """The integral of F from 0 to e under the rule below the table."""
        return below_floor_mass(self.host_arrays, self.kinds, e)

    # ---- one row ----

    def row(self, psi, w):
        """One row, at Psi and its depth w = p_0 - Psi: (s, d s / d lambda_u,
        d s / d lambda_p) at the nodes, the residual of the polish, the row's
        total, x_e."""
        vp = self.velocity_power
        x_nodes = self.x_nodes
        xe0 = self.support_edge(psi)
        below_edge = np.flatnonzero(x_nodes < xe0 * (1.0 - EDGE_MARGIN))
        if below_edge.size < 2:
            raise AccuracyError(
                f"The row at Psi = {psi:.6e}: its support ends at x_e = {xe0:.3e}, "
                "inside the velocity grid's first interval, where the lower "
                "refinement ends."
            )
        k_hi = int(below_edge[-1])
        k_hi = min(k_hi, x_nodes.size - 2)
        sub_hi_d, xe = self.hi_edges(psi, x_nodes[k_hi])
        # the masses of the grid's interior intervals, x_nodes[1] to x_nodes[k_hi]
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
                f"The row at Psi = {psi:.6e} carries no resolvable mass "
                f"({x_all.size} nodes)."
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
                f"The row at Psi = {psi:.6e}: the CDF is not monotone in (s, lambda_u)."
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
                    f"The row at Psi = {psi:.6e}: the true CDF residual is not finite "
                    f"at {int(np.sum(~np.isfinite(r)))} node(s)."
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
        """Nodes beyond the last construction node, on the tail with its logarithm."""
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
        """d s / d lambda_p at fixed u, from the density's own derivative in Psi."""
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
                f"The row at Psi = {psi:.6e}: the across-row slope is finite at "
                f"{int(good.sum())} node(s) only."
            )
        return np.where(
            good,
            rs_t,
            np.interp(np.arange(rs_t.size), np.flatnonzero(good), rs_t[good]),
        )

    # ---- the rows in kernels ----

    def kernel_inputs(self):
        """The row kernels' static config and traced numbers for this model."""
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
        """A chunk's layout on the host, with `_Rows.row`'s refusals: per row
        x_e, k_hi, d1 and the tail's remainder ratio; the upper edges in d; the
        interior panels' x edges, their interval and kind (1 above x_e / 2);
        the counts padded to multiples of COUNT_BUCKET."""
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
                    f"The row at Psi = {psi:.6e}: its support ends at x_e = {xe0:.3e}, "
                    "inside the velocity grid's first interval, where the lower "
                    "refinement ends."
                )
            k_hi[j] = min(int(below_edge[-1]), self.x_nodes.size - 2)
            d_hi, xe[j] = self.hi_edges(psi, self.x_nodes[k_hi[j]])
            d1[j] = d_hi[-1]
            d_his.append(d_hi)
            panels.append(self.interior_edges(xe[j], int(k_hi[j])))
        if self.rule == "tail" and self.tail_log_exponent != 0.0:
            # the chunk's rows in one call: the function works element by element
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
        size_h = COUNT_BUCKET * math.ceil(int(n_h.max()) / COUNT_BUCKET)
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
        """The interior intervals' panels as `interior_masses` forms them: the x
        edges of each panel, its kind (1 where its interval lies above x_e / 2)
        and the interval that owns it."""
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
            la, lb = np.log(a), np.log(b)  # libm's log, as math.log
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
        """`build` in the row kernels, ROWS_PER_CHUNK rows at a time."""
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
            # the chunk's refusals at once; the first row that fails names its
            # first failed check, as row by row
            few = host["n_2"][:real] < ROW_NODES_MIN
            failed = few | host["not_monotone"][:real] | (host["bad"][:real] > 0)
            if failed.any():
                j = int(np.argmax(failed))
                psi = float(pad[j])
                if few[j]:
                    raise AccuracyError(
                        f"The row at Psi = {psi:.6e} carries no resolvable mass "
                        f"({int(host['n_2'][j])} nodes)."
                    )
                if host["not_monotone"][j]:
                    raise ResultError(
                        f"The row at Psi = {psi:.6e}: the CDF is not monotone in "
                        "(s, lambda_u)."
                    )
                raise ResultError(
                    f"The row at Psi = {psi:.6e}: the true CDF residual is not "
                    f"finite at {int(host['bad'][j])} node(s)."
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
                        f"The row at Psi = {psi:.6e}: the across-row slope is finite "
                        f"at {int(n_good[j])} node(s) only."
                    )
                raise ResultError(
                    f"The row at Psi = {psi:.6e} is not finite and increasing."
                )
            res = np.max(
                np.where(host["inside"][:real], host["best_r"][:real], 0.0), axis=1
            )
            table[c0 : c0 + real], totals[c0 : c0 + real] = rows, host["tot"][:real]
            edges[c0 : c0 + real] = lay["xe"][:real]
            j = int(np.argmax(res))  # the first row of the chunk's largest
            if res[j] > residual:
                residual, residual_psi = float(res[j]), float(pad[j])
        return table, totals, edges, (residual, residual_psi)

    # ---- the rows ----

    def build(self, psis, ws):
        """The rows at the given Psi and depths w = p_0 - Psi: the table (n, 3,
        nodes), the totals, the edges, and the largest residual of the polish
        with its row's Psi. By the row kernels where KERNEL_ROWS (the default),
        else row by row on the host, the reference."""
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
                    f"The row at Psi = {psi:.6e} is not finite and increasing."
                )
            table[j], totals[j], edges[j] = row, tot, xe
            if res > residual:
                residual, residual_psi = res, psi
        return table, totals, edges, (residual, residual_psi)


def _panel_count(per_decade, ratio, where):
    """Panels for a span of `ratio` at `per_decade` panels a decade, at least
    one; a span that is not a positive ratio above 1 is refused."""
    if not ratio > 1.0:
        raise ResultError(
            f"The speed integral's panels at the {where} span a ratio of {ratio!r}, "
            "not above 1."
        )
    return int(per_decade * math.log10(ratio)) + 1


def _odd_row_error(table, dlam, lnu0, stride=1, in_speed=False):
    """Every interior row predicted from its neighbors at twice the spacing by
    the draw's interpolant, against the row itself: the largest miss in u (or
    in x / x), over the nodes a uniform double can address, and its row."""
    n_nodes = table.shape[2]
    lam_t = np.linspace(lnu0, -lnu0, n_nodes)
    reach = np.abs(lam_t) <= 53.0 * math.log(2.0)
    s, ms, rs = table[:, 0, :], table[:, 1, :], table[:, 2, :]
    k2 = 2 * int(stride)
    pred = 0.5 * (s[:-k2] + s[k2:]) + (k2 * dlam / 8.0) * (rs[:-k2] - rs[k2:])
    if in_speed:
        miss = np.abs(pred - s[stride:-stride]) / (1.0 + np.exp(s[stride:-stride]))
    else:
        w = 1.0 / ((1.0 + np.exp(-lam_t)) * (1.0 + np.exp(lam_t)))
        miss = np.abs(pred - s[stride:-stride]) * w[None, :] / ms[stride:-stride]
    by_row = np.max(np.where(reach[None, :], miss, 0.0), axis=1)
    worst = int(np.argmax(by_row))
    return float(by_row[worst]), worst + int(stride)


def build_speed_table(model, settings):
    """Return the arrays of the speed table of a one-component model.

    Returns
    -------
    fields : dict
        The fields of `SpeedTable`, as arrays of one component. Its
        `validation` holds the checks `SPEED_TABLE_KEYS`, each at its row's
        Psi: the table's own error across rows (where the refinement stopped,
        at the goal or at ``sampling.n_rows_max``) and the polish's largest
        residual.

    Raises
    ------
    AccuracyError
        If either check lies beyond the hard limit.
    """
    sampling = settings.sampling
    rows = _Rows(model, settings)
    p_0 = rows.p_0
    psi_profile = np.asarray(model.profile.p_trunc, dtype=np.float64)
    psi_pos = psi_profile[(psi_profile > 0.0) & (psi_profile < p_0)]
    if psi_pos.size < 2:
        raise ResultError(
            f"The profile holds {psi_pos.size} potential(s) inside (0, p_0 = {p_0!r}): "
            "the speed table's rows span that range."
        )
    if rows.continues or rows.n_below > 0:
        psi_lo = float(np.min(psi_pos))
    else:
        psi_lo = max(float(np.min(psi_pos)), rows.q_min_written / math.sqrt(GOAL))
    mass = np.asarray(model.profile.mass_trunc, dtype=np.float64)
    below_rows = (psi_profile > 0.0) & (psi_profile < psi_lo)
    mass_beyond = float(
        1.0
        - (np.min(mass[below_rows]) if below_rows.any() else mass.max()) / mass.max()
    )
    if mass_beyond > GOAL:
        raise AccuracyError(
            f"{mass_beyond:.3e} of the model's mass lies where Psi is below the "
            f"lowest row ({psi_lo:.4e}), above the goal {GOAL:g}."
        )
    psi_hi = float(np.max(psi_pos))
    # the top row at the depth of its potential in the solve's own column
    # u_trunc, exact where p_0 - Psi has lost its digits (on a cored model the
    # largest potential below p_0 is an ulp under it); p_0 - Psi where the
    # column does not cover the row
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
    # each row's Psi and its depth p_0 - Psi, both from lambda_p: the depth is
    # not formed by subtraction, which near the top leaves only its last bits
    psi_rows = p_0 / (1.0 + np.exp(-lam_rows))
    w_rows = p_0 / (1.0 + np.exp(lam_rows))
    table, totals, edges, residual = rows.build(psi_rows, w_rows)
    previous = math.nan
    while True:
        row_error = max(
            _odd_row_error(table, dlam, rows.lnu0),
            _odd_row_error(table, dlam, rows.lnu0, in_speed=True),
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
        # beyond the hard limit the table is refused, unless the settings ask
        # for it anyway: then its report carries the verdict (tier "refused")
        # and the sampler announces it
        if check.level == 3 and not settings.write_anyway:
            # what was observed, and for an Osipkov-Merritt or COM Type I
            # model its inputs (the family whose rows the Sampler's notes find
            # reaching the cap)
            capped = key == "row_error" and 2 * n_rows - 3 > int(sampling["n_rows_max"])
            anisotropy = model.components[0].anisotropy
            raise AccuracyError(
                f"The speed table's {key.replace('_', ' ')} is {check.error:.2e} at "
                f"Psi = {check.where:.6e}, beyond the hard limit {HARD_LIMIT:g}"
                + (
                    f", with its rows at n_rows_max ({n_rows}) before the error "
                    "across them reached the goal"
                    if capped
                    else ""
                )
                + (
                    f", for this {anisotropy.family.replace('_', ' Type ')} model "
                    f"(beta_c = {anisotropy.beta_c:g}, r_a = {anisotropy.r_a:g}; see "
                    "Sampler)."
                    if capped and anisotropy.family in ("OM_I", "COM_I")
                    else "."
                )
            )
    # the cause where the rows stopped at the cap above the goal: not a check
    # (the record holds the judgments), the table's log line and the
    # sampler's announcement say it
    row_cap = (
        f"its rows reached n_rows_max ({n_rows})"
        + (
            f" with the error across them still falling, "
            f"{previous / row_error[0]:.1f} times over the last doubling of the rows"
            if math.isfinite(previous)
            else ""
        )
        if row_error[0] > goal
        else ""
    )
    if row_cap:
        _log.info("The speed table's row error is %.2e: %s.", row_error[0], row_cap)
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
        validation=ValidationReport(keys=SPEED_TABLE_KEYS, checks=checks),
        row_cap=row_cap,
    )
