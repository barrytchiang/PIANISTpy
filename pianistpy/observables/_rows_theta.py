r"""The explicit part of a LOSVD row's derivative in the anisotropy: at a
fixed DF table, the rows held at their own rule.

A row's value at its speed nodes is the integral, along each line of sight's
nodes in t, of the velocity plane's integral (`EllipseRoute.plane`): of F
times the L factor :math:`(L^2)^{-\beta_c}` over (s, phi) where the model has
the factor, of :math:`2\pi\,G(Q_c)` where it has none. At a fixed table the
anisotropy moves the plane's integrand through q(r), and with it a, Q_c, the
Jacobian, the split at the L = 0 point and the s nodes, and through
:math:`\beta_c` in the prefactor, the s map toward the split and the angle
integral. The rule's decisions are held: the supports and the L = 0 splits
in t, each node's level, the layers of each group of pairs, which side of
the split each pair takes and each node's angle count. The derivative is
JAX's on a twin of the plane at those decisions (`_plane_twin`), F read by
the traced lookup and G by its twin, the angle integral by the rows' own
kernel (`_angle_rule._kernel`) in its general power form; the twin is held
to the rows the rule computes before its derivative is taken.

Where the support's edge moves and F is not zero there, the derivative has a
term at the edge that the held supports do not carry: a DF that ends at its
floor with a positive value (the rule ``"zero"``) and one that diverges at
zero energy (``"tail"``) are refused by name.
"""

__all__ = ["losvd_rows_explicit"]

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..df import Model
from ..df.lookup import K_0, LookupArrays, evaluate_lookup, prepare_lookup
from ..exceptions import InputError, ResultError
from ..numerics.quadrature import gauss_legendre_unit
from ..validation import GOAL
from ._angle_rule import _PAD, _kernel
from ._ellipse_rules import gl01, seg_unit
from ._table_vjp import (
    DF_PARAMETERS,
    _g_twin,
    _lookup_parts,
    _row_by_levels,
    _row_levels,
)

# The refusal of an isotropic model's beta_c by the rows (`DF_PARAMETERS`).
ISOTROPIC_ROWS = (
    "An isotropic model's rows read G, with no L factor: their derivative in "
    "beta_c needs the factor's integral at beta_c = 0, which no row computes."
)
# The relative agreement asked of the twin with the rows the rule computes.
_TWIN_TOLERANCE = 1e-12
# Pairs of one group taken at once under JAX, which bounds the memory.
_PAIR_CHUNK = 32
# The least padded count of a group's s-nodes.
_S_BUCKET = 64


def _refuse(model: Model, route_of: Callable[[], Any], names: tuple[str, ...]) -> Any:
    """Refuse, by name, what the twin does not carry; return the route
    (`route_of`), built once the model's own refusals are passed."""
    family = model.components[0].anisotropy.family
    if family == "COM_II":
        raise InputError(
            "The DF path takes no derivative in the anisotropy of a COM Type II "
            "model: the DF table's rule refuses it."
        )
    if family == "isotropic" and "beta_c" in names:
        raise InputError(ISOTROPIC_ROWS)
    allowed = DF_PARAMETERS[family]
    for name in names:
        if name not in allowed:
            raise InputError(
                f"A {family} model's rows take derivatives in {list(allowed)}; "
                f"got {name!r}."
            )
    rule = model.form.below_floor[0]
    if rule == "tail":
        raise InputError(
            "A DF that diverges at zero energy (the rule 'tail') moves the rows' "
            "support edge where F is not zero: its edge term is not carried."
        )
    route = route_of()
    if rule == "zero":
        k_0 = float(_lookup_parts(model)[0].params[K_0])
        if float(route._df().F(np.array([k_0 * (1.0 + 1e-12)]))[0]) > 0.0:
            raise InputError(
                "A DF that ends at its floor K_0 with a positive value (the rule "
                "'zero') moves the rows' support edge where F is not zero: its "
                "edge term is not carried."
            )
    if route.lam_closed:
        raise InputError(
            "At beta_c = +-1/2 the rows take the angle integral in closed form, "
            "which has no twin here."
        )
    if route.lfac and not route.rules.kernel:
        raise InputError(
            "The rows' angle rule is the numpy rule (rules.kernel false), which "
            "has no twin here."
        )
    return route


def _chunk(
    bc: Any,
    c1a: Any,
    c2a: Any,
    c3a: Any,
    pre: Any,
    split: Any,
    qc: Any,
    dqc: Any,
    top_half: Any,
    sa: Any,
    sb: Any,
    omb_all: Any,
    wa_: Any,
    wb_: Any,
    isb: Any,
    k: Any,
    read_f: Callable[..., Any],
    counts: tuple[tuple[int, int], ...],
) -> Any:
    """`_plane_s` at a chunk of pairs: the s-nodes, F there (by the depth `dqc`
    + s where K lies above `top_half`), the angle rule at the count each
    node's rule chose (each (index, count) of the rule taken over the chunk,
    and each node's own kept), the sum over s."""
    sp, rest = split[:, None], (qc - split)[:, None]
    s = sp * sa + rest * sb
    ws = sp * wa_ + rest * wb_
    k_s = jnp.where(isb[None, :], rest * omb_all[None, :], qc[:, None] - s)
    # piece a in the top half by the depth, as `EllipseRoute._plane_s`
    u_s = jnp.where(~isb[None, :] & (k_s > top_half), dqc[:, None] + s, jnp.nan)
    fs = read_f(k_s, u_s)
    # the nodes of no weight read no angle rule: s = 1 there, so that no
    # derivative a masked branch drops is infinite (the forward's values
    # where the weight is positive are the same)
    s_w = jnp.where(ws > 0.0, s, 1.0)
    c1 = c1a[:, None]
    c2 = c2a[:, None] * jnp.sqrt(s_w)
    c3 = c3a[:, None] * s_w
    kap = c1 * c2 + c3 - c2 * c2
    d2 = (c1 - c2) ** 2
    e2 = c2 * c2 - c3
    lam = jnp.where(
        kap > 0.0,
        jnp.sqrt(jnp.maximum(d2 / jnp.where(kap > 0.0, kap, 1.0), 1e-300)),
        jnp.pi,
    )
    lam = jnp.clip(lam, 1e-15, jnp.pi)
    kpi = c3 - c1 * c2 - c2 * c2
    lpi = jnp.where(
        kpi > 0.0,
        jnp.sqrt(jnp.maximum((c1 + c2) ** 2 / jnp.where(kpi > 0.0, kpi, 1.0), 1e-300)),
        jnp.pi,
    )
    lpi = jnp.clip(lpi, 1e-15, jnp.pi)
    shape = lam.shape
    f1, f2, f3, f4, f5, f6, f7, f8 = (
        jnp.broadcast_to(x_, shape).ravel()
        for x_ in (c1, c2, c3, kap, d2, e2, lam, lpi)
    )
    kf = k.ravel()
    lam_v = jnp.zeros(shape[0] * shape[1])
    for j, n in counts:
        xp_, wp_ = gauss_legendre_unit(int(n))
        # each count reads its own nodes; the others take the padding, P = 1
        chosen = kf == j
        g1, g2, g3, g4, g5, g6, g7, g8 = (
            jnp.where(chosen, f_, pad)
            for f_, pad in zip((f1, f2, f3, f4, f5, f6, f7, f8), _PAD, strict=True)
        )
        vals = _kernel(
            bc,
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g7,
            g8,
            jnp.asarray(xp_),
            jnp.asarray(wp_),
            "pow",
        )
        lam_v = jnp.where(chosen, vals, lam_v)
    lam_v = lam_v.reshape(shape)
    return pre * jnp.sum(ws * fs * jnp.where(ws > 0.0, lam_v, 0.0), axis=1)


_chunk_jit = jax.jit(_chunk, static_argnames=("read_f", "counts"))


def _plane_twin(
    route: Any,
    pairs: dict[str, Any],
    read_f: Callable[..., Any],
    read_g: Callable[..., Any] | None,
    bc: Any,
    ra: Any,
    part: tuple[int, int] | None = None,
) -> Any:
    """The plane's values at the pairs of one level under JAX, at the rule's
    decisions there: `EllipseRoute.plane` and `_plane_s`, the anisotropy's
    numbers `bc` and `ra` and the DF traced. With `part` (group, first pair
    of a chunk, `_twin_parts`), only that chunk's values, zero elsewhere."""
    r2d = float(pairs["r2d"])
    u = np.asarray(pairs["u"], dtype=np.float64)
    v = np.asarray(pairs["v"], dtype=np.float64)
    r = np.sqrt(r2d * r2d + u * u)
    psi = np.asarray(route.psi(r), dtype=np.float64)
    sgn = route.SGN
    q = 1.0 + sgn * (r / ra) ** 2 if sgn else jnp.ones_like(r)
    a = 1.0 + q * (u / r2d) ** 2
    kinetic = 0.5 * q * v * v * r * r / (r2d * r2d + q * u * u)
    qc = jnp.maximum(psi - kinetic, 0.0)
    # Q_c's depth below P0, as `EllipseRoute.plane` forms it
    depth = getattr(route.psi, "depth", None)
    dqc = (
        jnp.asarray(depth(r)) + kinetic
        if depth is not None
        else jnp.full(len(u), jnp.nan)
    )
    jac = (r / r2d) / jnp.sqrt(a * q)
    if not route.lfac:
        assert read_g is not None
        return jac * 2.0 * jnp.pi * read_g(qc, dqc)
    out = jnp.zeros(len(u))
    # which side of the split each pair takes, as the rule found it
    u_safe = np.where(u > 0.0, u, 1.0)
    s0 = jnp.where(u > 0.0, v * v * r * r / (2.0 * a * u_safe * u_safe), jnp.inf)
    s0_host = np.asarray(jax.lax.stop_gradient(s0))
    qc_host = np.asarray(jax.lax.stop_gradient(qc))
    at_s0 = (s0_host >= 0.0) & (s0_host < qc_host)
    split = jnp.where(at_s0, jnp.where(at_s0, s0, 0.0), 0.5 * qc)
    counts_of = np.full(len(u), -1, dtype=object)
    for at, chosen in pairs["counts"]:
        for row, i in enumerate(np.asarray(at)):
            counts_of[i] = chosen[row]
    rr = route.rules
    sr = rr.s_ratio
    df = route._df()
    counts = tuple(rr.angle_counts)
    for n_group, (group, (k0, ksa, ksb, kq, m, nphi)) in enumerate(pairs["groups"]):
        if part is not None and part[0] != n_group:
            continue
        at = np.asarray(group)
        u1, u2 = seg_unit(k0, ksa, sr), seg_unit(ksb, kq, sr)
        xg, wg = gl01(m)
        n1 = (len(u1) - 1) * m
        xb = jnp.asarray((u2[:-1, None] + np.diff(u2)[:, None] * xg).ravel())
        wb = jnp.asarray((np.diff(u2)[:, None] * wg).ravel())
        if 0.0 < route.BC < 1.0:
            p = 1.0 / (1.0 - bc)
            d = float(u2[1] - u2[0])
            xb = jnp.concatenate([u2[0] + d * xg**p, xb[m:]])
            wb = jnp.concatenate([d * p * xg ** (p - 1.0) * wg, wb[m:]])
        omb = 1.0 - xb
        if df.zero_map(m, kq):
            pz = 1.0 / (1.0 + df.p_zero)
            d = float(u2[-1] - u2[-2])
            lw = int(np.ceil(np.log(GOAL) / np.log(sr))) if df.k_zero != 0.0 else 0
            ew = np.concatenate([[0.0], sr ** np.arange(lw, -1, -1)])
            wn = (ew[:-1, None] + np.diff(ew)[:, None] * xg).ravel()
            ww = (np.diff(ew)[:, None] * wg).ravel()
            xb = jnp.concatenate([xb[:-m], 1.0 - d * wn**pz])
            wb = jnp.concatenate([wb[:-m], d * pz * wn ** (pz - 1.0) * ww])
            omb = jnp.concatenate([omb[:-m], d * wn**pz])
        n2 = int(xb.shape[0])
        sa = np.concatenate(
            [(u1[:-1, None] + np.diff(u1)[:, None] * xg).ravel(), np.ones(n2)]
        )
        sb = jnp.concatenate([jnp.zeros(n1), xb])
        omb_all = jnp.concatenate([jnp.zeros(n1), omb])
        isb = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
        wa_ = np.concatenate([(np.diff(u1)[:, None] * wg).ravel(), np.zeros(n2)])
        wb_ = jnp.concatenate([jnp.zeros(n1), wb])
        chosen = np.stack([np.asarray(counts_of[i]) for i in at])
        all_counts = tuple(enumerate(counts + (nphi // 2,)))
        # the s-nodes padded to a power of two, of no weight at s = 0 (Q_c,
        # where F is read), read by no angle rule: few shapes to compile
        n_s = len(sa)
        extra = max(_S_BUCKET, 1 << (n_s - 1).bit_length()) - n_s
        zeros = np.zeros(extra)
        sa_p = jnp.asarray(np.concatenate([sa, zeros]))
        sb_p = jnp.concatenate([sb, zeros])
        omb_p = jnp.concatenate([omb_all, zeros])
        wa_p = jnp.asarray(np.concatenate([wa_, zeros]))
        wb_p = jnp.concatenate([wb_, zeros])
        isb_p = jnp.asarray(np.concatenate([isb, np.zeros(extra, dtype=bool)]))
        chosen = np.concatenate(
            [chosen, np.full((len(at), extra), -1, dtype=chosen.dtype)], axis=1
        )
        for c0 in range(0, len(at), _PAIR_CHUNK):
            if part is not None and part[1] != c0:
                continue
            g = at[c0 : c0 + _PAIR_CHUNK]
            k = chosen[c0 : c0 + _PAIR_CHUNK]
            # one shape a chunk: padded with its first pair, cut off on return
            pad = np.concatenate([g, np.full(_PAIR_CHUNK - len(g), g[0])])
            kp = np.concatenate([k, np.repeat(k[:1], _PAIR_CHUNK - len(g), axis=0)])
            value = _chunk_jit(
                bc,
                v[pad] * r[pad] / r2d / a[pad],
                u[pad] / r2d * jnp.sqrt(2.0 / a[pad]),
                2.0 / q[pad],
                jac[pad] * 2.0 * r[pad] ** (-2.0 * bc),
                split[pad],
                qc[pad],
                dqc[pad],
                0.5 * df.Qend,
                sa_p,
                sb_p,
                omb_p,
                wa_p,
                wb_p,
                isb_p,
                jnp.asarray(kp.astype(np.int64)),
                read_f=read_f,
                counts=all_counts,
            )
            out = out.at[g].set(value[: len(g)])
    return out


def _twin_parts(route: Any, pairs: dict[str, Any]) -> list[tuple[int, int] | None]:
    """The chunks of `_plane_twin` at the pairs of one level, each a part its
    derivative is taken at alone (the pairing is a sum over pairs), which
    bounds the memory JAX holds for the derivative to one chunk."""
    if not route.lfac:
        return [None]
    return [
        (n_group, c0)
        for n_group, (group, _) in enumerate(pairs["groups"])
        for c0 in range(0, len(group), _PAIR_CHUNK)
    ]


def losvd_rows_explicit(
    model: Model,
    table: Any,
    settings: Any,
    cotangent: Sequence[Any],
    parameters: Sequence[str],
) -> dict[str, float]:
    r"""Return the explicit part of the transpose of a LOSVD table's rows: a
    cotangent on the rows, carried to the anisotropy at a fixed DF table.

    Parameters
    ----------
    model : Model
        The model of one component the table belongs to.
    table : LosvdTable
        Its table, each row's rule held.
    settings : Settings
        The settings the table was made with.
    cotangent : sequence of array or None
        One per row: a cotangent on ``row.losvd[0]``, or None. A row whose
        cotangent is None or zero is skipped.
    parameters : sequence of {"beta_c", "r_a"}
        The parameters.

    Returns
    -------
    explicit : dict of str to float

    Raises
    ------
    InputError
        For what the twin does not carry (`_refuse`).
    ResultError
        Where a row is not of its own rule, or the twin does not reproduce
        it to :math:`10^{-12}`.

    Notes
    -----
    Each row is recomputed by its rule and held to the stored row bit for
    bit (as `losvd_rows_vjp`), its normalization transposed the same way;
    each level's pairs are then read again by the twin of the plane, whose
    rows are held to the rule's to :math:`10^{-12}`, and the pairing of the
    cotangent with the twin's rows is differentiated by JAX in
    :math:`\beta_{\rm c}` and :math:`R_{\rm a}`.
    """
    from ._ellipse_model import DfReading, ellipse_route
    from ._losvd_stages import ellipse_rules

    names = tuple(parameters)

    def route_of() -> Any:
        route = ellipse_route(model, ellipse_rules(settings))
        route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
        return route

    route = _refuse(model, route_of, names)
    anisotropy = model.components[0].anisotropy
    reading = route._df().F
    if not isinstance(reading, DfReading):
        raise ResultError("The route reads its DF by no lookup: no transpose here.")
    shift = reading.shift
    _, _, leaves, _ = _lookup_parts(model)
    lookup, lookup_kinds = prepare_lookup(model.table, model.form, 0)
    fixed = (jnp.asarray(lookup.k), jnp.asarray(lookup.width), jnp.asarray(lookup.n))
    y0, m0, p0 = leaves

    def read_f(k: Any, u: Any = None) -> Any:
        a = LookupArrays(fixed[0], y0, m0, fixed[1], fixed[2], p0)
        log_f = evaluate_lookup(a, lookup_kinds, k, jnp, u)
        ok = jnp.isfinite(log_f)
        return jnp.where(ok, 10.0 ** (jnp.where(ok, log_f, 0.0) - shift), 0.0)

    read_g = None
    if not route.lfac:
        g_of = _g_twin(model, route._df(), shift)

        def read_g(q: Any, depth: Any = None) -> Any:
            return g_of(y0, m0, p0, q, depth)

    bc0 = float(anisotropy.beta_c)
    ra0 = float(anisotropy.r_a)
    total = {"beta_c": 0.0, "r_a": 0.0}
    for i, row in enumerate(table.rows):
        c = cotangent[i]
        # a row whose cotangent is zero adds nothing, as in `losvd_rows_vjp`:
        # it is neither recomputed nor read again by the twin
        if c is None or not np.any(np.asarray(c)):
            continue
        c = np.asarray(c, dtype=np.float64).ravel()
        r2d = float(row.r_2d)
        rule, _ = route.rule(r2d)
        vp, wp = route.panels(r2d)[:2]
        d, va, inv, trace = _row_levels(route, r2d, vp, wp, rule)
        norm = 2.0 * float(np.dot(wp, d))
        if (d / norm).tobytes() != np.asarray(row.losvd[0]).tobytes():
            raise ResultError(
                f"Row {i} recomputed by its own rule is not the row stored: its rule "
                "is not the forward's, so it has no transpose here."
            )
        collect: list[tuple[Any, ...]] = []
        by_speed = _row_by_levels(route, r2d, va, rule, trace, collect)
        c_d = (c - 2.0 * wp * float(c @ (d / norm))) / norm
        c_speed = np.bincount(inv, weights=c_d, minlength=len(va))
        twin_speed = np.zeros(len(va))
        for kind, speed, pairs, weight in collect:
            if kind != "P":
                continue

            for part in _twin_parts(route, pairs):

                def pairing(
                    bc: Any,
                    ra: Any,
                    pairs: Any = pairs,
                    w: Any = weight,
                    cw: Any = c_speed[speed],
                    part: Any = part,
                ) -> Any:
                    values = _plane_twin(route, pairs, read_f, read_g, bc, ra, part)
                    return jnp.sum(cw * w * values), values

                (value, values), grads = jax.value_and_grad(
                    pairing, argnums=(0, 1), has_aux=True
                )(jnp.asarray(bc0), jnp.asarray(ra0))
                twin_speed += np.bincount(
                    speed,
                    weights=np.asarray(weight) * np.asarray(values),
                    minlength=len(va),
                )
                total["beta_c"] += float(grads[0])
                total["r_a"] += float(grads[1])
        scale = float(np.max(np.abs(by_speed)))
        if not np.allclose(
            twin_speed, by_speed, rtol=0.0, atol=_TWIN_TOLERANCE * scale
        ):
            raise ResultError(
                f"Row {i}: the twin of the plane does not reproduce the row to "
                f"{_TWIN_TOLERANCE:.0e} of its largest value."
            )
    return {name: total[name] for name in names}
