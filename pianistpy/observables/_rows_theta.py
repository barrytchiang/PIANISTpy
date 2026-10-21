r"""The explicit part of the derivative of LOSVD rows in the anisotropy.

`Differentiable` calls `losvd_rows_explicit` when a gradient of a LOSVD in
the central anisotropy :math:`\beta_{\rm c}` or the anisotropy radius
:math:`R_{\rm a}` is asked. The explicit part is the derivative of the
LOSVD table's rows in :math:`\beta_{\rm c}` and :math:`R_{\rm a}` at a fixed
DF table, with every decision of the rows' quadrature held fixed. The
derivative through the DF table is a separate part (`losvd_rows_vjp`).

A row holds the LOSVD at one projected radius :math:`R_{\rm 2D}` and at the
row's velocity nodes. The row's value at one velocity is the integral, over the
line-of-sight nodes in t, of the velocity-plane integral
(`EllipseRoute.plane`). For a model with the factor
:math:`(L^2)^{-\beta_{\rm c}}`, the velocity-plane integral is the integral
of :math:`F (L^2)^{-\beta_{\rm c}}` over the depth s below :math:`Q_c` and
the angle :math:`\phi`, where F is the DF and :math:`Q_c` is the largest
energy variable on the velocity plane. For a model without the factor, the
velocity-plane integral is :math:`2\pi\,G(Q_c)`, with
:math:`G(Q_c) = \int_0^{Q_c} F(Q_c - s)\,{\rm d}s`.

At a fixed DF table, the anisotropy changes the velocity-plane integral in
two ways. First, :math:`R_{\rm a}` enters
:math:`q(r) = 1 + s_{\rm a}(r/R_{\rm a})^2`, with :math:`s_{\rm a}` the
sign of the anisotropy family, and q changes a, :math:`Q_c`, the
Jacobian, the split of the depth interval at the point where L = 0, and
the depth nodes. Second, :math:`\beta_{\rm c}` enters the prefactor
:math:`r^{-2\beta_{\rm c}}`, the map of the depth nodes toward the split,
and the angle integral.

The quadrature's decisions are held fixed at the values the forward
computation chose. The decisions are the supports and the L = 0 splits in
t, each node's level, the layers of each group of pairs, the side of the
split that each pair takes, and each node's angle count. JAX
differentiates a twin of the velocity-plane integral (`_plane_twin`) at
those decisions. In the twin, the traced lookup reads F, the twin of G
(`_g_twin`) reads G, and the kernel `_angle_rule._kernel` evaluates the
angle integral in its general power form. With
``Settings(check_rows=True)``, each row recomputed by the forward
computation must equal the stored row bit for bit, and the twin must
reproduce the row to a relative :math:`10^{-12}` (`_TWIN_TOLERANCE`).

Where the edge of the DF's support moves and F is not zero at that edge,
the derivative has an edge term that the held supports do not carry.
`losvd_rows_explicit` therefore raises InputError for a DF that ends at its
floor with a positive value (the rule ``"zero"``) and for a DF that
diverges at zero energy (the rule ``"tail"``). Notes Section T:7.3 states
the tangent rows of the observables.
"""

__all__ = ["losvd_rows_explicit"]

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..df import Model
from ..df.lookup import K_0, LookupArrays, LookupKinds, evaluate_lookup
from ..exceptions import InputError, ResultError
from ..numerics.quadrature import gauss_legendre_unit
from ..validation import TIER_BOUNDS
from ._angle_rule import _PAD, _kernel
from ._ellipse_rules import gl01, seg_unit
from ._table_vjp import (
    DF_PARAMETERS,
    RowsBasis,
    _g_twin,
    _row_by_levels,
    _row_levels,
    _row_reads,
)

# The message of the InputError that losvd_rows_explicit raises when the
# derivative of an isotropic model's rows in beta_c is asked.
ISOTROPIC_ROWS = (
    "The line-of-sight velocity distributions of an isotropic model are computed "
    "from G alone, with no L factor. The derivative of the distributions in beta_c "
    "needs the integral of the L factor at beta_c = 0, and the computation of the "
    "distributions does not compute the integral of the L factor."
)
# The largest difference allowed between the twin's row and the forward
# computation's row, relative to the row's largest value, with check_rows.
# The twin repeats the forward arithmetic in JAX, so the two rows differ by
# rounding only. The tolerance 1e-12 is below the bound 1e-6.
_TWIN_TOLERANCE = 1e-12
# The number of pairs of one group that _chunk evaluates at once under JAX. The
# chunk size bounds the memory JAX holds for one derivative.
_PAIR_CHUNK = 32
# The smallest padded count of a group's depth nodes.
_S_BUCKET = 64


def _refuse(
    model: Model, basis_of: Callable[[], RowsBasis], table: Any, names: tuple[str, ...]
) -> tuple[RowsBasis, Any]:
    """Raise InputError for a model or a request that the twin does not
    carry, and return the model's basis and route otherwise.

    The function raises InputError in each of the following cases.

    - The model is a COM Type II model.
    - The model is isotropic and `names` holds ``"beta_c"``.
    - `names` holds a parameter outside the family's entry of
      `DF_PARAMETERS`.
    - The DF's rule below the floor is ``"tail"``.
    - The DF's rule below the floor is ``"zero"`` and F is positive just
      above the floor :math:`K_0`.
    - The route takes the angle integral in closed form
      (:math:`\\beta_{\\rm c} = \\pm 1/2`).
    - The route has the factor :math:`(L^2)^{-\\beta_{\\rm c}}` and its angle
      rule is the numpy rule (``rules.kernel`` false).

    The function calls `basis_of` only after the first four checks pass.
    The returned pair is the basis (`RowsBasis`, the model's route and
    lookup) and the route that the basis forms with the table's kept
    ranges."""
    family = model.components[0].anisotropy.family
    if family == "COM_II":
        raise InputError(
            "The DF path takes no derivative in the anisotropy of a COM Type II "
            "model, because the derivative of the DF table is not computed for a "
            "COM Type II model."
        )
    if family == "isotropic" and "beta_c" in names:
        raise InputError(ISOTROPIC_ROWS)
    allowed = DF_PARAMETERS[family]
    for name in names:
        if name not in allowed:
            raise InputError(
                "The derivatives of the line-of-sight velocity distributions of a "
                f"model of the anisotropy family {family} are taken in the "
                f"parameters {list(allowed)}; the parameter given is {name!r}."
            )
    rule = model.form.below_floor[0]
    if rule == "tail":
        raise InputError(
            "For a DF that diverges at zero energy (the rule 'tail'), F is not zero "
            "at the edge of the integration range of the line-of-sight velocity "
            "distributions, and the edge moves with the parameters; the derivative "
            "does not include the resulting edge term."
        )
    basis = basis_of()
    route = basis.route(table)
    if rule == "zero":
        k_0 = float(basis.parts[0].params[K_0])
        if float(route._df().F(np.array([k_0 * (1.0 + 1e-12)]))[0]) > 0.0:
            raise InputError(
                "For a DF that ends at the DF's floor K_0 with a positive value (the "
                "rule 'zero'), F is not zero at the edge of the integration range of "
                "the line-of-sight velocity distributions, and the edge moves with "
                "the parameters; the derivative does not include the resulting edge "
                "term."
            )
    if route.lam_closed:
        raise InputError(
            "At beta_c = +-1/2, the line-of-sight velocity distributions take the "
            "angle integral in closed form, and the derivative of the closed form "
            "is not implemented."
        )
    if route.lfac and not route.rules.kernel:
        raise InputError(
            "The line-of-sight velocity distributions use the numpy angle rule "
            "(the setting observables.rows_route = 'numpy'), and the derivative of "
            "the numpy angle rule is not implemented."
        )
    return basis, route


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
    lookup: LookupArrays,
    shift: Any,
    kinds: LookupKinds,
    counts: tuple[tuple[int, int], ...],
) -> Any:
    """Return the twin of `EllipseRoute._plane_s` on one chunk of pairs.

    The function forms the depth nodes s of each pair, reads F at the
    energies :math:`K = Q_c - s`, evaluates the angle integral at each node,
    and sums over the depth nodes. Where K lies above `top_half`, the lookup
    reads F by the depth ``dqc + s`` below the central potential. The angle
    integral at a node uses the angle count that the forward computation
    chose for that node (the index in `k` into `counts`). The kernel runs
    once for each entry of `counts` on the whole chunk, and each node keeps
    the value of its own count. F is ``10**(log10 F - shift)``. The lookup's
    arrays and ``shift`` are traced, so one compilation serves every model
    with the same lookup kinds and the same counts."""
    sp, rest = split[:, None], (qc - split)[:, None]
    s = sp * sa + rest * sb
    ws = sp * wa_ + rest * wb_
    k_s = jnp.where(isb[None, :], rest * omb_all[None, :], qc[:, None] - s)
    # On piece a, where K lies above top_half, the lookup reads F by the depth
    # below the central potential, as EllipseRoute._plane_s does.
    u_s = jnp.where(~isb[None, :] & (k_s > top_half), dqc[:, None] + s, jnp.nan)
    log_f = evaluate_lookup(lookup, kinds, k_s, jnp, u_s)
    ok = jnp.isfinite(log_f)
    fs = jnp.where(ok, 10.0 ** (jnp.where(ok, log_f, 0.0) - shift), 0.0)
    # A node of zero weight takes s = 1 in the angle integral. At s = 0 the
    # factor sqrt(s) in c2 has an infinite derivative, and an infinite
    # derivative in a branch that jnp.where masks still turns the gradient
    # into NaN. The values at the nodes of positive weight do not change.
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
        # The nodes that chose the count n keep their arrays. Every other node
        # takes the padding values, at which P = 1.
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


_chunk_jit = jax.jit(_chunk, static_argnames=("kinds", "counts"))


def _plane_twin(
    route: Any,
    pairs: dict[str, Any],
    read_f: tuple[LookupArrays, Any, LookupKinds],
    read_g: Callable[..., Any] | None,
    bc: Any,
    ra: Any,
    part: tuple[int, int] | None = None,
) -> Any:
    """Return the twin of the velocity-plane integral at the pairs of one
    level, under JAX.

    The twin repeats `EllipseRoute.plane` and `EllipseRoute._plane_s` at the
    decisions that the forward computation recorded in `pairs`. The central
    anisotropy `bc`, the anisotropy radius `ra` and the DF's lookup arrays
    are traced. The tuple `read_f` holds the lookup's arrays, the shift and
    the lookup's kinds, as `_chunk` takes them. The callable `read_g` reads
    G for a model without the factor :math:`(L^2)^{-\\beta_{\\rm c}}`, and is
    None otherwise. With `part`, a pair of a group index and the first pair
    of a chunk (`_twin_parts`), the function returns the values of that
    chunk's pairs and zero at every other pair."""
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
    # The depth of Q_c below the central potential P0, formed as
    # EllipseRoute.plane forms it. The depth is NaN where the potential holds
    # no depth function.
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
    # The side of the split that each pair takes is decided on the host, from
    # values without gradient, so the decision is the forward computation's.
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
            lw = (
                int(np.ceil(np.log(TIER_BOUNDS[0]) / np.log(sr)))
                if df.k_zero != 0.0
                else 0
            )
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
        # The depth nodes are padded to a power of two, at least _S_BUCKET, so
        # that few array shapes are compiled. A padding node lies at s = 0,
        # where F is read at K = Q_c, has zero weight, and takes no angle
        # integral.
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
            # Every chunk has _PAIR_CHUNK pairs. A short chunk is padded with
            # copies of its first pair, and the padding is cut off on return.
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
                lookup=read_f[0],
                shift=read_f[1],
                kinds=read_f[2],
                counts=all_counts,
            )
            out = out.at[g].set(value[: len(g)])
    return out


def _twin_parts(route: Any, pairs: dict[str, Any]) -> list[tuple[int, int] | None]:
    """Return the chunks of `_plane_twin` at the pairs of one level.

    Each chunk is a pair of a group index and the index of the chunk's first
    pair in the group. The pairing of the cotangent with the twin is a sum
    over pairs, so the derivative is the sum of the derivatives of the
    chunks. Each chunk's derivative is taken alone, which bounds the memory
    JAX holds to one chunk. A model without the factor
    :math:`(L^2)^{-\\beta_{\\rm c}}` has the single part None, the whole
    level."""
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
    basis: RowsBasis | None = None,
) -> dict[str, float]:
    r"""Return the explicit part of the transpose of a LOSVD table's rows.

    The transpose carries a cotangent on the rows to the anisotropy
    parameters :math:`\beta_{\rm c}` and :math:`R_{\rm a}`, at a fixed DF
    table and with the quadrature's decisions of every row held fixed.

    Parameters
    ----------
    model : Model
        The model of one component that the table belongs to.
    table : LosvdTable
        The model's LOSVD table.
    settings : Settings
        The settings with which the table was computed.
    cotangent : sequence of array or None
        One entry for each row of `table`: a cotangent on ``row.losvd[0]``,
        or None. A row whose cotangent is None or all zero is skipped.
    parameters : sequence of {"beta_c", "r_a"}
        The anisotropy parameters in which the derivative is taken.
    basis : RowsBasis, optional
        The model's route and lookup, where the caller already holds them.
        Default: ``None``, which means that the function forms the basis
        from `model` and `settings`.

    Returns
    -------
    explicit : dict of str to float
        The explicit part of the derivative of the pairing of `cotangent`
        with the rows, keyed by each name in `parameters`, in the order of
        `parameters`.

    Raises
    ------
    InputError
        If the model or `parameters` is outside what the twin carries
        (`_refuse` lists the cases).
    ResultError
        If the route reads its DF by no lookup.
    ResultError
        If ``settings.check_rows`` is true and a row recomputed by the
        forward computation is not bit for bit the stored row.
    ResultError
        If ``settings.check_rows`` is true and the twin's row differs from
        the forward computation's row by more than :math:`10^{-12}` of the
        row's largest value.

    Notes
    -----
    The function recomputes each row once by the forward computation and
    collects the pairs of each level, as `losvd_rows_vjp` does. The
    cotangent is carried through the row's normalization in the same way as
    in `losvd_rows_vjp`. The twin of the velocity-plane integral then reads
    each level's pairs again. JAX differentiates the pairing of the
    cotangent with the twin's values in :math:`\beta_{\rm c}` and
    :math:`R_{\rm a}`. With ``settings.check_rows``, the function also
    collects each row's pairs level by level (`_row_by_levels`) and checks
    the twin's row against the forward computation's row.
    """
    from ._ellipse_model import DfReading

    names = tuple(parameters)

    def basis_of() -> RowsBasis:
        return RowsBasis(model, settings) if basis is None else basis

    held, route = _refuse(model, basis_of, table, names)
    anisotropy = model.components[0].anisotropy
    reading = route._df().F
    if not isinstance(reading, DfReading):
        raise ResultError(
            "The line-of-sight integrator's DF is not a DfReading lookup, so the "
            "derivative of the line-of-sight velocity distributions cannot be "
            "computed."
        )
    shift = reading.shift
    _, _, leaves, _ = held.parts
    lookup, lookup_kinds = held.lookup
    fixed = (jnp.asarray(lookup.k), jnp.asarray(lookup.width), jnp.asarray(lookup.n))
    y0, m0, p0 = leaves

    read_f = (
        LookupArrays(fixed[0], y0, m0, fixed[1], fixed[2], p0),
        jnp.asarray(shift, dtype=jnp.float64),
        lookup_kinds,
    )

    read_g = None
    if not route.lfac:
        g_tables, g_read = _g_twin(model, route._df(), shift, held.lookup)
        # The branches of G depend only on the lookup's arrays, which are
        # fixed in this function. The branches are therefore formed once and
        # read by every level of every row. Only the reading of G at Q_c
        # depends on the anisotropy.
        branches = g_tables(y0, m0, p0)

        def read_g(q: Any, depth: Any = None) -> Any:
            return g_read(branches, q, depth)

    check = settings.check_rows
    bc0 = float(anisotropy.beta_c)
    ra0 = float(anisotropy.r_a)
    total = {"beta_c": 0.0, "r_a": 0.0}
    for i, row in enumerate(table.rows):
        c = cotangent[i]
        # A row whose cotangent is None or all zero adds nothing to the
        # derivative, as in losvd_rows_vjp. The row is neither recomputed nor
        # read by the twin.
        if c is None or not np.any(np.asarray(c)):
            continue
        c = np.asarray(c, dtype=np.float64).ravel()
        r2d = float(row.r_2d)
        rule, _ = route.rule(r2d)
        vp, wp = route.panels(r2d)[:2]
        if not check:
            d, va, inv, collect = _row_reads(route, r2d, vp, wp, rule)
            norm = 2.0 * float(np.dot(wp, d))
        else:
            d, va, inv, trace = _row_levels(route, r2d, vp, wp, rule)
            norm = 2.0 * float(np.dot(wp, d))
            if (d / norm).tobytes() != np.asarray(row.losvd[0]).tobytes():
                raise ResultError(
                    "The line-of-sight velocity distribution at the radius index "
                    f"{i}, recomputed by the distribution's own rule, differs from "
                    "the stored distribution, so the stored distribution was not "
                    "made by the forward computation's rule and has no transpose."
                )
            collect = []
            by_velocity = _row_by_levels(route, r2d, va, rule, trace, collect)
        c_d = (c - 2.0 * wp * float(c @ (d / norm))) / norm
        c_velocity = np.bincount(inv, weights=c_d, minlength=len(va))
        twin_velocity = np.zeros(len(va))
        for kind, velocity, pairs, weight in collect:
            if kind != "P":
                continue

            for part in _twin_parts(route, pairs):

                def pairing(
                    bc: Any,
                    ra: Any,
                    pairs: Any = pairs,
                    w: Any = weight,
                    cw: Any = c_velocity[velocity],
                    part: Any = part,
                ) -> Any:
                    values = _plane_twin(route, pairs, read_f, read_g, bc, ra, part)
                    return jnp.sum(cw * w * values), values

                (value, values), grads = jax.value_and_grad(
                    pairing, argnums=(0, 1), has_aux=True
                )(jnp.asarray(bc0), jnp.asarray(ra0))
                if check:
                    twin_velocity += np.bincount(
                        velocity,
                        weights=np.asarray(weight) * np.asarray(values),
                        minlength=len(va),
                    )
                total["beta_c"] += float(grads[0])
                total["r_a"] += float(grads[1])
        if not check:
            continue
        scale = float(np.max(np.abs(by_velocity)))
        if not np.allclose(
            twin_velocity, by_velocity, rtol=0.0, atol=_TWIN_TOLERANCE * scale
        ):
            raise ResultError(
                f"At the radius index {i}, the twin of the plane integral does not "
                f"reproduce the line-of-sight velocity distribution to "
                f"{_TWIN_TOLERANCE:.0e} of the distribution's largest value."
            )
    return {name: total[name] for name in names}
