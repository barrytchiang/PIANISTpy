r"""The truncated DF and the Poisson source of one component.

`truncated_source` takes the inversion's table of an untruncated
distribution function (DF) :math:`F(Q)` (the energies, the DF, the DF's
exact derivative and whether the DF changes sign), a truncation energy
:math:`\mathcal{K}_{\rm t}` and the truncation settings. `truncated_source`
returns a `TruncatedSource`: the truncated DF and the auxiliary density on
the source grid that `q_fast_grid` builds. The module also defines the
floors, the row-count bins (`row_bin_edges`) and the row-origin codes that
the Poisson solve, the fills and the profile checks share.

The truncated DF is

.. math::

    F_{\rm t}(K) = F(K + \mathcal{K}_{\rm t}) - F(\mathcal{K}_{\rm t}),

and the auxiliary density, the Poisson solve's source, is

.. math::

    \rho_{\rm aux}(P) = \int_0^P F_{\rm t}(Q) \, [2 (P - Q)]^n \, dQ,

with :math:`K`, :math:`Q` and :math:`P` relative energies and potentials in
internal units, and the kernel exponent :math:`n = 1/2 - \beta_{\rm c}`
(:math:`n = 1/2` for an Osipkov-Merritt model). The Poisson solve
multiplies :math:`\rho_{\rm aux}` by the anisotropy family's constant
prefactor and divides it by the anisotropy factor :math:`h(r)` to form the
density. Near the edge :math:`K \to 0` of a truncated model, the
difference :math:`F_{\rm t}(K)` is replaced by the integral of
:math:`dF/dQ` from :math:`\mathcal{K}_{\rm t}` to
:math:`K + \mathcal{K}_{\rm t}` and, closer to the edge, by the Taylor form
:math:`F_1 K + F_2 K^2 / 2`, with :math:`F_1` and :math:`F_2` the first and
second derivatives of :math:`F` at :math:`\mathcal{K}_{\rm t}`.
`truncated_source` raises AccuracyError where either replacement differs
from the quantity it replaces by more than :math:`10^{-2}` relative at the
grid node where the replacement takes over. The method is in Notes Section
T:3.2.
"""

__all__ = [
    "CEILING_GAP",
    "INNER_CEILING",
    "INNER_DEPTH",
    "OUTER_ANCHOR",
    "K_FLOOR",
    "LOG_FLOOR",
    "ORIGINS",
    "ROW_BINS_TOP",
    "RowBins",
    "TruncatedSource",
    "inner_zone_depth",
    "q_fast_grid",
    "row_bin_edges",
    "truncated_source",
]

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import betainc as _jbetainc
from jax.scipy.special import gammaln, gammasgn

from ..exceptions import AccuracyError, InputError, ResultError, UnphysicalModelError
from ..numerics.pchip import interp1d, pchip_slopes
from ..typing import Array
from ..validation import TIER_BOUNDS

# The constants below belong to the method: no settings level changes them.
K_FLOOR = (
    1e-10  # The potential floor: the source grid's first node and the solve's edge.
)
CEILING_GAP = 1e-9  # Gap from the source grid's top node up to P_0 - K_t.
# The bound on the rounding of the difference F(K + K_t) - F(K_t), in units
# of the machine epsilon of doubles times |F(K_t)|. The lookup of F takes one
# logarithm of K, ten operations of the cubic, one exponential and the
# difference, each rounded within one unit in the last place; 16 units bound
# the 13 roundings of the lookup and the difference.
CLIP_ULPS = 16
# The depth of the inner zone's end that the validated records carry, as a
# fraction of P_0: 1.01 times float32's machine epsilon. In the package,
# INNER_DEPTH sets only OUTER_ANCHOR; the package's own Poisson solves place
# the inner zone's end by inner_zone_depth.
INNER_DEPTH = 1.2040138130942069e-07
# The potential, as a fraction of P_0, at which the outer fill anchors the
# slope of its log-log extrapolation of the density.
OUTER_ANCHOR = 3.0 * INNER_DEPTH
LOG_FLOOR = 1e-30  # Lower clamp in the source quadrature's logs and powers.
ROW_BINS_TOP = 0.1  # Top of both sets of row-count bins, a fraction of P_0.
# The center's amplitude is read from the Poisson solve's rows whose depth
# below P_0 lies between INNER_CEILING and 3 INNER_CEILING, in internal units.
INNER_CEILING = 1e-4  # Inner edge of the center's window of depth below P_0.
# The origin of each row of a profile, one small integer per row: the Poisson
# solve, the edge continuation, the floor event, the inner fill, the outer
# fill, the outer fill's closed form, and the edge row.
ORIGINS = {
    "solve": 0,
    "continuation": 1,
    "event": 2,
    "inner_fill": 3,
    "outer_fill": 4,
    "closed_outer": 5,
    "edge": 6,
}


class RowBins(NamedTuple):
    r"""Row-count bins of a profile's rows in the potential and in the depth.

    `row_bin_edges` builds the record. All values are in internal units.

    Parameters
    ----------
    psi_edges : array
        Bin edges in the potential :math:`\Psi`, logarithmic with a width of
        ``profile_bin_decades`` decades, from `K_FLOOR`.
    psi_top : float
        The top of the potential's bins, ROW_BINS_TOP times :math:`P_0`.
    u_edges : array or None
        Bin edges in the depth :math:`u = P_0 - \Psi`, logarithmic with a
        width of ``profile_bin_decades`` decades, from `K_FLOOR`; None where
        `u_top` is at or below `K_FLOOR`.
    u_top : float
        The top of the depth's bins: the smaller of `psi_top` and the
        deepest depth among the rows counted.
    min_per_bin : int
        The fewest rows each bin must hold.
    """

    psi_edges: Array
    psi_top: float
    u_edges: Array | None
    u_top: float
    min_per_bin: int


def row_bin_edges(
    p_0: float, settings: Mapping[str, Any], u_max: float = np.inf
) -> RowBins:
    r"""Return the row-count bins of a profile's rows.

    The fills (`fill.fill_profile`) and the row-density check
    (`profile_checks.row_density`) both read the bins from `row_bin_edges`, so
    that the fills and the check count rows in the same bins. The bins are
    logarithmic, of width ``profile_bin_decades`` decades, starting at
    `K_FLOOR`. The potential's bins cover :math:`\Psi` up to ROW_BINS_TOP
    times :math:`P_0`. The depth's bins cover :math:`u = P_0 - \Psi` up to
    the smaller of ROW_BINS_TOP times :math:`P_0` and `u_max`. The last bin
    of either set may end above that set's top. Each bin must hold at least
    ``profile_rows_per_decade`` times ``profile_bin_decades`` rows, rounded,
    and at least one row.

    Parameters
    ----------
    p_0 : float
        The central potential :math:`P_0` of the truncated model, in
        internal units.
    settings : mapping
        The truncation settings (`Settings.truncation`).
    u_max : float, optional
        The deepest depth among the rows counted, in internal units.
        Default: ``np.inf``, which means the depth's bins end at
        ROW_BINS_TOP times :math:`P_0`.

    Returns
    -------
    bins : RowBins
        The bin edges and the top of the potential's bins and of the depth's
        bins, and the fewest rows per bin.
    """
    dlog = float(settings["profile_bin_decades"])
    min_per_bin = max(1, int(np.rint(int(settings["profile_rows_per_decade"]) * dlog)))
    psi_top = ROW_BINS_TOP * float(p_0)
    psi_edges = 10.0 ** np.arange(np.log10(K_FLOOR), np.log10(psi_top) + dlog, dlog)
    u_top = min(psi_top, float(u_max))
    u_edges = None
    if u_top > K_FLOOR:
        u_edges = 10.0 ** np.arange(np.log10(K_FLOOR), np.log10(u_top) + dlog, dlog)
    return RowBins(psi_edges, psi_top, u_edges, u_top, min_per_bin)


@dataclass(frozen=True)
class TruncatedSource:
    r"""The truncated DF and the auxiliary density on the source grid.

    All energies and potentials are in internal units.

    Parameters
    ----------
    q : array, shape (n,)
        The source grid in the truncated relative energy, from `K_FLOOR` to
        :math:`P_0 - \mathcal{K}_{\rm t}` less `CEILING_GAP`, where
        :math:`P_0` is the untruncated central potential.
    df : array, shape (n,)
        The truncated DF :math:`F_{\rm t}` at `q`.
    rho_aux : array, shape (n,)
        The auxiliary density :math:`\rho_{\rm aux}` at `q`.
    df_at_k_t : float
        The untruncated DF at :math:`\mathcal{K}_{\rm t}`, the constant that
        the truncation subtracts.
    taylor : tuple of float or None
        :math:`(F_1, F_2, K_{\rm switch})`: below :math:`K_{\rm switch}` the
        truncated DF is :math:`F_1 K + F_2 K^2 / 2`, with :math:`F_1` and
        :math:`F_2` the untruncated DF's first and second derivatives at
        :math:`\mathcal{K}_{\rm t}`. None for an untruncated DF, where no
        derivative table is given, or where the derivative table has fewer
        than four nodes with a finite positive :math:`dF/dQ` or does not
        bracket :math:`\mathcal{K}_{\rm t}`.
    taylor_check : dict or None
        The two joins near the edge. ``K_splice``: the energy where the edge
        integral of :math:`dF/dQ` hands over to the difference
        :math:`F(K + \mathcal{K}_{\rm t}) - F(\mathcal{K}_{\rm t})`.
        ``splice_rel_diff``: the edge integral over the difference, less 1,
        at ``K_splice``. ``switch_rel_diff``: the Taylor form over the edge
        integral, less 1, at the last grid node below
        :math:`K_{\rm switch}`; NaN where no grid node lies below
        :math:`K_{\rm switch}`. ``splice_reached``: whether the edge integral
        reaches :math:`|F(\mathcal{K}_{\rm t})|` on the grid. None wherever
        `taylor` is None.
    sentinel_q : float or None
        Where the top of the DF table holds +inf values (a DF above the
        largest double), the highest energy of the finite part of the table;
        None where the table has no +inf value.
    lookup : dict or None
        The lookup of the untruncated DF that the truncation reads: the
        table ``xp`` and ``fp`` (finite part only), the value ``right``
        returned above the table, the node slopes ``m`` (None where the
        lookup uses PCHIP's slopes at every node), and ``logf``, whether the
        lookup interpolates :math:`\ln F` rather than :math:`F`.
    edge_nodes : tuple of array or None
        The nodes :math:`(Q, dF/dQ)` of the Lagrange interpolant of the edge
        integral; None where `taylor` is None.
    clip : tuple of float
        The magnitude of the most negative value of the difference
        :math:`F(K + \mathcal{K}_{\rm t}) - F(\mathcal{K}_{\rm t})` set to
        zero, divided by :math:`|F(\mathcal{K}_{\rm t})|`, and the energy
        :math:`K` of the most negative value; ``(0.0, nan)`` where no value was set to
        zero. Each value set to zero is at most `CLIP_ULPS` units of
        rounding of :math:`F(\mathcal{K}_{\rm t})` in size.
    """

    q: Array
    df: Array
    rho_aux: Array
    df_at_k_t: float
    taylor: tuple[float, float, float] | None
    taylor_check: dict[str, Any] | None
    sentinel_q: float | None
    lookup: dict[str, Any] | None = None
    edge_nodes: tuple[Array, Array] | None = None
    clip: tuple[float, float] = (0.0, float("nan"))


def q_fast_grid(q_max: float, settings: Mapping[str, Any]) -> Array:
    r"""Return the source grid on [K_FLOOR, q_max - CEILING_GAP].

    The grid has a uniform middle part between two logarithmic tails. The
    lower tail is logarithmic in Q, from `K_FLOOR` to the middle part's
    start. The upper tail is logarithmic in ``q_max - Q``, from the middle
    part's end to ``q_max - CEILING_GAP``. The node counts and tail extents
    come from the settings ``n_source_rows``, ``n_source_tail_rows``,
    ``source_tail_extent`` and ``source_tail_floor``. A node whose distance
    to the node below it is at most 0.05 times the smaller of the next
    spacing and the median spacing is dropped.

    Parameters
    ----------
    q_max : float
        The top of the grid's range, :math:`P_0 - \mathcal{K}_{\rm t}`,
        in internal units.
    settings : mapping
        The truncation settings (`Settings.truncation`).

    Returns
    -------
    q : array
        The source grid, increasing, in internal units.
    """
    floor, gap = K_FLOOR, CEILING_GAP
    tail_floor = float(settings["source_tail_floor"])
    poisson_n, n_a_c, psi_extent = (
        int(settings["n_source_rows"]),
        int(settings["n_source_tail_rows"]),
        float(settings["source_tail_extent"]),
    )
    _tail_ext = min(psi_extent, 0.01 * float(q_max))
    _n_tail = min(n_a_c, poisson_n // 3)
    _n_lin = poisson_n - 2 * _n_tail
    tail_floor = tail_floor * float(q_max)
    _q_tail_lo = max(_tail_ext, tail_floor)
    _q_tail_hi = float(q_max) - max(_tail_ext, tail_floor)
    _grid_lin = np.linspace(_q_tail_lo, _q_tail_hi, _n_lin, endpoint=False)[1:]
    _omq_hi = np.logspace(np.log10(max(_tail_ext, tail_floor)), np.log10(gap), _n_tail)
    _grid_hi = float(q_max) * (1.0 - _omq_hi / float(q_max))
    _grid_lo = np.logspace(np.log10(floor), np.log10(_q_tail_lo), _n_tail)
    merged = np.unique(np.concatenate([_grid_lin, _grid_hi, _grid_lo]))
    _dm = np.diff(merged)
    _keep = np.concatenate(
        [[True], _dm > 0.05 * np.minimum(np.append(_dm[1:], _dm[-1]), np.median(_dm))]
    )
    merged = merged[_keep]
    merged = np.clip(merged, floor, float(q_max) - gap)
    return np.unique(merged)


def _lagrange4(xq: Array, yq: Array, x: Any, deriv: bool = False) -> Array:
    """Return the four-node Lagrange interpolant of (xq, yq) at x.

    The interpolant is a cubic in x through the four nodes of `xq` around
    each point of `x` (two below and two above where the table allows).
    With ``deriv=True`` the function returns the interpolant's derivative
    in x instead of its value.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    i0 = np.clip(np.searchsorted(xq, x) - 2, 0, len(xq) - 4)[:, None] + np.arange(4)
    xs, ys = xq[i0], yq[i0]
    out = np.zeros_like(x)
    for k in range(4):
        oth = [m for m in range(4) if m != k]
        den = np.prod([xs[:, k] - xs[:, m] for m in oth], axis=0)
        if deriv:
            num = sum(
                np.prod([x - xs[:, j] for j in oth if j != m], axis=0) for m in oth
            )
        else:
            num = np.prod([x - xs[:, m] for m in oth], axis=0)
        out += ys[:, k] * num / den
    return out


def _gamma_quotient(a, b, c):
    """Return Gamma(a) Gamma(b) / Gamma(c) from log-Gammas and signs.

    Gamma(x) exceeds the largest double for every x of at least 171.63,
    while the quotient Gamma(a) Gamma(b) / Gamma(c) can still be a finite
    double. The quotient
    is therefore formed as the exponential of a sum of log-Gammas, never as
    a quotient of Gammas. The argument that grows large in the source
    quadrature is ``1 + n``, with the kernel exponent
    ``n = 1/2 - beta_c``. At c = 0, a pole of Gamma(c), the function
    returns 0; at a negative integer c it returns NaN.
    """
    sign = gammasgn(a) * gammasgn(b) * gammasgn(c)
    return sign * jnp.exp(gammaln(a) + gammaln(b) - gammaln(c))


@lru_cache(maxsize=8)
def _p_rhs_kernel(abel_n: int, split_frac: float, underflow: float):
    """Return the jitted quadrature of rho_aux at a grid of potentials.

    The three arguments are closed over, so that each combination compiles
    once: `abel_n`, the Gauss-Legendre nodes per interval; `split_frac`,
    the fraction of the potential Psi where the first interval ends; and
    `underflow`, the clamp before every logarithm. The returned kernel
    integrates ``F(Q) (2 (Psi - Q))^n`` over [0, Psi] at each Psi of
    `p_grid`, with n = 1/2 for an Osipkov-Merritt model and
    ``n = 1/2 - bc`` otherwise. The kernel reads F from (`Q_grid`,
    `df_grid`) by a cubic lookup in ln Q, or, where `P0` is given, in the
    logit ``ln(Q / (P0 - Q))``, interpolating ln F unless `df_has_neg` is
    True. The kernel integrates the power-law tail ``tail_A Q^(-a)``, with
    ``a = min(tail_a, 0.99)``, in closed form, and the quadrature takes the
    DF less that power-law tail; the caller passes a nonzero `tail_A` only
    for an untruncated model. Where `n9` is given, the kernel adds the exact
    piece of rho_aux from energies below the grid's first node.
    """
    gl_xi, gl_w = np.polynomial.legendre.leggauss(abel_n)

    @partial(jax.jit, static_argnames=["model_type", "bc", "df_has_neg"])
    def kernel(
        Q_grid,
        df_grid,
        p_grid,
        bc,
        model_type,
        k_stretch=2.0,
        tail_A=0.0,
        tail_a=0.0,
        n9=None,
        P0=None,
        df_has_neg=False,
    ):
        n = 0.5 if model_type in ("OM_I", "OM_II") else 0.5 - bc
        _k2 = k_stretch if n >= 0.0 else 1.0 / (1.0 + n)
        _t_gl = jnp.array(0.5 * (gl_xi + 1.0), dtype=jnp.float64)
        _w_gl = jnp.array(0.5 * gl_w, dtype=jnp.float64)

        def _F_at(Qq):
            if P0 is None:
                return interp1d(
                    Qq,
                    Q_grid,
                    df_grid,
                    0.0,
                    df_grid[-1],
                    order="cubic",
                    logx=True,
                    logf=not df_has_neg,
                )
            _xq = jnp.log(jnp.maximum(Qq, underflow)) - jnp.log(
                jnp.maximum(P0 - Qq, underflow)
            )
            _xg = jnp.log(jnp.maximum(Q_grid, underflow)) - jnp.log(
                jnp.maximum(P0 - Q_grid, underflow)
            )
            return interp1d(
                _xq,
                _xg,
                df_grid,
                0.0,
                df_grid[-1],
                order="cubic",
                logx=False,
                logf=not df_has_neg,
            )

        def single_Psi_solver(Psi):
            split = jnp.minimum(split_frac * Psi, Psi * (1.0 - 1e-12))
            # Interval 1 is [0, split]. Its Gauss-Legendre nodes lie in ln Q
            # over [q0, split], with q0 the grid's first node. Below q0 an
            # untruncated model adds the tail's closed form or the exact piece
            # n9, and a truncated model adds nothing.
            _y0 = jnp.log(jnp.maximum(Q_grid[0], underflow))
            _y1 = jnp.log(jnp.maximum(split, underflow))
            _wy = jnp.maximum(_y1 - _y0, 0.0)
            Q1 = jnp.exp(_y0 + _wy * _t_gl)
            jac1 = _wy * Q1
            f1 = _F_at(Q1)
            kernel1 = (2.0 * jnp.maximum(Psi - Q1, 0.0)) ** n
            integral1 = jnp.dot(_w_gl, f1 * kernel1 * jac1)
            # The power-law tail A Q^-a, with a capped at 0.99, is integrated
            # in closed form, and the quadrature takes only the DF less the
            # power-law tail.
            _a = jnp.minimum(tail_a, 0.99)
            _x1 = jnp.clip(
                split / jnp.maximum(Psi, underflow),
                0.0,
                1.0,
            )
            _Bab = _gamma_quotient(1.0 - _a, 1.0 + n, 2.0 + n - _a)
            _ftail1 = tail_A * jnp.maximum(Q1, underflow) ** (-_a)
            _f1s = jnp.where(
                Q1 >= Q_grid[0],
                f1 - _ftail1,
                0.0,
            )
            if n9 is not None:
                # With n9, the tail's closed form covers only [q0, split], and
                # the exact piece n9 covers the energies below q0.
                _x0 = jnp.clip(Q_grid[0] / jnp.maximum(Psi, underflow), 0.0, 1.0)
                _cf_above = (
                    tail_A
                    * (2.0**n)
                    * Psi ** (1.0 + n - _a)
                    * _Bab
                    * (
                        _jbetainc(1.0 - _a, 1.0 + n, _x1)
                        - _jbetainc(1.0 - _a, 1.0 + n, _x0)
                    )
                )
                _nk = n9["n"]
                _Pj = n9["P"]
                _dP = jnp.maximum(Psi - _Pj, underflow)
                _xj = jnp.clip((Q_grid[0] - _Pj) / _dP, 0.0, 1.0)
                _Kj = (
                    (2.0**n)
                    * _dP ** (1.0 - _nk + n)
                    * _jbetainc(1.0 - _nk, 1.0 + n, _xj)
                )
                _dPm = jnp.maximum(Psi - n9["Pmin"], underflow)
                _Km = (
                    (2.0**n)
                    * _dPm ** (1.0 - _nk + n)
                    * _jbetainc(
                        1.0 - _nk,
                        1.0 + n,
                        jnp.clip((Q_grid[0] - n9["Pmin"]) / _dPm, 0.0, 1.0),
                    )
                )
                _Bk = _gamma_quotient(1.0 - _nk, 1.0 + n, 2.0 + n - _nk)
                if "wr" in n9:
                    # Where rho_aux's derivative is unbounded at P = 0, the
                    # piece below q0 is integrated by parts once more.
                    _dq = n9["d"] + (Q_grid[0] - n9["q0"])
                    _G1 = _gamma_quotient(1.0 - _nk, 1.0 + n, 1.0 + n - _nk)
                    _t1 = (
                        jnp.maximum(_dq, underflow) ** (-_nk)
                        * (2.0 * jnp.maximum(Psi - Q_grid[0], 0.0)) ** n
                    )
                    _t2 = (
                        (2.0**n) * _dP ** (n - _nk) * _G1 * _jbetainc(1.0 - _nk, n, _xj)
                    )
                    _below = n9["pref"] * jnp.dot(n9["wr"], _t1 + _t2)
                else:
                    _below = (
                        n9["pref"] * _Bk * (jnp.dot(n9["wg"], _Kj) + n9["dm_min"] * _Km)
                    )
                integral1 = jnp.dot(_w_gl, _f1s * kernel1 * jac1) + _cf_above + _below
            else:
                integral1 = jnp.dot(_w_gl, _f1s * kernel1 * jac1) + tail_A * (
                    2.0**n
                ) * Psi ** (1.0 + n - _a) * _Bab * _jbetainc(1.0 - _a, 1.0 + n, _x1)
            # Interval 2 is [split, Psi], mapped by Q = Psi - u^k. For n < 0
            # the power k = 1/(1 + n) cancels the kernel's singularity at
            # Q = Psi. For n >= 0 the power k is k_stretch.
            u_max2 = (Psi - split) ** (1.0 / _k2)
            u_nodes2 = u_max2 * _t_gl
            Q2 = Psi - u_nodes2**_k2
            dQ_dtau2 = _k2 * u_max2 * (jnp.maximum(u_nodes2, underflow) ** (_k2 - 1.0))
            f2 = _F_at(Q2)
            kernel2 = (2.0 * jnp.maximum(Psi - Q2, 0.0)) ** n
            integral2 = jnp.dot(_w_gl, f2 * kernel2 * dQ_dtau2)
            return integral1 + integral2

        return jax.vmap(single_Psi_solver)(p_grid)

    return kernel


def truncated_source(
    q_table: Array,
    df_table: Array,
    ddf_dq: Array | None,
    df_has_neg: bool,
    k_t: float,
    p_center: float,
    beta_c: float,
    model_type: str,
    settings: Mapping[str, Any],
    tail: tuple[Any, Any] | None = None,
    below_floor: Mapping[str, Any] | None = None,
) -> TruncatedSource:
    r"""Return the truncated DF and the auxiliary density of one component.

    All energies and potentials are in internal units.

    Parameters
    ----------
    q_table : array, shape (m,)
        Energies of the inversion's table, increasing.
    df_table : array, shape (m,)
        The untruncated DF at `q_table`, in linear :math:`F`. A contiguous
        band at the top of the table may hold +inf where the DF exceeds the
        largest double.
    ddf_dq : array, shape (k,), or None
        The DF's derivative :math:`dF/dQ` at the first :math:`k` energies of
        `q_table`, with :math:`k \leq m`, NaN where the inversion did not
        compute :math:`dF/dQ`; None where the inversion gives no derivative.
    df_has_neg : bool
        Whether the DF changes sign. Where True, the lookups interpolate
        :math:`F` rather than :math:`\ln F`.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`; 0 for an
        untruncated model.
    p_center : float
        The central relative potential :math:`P_0` of the untruncated model.
    beta_c : float
        The central anisotropy :math:`\beta_{\rm c}`. The kernel exponent is
        :math:`n = 1/2 - \beta_{\rm c}`, or 1/2 for an Osipkov-Merritt model.
    model_type : str
        The anisotropy family: "OM_I", "OM_II", "COM_I", "COM_II", or any
        other value for constant anisotropy.
    settings : mapping
        The truncation settings (`Settings.truncation`).
    tail : tuple of (float, float), optional
        :math:`(A, a)` of the DF's power-law tail :math:`A K^{-a}` below the
        table; read only for an untruncated model, which requires it.
        Default: ``None``, which means no tail (allowed only where
        :math:`\mathcal{K}_{\rm t} > 0`).
    below_floor : mapping, optional
        The exact piece of :math:`\rho_{\rm aux}` from energies below the
        table's first energy; read only for an untruncated model. Default:
        ``None``, which means the tail's closed form covers all energies
        below the source grid's split point.

    Returns
    -------
    source : TruncatedSource
        The truncated DF and :math:`\rho_{\rm aux}` on the source grid, the
        edge's Taylor form and joins, and the DF lookup the truncation used.

    Raises
    ------
    ResultError
        If the DF table holds NaN or -inf, if the table's +inf values do not
        form one contiguous band at the top, or if every value of the table
        is +inf.
    AccuracyError
        If the truncated DF has a non-finite value on the source grid.
    UnphysicalModelError
        If the truncated DF is negative at a grid node by more than
        `CLIP_ULPS` units of rounding of :math:`F(\mathcal{K}_{\rm t})`.
    InputError
        If the derivative table ends too close above
        :math:`\mathcal{K}_{\rm t}` for the edge integral to cover any grid
        node.
    AccuracyError
        If the edge integral and the difference differ by more than
        :math:`10^{-2}` relative at the splice, or the Taylor form and the
        edge integral differ by more than :math:`10^{-2}` relative at the
        switch. Smaller differences are stored in ``taylor_check`` for the
        model's validation report.
    InputError
        If `k_t` is 0 and `tail` is None.
    AccuracyError
        If :math:`\rho_{\rm aux}` has a non-finite value on the source grid.
    """
    order = "cubic"
    limit = TIER_BOUNDS[2]
    q_sampling = jnp.array(q_table, dtype=jnp.float64)
    df_raw = jnp.array(df_table, dtype=jnp.float64)
    zt_val = jnp.array(float(k_t), dtype=jnp.float64)
    bc_val = float(beta_c)
    model_type_val = str(model_type)

    # The top of the DF table may hold +inf where the DF exceeds the largest
    # double. Those values must form one contiguous band at the top, and the
    # truncation reads only the finite part of the table below that band.
    sentinel_q = None
    _df_np_g = np.asarray(df_raw, dtype=np.float64)
    _bad_g = np.isnan(_df_np_g) | np.isneginf(_df_np_g)
    if _bad_g.any():
        raise ResultError(
            f"{int(_bad_g.sum())} NaN/-inf values in the DF table: a corrupt table"
        )
    _pinf_g = np.isposinf(_df_np_g)
    if _pinf_g.any():
        _i_first_inf = int(np.argmax(_pinf_g))
        if not _pinf_g[_i_first_inf:].all():
            raise ResultError(
                "+inf rows of the DF table are not a contiguous top band: a "
                "corrupt table"
            )
        if _i_first_inf == 0:
            raise ResultError("the whole DF table is beyond the range of a double")
        sentinel_q = float(q_sampling[_i_first_inf - 1])
        q_sampling = q_sampling[:_i_first_inf]
        df_raw = df_raw[:_i_first_inf]

    # The cubic lookups of F take the exact slope from dF/dQ at each table
    # node where dF/dQ and F are finite, Q is positive and, for a DF that does
    # not change sign, F is positive; the lookups take PCHIP's slope at the
    # other nodes. The exact slopes are used only where at least four nodes
    # qualify. The slopes are d ln F / d ln Q, or dF / d ln Q where the DF
    # changes sign.
    _m_resample = None
    if ddf_dq is not None and len(ddf_dq) >= 4:
        _qs_m = np.asarray(q_sampling, dtype=np.float64)
        _fs_m = np.asarray(df_raw, dtype=np.float64)
        _n_m = min(len(ddf_dq), _qs_m.size)
        _ds_m = np.full(_qs_m.size, np.nan)
        _ds_m[:_n_m] = np.asarray(ddf_dq, dtype=np.float64)[:_n_m]
        _ok_m = (
            np.isfinite(_ds_m)
            & np.isfinite(_fs_m)
            & (_qs_m > 0.0)
            & (df_has_neg | (_fs_m > 0.0))
        )
        if int(_ok_m.sum()) >= 4:
            _xg_m = jnp.log(jnp.maximum(jnp.array(_qs_m), 1e-300))
            _fg_m = (
                jnp.array(_fs_m)
                if df_has_neg
                else jnp.log(jnp.maximum(jnp.array(_fs_m), 1e-300))
            )
            _m_np = np.array(pchip_slopes(_xg_m, _fg_m), dtype=np.float64)
            _m_np[_ok_m] = (
                (_ds_m * _qs_m)[_ok_m] if df_has_neg else (_ds_m * _qs_m / _fs_m)[_ok_m]
            )
            _m_resample = jnp.array(_m_np, dtype=jnp.float64)
    df_at_zt = float(
        interp1d(
            jnp.array(float(zt_val)),
            q_sampling,
            df_raw,
            0.0,
            float(df_raw[-1]),
            order=order,
            logx=True,
            logf=not df_has_neg,
            m=_m_resample,
        )
    )
    q_trunc = jnp.array(
        q_fast_grid(float(p_center - zt_val), settings), dtype=jnp.float64
    )

    # The truncated DF is F(K + K_t) - F(K_t), with the lookup energy capped
    # at the top of the table's finite part and at CEILING_GAP below the
    # central potential. A negative difference of at most CLIP_ULPS units of
    # rounding of F(K_t) is rounding: it is set to zero, and the largest such
    # value is stored in clip. A larger negative difference raises
    # UnphysicalModelError, because the truncated DF would be negative there.
    _Q_MAX_SAFE = min(float(q_sampling[-1]), float(p_center) - CEILING_GAP)
    _q_lookup = jnp.minimum(q_trunc + zt_val, _Q_MAX_SAFE)
    _df_trunc_raw = (
        interp1d(
            _q_lookup,
            q_sampling,
            df_raw,
            0.0,
            float(df_raw[-1]),
            order=order,
            logx=True,
            logf=not df_has_neg,
            m=_m_resample,
        )
        - df_at_zt
    )
    _raw_np = np.asarray(_df_trunc_raw, dtype=np.float64)
    _q_np = np.asarray(q_trunc, dtype=np.float64)
    _rounding = CLIP_ULPS * float(np.finfo(np.float64).eps) * abs(float(df_at_zt))
    if not np.all(np.isfinite(_raw_np)):
        _bad = np.flatnonzero(~np.isfinite(_raw_np))
        raise AccuracyError(
            f"{_bad.size} non-finite values of the truncated DF F(K + K_t) - F(K_t), "
            f"the first at K = {_q_np[_bad[0]]:.6e}: refused"
        )
    _j = int(np.argmin(_raw_np))
    if _raw_np[_j] < -_rounding:
        raise UnphysicalModelError(
            f"NEGATIVE TRUNCATED DF: F(K + K_t) - F(K_t) = {_raw_np[_j]:.6e} at "
            f"K = {_q_np[_j]:.6e}, beyond the rounding of the difference "
            f"({CLIP_ULPS} ulps of F(K_t) = {float(df_at_zt):.6e}): the lowered DF "
            "is negative there"
        )
    clip = (
        (float(-_raw_np[_j] / abs(float(df_at_zt))), float(_q_np[_j]))
        if _raw_np[_j] < 0.0
        else (0.0, float("nan"))
    )
    df_trunc = jnp.maximum(_df_trunc_raw, 0.0)

    # Near the edge K -> 0, the difference F(K + K_t) - F(K_t) loses digits
    # to cancellation, because the difference is small against F(K_t). There
    # the truncated DF is the integral of dF/dQ from K_t to K + K_t: four-node
    # Gauss-Legendre on each step of the source grid, through a four-node
    # Lagrange interpolant of the dF/dQ table. The integral replaces the
    # difference below the splice: the first grid node where the integral
    # reaches |F(K_t)|, or the last node the integral covers where the
    # integral stays below |F(K_t)|. At the grid nodes below K_switch the
    # Taylor form F_1 K + F_2 K^2 / 2 replaces the integral.
    taylor = taylor_check = None
    _d10_ok = False
    if float(zt_val) > 0.0 and ddf_dq is not None:
        _n_d10 = min(int(q_sampling.shape[0]), len(ddf_dq))
        _qs_np = np.asarray(q_sampling, dtype=np.float64)[:_n_d10]
        _dfd_np = np.asarray(ddf_dq, dtype=np.float64)[:_n_d10]
        _okd = np.isfinite(_dfd_np) & (_dfd_np > 0) & (_qs_np > 0)
        _okd &= np.concatenate([[True], np.diff(_qs_np) > 0.0])
        _d10_ok = int(_okd.sum()) >= 4 and (
            _qs_np[_okd][0] <= float(zt_val) <= _qs_np[_okd][-1]
        )
    if _d10_ok:
        _xq, _yq = _qs_np[_okd], _dfd_np[_okd]
        _Kt = float(zt_val)
        _F1 = float(_lagrange4(_xq, _yq, _Kt)[0])
        _F2 = float(_lagrange4(_xq, _yq, _Kt, deriv=True)[0])
        _qt = np.asarray(q_trunc, dtype=np.float64)
        # K_switch is where the neglected cubic term, with F'''(K_t) estimated
        # as F''(K_t) / K_t, reaches 1e-9 of the linear term F_1 K; K_switch
        # is at most 0.1 K_t.
        _Ksw = (
            min(0.1 * _Kt, (6e-9 * _F1 * _Kt / max(abs(_F2), 1e-300)) ** 0.5)
            if _F2 != 0
            else 0.1 * _Kt
        )
        _m10 = _qt < _Ksw
        _taylor = _F1 * _qt + 0.5 * _F2 * _qt**2
        _dt_np = np.array(df_trunc, dtype=np.float64)
        _xg4, _wg4 = np.polynomial.legendre.leggauss(4)
        _ksup = _qt < (_xq[-1] - _Kt)
        _lo = np.concatenate([[0.0], _qt[:-1]])[_ksup]
        _hw = 0.5 * (_qt[_ksup] - _lo)
        _Fint = np.cumsum(
            _hw
            * (
                _lagrange4(
                    _xq,
                    _yq,
                    ((_Kt + _lo + _hw)[:, None] + _hw[:, None] * _xg4[None, :]).ravel(),
                ).reshape(-1, 4)
                @ _wg4
            )
        )
        if _Fint.size == 0:
            raise InputError(
                f"the dF/dK table ends at Q = {_xq[-1]:.6e}, not above "
                f"K_t = {_Kt:.6e}: no edge integral"
            )
        _reach = np.flatnonzero(_Fint >= abs(float(df_at_zt)))
        _jx = int(_reach[0]) if _reach.size else int(_Fint.size) - 1
        _relx = float(_Fint[_jx] / max(_dt_np[_jx], 1e-300) - 1.0)
        if not abs(_relx) <= limit:
            raise AccuracyError(
                f"the edge integral of dF/dK and the difference table disagree "
                f"by {_relx:+.3e} at the splice "
                f"K = {_qt[_jx]:.3e} (> 0.01), where both "
                f"should be reliable"
            )
        _dt_np[:_jx] = _Fint[:_jx]
        _rel10 = float("nan")
        if _m10.any():
            _j = np.flatnonzero(_m10)[-1]
            _rel10 = float(
                _taylor[_j] / max(_Fint[_j] if _j < _Fint.size else _dt_np[_j], 1e-300)
                - 1.0
            )
            if not abs(_rel10) <= limit:
                raise AccuracyError(
                    f"the Taylor edge and the edge integral disagree by "
                    f"{_rel10:+.3e} at the switch "
                    f"K = {float(np.asarray(q_trunc)[_j]):.3e} "
                    f"(> 0.01): "
                    f"F'(K_t) = {_F1:.6e}, F''(K_t) = {_F2:.6e}"
                )
        _dt_np[_m10] = _taylor[_m10]
        df_trunc = jnp.array(_dt_np, dtype=jnp.float64)
        taylor = (float(_F1), float(_F2), float(_Ksw))
        taylor_check = dict(
            K_splice=float(_qt[_jx]),
            splice_rel_diff=float(_relx),
            switch_rel_diff=float(_rel10),
            splice_reached=bool(_reach.size),
        )

    # The auxiliary density on the source grid. Only an untruncated model adds
    # the DF's tail and the exact piece below the grid's first node.
    kernel = _p_rhs_kernel(
        int(settings["n_source_nodes"]),
        float(settings["quadrature_split"]),
        LOG_FLOOR,
    )
    untruncated = float(zt_val) == 0.0
    if untruncated and tail is None:
        raise InputError("An untruncated source needs the DF's tail (A, a).")
    p_rhs = kernel(
        q_trunc,
        df_trunc,
        q_trunc,
        bc_val,
        model_type_val,
        k_stretch=settings["quadrature_stretch"],
        tail_A=(tail[0] if untruncated and tail is not None else 0.0),
        tail_a=(tail[1] if untruncated and tail is not None else 0.0),
        n9=(below_floor if untruncated else None),
        P0=float(p_center) - float(zt_val),
        df_has_neg=bool(df_has_neg),
    )
    if float(zt_val) > 0.0 and taylor is not None:
        # Below K_switch the truncated DF is the Taylor form, so rho_aux there
        # is 2^n [F_1 B(2, n + 1) K^(n + 2) + F_2 / 2 B(3, n + 1) K^(n + 3)],
        # with B the beta function.
        from scipy.special import beta as _sbeta

        _F1e, _F2e, _Kswe = taylor
        _n = 0.5 - float(bc_val)
        _qf = np.asarray(q_trunc, dtype=np.float64)
        _me = _qf < _Kswe
        _closed = (2.0**_n) * (
            _F1e * _sbeta(2.0, _n + 1.0) * _qf ** (_n + 2.0)
            + 0.5 * _F2e * _sbeta(3.0, _n + 1.0) * _qf ** (_n + 3.0)
        )
        _pr = np.array(p_rhs, dtype=np.float64)
        _pr[_me] = _closed[_me]
        p_rhs = jnp.array(_pr, dtype=jnp.float64)
    n_nonfinite = int(jnp.sum(~jnp.isfinite(p_rhs)))
    if n_nonfinite > 0:
        raise AccuracyError(
            f"{n_nonfinite} non-finite values in the Poisson source: refused"
        )
    return TruncatedSource(
        q=np.asarray(q_trunc, dtype=np.float64),
        df=np.asarray(df_trunc, dtype=np.float64),
        rho_aux=np.asarray(p_rhs, dtype=np.float64),
        df_at_k_t=float(df_at_zt),
        taylor=taylor,
        taylor_check=taylor_check,
        sentinel_q=sentinel_q,
        lookup=dict(
            xp=np.asarray(q_sampling, dtype=np.float64),
            fp=np.asarray(df_raw, dtype=np.float64),
            right=float(df_raw[-1]),
            m=None if _m_resample is None else np.asarray(_m_resample),
            logf=not df_has_neg,
        ),
        edge_nodes=(_xq, _yq) if _d10_ok else None,
        clip=clip,
    )


def inner_zone_depth(p_0: float, ode_rtol: float) -> float:
    r"""Return the depth where the Poisson solve's inner zone ends.

    The depth is :math:`P_0 \epsilon_{64} / {\rm ode\_rtol}`, with
    :math:`P_0` the central potential and :math:`\epsilon_{64}` the spacing
    of doubles at 1. A depth formed as :math:`u = P_0 - \Phi` carries the
    rounding :math:`\epsilon_{64} P_0` of the potential :math:`\Phi`, a
    relative error :math:`\epsilon_{64} P_0 / u` in :math:`u`. At the
    returned depth, the relative error :math:`\epsilon_{64} P_0 / u` equals
    the solve's relative tolerance `ode_rtol`. At smaller depths the solve
    carries the depth :math:`u` as its own variable instead of forming
    :math:`u` from :math:`\Phi`.

    >>> inner_zone_depth(1.0, 1e-11)
    2.220446049250313e-05
    """
    return float(p_0) * float(np.finfo(np.float64).eps) / float(ode_rtol)
