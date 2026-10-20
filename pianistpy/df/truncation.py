"""The truncated DF and the Poisson source of a model.

From the table the inversion leaves (the energies, the DF, its exact
derivative and whether the DF changes sign), the truncation energy and the
run's settings: the truncated DF ``F(K + K_t) - F(K_t)`` on the source grid, its
Taylor form at the edge with the edge integral that joins it to the table,
and the auxiliary density rho_aux(P) = INT_0^P F(Q) (2 (P - Q))^n dQ that
the Poisson solve takes as its source. The density of a component is rho_aux
times its anisotropy's factor in r, which the Poisson solve applies.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend; the settings are read
by the names the pipeline uses.
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
from ..validation import HARD_LIMIT

# The method's own numbers; no tier changes them.
K_FLOOR = (
    1e-10  # the potential's floor: the source grid's first node and the solve's edge
)
CEILING_GAP = 1e-9  # the source grid's top lies this far below the central potential
# the rounding of the truncated DF's difference F(K + K_t) - F(K_t), in units
# of F(K_t): the lookup's operations (the logarithm of K, the cubic's ten, the
# exponential and the difference), each within one unit in the last place
CLIP_ULPS = 16
# the validated records' inner zone's end (float32's epsilon), a fraction of P_0:
# the value such a record carries; the package's own solves follow inner_zone_depth
INNER_DEPTH = 1.2040138130942069e-07
# the outer fill's slope anchor, a fraction of P_0
OUTER_ANCHOR = 3.0 * INNER_DEPTH
LOG_FLOOR = 1e-30  # the clamp before a logarithm in the source quadrature
ROW_BINS_TOP = 0.1  # both sides' row-count bins end at this fraction of P_0
INNER_CEILING = 1e-4  # the depth of the solve's rows the center's amplitude is read at
# where each row of a profile comes from, one small integer per row
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
    """The row-count bins of a profile's rows on both sides (`row_bin_edges`)."""

    psi_edges: Array
    psi_top: float
    u_edges: Array | None
    u_top: float
    min_per_bin: int


def row_bin_edges(
    p_0: float, settings: Mapping[str, Any], u_max: float = np.inf
) -> RowBins:
    """Return the row-count bins of a profile's rows: the one definition the
    fill (`fill.fill_profile`) and the check (`profile_checks.row_density`) read.

    The bins are decades of ``profile_bin_decades`` from K_FLOOR, on two
    sides: the potential Psi up to ROW_BINS_TOP times :math:`P_0`, and the
    depth ``u = P_0 - Psi`` up to the smaller of ROW_BINS_TOP times
    :math:`P_0` and the rows' deepest depth `u_max`. The last bin of a side may
    reach beyond its top.

    Parameters
    ----------
    p_0 : float
        The central potential of the truncated model.
    settings : mapping
        The truncation's settings (`Settings.truncation`).
    u_max : float, optional
        The deepest depth among the rows counted, at most ROW_BINS_TOP times
        :math:`P_0`.

    Returns
    -------
    bins : RowBins
        The edges and the top of each side, `u_edges` None where the depth's
        top lies at or below K_FLOOR, and the rows each bin must hold.
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
    r"""The truncated DF and the Poisson source on the source grid.

    Parameters
    ----------
    q : array, shape (n,)
        The source grid in the truncated relative potential, from the grid
        floor to ``P_0 - K_t`` less the ceiling gap.
    df : array, shape (n,)
        The truncated DF on `q`.
    rho_aux : array, shape (n,)
        The auxiliary density on `q`.
    df_at_k_t : float
        The untruncated DF at :math:`\mathcal{K}_{\rm t}`, the constant the
        truncation subtracts.
    taylor : tuple of float or None
        (F_1, F_2, K_switch): the DF is F_1 K + F_2 K^2 / 2 below K_switch.
        None for an untruncated DF or where the derivative table is unusable.
    taylor_check : dict or None
        The two joins of the edge: ``K_splice`` and ``splice_rel_diff`` (the
        edge integral against the difference of the table at the splice),
        ``switch_rel_diff`` (the Taylor form against the edge integral at the
        switch; NaN where no row takes the Taylor form), ``splice_reached``.
    sentinel_q : float or None
        The last energy of the table's finite band where its top carries
        unrepresentable (+inf) rows; None where it does not.
    lookup : dict
        The lookup of the untruncated DF the truncation reads: its table
        ``xp`` and ``fp`` (the finite band), the value ``right`` above it,
        its node slopes ``m`` (None for PCHIP's), and whether it is read in
        logarithms of F (``logf``).
    edge_nodes : tuple of array or None
        The nodes (Q, dF/dQ) of the edge's Lagrange interpolation; None where
        the edge has none.
    clip : tuple of float
        The largest negative of the difference set to zero, over ``F(K_t)``,
        and the energy K where it lies (0 and NaN where none was): the rounding
        of the difference, within `CLIP_ULPS` of ``F(K_t)``.
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
    """Return the source grid on [floor, q_max - ceiling gap].

    A uniform part between two logarithmic tails, one toward the floor and
    one toward q_max, with near-duplicate nodes at the joins dropped.
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
    _q_tail_lo = max(_tail_ext * float(q_max), tail_floor)
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
    """4-node Lagrange of (xq, yq) in linear x at x, or its derivative."""
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
    """Gamma(a) Gamma(b) / Gamma(c), from the log of each Gamma with its sign:
    never the quotient of the Gammas, whose factors overflow past an argument
    of 171.6 (here 1 + n, with ``n = 1/2 - beta_c``) although the quotient itself
    is an ordinary number; 0 at a pole of Gamma(c), as the quotient gave."""
    sign = gammasgn(a) * gammasgn(b) * gammasgn(c)
    return sign * jnp.exp(gammaln(a) + gammaln(b) - gammaln(c))


@lru_cache(maxsize=8)
def _p_rhs_kernel(abel_n: int, split_frac: float, underflow: float):
    """The jitted quadrature of rho_aux on a grid, its constants closed over
    as the pipeline's module constants are."""
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
            # interval 1, [0, split]: uniform in ln Q over [q0, split]; below q0
            # the exact piece (untruncated) or nothing
            _y0 = jnp.log(jnp.maximum(Q_grid[0], underflow))
            _y1 = jnp.log(jnp.maximum(split, underflow))
            _wy = jnp.maximum(_y1 - _y0, 0.0)
            Q1 = jnp.exp(_y0 + _wy * _t_gl)
            jac1 = _wy * Q1
            f1 = _F_at(Q1)
            kernel1 = (2.0 * jnp.maximum(Psi - Q1, 0.0)) ** n
            integral1 = jnp.dot(_w_gl, f1 * kernel1 * jac1)
            # the power-law tail A Q^-a in closed form over [0, split], the
            # remainder numerically
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
                # the tail above the floor in closed form, and the exact
                # piece below the floor
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
                    # the piece integrated by parts once more, where
                    # rho_aux's derivative is unbounded at P = 0
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
            # interval 2, [split, Psi]: Q = Psi - u^k, which removes the
            # kernel's singularity for n < 0
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

    Parameters
    ----------
    q_table, df_table : array, shape (m,)
        The inversion's table: energies and the DF (linear F; +inf on a top
        band where F leaves the range of a double).
    ddf_dq : array, shape (<= m,), or None
        The DF's derivative at the table's energies, NaN where not computed;
        None where the inversion gives none.
    df_has_neg : bool
        Whether the DF changes sign: the lookups are then in linear F.
    k_t : float
        Truncation energy; 0 for an untruncated model.
    p_center : float
        Central relative potential of the untruncated model.
    beta_c : float
        Central anisotropy; the kernel exponent is ``1/2 - beta_c`` (1/2 for
        Osipkov-Merritt).
    model_type : str
        The anisotropy's family, as the pipeline names it.
    settings : mapping
        The truncation's settings (`Settings.truncation`).
    tail : (A, a), optional
        The power-law tail A K^-a below the table (untruncated only).
    below_floor : mapping, optional
        The exact piece of the density below the table's first energy
        (untruncated only).

    Returns
    -------
    source : TruncatedSource

    Raises
    ------
    InputError
        Where the table ends below :math:`\mathcal{K}_{\rm t}`.
    ResultError
        Where the DF table is corrupt.
    AccuracyError
        Where the edge integral or the Taylor form disagree with the table
        beyond the hard limit (the model's report judges the joins below it:
        ``taylor_check``), or where the source is not finite.
    """
    order = "cubic"
    limit = HARD_LIMIT
    q_sampling = jnp.array(q_table, dtype=jnp.float64)
    df_raw = jnp.array(df_table, dtype=jnp.float64)
    zt_val = jnp.array(float(k_t), dtype=jnp.float64)
    bc_val = float(beta_c)
    model_type_val = str(model_type)

    # the table's top may carry +inf rows (a DF beyond the range of a double):
    # the finite band is the table
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

    # the lookups of F take the exact node slopes where the derivative is
    # finite, PCHIP's elsewhere
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

    # F(K + K_t) - F(K_t), the lookup kept inside the table: a negative within
    # the rounding of the difference is zero, and the largest is reported; one
    # beyond it is refused (the lowered DF would be negative there)
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

    # the edge: the integral of F' from K_t up to where the difference is
    # reliable, the Taylor form below K_switch
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
        # the switch, where the neglected cubic term is below 1e-9 of the Taylor value
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
                f"K = {_qt[_jx]:.3e} (> {limit:.0e}, the hard limit), where both "
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
                    f"(> {limit:.0e}, the hard limit): "
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

    # the auxiliary density on the source grid; the tail and the piece below
    # the floor only untruncated
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
        # below the switch the DF is the Taylor form, and rho_aux follows in closed form
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
    """The depth where the solve's inner zone ends: ``P_0 eps_64 / ode_rtol``.

    Phi's rounding, ``eps_64 P_0``, is a relative error ``eps_64 P_0 / u`` in a
    depth formed as ``P_0 - Phi``; it reaches the solve's relative tolerance at
    this depth, below which the depth is the solve's u alone.

    >>> inner_zone_depth(1.0, 1e-11)
    2.220446049250313e-05
    """
    return float(p_0) * float(np.finfo(np.float64).eps) / float(ode_rtol)
