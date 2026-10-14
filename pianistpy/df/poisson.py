"""The Poisson solve of a truncated model.

From the Poisson source rho_aux(P) on its grid (`truncated_source`), the
anisotropy's factor in r and the source profile: the potential Phi(r) and
its gradient on a grid in r, from the center to the edge where Phi reaches
the floor (the edge of a spatially finite model), or to the outer radius.

The state is [Phi, W = r dPhi/dr, u = P_0 - Phi] in s = ln r under a
relative tolerance, started from the cusp's series; a truncated model's
last rows are re-solved on a grid geometric in R_t - r, and the row at the
floor is the event's crossing. The arithmetic is the working pipeline's,
operation for operation; the settings are read by the names it uses.
"""

__all__ = [
    "EVENT_ATOL",
    "EVENT_RTOL",
    "PoissonSolve",
    "cuddeford_prefactor",
    "poisson_source",
    "profile_density",
    "run_poisson_ode",
    "source_slopes",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma as _sgamma

from ..exceptions import AccuracyError, InputError
from ..numerics.pchip import interp1d, pchip_slopes
from ..typing import Array
from .truncation import INNER_DEPTH, K_FLOOR, ROW_BINS_TOP

# the lookup's margins inside the source table, and the floor of a physical potential
PHI_CLIP_EPS = 1e-14
PHYS_FLOOR = 1e-13
# the floor events' root finder, on ln r: |delta ln r| <= EVENT_ATOL + EVENT_RTOL |ln r|
EVENT_RTOL = 1e-12
EVENT_ATOL = 1e-14


@dataclass(frozen=True)
class PoissonSolve:
    """The Poisson solve on its grid.

    Parameters
    ----------
    r : array, shape (n,)
        Radii of the rows, increasing.
    phi, dphi_dr : array, shape (n,)
        Potential and its radial derivative; +inf past the event on a
        model the edge continuation did not re-solve.
    u_r, u : array, shape (m,)
        The central coordinate u = P_0 - Phi at the pass-1 nodes.
    event_hit : bool
        Whether Phi reached the floor inside the outer radius.
    r_event : float
        Where it did; infinite where it did not.
    r_floor_request : float
        The innermost radius the grid asked for, before the start moved out
        to the source table's top.
    edge_continuation : dict or None
        The continuation's start ``R_c``, ``R_t``, ``psi_c`` and nodes per
        decade ``per_dec``; None where it did not run.
    evaluate : callable or None
        The pass-1 solve's dense interpolant of [Phi, W, u] at radii.
    evaluate_range : tuple of float
        The radii between which ``evaluate`` is the solve's: its first node
        and its last time (the event's, where it was hit).
    """

    r: Array
    phi: Array
    dphi_dr: Array
    u_r: Array
    u: Array
    event_hit: bool
    r_event: float
    r_floor_request: float
    edge_continuation: dict[str, Any] | None
    evaluate: Callable[[Any], Array] | None
    evaluate_range: tuple[float, float]


def cuddeford_prefactor(bc: float) -> float:
    """C(beta) = 2 pi^(3/2) Gamma(1 - beta) / Gamma(3/2 - beta).

    rho_aux = C INT F (2 (P - Q))^(1/2 - beta) dQ.
    """
    return float(2.0 * np.pi**1.5 * _sgamma(1.0 - bc) / _sgamma(1.5 - bc))


def source_slopes(
    q: Any, p_rhs: Any, p_0: float | None, p_s: float, order: str = "cubic"
) -> Array:
    """Return the source lookup's node slopes, d ln rho_aux / dx.

    x is the logit ln(Q / (P_0 - Q)) where P_0 is given, ln Q otherwise. A
    not-a-knot cubic spline over each unbroken run of positive rho_aux,
    split at the inversion's interior point P_s where it is finite; PCHIP's
    slopes on the nodes outside such runs.
    """
    Q = np.asarray(q, dtype=np.float64)
    pv = np.asarray(p_rhs, dtype=np.float64)
    x = np.log(np.maximum(Q, 1e-300))
    if p_0 is not None:
        x = x - np.log(np.maximum(float(p_0) - Q, 1e-300))
    y = np.log(np.maximum(pv, 1e-300))
    m = np.array(pchip_slopes(jnp.asarray(x), jnp.asarray(y)), dtype=np.float64)
    ok = (pv > 0.0) & np.isfinite(x) & np.isfinite(y)
    seg = np.cumsum(np.r_[0, np.diff(ok.astype(int)) != 0])
    if np.isfinite(p_s):
        seg = seg * 2 + (Q > p_s)
    for g in np.unique(seg[ok]):
        idx = np.flatnonzero((seg == g) & ok)
        if idx.size >= 4 and np.all(np.diff(x[idx]) > 0.0):
            m[idx] = CubicSpline(x[idx], y[idx], bc_type="not-a-knot")(x[idx], 1)
    return jnp.asarray(m)


def poisson_source(
    phi, r, ra, model_type, bc, q, p_rhs, p_0=None, u=None, m=None, order="cubic"
):
    """Return the density at (Phi, r) from the source table, and whether Phi
    is physical there. Exact vacuum below the table."""
    phi_safe = jnp.clip(phi, q[0] + PHI_CLIP_EPS, q[-1] - PHI_CLIP_EPS)
    is_physical = phi_safe > PHYS_FLOOR
    log_phi = jnp.log(jnp.maximum(phi_safe, 1e-300))
    log_q = jnp.log(jnp.maximum(q, 1e-300))
    if p_0 is not None:
        u_q = (
            (p_0 - phi_safe)
            if u is None
            else jnp.where(phi == phi_safe, u, p_0 - phi_safe)
        )
        log_phi = log_phi - jnp.log(jnp.maximum(u_q, 1e-300))
        log_q = log_q - jnp.log(jnp.maximum(p_0 - q, 1e-300))
    log_p = jnp.log(jnp.maximum(p_rhs, 1e-300))
    rhs = jnp.exp(
        interp1d(
            log_phi,
            log_q,
            log_p,
            log_p[0],
            log_p[-1],
            order=order,
            m=(None if order != "cubic" else m),
        )
    )
    rhs = jnp.where(phi < q[0], 0.0, rhs)
    # rho = rho_aux over the anisotropy's factor in r (zero beyond R_a for Type II)
    if model_type == "OM_I":
        source = 4.0 * jnp.pi * rhs / (1.0 + (r / ra) ** 2)
    elif model_type == "OM_II":
        source = jnp.where(
            r < ra, 4.0 * jnp.pi * rhs / jnp.maximum(1.0 - (r / ra) ** 2, 1e-30), 0.0
        )
    elif model_type == "COM_I":
        source = (
            cuddeford_prefactor(bc)
            * rhs
            / (r ** (2.0 * bc) * (1.0 + (r / ra) ** 2) ** (1.0 - bc))
        )
    elif model_type == "COM_II":
        source = jnp.where(
            r < ra,
            cuddeford_prefactor(bc)
            * rhs
            / (r ** (2.0 * bc) * jnp.maximum(1.0 - (r / ra) ** 2, 1e-30) ** (1.0 - bc)),
            0.0,
        )
    else:
        source = cuddeford_prefactor(bc) * rhs / (r ** (2.0 * bc))
    return source, is_physical


@partial(jax.jit, static_argnames=("model_type", "bc", "order"))
def profile_density(
    r, phi, q, p_rhs, ra, bc, model_type, p_0=None, m=None, order="cubic"
):
    """Return the density on the solve's rows: rho_aux at Phi over the
    anisotropy's factor in r, zero beyond R_a for Type II, below the table
    and at r <= 1e-13 (those rows are replaced by the inner fill).

    Raises
    ------
    InputError
        Where the cubic lookup is asked for without its slopes ``m``.
    """
    phi_safe = jnp.clip(phi, q[0] + PHI_CLIP_EPS, q[-1])
    log_phi = jnp.log(jnp.maximum(phi_safe, 1e-300))
    log_q = jnp.log(jnp.maximum(q, 1e-300))
    if p_0 is not None:
        log_phi = log_phi - jnp.log(jnp.maximum(p_0 - phi_safe, 1e-300))
        log_q = log_q - jnp.log(jnp.maximum(p_0 - q, 1e-300))
    log_p = jnp.log(jnp.maximum(p_rhs, 1e-300))
    if order == "cubic" and m is None:
        raise InputError("profile_density needs the source lookup's slopes m")
    rhs = jnp.exp(
        interp1d(
            log_phi,
            log_q,
            log_p,
            log_p[0],
            log_p[-1],
            order=order,
            m=(m if order == "cubic" else None),
        )
    )
    rhs = jnp.where(phi < q[0], 0.0, rhs)
    if model_type == "OM_I":
        rho = 4.0 * jnp.pi * rhs / (1.0 + (r / ra) ** 2)
    elif model_type == "OM_II":
        rho = jnp.where(
            r < ra, 4.0 * jnp.pi * rhs / jnp.maximum(1.0 - (r / ra) ** 2, 1e-30), 0.0
        )
    elif model_type == "COM_I":
        rho = (
            cuddeford_prefactor(bc)
            * rhs
            / (r ** (2.0 * bc) * (1.0 + (r / ra) ** 2) ** (1.0 - bc))
        )
    elif model_type == "COM_II":
        rho = (
            cuddeford_prefactor(bc)
            * rhs
            / (r ** (2.0 * bc) * jnp.maximum(1.0 - (r / ra) ** 2, 1e-30) ** (1.0 - bc))
            * (r < ra)
        )
    else:
        rho = cuddeford_prefactor(bc) * rhs / (r ** (2.0 * bc))
    return jnp.where(r > 1e-13, rho, 0.0)


def phi_ic_series(p_0, r0, q, p_rhs, ra, model_type, bc, gamma_inner):
    """Return u(r0) = P_0 - Phi(r0) of the cusp's series,
    u = source(P_0 - u, r0) r0^2 / ((3 - g)(2 - g)),
    by bisection in ln u.

    Raises
    ------
    InputError
        Where the deficit exceeds half the well, or the source vanishes at r0.
    """
    q_np = np.asarray(q, dtype=np.float64)
    p_np = np.asarray(p_rhs, dtype=np.float64)
    g = float(gamma_inner)
    pref_g = (3.0 - g) * (2.0 - g)

    def _src(phi):
        phi_c = min(max(phi, q_np[0]), q_np[-1])
        rhs = float(
            np.exp(
                np.interp(
                    np.log(max(phi_c, 1e-300)),
                    np.log(np.maximum(q_np, 1e-300)),
                    np.log(np.maximum(p_np, 1e-300)),
                )
            )
        )
        if model_type == "OM_I":
            return 4.0 * np.pi * rhs / (1.0 + (r0 / ra) ** 2)
        if model_type == "OM_II":
            return 4.0 * np.pi * rhs / max(1.0 - (r0 / ra) ** 2, 1e-30)
        if model_type == "COM_I":
            return (
                cuddeford_prefactor(bc)
                * rhs
                / (r0 ** (2.0 * bc) * (1.0 + (r0 / ra) ** 2) ** (1.0 - bc))
            )
        if model_type == "COM_II":
            return (
                cuddeford_prefactor(bc)
                * rhs
                / (r0 ** (2.0 * bc) * max(1.0 - (r0 / ra) ** 2, 1e-30) ** (1.0 - bc))
            )
        return cuddeford_prefactor(bc) * rhs / (r0 ** (2.0 * bc))

    def _resid(lu):
        u = np.exp(lu)
        return lu - np.log(max(_src(p_0 - u) * r0 * r0 / pref_g, 1e-300))

    lo, hi = np.log(1e-300), np.log(0.5 * p_0)
    if _resid(hi) < 0.0:
        raise InputError(
            f"the series deficit at r0 = {r0:.3e} exceeds half the potential "
            f"well (gamma = {g:.3f}): "
            "the central resolution asked for is not representable"
        )
    if _resid(lo) > 0.0:
        raise InputError(
            f"the Poisson source vanishes at the inner floor r0 = {r0:.3e} "
            f"(gamma = {g:.3f})"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _resid(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


class _SourceArgs(NamedTuple):
    """The model's quantities the solve's vector fields read, passed as the
    solve's arguments: the arrays and P_0, R_a traced, so one compiled solve
    serves every model of the same family, beta and grid size."""

    p_0: Any
    ra: Any
    q: Any
    p_rhs: Any
    m: Any
    bc: float
    model_type: str
    order: str


def _logr_term(s, y, args):
    R = jnp.exp(s)
    Phi, W, _uu = y
    _src, _is_phys = poisson_source(
        args.p_0 - _uu,
        R,
        args.ra,
        args.model_type,
        args.bc,
        args.q,
        args.p_rhs,
        p_0=args.p_0,
        u=_uu,
        m=args.m,
        order=args.order,
    )
    return jnp.where(_is_phys, jnp.array([W, -W - R * R * _src, -W]), jnp.zeros(3))


def _logr_term_lnphi(s, y, args):
    R = jnp.exp(s)
    L, W = y
    Phi = jnp.exp(L)
    _src, _is_phys = poisson_source(
        Phi,
        R,
        args.ra,
        args.model_type,
        args.bc,
        args.q,
        args.p_rhs,
        p_0=args.p_0,
        m=args.m,
        order=args.order,
    )
    return jnp.where(_is_phys, jnp.array([W / Phi, -W - R * R * _src]), jnp.zeros(2))


def _floor_cond(t, y, args, **kwargs):
    return y[0] - K_FLOOR


def _ln_floor_cond(t, y, args, **kwargs):
    # the root of ln Phi = ln floor, in Phi's units as the main solve's event
    # (floor (e^(ln Phi - ln floor) - 1) = Phi - floor): the root finder's
    # tolerance on the function is absolute
    return K_FLOOR * jnp.expm1(y[0] - np.log(K_FLOOR))


def run_poisson_ode(
    q: Any,
    p_rhs: Any,
    p_0: float,
    ra: float,
    bc: float,
    model_type: str,
    gamma_inner: float,
    potential: Callable[[Any], Any],
    settings: Mapping[str, Any],
    p_s: float = float("nan"),
    sentinel_q: float | None = None,
    edge_continuation: bool = True,
) -> PoissonSolve:
    """Return the Poisson solve of a model on the pipeline's grid.

    Parameters
    ----------
    q, p_rhs : array, shape (n,)
        The source grid (shifted by the boundary-value closure where the
        run applies one) and rho_aux on it.
    p_0 : float
        The model's central potential, P_0 - K_t.
    ra, bc, model_type : float, float, str
        The anisotropy.
    gamma_inner : float
        The source profile's inner slope (the cusp's series).
    potential : callable
        The source profile's potential at an array of radii, as the
        pipeline evaluates it (it places the grid's inner zone).
    settings : mapping
        The truncation's settings (`Settings.truncation`), with
        ``log10_r_max`` the run's own (raised where the profile needed it).
    p_s : float
        The inversion's interior point, where the source's slopes split.
    sentinel_q : float or None
        The top of the DF table's finite band, where it has one.
    edge_continuation : bool
        Re-solve the last rows of a truncated model.

    Returns
    -------
    solve : PoissonSolve
    """
    # imported here, not with the module: an import of optimistix (0.0.11)
    # creates an array, which starts JAX's backend before set_platform can act
    import diffrax
    import optimistix

    order = "cubic"
    psi_floor = K_FLOOR
    r_min_exp = float(settings["log10_r_min"])
    r_max_exp = float(settings["log10_r_max"])
    n_inner, n_outer = int(settings["n_rows_inner"]), int(settings["n_rows_outer"])
    f32_eps = INNER_DEPTH
    ode_rtol, ode_atol = float(settings["ode_rtol"]), float(settings["ode_atol"])

    # the radius where Phi reaches P_0 (1 - INNER_DEPTH), log-log on the
    # profile's potential
    _psi_ref_arr = jnp.array(potential(jnp.logspace(r_min_exp, -3, 10000)))
    _r_ref_arr = jnp.logspace(r_min_exp, -3, 10000)
    _log_psi_ref = jnp.log(jnp.maximum(jnp.flip(_psi_ref_arr), 1e-300))
    _log_r_ref = jnp.log(jnp.maximum(jnp.flip(_r_ref_arr), 1e-300))
    _log_psi_tgt = jnp.log(jnp.maximum(jnp.array(p_0 * (1.0 - f32_eps)), 1e-300))
    _r_split = float(jnp.exp(jnp.interp(_log_psi_tgt, _log_psi_ref, _log_r_ref)))

    # the grid: an inner zone to r_split, four overlapping outer zones, and dense points
    _eps = 1e-5 * np.pi
    _n1 = round(0.4 * n_outer)
    _n2 = round(0.3 * n_outer)
    _n3 = round(0.2 * n_outer)
    _n4 = n_outer - (_n1 + _n2 + _n3)
    _lr = float(np.log10(_r_split))
    r_ode = jnp.array(
        np.unique(
            np.concatenate(
                [
                    np.logspace(r_min_exp, _lr, n_inner, endpoint=False),
                    np.logspace(_lr, r_max_exp, _n1),
                    np.logspace(max(-1.5, _lr), min(5.0, r_max_exp) + 1 * _eps, _n2),
                    np.logspace(max(-0.5, _lr), min(3.5, r_max_exp) + 2 * _eps, _n3),
                    np.logspace(max(0.5, _lr), min(2.0, r_max_exp) + 3 * _eps, _n4),
                ]
            )
        )
    )
    _n_dense = max(
        500,
        min(int(settings["n_source_rows"]), int(settings["n_source_tail_rows"])) // 10,
    )
    _r_dense = np.logspace(
        float(np.log10(float(r_ode[0]))), float(np.log10(float(r_ode[-1]))), _n_dense
    )
    r_ode = jnp.array(np.unique(np.concatenate([np.array(r_ode), _r_dense])))

    _q_fi, _p_rhs = q, p_rhs
    _u_ic = phi_ic_series(
        float(p_0),
        float(r_ode[0]),
        _q_fi,
        _p_rhs,
        float(ra),
        model_type,
        float(bc),
        float(gamma_inner),
    )
    # the inner floor where the series' error meets the solver's absolute tolerance
    _u_floor_target = float(np.sqrt(ode_atol))
    if float(gamma_inner) < 2.0 and _u_ic > _u_floor_target:
        _A_cusp = _u_ic / float(r_ode[0]) ** (2.0 - float(gamma_inner))
        _r_min_g4 = (_u_floor_target / _A_cusp) ** (1.0 / (2.0 - float(gamma_inner)))
        _dec_ext = float(np.log10(float(r_ode[0]) / _r_min_g4))
        _dec_inner = max(float(np.log10(float(_r_split))) - float(r_min_exp), 1.0)
        _n_per_dec = float(n_inner) / _dec_inner
        _n_ext = int(np.ceil(_n_per_dec * _dec_ext))
        _r_ext = np.logspace(
            np.log10(_r_min_g4),
            float(np.log10(float(r_ode[0]))),
            _n_ext,
            endpoint=False,
        )
        r_ode = jnp.array(np.concatenate([_r_ext, np.array(r_ode)]))
        _u_ic = phi_ic_series(
            float(p_0),
            float(r_ode[0]),
            _q_fi,
            _p_rhs,
            float(ra),
            model_type,
            float(bc),
            float(gamma_inner),
        )
    # the solve starts where the source table ends: u(r0) = u_top
    r_floor_request = float(r_ode[0])
    _u_top_tab = float(p_0) - float(np.asarray(_q_fi, dtype=np.float64)[-1])
    if _u_top_tab > 0.0 and _u_ic < _u_top_tab:

        def _u_at(_r):
            return phi_ic_series(
                float(p_0),
                float(_r),
                _q_fi,
                _p_rhs,
                float(ra),
                model_type,
                float(bc),
                float(gamma_inner),
            )

        _r_np0 = np.array(r_ode, dtype=np.float64)
        _j = 1
        while _j < _r_np0.size and _u_at(_r_np0[_j]) < _u_top_tab:
            _j *= 2
        _lo, _hi = np.log(_r_np0[_j // 2]), np.log(_r_np0[min(_j, _r_np0.size - 1)])
        for _ in range(80):
            _mid = 0.5 * (_lo + _hi)
            if _u_at(np.exp(_mid)) < _u_top_tab:
                _lo = _mid
            else:
                _hi = _mid
        _r0_new = float(np.exp(_hi))
        r_ode = jnp.array(np.concatenate([[_r0_new], _r_np0[_r_np0 > _r0_new]]))
        _u_ic = _u_at(_r0_new)
    if sentinel_q is not None and float(p_0) - _u_ic > sentinel_q:
        raise InputError(
            f"the ODE starts at Phi = P0 - {_u_ic:.3e}, above the DF table's "
            f"finite band (Q = {sentinel_q:.12f}): "
            "the source there comes from a capped DF"
        )

    _s_ode = jnp.log(r_ode)
    if int(np.sum(np.diff(np.log(np.array(r_ode))) <= 0.0)) > 0:
        _s_np = np.log(np.array(r_ode))
        _keep_s = np.concatenate([[True], np.diff(_s_np) > 0.0])
        r_ode = jnp.array(np.array(r_ode)[_keep_s])
        _s_ode = jnp.log(r_ode)
    _P0_u = float(p_0)
    _m_src_logit = (
        source_slopes(_q_fi, _p_rhs, _P0_u, p_s) if order == "cubic" else None
    )
    y0 = jnp.array(
        [_P0_u - _u_ic, -(2.0 - float(gamma_inner)) * _u_ic, _u_ic], dtype=jnp.float64
    )
    _args = _SourceArgs(
        jnp.asarray(_P0_u, dtype=jnp.float64),
        jnp.asarray(ra, dtype=jnp.float64),
        jnp.asarray(_q_fi, dtype=jnp.float64),
        jnp.asarray(_p_rhs, dtype=jnp.float64),
        _m_src_logit,
        float(bc),
        model_type,
        order,
    )

    _event = diffrax.Event(
        cond_fn=_floor_cond,
        root_finder=optimistix.Newton(rtol=EVENT_RTOL, atol=EVENT_ATOL),
    )
    sol_obj = diffrax.diffeqsolve(
        diffrax.ODETerm(_logr_term),
        diffrax.Dopri5(),
        t0=_s_ode[0],
        t1=_s_ode[-1],
        dt0=1e-3,
        y0=y0,
        args=_args,
        stepsize_controller=diffrax.PIDController(rtol=ode_rtol, atol=0.0),
        saveat=diffrax.SaveAt(
            subs=[diffrax.SubSaveAt(ts=_s_ode), diffrax.SubSaveAt(t1=True)], dense=True
        ),
        event=_event,
        max_steps=200_000,
    )
    _ys_uw = sol_obj.ys[0]
    # the smaller of Phi and u is the primary variable, the other P_0 less it
    _u_prim = _ys_uw[:, 2] < _ys_uw[:, 0]
    _Phi_sv = jnp.where(_u_prim, _P0_u - _ys_uw[:, 2], _ys_uw[:, 0])
    _u_sv = jnp.where(_u_prim, _ys_uw[:, 2], _P0_u - _ys_uw[:, 0])
    _ys_ts = jnp.stack([_Phi_sv, _ys_uw[:, 1]], axis=1)
    u_r, u = np.array(r_ode, dtype=np.float64), np.array(_u_sv, dtype=np.float64)
    _ev_uw = jax.jit(jax.vmap(sol_obj.evaluate))

    def evaluate(R, _f=_ev_uw):
        return np.asarray(
            _f(jnp.log(jnp.asarray(R, dtype=jnp.float64))), dtype=np.float64
        )

    _y_t1 = np.array(sol_obj.ys[1], dtype=np.float64).reshape(-1)[-3:-1]
    _s_t1 = float(np.array(sol_obj.ts[1], dtype=np.float64).reshape(-1)[-1])
    evaluate_range = (
        float(np.exp(float(_s_ode[0]))),
        float(np.exp(_s_t1))
        if np.isfinite(_s_t1)
        else float(np.exp(float(_s_ode[-1]))),
    )
    _ev_hit = (
        bool(np.array(sol_obj.event_mask))
        and np.isfinite(_s_t1)
        and np.all(np.isfinite(_y_t1))
    )
    r_event = float(np.exp(_s_t1)) if _ev_hit else float("inf")
    sol = jnp.stack([_ys_ts[:, 0], _ys_ts[:, 1] / r_ode], axis=1)

    # a truncated model's last rows, re-solved on a grid geometric in R_t - r
    # up to the crossing
    cont = None
    if edge_continuation and _ev_hit:
        _sol_np = np.array(sol)
        _r_np = np.array(r_ode)
        _R_t = float(np.exp(_s_t1))
        _Phi_t, _W_t = float(_y_t1[0]), float(_y_t1[1])
        _a_edge = -_W_t / _R_t
        _fin = np.isfinite(_sol_np[:, 0]) & (_r_np < _R_t)
        _psi_c = float(settings["continuation_start"]) * float(p_0)
        _dlog = float(settings["profile_bin_decades"])
        _edges_c = 10.0 ** np.arange(
            np.log10(psi_floor), np.log10(ROW_BINS_TOP * float(p_0)) + _dlog, _dlog
        )
        _min_bin_c = max(
            1, int(np.rint(float(settings["profile_rows_per_decade"]) * _dlog))
        )
        _p_fin = _sol_np[_fin, 0]
        _cnt_c, _ = np.histogram(_p_fin[_p_fin >= psi_floor], bins=_edges_c)
        _under_c = np.flatnonzero(_cnt_c < _min_bin_c)
        _psi_u = float(_edges_c[_under_c[-1] + 1]) if _under_c.size else 0.0
        _psi_c = max(_psi_c, _psi_u)
        _cand = np.flatnonzero(_fin & (_sol_np[:, 0] >= _psi_c))
        if not (_a_edge * _R_t <= 100.0 * psi_floor or _cand.size == 0):
            _ic = int(_cand[-1])
            _R_c = float(_r_np[_ic])
            _d_max = _R_t - _R_c
            _d_min = min(
                max(
                    float(settings["continuation_floor_fraction"])
                    * psi_floor
                    / _a_edge,
                    1e-12 * _d_max,
                ),
                0.5 * _d_max,
            )
            _n_dec = float(np.log10(_d_max / _d_min))
            _m_sl = _fin & (np.arange(_r_np.size) >= _ic)
            with np.errstate(all="ignore"):
                _sl = np.diff(np.log(_sol_np[_m_sl, 0])) / np.diff(
                    np.log(_R_t - _r_np[_m_sl])
                )
            _sl_max = (
                float(np.nanmax(_sl)) if _sl.size and np.any(np.isfinite(_sl)) else 1.0
            )
            _per_dec = int(
                np.ceil(
                    float(settings["continuation_rows_per_decade"]) * max(1.0, _sl_max)
                )
            )
            _n_cont = int(np.ceil(_per_dec * _n_dec)) + 1
            _d_grid = _d_max * 10.0 ** (-np.linspace(0.0, _n_dec, _n_cont))
            _s_cont = np.log(_R_t - _d_grid[1:])
            # ln Phi and W, relative-only as the main solve: rtol holds Phi's
            # relative error over the decades the continuation spans; its own
            # floor event on ln Phi ends it
            _y0_c = jnp.array(
                [np.log(_sol_np[_ic, 0]), _sol_np[_ic, 1] * _R_c], dtype=jnp.float64
            )
            _sol_c = diffrax.diffeqsolve(
                diffrax.ODETerm(_logr_term_lnphi),
                diffrax.Dopri5(),
                t0=float(np.log(_R_c)),
                t1=_s_t1 + 1e-3,
                dt0=1e-4,
                y0=_y0_c,
                args=_args,
                stepsize_controller=diffrax.PIDController(rtol=ode_rtol, atol=0.0),
                saveat=diffrax.SaveAt(
                    subs=[
                        diffrax.SubSaveAt(ts=jnp.array(_s_cont, dtype=jnp.float64)),
                        diffrax.SubSaveAt(t1=True),
                    ]
                ),
                event=diffrax.Event(
                    cond_fn=_ln_floor_cond,
                    root_finder=optimistix.Newton(rtol=EVENT_RTOL, atol=EVENT_ATOL),
                ),
                max_steps=200_000,
            )
            _ys_c = np.array(_sol_c.ys[0], dtype=np.float64)
            _y_e = np.array(_sol_c.ys[1], dtype=np.float64).reshape(-1)[-2:]
            _R_e = float(
                np.exp(np.array(_sol_c.ts[1], dtype=np.float64).reshape(-1)[-1])
            )
            if not (bool(np.array(_sol_c.event_mask)) and np.all(np.isfinite(_y_e))):
                raise AccuracyError(
                    f"the edge continuation from R = {_R_c:.6e} did not reach the "
                    f"floor by R = {float(np.exp(_s_t1 + 1e-3)):.6e}"
                )
            _r_c = np.exp(_s_cont)
            _phi_c = np.exp(_ys_c[:, 0])
            _ok_c = np.isfinite(_ys_c[:, 0]) & (_phi_c > psi_floor) & (_r_c < _R_e)
            r_ode = jnp.array(np.concatenate([_r_np[: _ic + 1], _r_c[_ok_c], [_R_e]]))
            sol = jnp.array(
                np.concatenate(
                    [
                        _sol_np[: _ic + 1],
                        np.column_stack([_phi_c[_ok_c], _ys_c[_ok_c, 1] / _r_c[_ok_c]]),
                        [[psi_floor, _y_e[1] / _R_e]],
                    ]
                )
            )
            cont = {"R_c": _R_c, "R_t": _R_e, "psi_c": _psi_c, "per_dec": _per_dec}

    # the terminal row: the first row at or below the floor, interpolated onto it
    _sol_psi_np = np.array(sol[:, 0])
    _sol_dpsi_np = np.array(sol[:, 1])
    _r_ode_np_aug = np.array(r_ode)
    _valid_sol = _sol_psi_np.copy()
    _valid_sol[~np.isfinite(_sol_psi_np)] = -np.inf
    _above_floor = _valid_sol > psi_floor
    if np.any(_above_floor) and not np.all(_above_floor):
        _i_last_above = int(np.where(_above_floor)[0][-1])
        _i_first_at = _i_last_above + 1
        if _i_first_at < len(_sol_psi_np):
            _psi_a = _sol_psi_np[_i_last_above]
            _psi_b = _sol_psi_np[_i_first_at]
            _r_a = _r_ode_np_aug[_i_last_above]
            _r_b = _r_ode_np_aug[_i_first_at]
            if np.isfinite(_psi_b):
                _t_interp = float(
                    np.clip((psi_floor - _psi_a) / (_psi_b - _psi_a + 1e-300), 0.0, 1.0)
                )
                _r_exact = _r_a + _t_interp * (_r_b - _r_a)
                _dpsi_exact = _sol_dpsi_np[_i_last_above] + _t_interp * (
                    _sol_dpsi_np[_i_first_at] - _sol_dpsi_np[_i_last_above]
                )
            else:
                _i_prev = max(0, _i_last_above - 1)
                _dpsi_dr = (_sol_psi_np[_i_last_above] - _sol_psi_np[_i_prev]) / max(
                    _r_ode_np_aug[_i_last_above] - _r_ode_np_aug[_i_prev], 1e-300
                )
                _dpsi_dr = min(_dpsi_dr, -1e-30)
                _r_exact = float(
                    np.clip(_r_a + (psi_floor - _psi_a) / _dpsi_dr, _r_a, _r_a * 2.0)
                )
                _dpsi_exact = _sol_dpsi_np[_i_last_above]
            sol = jnp.concatenate(
                [
                    sol[: _i_last_above + 1],
                    jnp.array([[psi_floor, _dpsi_exact]], dtype=jnp.float64),
                ]
            )
            r_ode = jnp.array(np.append(_r_ode_np_aug[: _i_last_above + 1], _r_exact))
    return PoissonSolve(
        r=np.asarray(r_ode, dtype=np.float64),
        phi=np.asarray(sol[:, 0], dtype=np.float64),
        dphi_dr=np.asarray(sol[:, 1], dtype=np.float64),
        u_r=u_r,
        u=u,
        event_hit=bool(_ev_hit),
        r_event=r_event,
        r_floor_request=r_floor_request,
        edge_continuation=cont,
        evaluate=evaluate,
        evaluate_range=evaluate_range,
    )
