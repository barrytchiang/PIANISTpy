r"""The inversion's rows: the Cuddeford inversion operator at every energy of
the grid, and the completions that the rows take below the table's floor.

Every anisotropy model uses one operator (Notes Eq. (T:2.16)):

    F(Q) = C(n) d/dQ INT_0^Q rho_aux^(j)(P) (Q - P)^-n dP
           [+ C(n_f) rho_aux^(m_f)(0) Q^-n_f where j = m_f + 1].

Here F is the DF at the relative energy Q, P is the relative potential,
rho_aux^(j) is the j-th derivative of the auxiliary density in P, n is the
kernel exponent of the order pair (j, n), C(n) is the inversion's
prefactor, and the bracketed boundary term with the floor pair (m_f, n_f)
enters for a switched pair, j = m_f + 1.

Each row (one energy Q) is evaluated in the form that the model fixes
(Notes Section T:2.3): the local closed form at n = 0; the form integrated
by parts; the split form with the derivative outside the integral, used
for every row of a switched pair, for the rows above an interior point and
for the rows of an unbounded outer end from ``outer_from`` up; and
Gauss-Jacobi rules on either side of an interior point. The operator one
derivative higher gives dF/dQ at the energies of the rows of F.

The quadrature of a row (Notes Section T:2.3.4) has three pieces: a
Gauss-Legendre rule in ln P on [q_min, s], a Gauss-Legendre rule in P on
[s, c], and, on [c, Q], the composite map ``P = P_0 - (P_0 - Q) e^y`` with
``y = v^k`` and ``k = 1 / (1 - n)``. The composite map resolves a source
that is steep toward the central potential :math:`P_0`, and the Jacobian
of ``y = v^k`` cancels the singular factor (Q - P)^-n at P = Q, since
k (1 - n) = 1. The split points are c = 0.9 Q and
:math:`s = \max(0.1 c, q_{\rm min} (1 + 10^{-9}))`, with q_min the
table's floor. For
dF/dQ and for a switched pair, c is moved down to the nearest grid energy
at or below 0.9 Q.

The rows of `inversion_kernel` start at the table's floor. The caller adds
the completion below the floor (`floor_completion`, `complete_rows`).
"""

__all__ = [
    "KernelRules",
    "below_floor",
    "complete_rows",
    "edge_term",
    "floor_completion",
    "floor_source",
    "inversion_kernel",
    "kernel_rules",
    "method2_floor_tail",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.special import gamma
from scipy.interpolate import CubicSpline
from scipy.special import gamma as scipy_gamma
from scipy.special import hyp2f1, roots_jacobi

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..numerics.pchip import interp1d
from ..numerics.quadrature import gauss_legendre
from ..profiles import DoublePowerLaw
from ..settings import Settings
from ..typing import Array
from ..validation import TIER_BOUNDS
from .form import InversionFormRecord, tail_exponent_analytic
from .grid import TABLE_FLOOR
from .inversion import inversion_prefactor, order_pair
from .stages import AuxDerivatives

# The lowest energy P_min of the completion below the table's floor: the
# exact integral of the completion starts at P_min.
FLOOR_P_MIN = 1e-42
# The step of the one-sided differences of the source g at the floor q0, as
# a fraction of q0.
FLOOR_STEP = 1e-4
# The DF below the written floor: BELOW_PER_DECADE candidate nodes per decade
# from BELOW_FROM P_min up; the second truncation of the integral at
# BELOW_SECOND P_min; BELOW_SOURCE_PER_DECADE source nodes per decade; and,
# for each energy Q, BELOW_LOG_NODES Gauss-Legendre nodes in ln P followed by
# BELOW_END_NODES nodes in w = (Q - P)^(1 - n).
BELOW_PER_DECADE = 12
BELOW_FROM = 100.0
BELOW_SECOND = 10.0
BELOW_SOURCE_PER_DECADE = 24
BELOW_LOG_NODES = 120
BELOW_END_NODES = 60
# The fraction of Q at which the top piece of a row's integral starts.
GL_SPLIT_FRAC = 0.9
# The smallest gap P_0 - Q that the composite map of the top piece takes,
# and the smallest Q in the boundary terms Q^-n of a row.
INTEGRATOR_EPSILON = 1e-10
# The smallest energy whose logarithm is taken. A smaller energy is raised
# to UNDERFLOW_GUARD before its logarithm is taken.
UNDERFLOW_GUARD = 1e-30
# The factor by which the top piece of a row with a steep source (every row
# of a switched pair, and every row of dF/dQ) has more nodes than the top
# piece of any other row.
STEEP_FACTOR = 8


@dataclass(frozen=True)
class KernelRules:
    """Numbers of Gauss-Legendre nodes of the three pieces of a row's
    integral.

    Parameters
    ----------
    n_low : int
        Number of nodes of the low piece, the rule in ln P on [q_min, s].
    n_mid : int
        Number of nodes of the middle piece, the rule in P on [s, c].
    n_top : int
        Number of nodes of the top piece, the composite map on [c, Q], for
        a row whose source is not steep.
    """

    n_low: int
    n_mid: int
    n_top: int

    @property
    def n_top_steep(self) -> int:
        """Number of nodes of the top piece for a row with a steep source,
        STEEP_FACTOR times `n_top`."""
        return STEEP_FACTOR * self.n_top


def kernel_rules(settings: Settings) -> KernelRules:
    """Return the numbers of nodes of the three pieces for ``settings``.

    The low, middle and top pieces take 20, 60 and 20 percent of the
    inversion's ``n_nodes``, rounded, with at least 10 nodes each. The low
    piece takes at least 12 nodes per decade between the table's floor
    TABLE_FLOOR and 1."""
    abel_n = int(settings.inversion["n_nodes"])
    floor = TABLE_FLOOR
    n_low = max(10, int(round(0.20 * abel_n)))
    n_low = max(n_low, int(12 * np.log10(1.0 / max(floor, 1e-300))))
    n_mid = max(10, int(round(0.60 * abel_n)))
    n_top = max(10, int(round(0.20 * abel_n)))
    return KernelRules(n_low, n_mid, n_top)


def _unit(n: int) -> tuple[Any, Any]:
    nodes, weights = gauss_legendre(n)
    return (
        jnp.array(0.5 * (nodes + 1.0), dtype=jnp.float64),
        jnp.array(0.5 * weights, dtype=jnp.float64),
    )


def inversion_kernel(
    q: Array,
    g1: Array,
    beta_c: float,
    is_om: bool,
    rules: KernelRules,
    dm_j: float = 0.0,
    p_0: float = 1.0,
    deriv: bool = False,
    g2: Array | None = None,
    g0: Array | None = None,
    dm_jm1: float = 0.0,
    switched: bool = False,
    p_s: float = 0.0,
    s: float = 0.0,
    g1_slope: Array | None = None,
    outer_from: float = 0.0,
) -> Any:
    r"""Return the kernel's F, or dF/dQ, at every energy of the grid.

    The rows start at the table's floor q_min; the completion below the
    floor is not included.

    Parameters
    ----------
    q : array, shape (n,)
        Relative energies Q, increasing, in internal units. The first
        energy is the table's floor q_min.
    g1 : array, shape (n,)
        Source of the integral at the energies `q`, rho_aux^(j+1). Where the
        kernel exponent n is 0, the closed form returns the prefactor times
        `g0` for F and the prefactor times `g1` for dF/dQ.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}` in the prefactor. The
        caller passes 0 for an Osipkov-Merritt model.
    is_om : bool
        Whether the model is Osipkov-Merritt, whose order pair is the
        isotropic one.
    rules : KernelRules
        Numbers of nodes of the three pieces of each row.
    dm_j : float, optional
        rho_aux^(j) at the floor q_min. Default: ``0.0``.
    p_0 : float, optional
        Central potential :math:`P_0` of the composite map. Default:
        ``1.0``, the central potential in internal units.
    deriv : bool, optional
        Whether to return dF/dQ, the operator one derivative higher,
        instead of F. Default: ``False``.
    g2 : array, shape (n,), optional
        rho_aux^(j+2), the source of the top piece of dF/dQ. Default:
        ``None``, which means no such source. Every call with `deriv` true
        and n not 0 needs `g2`.
    g0 : array, shape (n,), optional
        rho_aux^(j), the source of the rows with the derivative outside the
        integral, and of the closed form at n = 0. Default: ``None``, which
        means no such source.
    dm_jm1 : float, optional
        rho_aux^(j-1) at the floor q_min, read by a switched pair. Default:
        ``0.0``.
    switched : bool, optional
        Whether the order pair is switched (j = m_f + 1, n < 0). Default:
        ``False``.
    p_s : float, optional
        Energy of the interior point of rho_aux. Default: ``0.0``, which
        means no interior point.
    s : float, optional
        Exponent of rho_aux^(j) at the interior point, in (-1, 0].
        Default: ``0.0``.
    g1_slope : array, shape (n,), optional
        Exact slopes of `g1` in P at the energies `q` (rho_aux^(j+2)), used
        by the interpolation of `g1`. Default: ``None``, which means the
        interpolation takes its own slopes.
    outer_from : float, optional
        Energy from which the rows of an unbounded outer end take the form
        with the derivative outside the integral and no boundary term at
        the floor. Ignored for a switched pair. Default: ``0.0``, which
        means no such rows.

    Returns
    -------
    rows : jax array, shape (n,)
        F, or dF/dQ where `deriv` is true, at the energies `q`, in internal
        units.

    Raises
    ------
    InputError
        If the closed form at n = 0 has no source rho_aux^(j).
    InputError
        If `switched` is true and :math:`n \ge 0`, or `switched` is false
        and :math:`n < 0`.
    InputError
        If a switched pair, the rows above an interior point or the rows of
        an unbounded outer end miss `g0`, or miss `g2` for dF/dQ.
    InputError
        If the interior point's exponent `s` lies outside (-1, 0], fewer
        than 3 energies lie on either side of the interior point, or a
        source table is not finite on one side of the interior point.
    """
    pair = order_pair(0.0 if is_om else beta_c)
    n = pair.kernel_exponent
    if n == 0.0:
        source = g1 if deriv else g0
        if source is None:
            raise InputError("The closed form at n = 0 needs rho_aux^(j).")
        return inversion_prefactor(0.0, beta_c) * source
    q_sampling = q
    if g0 is not None:
        # g0 becomes a JAX array because the rows with the derivative outside
        # the integral index g0 at energies that JAX traces.
        g0 = jnp.asarray(g0, dtype=jnp.float64)
    g_logf = bool(np.all(np.asarray(g1) > 0.0))
    m_src = None
    if g1_slope is not None:
        gs, gg, qq = (jnp.asarray(a, dtype=jnp.float64) for a in (g1_slope, g1, q))
        m_src = (
            jnp.where(gg > 0, gs * qq / jnp.where(gg > 0, gg, 1.0), 0.0)
            if g_logf
            else gs * qq
        )
    if deriv:
        gp_logf = bool(np.all(np.asarray(g2) > 0.0))
    outer_rows = bool(float(outer_from) > 0.0) and not switched
    if outer_rows:
        if g0 is None or (deriv and g2 is None):
            raise InputError(
                "The rows of an unbounded outer end need rho_aux^(j) and, for "
                "dF/dQ, rho_aux^(j+2)."
            )
        s0_logf_o = bool(np.all(np.asarray(g0) > 0.0))
    if switched != (n < 0.0):
        raise InputError(
            f"A switched pair has n < 0 and no other does; got n = {n:+.7f} "
            f"with switched = {switched}."
        )
    if switched:
        if g0 is None or (deriv and g2 is None):
            raise InputError(
                "A switched pair needs rho_aux^(j) and, for dF/dQ, rho_aux^(j+2)."
            )
        s0_logf = bool(np.all(np.asarray(g0) > 0.0))

    bp = bool(float(p_s) > 0.0)
    if bp:
        ps, ss = float(p_s), float(s)
        if not (-1.0 < ss <= 0.0):
            raise InputError(
                f"The interior point's exponent {ss:+.4f} lies outside (-1, 0]: "
                "rho_aux^(j) is not integrable there."
            )
        if g0 is None or (deriv and g2 is None):
            raise InputError(
                "The rows above an interior point need rho_aux^(j) and, for "
                "dF/dQ, rho_aux^(j+2)."
            )
        qn_s = np.asarray(q, dtype=np.float64)
        ml_s, mr_s = qn_s < ps, qn_s > ps
        if int(ml_s.sum()) < 3 or int(mr_s.sum()) < 3:
            raise InputError(
                "The source tables need at least 3 nodes on each side of the "
                "interior point."
            )

        def t2s_end(d, h):
            i = np.argsort(d)[:3]
            dd, hh = d[i], h[i]
            return float(
                sum(
                    hh[k]
                    * np.prod(
                        [(-dd[jj]) / (dd[k] - dd[jj]) for jj in range(3) if jj != k]
                    )
                    for k in range(3)
                )
            )

        dl_s = ps - qn_s[ml_s]
        dr_s = qn_s[mr_s] - ps

        def e_ord(k):
            return (ss - k) if ss != 0.0 else 0.0

        def h_side(grid, k, left):
            g = np.asarray(grid, dtype=np.float64)[ml_s if left else mr_s]
            h = g * (dl_s if left else dr_s) ** (-e_ord(k))
            if not np.all(np.isfinite(h)):
                raise InputError(
                    f"The source table of order j+{k} is not finite "
                    f"{'left' if left else 'right'} of the interior point."
                )
            if left:
                x, f = np.append(qn_s[ml_s], ps), np.append(h, t2s_end(dl_s, h))
            else:
                x, f = np.insert(qn_s[mr_s], 0, ps), np.insert(h, 0, t2s_end(dr_s, h))
            # The spline's slopes are linear in the column f. The slopes are
            # therefore formed on f / 2^e, with 2^e the largest power of 2 not
            # above max |f|, and then multiplied by 2^e. A column near the
            # largest double then keeps finite secants. Division and
            # multiplication by a power of 2 are exact for normal doubles, so
            # the slopes of a column without overflow are the unscaled slopes
            # to the bit.
            top = float(np.max(np.abs(f)))
            scale = float(2.0 ** np.floor(np.log2(top))) if top > 0.0 else 1.0
            m = CubicSpline(np.log(x), f / scale).derivative()(np.log(x)) * scale
            return tuple(jnp.array(a, dtype=jnp.float64) for a in (x, f, m))

        xl0, fl0, ml0 = h_side(g0, 0, True)
        xr0, fr0, mr0 = h_side(g0, 0, False)
        xr1, fr1, mr1 = h_side(g1, 1, False)
        if deriv:
            xr2, fr2, mr2 = h_side(g2, 2, False)
        n_t2s = (rules.n_low + rules.n_mid) // 2
        xj, wj = roots_jacobi(n_t2s, ss, 0.0)
        xjl_s, wjl_s = (
            jnp.array(xj, dtype=jnp.float64),
            jnp.array(wj, dtype=jnp.float64),
        )
        xj, wj = roots_jacobi(n_t2s, 0.0, ss)
        xjr_s, wjr_s = (
            jnp.array(xj, dtype=jnp.float64),
            jnp.array(wj, dtype=jnp.float64),
        )
        ys_s = float(np.log(ps))

    t_s1, w_s1 = _unit(rules.n_low)
    t_s2, w_s2 = _unit(rules.n_mid)
    steep_top = bool(deriv or switched)
    t_s3, w_s3 = _unit(rules.n_top_steep if steep_top else rules.n_top)

    def _interp(x, xp, fp, logf, m=None):
        return interp1d(x, xp, fp, 0.0, 0.0, order="cubic", logx=True, logf=logf, m=m)

    def single_row(Q):
        psi_cut = GL_SPLIT_FRAC * Q
        if deriv or switched:
            psi_cut = q_sampling[
                jnp.maximum(jnp.searchsorted(q_sampling, psi_cut, side="right") - 1, 0)
            ]
        epsilon = INTEGRATOR_EPSILON

        def klo(d):
            d = jnp.maximum(d, 1e-15 * Q)
            return (-n) * d ** (-n - 1.0) if deriv else d ** (-n)

        q_min = q_sampling[0]
        eps_log = UNDERFLOW_GUARD
        split_sub = jnp.maximum(0.1 * psi_cut, q_min * (1.0 + 1e-9))
        split_sub = jnp.minimum(split_sub, Q)
        psi_cut = jnp.clip(psi_cut, split_sub, Q)
        log_a = jnp.log(jnp.maximum(q_min, eps_log))
        log_b = jnp.log(jnp.maximum(split_sub, eps_log))
        log_p_s1 = log_a + (log_b - log_a) * t_s1
        p_s1 = jnp.exp(log_p_s1)
        jac_s1 = (log_b - log_a) * p_s1
        width_s2 = psi_cut - split_sub
        p_s2 = split_sub + width_s2 * t_s2
        jac_s2 = width_s2

        def g_top(P):
            if deriv:
                return _interp(P, q_sampling, g2, gp_logf)
            return _interp(P, q_sampling, g1, g_logf, m_src)

        k_hi = 1.0 / jnp.maximum(1.0 - n, 0.5)

        def top_piece(psi_lo, src_fn):
            gap = jnp.maximum(p_0 - Q, epsilon)
            big_y = jnp.log1p(jnp.maximum(Q - psi_lo, 0.0) / gap)
            big_v = big_y ** (1.0 / k_hi)
            v = big_v * t_s3
            vs = jnp.where(big_v > 0.0, v, 1.0)
            y = vs**k_hi
            p_hi = p_0 - gap * jnp.exp(y)
            g_hi = src_fn(p_hi)
            kern_hi = (gap * jnp.expm1(y)) ** (-n)
            jac_hi = gap * jnp.exp(y) * k_hi * big_v * vs ** (k_hi - 1.0)
            return jnp.where(big_v > 0.0, jnp.dot(w_s3, kern_hi * g_hi * jac_hi), 0.0)

        prefactor = (2.0**beta_c) / (
            (2.0 * jnp.pi) ** 1.5 * gamma(1.0 - n) * gamma(1.0 - beta_c)
        )

        if not switched:
            g_s1 = _interp(p_s1, q_sampling, g1, g_logf, m_src)
            kern_s1 = klo(Q - p_s1)
            integral_s1 = jnp.dot(w_s1, kern_s1 * g_s1 * jac_s1)
            g_s2 = _interp(p_s2, q_sampling, g1, g_logf, m_src)
            kern_s2 = klo(Q - p_s2)
            integral_s2 = jnp.dot(w_s2, kern_s2 * g_s2 * jac_s2)
            integral_lo = integral_s1 + integral_s2
            integral_hi = top_piece(psi_cut, g_top)
            boundary_term = prefactor * (jnp.maximum(Q, epsilon) ** (-n)) * dm_j
            if deriv:
                gc = _interp(psi_cut, q_sampling, g1, g_logf)
                gap_c = Q - psi_cut
                boundary_term = prefactor * (-n) * jnp.maximum(Q, epsilon) ** (
                    -n - 1.0
                ) * dm_j + jnp.where(
                    gap_c > 0.0,
                    prefactor * gc * jnp.maximum(gap_c, 1e-300) ** (-n),
                    jnp.nan,
                )
            f_rows = prefactor * (integral_lo + integral_hi) + boundary_term
            if outer_rows:
                co = q_sampling[
                    jnp.maximum(
                        jnp.searchsorted(q_sampling, GL_SPLIT_FRAC * Q, side="right")
                        - 1,
                        0,
                    )
                ]
                co = jnp.clip(co, q_min, Q)
                sso = jnp.minimum(jnp.maximum(0.1 * co, q_min * (1.0 + 1e-9)), co)
                lbo = jnp.log(jnp.maximum(sso, eps_log))
                p1o = jnp.exp(log_a + (lbo - log_a) * t_s1)
                j1o = (lbo - log_a) * p1o
                p2o = sso + (co - sso) * t_s2
                j2o = co - sso

                def s0o(P):
                    return _interp(P, q_sampling, g0, s0_logf_o)

                if deriv:

                    def kdo(d):
                        return n * (n + 1.0) * jnp.maximum(d, 1e-15 * Q) ** (-n - 2.0)

                else:

                    def kdo(d):
                        return (-n) * jnp.maximum(d, 1e-15 * Q) ** (-n - 1.0)

                lowo = jnp.dot(w_s1, kdo(Q - p1o) * s0o(p1o) * j1o) + jnp.dot(
                    w_s2, kdo(Q - p2o) * s0o(p2o) * j2o
                )
                gapo = jnp.maximum(Q - co, 1e-300)
                if deriv:
                    s1o = _interp(co, q_sampling, g1, g_logf)
                    spo = -n * s0o(co) * gapo ** (-n - 1.0) + s1o * gapo ** (-n)
                else:
                    spo = s0o(co) * gapo ** (-n)
                fo = prefactor * (lowo + spo + top_piece(co, g_top))
                f_rows = jnp.where(Q >= outer_from, fo, f_rows)
            if not bp:
                return f_rows

        qg = jnp.maximum(Q, epsilon)
        if deriv:

            def kd(d):
                return n * (n + 1.0) * jnp.maximum(d, 1e-15 * Q) ** (-n - 2.0)

        else:

            def kd(d):
                return (-n) * jnp.maximum(d, 1e-15 * Q) ** (-n - 1.0)

        if switched:
            closure = (
                n * (n + 1.0) * dm_jm1 * qg ** (-n - 2.0)
                if deriv
                else (-n) * dm_jm1 * qg ** (-n - 1.0)
            )
        else:
            rq = jnp.minimum(q_min / qg, 0.5)
            closure = (
                (-n)
                * dm_j
                * qg ** (-n - 1.0)
                * (-jnp.expm1((-n - 1.0) * jnp.log1p(-rq)))
                if deriv
                else dm_j * qg ** (-n) * (-jnp.expm1(-n * jnp.log1p(-rq)))
            )
            if outer_rows:
                closure = jnp.where(Q >= outer_from, 0.0, closure)

        def outside(lower, s0c, s1c, gap, top):
            gg = jnp.maximum(gap, 1e-300)
            if deriv:
                sp = jnp.where(
                    gap > 0.0, -n * s0c * gg ** (-n - 1.0) + s1c * gg ** (-n), jnp.nan
                )
            else:
                sp = jnp.where(gap > 0.0, s0c * gg ** (-n), 0.0)
            return prefactor * (lower + sp + top + closure)

        if switched:

            def s0f(P):
                return _interp(P, q_sampling, g0, s0_logf)

            low = jnp.dot(w_s1, kd(Q - p_s1) * s0f(p_s1) * jac_s1) + jnp.dot(
                w_s2, kd(Q - p_s2) * s0f(p_s2) * jac_s2
            )
            f_rows = outside(
                low,
                s0f(psi_cut),
                _interp(psi_cut, q_sampling, g1, g_logf),
                Q - psi_cut,
                top_piece(psi_cut, g_top),
            )
            if not bp:
                return f_rows

        act_s = ps < Q
        c = jnp.where(act_s, jnp.maximum(GL_SPLIT_FRAC * Q, 0.5 * (ps + Q)), ps * np.e)
        kc = jnp.searchsorted(xr0[1:], c, side="right") - 1
        cn = xr0[1:][jnp.maximum(kc, 0)]
        c = jnp.where((kc >= 0) & (cn - ps >= 0.5 * (c - ps)), cn, c)
        yb_s = jnp.log(c)
        dl = 0.5 * (ys_s - log_a) * (1.0 - xjl_s)
        pl = jnp.exp(ys_s - dl)
        rl = (ps * (-jnp.expm1(-dl)) / dl) ** ss
        hl = interp1d(
            pl, xl0, fl0, 0.0, 0.0, order="cubic", logx=True, logf=False, m=ml0
        )
        dr = 0.5 * (yb_s - ys_s) * (1.0 + xjr_s)
        pr = jnp.exp(ys_s + dr)
        rr = (ps * jnp.expm1(dr) / dr) ** ss
        hr = interp1d(
            pr, xr0, fr0, 0.0, 0.0, order="cubic", logx=True, logf=False, m=mr0
        )
        lower = (0.5 * (ys_s - log_a)) ** (1.0 + ss) * jnp.dot(
            wjl_s, kd(Q - pl) * hl * rl * pl
        ) + (0.5 * (yb_s - ys_s)) ** (1.0 + ss) * jnp.dot(
            wjr_s, kd(Q - pr) * hr * rr * pr
        )

        def reg(P, x, f, m, k):
            return interp1d(
                P, x, f, 0.0, 0.0, order="cubic", logx=True, logf=False, m=m
            ) * jnp.maximum(P - ps, 1e-300) ** e_ord(k)

        if deriv:
            topk = top_piece(c, lambda P: reg(P, xr2, fr2, mr2, 2))
        else:
            topk = top_piece(c, lambda P: reg(P, xr1, fr1, mr1, 1))
        f_kink = outside(
            lower,
            reg(c, xr0, fr0, mr0, 0),
            reg(c, xr1, fr1, mr1, 1),
            jnp.where(act_s, Q - c, 1.0),
            topk,
        )
        return jnp.where(act_s, f_kink, f_rows)

    return vmap(single_row)(q)


def edge_term(q: Array, p_e: float, amplitude: float, beta_c: float) -> Array:
    r"""Return the closed-form DF of the edge power law removed from a Type II
    model's source.

    Above the edge energy :math:`P_{\rm e}` (`p_e`), the DF is

    .. math::

        F(Q) = \frac{A \, 2^{\beta_{\rm c}} (1 - \beta_{\rm c})}
        {2^{3/2} \pi^2} (Q - P_{\rm e})^{-1/2},

    with :math:`A` the `amplitude` and :math:`\beta_{\rm c}` the central
    anisotropy `beta_c`; the DF diverges as :math:`(Q - P_{\rm e})^{-1/2}`
    at :math:`P_{\rm e}`. At and below :math:`P_{\rm e}`, where the model is
    truncated, the DF is 0."""
    dq = np.asarray(q, dtype=np.float64) - float(p_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            float(amplitude)
            * 2.0 ** float(beta_c)
            * (1.0 - float(beta_c))
            / (2.0**1.5 * np.pi**2)
            * np.where(dq > 0, dq**-0.5, 0.0)
        )


def floor_source(
    aux: AuxDerivatives,
    record: InversionFormRecord,
    profile: DoublePowerLaw,
    anisotropy: Anisotropy,
    q0: float,
    n_nodes: int,
) -> dict[str, Any] | None:
    """Return the quadrature rule and the exact source of the completion
    below the table's floor q0, as `floor_completion` reads them, or None
    where the model has no completion (Notes Section T:2.3.2).

    The rule's nodes are Gauss-Legendre in ln P on [P_min, q0/10], with
    P_min = FLOOR_P_MIN, and Gauss-Legendre in w = (q0 - P)^(1-n) on
    [q0/10, q0]. The distance q0 - P of each node of the rule in w is taken
    exactly from the map in w. In the floor pair (m, n), the source is
    g = rho_aux^(m+1). The returned values of g at the nodes and at q0, and
    of rho_aux^(m) at P_min and at q0, are exact derivatives. The returned
    first and second derivatives of g at q0 are one-sided differences of
    the exact g with step FLOOR_STEP q0. Where
    rho_aux^(m) is unbounded at P = 0 (an unbounded outer end), the
    returned values also include rho_aux^(m) at the nodes and
    rho_aux^(m-1) at P_min. A model has the completion where the exponent
    of its DF's tail is below 1, and also at an unbounded outer end.

    Parameters
    ----------
    aux : AuxDerivatives
        Exact derivatives of the component's auxiliary density.
    record : InversionFormRecord
        The component's form, which gives the floor pair and the kind of
        outer end.
    profile : DoublePowerLaw
        The component's profile.
    anisotropy : Anisotropy
        The component's anisotropy.
    q0 : float
        The table's floor, the grid's first energy, in internal units.
    n_nodes : int
        Number of nodes in ln P. The rule in w takes a third of `n_nodes`,
        and at least 40 nodes.

    Returns
    -------
    source : dict or None
        The rule and the source, with the keys that `floor_completion`
        documents for its argument ``src``, and also ``"p"`` (the nodes'
        energies P), ``"m"`` (the floor pair's order) and ``"half_int"``
        (whether the kernel exponent is 0). None where the model has no
        completion.

    Raises
    ------
    ResultError
        If the source g, or rho_aux^(m) at an unbounded outer end, is not
        finite on the nodes.
    """
    a = tail_exponent_analytic(profile, anisotropy)
    unbounded = record.outer_end == "unbounded"
    if not (a < 1.0 or unbounded):
        return None
    m, n = int(record.order_m), float(record.order_n)
    beta_c = 0.0 if anisotropy.model == "om" else float(anisotropy.beta_c)
    xi, w_l = gauss_legendre(int(n_nodes))
    ya, yb = np.log(float(FLOOR_P_MIN)), np.log(q0 / 10.0)
    y = 0.5 * (yb - ya) * xi + 0.5 * (yb + ya)
    p_a = np.exp(y)
    w_a = 0.5 * (yb - ya) * w_l * p_a
    xib, w_b = gauss_legendre(max(40, int(n_nodes) // 3))
    om = 1.0 - n
    w_max = (q0 - q0 / 10.0) ** om
    w_n = 0.5 * w_max * (xib + 1.0)
    dw_b = w_n ** (1.0 / om)
    p_b = q0 - dw_b
    w_bb = 0.5 * w_max * w_b * (1.0 / om) * w_n ** (1.0 / om - 1.0)
    p = np.concatenate([p_a, p_b[::-1]])
    w = np.concatenate([w_a, w_bb[::-1]])
    d = np.concatenate([q0 - p_a, dw_b[::-1]])
    g = np.asarray(aux.at(p, m + 1), dtype=np.float64)
    if not np.all(np.isfinite(g)):
        raise ResultError("The source below the floor is not finite on its nodes.")
    pref = (2.0**beta_c) / (
        (2.0 * np.pi) ** 1.5 * scipy_gamma(1.0 - n) * scipy_gamma(1.0 - beta_c)
    )
    gq0 = float(aux.at(np.array([q0]), m + 1)[0])
    h = FLOOR_STEP * q0
    gm1 = float(aux.at(np.array([q0 - h]), m + 1)[0])
    gm2 = float(aux.at(np.array([q0 - 2 * h]), m + 1)[0])
    r = rm1min = None
    if unbounded:
        r = np.asarray(aux.at(p, m), dtype=np.float64)
        if not np.all(np.isfinite(r)):
            raise ResultError("The source below the floor is not finite on its nodes.")
        rm1min = float(aux.at(np.array([float(FLOOR_P_MIN)]), m - 1)[0])
    return {
        "q0": float(q0),
        "p_min": float(FLOOR_P_MIN),
        "n": n,
        "pref": float(pref),
        "dq0": float(aux.at(np.array([q0]), m)[0]),
        "dmm": float(aux.at(np.array([float(FLOOR_P_MIN)]), m)[0]),
        "gq0": gq0,
        "gpq0": (3.0 * gq0 - 4.0 * gm1 + gm2) / (2.0 * h),
        "gppq0": (gq0 - 2.0 * gm1 + gm2) / (h**2),
        "w": np.asarray(w, dtype=np.float64),
        "d": np.asarray(d, dtype=np.float64),
        "p": np.asarray(p, dtype=np.float64),
        "m": m,
        "g": g,
        "outer": unbounded,
        "outer_from": float(record.outer_from),
        "r": r,
        "rm1min": rm1min,
        "half_int": bool(record.kernel_exponent == 0.0),
    }


def _below_nodes(
    qs: Array, p_lo: float, n: float
) -> tuple[Array, Array, Array, list[int], Array]:
    """Return the quadrature nodes of `below_integral` for every energy Q in
    ``qs``, concatenated over Q.

    For each Q the nodes are BELOW_LOG_NODES Gauss-Legendre nodes in ln P
    on [p_lo, Q/10] and BELOW_END_NODES Gauss-Legendre nodes in
    w = (Q - P)^(1-n) on [Q/10, Q]. The returned arrays are the nodes P,
    the weights, the exact distances Q - P, the number of nodes of each Q,
    and the weights times the kernel (Q - P)^-n."""
    xa, wa = np.polynomial.legendre.leggauss(BELOW_LOG_NODES)
    xb, wb = np.polynomial.legendre.leggauss(BELOW_END_NODES)
    om = 1.0 - n
    p_all, w_all, d_all, seg, k_all = [], [], [], [], []
    for qq in qs:
        ya, yb = np.log(float(p_lo)), np.log(qq / 10.0)
        p_a = np.exp(0.5 * (yb - ya) * xa + 0.5 * (yb + ya))
        w_a = 0.5 * (yb - ya) * wa * p_a
        wmx = (qq - qq / 10.0) ** om
        wn = 0.5 * wmx * (xb + 1.0)
        dq_b = wn ** (1.0 / om)
        w_b = 0.5 * wmx * wb * (1.0 / om) * wn ** (1.0 / om - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            kw_b = w_b * dq_b ** (-n)
        # Where Q - P = w^(1/(1-n)) underflows to 0, the weight times the
        # kernel takes its exact value in w,
        # wmx w_i / (2 (1 - n)), instead of the product 0 times infinity.
        under = dq_b == 0.0
        kw_b[under] = 0.5 * wmx * wb[under] / om
        p_all.append(np.concatenate([p_a, qq - dq_b]))
        w_all.append(np.concatenate([w_a, w_b]))
        d_all.append(np.concatenate([qq - p_a, dq_b]))
        seg.append(p_a.size + dq_b.size)
        k_all.append(np.concatenate([w_a * (qq - p_a) ** (-n), kw_b]))
    return (
        np.concatenate(p_all),
        np.concatenate(w_all),
        np.concatenate(d_all),
        seg,
        np.concatenate(k_all),
    )


def below_integral(
    qs: Array,
    p_lo: float,
    dm_lo: float,
    n: float,
    pref: float,
    g_i: Any,
    r_i: Any = None,
) -> Array:
    """Return F below the written floor at the energies ``qs`` from the
    inversion identity truncated at the energy p_lo (`below_floor`).

    Without ``r_i``, F(Q) = pref [INT_{p_lo}^{Q} g (Q - P)^-n dP +
    dm_lo (Q - p_lo)^-n], with g = ``g_i`` and dm_lo = rho_aux^(m)(p_lo).
    With ``r_i``, the function rho_aux^(m) at an unbounded outer end, the
    part of the integral on [p_lo, Q/10] is integrated by parts once more,
    with ``r_i`` as its source, and the term in dm_lo is not used. At fixed
    n and pref, F is linear in the
    functions ``g_i`` and ``r_i`` of P and in ``dm_lo``."""
    qs = np.atleast_1d(np.asarray(qs, dtype=np.float64))
    n_a = BELOW_LOG_NODES
    pc, wc, dc, seg, wk = _below_nodes(qs, p_lo, n)
    gc = g_i(pc)
    out = np.empty(qs.size)
    o = 0
    for iq, qq in enumerate(qs):
        sl = slice(o, o + seg[iq])
        o += seg[iq]
        if r_i is None:
            out[iq] = pref * (
                float(np.dot(wk[sl], gc[sl])) + dm_lo * (qq - float(p_lo)) ** (-n)
            )
        else:
            la = slice(sl.start, sl.start + n_a)
            lb = slice(sl.start + n_a, sl.stop)
            out[iq] = pref * (
                float(r_i(np.array([0.1 * qq]))[0]) * (0.9 * qq) ** (-n)
                - n * float(np.dot(wc[la] * dc[la] ** (-n - 1.0), r_i(pc[la])))
                + float(np.dot(wk[lb], gc[lb]))
            )
    return out


def below_floor(
    aux: AuxDerivatives, floor: Mapping[str, Any], k_written: float
) -> dict[str, Any]:
    r"""Return the DF below the written floor from the exact source of the
    completion (Notes Section T:2.3.2).

    F is the inversion identity truncated at P_min,
    F(Q) = pref [INT_{P_min}^{Q} g (Q - P)^-n dP
    + rho_aux^(m)(P_min) (Q - P_min)^-n], with g = rho_aux^(m+1) in the
    floor pair (m, n). This F is the whole DF, not a correction. F is
    evaluated on a grid uniform in ln Q that ends on the table's first
    written node.

    Each Q has its own rule: Gauss-Legendre in ln P on [P_min, Q/10] and in
    w = (Q - P)^(1-n) on [Q/10, Q]. The source g is evaluated once on a
    grid uniform in ln P, with BELOW_SOURCE_PER_DECADE nodes per decade from
    P_min to the table's first written node, and is read between its nodes by a cubic
    spline in (ln P, ln |g|). The spline is checked against the exact
    source between the nodes, in value and in slope. Where rho_aux^(m) is
    unbounded at P = 0, the part on [P_min, Q/10] is integrated by parts
    once more, with rho_aux^(m) as source, read by its own spline. The
    lowest node of the returned grid is the first candidate node above
    which the identity truncated at P_min and the identity truncated at
    10 P_min agree to :math:`10^{-6}`. dF/dQ is the slope of the cubic
    spline in (ln Q, ln F) through F.

    Parameters
    ----------
    aux : AuxDerivatives
        Exact derivatives of the component's auxiliary density.
    floor : mapping
        The completion's source (`floor_source`), of which the keys
        ``p_min``, ``n``, ``m``, ``pref``, ``dmm`` and ``outer`` are read.
    k_written : float
        Energy of the table's first written node, in internal units.

    Returns
    -------
    below : dict
        ``K``, ``F`` and ``dFdK`` on the returned grid; ``eval``, a function
        that returns F and dF/dQ at any Q below the written floor; the
        source's splines ``g_i`` and ``r_i`` (None without an unbounded
        outer end) and their grid ``p_source``; and the scalars ``P_min``,
        ``n``, ``m``, ``K0_written``, ``source_interp_rel_err`` (the
        splines' largest relative miss of the exact source),
        ``bottom_discarded_decades`` and ``truncation_agreement_rel``.
        Where F of either truncation is not positive at a candidate node,
        the dict is instead ``{"not_positive": True, "F_min": ...}``, with
        ``F_min`` the least F of the two truncations at the candidate nodes.

    Raises
    ------
    ResultError
        If the source g, or rho_aux^(m) at an unbounded outer end, changes
        sign below the floor.
    ResultError
        If a spline of the source misses the exact source by
        :math:`10^{-6}` or more.
    ResultError
        If F of either truncation is not finite at a candidate node.
    ResultError
        If the two truncations agree to :math:`10^{-6}` on less than one
        decade below the written floor.
    ResultError
        If ``eval`` finds F not finite or not positive.
    """
    p_min, n, m = float(floor["p_min"]), float(floor["n"]), int(floor["m"])
    pref, dmm = float(floor["pref"]), float(floor["dmm"])
    outer = bool(floor["outer"])
    k0w = float(k_written)
    n_c = int(round(BELOW_PER_DECADE * np.log10(k0w / (BELOW_FROM * p_min)))) + 1
    q_c = np.exp(np.linspace(np.log(BELOW_FROM * p_min), np.log(k0w), n_c))
    n_d = int(round(BELOW_SOURCE_PER_DECADE * np.log10(k0w / p_min))) + 1
    p_d = np.exp(np.linspace(np.log(p_min), np.log(k0w), n_d))
    g_d = np.asarray(aux.at(p_d, m + 1), dtype=np.float64)
    if not (np.all(g_d > 0.0) or np.all(g_d < 0.0)):
        raise ResultError(
            "The exact source changes sign below the floor: no cubic in ln |g| "
            "describes it."
        )
    sg = 1.0 if g_d[0] > 0.0 else -1.0
    cs_g = CubicSpline(np.log(p_d), np.log(sg * g_d))

    def g_i(pp: Array) -> Array:
        return sg * np.exp(cs_g(np.log(pp)))

    p_chk = np.sqrt(p_d[:-1] * p_d[1:])[:: max(1, (n_d - 1) // 12)]
    g_chk = np.asarray(aux.at(p_chk, m + 1), dtype=np.float64)
    gp_chk = np.asarray(aux.at(p_chk, m + 2), dtype=np.float64)
    h_ln = float(np.log(p_d[1] / p_d[0]))
    slope_err = float(
        np.max(np.abs(cs_g(np.log(p_chk), 1) - p_chk * gp_chk / g_chk)) * h_ln
    )
    interp_err = float(max(np.max(np.abs(g_i(p_chk) / g_chk - 1.0)), slope_err))
    r_i = None
    if outer:
        r_d = np.asarray(aux.at(p_d, m), dtype=np.float64)
        if not (np.all(r_d > 0.0) or np.all(r_d < 0.0)):
            raise ResultError(
                "rho_aux^(m) changes sign below the floor: no cubic in "
                "ln |rho_aux^(m)| describes it."
            )
        sr = 1.0 if r_d[0] > 0.0 else -1.0
        cs_r = CubicSpline(np.log(p_d), np.log(sr * r_d))

        def r_i(pp: Array) -> Array:
            return sr * np.exp(cs_r(np.log(pp)))

        r_chk = np.asarray(aux.at(p_chk, m), dtype=np.float64)
        interp_err = float(max(interp_err, np.max(np.abs(r_i(p_chk) / r_chk - 1.0))))
    if not (interp_err < TIER_BOUNDS[0]):
        raise ResultError(
            f"The source cubic misses the exact source by {interp_err:.2e} below "
            f"the floor, above 0.000001: the exact source is not smooth on its grid."
        )

    def f_of(qs: Array, p_lo: float = p_min, dm_lo: float = dmm) -> Array:
        return below_integral(qs, p_lo, dm_lo, n, pref, g_i, r_i)

    p10 = BELOW_SECOND * p_min
    dm10 = float(aux.at(np.array([p10]), m)[0])
    f1, f2 = f_of(q_c), f_of(q_c, p10, dm10)
    for f_t, at in ((f1, "P_min"), (f2, f"{BELOW_SECOND:g} P_min")):
        bad_f = np.flatnonzero(~np.isfinite(f_t))
        if bad_f.size:
            raise ResultError(
                f"F below the floor, truncated at {at}, is not finite at "
                f"{bad_f.size} of {q_c.size} candidate nodes: F = "
                f"{f_t[bad_f[0]]!r} at Q = {q_c[bad_f[0]]:.6e}."
            )
    if not (np.all(f1 > 0.0) and np.all(f2 > 0.0)):
        return {"not_positive": True, "F_min": float(min(np.min(f1), np.min(f2)))}
    trc = np.abs(f2 / f1 - 1.0)
    bad = np.nonzero(trc >= TIER_BOUNDS[0])[0]
    i0 = int(bad[-1]) + 1 if bad.size else 0
    if i0 >= n_c - BELOW_PER_DECADE:
        raise ResultError(
            f"The two truncations of the identity (P_min = {p_min:.0e} and "
            f"{BELOW_SECOND:g} P_min) agree to 0.000001 on less than one decade "
            f"below the written floor (worst {float(np.max(trc)):.2e}, last "
            f"disagreement at Q = {float(q_c[bad[-1]]):.2e})."
        )
    q_b = q_c[i0:]

    def evaluate(
        qs: Array, q_master: Array = q_b, f_master: Array | None = None
    ) -> tuple[Array, Array]:
        qs = np.atleast_1d(np.asarray(qs, dtype=np.float64))
        q_all = np.unique(np.concatenate([q_master, qs]))
        if f_master is not None and np.array_equal(q_all, q_master):
            # The requested energies are all master nodes, whose F was already
            # computed, so F is not computed again.
            f_all = f_master
        else:
            f_all = f_of(q_all)
        if not (np.all(np.isfinite(f_all)) and np.all(f_all > 0.0)):
            raise ResultError("F below the floor is not finite and positive.")
        cs = CubicSpline(np.log(q_all), np.log(f_all))
        f = np.exp(cs(np.log(qs)))
        return f, f * cs(np.log(qs), 1) / qs

    f_b, df_b = evaluate(q_b, f_master=f1[i0:])
    return {
        "K": q_b,
        "F": f_b,
        "dFdK": df_b,
        "eval": evaluate,
        "P_min": p_min,
        "n": n,
        "m": m,
        "K0_written": k0w,
        "source_interp_rel_err": interp_err,
        "bottom_discarded_decades": float(np.log10(q_b[0] / q_c[0])),
        "truncation_agreement_rel": float(np.max(trc[i0:])),
        "p_source": p_d,
        "g_i": g_i,
        "r_i": r_i,
    }


def floor_completion(
    q: Array, src: Mapping[str, Any], deriv: bool = True
) -> tuple[Array, Array | None]:
    """Return the correction that adds to the kernel's rows the part of the
    integral below the table's floor q0, and the correction's derivative in
    Q (Notes Section T:2.3.2).

    The kernel integrates from q0 and carries the boundary term
    rho_aux^(m)(q0) Q^-n. The exact form from P_min adds the correction
    pref [INT_{P_min}^{q0} g (Q - P)^-n dP + rho_aux^(m)(P_min)
    (Q - P_min)^-n - rho_aux^(m)(q0) Q^-n], in the floor pair (m, n), with
    g = rho_aux^(m+1). At the nodes with d = q0 - P below 0.01 q0, the
    difference g - g(q0) is replaced by its Taylor form
    -g'(q0) d + g''(q0) d^2 / 2 in the exact d. Where rho_aux^(m) is
    unbounded at P = 0, the correction is integrated by parts once more,
    with rho_aux^(m) as source, and the rows from ``outer_from`` up take
    the general form with no subtraction at q0.

    Parameters
    ----------
    q : array, shape (n,)
        Relative energies Q of the rows, in internal units.
    src : mapping
        The rule and the exact source on its nodes (`floor_source`):
        ``q0``, ``p_min``, ``n``, ``pref``, ``dq0`` (rho_aux^(m)(q0)),
        ``dmm`` (rho_aux^(m)(P_min)), ``gq0``, ``gpq0`` and ``gppq0`` (g and
        its first two derivatives at q0), ``w`` and ``d`` (the nodes'
        weights and exact distances q0 - P), ``g`` (g at the nodes),
        ``outer`` (whether the outer end is unbounded), ``outer_from``, and,
        for an unbounded outer end, ``r`` (rho_aux^(m) at the nodes) and
        ``rm1min`` (rho_aux^(m-1)(P_min)).
    deriv : bool, optional
        Whether to compute the correction to dF/dQ. The correction to F
        does not depend on `deriv`. Default: ``True``.

    Returns
    -------
    corr : array, shape (n,)
        Correction to F at `q`.
    dcorr : array, shape (n,) or None
        Correction to dF/dQ at `q`, NaN at Q = q0 except at an unbounded
        outer end with ``outer_from`` 0, where the correction at q0 is
        finite; None where `deriv` is false.
    """
    qs = np.asarray(q, dtype=np.float64)
    q0t, n_n9, pref9 = float(src["q0"]), float(src["n"]), float(src["pref"])
    dq0, dmm = float(src["dq0"]), float(src["dmm"])
    gq0, gpq0, gppq0 = float(src["gq0"]), float(src["gpq0"]), float(src["gppq0"])
    w_p9 = np.asarray(src["w"], dtype=np.float64)
    d_ex = np.asarray(src["d"], dtype=np.float64)
    g9 = np.asarray(src["g"], dtype=np.float64)
    p_mn = float(src["p_min"])
    om = 1.0 - n_n9
    q_m_p = np.maximum((qs - q0t)[:, None] + d_ex[None, :], 1e-30)
    k10 = q_m_p ** (-n_n9)
    i_g0 = ((qs - p_mn) ** om - (qs - q0t) ** om) / om
    if src["outer"]:
        r9 = np.asarray(src["r"], dtype=np.float64)
        rm1min = float(src["rm1min"])
        rdif = np.where(d_ex < 0.01 * q0t, -gq0 * d_ex + 0.5 * gpq0 * d_ex**2, r9 - dq0)
        corr = pref9 * (
            dq0 * ((qs - p_mn) ** (-n_n9) - qs ** (-n_n9))
            - n_n9 * ((q_m_p ** (-n_n9 - 1.0)) @ (w_p9 * rdif))
            - n_n9 * rm1min * qs ** (-n_n9 - 1.0)
        )
        dcorr = None
        if deriv:
            with np.errstate(divide="ignore", invalid="ignore"):
                dcorr = pref9 * (
                    (-n_n9) * dq0 * ((qs - p_mn) ** (-n_n9 - 1.0) - qs ** (-n_n9 - 1.0))
                    + n_n9 * (n_n9 + 1.0) * ((q_m_p ** (-n_n9 - 2.0)) @ (w_p9 * rdif))
                    + n_n9 * (n_n9 + 1.0) * rm1min * qs ** (-n_n9 - 2.0)
                )
        ofr = float(src["outer_from"])
        if ofr > 0.0:
            co_out = (
                pref9
                * (-n_n9)
                * (
                    (q_m_p ** (-n_n9 - 1.0)) @ (w_p9 * r9)
                    + rm1min * qs ** (-n_n9 - 1.0)
                )
            )
            corr = np.where(qs >= ofr, co_out, corr)
            if deriv:
                dco_out = (
                    pref9
                    * n_n9
                    * (n_n9 + 1.0)
                    * (
                        (q_m_p ** (-n_n9 - 2.0)) @ (w_p9 * r9)
                        + rm1min * qs ** (-n_n9 - 2.0)
                    )
                )
                assert dcorr is not None
                dcorr = np.where(qs >= ofr, dco_out, dcorr)
                dcorr = np.where(qs > q0t, dcorr, np.nan)
        return corr, dcorr
    gdif = np.where(d_ex < 0.01 * q0t, -gpq0 * d_ex + 0.5 * gppq0 * d_ex**2, g9 - gq0)
    corr = pref9 * (
        k10 @ (w_p9 * gdif)
        + gq0 * i_g0
        + dmm * (qs - p_mn) ** (-n_n9)
        - dq0 * qs ** (-n_n9)
    )
    if not deriv:
        return corr, None
    with np.errstate(divide="ignore"):
        di_g0 = (qs - p_mn) ** (-n_n9) - np.where(qs > q0t, (qs - q0t), np.nan) ** (
            -n_n9
        )
    dcorr = pref9 * (
        (-n_n9) * ((q_m_p ** (-n_n9 - 1.0)) @ (w_p9 * gdif))
        + gq0 * di_g0
        + (-n_n9) * (dmm * (qs - p_mn) ** (-n_n9 - 1.0) - dq0 * qs ** (-n_n9 - 1.0))
    )
    return corr, dcorr


def method2_floor_tail(q: Array, tail: Mapping[str, Any]) -> tuple[Array, Array]:
    """Return the correction that turns the boundary term the kernel carries
    into a Method II child's exact contribution from below its table's floor,
    and the correction's derivative in Q (Notes Eq. (T:2.23)).

    Each power P^q of the two-term edge expansion of rho_aux adds
    C(n_f) a_q Gamma(q+1) / Gamma(a+1) Q^(a - n_f) x^a [2F1(a, n_f; a+1; x) - 1],
    with a = q - m_f and x = q_min / Q.

    Parameters
    ----------
    q : array, shape (n,)
        Relative energies Q, in internal units. The smallest energy is the
        floor q_min.
    tail : mapping
        ``a_q`` (a dict from each power q to its coefficient a_q), ``m_f``
        and ``n_f`` (the floor pair) and ``pref`` (the prefactor C(n_f)).

    Returns
    -------
    corr : array, shape (n,)
        Correction to F at `q`.
    dcorr : array, shape (n,)
        Correction to dF/dQ at `q`, NaN at Q = q_min.
    """
    qm = np.asarray(q, dtype=np.float64)
    q0m = float(np.min(qm))
    xm = np.clip(q0m / np.maximum(qm, 1e-300), 0.0, 1.0)
    mfm, nfm, cfm = int(tail["m_f"]), float(tail["n_f"]), float(tail["pref"])
    corr = np.zeros_like(qm)
    dcorr = np.zeros_like(qm)
    for qq, aq in tail["a_q"].items():
        am = qq - mfm
        bm = 1.0 - nfm
        sm = am - nfm
        cq = cfm * aq * scipy_gamma(qq + 1.0) / scipy_gamma(am + 1.0)
        phi = xm**am * (hyp2f1(am, nfm, am + 1.0, xm) - 1.0)
        corr += cq * qm**sm * phi
        with np.errstate(divide="ignore", invalid="ignore"):
            dphi = np.where(
                xm < 1.0, am * xm**am * ((1.0 - xm) ** (bm - 1.0) - 1.0), np.nan
            )
        dcorr += cq * qm ** (sm - 1.0) * (sm * phi - dphi)
    return corr, dcorr


def complete_rows(
    q: Array,
    f: Array,
    ddf: Array | None,
    edge: Mapping[str, float] | None = None,
    floor: Mapping[str, Any] | None = None,
    m2_tail: Mapping[str, Any] | None = None,
) -> tuple[Array, Array | None]:
    """Return the inversion's complete rows F and dF/dQ.

    F is the kernel's rows plus, in this order, the Type II edge term
    (`edge_term`, where ``edge`` is given), the completion below the floor
    (`floor_completion`, where ``floor`` is given and the kernel exponent is
    not 0, since the closed form at n = 0 needs no completion) and a
    Method II child's floor tail (`method2_floor_tail`, where ``m2_tail`` is
    given). dF/dQ is ``ddf`` plus the derivatives of the completion below
    the floor and of the Method II floor tail, or None where ``ddf`` is
    None. dF/dQ does not include a derivative of the Type II edge term."""
    rows = np.asarray(f, dtype=np.float64)
    if edge is not None:
        rows = np.asarray(rows + edge_term(q, edge["Pe"], edge["A"], edge["bc"]))
    dcorr = None
    if floor is not None and not floor["half_int"]:
        corr, dcorr = floor_completion(q, floor, deriv=ddf is not None)
        rows = np.asarray(rows + corr)
    dtail = None
    if m2_tail is not None:
        corr_m, dtail = method2_floor_tail(q, m2_tail)
        rows = np.asarray(rows + corr_m)
    d_rows = None
    if ddf is not None:
        d_rows = np.asarray(ddf, dtype=np.float64)
        if dcorr is not None:
            d_rows = d_rows + dcorr
        if dtail is not None:
            d_rows = d_rows + dtail
    return rows, d_rows
