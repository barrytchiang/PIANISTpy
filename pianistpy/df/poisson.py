r"""The Poisson solve of a model from its auxiliary density.

`run_poisson_ode` takes the auxiliary density :math:`\rho_{\rm aux}(P)` on
the source grid (`truncation.truncated_source`), the anisotropy and the
source profile, and returns a `PoissonSolve`: the relative potential
:math:`\Phi(r)` and its radial derivative on a grid of radii. The grid runs
from its innermost radius, near the center, out to the radius where
:math:`\Phi` falls to the floor `K_FLOOR` (the edge of a spatially finite
model), or out to the grid's outer radius where :math:`\Phi` stays above
the floor. The module also holds the density lookup that the fills read
(`profile_density`, `source_slopes`).

With :math:`s = \ln r`, the solve integrates the state
:math:`(\Phi, W, u)`, with :math:`W = r \, d\Phi/dr` and the depth
:math:`u = P_0 - \Phi` below the central potential :math:`P_0`:

.. math::

    \frac{d\Phi}{ds} = W, \qquad
    \frac{dW}{ds} = -W - r^2 S(\Phi, r), \qquad
    \frac{du}{ds} = -W,

where :math:`S(\Phi, r)` is the anisotropy family's prefactor times
:math:`\rho_{\rm aux}(\Phi)`, divided by the anisotropy factor :math:`h(r)`
(`poisson_source`). All quantities are in internal units. The solve starts
from the inner series of the cusp and holds a relative tolerance
``ode_rtol`` (default :math:`10^{-11}`) per step. A truncated model's last
rows are solved again on a grid geometric in :math:`R_{\rm t} - r`, and the
last row lies at the radius where :math:`\Phi` crosses the floor. The method
is in Notes Section T:3.3.
"""

__all__ = [
    "EVENT_ATOL",
    "EVENT_RTOL",
    "PoissonSolve",
    "cuddeford_prefactor",
    "density_cut",
    "poisson_source",
    "profile_density",
    "run_poisson_ode",
    "source_slopes",
]

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma as _sgamma

from ..anisotropy.factor import inverse_factor
from ..exceptions import AccuracyError, InputError, UnphysicalModelError
from ..numerics.pchip import interp1d, pchip_slopes
from ..typing import Array
from .truncation import K_FLOOR, ROW_BINS_TOP, inner_zone_depth

# The margin, in internal units, by which the density lookup keeps the
# potential below the top of the source table.
PHI_CLIP_EPS = 1e-14
# The solve's tables and rows are padded to the least length ROW_BUCKET * 2**k,
# with k a non-negative integer, at or above their length, so that one
# compiled solve serves every truncation whose lengths fall in the same
# buckets. The bucket sizes grow geometrically, so a
# change of a length by a small fraction changes the padded length only where
# the length crosses a bucket's size.
ROW_BUCKET = 5
# The potential, in internal units, above which the solve applies the density
# source; at or below PHYS_FLOOR the solve's derivatives are zero.
PHYS_FLOOR = 1e-13
# The tolerances of the floor events' root finder in s = ln r:
# |delta ln r| <= EVENT_ATOL + EVENT_RTOL |ln r|.
EVENT_RTOL = 1e-12
EVENT_ATOL = 1e-14


@dataclass(frozen=True)
class PoissonSolve:
    r"""The Poisson solve of a model on its grid of radii.

    All quantities are in internal units. The main solve is the first
    integration from the center; the edge continuation is the second
    integration of a truncated model's last rows.

    Parameters
    ----------
    r : array, shape (n,)
        Radii of the rows, increasing. Where :math:`\Phi` reaches the floor,
        the last row lies at the floor's radius.
    phi : array, shape (n,)
        The relative potential :math:`\Phi` at `r`.
    dphi_dr : array, shape (n,)
        The radial derivative :math:`d\Phi/dr` at `r`.
    u_r : array, shape (m,)
        Radii of the main solve's grid.
    u : array, shape (m,)
        The depth :math:`u = P_0 - \Phi` at `u_r`, taken from the main
        solve's own variable :math:`u` wherever :math:`u < \Phi`, and
        computed as :math:`P_0 - \Phi` from the main solve's :math:`\Phi`
        elsewhere.
    event_hit : bool
        Whether :math:`\Phi` reached the floor `K_FLOOR` inside the grid's
        outer radius.
    r_event : float
        The radius where the main solve's floor event fired; ``inf`` where
        `event_hit` is False.
    r_floor_request : float
        The grid's innermost radius, read before any outward move of the
        solve's start. Let :math:`u_{\rm top}` be :math:`P_0` minus the
        source table's highest potential. Where the inner series' depth at
        the innermost radius is below :math:`u_{\rm top}`, the solve's start
        is moved outward to the radius where the series' depth equals
        :math:`u_{\rm top}`. Elsewhere `r_floor_request` equals the solve's
        first radius.
    edge_continuation : dict or None
        The edge continuation's start radius ``R_c``, its floor radius
        ``R_t``, the potential ``psi_c`` below which the continuation
        replaces the main solve's rows, and its nodes per decade of
        :math:`R_{\rm t} - r`, ``per_dec``; None where the continuation did
        not run.
    evaluate : callable or None
        The main solve's dense interpolant of :math:`(\Phi, W, u)` at an
        array of radii, returning an array of shape (k, 3).
    evaluate_range : tuple of float
        The radii between which `evaluate` interpolates the main solve: the
        grid's first radius and the main solve's last radius (the floor
        event's radius where `event_hit` is True).
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
    r"""Return the Cuddeford prefactor C of a model's density.

    The prefactor is

    .. math::

        C = \frac{2 \pi^{3/2} \, \Gamma(1 - \beta_{\rm c})}
        {\Gamma(3/2 - \beta_{\rm c})},

    with :math:`\beta_{\rm c}` the central anisotropy `bc`. The density of a
    constant-anisotropy or COM model is :math:`C \rho_{\rm aux} / h(r)`,
    with :math:`h(r)` the anisotropy factor and

    .. math::

        \rho_{\rm aux}(P) = \int F(Q) \, [2 (P - Q)]^{1/2 - \beta_{\rm c}}
        \, {\rm d}Q.
    """
    return float(2.0 * np.pi**1.5 * _sgamma(1.0 - bc) / _sgamma(1.5 - bc))


def _density_of(rhs, r, ra, model_type, bc):
    r"""Return the density: the family's prefactor times `rhs`, over h(r).

    `rhs` is :math:`\rho_{\rm aux}`, and h(r) is the anisotropy factor
    (`inverse_factor`). The prefactor is :math:`4 \pi` for an
    Osipkov-Merritt model and `cuddeford_prefactor` otherwise. A Type II
    model is not defined at or beyond :math:`R_{\rm a}`, so a Type II
    model's density at or beyond :math:`R_{\rm a}` is zero. At radii at or
    beyond :math:`R_{\rm a}`, h is evaluated at :math:`R_{\rm a} / 2`
    instead, so that no value of h beyond :math:`R_{\rm a}` enters the
    computation. `run_poisson_ode` raises UnphysicalModelError after the
    solve where a Type II model's edge lies at or beyond
    :math:`R_{\rm a}`.
    """
    if model_type in ("OM_I", "OM_II"):
        pref = 4.0 * jnp.pi
    else:
        pref = cuddeford_prefactor(bc)
    if model_type not in ("OM_I", "OM_II", "COM_I", "COM_II"):
        (h,) = inverse_factor(r, "constant", bc, math.inf)
        return pref * rhs / h
    if model_type in ("OM_I", "COM_I"):
        (h,) = inverse_factor(r, model_type, bc, ra)
        return pref * rhs / h
    inside = r < ra
    (h,) = inverse_factor(jnp.where(inside, r, 0.5 * ra), model_type, bc, ra)
    return jnp.where(inside, pref * rhs / h, 0.0)


def source_slopes(
    q: Any, p_rhs: Any, p_0: float | None, p_s: float, order: str = "cubic"
) -> Array:
    r"""Return the node slopes d ln rho_aux / dx of the source lookup.

    The variable x is the logit :math:`\ln[Q / (P_0 - Q)]` where `p_0` is
    given, and :math:`\ln Q` where `p_0` is None. The table's nodes are cut
    into runs of consecutive nodes with positive :math:`\rho_{\rm aux}` and
    finite x; where `p_s` is finite, each run is cut again at `p_s`. On a run
    of at least four nodes with increasing x, the slopes are those of a
    not-a-knot cubic spline of :math:`\ln \rho_{\rm aux}` in x. At every
    other node the slopes are PCHIP's. `order` is not read.
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
    phi,
    r,
    ra,
    model_type,
    bc,
    q,
    p_rhs,
    p_0=None,
    u=None,
    m=None,
    order="cubic",
    n=None,
):
    """Return the density at (Phi, r) from the source table, and whether Phi
    is above PHYS_FLOOR.

    The lookup interpolates ln rho_aux in the logit of Phi where `p_0` is
    given (with the depth `u` in place of ``p_0 - Phi`` where `u` is given
    and Phi lies inside the table's range of potentials),
    and in ln Phi otherwise. The density is exactly zero where Phi lies below
    the table's first node. With `n`, the table holds `n` nodes, padded to
    its full length by repeats of its last node (`_padded`).
    """
    # The lower clip is the table's first node with no margin. The first node
    # lies at the floor K_FLOOR = 1e-10, and a margin of PHI_CLIP_EPS = 1e-14
    # at the first node would move the floor's potential by a relative 1e-4.
    phi_safe = jnp.clip(phi, q[0], q[-1] - PHI_CLIP_EPS)
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
            n=n,
        )
    )
    rhs = jnp.where(phi < q[0], 0.0, rhs)
    # The density is the prefactor times rho_aux over the anisotropy factor
    # h(r), and zero at or beyond R_a for a Type II model.
    return _density_of(rhs, r, ra, model_type, bc), is_physical


def density_cut(c: float, gamma: float) -> float:
    """Return the radius below which the cusp density exceeds the largest double.

    The cusp density is C r^(-gamma) with C = `c` and gamma = `gamma`. The
    radius is (C / DBL_MAX)^(1 / gamma), with DBL_MAX the largest double; 0
    where gamma is not positive.
    """
    if not float(gamma) > 0.0:
        return 0.0
    log_max = float(np.log(np.finfo(np.float64).max))
    return float(np.exp((np.log(float(c)) - log_max) / float(gamma)))


def _bucket(n: int) -> int:
    """Return the padded length of `n` rows, ``ROW_BUCKET * 2**k`` for the
    least non-negative integer k at which ``ROW_BUCKET * 2**k`` is at least
    `n`.

    >>> _bucket(11999), _bucket(35910), _bucket(60498)
    (20480, 40960, 81920)
    """
    size = ROW_BUCKET
    while size < n:
        size *= 2
    return size


def _padded(a: Any, size: int) -> Any:
    """Return `a` padded along its first axis to length `size`.

    The padding repeats the last entry of `a`. The padding serves a table
    that `interp1d` reads with its true node count ``n``, and rows whose
    padded entries are cut off after the read.
    """
    a = jnp.asarray(a)
    return jnp.concatenate([a, jnp.repeat(a[-1:], size - a.shape[0], axis=0)])


def profile_density(
    r, phi, q, p_rhs, ra, bc, model_type, p_0=None, m=None, order="cubic", *, r_cut
):
    r"""Return the density at the solve's rows (r, Phi).

    The density is the family's prefactor times :math:`\rho_{\rm aux}(\Phi)`,
    over the anisotropy factor h(r). The density is zero at or beyond
    :math:`R_{\rm a}` for a Type II model, where :math:`\Phi` lies below the
    table's first node, and at :math:`r \le` `r_cut`, the profile's
    `density_cut`; the inner fill replaces the rows at :math:`r \le` `r_cut`.
    The rows and the table are padded to `_bucket` lengths, so that one
    compiled read serves every model of the same anisotropy and the same
    padded lengths.

    Raises
    ------
    InputError
        If `order` is "cubic" and the slopes `m` are None.
    """
    if order == "cubic" and m is None:
        raise InputError("profile_density needs the source lookup's slopes m")
    if order != "cubic":
        return _profile_density(
            r, phi, q, p_rhs, ra, bc, model_type, p_0, None, order, r_cut, None
        )
    k, nq = len(r), len(q)
    rows, nodes = _bucket(k), _bucket(nq)
    return _profile_density(
        _padded(r, rows),
        _padded(phi, rows),
        _padded(q, nodes),
        _padded(p_rhs, nodes),
        ra,
        bc,
        model_type,
        p_0,
        _padded(m, nodes),
        order,
        r_cut,
        jnp.asarray(nq),
    )[:k]


@partial(jax.jit, static_argnames=("model_type", "bc", "order"))
def _profile_density(r, phi, q, p_rhs, ra, bc, model_type, p_0, m, order, r_cut, n):
    """Return `profile_density` at padded rows from a padded table of `n` nodes."""
    # The lower clip is the table's first node with no margin, as in
    # poisson_source, so that the floor row's density is read at the floor's
    # potential.
    phi_safe = jnp.clip(phi, q[0], q[-1])
    log_phi = jnp.log(jnp.maximum(phi_safe, 1e-300))
    log_q = jnp.log(jnp.maximum(q, 1e-300))
    if p_0 is not None:
        log_phi = log_phi - jnp.log(jnp.maximum(p_0 - phi_safe, 1e-300))
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
            m=(m if order == "cubic" else None),
            n=n,
        )
    )
    rhs = jnp.where(phi < q[0], 0.0, rhs)
    rho = _density_of(rhs, r, ra, model_type, bc)
    return jnp.where(r > r_cut, rho, 0.0)


def phi_ic_series(p_0, r0, q, p_rhs, ra, model_type, bc, gamma_inner):
    r"""Return the depth :math:`u(r_0) = P_0 - \Phi(r_0)` of the inner series.

    Here :math:`P_0` is the central potential `p_0` and :math:`r_0` is the
    radius `r0` at which the cusp's inner series starts.

    The depth solves
    :math:`u = S(P_0 - u, r_0)\,r_0^2 / [(3 - \gamma)(2 - \gamma)]`, with
    :math:`\gamma` the inner slope `gamma_inner` and :math:`S` the density
    source of `poisson_source`, read here by linear interpolation of
    :math:`\ln \rho_{\rm aux}` in :math:`\ln \Phi`. The root is found by 200
    bisection steps in ln u between :math:`\ln 10^{-300}` and
    :math:`\ln (P_0 / 2)`.

    Raises
    ------
    InputError
        If the root lies above :math:`P_0 / 2`, half the depth of the
        potential well.
    InputError
        If the root lies below :math:`10^{-300}`, as where the source
        :math:`S` is zero at :math:`r_0`.
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
        if model_type in ("OM_I", "OM_II"):
            (h,) = inverse_factor(r0, model_type, bc, ra)
            return 4.0 * np.pi * rhs / h
        if model_type in ("COM_I", "COM_II"):
            (h,) = inverse_factor(r0, model_type, bc, ra)
            return cuddeford_prefactor(bc) * rhs / h
        (h,) = inverse_factor(r0, "constant", bc, math.inf)
        return cuddeford_prefactor(bc) * rhs / h

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
    r"""The model's quantities that the solve's vector fields read.

    The record is passed as the solve's arguments. The arrays, `p_0` and
    `ra` are traced, and `bc`, `model_type` and `order` are static, so one
    compiled solve serves every model of the same family, the same central
    anisotropy :math:`\beta_{\rm c}` and the same padded table length. `n`
    is the table's true node count; `m` holds the lookup's node slopes, or
    None.
    """

    p_0: Any
    ra: Any
    q: Any
    p_rhs: Any
    m: Any
    n: Any
    bc: float
    model_type: str
    order: str


def _logr_term(s, y, args):
    r"""Return :math:`d(\Phi, W, u)/ds = (W, -W - r^2 S, -W)` at :math:`s = \ln r`.

    :math:`S` is the density source of `poisson_source`, read at the
    potential :math:`P_0 - u` with the depth :math:`u`. The derivatives are
    zero where the potential :math:`P_0 - u`, clipped to the source table's
    range, is at or below `PHYS_FLOOR`.
    """
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
        n=args.n,
    )
    return jnp.where(_is_phys, jnp.array([W, -W - R * R * _src, -W]), jnp.zeros(3))


def _logr_term_lnphi(s, y, args):
    """Return d(ln Phi, W)/ds = (W / Phi, -W - r^2 S) at s = ln r.

    S is the density source of `poisson_source` at Phi. The derivatives are
    zero where Phi, clipped to the source table's range, is at or below
    PHYS_FLOOR.
    """
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
        n=args.n,
    )
    return jnp.where(_is_phys, jnp.array([W / Phi, -W - R * R * _src]), jnp.zeros(2))


@jax.jit
def _evaluate_at(dense: Any, s: Any) -> Any:
    """Return the solve's dense interpolation `dense` at the log-radii `s`.

    The values equal those of the solution's own ``evaluate``.
    `_evaluate_at` is compiled once per set of array shapes, and the
    compiled function serves every solve of those shapes.
    """
    return jax.vmap(dense.evaluate)(s)


def _floor_cond(t, y, args, **kwargs):
    return y[0] - K_FLOOR


def _ln_floor_cond(t, y, args, **kwargs):
    # The event function is K_FLOOR (exp(ln Phi - ln K_FLOOR) - 1), which
    # equals Phi - K_FLOOR, the main solve's event function in units of Phi.
    # The units matter because the root finder's tolerance on the event
    # function is absolute.
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
    r"""Return the Poisson solve of a model on its grid of radii.

    All quantities are in internal units.

    Parameters
    ----------
    q : array, shape (n,)
        The source grid in the potential, increasing (shifted by the
        boundary-value closure where the caller applies one).
    p_rhs : array, shape (n,)
        The auxiliary density :math:`\rho_{\rm aux}` at `q`.
    p_0 : float
        The model's central potential, :math:`P_0 - \mathcal{K}_{\rm t}`
        for a truncated model.
    ra : float
        The anisotropy radius :math:`R_{\rm a}`; ``inf`` where the model has
        none.
    bc : float
        The central anisotropy :math:`\beta_{\rm c}`.
    model_type : str
        The anisotropy family: "OM_I", "OM_II", "COM_I", "COM_II", or any
        other value for constant anisotropy.
    gamma_inner : float
        The source profile's inner slope :math:`\gamma`, which sets the
        cusp's inner series.
    potential : callable
        The source profile's potential at an array of radii, equal to 1 at
        the center; the inner zone's end is read from it.
    settings : mapping
        The truncation settings (`Settings.truncation`). ``log10_r_max`` is
        the caller's value, raised where the profile needs a larger outer
        radius.
    p_s : float, optional
        The inversion's interior point, where the source lookup's slopes
        split (`source_slopes`). Default: ``nan``, which means no split.
    sentinel_q : float or None, optional
        The highest energy of the finite part of the DF table, where the
        table's top holds +inf. Default: ``None``, which means the table
        holds no +inf.
    edge_continuation : bool, optional
        Whether to solve a truncated model's last rows again on a grid
        geometric in :math:`R_{\rm t} - r`. Default: ``True``.

    Returns
    -------
    solve : PoissonSolve
        The rows of :math:`\Phi` and :math:`d\Phi/dr`, the main solve's
        depths and dense interpolant, and the floor event's radius.

    Raises
    ------
    InputError
        If the inner series has no root at the grid's first radius
        (`phi_ic_series`).
    InputError
        If the radius where the inner series' depth reaches
        the square root of ``ode_atol`` lies below the smallest positive
        normal double.
    InputError
        If the potential at the solve's start, :math:`P_0 - u`, lies above
        `sentinel_q`, the highest energy of the finite part of the DF table.
    UnphysicalModelError
        If a Type II model's potential is above the floor at
        :math:`R_{\rm a}`, or the model's edge lies at or beyond
        :math:`R_{\rm a}`.
    AccuracyError
        If the main solve stops before its last radius, or the edge
        continuation does not reach the floor.
    """
    # diffrax and optimistix are imported inside the function, not with the
    # module: importing optimistix (version 0.0.11) creates a JAX array, which
    # starts JAX's backend before set_platform can select the platform.
    import diffrax
    import optimistix

    order = "cubic"
    psi_floor = K_FLOOR
    r_min_exp = float(settings["log10_r_min"])
    r_max_exp = float(settings["log10_r_max"])
    n_inner, n_outer = int(settings["n_rows_inner"]), int(settings["n_rows_outer"])
    ode_rtol, ode_atol = float(settings["ode_rtol"]), float(settings["ode_atol"])

    # The inner zone ends where the depth reaches inner_zone_depth(p_0,
    # ode_rtol). Near the center the truncated model's depth equals the
    # untruncated profile's depth, so the inner zone's end r_split is the
    # radius where the profile's own potential (1 at the center) equals 1 minus
    # inner_zone_depth(p_0, ode_rtol), found by log-log interpolation of the
    # profile's potential at radii from 10^log10_r_min to 1e-3.
    _psi_ref_arr = jnp.array(potential(jnp.logspace(r_min_exp, -3, 10000)))
    _r_ref_arr = jnp.logspace(r_min_exp, -3, 10000)
    _log_psi_ref = jnp.log(jnp.maximum(jnp.flip(_psi_ref_arr), 1e-300))
    _log_r_ref = jnp.log(jnp.maximum(jnp.flip(_r_ref_arr), 1e-300))
    _log_psi_tgt = jnp.log(
        jnp.maximum(jnp.array(1.0 - inner_zone_depth(p_0, ode_rtol)), 1e-300)
    )
    _r_split = float(jnp.exp(jnp.interp(_log_psi_tgt, _log_psi_ref, _log_r_ref)))

    # The grid is the union of an inner zone of n_rows_inner radii up to
    # r_split, four overlapping logarithmic zones beyond r_split that share
    # n_rows_outer radii as 40, 30, 20 and 10 percent, and a set of radii
    # uniform in log r over the whole grid.
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
    # The grid extends inward to the radius where the series' depth equals
    # sqrt(ode_atol), the depth at which the series' error meets the solver's
    # absolute tolerance ode_atol.
    _u_floor_target = float(np.sqrt(ode_atol))
    if float(gamma_inner) < 2.0 and _u_ic > _u_floor_target:
        _A_cusp = _u_ic / float(r_ode[0]) ** (2.0 - float(gamma_inner))
        # The radius is formed as its logarithm first: for a steep cusp the
        # radius lies below the smallest double, where the power that forms
        # the radius would round to 0.
        _log10_r_g4 = (np.log10(_u_floor_target) - np.log10(_A_cusp)) / (
            2.0 - float(gamma_inner)
        )
        _tiny = float(np.finfo(np.float64).tiny)
        if _log10_r_g4 < np.log10(_tiny):
            raise InputError(
                "the series meets the solver's absolute tolerance only at R = "
                f"(sqrt(ode_atol) / A)^(1 / (2 - gamma)) = 10^{_log10_r_g4:.1f} "
                f"(ode_atol = {ode_atol:.1e}, A = {_A_cusp:.3e}, "
                f"gamma = {float(gamma_inner):.3f}), below the double's range "
                f"({_tiny:.3e}): the inner floor is not representable"
            )
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
    # Where the series' depth at the first radius lies below u_top, the depth
    # of the source table's top, the solve starts farther out: at the radius
    # where the series' depth equals u_top, bracketed by doubling the grid
    # index and then found by 80 bisection steps in ln r.
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
    # The source table is padded to a _bucket length and read with its true
    # node count, and the save times are padded by repeats of the last save
    # time, whose rows are cut off after the solve. One compiled solve then
    # serves every truncation whose lengths fall in the same buckets.
    _n_q = len(_q_fi)
    _nodes = _bucket(_n_q)
    _args = _SourceArgs(
        jnp.asarray(_P0_u, dtype=jnp.float64),
        jnp.asarray(ra, dtype=jnp.float64),
        _padded(jnp.asarray(_q_fi, dtype=jnp.float64), _nodes),
        _padded(jnp.asarray(_p_rhs, dtype=jnp.float64), _nodes),
        None if _m_src_logit is None else _padded(_m_src_logit, _nodes),
        jnp.asarray(_n_q),
        float(bc),
        model_type,
        order,
    )
    _n_s = int(_s_ode.shape[0])
    _s_save = _padded(_s_ode, _bucket(_n_s))

    _event = diffrax.Event(
        cond_fn=_floor_cond,
        root_finder=optimistix.Newton(rtol=EVENT_RTOL, atol=EVENT_ATOL),
    )
    # The solve runs on the CPU on every platform: the ODE is serial, and on
    # the GPU each step would pay a kernel launch latency.
    cpu = jax.devices("cpu")[0]
    y0, _s_ode, _s_save, _args = jax.tree.map(
        lambda x: jax.device_put(x, cpu) if isinstance(x, jax.Array) else x,
        (y0, _s_ode, _s_save, _args),
    )
    with jax.default_device(cpu):
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
                subs=[diffrax.SubSaveAt(ts=_s_save), diffrax.SubSaveAt(t1=True)],
                dense=True,
            ),
            event=_event,
            max_steps=200_000,
            throw=False,
        )
    if not bool(diffrax.is_okay(sol_obj.result)):
        # A Type II model whose potential is still above the floor at R_a has
        # a density rho_aux / (1 - r^2/R_a^2) with a pole at R_a. The solve
        # approaches the pole in steps proportional to R_a - r and stops
        # before R_a.
        _s_stop = float(np.asarray(sol_obj.ts[1], dtype=np.float64).reshape(-1)[-1])
        _y_stop = np.asarray(sol_obj.ys[1], dtype=np.float64).reshape(-1)
        _r_stop = float(np.exp(_s_stop))
        if model_type in ("OM_II", "COM_II") and abs(_r_stop / float(ra) - 1.0) < 1e-6:
            raise UnphysicalModelError(
                f"TYPE II EDGE: the model's potential at R_a = {float(ra):g} is "
                f"Phi = {_y_stop[0]:.6e}, above the floor, so its truncation radius "
                "lies beyond R_a, where the Type II density rho_aux / (1 - r^2/R_a^2) "
                "has a pole (the solve stopped at R_a - r = "
                f"{float(ra) - _r_stop:.3e}); a larger K_t moves the edge inside R_a"
            )
        # Any other early stop is a failure of the solve itself, and raises
        # AccuracyError with diffrax's result code.
        raise AccuracyError(
            f"the Poisson solve stopped at R = {_r_stop:.6e} before its last radius "
            f"R = {float(np.exp(float(_s_ode[-1]))):.6e} ({sol_obj.result}): refused"
        )
    _ys_uw = sol_obj.ys[0][:_n_s]
    # At each row, the smaller of Phi and u is taken from the solve and the
    # larger is P_0 less the smaller, so that the smaller quantity keeps its
    # own relative precision.
    _u_prim = _ys_uw[:, 2] < _ys_uw[:, 0]
    _Phi_sv = jnp.where(_u_prim, _P0_u - _ys_uw[:, 2], _ys_uw[:, 0])
    _u_sv = jnp.where(_u_prim, _ys_uw[:, 2], _P0_u - _ys_uw[:, 0])
    _ys_ts = jnp.stack([_Phi_sv, _ys_uw[:, 1]], axis=1)
    u_r, u = np.array(r_ode, dtype=np.float64), np.array(_u_sv, dtype=np.float64)

    def evaluate(R, _dense=sol_obj.interpolation):
        return np.asarray(
            _evaluate_at(_dense, jnp.log(jnp.asarray(R, dtype=jnp.float64))),
            dtype=np.float64,
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
    if model_type in ("OM_II", "COM_II") and _ev_hit and not r_event < float(ra):
        # The solve passed R_a with the potential above the floor, carrying
        # zero density beyond R_a, and placed the edge beyond R_a, where the
        # Type II model is not defined.
        raise UnphysicalModelError(
            f"TYPE II EDGE: the model's edge R_t = {r_event:.10g} lies at or beyond "
            f"R_a = {float(ra):g}, where the Type II model is not defined; a "
            "larger K_t moves the edge inside R_a"
        )
    sol = jnp.stack([_ys_ts[:, 0], _ys_ts[:, 1] / r_ode], axis=1)

    # The edge continuation solves a truncated model's last rows again on a
    # grid geometric in R_t - r, up to the floor crossing. The continuation
    # starts at R_c, the last finite row inside R_t whose potential is at or
    # above psi_c. psi_c is the larger of continuation_start times P_0 and the
    # top of the highest potential bin, of profile_bin_decades decades between
    # K_FLOOR and ROW_BINS_TOP times P_0, that holds fewer than
    # max(1, profile_rows_per_decade * profile_bin_decades) rows. The
    # continuation is skipped where a_edge R_t is at most 100 K_FLOOR, with
    # a_edge = -dPhi/dr at R_t, or where no row qualifies as R_c.
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
            _n_c = _s_cont.size
            # The continuation integrates ln Phi and W with the relative
            # tolerance ode_rtol and no absolute tolerance, as the main solve
            # does. The step control on ln Phi then holds the absolute error of
            # ln Phi per step, which is Phi's relative error per step, near
            # ode_rtol |ln Phi| at every decade of R_t - r that the
            # continuation spans.
            # The continuation's own floor event on ln Phi ends the
            # continuation.
            _y0_c = jnp.array(
                [np.log(_sol_np[_ic, 0]), _sol_np[_ic, 1] * _R_c], dtype=jnp.float64
            )
            _sol_c = diffrax.diffeqsolve(
                diffrax.ODETerm(_logr_term_lnphi),
                diffrax.Dopri5(),
                t0=jnp.asarray(float(np.log(_R_c)), dtype=jnp.float64),
                t1=jnp.asarray(_s_t1 + 1e-3, dtype=jnp.float64),
                dt0=1e-4,
                y0=_y0_c,
                args=_args,
                stepsize_controller=diffrax.PIDController(rtol=ode_rtol, atol=0.0),
                saveat=diffrax.SaveAt(
                    subs=[
                        diffrax.SubSaveAt(
                            ts=_padded(
                                jnp.array(_s_cont, dtype=jnp.float64), _bucket(_n_c)
                            )
                        ),
                        diffrax.SubSaveAt(t1=True),
                    ]
                ),
                event=diffrax.Event(
                    cond_fn=_ln_floor_cond,
                    root_finder=optimistix.Newton(rtol=EVENT_RTOL, atol=EVENT_ATOL),
                ),
                max_steps=200_000,
            )
            _ys_c = np.array(_sol_c.ys[0], dtype=np.float64)[:_n_c]
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

    # The terminal row replaces the first row that is not above the floor (at
    # or below the floor, or not finite), and every row after that first row,
    # with one row at the floor. The terminal row's radius and dPhi/dr are
    # interpolated linearly in Phi between the last row above the floor and
    # the first row not above the floor. Where the first row not above the
    # floor has a non-finite Phi, the terminal radius is extrapolated with
    # the slope dPhi/dr of the last two rows above the floor, clipped to
    # between 1 and 2 times the last such row's radius, and the terminal
    # dPhi/dr is that of the last row above the floor.
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
