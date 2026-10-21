r"""Build the ellipse route of a model of one component.

The ellipse route (`EllipseRoute`) computes the rows of the line-of-sight
velocity distribution that the public calls `pianistpy.losvd`,
`pianistpy.aperture_losvd` and `pianistpy.LosTable` read. `ellipse_route`
builds the route from a model. This module also holds the parts of a model
that the route reads: the potential and its log-slopes
(`PotentialReading`, `LogPotentialSlopes`), the DF (`DfReading`), the
cumulative integral of the DF (`cumulative_df`), the constants of the model
(`ellipse_model`) and the radius where the rows end (`floor_radius`).

The cumulative integral of the DF is

.. math::

    G(Q) = \int_0^Q F(K)\,{\rm d}K,

with F the DF and K its energy variable. `cumulative_df` doubles its
Gauss-Legendre order until two successive orders agree to a relative
:math:`10^{-7}` (Notes Section T:6.4).
"""

__all__ = [
    "DfReading",
    "LogPotentialSlopes",
    "PotentialReading",
    "cumulative_df",
    "ellipse_model",
    "ellipse_route",
    "floor_radius",
    "unconverged_cumulative",
]

import math

import numpy as np

from ..df import Model
from ..df.central import LOG10_DF_CEILING
from ..df.lookup import (
    K_0,
    LookupArrays,
    LookupKinds,
    below_floor_mass,
    evaluate_lookup,
    prepare_lookup,
)
from ..exceptions import AccuracyError, InputError, ResultError
from ..numerics import hermite
from ..profiles import TruncatedProfile
from ..typing import Array, ArrayLike
from ..validation import TIER_BOUNDS
from ._ellipse import EllipseRoute
from ._ellipse_df import EllipseDF
from ._ellipse_rules import EllipseModel, EllipseRules, layer_counts
from ._tables import _kernel_beta

# The Gauss-Legendre order of `cumulative_df` starts at CUMULATIVE_FIRST_ORDER
# and doubles up to CUMULATIVE_LAST_ORDER. The doubling stops once two
# successive orders agree to a relative CUMULATIVE_TOLERANCE, which is 0.1 of
# 1e-6 and serves the bound 1e-6.
CUMULATIVE_FIRST_ORDER = 16
CUMULATIVE_LAST_ORDER = 512
CUMULATIVE_TOLERANCE = 0.1 * TIER_BOUNDS[0]


class LogPotentialSlopes:
    r"""The first and second derivatives of :math:`\ln\mathcal{P}` in :math:`\ln r`.

    :math:`\mathcal{P}` is the relative potential of a profile, and
    :math:`y = \ln r`. Between the profile's rows, :math:`\ln\mathcal{P}` is
    the cubic Hermite interpolant in y that `TruncatedProfile.potential`
    evaluates. On each interval between two rows, the first derivative is
    therefore a quadratic in y and the second derivative is linear in y.
    `EllipseRoute.f_extrema` bounds the first derivative on an interval from
    the quadratic form of the first derivative.

    A profile with an edge row has a Kepler interval from the last row to
    the edge :math:`r_{\rm t}`. On the Kepler interval,
    :math:`\mathcal{P} = M_{\rm t}(1/r - 1/r_{\rm t})`, with
    :math:`M_{\rm t}` the mass of the truncated profile. The first derivative there is
    :math:`-r_{\rm t}/(r_{\rm t} - r)`, and the second derivative is
    :math:`-r_{\rm t} r/(r_{\rm t} - r)^2`. Both derivatives decrease
    monotonically in r and tend to minus infinity at :math:`r_{\rm t}`.

    Parameters
    ----------
    profile : TruncatedProfile
        The profile.

    Attributes
    ----------
    x : array
        The knots in ln r: the rows, followed by :math:`\ln r_{\rm t}` where
        the profile has an edge row.
    """

    def __init__(self, profile: TruncatedProfile) -> None:
        tables = profile.interpolation_tables()
        self._log_r = np.asarray(tables.log_r, dtype=np.float64)
        self._log_p = np.asarray(tables.log_p, dtype=np.float64)
        self._slope = np.asarray(tables.log_p_slope, dtype=np.float64)
        self._edge = bool(tables.has_edge_row)
        self._r_t = float(tables.r_t)
        self.x = (
            np.append(self._log_r, math.log(self._r_t)) if self._edge else self._log_r
        )

    def __call__(self, y: ArrayLike, nu: int) -> Array:
        """Return the `nu`-th derivative at `y` as the right-hand limit at a knot.

        The derivative is read on the interval that starts at or below `y`.
        `nu` is 1 or 2.
        """
        return self._read(y, nu, "right")

    def left(self, y: ArrayLike, nu: int) -> Array:
        """Return the `nu`-th derivative at `y` as the left-hand limit at a knot.

        The derivative is read on the interval that ends at or above `y`.
        `nu` is 1 or 2.
        """
        return self._read(y, nu, "left")

    def both(self, y: ArrayLike, side: str = "right") -> tuple[Array, Array]:
        """Return the first and second derivatives at `y` from one interval lookup.

        With ``side="right"`` each derivative equals the value of `__call__`.
        With ``side="left"`` each derivative equals the value of `left`.
        """
        d1, d2 = self._read_many(y, (1, 2), side)
        return d1, d2

    def _read(self, y: ArrayLike, nu: int, side: str) -> Array:
        if nu not in (1, 2):
            raise ValueError(
                f"The derivative order nu must be 1 or 2; the order given is {nu}."
            )
        return self._read_many(y, (nu,), side)[0]

    def _read_many(
        self, y: ArrayLike, orders: tuple[int, ...], side: str
    ) -> tuple[Array, ...]:
        y = np.asarray(y, dtype=np.float64)
        x = self._log_r
        n = x.size
        i = np.clip(
            np.searchsorted(x, y, side="right" if side == "right" else "left") - 1,
            0,
            n - 2,
        ).astype(np.intp, copy=False)
        i1 = (i + 1).astype(np.intp, copy=False)
        h = x[i1] - x[i]
        t = (y - x[i]) / h
        y0, y1 = self._log_p[i], self._log_p[i1]
        m0, m1 = self._slope[i] * h, self._slope[i1] * h
        if self._edge:
            # The Kepler interval starts at the last row. At the last row the
            # right-hand limit is the Kepler form and the left-hand limit is
            # the Hermite interpolant.
            kepler = (y >= x[-1]) if side == "right" else (y > x[-1])
            r = np.exp(y)
        out = []
        for nu in orders:
            if nu == 1:
                inside = (
                    (6.0 * t * t - 6.0 * t) * y0
                    + (3.0 * t * t - 4.0 * t + 1.0) * m0
                    + (-6.0 * t * t + 6.0 * t) * y1
                    + (3.0 * t * t - 2.0 * t) * m1
                ) / h
            else:
                inside = (
                    (12.0 * t - 6.0) * y0
                    + (6.0 * t - 4.0) * m0
                    + (-12.0 * t + 6.0) * y1
                    + (6.0 * t - 2.0) * m1
                ) / (h * h)
            if not self._edge:
                out.append(inside)
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = self._r_t - r
                value = -self._r_t / gap if nu == 1 else -self._r_t * r / (gap * gap)
                value = np.where(gap > 0.0, value, -np.inf)
            out.append(np.where(kepler, value, inside))
        return tuple(out)


class PotentialReading:
    r"""The relative potential of a profile at radii, as the ellipse route reads it.

    A call returns the values of `TruncatedProfile.potential`. The attribute
    `spline` gives the derivatives of the same Hermite interpolant
    (`LogPotentialSlopes`).

    Parameters
    ----------
    profile : TruncatedProfile
        The profile.
    p_0 : float, optional
        The model's top energy :math:`P_0`, the reference of `depth`.
        Default: ``nan``, with which `depth` returns NaN at every radius.

    Attributes
    ----------
    spline : LogPotentialSlopes
        The derivatives of :math:`\ln\mathcal{P}` in ln r and their knots.
    R_knots : tuple of float
        The radius of the profile's first row and the largest radius of the
        profile's table (``profile.r_table_max``).
    """

    def __init__(self, profile: TruncatedProfile, p_0: float = math.nan) -> None:
        self.profile = profile
        self.spline = LogPotentialSlopes(profile)
        self.R_knots = (float(profile.r[0]), float(profile.r_table_max))
        self.p_0 = float(p_0)
        # The depths u = P_0 - Phi of the solve's own column u_trunc, kept on
        # the inner rows where 0 < u < P_0 / 2. On the rows with u < P_0 / 2,
        # the difference P_0 - Phi loses digits. The exact slope of ln u is
        # d ln u / d ln r = M / (r u), with M the enclosed mass.
        n = profile._n_interior
        u = np.asarray(profile.u_trunc, dtype=np.float64)
        self._deep = None
        if np.isfinite(self.p_0) and u.size >= n:
            r = np.asarray(profile.r[:n], dtype=np.float64)
            mass = np.asarray(profile.mass_trunc[:n], dtype=np.float64)
            deep = np.flatnonzero((u[:n] > 0.0) & (u[:n] < 0.5 * self.p_0))
            if deep.size >= 2 and deep[-1] == deep.size - 1:
                u, r, mass = u[deep], r[deep], mass[deep]
                self._deep = (np.log(r), np.log(u), mass / (r * u), float(r[-1]))

    def __call__(self, r: ArrayLike) -> Array:
        return self.profile.potential(np.atleast_1d(np.asarray(r, dtype=np.float64)))

    def depth(self, r: ArrayLike) -> Array:
        r"""Return the depth :math:`u = P_0 - \Phi` below the top energy at radii.

        :math:`\Phi` is the relative potential of the profile. The deep rows
        are the profile's first rows, each with :math:`0 < u < P_0/2` in the
        solve's column ``u_trunc``. At radii up to the outermost deep row, the
        depth is the cubic Hermite interpolant in ln r of ln u through the
        deep rows, with the exact slopes :math:`M/(r u)`. The interpolant
        keeps the digits that the difference :math:`P_0 - \Phi` loses near
        the center. At larger radii the depth is the difference
        :math:`P_0 - \Phi`. The difference is used at every radius where
        :math:`P_0` is NaN or fewer than two deep rows exist.
        """
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        out = self.p_0 - self(r)
        if self._deep is None:
            return out
        log_r, log_u, slope, r_end = self._deep
        inside = r <= r_end
        if np.any(inside):
            log_d, _ = hermite(np.log(r[inside]), log_r, log_u, slope)
            out[inside] = np.exp(log_d)
        return out


class DfReading:
    r"""The DF F at energies from a prepared lookup, scaled by a constant.

    A call returns :math:`10^{\log_{10}F - c}`, with c the shift. The value is
    0 where the lookup gives no finite :math:`\log_{10}F`.

    Parameters
    ----------
    arrays : LookupArrays
        The arrays of the prepared lookup (`prepare_lookup`).
    kinds : LookupKinds
        The rules of the prepared lookup (`prepare_lookup`).
    shift : float
        The shift c of :math:`\log_{10}F`.
    """

    def __init__(self, arrays: LookupArrays, kinds: LookupKinds, shift: float) -> None:
        self.arrays, self.kinds, self.shift = arrays, kinds, float(shift)

    def __call__(self, q: ArrayLike, u: ArrayLike | None = None) -> Array:
        """Return the scaled F at the energies `q`.

        Where the depths ``u = P_0 - q`` are given and finite, the lookup reads
        F by the depth u instead of by q (`evaluate_lookup`).
        """
        log_f = evaluate_lookup(self.arrays, self.kinds, q, u=u)
        return np.where(np.isfinite(log_f), 10.0 ** (log_f - self.shift), 0.0)

    def log10(self, q: ArrayLike) -> Array:
        """Return the base-10 logarithm of F at the energies `q`, without the shift."""
        return evaluate_lookup(self.arrays, self.kinds, q)


def _cumulative_at_order(
    arrays: LookupArrays, kinds: LookupKinds, q: float, order: int
) -> float:
    r"""Return :math:`G(Q)` for Q = `q` at one Gauss-Legendre order.

    The value is the lookup's `below_floor_mass` at the floor :math:`K_0`
    plus Gauss-Legendre panels of `order` nodes from :math:`K_0` to Q. The
    panel edges are :math:`K_0`, the lookup's nodes between :math:`K_0` and Q,
    Q/2 where Q/2 lies above :math:`K_0`, and Q. A panel that ends at or
    below Q/2 is integrated in ln K. A panel above Q/2 is integrated in
    ln(Q - K). The panel that ends at Q is cut where Q - K falls below
    :math:`e^{-60}` times the value of Q - K at the panel's lower edge.
    """
    n = int(arrays.n)
    k_nodes = np.asarray(arrays.k[:n], dtype=np.float64)
    k_0 = float(arrays.params[K_0])
    below = float(below_floor_mass(arrays, kinds, k_0)[0])
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
    a, b = nodes[:-1], nodes[1:]
    keep = b > a
    a, b = a[keep], b[keep]
    xg, wg = np.polynomial.legendre.leggauss(order)
    low = b <= half
    total = 0.0
    if np.any(low):
        ua, ub = np.log(a[low]), np.log(b[low])
        um, uw = 0.5 * (ua + ub), 0.5 * (ub - ua)
        u = um[:, None] + uw[:, None] * xg[None, :]
        k = np.clip(np.exp(u), k_0, q)
        ln_f = evaluate_lookup(arrays, kinds, k) * math.log(10.0)
        total += float(np.sum(uw[:, None] * wg[None, :] * np.exp(ln_f + u)))
    if np.any(~low):
        ah, bh = a[~low], b[~low]
        ta = np.log(q - ah)
        with np.errstate(divide="ignore"):
            tb = np.log(np.maximum(q - bh, 0.0))
        tb = np.where(np.isfinite(tb) & (tb > ta - 60.0), tb, ta - 60.0)
        hm, hw = 0.5 * (ta + tb), 0.5 * (ta - tb)
        t = hm[:, None] + hw[:, None] * xg[None, :]
        k = np.clip(q - np.exp(t), k_0, q)
        ln_f = evaluate_lookup(arrays, kinds, k) * math.log(10.0)
        total += float(np.sum(hw[:, None] * wg[None, :] * np.exp(ln_f + t)))
    return below + total


def cumulative_df(arrays: LookupArrays, kinds: LookupKinds, q: float) -> float:
    r"""Return :math:`G(Q) = \int_0^Q F\,{\rm d}K` from a prepared lookup.

    At Q at or below the lookup's floor energy :math:`K_0`, G is the lookup's
    `below_floor_mass` at Q. Above :math:`K_0`, G is
    `_cumulative_at_order`. The order starts at `CUMULATIVE_FIRST_ORDER` and
    doubles up to `CUMULATIVE_LAST_ORDER`. The function returns the value of
    the first order whose relative difference from the previous order is at
    or below `CUMULATIVE_TOLERANCE` (:math:`10^{-7}`).

    Raises
    ------
    AccuracyError
        If orders `CUMULATIVE_LAST_ORDER` / 2 and `CUMULATIVE_LAST_ORDER`
        differ by more than `CUMULATIVE_TOLERANCE` relative to the value.
    """
    q = float(q)
    if q <= float(arrays.params[K_0]):
        return float(below_floor_mass(arrays, kinds, q)[0])
    order = CUMULATIVE_FIRST_ORDER
    value = _cumulative_at_order(arrays, kinds, q, order)
    while order < CUMULATIVE_LAST_ORDER:
        order *= 2
        new = _cumulative_at_order(arrays, kinds, q, order)
        if abs(new - value) <= CUMULATIVE_TOLERANCE * abs(new):
            return new
        last, value = value, new
    raise unconverged_cumulative(q, value, last)


def unconverged_cumulative(q: float, value: float, previous: float) -> AccuracyError:
    """Return the AccuracyError of a cumulative integral G(Q) that does not converge.

    The message gives Q, the orders `CUMULATIVE_LAST_ORDER` / 2 and
    `CUMULATIVE_LAST_ORDER`, and the relative difference between `value` and
    `previous`, the values at orders `CUMULATIVE_LAST_ORDER` / 2 and
    `CUMULATIVE_LAST_ORDER`.
    """
    return AccuracyError(
        f"The DF's cumulative integral G(Q) at Q = {q:.6e} is not converged. The "
        f"values of G(Q) at the orders {CUMULATIVE_LAST_ORDER // 2} and "
        f"{CUMULATIVE_LAST_ORDER} differ by the relative amount "
        f"{abs(value - previous) / max(abs(value), 1e-300):.3e}, more than the "
        f"tolerance {CUMULATIVE_TOLERANCE:g}. The order cap {CUMULATIVE_LAST_ORDER} "
        f"and the tolerance {CUMULATIVE_TOLERANCE:g} are constants of the method, "
        f"and no setting moves the order cap or the tolerance."
    )


def _edge_gamma(arrays: LookupArrays, kinds: LookupKinds, p_edge: float) -> float:
    r"""Return the least :math:`QF(Q)/G(Q)` near the potential at the profile's edge.

    :math:`\mathcal{P}_{\rm edge}` = `p_edge` is the potential at the
    profile's last row of support. The ratio is taken at 13 energies Q
    spaced evenly in log Q from :math:`10^{-6}\,\mathcal{P}_{\rm edge}` to
    :math:`\mathcal{P}_{\rm edge}`, with F unscaled from the lookup and G
    from `cumulative_df`. For :math:`G \propto Q^\gamma` the ratio equals
    :math:`\gamma`. The least ratio serves as the exponent :math:`\gamma` of G
    near the edge energy. The result is NaN where no ratio is finite and
    positive.

    Raises
    ------
    AccuracyError
        If `cumulative_df` raises AccuracyError at one of the 13 energies.
    """
    q = p_edge * np.logspace(math.log10(TIER_BOUNDS[0]), 0.0, 13)
    g = np.array(
        [
            x
            * 10.0 ** float(evaluate_lookup(arrays, kinds, np.array([x]))[0])
            / cumulative_df(arrays, kinds, x)
            for x in q
        ]
    )
    g = g[np.isfinite(g) & (g > 0.0)]
    return float(g.min()) if g.size else float("nan")


def ellipse_model(
    model: Model, rules: EllipseRules, arrays: LookupArrays, kinds: LookupKinds
) -> EllipseModel:
    r"""Return the constants of a model that the ellipse route reads.

    Parameters
    ----------
    model : Model
        The model, of one component.
    rules : EllipseRules
        The route's rules.
    arrays : LookupArrays
        The arrays of the model's prepared lookup.
    kinds : LookupKinds
        The rules of the model's prepared lookup.

    Returns
    -------
    constants : EllipseModel
        The anisotropy, the layer counts and the velocity breaks V_edge and
        V_c of the model.

    Raises
    ------
    ResultError
        If `layer_counts` raises ResultError.
    AccuracyError
        If `cumulative_df` raises AccuracyError in `_edge_gamma`.

    Notes
    -----
    The layer counts toward the L = 0 point follow from
    :math:`\beta_{\rm c}` alone (`layer_counts`).

    The support of the profile is the set of rows with positive potential
    :math:`\mathcal{P}`, positive density and positive radius.
    :math:`\mathcal{P}_{\rm edge}` is the potential at the last row of
    support. V_edge is a velocity break where the profile is not spatially
    finite and :math:`\mathcal{P}_{\rm edge} > 0`. The layer count toward
    V_edge is the least count n with
    :math:`n \ge [\ln(10^{-6})/(\gamma + 2) - \ln 0.5]/\ln r_V`, and at
    least 0. Here :math:`\gamma` is `_edge_gamma` and :math:`r_V` is
    ``rules.vp_ratio``. Where :math:`\gamma` is NaN, the count is
    ``rules.vp_layers_kink``.

    The exponent of the turnover of D at V_c is

    .. math::

        p = -\kappa - (1 - b) - \frac{1 - 2b}{\alpha}.

    Here :math:`\kappa` is the exponent of :math:`F \sim (P_0 - Q)^\kappa`
    at the top energy :math:`P_0`. The exponent b of the L factor is
    :math:`\beta_{\rm c}` for the constant and COM classes and 0 otherwise.
    :math:`\alpha` is the least-squares slope of
    :math:`\ln(P_0 - \mathcal{P})` against ln r. The fit takes the rows of
    support with :math:`P_0 - \mathcal{P} > 0` whose radius is at most 10
    times the least radius of those rows. p is computed only where
    :math:`P_0` is finite, the fit has at least three rows and
    :math:`\kappa` is finite. V_c is a velocity break where
    :math:`(1/r_V - 1)\,p` exceeds ``rules.vp_nodes``.

    At the innermost row of support, with potential
    :math:`\mathcal{P}_{\rm in}`, the velocities are
    :math:`V_c = [2(P_0 - \mathcal{P}_{\rm in})]^{1/2}` and
    :math:`V_{\rm esc} = (2\mathcal{P}_{\rm in})^{1/2}`. Where
    :math:`0 < V_c < V_{\rm esc}`, the layer count toward V_c is the least
    count n with :math:`n \ge \ln[2V_c/(V_{\rm esc} - V_c)]/\ln r_V`, and at
    least 0. Otherwise the count is ``rules.vp_layers_zero``.
    """
    anisotropy = model.components[0].anisotropy
    profile = model.profile
    beta_c = float(anisotropy.beta_c)
    counts = layer_counts(beta_c, rules)
    r = np.asarray(profile.r, dtype=np.float64)
    p = np.asarray(profile.p_trunc, dtype=np.float64)
    rho = np.sum(np.asarray(profile.rho_trunc, dtype=np.float64), axis=0)
    support = (p > 0.0) & (rho > 0.0) & (r > 0.0)
    p_edge = float(p[support][-1])
    edge_alive = (not model.spatially_finite) and p_edge > 0.0
    edge_gamma = _edge_gamma(arrays, kinds, p_edge) if edge_alive else float("nan")
    if not edge_alive:
        vp_layers_edge = 0
    elif not np.isfinite(edge_gamma):
        vp_layers_edge = rules.vp_layers_kink
    else:
        vp_layers_edge = max(
            0,
            int(
                np.ceil(
                    (np.log(TIER_BOUNDS[0]) / (edge_gamma + 2.0) - np.log(0.5))
                    / np.log(rules.vp_ratio)
                )
            ),
        )
    p0_top = float(model.form.p_0)
    kappa_top = float(model.form.central_exponent[0])
    b = _kernel_beta(model)
    vc_p = float("nan")
    if np.isfinite(p0_top):
        ok = support & (p0_top - p > 0.0)
        r_ok, depth = r[ok], p0_top - p[ok]
        decade = r_ok <= 10.0 * np.min(r_ok)
        if decade.sum() >= 3 and np.isfinite(kappa_top):
            alpha = float(np.polyfit(np.log(r_ok[decade]), np.log(depth[decade]), 1)[0])
            vc_p = -kappa_top - (1.0 - b) - (1.0 - 2.0 * b) / alpha
    vc_alive = bool(
        np.isfinite(vc_p) and (1.0 / rules.vp_ratio - 1.0) * vc_p > rules.vp_nodes
    )
    vp_layers_c = 0
    if vc_alive:
        p_in = float(p[support][np.argmin(r[support])])
        vc, ve = np.sqrt(2.0 * max(p0_top - p_in, 0.0)), np.sqrt(2.0 * p_in)
        vp_layers_c = (
            max(0, int(np.ceil(np.log(2.0 * vc / (ve - vc)) / np.log(rules.vp_ratio))))
            if 0.0 < vc < ve
            else rules.vp_layers_zero
        )
    return EllipseModel(
        family=anisotropy.family,
        r_a=float(anisotropy.r_a),
        beta_c=beta_c,
        k_end=counts[0],
        k_end_floor=counts[1],
        k_end_v0=counts[2],
        k_end_v0_floor=counts[3],
        k_map=counts[4],
        edge_alive=bool(edge_alive),
        edge_gamma=float(edge_gamma),
        vp_layers_edge=int(vp_layers_edge),
        p0_top=p0_top,
        kappa_top=kappa_top,
        vc_p=float(vc_p),
        vc_alive=vc_alive,
        vp_layers_c=int(vp_layers_c),
    )


def floor_radius(model: Model) -> float:
    r"""Return the radius where the rows of a model end.

    The radius is ``profile.r_table_max``, the profile's last tabulated
    radius, unless the case below applies. The lookup's rule ``"zero"`` sets
    the DF to 0 below the floor energy :math:`K_0`. Under the rule
    ``"zero"``, where the potential at the last tabulated radius is below
    :math:`K_0`, the radius is instead the radius where
    :math:`\mathcal{P} = K_0`, found by 200 bisections in ln r. Beyond the
    radius where :math:`\mathcal{P} = K_0`, the DF is 0 along every line of
    sight.

    Raises
    ------
    ResultError
        If the rule is ``"zero"`` and the potential at the profile's first
        row is at or below :math:`K_0`.
    """
    profile = model.profile
    r_last = float(profile.r_table_max)
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    k_0 = float(arrays.params[K_0])

    def p(log_r: float) -> float:
        return float(np.asarray(profile.potential(np.array([math.exp(log_r)])))[0])

    if kinds.below_floor != "zero" or p(math.log(r_last)) >= k_0:
        return r_last
    lo, hi = math.log(float(profile.r[0])), math.log(r_last)
    if p(lo) <= k_0:
        raise ResultError(
            f"The DF is zero below the DF's floor K_0 = {k_0:.6e} (the rule "
            f"'zero'), which the potential reaches at or inside the profile's first "
            f"tabulated radius r = {math.exp(lo):.6e}, so the model has no line of "
            f"sight to tabulate."
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p(mid) > k_0:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def ellipse_route(
    model: Model, rules: EllipseRules, with_df: bool = True
) -> EllipseRoute:
    """Return the ellipse route of a model of one component.

    The route reads the model alone: the potential of the model's profile
    (`PotentialReading`), the lookup of the model's DF (`EllipseDF`), the
    constants of `ellipse_model` and the end radius of `floor_radius`.

    Parameters
    ----------
    model : Model
        The model.
    rules : EllipseRules
        The route's rules.
    with_df : bool, optional
        True builds a route that carries the DF and computes rows. False
        builds a route without the DF, which places velocity breaks only.
        Default: ``True``.

    Returns
    -------
    route : EllipseRoute
        The route of the model.

    Raises
    ------
    InputError
        If the model has more than one component.
    InputError
        If `with_df` is True and the exponent p of ``F ~ K^p`` at K = 0 is
        not above -1 (`EllipseDF`).
    ResultError
        If the potential reaches the DF's floor energy at or inside the
        profile's first row under the lookup's rule ``"zero"``
        (`floor_radius`), or if `layer_counts` raises ResultError.
    AccuracyError
        If `cumulative_df` raises AccuracyError.
    """
    if len(model.components) != 1:
        raise InputError(
            f"The line-of-sight velocity distributions are computed for a model of "
            f"one component; the model given has {len(model.components)} components."
        )
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    constants = ellipse_model(model, rules, arrays, kinds)
    df = None
    if with_df:
        n = int(arrays.n)
        log_df = np.asarray(arrays.log_df[:n], dtype=np.float64)
        # The route carries the DF as 10^(log10 F - shift). The shift keeps the
        # largest tabulated value of the carried DF at or below
        # 10^LOG10_DF_CEILING. The shift cancels in every row normalized to
        # unit integral.
        shift = max(0.0, float(np.max(log_df[np.isfinite(log_df)])) - LOG10_DF_CEILING)
        f = DfReading(arrays, kinds, shift)
        # EllipseDF takes the table's first and last written energies, at the
        # indices i_first and i_last, as its floor Q_0 and top Q_top. The
        # branches of G split at the floor Q_0. Table energies below index i_first
        # lie on the lower branch of G, with the tail below the floor.
        table = model.table
        written = np.asarray(
            table.k[0, [table.i_first[0], table.i_last[0]]], dtype=np.float64
        )
        df = EllipseDF(
            written,
            f,
            f.log10,
            _Cumulative(arrays, kinds),
            {
                "P0": float(model.form.p_0),
                "below_floor": kinds.below_floor,
                "tail_a": float(model.form.tail_exponent[0]),
                "tail_k": float(model.form.tail_log_exponent[0]),
                "edge_law_p": (
                    float(model.form.below_floor_coefficients[0][0])
                    if kinds.below_floor == "edge_law"
                    else 0.0
                ),
                "kappa": float(model.form.central_exponent[0]),
            },
            rules,
        )
    return EllipseRoute(
        PotentialReading(model.profile, float(model.form.p_0)),
        floor_radius(model),
        constants,
        df,
        rules,
    )


class _Cumulative:
    """A picklable callable that returns `cumulative_df` on one prepared lookup."""

    def __init__(self, arrays: LookupArrays, kinds: LookupKinds) -> None:
        self.arrays, self.kinds = arrays, kinds

    def __call__(self, q: float) -> float:
        return cumulative_df(self.arrays, self.kinds, q)
