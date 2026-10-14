"""The ellipse route's readers of a model: its potential with the slopes the
extrema of a line of sight are bounded on, its DF with the cumulative
integral, and the constants of the model the route's rules read."""

__all__ = [
    "DfReading",
    "LogPotentialSlopes",
    "PotentialReading",
    "cumulative_df",
    "ellipse_model",
    "ellipse_route",
]

import math

import numpy as np

from ..df import Model
from ..df.lookup import (
    K_0,
    LookupArrays,
    LookupKinds,
    below_floor_mass,
    evaluate_lookup,
    prepare_lookup,
)
from ..exceptions import InputError
from ..profiles import TruncatedProfile
from ..typing import Array, ArrayLike
from ..validation import GOAL
from ._ellipse import EllipseRoute
from ._ellipse_df import EllipseDF
from ._ellipse_rules import EllipseModel, EllipseRules, layer_counts
from ._tables import _kernel_beta

# The cumulative integral's Gauss-Legendre order starts here and doubles until
# two orders agree to this fraction of the goal, or the order reaches the cap.
CUMULATIVE_FIRST_ORDER = 16
CUMULATIVE_LAST_ORDER = 512
CUMULATIVE_TOLERANCE = 0.1 * GOAL
# The DF is carried as 10^(log10 F - shift), the shift keeping its largest
# value at most 10^this (it cancels in every normalized row).
LOG10_DF_CEILING = 300.0


class LogPotentialSlopes:
    r"""The first and second derivatives of :math:`\ln\mathcal{P}` in
    :math:`y = \ln r` of a profile's own potential, interval by interval.

    Inside the rows :math:`\ln\mathcal{P}` is the profile's cubic Hermite in
    :math:`y` (`TruncatedProfile.potential`), so on each interval the first
    derivative is a quadratic and the second is linear, the form the bound of
    `EllipseRoute.f_extrema` rests on. On the Kepler interval of a profile
    with an edge row, :math:`\mathcal{P} = M_t(1/r - 1/r_t)`, the first
    derivative :math:`-r_t/(r_t - r)` and the second
    :math:`-r_t r/(r_t - r)^2` are monotone; both go to minus infinity at the
    edge.

    Parameters
    ----------
    profile : TruncatedProfile
        The profile.

    Attributes
    ----------
    x : array
        The knots in :math:`\ln r`: the rows, and the edge where the profile
        has an edge row.
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
        """The `nu`-th derivative at `y`, from the interval that starts at or
        below `y` (the right-hand limit at a knot)."""
        return self._read(y, nu, "right")

    def left(self, y: ArrayLike, nu: int) -> Array:
        """The `nu`-th derivative at `y`, from the interval that ends at or
        above `y` (the left-hand limit at a knot)."""
        return self._read(y, nu, "left")

    def both(self, y: ArrayLike, side: str = "right") -> tuple[Array, Array]:
        """The first and second derivatives at `y` from one lookup of the
        interval, each the number `__call__` (``side="right"``) or `left`
        (``side="left"``) gives."""
        d1, d2 = self._read_many(y, (1, 2), side)
        return d1, d2

    def _read(self, y: ArrayLike, nu: int, side: str) -> Array:
        if nu not in (1, 2):
            raise ValueError(f"nu = {nu}: only the first and second derivatives.")
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
        )
        h = x[i + 1] - x[i]
        t = (y - x[i]) / h
        y0, y1 = self._log_p[i], self._log_p[i + 1]
        m0, m1 = self._slope[i] * h, self._slope[i + 1] * h
        if self._edge:
            # the Kepler interval starts at the last row: its right-hand limit
            # there is Kepler's, its left-hand limit the Hermite's
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
    """The potential of a profile at radii, as the ellipse route reads it.

    The rows' one reading of the potential: its values are
    the profile's own (`TruncatedProfile.potential`), its derivatives those of
    the same Hermite (`LogPotentialSlopes`, from the profile's interpolation
    tables), and nothing else in the rows holds a potential. The one clip left
    is `EllipseRoute.f_extrema`'s of ln R to the knots, behind its refusal of a
    line of sight outside them: it takes the end's rounding (R_2D cosh(arccosh(
    R_t / R_2D)) can exceed R_t by one unit in the last place) to R_t.

    Parameters
    ----------
    profile : TruncatedProfile
        The profile.

    Attributes
    ----------
    spline : LogPotentialSlopes
        The derivatives of ln P in ln r on each interval, and its knots.
    R_knots : tuple of float
        The first row and the end of the lines of sight.
    """

    def __init__(self, profile: TruncatedProfile) -> None:
        self.profile = profile
        self.spline = LogPotentialSlopes(profile)
        self.R_knots = (float(profile.r[0]), float(profile.r_table_max))

    def __call__(self, r: ArrayLike) -> Array:
        return self.profile.potential(np.atleast_1d(np.asarray(r, dtype=np.float64)))


class DfReading:
    r"""F at energies from a prepared lookup, as
    :math:`10^{\log_{10}F - c}`, zero where the DF has no value.

    Parameters
    ----------
    arrays, kinds
        The prepared lookup (`prepare_lookup`).
    shift : float
        The shift c.
    """

    def __init__(self, arrays: LookupArrays, kinds: LookupKinds, shift: float) -> None:
        self.arrays, self.kinds, self.shift = arrays, kinds, float(shift)

    def __call__(self, q: ArrayLike) -> Array:
        log_f = evaluate_lookup(self.arrays, self.kinds, q)
        return np.where(np.isfinite(log_f), 10.0 ** (log_f - self.shift), 0.0)

    def log10(self, q: ArrayLike) -> Array:
        """log10 F, without the shift."""
        return evaluate_lookup(self.arrays, self.kinds, q)


def _cumulative_at_order(
    arrays: LookupArrays, kinds: LookupKinds, q: float, order: int
) -> float:
    r"""The panels of `cumulative_df` at one Gauss-Legendre order."""
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

    At or below the table's first energy :math:`K_0` it is the lookup's own
    `below_floor_mass`. Above it, that at :math:`K_0` plus Gauss-Legendre
    panels between the lookup's nodes: in :math:`\ln K` up to :math:`Q/2`, in
    :math:`\ln(Q - K)` above, the order doubling from
    `CUMULATIVE_FIRST_ORDER` until two orders agree to
    `CUMULATIVE_TOLERANCE`. The working code's rule (``floor_term``), here
    until the DF's lookup states its own cumulative integral.
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
        value = new
    return value


def _edge_gamma(arrays: LookupArrays, kinds: LookupKinds, p_edge: float) -> float:
    r"""The least :math:`QF/G` over Q from GOAL :math:`\mathcal{P}_{\rm edge}`
    to :math:`\mathcal{P}_{\rm edge}`; NaN where G vanishes there."""
    q = p_edge * np.logspace(math.log10(GOAL), 0.0, 13)
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
    r"""Return the constants of a model that the ellipse route's rules read.

    Parameters
    ----------
    model : Model
        The model, of one component.
    rules : EllipseRules
        The route's rules.
    arrays, kinds
        The model's prepared lookup.

    Returns
    -------
    constants : EllipseModel

    Notes
    -----
    The layers toward the L = 0 point follow from :math:`\beta_c` alone
    (`layer_counts`). Where the profile is not spatially finite and its last
    row has :math:`\mathcal{P} > 0`, the line of sight's end :math:`V_{\rm
    edge}` is a break, its layers from the exponent :math:`\gamma` of G there.
    :math:`V_c` is a break where D's turnover exponent
    :math:`p = -\kappa - (1 - b) - (1 - 2b)/\alpha` exceeds what a sub-panel
    resolves, :math:`\alpha` the slope of :math:`\ln(P_0 - \mathcal{P})`
    against :math:`\ln r` over the profile's innermost decade.
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
                    (np.log(GOAL) / (edge_gamma + 2.0) - np.log(0.5))
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
        decade = r_ok <= 10.0 * r_ok.min()
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


def ellipse_route(
    model: Model, rules: EllipseRules, with_df: bool = True
) -> EllipseRoute:
    """Return the ellipse route of a model of one component, from the model
    alone: its profile's potential, its DF's lookup and `ellipse_model`.

    Parameters
    ----------
    model : Model
        The model.
    rules : EllipseRules
        The route's rules.
    with_df : bool, optional
        Whether the route carries the DF; without it, it places breaks only.

    Raises
    ------
    InputError
        For a model of more than one component.
    """
    if len(model.components) != 1:
        raise InputError(
            f"The line-of-sight velocity distributions are computed for a model of "
            f"one component; this one has {len(model.components)}."
        )
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    constants = ellipse_model(model, rules, arrays, kinds)
    df = None
    if with_df:
        n = int(arrays.n)
        log_df = np.asarray(arrays.log_df[:n], dtype=np.float64)
        shift = max(0.0, float(np.max(log_df[np.isfinite(log_df)])) - LOG10_DF_CEILING)
        f = DfReading(arrays, kinds, shift)
        # G's branches split at the table's own first energy: the nodes the
        # inversion records below it lie on the lower branch, as the tail does
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
        PotentialReading(model.profile),
        float(model.profile.r_table_max),
        constants,
        df,
        rules,
    )


class _Cumulative:
    """`cumulative_df` on one lookup, as a picklable callable."""

    def __init__(self, arrays: LookupArrays, kinds: LookupKinds) -> None:
        self.arrays, self.kinds = arrays, kinds

    def __call__(self, q: float) -> float:
        return cumulative_df(self.arrays, self.kinds, q)
