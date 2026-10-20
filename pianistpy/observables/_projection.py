"""Surface density and line-of-sight dispersion over projected radius, and the
reading of their table between rows."""

__all__ = [
    "DispersionReading",
    "SurfaceDensityReading",
    "certify_projection",
    "dispersion_table_reader",
    "edge_variable",
    "edge_variable_inverse",
    "project",
    "project_tangent",
    "surface_density_reader",
]

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ._dispersion import beta_sigma, beta_sigma_tangent

# The smallest value taken a logarithm of, against an underflow to zero.
TINY = 1e-300
# The relative tolerance by which a radius at a table's end, computed on
# another grid, is still that end: grids ending at the same radius differ by
# units in the last place.
END_TOLERANCE = 1e-12


def spline_value(breaks: Any, coefficients: Any, x: Any, xp: Any = np) -> Any:
    """Return a cubic of SciPy's piecewise form at `x`, under either array
    module: each interval's own polynomial, the end ones continued beyond
    the breakpoints."""
    breaks = xp.asarray(breaks)
    c = xp.asarray(coefficients)
    i = xp.clip(xp.searchsorted(breaks, x, side="right") - 1, 0, breaks.size - 2)
    dx = x - breaks[i]
    return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]


def _line_of_sight_nodes(n_nodes: int) -> tuple[Array, Array, Array, Array]:
    """Return the unit Gauss-Legendre nodes and weights of the two panels."""
    n_a = n_nodes // 2
    n_b = n_nodes - n_a
    x_a, w_a = np.polynomial.legendre.leggauss(n_a)
    x_b, w_b = np.polynomial.legendre.leggauss(n_b)
    return 0.5 * (x_a + 1.0), 0.5 * w_a, 0.5 * (x_b + 1.0), 0.5 * w_b


def _line_of_sight_rule(
    r_2d_rows: Array, r_t: float, nodes: tuple[Array, Array, Array, Array]
) -> tuple[Array, Array]:
    """Return the radii of the line-of-sight nodes of a chunk of projected
    radii, shape (rows, nodes), and their weights in u (``du = R dt``)."""
    t_a, w_a, t_b, w_b = nodes
    u_max = np.sqrt(np.maximum(r_t**2 - r_2d_rows**2, 0.0))
    t_max = np.arcsinh(u_max / r_2d_rows)
    t_h = np.maximum(
        np.arccosh(np.maximum(r_2d_rows * np.cosh(t_max) / (2.0 * r_2d_rows), 1.0)),
        0.5 * t_max,
    )
    t = np.concatenate([t_h * t_a, t_h + (t_max - t_h) * t_b], axis=1)
    w = np.concatenate([t_h * w_a, (t_max - t_h) * w_b], axis=1)
    r = r_2d_rows * np.cosh(t)
    return r, w * r


def project(
    r_2d: ArrayLike,
    density: Callable[[Array], Array],
    sigma_r2: Callable[[Array], Array],
    r_t: float,
    anisotropy: Anisotropy,
    n_nodes: int,
    chunk: int = 256,
) -> tuple[Array, Array]:
    r"""Return the surface density and the squared line-of-sight dispersion.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radius, in internal units.
    density, sigma_r2 : callable
        The density and :math:`\sigma_r^2` at radii.
    r_t : float
        The radius where every line of sight ends.
    anisotropy : Anisotropy
        The anisotropy of the component.
    n_nodes : int
        Gauss-Legendre nodes of each line-of-sight integral (at least 20).
    chunk : int, optional
        Projected radii computed together.

    Returns
    -------
    surface_density, sigma_los2 : array, shape (n,)
        :math:`\Sigma` and :math:`\sigma_{\rm los}^2`, in internal units;
        :math:`\sigma_{\rm los}^2` is zero where :math:`\Sigma` is.

    Raises
    ------
    ResultError
        If a value of :math:`\sigma_{\rm los}^2` is not finite.

    Notes
    -----
    .. math::

        \Sigma = 2\int_0^{u_{\max}}\varrho\,{\rm d}u, \qquad
        \Sigma\sigma_{\rm los}^2 = 2\int_0^{u_{\max}}\sigma_{r}^2\varrho
        \Big(1 - \beta_\sigma\frac{R_{\rm 2D}^2}{R^2}\Big)\,{\rm d}u,

    with :math:`u_{\max} = (R_t^2 - R_{\rm 2D}^2)^{1/2}`, in
    ``t = asinh(u / R_2D)`` (``R = R_2D cosh t``, ``du = R dt``) on two panels
    of half the nodes each, split at
    ``t_h = max(acosh(R_hi / (2 R_2D)), t_max / 2)`` with ``R_hi = R_2D cosh t_max``:
    a density cusp of width ``R_2D`` is smooth in ``t``, and the outer half of the
    radius range keeps its own panel.

    References
    ----------
    Chiang+(2026d), Eqs. (B45), (B46) [eqn:los_sigma_1].
    """
    if not n_nodes >= 20:
        raise InputError(f"n_nodes must be at least 20; got {n_nodes}.")
    nodes = _line_of_sight_nodes(n_nodes)
    r_2d = np.asarray(r_2d, dtype=np.float64)
    surface_density = np.empty(r_2d.size)
    sigma_los2 = np.empty(r_2d.size)
    for i0 in range(0, r_2d.size, chunk):
        r_2d_rows = r_2d[i0 : i0 + chunk, None]
        r, weights = _line_of_sight_rule(r_2d_rows, r_t, nodes)
        rho = density(r.ravel()).reshape(r.shape)
        sr2rho = sigma_r2(r.ravel()).reshape(r.shape) * rho
        beta = beta_sigma(r, anisotropy)
        sigma_rows = 2.0 * np.sum(weights * rho, axis=1)
        term1 = 2.0 * np.sum(weights * sr2rho, axis=1)
        term2 = (
            2.0 * r_2d_rows[:, 0] ** 2 * np.sum(weights * beta * sr2rho / r**2, axis=1)
        )
        surface_density[i0 : i0 + chunk] = sigma_rows
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_los2[i0 : i0 + chunk] = np.where(
                sigma_rows > 0.0, (term1 - term2) / sigma_rows, 0.0
            )

    if not np.all(np.isfinite(sigma_los2)):
        raise ResultError(
            f"{int(np.sum(~np.isfinite(sigma_los2)))} values of the squared "
            f"line-of-sight dispersion are not finite."
        )
    log.debug(
        "Projected %d radii: %d non-decreasing steps of the surface density, "
        "%d negative squared dispersions.",
        r_2d.size,
        int(np.sum(np.diff(surface_density) >= 0)),
        int(np.sum(sigma_los2 < 0.0)),
    )
    return surface_density, sigma_los2


def project_tangent(
    r_2d: ArrayLike,
    density: Callable[[Array], Array],
    sigma_r2: Callable[[Array], Array],
    tangent: Callable[[Array], Array],
    r_t: float,
    anisotropy: Anisotropy,
    n_nodes: int,
    parameter: str,
    chunk: int = 256,
) -> Array:
    r"""Return the derivative of the squared line-of-sight dispersion in one
    anisotropy parameter, on a fixed profile, at the nodes of `project`.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radius, in internal units.
    density, sigma_r2 : callable
        The density and :math:`\sigma_r^2` at radii, as given to `project`.
    tangent : callable
        :math:`\partial\sigma_r^2/\partial\theta` at radii
        (`radial_integral_tangent_reader`).
    r_t : float
        The radius where every line of sight ends.
    anisotropy : Anisotropy
        The anisotropy of the component.
    n_nodes : int
        Gauss-Legendre nodes of each line-of-sight integral, as for `project`.
    parameter : {"beta_c", "r_a"}
        The parameter, one of the family's `TANGENT_PARAMETERS`.
    chunk : int, optional
        Projected radii computed together.

    Returns
    -------
    tangent : array, shape (n,)
        :math:`\partial\sigma_{\rm los}^2/\partial\theta`, in internal
        units; zero where :math:`\Sigma` is.

    Notes
    -----
    :math:`\Sigma` does not move, so

    .. math::

        \Sigma\frac{\partial\sigma_{\rm los}^2}{\partial\theta} = 2\int_0^{u_{\max}}
        \varrho\Big[\frac{\partial\sigma_{r}^2}{\partial\theta}\Big(1 -
        \beta_\sigma\frac{R_{\rm 2D}^2}{R^2}\Big) - \sigma_{r}^2
        \frac{\partial\beta_\sigma}{\partial\theta}\frac{R_{\rm 2D}^2}{R^2}
        \Big]\,{\rm d}u

    on the nodes of `project` at the same projected radii.
    """
    nodes = _line_of_sight_nodes(n_nodes)
    r_2d = np.asarray(r_2d, dtype=np.float64)
    out = np.empty(r_2d.size)
    for i0 in range(0, r_2d.size, chunk):
        r_2d_rows = r_2d[i0 : i0 + chunk, None]
        r, weights = _line_of_sight_rule(r_2d_rows, r_t, nodes)
        rho = density(r.ravel()).reshape(r.shape)
        sr2rho = sigma_r2(r.ravel()).reshape(r.shape) * rho
        d_sr2rho = tangent(r.ravel()).reshape(r.shape) * rho
        beta = beta_sigma(r, anisotropy)
        d_beta = beta_sigma_tangent(r, anisotropy, parameter)
        sigma_rows = 2.0 * np.sum(weights * rho, axis=1)
        d_term1 = 2.0 * np.sum(weights * d_sr2rho, axis=1)
        d_term2 = (
            2.0
            * r_2d_rows[:, 0] ** 2
            * np.sum(weights * (beta * d_sr2rho + d_beta * sr2rho) / r**2, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i0 : i0 + chunk] = np.where(
                sigma_rows > 0.0, (d_term1 - d_term2) / sigma_rows, 0.0
            )
    return out


def edge_variable(r: ArrayLike, r_t: float) -> Array:
    """Return ``x = ln(R / (1 - R / R_t))``: ln R far inside the edge,
    ``-ln(1 - R / R_t)`` at it."""
    r = np.asarray(r, dtype=np.float64)
    return np.log(r) - np.log1p(-r / r_t)


def edge_variable_inverse(x: ArrayLike, r_t: float) -> Array:
    """Return R from ``x = ln(R / (1 - R / R_t))``."""
    e = np.exp(np.asarray(x, dtype=np.float64))
    return r_t * e / (r_t + e)


def _refuse_below(r_2d: Array, r_min: float, what: str) -> None:
    """Refuse a projected radius inside a table's innermost row."""
    below = r_2d < r_min * (1.0 - END_TOLERANCE)
    if np.any(below):
        raise InputError(
            f"The {what} is asked for at R_2D = {float(np.min(r_2d[below])):.4e}, "
            f"inside the innermost projected radius of its table {r_min:.4e}; the "
            f"table is not extrapolated inward."
        )


class SurfaceDensityReading:
    """The reading of a table of the surface density between its rows.

    Built by `surface_density_reader`, whose notes it follows. Called with
    projected radii, it returns the surface density there.

    Attributes
    ----------
    r_t : float or None
        Where the edge factor vanishes; None where the table reaches it.
    var : str
        ``"x"`` or ``"lnR"``, the variable it interpolates in.
    r_min, r_max : float
        The first and last rows.
    inner_disk : float
        The estimated integral of Sigma R dR inside the first row.
    inner_note : str
        A sentence a refusal inside the first row adds; empty unless the
        caller cut the rows there.
    """

    def __init__(
        self,
        r_2d: ArrayLike,
        surface_density: ArrayLike,
        r_t: float,
        spatially_finite: bool,
    ) -> None:
        r_rows = np.asarray(r_2d, dtype=np.float64)
        sigma_rows = np.asarray(surface_density, dtype=np.float64)
        log_r_rows = np.log(np.maximum(r_rows, TINY))
        log_sigma = np.log(np.maximum(sigma_rows, TINY))
        r_min = float(r_rows[0])
        r_max = float(r_rows[-1])
        has_edge = r_t is not None and float(r_t) > r_max
        self._has_edge = has_edge
        # The edge, read only where there is one.
        self._edge = float(r_t) if has_edge else 0.0
        log_h = (
            np.log(np.maximum(sigma_rows / self._edge_factor(r_rows), TINY))
            if has_edge
            else log_sigma
        )
        self._use_x = spatially_finite and has_edge
        if self._use_x:
            self._spline = CubicSpline(edge_variable(r_rows, self._edge), log_h)
        else:
            self._spline = CubicSpline(log_r_rows, log_h)
        self.r_t: float | None = self._edge if has_edge else None
        self.var = "x" if self._use_x else "lnR"
        slope = float(
            np.clip(
                -(log_sigma[1] - log_sigma[0]) / (log_r_rows[1] - log_r_rows[0]),
                0.0,
                1.99,
            )
        )
        self.r_min = r_min
        self.r_max = r_max
        self.inner_disk = float(np.exp(log_sigma[0])) * r_min**2 / (2.0 - slope)
        # what a refusal inside the first row adds: the setting that placed it
        self.inner_note = ""

    def _edge_factor(self, r: Array) -> Array:
        if self._has_edge:
            return np.sqrt(np.maximum(self._edge * self._edge - r * r, 0.0))
        return np.ones_like(r)

    def _variable(self, r: Array) -> Array:
        if self._use_x:
            return edge_variable(r, self._edge)
        return np.log(np.maximum(r, TINY))

    def traced(self, r: Any, xp: Any = np) -> Any:
        """The same reading under either array module, `r` traceable: zero
        beyond the last row, the first rows' cubic continued inside the
        first (a caller refuses those radii before it traces)."""
        if self._use_x:
            variable = xp.log(r) - xp.log1p(-r / self._edge)
        else:
            variable = xp.log(xp.maximum(r, TINY))
        factor = (
            xp.sqrt(xp.maximum(self._edge * self._edge - r * r, 0.0))
            if self._has_edge
            else 1.0
        )
        value = xp.exp(spline_value(self._spline.x, self._spline.c, variable, xp))
        return xp.where(r > self.r_max * (1.0 + END_TOLERANCE), 0.0, value * factor)

    def __call__(self, r_any: ArrayLike) -> Array:
        r_any = np.atleast_1d(np.asarray(r_any, dtype=np.float64))
        _refuse_below(r_any, self.r_min, "surface density")
        log_s = self._spline(self._variable(r_any))
        return np.where(
            r_any > self.r_max * (1.0 + END_TOLERANCE),
            0.0,
            np.exp(log_s) * self._edge_factor(r_any),
        )


def surface_density_reader(
    r_2d: ArrayLike, surface_density: ArrayLike, r_t: float, spatially_finite: bool
) -> SurfaceDensityReading:
    r"""Return the reading of a table of the surface density between its rows.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radius of the rows, increasing, in internal units.
    surface_density : array-like, shape (n,)
        :math:`\Sigma` at the rows.
    r_t : float
        The radius where every line of sight ends.
    spatially_finite : bool
        Whether the model reaches its edge at a finite radius.

    Returns
    -------
    read : SurfaceDensityReading
        :math:`\Sigma` at projected radii: zero beyond the last row, refused
        inside the first. Its attributes: ``r_t`` (where the edge factor
        vanishes, None where the table reaches it), ``var`` (``"x"`` or
        ``"lnR"``, the variable it interpolates in), ``r_min``, ``r_max`` and
        ``inner_disk``.

    Notes
    -----
    The interpolant carries ``h = Sigma / (R_t^2 - R^2)^(1/2)``, smooth to
    the last row, and puts the factor back at evaluation. ln h is a
    not-a-knot cubic in ``x = ln(R / (1 - R / R_t))`` on a spatially finite
    model, where ``h`` goes as a power of ``R_t - R``, and in ln R otherwise.
    ``inner_disk`` estimates the part inside the first row from the
    innermost slope.
    """
    return SurfaceDensityReading(r_2d, surface_density, r_t, spatially_finite)


class DispersionReading:
    """The reading of a table of the squared line-of-sight dispersion.

    Built by `dispersion_table_reader`. Called with projected radii, it
    returns the value clipped at zero; `raw` returns it before the clip, and
    ``n_clipped`` and ``min_clipped`` count where the clip acted.
    """

    def __init__(self, r_2d: ArrayLike, sigma_los2: ArrayLike) -> None:
        r_rows = np.asarray(r_2d, dtype=np.float64)
        log_r_rows = np.log(np.maximum(r_rows, TINY))
        self._spline = CubicSpline(log_r_rows, sigma_los2, extrapolate=False)
        self._r_min = float(r_rows[0])
        self.n_clipped = 0
        self.min_clipped = 0.0

    def traced(self, r: Any, xp: Any = np) -> Any:
        """The same reading under either array module, `r` traceable: clipped
        at zero, zero beyond the last row, the first rows' cubic continued
        inside the first (a caller refuses those radii before it traces)."""
        log_r = xp.log(xp.maximum(r, TINY))
        value = spline_value(self._spline.x, self._spline.c, log_r, xp)
        value = xp.where(log_r > float(self._spline.x[-1]), 0.0, value)
        return xp.maximum(value, 0.0)

    def raw(self, r_any: ArrayLike) -> Array:
        """Return the value before the clip at zero."""
        r_any = np.atleast_1d(np.asarray(r_any, dtype=np.float64))
        _refuse_below(r_any, self._r_min, "squared line-of-sight dispersion")
        value = self._spline(np.log(np.maximum(r_any, TINY)))
        return np.where(np.isfinite(value), value, 0.0)

    def __call__(self, r_any: ArrayLike) -> Array:
        value = self.raw(r_any)
        n_negative = int(np.sum(value < 0.0))
        if n_negative:
            self.n_clipped += n_negative
            self.min_clipped = min(self.min_clipped, float(value.min()))
        return np.maximum(value, 0.0)


def dispersion_table_reader(
    r_2d: ArrayLike, sigma_los2: ArrayLike
) -> DispersionReading:
    r"""Return the reading of a table of :math:`\sigma_{\rm los}^2` between rows.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radius of the rows, increasing, in internal units.
    sigma_los2 : array-like, shape (n,)
        :math:`\sigma_{\rm los}^2` at the rows.

    Returns
    -------
    read : DispersionReading
        :math:`\sigma_{\rm los}^2` at projected radii, clipped at zero: zero
        beyond the last row, refused inside the first. ``read.raw`` gives the
        values before the clip; ``read.n_clipped`` and ``read.min_clipped``
        count where the clip acted.

    Notes
    -----
    A not-a-knot cubic in ln R, the interpolant the table checks its rows
    for (`certify_projection`).
    """
    return DispersionReading(r_2d, sigma_los2)


def certify_projection(
    r_2d: ArrayLike,
    surface_density: ArrayLike,
    sigma_los2: ArrayLike,
    r_t: float,
    spatially_finite: bool,
    rule: Callable[[Array], tuple[Array, Array]],
    tolerance: float,
) -> tuple[Array, Array, Array, dict[str, Any]]:
    """Return the rows of a projected table that reads itself between rows.

    Parameters
    ----------
    r_2d, surface_density, sigma_los2 : array-like, shape (n,)
        The table's rows, in increasing projected radius.
    r_t : float
        The radius where every line of sight ends.
    spatially_finite : bool
        Whether the model reaches its edge at a finite radius.
    rule : callable
        The surface density and the squared dispersion at projected radii, by
        the rule the rows come from (`project`).
    tolerance : float
        The largest relative difference allowed between the table read at an
        interval's midpoint and the rule there.

    Returns
    -------
    r_2d, surface_density, sigma_los2 : array
        The rows, with those the check added.
    record : dict
        ``end`` (``"tolerance"`` or ``"stalled"``), ``rounds``,
        ``rows_added``, ``worst_first`` (the surface density's and the
        dispersion's before), ``worst_last``, ``rule_radii``, ``seconds`` and
        ``var``.

    Notes
    -----
    At every interval's midpoint, in each reader's own variable, the readers
    of `surface_density_reader` and `dispersion_table_reader` are compared
    with the rule; where either misses the tolerance the rule's values there
    become a row. Round by round until none misses, or two rounds running do
    not halve the worst (the cubic is global, and a row added near the edge
    can leave the worst unchanged for a round before it falls). The rule is
    evaluated once at each radius.
    """
    r_rows, sigma_rows, los_rows = (
        np.array(a, dtype=np.float64) for a in (r_2d, surface_density, sigma_los2)
    )
    cache: dict[float, tuple[float, float]] = {}

    def rule_at(r_any: Array) -> tuple[Array, Array]:
        new = np.array([x for x in np.unique(r_any).tolist() if x not in cache])
        if new.size:
            a, b = rule(new)
            cache.update(
                zip(new.tolist(), zip(a.tolist(), b.tolist(), strict=True), strict=True)
            )
        values = np.array([cache[x] for x in r_any.tolist()])
        return values[:, 0], values[:, 1]

    t0 = time.perf_counter()
    worst: list[float] = []
    end, rounds, added, first = "tolerance", 0, 0, None
    while True:
        sigma_read = surface_density_reader(r_rows, sigma_rows, r_t, spatially_finite)
        los_read = dispersion_table_reader(r_rows, los_rows)
        if sigma_read.var == "x":
            mid_sigma = edge_variable_inverse(
                0.5
                * (edge_variable(r_rows[:-1], r_t) + edge_variable(r_rows[1:], r_t)),
                r_t,
            )
        else:
            mid_sigma = np.exp(0.5 * (np.log(r_rows[:-1]) + np.log(r_rows[1:])))
        mid_los = np.exp(0.5 * (np.log(r_rows[:-1]) + np.log(r_rows[1:])))
        sigma_rule, _ = rule_at(mid_sigma)
        _, los_rule = rule_at(mid_los)
        error_sigma = np.abs(sigma_read(mid_sigma) / sigma_rule - 1.0)
        error_los = np.abs(los_read.raw(mid_los) - los_rule) / np.maximum(
            np.abs(los_rule), TINY
        )
        worst.append(float(max(error_sigma.max(), error_los.max())))
        if first is None:
            first = (float(error_sigma.max()), float(error_los.max()))
        if worst[-1] <= tolerance:
            break
        if (
            len(worst) >= 3
            and worst[-1] > 0.5 * worst[-2]
            and worst[-2] > 0.5 * worst[-3]
        ):
            end = "stalled"
            break
        new = np.unique(
            np.concatenate(
                [mid_sigma[error_sigma > tolerance], mid_los[error_los > tolerance]]
            )
        )
        a, b = rule_at(new)
        r_rows = np.concatenate([r_rows, new])
        sigma_rows = np.concatenate([sigma_rows, a])
        los_rows = np.concatenate([los_rows, b])
        order = np.argsort(r_rows)
        r_rows, sigma_rows, los_rows = r_rows[order], sigma_rows[order], los_rows[order]
        added += len(new)
        rounds += 1
    record = dict(
        end=end,
        rounds=rounds,
        rows_added=added,
        worst_first=first,
        worst_last=worst[-1],
        rule_radii=len(cache),
        seconds=time.perf_counter() - t0,
        var=sigma_read.var,
    )
    log.debug(
        "The projected table checked itself: %s after %d rounds, %d rows added, "
        "worst %.3e.",
        end,
        rounds,
        added,
        worst[-1],
    )
    return r_rows, sigma_rows, los_rows, record
