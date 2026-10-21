"""Surface density and squared line-of-sight velocity dispersion over projected
radius, and the interpolants of their tables.

The stage `stages.projected_dispersion` builds the projected table of a
`LosTable` with `project` and `certify_projection`. The public calls
`sigma_los` and `ProjectedSampler` read that table through
`surface_density_reader` and `dispersion_table_reader`.

For a component truncated at the radius ``R_t``, with density ``rho``,
squared radial dispersion ``sigma_r^2`` and anisotropy profile ``beta_sigma``,
the surface density ``Sigma`` and the squared line-of-sight dispersion
``sigma_los^2`` at the projected radius ``R_2D`` are

- ``Sigma = 2 * integral of rho du``;
- ``Sigma * sigma_los^2 = 2 * integral of sigma_r^2 rho``
  ``(1 - beta_sigma R_2D^2 / R^2) du``.

Both integrals run along the line of sight from ``u = 0`` to
``u = (R_t^2 - R_2D^2)^(1/2)``, at the 3D radius ``R = (R_2D^2 + u^2)^(1/2)``.
The caller sets the tolerance of `certify_projection`. `LosTable` passes
:math:`10^{-8}`, a factor 100 below the bound :math:`10^{-6}`. The formula
is Chiang+(2026d), Eq. (B46).
"""

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
from ..exceptions import AccuracyError, InputError, ResultError
from ..runtime import log
from ..typing import Array, ArrayLike
from ._dispersion import beta_sigma, beta_sigma_tangent

# The floor applied to a value before its logarithm is taken, so that a value
# that underflows to zero still has a finite logarithm.
TINY = 1e-300
# The largest inner slope s for which the disk inside a table's first row has
# a finite mass. With Sigma proportional to R^-s inside the first row R_0, the
# disk holds Sigma_0 R_0^2 / (2 - s), which is finite only for s below 2
# (Section T:6.3).
INNER_SLOPE_MAX = 2.0
# The relative distance within which a radius still counts as a table's end
# radius. Two grids that end at the same radius can differ there by a few units
# in the last place, so a radius computed on one grid can miss the end of the
# other grid by that amount.
END_TOLERANCE = 1e-12


def spline_value(breaks: Any, coefficients: Any, x: Any, xp: Any = np) -> Any:
    """Return a piecewise cubic in SciPy's coefficient layout at `x`.

    `xp` is the array module, NumPy or JAX's NumPy. Each interval between
    consecutive `breaks` uses its own cubic. Beyond the first and the last
    breakpoint, the cubic of the end interval is continued.
    """
    breaks = xp.asarray(breaks)
    c = xp.asarray(coefficients)
    i = xp.clip(xp.searchsorted(breaks, x, side="right") - 1, 0, breaks.size - 2)
    dx = x - breaks[i]
    return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]


def _line_of_sight_nodes(n_nodes: int) -> tuple[Array, Array, Array, Array]:
    """Return the Gauss-Legendre nodes and weights on [0, 1] of the two
    line-of-sight panels of `project`.

    The inner panel gets ``n_nodes // 2`` nodes, and the outer panel gets the
    remaining nodes.
    """
    n_a = n_nodes // 2
    n_b = n_nodes - n_a
    x_a, w_a = np.polynomial.legendre.leggauss(n_a)
    x_b, w_b = np.polynomial.legendre.leggauss(n_b)
    return 0.5 * (x_a + 1.0), 0.5 * w_a, 0.5 * (x_b + 1.0), 0.5 * w_b


def _line_of_sight_rule(
    r_2d_rows: Array, r_t: float, nodes: tuple[Array, Array, Array, Array]
) -> tuple[Array, Array]:
    """Return the 3D radii of the line-of-sight nodes and the nodes' weights
    in ``u``, for a chunk of projected radii.

    Both arrays have the shape (number of projected radii, number of nodes).
    The weights in ``u`` are the weights in ``t`` times ``R``, since
    ``du = R dt``.
    """
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
        User-specified projected radii :math:`R_{\rm 2D}`, in internal units.
    density : callable
        The density :math:`\varrho` at 3D radii, in internal units.
    sigma_r2 : callable
        The squared radial dispersion :math:`\sigma_r^2` at 3D radii, in
        internal units.
    r_t : float
        The 3D radius :math:`R_{\rm t}` where every line of sight ends.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    n_nodes : int
        Gauss-Legendre node count of each line-of-sight integral. `n_nodes`
        must be at least 20.
    chunk : int, optional
        The number of projected radii computed in one array. Default:
        ``256``, which means that the nodes of 256 projected radii are
        evaluated at once.

    Returns
    -------
    surface_density : array, shape (n,)
        :math:`\Sigma(R_{\rm 2D})`, in internal units.
    sigma_los2 : array, shape (n,)
        :math:`\sigma_{\rm los}^2(R_{\rm 2D})`, in internal units.
        :math:`\sigma_{\rm los}^2` is zero where :math:`\Sigma` is zero.

    Raises
    ------
    InputError
        If `n_nodes` is below 20.
    ResultError
        If a value of :math:`\sigma_{\rm los}^2` is not finite.

    Notes
    -----
    .. math::

        \Sigma = 2\int_0^{u_{\max}}\varrho\,{\rm d}u, \qquad
        \Sigma\sigma_{\rm los}^2 = 2\int_0^{u_{\max}}\sigma_{r}^2\varrho
        \Big(1 - \beta_\sigma\frac{R_{\rm 2D}^2}{R^2}\Big)\,{\rm d}u,

    with the distance :math:`u` along the line of sight, the 3D radius
    :math:`R = (R_{\rm 2D}^2 + u^2)^{1/2}`, the anisotropy profile
    :math:`\beta_\sigma(R)` and
    :math:`u_{\max} = (R_{\rm t}^2 - R_{\rm 2D}^2)^{1/2}`. The quadrature
    variable is ``t = asinh(u / R_2D)``, so ``R = R_2D cosh t`` and
    ``du = R dt``. The interval ``[0, t_max]`` is split into two panels at
    ``t_h = max(acosh(R_hi / (2 R_2D)), t_max / 2)``, with
    ``R_hi = R_2D cosh t_max``. The inner panel gets ``n_nodes // 2`` nodes,
    and the outer panel gets the remaining nodes. A density cusp of width
    ``R_2D`` is smooth in ``t``. The outer panel starts at the 3D radius
    ``R_hi / 2`` or farther out, and ends at ``R_hi``.

    References
    ----------
    Chiang+(2026d), Eq. (B46).
    """
    if not n_nodes >= 20:
        raise InputError(
            "The number of line-of-sight nodes n_nodes must be at least 20; the "
            f"value given is {n_nodes}."
        )
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
            f"The squared line-of-sight dispersion is not finite at "
            f"{int(np.sum(~np.isfinite(sigma_los2)))} of the {r_2d.size} projected "
            f"radii."
        )
    log.debug(
        "The projection computed %d projected radii; the surface density does not "
        "decrease at %d steps between neighboring radii, and the squared "
        "line-of-sight dispersion is negative at %d radii.",
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
    anisotropy parameter, on a fixed profile, with the nodes of `project`.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        User-specified projected radii :math:`R_{\rm 2D}`, in internal units.
    density : callable
        The density :math:`\varrho` at 3D radii, the `density` given to
        `project`.
    sigma_r2 : callable
        The squared radial dispersion :math:`\sigma_r^2` at 3D radii, the
        `sigma_r2` given to `project`.
    tangent : callable
        :math:`\partial\sigma_r^2/\partial\theta` at 3D radii, from
        `radial_integral_tangent_reader`.
    r_t : float
        The 3D radius :math:`R_{\rm t}` where every line of sight ends.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    n_nodes : int
        Gauss-Legendre node count of each line-of-sight integral, the
        `n_nodes` given to `project`.
    parameter : {"beta_c", "r_a"}
        The anisotropy parameter :math:`\theta`, one of the entries of
        ``TANGENT_PARAMETERS`` for the family of `anisotropy`.
    chunk : int, optional
        The number of projected radii computed in one array. Default:
        ``256``, which means that the nodes of 256 projected radii are
        evaluated at once.

    Returns
    -------
    tangent : array, shape (n,)
        :math:`\partial\sigma_{\rm los}^2/\partial\theta`, in internal
        units. The derivative is zero where :math:`\Sigma` is zero.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.

    Notes
    -----
    On a fixed profile, :math:`\Sigma` does not depend on :math:`\theta`, so

    .. math::

        \Sigma\frac{\partial\sigma_{\rm los}^2}{\partial\theta} = 2\int_0^{u_{\max}}
        \varrho\Big[\frac{\partial\sigma_{r}^2}{\partial\theta}\Big(1 -
        \beta_\sigma\frac{R_{\rm 2D}^2}{R^2}\Big) - \sigma_{r}^2
        \frac{\partial\beta_\sigma}{\partial\theta}\frac{R_{\rm 2D}^2}{R^2}
        \Big]\,{\rm d}u

    with the integral on the nodes of `project` at the same projected radii.
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
    """Return the edge variable ``x = ln(R / (1 - R / R_t))``.

    For ``R`` much smaller than ``R_t``, ``x`` approaches ``ln R``. As ``R``
    approaches ``R_t``, ``x`` approaches ``ln R_t - ln(1 - R / R_t)``.
    """
    r = np.asarray(r, dtype=np.float64)
    return np.log(r) - np.log1p(-r / r_t)


def edge_variable_inverse(x: ArrayLike, r_t: float) -> Array:
    """Return the radius ``R`` at the edge variable
    ``x = ln(R / (1 - R / R_t))``, the inverse of `edge_variable`."""
    e = np.exp(np.asarray(x, dtype=np.float64))
    return r_t * e / (r_t + e)


def _refuse_below(r_2d: Array, r_min: float, what: str, note: str = "") -> None:
    """Raise InputError if a projected radius in `r_2d` is inside the
    innermost row `r_min` of a table.

    A radius below ``r_min * (1 - END_TOLERANCE)`` counts as inside. `what`
    is the name of the tabulated quantity in the message. `note` is a
    sentence, appended to the message, that states the setting that places
    the innermost row.
    """
    below = r_2d < r_min * (1.0 - END_TOLERANCE)
    if np.any(below):
        raise InputError(
            f"The {what} table is read at R_2D = {float(np.min(r_2d[below])):.4e}, "
            f"inside the innermost projected radius {r_min:.4e} of the table; the "
            f"table is not extrapolated inward." + (f" {note}" if note else "")
        )


def _refuse_beyond(r_2d: Array, r_max: float, what: str, note: str = "") -> None:
    """Raise InputError if a projected radius in `r_2d` is beyond the
    outermost row `r_max` of a table.

    A radius above ``r_max * (1 + END_TOLERANCE)`` counts as beyond. `what`
    is the name of the tabulated quantity in the message. `note` is a
    sentence, appended to the message, that states the setting that places
    the outermost row.
    """
    beyond = r_2d > r_max * (1.0 + END_TOLERANCE)
    if np.any(beyond):
        raise InputError(
            f"The {what} table is read at R_2D = {float(np.max(r_2d[beyond])):.4e}, "
            f"beyond the outermost projected radius {r_max:.4e} of the table; the "
            f"table is not extrapolated outward." + (f" {note}" if note else "")
        )


class SurfaceDensityReading:
    """Interpolant of a table of the surface density over projected radius.

    `surface_density_reader` builds the interpolant, and the Notes of
    `surface_density_reader` describe the interpolation. A call with
    projected radii returns the surface density at those radii.

    Attributes
    ----------
    r_t : float or None
        The radius ``R_t`` where the edge factor ``(R_t^2 - R^2)^(1/2)``
        vanishes. ``r_t`` is None where the table's last row is at or beyond
        ``R_t``, and the interpolant then applies no edge factor.
    var : str
        The interpolation variable: ``"x"`` for the edge variable of
        `edge_variable`, or ``"lnR"`` for the logarithm of the projected
        radius.
    r_min : float
        The projected radius of the table's first row.
    r_max : float
        The projected radius of the table's last row.
    inner_disk : float
        The estimated integral of ``Sigma R dR`` from ``R = 0`` to the first
        row.
    inner_note : str
        A sentence appended to the InputError message for a radius inside the
        first row. The sentence states the setting that places the first row.
        Default: ``""``, which means that the message has no such sentence.
    outer_note : str
        A sentence appended to the InputError message for a radius beyond the
        last row. The sentence states the setting that places the last row.
        Default: ``""``, which means that the message has no such sentence.
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
        # The edge radius R_t, set only where the table ends inside R_t.
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
        self._inner_slope = float(
            -(log_sigma[1] - log_sigma[0]) / (log_r_rows[1] - log_r_rows[0])
        )
        self._sigma_0 = float(np.exp(log_sigma[0]))
        self.r_min = r_min
        self.r_max = r_max
        # The callers set the two notes, the sentences that state the settings
        # that place the first and the last row.
        self.inner_note = ""
        self.outer_note = ""

    @property
    def inner_disk(self) -> float:
        """The estimated integral of ``Sigma R dR`` from ``R = 0`` to the first
        row.

        The estimate is ``Sigma_0 R_0^2 / (2 - s)``, with ``R_0`` the first
        row's radius, ``Sigma_0`` the first row's surface density and ``s``
        the slope ``-d ln Sigma / d ln R`` between the first two rows.

        Raises
        ------
        AccuracyError
            If ``s`` is not below ``INNER_SLOPE_MAX``.
        """
        slope = self._inner_slope
        if not slope < INNER_SLOPE_MAX:
            raise AccuracyError(
                f"The slope -d ln Sigma / d ln R = {slope:.6g} of the surface "
                f"density between the table's first two projected radii, at R_2D = "
                f"{self.r_min:.4e}, is not below {INNER_SLOPE_MAX:g}. At a slope of "
                f"{INNER_SLOPE_MAX:g} or more, the mass of the disk inside the first "
                f"projected radius diverges (Section T:6.3), so the mass of the disk "
                f"is not estimated."
            )
        return self._sigma_0 * self.r_min**2 / (2.0 - slope)

    def _edge_factor(self, r: Array) -> Array:
        if self._has_edge:
            return np.sqrt(np.maximum(self._edge * self._edge - r * r, 0.0))
        return np.ones_like(r)

    def _variable(self, r: Array) -> Array:
        if self._use_x:
            return edge_variable(r, self._edge)
        return np.log(np.maximum(r, TINY))

    def traced(self, r: Any, xp: Any = np) -> Any:
        """Return the interpolant at `r` under the array module `xp`, NumPy
        or JAX's NumPy, with `r` traceable by JAX.

        The traced interpolant returns zero beyond the last row. Inside the
        first row, the traced interpolant continues the cubic of the first
        interval. The traced interpolant raises no error, so a caller must
        raise InputError for radii outside the rows before tracing, as
        `__call__` does.
        """
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
        _refuse_below(r_any, self.r_min, "surface density", self.inner_note)
        _refuse_beyond(r_any, self.r_max, "surface density", self.outer_note)
        log_s = self._spline(self._variable(r_any))
        return np.exp(log_s) * self._edge_factor(r_any)


def surface_density_reader(
    r_2d: ArrayLike, surface_density: ArrayLike, r_t: float, spatially_finite: bool
) -> SurfaceDensityReading:
    r"""Return the interpolant of a table of the surface density over
    projected radius.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radii :math:`R_{\rm 2D}` of the table's rows, increasing,
        in internal units.
    surface_density : array-like, shape (n,)
        :math:`\Sigma` at the rows, in internal units.
    r_t : float
        The 3D radius :math:`R_{\rm t}` where every line of sight ends.
    spatially_finite : bool
        Whether the model's density ends at a finite edge radius.

    Returns
    -------
    read : SurfaceDensityReading
        The interpolant. A call of `read` with projected radii returns
        :math:`\Sigma` at those radii. A call at a radius outside the rows
        raises InputError, whose message ends with ``read.inner_note`` or
        ``read.outer_note`` where the caller sets them.

    Notes
    -----
    Where the last row is inside :math:`R_{\rm t}`, the interpolant
    tabulates ``h = Sigma / (R_t^2 - R^2)^(1/2)``, which stays smooth up to
    the last row, and multiplies by the edge factor
    ``(R_t^2 - R^2)^(1/2)`` at evaluation. Where the last row is at or
    beyond :math:`R_{\rm t}`, ``h`` is ``Sigma`` itself. The interpolant of
    ``ln h`` is a not-a-knot cubic spline. On a spatially finite model with
    the last row inside :math:`R_{\rm t}`, the spline variable is
    ``x = ln(R / (1 - R / R_t))``, since ``h`` goes as a power of
    ``R_t - R`` there. On every other model, the spline variable is
    ``ln R``.

    ``read.inner_disk`` estimates the integral of ``Sigma R dR`` inside the
    first row from the slope ``s = -d ln Sigma / d ln R`` between the first
    two rows. With :math:`\Sigma \propto R^{-s}` inside the first row, the
    estimate is :math:`\Sigma_0 R_0^2 / (2 - s)`, with :math:`R_0` the
    first row's radius and :math:`\Sigma_0` the first row's surface
    density. ``read.inner_disk`` raises AccuracyError if ``s`` is not below
    ``INNER_SLOPE_MAX``, since the estimate diverges at ``s = 2``.
    """
    return SurfaceDensityReading(r_2d, surface_density, r_t, spatially_finite)


class DispersionReading:
    """Interpolant of a table of the squared line-of-sight dispersion over
    projected radius.

    `dispersion_table_reader` builds the interpolant. A call with projected
    radii inside the rows returns the squared dispersion at those radii, and
    raises AccuracyError where the interpolated value is negative. The
    method `raw` returns the interpolated value with its sign, for a table
    whose rows may be negative, such as a table of derivatives in an
    anisotropy parameter. A radius outside the rows raises InputError. The
    InputError message ends with ``inner_note`` or ``outer_note``, the
    sentence that states the setting that places the end row, where the
    caller sets the note.
    """

    def __init__(self, r_2d: ArrayLike, sigma_los2: ArrayLike) -> None:
        r_rows = np.asarray(r_2d, dtype=np.float64)
        log_r_rows = np.log(np.maximum(r_rows, TINY))
        self._spline = CubicSpline(log_r_rows, sigma_los2, extrapolate=False)
        self._r_min = float(r_rows[0])
        self._r_max = float(r_rows[-1])
        # The callers set the two notes, the sentences that state the settings
        # that place the first and the last row.
        self.inner_note = ""
        self.outer_note = ""

    def traced(self, r: Any, xp: Any = np) -> Any:
        """Return the interpolant at `r` under the array module `xp`, NumPy
        or JAX's NumPy, with `r` traceable by JAX.

        The traced interpolant returns NaN where the interpolated value is
        negative, at the radii where `__call__` raises AccuracyError. The
        traced interpolant returns zero beyond the last row. Inside the first
        row, the traced interpolant continues the cubic of the first
        interval. The traced interpolant raises no error, so a caller must
        raise InputError for radii outside the rows before tracing, as
        `__call__` does.
        """
        log_r = xp.log(xp.maximum(r, TINY))
        value = spline_value(self._spline.x, self._spline.c, log_r, xp)
        value = xp.where(log_r > float(self._spline.x[-1]), 0.0, value)
        return xp.where(value < 0.0, xp.nan, value)

    def raw(self, r_any: ArrayLike) -> Array:
        """Return the interpolated value at projected radii `r_any`, with its
        sign.

        Raises
        ------
        InputError
            If a radius is outside the rows by more than ``END_TOLERANCE``.
        """
        r_any = np.atleast_1d(np.asarray(r_any, dtype=np.float64))
        what = "squared line-of-sight dispersion"
        _refuse_below(r_any, self._r_min, what, self.inner_note)
        _refuse_beyond(r_any, self._r_max, what, self.outer_note)
        # A radius outside an end row by at most END_TOLERANCE is clipped to
        # that end row.
        x = self._spline.x
        return self._spline(np.clip(np.log(np.maximum(r_any, TINY)), x[0], x[-1]))

    def __call__(self, r_any: ArrayLike) -> Array:
        value = self.raw(r_any)
        negative = value < 0.0
        if np.any(negative):
            r_any = np.atleast_1d(np.asarray(r_any, dtype=np.float64))
            k = int(np.argmin(value))
            raise AccuracyError(
                f"The interpolated squared line-of-sight dispersion is "
                f"{float(value[k]):.3e}, below 0, at R_2D = {float(r_any[k]):.4e}, "
                f"between the table's tabulated projected radii, so a negative "
                f"variance is not returned. The setting observables.n_radii places "
                f"the tabulated projected radii, and the setting "
                f"observables.n_projection_nodes gives the number of nodes of the "
                f"projection integral."
            )
        return value


def dispersion_table_reader(
    r_2d: ArrayLike, sigma_los2: ArrayLike
) -> DispersionReading:
    r"""Return the interpolant of a table of :math:`\sigma_{\rm los}^2` over
    projected radius.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radii :math:`R_{\rm 2D}` of the table's rows, increasing,
        in internal units.
    sigma_los2 : array-like, shape (n,)
        :math:`\sigma_{\rm los}^2` at the rows, in internal units.

    Returns
    -------
    read : DispersionReading
        The interpolant. A call of `read` with projected radii returns
        :math:`\sigma_{\rm los}^2` at those radii. A call raises
        AccuracyError where the interpolated value is negative, and raises
        InputError at a radius outside the rows. ``read.raw`` returns the
        interpolated values with their sign.

    Notes
    -----
    The interpolant is a not-a-knot cubic spline in ``ln R``.
    `certify_projection` checks the table's rows with the not-a-knot cubic
    spline in ``ln R`` that `dispersion_table_reader` returns.
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
    """Return the rows of a projected table, with rows added until the
    table's interpolants meet `tolerance` between rows.

    Parameters
    ----------
    r_2d : array-like, shape (n,)
        Projected radii of the starting rows, increasing, in internal units.
    surface_density : array-like, shape (n,)
        The surface density at the starting rows, in internal units.
    sigma_los2 : array-like, shape (n,)
        The squared line-of-sight dispersion at the starting rows, in
        internal units.
    r_t : float
        The 3D radius ``R_t`` where every line of sight ends.
    spatially_finite : bool
        Whether the model's density ends at a finite edge radius.
    rule : callable
        A function of projected radii that returns the surface density and
        the squared line-of-sight dispersion by the quadrature that computed
        the starting rows, such as `project`.
    tolerance : float
        The largest relative difference allowed at an interval's midpoint
        between the interpolated value and the value of `rule`.

    Returns
    -------
    r_2d : array
        Projected radii of the rows, the starting rows and the added rows,
        increasing.
    surface_density : array
        The surface density at the returned rows.
    sigma_los2 : array
        The squared line-of-sight dispersion at the returned rows.
    record : dict
        The check's record, with the keys ``end`` (``"tolerance"`` or
        ``"stalled"``), ``rounds`` (the number of rounds that added rows),
        ``rows_added``, ``worst_first`` (the largest relative differences of
        the surface density and of the squared dispersion before any row is
        added), ``worst_last`` (the largest relative difference of the last
        round), ``rule_radii`` (the number of radii where `rule` was
        evaluated), ``seconds`` (the wall-clock time of the check) and
        ``var`` (the interpolation variable of the surface density, ``"x"``
        or ``"lnR"``).

    Notes
    -----
    Each round compares the interpolants of `surface_density_reader` and
    `dispersion_table_reader` with `rule` at the midpoint of every interval
    between rows. Each midpoint is the midpoint in the interpolation
    variable of the interpolant compared there. At a midpoint where an
    interpolant's relative difference exceeds `tolerance`, the values of
    `rule` at that midpoint become a new row. The rounds stop when one of
    two conditions holds: the largest relative difference of a round is at
    or below `tolerance` (``end = "tolerance"``); or each of two successive
    rounds leaves a largest relative difference above half the largest
    relative difference of the round before (``end = "stalled"``). The
    stalling test spans two rounds because the cubic spline is global: a
    row added near the edge can leave the largest relative difference
    unchanged for one round before the largest relative difference falls.
    `rule` is evaluated once at each radius.
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
        "The projected table's self-check ended with the status '%s' after %d "
        "rounds; the self-check added %d projected radii, and the worst relative "
        "difference is %.3e.",
        end,
        rounds,
        added,
        worst[-1],
    )
    return r_rows, sigma_rows, los_rows, record
