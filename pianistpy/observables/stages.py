"""Stages that compute the velocity observables of a model of one component.

The public calls of `pianistpy.observables` use these stages: `LosTable`,
`sigma_r`, `sigma_t`, `sigma_los`, `sigma_los_aperture`, `losvd`,
`aperture_losvd` and `pdf_v`. The stage `velocity_distribution` is public itself.

The second moments come from the model's density and potential, in three
stages::

    dispersion_profile -> projected_dispersion -> aperture_dispersion

The line-of-sight velocity distributions come from the model's distribution
function (DF), in four stages::

    kept_ranges -> losvd_row -> losvd_table -> aperture_losvd

The second-moment stages and the distribution stages share only the model.
The distributions are therefore checked against the second-moment stages, in
three checks: a row's integral over velocity against the surface density; a
row's second moment against the squared line-of-sight velocity dispersion;
and an aperture distribution's second moment against the squared
line-of-sight velocity dispersion averaged over the aperture. Each check's
relative error is reported in a validation report, whose tier is split at
:math:`10^{-6}`, :math:`10^{-4}` and :math:`10^{-2}`.

The module also holds the Gauss-Hermite moments of the distributions and the
local velocity distribution at a 3D radius (`velocity_distribution`). The technical
notes describe every stage in Section T:6.
"""

__all__ = [
    "END_TOLERANCE",
    "TINY",
    "ApertureLosvd",
    "DispersionProfile",
    "GaussHermiteTable",
    "KeptRanges",
    "LosvdRow",
    "LosvdTable",
    "ProjectedDispersion",
    "VelocityDistribution",
    "aperture_dispersion",
    "aperture_losvd",
    "aperture_nodes",
    "dispersion_profile",
    "gauss_hermite_moments",
    "gauss_hermite_table",
    "kept_ranges",
    "losvd_row",
    "losvd_table",
    "projected_dispersion",
    "velocity_distribution",
    "surface_density_reader",
    "validate_aperture",
    "validate_table",
]

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from ..df import Model
from ..df.truncation import K_FLOOR
from ..exceptions import AccuracyError, InputError
from ..runtime import get_cores, timed
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..validation import TIER_BOUNDS, Check, ValidationReport
from ._aperture import aperture_mean_square, aperture_nodes
from ._aperture_losvd import aperture_losvd_panels
from ._dispersion import (
    TANGENT_PARAMETERS,
    beta_sigma,
    radial_dispersion_tangent,
    radial_integral_reader,
    radial_integral_tangent_reader,
)
from ._ellipse_model import ellipse_route
from ._ellipse_rules import ROW_TOL_PW
from ._kept import extrema_in_pool, vk_kept_ranges, vrad_kept_ranges
from ._losvd_stages import (
    BREAK_NAMES,
    base_rows,
    cut_rows,
    ellipse_rules,
    row_start,
    row_stop,
    rows_of,
    served_radius,
    start_pool,
    table_stencil,
)
from ._losvd_table import losvd_rows, new_pool
from ._projection import (
    END_TOLERANCE,
    TINY,
    _refuse_below,
    _refuse_beyond,
    dispersion_table_reader,
    project_tangent,
    spline_value,
    surface_density_reader,
)
from ._tables import projected_table, radial_table
from ._velocity import (
    VelocityModel,
    angle_integral,
    graded_rule,
    velocity_breaks,
    velocity_model,
)
from .aperture import Aperture

# The largest relative difference allowed between the projected table read
# between two rows and the projection computed there. The value is a hundredth
# of the bound 1e-6: the interpolation adds at most 1e-8 at the midpoints.
CERTIFICATE_TOLERANCE = 0.01 * TIER_BOUNDS[0]
# The largest share of an aperture's weight allowed inside the projected
# table's first row. The disk inside the first row is estimated, not computed,
# so the share is held to the bound 1e-6.
INNER_DISK_TOLERANCE = TIER_BOUNDS[0]


@dataclass(frozen=True)
class DispersionProfile:
    r"""Intrinsic velocity dispersions of a model on rows of 3D radius.

    `dispersion_profile` builds the object. The method `read` returns the
    dispersions at any radius between the rows.

    Parameters
    ----------
    r : array, shape (n,)
        3D radius :math:`R` of the rows, increasing, in internal units.
    sigma_r : array, shape (n_components, n)
        Radial velocity dispersion :math:`\sigma_{\rm r}` at the rows, in
        internal units.
    sigma_t : array, shape (n_components, n)
        Tangential velocity dispersion :math:`\sigma_{\rm t}` at the rows, in
        internal units. :math:`\sigma_{\rm t}` combines both tangential
        components: :math:`\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2`.
        The anisotropy is therefore
        :math:`\beta_\sigma = 1 - \sigma_{\rm t}^2 / (2 \sigma_{\rm r}^2)`.
    beta_sigma : array, shape (n_components, n)
        Velocity anisotropy :math:`\beta_\sigma(R)` at the rows.
    edge_limit : tuple of float
        One value per component. On a spatially finite model, the value is
        the limit of :math:`\sigma_{\rm r}^2 / P` at the model's edge, with
        :math:`P` the relative potential. `read` uses the limit between the
        outermost row and the edge. The value is NaN on a model that is not
        spatially finite, and NaN where the DF states no power law at the
        edge.
    certificate : dict
        The record of the row placement by `radial_table`: ``tolerance``,
        ``n_jeans_nodes``, ``rule_end``, ``rule_change``, ``end``,
        ``rounds``, ``rows``, ``worst_first``, ``worst_last`` and
        ``uncertified`` (the intervals of radius whose midpoint error stayed
        above ``tolerance``).
    model : Model
        The model of the profile. `read` evaluates the model's potential and
        the anisotropy of the model's components.
    tangents : dict of str to array, shape (n_components, n)
        The derivative of :math:`\sigma_{\rm r}^2` at the rows in each
        anisotropy parameter that `dispersion_profile` was given in its
        argument ``tangents``, at fixed density profile, in internal units.
        The dict is empty when `dispersion_profile` was given no parameter.

    Notes
    -----
    The rows hold the Jeans integral
    :math:`I = \varrho_{\rm aux}\sigma_{\rm r}^2`, with
    :math:`\varrho_{\rm aux}` the auxiliary density. The derivative of
    :math:`I` in :math:`P` is :math:`\varrho_{\rm aux}`, exactly. `read`
    interpolates :math:`I` by the cubic Hermite polynomial in :math:`P`
    through the rows' values of :math:`I` and of :math:`\varrho_{\rm aux}`
    (`radial_integral_reader`). Every check of a velocity distribution
    against :math:`\sigma_{\rm r}` reads :math:`\sigma_{\rm r}` through the
    cubic Hermite interpolation. The interpolation uses the rows with a
    positive :math:`\sigma_{\rm r}^2` and a positive
    :math:`\varrho_{\rm aux}`, except the last row of a spatially finite
    model. `read` raises InputError at a radius inside the innermost used
    row. On a spatially finite model with a finite `edge_limit`, `read`
    interpolates :math:`h = \sigma_{\rm r}^2 / P` linearly in :math:`P`,
    from the outermost used row to the value `edge_limit` at the edge, where
    :math:`P = 0`. `read` raises InputError at a radius beyond the edge, or,
    on any other model, beyond the outermost used row.
    """

    r: Array
    sigma_r: Array
    sigma_t: Array
    beta_sigma: Array
    edge_limit: tuple[float, ...]
    certificate: dict[str, Any]
    model: Model
    tangents: dict[str, Array] = field(default_factory=dict)

    def _rho_aux(self, component: int) -> Callable[[Array], Array]:
        r"""Return the auxiliary density of one component as a function of radius.

        The auxiliary density is :math:`\varrho_{\rm aux} = \varrho / g`, with
        :math:`\varrho` the component's density and :math:`g` the factor of
        the component's anisotropy.
        """
        profile = self.model.profile
        anisotropy = self.model.components[component].anisotropy

        def rho_aux(r: Array) -> Array:
            return profile.density(r)[component] / anisotropy.factor(r)

        return rho_aux

    def _read_tangent(
        self, parameter: str, component: int = 0
    ) -> Callable[[ArrayLike], Array]:
        r"""Return the interpolated derivative of :math:`\sigma_{\rm r}^2` in
        one anisotropy parameter, for one component.

        The returned function interpolates the rows of `tangents` for the
        parameter whose name is `parameter`."""
        return radial_integral_tangent_reader(
            self.r,
            np.asarray(self.sigma_r)[component] ** 2,
            np.asarray(self.tangents[parameter])[component],
            self.model.profile.potential,
            self._rho_aux(component),
            self.model.components[component].anisotropy,
            parameter,
            self.model.spatially_finite,
            self.edge_limit[component],
        )

    def _read_sigma_r2(self, component: int = 0) -> Callable[[ArrayLike], Array]:
        r"""Return the interpolation of :math:`\sigma_{\rm r}^2` between the
        rows, for one component."""
        return radial_integral_reader(
            self.r,
            np.asarray(self.sigma_r)[component] ** 2,
            self.model.profile.potential,
            self._rho_aux(component),
            self.model.spatially_finite,
            self.edge_limit[component],
        )

    def read(self, r: ArrayLike) -> tuple[Array, Array, Array]:
        """Return `sigma_r`, `sigma_t` and `beta_sigma` at 3D radii `r`.

        Each of the three arrays has shape (n_components, len(r)). The
        velocity dispersions are in internal units. `beta_sigma` is evaluated
        from the anisotropy of each component, not interpolated.
        """
        r = np.asarray(r, dtype=np.float64)
        n_components = np.shape(self.sigma_r)[0]
        sigma_r2 = np.stack([self._read_sigma_r2(i)(r) for i in range(n_components)])
        beta = np.stack(
            [
                beta_sigma(r, self.model.components[i].anisotropy)
                for i in range(n_components)
            ]
        )
        return np.sqrt(sigma_r2), np.sqrt(2.0 * (1.0 - beta) * sigma_r2), beta


@dataclass(frozen=True)
class ProjectedDispersion:
    """Surface density and line-of-sight dispersion on rows of projected radius.

    `projected_dispersion` builds the object. One quadrature along the line
    of sight, with one density, gives both quantities at each row. Rows are
    added until the interpolation between rows agrees with the quadrature at
    every interval's midpoint to `CERTIFICATE_TOLERANCE`.

    Parameters
    ----------
    r_2d : array, shape (n,)
        Projected radius :math:`R_{\\rm 2D}` of the rows, increasing, in
        internal units.
    surface_density : array, shape (n_components, n)
        Surface density :math:`\\Sigma` at the rows, in internal units.
    sigma_los : array, shape (n_components, n)
        Line-of-sight velocity dispersion :math:`\\sigma_{\\rm los}` at the
        rows, in internal units.
    certificate : dict
        The record of the row placement by `certify_projection`: ``end``
        (``"tolerance"`` or ``"stalled"``), ``rounds``, ``rows_added``,
        ``worst_last``, ``rule_radii``, ``seconds``, ``var``,
        ``worst_first_surface_density`` and ``worst_first_sigma_los2``.
    model : Model
        The model of the table. The interpolation between rows uses the
        model's outermost tabulated radius :math:`R_{\\rm t}`, where every
        line of sight ends, and whether the model is spatially finite.
    tangents : dict of str to array, shape (n_components, n)
        The derivative of :math:`\\sigma_{\\rm los}^2` at the rows in each
        anisotropy parameter that `projected_dispersion` was given in its
        argument ``tangents``, at fixed density profile, in internal units.
        The dict is empty when `projected_dispersion` was given no
        parameter. The surface density has no derivative in an anisotropy
        parameter.

    Notes
    -----
    The surface density is interpolated through
    ``h = Sigma / (R_t**2 - R_2D**2)**0.5``. The interpolant of ``ln h`` is a
    not-a-knot cubic spline in ``x = ln(R_2D / (1 - R_2D / R_t))`` on a
    spatially finite model, and in ``ln R_2D`` on any other model.
    :math:`\\sigma_{\\rm los}^2` is interpolated by a not-a-knot cubic spline
    in ``ln R_2D``. The row placement checks these two interpolations. Both
    interpolations raise InputError at a projected radius inside the first
    row or beyond the last row. The message of the error states what places
    the first or the last row (`notes`).
    """

    r_2d: Array
    surface_density: Array
    sigma_los: Array
    certificate: dict[str, float | int | str]
    model: Model
    tangents: dict[str, Array] = field(default_factory=dict)

    def notes(self) -> tuple[str, str]:
        """Return the two sentences that state what places the first row and
        the last row.

        The interpolations of `read` append the first sentence to an
        InputError at a radius inside the first row, and the second sentence
        to an InputError at a radius beyond the last row. The first row is
        set by the setting ``truncation.log10_r_min``, or by the method's
        floor `K_FLOOR` when the first row differs from that setting's
        radius. The last row lies inside the edge of a spatially finite
        model, or inside the profile's outermost row on any other model.
        """
        r_t = float(self.model.profile.r_table_max)
        r_0 = float(self.r_2d[0])
        log10_r_min = float(self.model.settings.truncation["log10_r_min"])
        inner = (
            f"The table's innermost projected radius is the profile's innermost "
            f"radius {r_0:.4e}, which the setting truncation.log10_r_min sets."
            if r_0 == 10.0**log10_r_min
            else f"The table's innermost projected radius is the profile's innermost "
            f"radius {r_0:.4e}, at the method's floor, a depth {K_FLOOR:g} below the "
            f"center that no setting moves."
        )
        outer = (
            f"The table's outermost projected radius lies just inside the model's "
            f"edge {r_t:.4e}, which the model's truncation sets."
            if self.model.spatially_finite
            else f"The table's outermost projected radius lies just inside the "
            f"profile's outermost radius {r_t:.4e}, where the model's potential "
            f"reaches the method's floor {K_FLOOR:g}; no setting moves the "
            f"method's floor."
        )
        return inner, outer

    def read(self, r_2d: ArrayLike) -> tuple[Array, Array]:
        """Return the surface density and `sigma_los` at projected radii.

        Each of the two arrays has shape ``(n_components, len(r_2d))``, in
        internal units. A projected radius inside the first row or beyond the
        last row raises InputError. A negative interpolated value of the
        squared dispersion raises AccuracyError.
        """
        r_t = float(self.model.profile.r_table_max)
        finite = self.model.spatially_finite
        inner, outer = self.notes()
        surface, sigma_los = [], []
        for i in range(np.shape(self.surface_density)[0]):
            read_surface = surface_density_reader(
                self.r_2d, self.surface_density[i], r_t, finite
            )
            read_los2 = dispersion_table_reader(
                self.r_2d, np.asarray(self.sigma_los[i]) ** 2
            )
            for reader in (read_surface, read_los2):
                reader.inner_note, reader.outer_note = inner, outer
            surface.append(read_surface(r_2d))
            sigma_los.append(np.sqrt(read_los2(r_2d)))
        return np.stack(surface), np.stack(sigma_los)

    def read_traced(self, r_2d: Any, xp: Any = np) -> tuple[Any, Any]:
        """Return the surface density and the squared `sigma_los` at projected
        radii, with `r_2d` traceable by JAX.

        The method evaluates the cubic splines of `read` in the array module
        `xp` (``numpy`` or ``jax.numpy``). Between the first and the last row
        the values equal those of `read` to rounding, with `sigma_los`
        squared. Beyond the last row both values are zero. Inside the first
        row the cubic of the first interval is extended, and the values are
        not checked. The caller must therefore raise InputError for a radius
        inside the first row before it traces this method, with the relative
        margin `END_TOLERANCE` that `read` applies. Where the squared
        dispersion is negative, the returned squared dispersion is NaN.
        """
        r_t = float(self.model.profile.r_table_max)
        finite = self.model.spatially_finite
        surface, sigma_los2 = [], []
        for i in range(np.shape(self.surface_density)[0]):
            surface.append(
                surface_density_reader(
                    self.r_2d, self.surface_density[i], r_t, finite
                ).traced(r_2d, xp)
            )
            sigma_los2.append(
                dispersion_table_reader(
                    self.r_2d, np.asarray(self.sigma_los[i]) ** 2
                ).traced(r_2d, xp)
            )
        return xp.stack(surface), xp.stack(sigma_los2)


@dataclass(frozen=True)
class KeptRanges:
    """Ranges of projected radius in which two velocity breaks are kept.

    A break is a velocity at which the line-of-sight velocity distribution of
    a row is not smooth, and a velocity panel of the row ends there.
    `kept_ranges` decides the ranges once for a model, on the base rows of
    the model's table. Every computation of a row or of a table places its
    velocity panels with the same ranges. A row computed alone therefore
    agrees with the row at the same projected radius in the table.

    Parameters
    ----------
    v_rad : array, shape (n_ranges, 2)
        The ranges of projected radius, as (start, end) pairs in internal
        units, in which the break at the radial turning velocity V_rad is
        kept.
    v_k : array, shape (n_ranges, 2)
        The ranges of projected radius, as (start, end) pairs in internal
        units, in which the break at the kink velocity V_k is kept.
    """

    v_rad: Array
    v_k: Array


@dataclass(frozen=True)
class LosvdRow:
    """The line-of-sight velocity distribution at one projected radius (a row).

    Parameters
    ----------
    r_2d : float
        Projected radius of the row, in internal units.
    layout : str
        The layout code of the row: a decimal integer that identifies the order
        of the row's interior velocity breaks. ``"0"`` means no interior
        break. ``"1"`` to ``"15"`` are the orders of V_k, V_rad and V_edge.
        ``"16"`` to ``"64"`` are the orders that include V_c. A code that
        contains V_k is raised by 100 where a kept V_k lies inside the first
        velocity panel graded toward zero velocity. The interpolation between
        two rows requires the two rows to have the same layout code.
    v_breaks : dict of str to float
        The five break velocities that the row computed, by name:
        ``"v_k"``, ``"v_rad"``, ``"v_edge"``, ``"v_c"`` and ``"v_end"``, in
        internal units. The layout code says which of the five are interior
        breaks of the row. ``"v_end"`` is the largest velocity of the row.
    v_escape : float
        Escape velocity of the line of sight at `r_2d`, in internal units.
    v_nodes : array, shape (n,)
        Velocity nodes of the row's panels, at non-negative velocity only,
        in internal units. The distribution is even in velocity.
    v_weights : array, shape (n,)
        Quadrature weights of `v_nodes`.
    panel_edges : array
        Edges of the row's velocity panels: 0, the interior breaks, and
        V_end. Between nodes, the row is interpolated by the polynomial of
        each panel through that panel's own nodes.
    losvd : array, shape (n_components, n)
        The distribution at `v_nodes`, normalized so that its integral over
        both signs of velocity is 1: twice its sum with `v_weights` is 1.
        A row of a `LosvdTable` whose integral is not positive is stored
        without normalization. An aperture weights a row by the surface density of
        `ProjectedDispersion`, never by the row's integral before
        normalization. The ratio of the two is the row's norm check in
        `record`.
    record : dict
        The row's checks and the record of its quadrature along the line of
        sight. ``norm_against_surface_density`` is the row's integral before
        normalization over the surface density, less 1.
        ``m2_against_projected_dispersion`` is the row's second moment over
        :math:`\\sigma_{\\rm los}^2` of `ProjectedDispersion`, less 1.
        ``beyond_escape`` is the share of the row beyond `v_escape`.
        ``cusp`` says whether the row used the quadrature rule of a cusp.
        ``t_panels``, ``t_change`` and ``t_converged`` are the panels of the
        quadrature along the line of sight, the quadrature's last change and
        whether the quadrature converged. A row of a `LosvdTable` also holds
        ``seconds``, the wall-clock time of the row.
    """

    r_2d: float
    layout: str
    v_breaks: dict[str, float]
    v_escape: float
    v_nodes: Array
    v_weights: Array
    panel_edges: Array
    losvd: Array
    record: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class LosvdTable:
    """Rows of the line-of-sight velocity distribution over projected radius.

    `losvd_table` builds the object for a model of one component. Components
    in one potential do not share their velocity breaks, so each component
    has its own table. An aperture sums the components, each weighted by its
    surface density.

    Parameters
    ----------
    rows : tuple of LosvdRow
        The rows, in increasing projected radius.
    r_base : array
        The base rows (`base_rows`) that lie between the table's first and
        last row, in internal units. The base rows are placed before any row
        is added.
    r_breaks : array
        Projected radii at which the layout code of the rows changes, in
        internal units. The interpolation between rows never crosses one of
        these radii.
    stretch_ends : array of bool, shape (n_stretches, 2)
        One pair per stretch between two neighboring entries of `r_breaks`.
        Each flag says whether that end of the stretch is a true end: a
        radius in `r_breaks`, the edge of a spatially finite model, or the
        last radius of the table. The flags fix the variable in which the
        rows of the stretch are interpolated.
    kept : KeptRanges
        The kept ranges with which the rows placed their velocity panels.
    order : int
        The number of rows in the interpolation stencil with which the
        table was checked between rows. An interpolation with another number
        of rows reads an unchecked table.
    r_uncertified : array, shape (n_gaps, 2)
        Intervals of projected radius, in internal units, in which the
        interpolation between rows is not checked. A request inside one of
        these intervals raises an error that gives the interval.
    why_uncertified : tuple of str
        The reason for each interval of `r_uncertified`, quoted by that
        error.
    certificate : dict
        The record of the row placement: how the placement ended, the rows
        added, the rounds and the tolerances. The table's integer, float and
        string entries of the placement record are kept, and
        ``layer_floor`` is added (`validate_table`).
    identity : dict
        The model of the table: ``model`` (the model's identity digest),
        ``spatially_finite``, ``r_edge`` (the model's edge),
        ``r_end`` (the last row of a table cut inside the edge, else
        infinity) and ``r_table_max`` (the profile's outermost radius).
    extrema : dict
        The extrema along the line of sight (`EllipseRoute.f_extrema`) that
        the apertures read from this table have computed, keyed by the
        model's identity and the rules, then by projected radius. A second
        aperture on the same table reads the stored extrema instead of
        computing them again. Each stored extremum equals the value that its
        radius gives when computed alone. The field is not part of the
        table's value.
    pools : dict
        The pools of processes that the apertures read from this table keep
        open between calls (`table_pool`), keyed by the model's identity,
        the rules and the number of cores. A pool starts at the first
        aperture that needs a pool. A pool closes at `close_pools` or when the
        table is garbage-collected. The field is not part of the table's
        value.
    readers : dict
        The route without the DF and the stencil between rows with which
        `losvd` and its transpose read this table
        (`observables.core._reader_of`), keyed by the model's identity and
        the rules. Each entry is made at the first read, and the route's
        extrema are those of `extrema`. A table made from this table by
        `dataclasses.replace` starts with an empty dict. The field is not
        part of the table's value.
    """

    rows: tuple[LosvdRow, ...]
    r_base: Array
    r_breaks: Array
    stretch_ends: Array
    kept: KeptRanges
    order: int
    r_uncertified: Array
    why_uncertified: tuple[str, ...]
    certificate: dict[str, float | int | str]
    identity: dict[str, float | int | str | bool]
    extrema: dict[Any, dict[float, Array]] = field(
        default_factory=dict, repr=False, compare=False
    )
    pools: dict[Any, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    readers: dict[Any, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def close_pools(self) -> None:
        """Terminate and join every pool of processes in `pools`."""
        while self.pools:
            _, pool = self.pools.popitem()
            pool.terminate()
            pool.join()


def table_pool(table: LosvdTable, route: Any, key: Any) -> Any:
    """Return the pool of processes stored on `table` under `key`.

    The first call for a key starts a pool of `get_cores` processes and
    stores the pool in `LosvdTable.pools`. Each process receives a copy of
    `route` with an empty extrema cache. The route's own extrema cache stays
    with the caller. A process computes the extrema of each radius that the
    process receives, and each value equals the value that the radius gives when
    computed alone. When the table is garbage-collected before
    `LosvdTable.close_pools` is called, a finalizer terminates the pool.
    """
    if key not in table.pools:
        cache, route._fext = route._fext, {}
        try:
            pool = new_pool(get_cores(), route)
        finally:
            route._fext = cache
        table.pools[key] = pool
        weakref.finalize(table, pool.terminate)
    return table.pools[key]


@dataclass(frozen=True)
class ApertureLosvd:
    """The line-of-sight velocity distribution inside an aperture.

    `aperture_losvd` builds the object.

    Parameters
    ----------
    aperture : Aperture
        The aperture, in internal units.
    v_nodes : array, shape (n,)
        Velocity nodes of the distribution's panels, at non-negative velocity
        only, in internal units.
    v_weights : array, shape (n,)
        Quadrature weights of `v_nodes`. The moments of the distribution are
        sums with these weights.
    v_breaks : array
        Edges of the distribution's velocity panels, in internal units.
    losvd : array, shape (n_components, n)
        The distribution at `v_nodes`: the rows of the table averaged over
        the aperture with the surface density as weight. The distribution's
        integral over both signs of velocity is ``record["norm_file"]``,
        which `validate_aperture` checks against 1.
    mass : tuple of float
        Projected mass of each component inside the aperture, in internal
        units.
    record : dict
        The integer, float and string entries of the record of
        `aperture_losvd_panels`, and ``table_level``. ``norm_file`` and
        ``m2_file`` are the integral and the second moment over both signs of
        velocity with the distribution's own weights. ``norm_rows`` and
        ``m2_rows`` are the same moments by the rows' own quadrature rule.
        ``stencil_est`` estimates the error of the interpolation between rows
        over the aperture. ``refine_end``, ``refine_rounds``, ``gap_first``
        and ``gap_last`` record how the refinement of the velocity panels
        ended. ``table_level`` is the level of the tier of the table's report
        (`validate_table`).
    """

    aperture: Aperture
    v_nodes: Array
    v_weights: Array
    v_breaks: Array
    losvd: Array
    mass: tuple[float, ...]
    record: dict[str, float | int | str]


def _surface_and_dispersion(
    projected: ProjectedDispersion,
) -> Callable[[ArrayLike], tuple[Array, Array]]:
    """Return a function of projected radii that gives the first component's
    surface density and squared line-of-sight dispersion.

    The function reads `projected` with `ProjectedDispersion.read`. A row's
    norm check and second-moment check compare the row against the two
    returned values.
    """

    def read(r_2d: ArrayLike) -> tuple[Array, Array]:
        surface, sigma_los = projected.read(np.atleast_1d(r_2d))
        return surface[0], sigma_los[0] ** 2

    return read


def _tangent_parameters(model: Model, tangents: Sequence[str]) -> tuple[str, ...]:
    """Return the names in `tangents` as a tuple, after checking that the
    derivatives in those anisotropy parameters can be taken.

    An empty `tangents` returns an empty tuple without a check. Otherwise
    the function raises InputError in three cases: the model is not
    spatially finite; the model's DF below its table does not follow the
    edge law (the model is not a Method II model); or a name is not an
    anisotropy parameter of the model's anisotropy family
    (`TANGENT_PARAMETERS`). On a model that is not spatially finite, the
    radial dispersion's part below the table reads the DF, and the DF
    changes with the anisotropy. On a truncated model that is not a Method II
    model, the density profile follows the model's own DF, and the DF changes
    with the anisotropy.
    """
    names = tuple(tangents)
    if not names:
        return names
    component = model.components[0]
    if not model.spatially_finite:
        raise InputError(
            f"The tangents of {component.name!r} are not computed, because "
            f"{component.name!r} is untruncated; the part of the radial dispersion "
            "below the table depends on the DF, which moves with the anisotropy."
        )
    rule = model.form.below_floor[0]
    if rule != "edge_law":
        raise InputError(
            f"The tangents of {component.name!r} are not computed, because "
            f"{component.name!r} is not a Method II model (the DF below the DF "
            f"table follows the rule {rule!r}, not the edge law). The profile of a "
            "truncated model that is not a Method II model follows the model's own "
            "DF, so the profile moves with the anisotropy."
        )
    family = component.anisotropy.family
    for name in names:
        if name not in TANGENT_PARAMETERS[family]:
            raise InputError(
                f"The tangents of a model of the anisotropy family {family} are "
                f"taken in the parameters {list(TANGENT_PARAMETERS[family])}; the "
                f"parameter given is {name!r}."
            )
    return names


def dispersion_profile(
    model: Model, settings: Settings, tangents: Sequence[str] = ()
) -> DispersionProfile:
    r"""Return the intrinsic velocity dispersions of a model of one component.

    Parameters
    ----------
    model : Model
        The model.
    settings : Settings
        The numerical settings. The stage reads ``observables.profile_tolerance``
        and ``observables.n_profile_start``.
    tangents : sequence of {"beta_c", "r_a"}, optional
        The anisotropy parameters in which the stage also takes the
        derivative of :math:`\sigma_{\rm r}^2` at the rows, at fixed density
        profile (`DispersionProfile.tangents`). Each name must be a parameter
        of the model's anisotropy family (`TANGENT_PARAMETERS`). Default:
        ``()``, which means no derivative is taken.

    Returns
    -------
    profile : DispersionProfile
        The dispersions at the rows, with the interpolation between rows.

    Raises
    ------
    InputError
        If `tangents` lists a parameter and the model is not spatially
        finite.
    InputError
        If `tangents` lists a parameter and the model's DF below its table
        does not follow the edge law (the model is not a Method II model).
    InputError
        If `tangents` lists a parameter that the model's anisotropy family
        does not have.
    ResultError
        If the integral of the auxiliary density below the table's lowest
        potential is not finite, or if no row has a positive
        :math:`\sigma_{\rm r}^2`.

    Notes
    -----
    The stage evaluates the Jeans integral

    .. math::

        \varrho_{\rm aux}\sigma_{\rm r}^2(R) = \int_0^{P(R)}
        \varrho_{\rm aux}\,{\rm d}K,

    with :math:`R` the 3D radius, :math:`P` the relative potential,
    :math:`K` the potential as the variable of integration and
    :math:`\varrho_{\rm aux}` the auxiliary density. The part of the
    integral below the lowest potential of the profile's table is computed
    from the model's DF. That part is zero on a spatially finite model. The
    tangential dispersion is
    :math:`\sigma_{\rm t}^2 = 2(1 - \beta_\sigma)\sigma_{\rm r}^2`.

    The rows run from the profile's first radius to the profile's last
    radius. `radial_table` adds rows until the interpolation between rows
    agrees with the integral at every interval's midpoint to
    ``profile_tolerance`` in :math:`\sigma_{\rm r}`.

    The derivative rows apply the derivative of the same quadrature rule on
    the same rows and nodes (`radial_dispersion_tangent`). The rows are not
    moved. The limit `DispersionProfile.edge_limit` does not change with the
    anisotropy, because a Method II model's edge law changes with
    :math:`\beta_{\rm c}` as the factor of the model's anisotropy class
    does. The
    result without `tangents` is bit for bit the result with `tangents`,
    apart from `DispersionProfile.tangents`.

    References
    ----------
    Notes Eq. (T:6.1). Chiang+(2026d), Eq. (B43).
    """
    names = _tangent_parameters(model, tangents)
    with timed("dispersion_profile"):
        radial = radial_table(model, settings)
        r = radial["r"]
        sigma_r2 = radial["sigma_r2"][np.newaxis, :]
        beta = beta_sigma(r, model.components[0].anisotropy)[np.newaxis, :]
        profile = model.profile

        def density(radii: Array) -> Array:
            return profile.density(radii)[0]

        tangent_rows = {
            name: radial_dispersion_tangent(
                r,
                radial["sigma_r2"],
                profile.potential,
                density,
                profile.radius,
                model.components[0].anisotropy,
                radial["p_outer"],
                int(radial["certificate"]["n_jeans_nodes"]),
                profile.p_trunc,
                name,
            )[np.newaxis, :]
            for name in names
        }
        return DispersionProfile(
            r=r,
            sigma_r=np.sqrt(sigma_r2),
            sigma_t=np.sqrt(2.0 * (1.0 - beta) * sigma_r2),
            beta_sigma=beta,
            edge_limit=(float(radial["edge_limit"]),),
            certificate=radial["certificate"],
            model=model,
            tangents=tangent_rows,
        )


def projected_dispersion(
    model: Model,
    profile: DispersionProfile,
    settings: Settings,
    tangents: Sequence[str] = (),
) -> ProjectedDispersion:
    """Return the surface density and the line-of-sight velocity dispersion.

    Parameters
    ----------
    model : Model
        The model, of one component.
    profile : DispersionProfile
        The intrinsic velocity dispersions of `model`.
    settings : Settings
        The numerical settings. The stage reads ``observables.n_radii`` and
        ``observables.n_projection_nodes``.
    tangents : sequence of {"beta_c", "r_a"}, optional
        The anisotropy parameters in which the stage also takes the
        derivative of the squared line-of-sight dispersion at the rows
        (`ProjectedDispersion.tangents`). The stage computes each derivative
        from the derivative rows of `profile` in the same parameter.
        Default: ``()``, which means no derivative is taken.

    Returns
    -------
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion at the rows,
        with the interpolation between rows.

    Raises
    ------
    InputError
        If `tangents` lists a parameter, under the three conditions that
        `dispersion_profile` lists for its argument ``tangents``.
    InputError
        If `profile` holds no derivative rows in a parameter that `tangents`
        lists.
    AccuracyError
        If the quadrature gives a row a negative squared line-of-sight
        dispersion.

    Notes
    -----
    One quadrature along the line of sight (`project`), with one density,
    gives the surface density and the squared dispersion at each row. The
    first rows are ``n_radii`` projected radii, logarithmic from the
    profile's first radius to 0.9999 of the profile's outermost radius.
    `certify_projection` adds rows until the interpolation between rows
    agrees with the quadrature at every interval's midpoint to
    `CERTIFICATE_TOLERANCE`.

    The derivative rows apply the derivative of the same quadrature rule on
    the same rows and nodes (`project_tangent`). The derivative of
    :math:`\\sigma_{\\rm r}^2` between the profile's rows comes from
    `radial_integral_tangent_reader`. The result without `tangents` is bit
    for bit the result with `tangents`, apart from
    `ProjectedDispersion.tangents`.

    References
    ----------
    Chiang+(2026d), Eq. (B46).
    """
    names = _tangent_parameters(model, tangents)
    missing = [name for name in names if name not in profile.tangents]
    if missing:
        raise InputError(
            f"The dispersion profile holds no tangents in {missing}; a dispersion "
            "profile computed by dispersion_profile(..., tangents=...) in the same "
            "parameters holds the tangents."
        )
    with timed("projected_dispersion"):
        table = projected_table(
            model, profile._read_sigma_r2(0), settings, CERTIFICATE_TOLERANCE
        )
        negative = np.asarray(table["sigma_los2"]) < 0.0
        if np.any(negative):
            k = int(np.argmin(table["sigma_los2"]))
            raise AccuracyError(
                f"The projection integral gives the squared line-of-sight "
                f"dispersion {float(table['sigma_los2'][k]):.3e}, which is below 0, "
                f"at the projected radius R_2D = {float(table['r_2d'][k]):.4e}, so "
                f"the table does not hold the negative variance. The setting "
                f"observables.n_projection_nodes gives the projection integral's "
                f"number of nodes."
            )
        record = dict(table["certificate"])
        worst_sigma, worst_los = record.pop("worst_first")
        record["worst_first_surface_density"] = worst_sigma
        record["worst_first_sigma_los2"] = worst_los

        def density(radii: Array) -> Array:
            return model.profile.density(radii)[0]

        tangent_rows = {}
        for name in names:
            rows = project_tangent(
                table["r_2d"],
                density,
                profile._read_sigma_r2(0),
                profile._read_tangent(name),
                float(model.profile.r_table_max),
                model.components[0].anisotropy,
                int(settings.observables["n_projection_nodes"]),
                name,
            )
            tangent_rows[name] = rows[np.newaxis, :]
        return ProjectedDispersion(
            r_2d=table["r_2d"],
            surface_density=table["surface_density"][np.newaxis, :],
            sigma_los=np.sqrt(table["sigma_los2"])[np.newaxis, :],
            certificate=record,
            model=model,
            tangents=tangent_rows,
        )


def aperture_dispersion(
    projected: ProjectedDispersion, aperture: Aperture, settings: Settings
) -> Array:
    """Return the line-of-sight velocity dispersion averaged over an aperture.

    The result is the square root of the average of the squared
    line-of-sight dispersion over the aperture, with the surface density as
    weight. The stage reads no velocity distribution. The result is
    therefore an independent reference for the second moment of
    `aperture_losvd`.

    Parameters
    ----------
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion over projected
        radius.
    aperture : Aperture
        The aperture, in internal units.
    settings : Settings
        The numerical settings. The stage reads
        ``observables.n_aperture_nodes`` and
        ``observables.aperture_nodes_per_decade``.

    Returns
    -------
    sigma_los : array, shape (n_components,)
        The averaged line-of-sight dispersion of each component, in internal
        units. A negative average of the squared dispersion is returned as 0.

    Raises
    ------
    InputError
        If the aperture encloses no tabulated surface density.
    InputError
        If the disk inside the projected table's first row carries more than
        `INNER_DISK_TOLERANCE` of the aperture's weight.
    AccuracyError
        If the disk inside the first row must be estimated and the slope
        :math:`-{\\rm d}\\ln\\Sigma/{\\rm d}\\ln R_{\\rm 2D}` between the first
        two rows is not below 2 (``_projection.INNER_SLOPE_MAX``), so that
        the disk's estimate diverges.

    Notes
    -----
    The integral over the aperture is one-dimensional in projected radius.
    The integrand is weighted by the angular measure of the circle of each
    radius inside the aperture (`Aperture.arc`). The integral runs on
    panels between the radii where that angular measure is not smooth
    (`Aperture.breaks`). Each panel has ``n_aperture_nodes`` Gauss-Legendre
    nodes, or ``aperture_nodes_per_decade`` nodes per decade of the panel's
    span where that number is larger. The stage interpolates the table as
    `ProjectedDispersion.read` interpolates the table.

    References
    ----------
    Chiang+(2026d), Eq. (B48) (a rectangle). Chiang+(2026d), Eq. (B50) (a
    centered circle). Chiang+(2026d), Eq. (B51) (an offset circle).
    """
    observables = settings.observables
    r_t = float(projected.model.profile.r_table_max)
    finite = projected.model.spatially_finite
    with timed("aperture_dispersion"):
        mean_square = [
            aperture_mean_square(
                surface_density_reader(
                    projected.r_2d, projected.surface_density[i], r_t, finite
                ),
                dispersion_table_reader(
                    projected.r_2d, np.asarray(projected.sigma_los[i]) ** 2
                ),
                aperture,
                int(observables["n_aperture_nodes"]),
                int(observables["aperture_nodes_per_decade"]),
                INNER_DISK_TOLERANCE,
            )
            for i in range(np.shape(projected.surface_density)[0])
        ]
        return np.sqrt(np.maximum(np.asarray(mean_square), 0.0))


def kept_ranges(
    model: Model, projected: ProjectedDispersion, settings: Settings
) -> KeptRanges:
    r"""Return the ranges of projected radius in which the velocity breaks
    V_rad and V_k are kept.

    A break is kept where the measure of the break, defined in the Notes,
    is above `VRAD_KEEP` (:math:`10^{-8}`). The stage returns empty ranges
    for a model whose route has no angular-momentum factor (``route.lfac``
    false) and no anisotropy radius (``route.SGN`` zero). V_k is never kept
    on a model without an anisotropy radius.

    Parameters
    ----------
    model : Model
        The model, of one component.
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion of `model`.
    settings : Settings
        The numerical settings. The stage reads the ``losvd`` group.

    Returns
    -------
    kept : KeptRanges
        The ranges of projected radius in which each break is kept.

    Notes
    -----
    The measure of V_rad at a projected radius is
    :math:`B = 2D(V_{\rm rad})(V_{\rm end} - V_{\rm rad})/\Sigma`, with
    :math:`D(V_{\rm rad})` the distribution at V_rad from one row computed
    at that velocity alone, V_end the row's largest velocity and
    :math:`\Sigma` the surface density. Where
    :math:`V_{\rm end}^2 B/\sigma_{\rm los}^2` is larger than :math:`B`, the
    measure is :math:`V_{\rm end}^2 B/\sigma_{\rm los}^2`. The measure of V_k
    is the smaller of the same measure over the two panels next to V_k.
    Each measure is evaluated on the base rows. Where a measure crosses
    `VRAD_KEEP` between two base rows, the crossing is bisected in
    :math:`\ln R_{\rm 2D}`.
    """
    rules = ellipse_rules(settings)
    with timed("kept_ranges"):
        route = ellipse_route(model, rules, with_df=False)
        if not (route.lfac or route.SGN != 0.0):
            return KeptRanges(v_rad=np.zeros((0, 2)), v_k=np.zeros((0, 2)))
        route_df = ellipse_route(model, rules)
        base = base_rows(model, settings)
        with start_pool(route_df, key=(model.identity, rules)) as pool:
            v_rad, _ = vrad_kept_ranges(
                route, route_df, base, _surface_and_dispersion(projected), pool.map
            )
            v_k = (
                vk_kept_ranges(
                    route,
                    route_df,
                    base,
                    _surface_and_dispersion(projected),
                    pool.map,
                    float(projected.r_2d[-1]),
                )[0]
                if route.SGN != 0.0
                else np.zeros((0, 2))
            )
        return KeptRanges(v_rad=v_rad, v_k=v_k)


def losvd_row(
    model: Model,
    r_2d: float,
    kept: KeptRanges,
    projected: ProjectedDispersion,
    settings: Settings,
) -> LosvdRow:
    """Return the line-of-sight velocity distribution at one projected radius.

    The stage computes one row alone, with the same velocity panels as the
    row at the same projected radius in `losvd_table`.

    Parameters
    ----------
    model : Model
        The model, of one component.
    r_2d : float
        Projected radius of the row, in internal units.
    kept : KeptRanges
        The kept ranges of `model`.
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion of `model`. The
        row's norm and second moment are checked against these two values.
    settings : Settings
        The numerical settings. The stage reads the ``losvd`` group.

    Returns
    -------
    row : LosvdRow
        The row, normalized, with its checks in `LosvdRow.record`.

    Raises
    ------
    ResultError
        If the layout code of the row does not place the row's interior
        breaks in increasing order strictly between 0 and V_end.
    InputError
        If `r_2d` lies inside the first row or beyond the last row of
        `projected`.
    """
    rules = ellipse_rules(settings)
    with timed("losvd_row"):
        route = ellipse_route(model, rules)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        r_2d = float(r_2d)
        rule, cusp = route.rule(r_2d)
        v, w, breaks, v_escape, v_k, v_end, v_rad, code = route.panels(r_2d)
        report: dict = {}
        d = route.row(r_2d, v, rule, report=report, w=w)
        norm = 2.0 * float(np.dot(w, d))
        surface, sigma_los = projected.read(np.array([r_2d]))
        m2 = 2.0 * float(np.dot(w, v * v * d)) / norm
        values = (
            v_k,
            v_rad,
            float(route.v_edge_many(np.array([r_2d]))[0]),
            float(route.v_c_many(np.array([r_2d]))[0]),
            v_end,
        )
        return LosvdRow(
            r_2d=r_2d,
            layout=str(int(code)),
            v_breaks={k: float(x) for k, x in zip(BREAK_NAMES, values, strict=True)},
            v_escape=float(v_escape),
            v_nodes=v,
            v_weights=w,
            panel_edges=np.asarray(breaks, dtype=np.float64),
            losvd=(d / norm)[np.newaxis, :],
            record=dict(
                norm_against_surface_density=norm
                / (float(surface[0, 0]) * route._df().scale)
                - 1.0,
                m2_against_projected_dispersion=m2 / float(sigma_los[0, 0]) ** 2 - 1.0,
                beyond_escape=2.0
                * float(np.dot(w[v > v_escape], d[v > v_escape]))
                / norm,
                cusp=bool(cusp),
                t_panels=int(report.get("panels", 0)),
                t_change=float(report.get("change", np.nan)),
                t_converged=bool(report.get("converged", False)),
            ),
        )


def losvd_table(
    model: Model,
    kept: KeptRanges,
    projected: ProjectedDispersion,
    settings: Settings,
) -> LosvdTable:
    """Return the table of rows of the line-of-sight velocity distribution.

    Rows are added until the distribution interpolated between two
    neighboring rows agrees with a row computed at the same projected
    radius, to the tolerances of the ``losvd`` settings. An interval of
    projected radius in which that agreement is not reached is recorded in
    `LosvdTable.r_uncertified`, with the interval's reason in
    `LosvdTable.why_uncertified`.

    Parameters
    ----------
    model : Model
        The model, of one component.
    kept : KeptRanges
        The kept ranges of `model`.
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion of `model`.
    settings : Settings
        The numerical settings. The stage reads the ``losvd`` group.

    Returns
    -------
    table : LosvdTable
        The rows, with the record of their placement and of their checks.

    Notes
    -----
    The base rows (`base_rows`) are ``rows_per_decade`` projected radii per
    decade. The base rows run from the profile's first radius to 0.9999 of
    the radius where the rows end (`floor_radius`). The rows end at the
    profile's last radius, or, on a model whose DF is zero below a floor
    energy, at the radius where the potential reaches that floor energy.
    `rows_with_r_breaks` adds rows toward each radius in
    `LosvdTable.r_breaks`, toward the end radius, and inside short stretches
    between breaks.

    The table keeps the rows from the last row at or inside `row_start` to
    the first row at or beyond `row_stop`. Each kept row is the row of the
    full grid at the same radius. `row_start` is set by the setting
    ``losvd.innermost_read_radius``, and `row_stop` by the setting
    ``losvd.outer_mass_share``. A table cut inside the model's edge is read
    only up to its last row.

    Each row is computed in a pool of `get_cores` processes, started by
    spawn, so each process imports the package once. Each row is checked
    against the surface density and the squared line-of-sight dispersion of
    `projected`.
    """
    rules = ellipse_rules(settings)
    with timed("losvd_table"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        route_df = ellipse_route(model, rules)
        base = base_rows(model, settings)
        r_full, r_breaks = route.rows_with_r_breaks(base)
        # The table keeps the rows of the full grid between row_start and
        # row_stop, so each kept row equals the full grid's row at its radius.
        r_rows = cut_rows(
            r_full,
            row_start(model, projected, float(settings.losvd["innermost_read_radius"])),
            row_stop(model, projected, float(settings.losvd["outer_mass_share"])),
        )
        base = base[(base >= r_rows[0]) & (base <= r_rows[-1])]
        read = _surface_and_dispersion(projected)
        surface, sigma_los2 = read(r_rows)
        r_edge = float(route.R_t)
        # A table cut inside the edge is read up to its last row only. A table
        # cut at its inner end is read from its first row only. The radii of
        # r_breaks are those of the full grid in both cases.
        cut = bool(r_rows[-1] < r_full[-1])
        read_range = (
            float(r_rows[0]) if r_rows[0] > r_full[0] else 0.0,
            float(r_rows[-1]) if cut else float("inf"),
        )
        with start_pool(route_df, key=(model.identity, rules)) as pool:
            table, norm_rel, m2_rel, beyond_rel, info = losvd_rows(
                route,
                r_rows,
                surface,
                sigma_los2,
                r_breaks=r_breaks,
                vrad_kept=kept.v_rad,
                vk_kept=kept.v_k,
                sigma_at=read,
                certify=True,
                r_edge=r_edge,
                pool=pool,
                df_scale=route_df._df().scale,
                read_range=read_range,
            )
        rows = tuple(
            LosvdRow(**fields)
            for fields in rows_of(table, norm_rel, m2_rel, beyond_rel, info)
        )
        n_stretches = len(table._st_lo)
        certificate = {
            key: value
            for key, value in info.items()
            if isinstance(value, (int, float, str)) and not isinstance(value, bool)
        }
        floors = (route.model.k_end_floor, route.model.k_end_v0_floor)
        certificate["layer_floor"] = max(
            (float(f) for f in floors if f is not None), default=0.0
        )
        return LosvdTable(
            rows=rows,
            r_base=base,
            r_breaks=np.asarray(table.R_breaks, dtype=np.float64),
            stretch_ends=np.array(
                [
                    [bool(table._st_lt[i]), bool(table._st_ht[i])]
                    for i in range(n_stretches)
                ]
            ).reshape(-1, 2),
            kept=kept,
            order=int(rules.row_order),
            r_uncertified=np.asarray(table.uncert, dtype=np.float64).reshape(-1, 2),
            why_uncertified=tuple(table.uncert_why),
            certificate=certificate,
            identity=dict(
                model=model.identity,
                spatially_finite=bool(model.spatially_finite),
                r_edge=r_edge,
                r_end=float(r_rows[-1]) if cut else float("inf"),
                r_table_max=float(model.profile.r_table_max),
            ),
        )


def aperture_losvd(
    model: Model,
    table: LosvdTable,
    projected: ProjectedDispersion,
    aperture: Aperture,
    settings: Settings,
    record: list[tuple[Array, Array, dict[str, Array]]] | None = None,
) -> ApertureLosvd:
    """Return the line-of-sight velocity distribution inside an aperture.

    The stage interpolates the rows of `table` at the projected radii of the
    aperture's quadrature. The velocity breaks at each radius come from the
    model, and the weights from the surface density. The stage computes no
    new row.

    Parameters
    ----------
    model : Model
        The model, of one component.
    table : LosvdTable
        The table of rows of `model`, covering the aperture.
    projected : ProjectedDispersion
        The surface density and the line-of-sight dispersion of `model`.
    aperture : Aperture
        The aperture, in internal units.
    settings : Settings
        The numerical settings. The stage reads ``observables.n_aperture_nodes``,
        ``observables.aperture_nodes_per_decade`` and the ``losvd`` group.
    record : list, optional
        A list to which the stage appends, for each value of ``losvd[0]`` in
        order, the projected radii at which the value reads the table, their
        weights and the velocity breaks at those radii
        (`aperture_losvd_panels`). Default: ``None``, which means nothing is
        recorded.

    Returns
    -------
    result : ApertureLosvd
        The distribution, the projected mass inside the aperture and the
        record of the computation. ``result.record["table_level"]`` holds the
        level of the tier of the report of `table` (`validate_table`).

    Raises
    ------
    InputError
        If the aperture encloses no tabulated surface density.
    InputError
        If the disk inside the first row of `table`, or of `projected` for a
        table that is not cut at its inner end, carries more than
        `INNER_DISK_TOLERANCE` of the aperture's weight.
    AccuracyError
        If the disk inside the first row must be estimated and the slope
        :math:`-{\\rm d}\\ln\\Sigma/{\\rm d}\\ln R_{\\rm 2D}` between the first
        two rows is not below 2 (``_projection.INNER_SLOPE_MAX``).
    """
    rules = ellipse_rules(settings)
    observables = settings.observables
    n_nodes = int(observables["n_aperture_nodes"])
    per_decade = int(observables["aperture_nodes_per_decade"])
    with timed("aperture_losvd"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
        # The route reads and extends the extrema that earlier apertures on
        # the same table computed and stored in LosvdTable.extrema.
        route._fext = table.extrema.setdefault((model.identity, rules), {})
        stencil = table_stencil(route, table)
        # On a table cut at its inner end, the integral over the aperture
        # starts at the table's first row: the surface density is read from
        # the first row of projected at or beyond that radius.
        r_first = float(table.rows[0].r_2d)
        r_p = np.asarray(projected.r_2d, dtype=np.float64)
        k = (
            int(np.searchsorted(r_p, r_first * (1.0 - END_TOLERANCE)))
            if r_first > r_p[0] * (1.0 + END_TOLERANCE)
            else 0
        )
        surface = surface_density_reader(
            r_p[k:],
            np.asarray(projected.surface_density[0])[k:],
            float(model.profile.r_table_max),
            model.spatially_finite,
        )
        if k:
            surface.inner_note = (
                f"The table's innermost projected radius is {r_first:.4e}, which the "
                f"setting losvd.innermost_read_radius sets; the smallest centered "
                f"circle that the table computes to a relative error of 0.000001 "
                f"has the radius {served_radius(model, projected, r_first):.4e}."
            )
        n_workers = get_cores()

        def pool() -> Any:
            # The table's pool starts when the first radius needs it, and the
            # table keeps the pool for its next apertures.
            return table_pool(table, route, (model.identity, rules, n_workers))

        def extrema(r2: Array) -> None:
            extrema_in_pool(route, r2, pool, 4 * n_workers)

        v, w, d, breaks, info, _, _ = aperture_losvd_panels(
            surface,
            stencil,
            route,
            aperture,
            n_nodes,
            per_decade,
            INNER_DISK_TOLERANCE,
            record=record,
            extrema=extrema if n_workers > 1 else None,
        )
        r, weights, sigma = aperture_nodes(
            surface, aperture, n_nodes, per_decade, INNER_DISK_TOLERANCE
        )
        centered = aperture.shape == "circle" and aperture.r_off == 0.0
        mass = float(np.dot(weights, sigma)) * (2.0 * np.pi if centered else 1.0)
        info["table_level"] = validate_table(table, projected).level
        return ApertureLosvd(
            aperture=aperture,
            v_nodes=v,
            v_weights=w,
            v_breaks=np.asarray(breaks, dtype=np.float64),
            losvd=np.asarray(d)[np.newaxis, :],
            mass=(mass,),
            record={
                key: value
                for key, value in info.items()
                if isinstance(value, (int, float, str)) and not isinstance(value, bool)
            },
        )


def gauss_hermite_moments(v: Array, weights: Array, losvd: Array) -> dict[str, float]:
    r"""Return the Gauss-Hermite moments of a velocity distribution.

    Parameters
    ----------
    v : array, shape (n,)
        Velocity nodes over the whole distribution, at both signs of
        velocity, in internal units. A `LosvdRow` or an `ApertureLosvd`
        holds nodes at non-negative velocity only, so its nodes, weights and
        values must be mirrored to negative velocity first.
    weights : array, shape (n,)
        Quadrature weights of `v`.
    losvd : array, shape (n,)
        The distribution :math:`D(V)` at `v`. The function divides
        :math:`D` by its integral with `weights` before it takes the moments.

    Returns
    -------
    moments : dict
        ``mean`` and ``sigma`` are the mean and the dispersion of the
        normalized distribution, in internal units. The function uses them
        as the mean :math:`V_*` and the dispersion :math:`\sigma_*` of the
        Gaussian. ``h3`` and ``h4`` are the Gauss-Hermite moments
        :math:`h_3` and :math:`h_4`. ``s3`` and ``s4`` are the standardized
        third and fourth moments :math:`s_3` and :math:`s_4`, by direct sums
        over the nodes.

    Notes
    -----
    With :math:`y = (V - V_*)/\sigma_*`, the moments are

    .. math::

        h_n = 2\sqrt{\pi}\int D(V)\,H_n(y)\,\frac{e^{-y^2/2}}{\sqrt{2\pi}}
        \,{\rm d}V,

    with the Hermite polynomials of van der Marel and Franx (1993),
    :math:`H_3 = (2y^3 - 3y)/\sqrt{3}` and
    :math:`H_4 = (4y^4 - 12y^2 + 3)/(2\sqrt{6})`. To first order in
    :math:`h_3` and :math:`h_4`, :math:`s_3 = 4\sqrt{3}\,h_3` and
    :math:`s_4 = 3 + 8\sqrt{6}\,h_4`.

    References
    ----------
    Chiang+(2026d), Eq. (B63). Chiang+(2026d), Eq. (B61). Notes Eq. (T:6.3).
    """
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(losvd, dtype=np.float64)
    d = d / float(np.dot(w, d))
    mean = float(np.dot(w, v * d))
    dv = v - mean
    sigma = float(np.sqrt(np.dot(w, dv * dv * d)))
    y = dv / sigma
    gauss = 2.0 * np.sqrt(np.pi) * np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)
    h3 = (2.0 * y**3 - 3.0 * y) / np.sqrt(3.0)
    h4 = (4.0 * y**4 - 12.0 * y * y + 3.0) / (2.0 * np.sqrt(6.0))
    return {
        "mean": mean,
        "sigma": sigma,
        "h3": float(np.dot(w, d * h3 * gauss)),
        "h4": float(np.dot(w, d * h4 * gauss)),
        "s3": float(np.dot(w, d * y**3)),
        "s4": float(np.dot(w, d * y**4)),
    }


GAUSS_HERMITE_NAMES = ("mean", "sigma", "h3", "h4")


@dataclass(frozen=True)
class GaussHermiteTable:
    """The Gauss-Hermite moments of a table's rows, interpolated in radius.

    `gauss_hermite_table` builds the object.

    Parameters
    ----------
    r_2d : array, shape (n,)
        Projected radius of each row that has a distribution with a positive
        integral, increasing, in internal units.
    moments : dict of str to array, shape (n,)
        ``mean``, ``sigma``, ``h3`` and ``h4`` of each row
        (`gauss_hermite_moments`).

    Notes
    -----
    Each moment is interpolated between rows by a not-a-knot cubic spline in
    ``ln R_2D``, the interpolation that `dispersion_table_reader` applies to
    the squared line-of-sight dispersion. `read` raises InputError at a
    projected radius inside the first row or beyond the last row. A caller
    of `read_traced` must raise that error before it traces `read_traced`.
    At the projected radius of a row, `read` and `read_traced` return the
    row's own moments to the bit.
    """

    r_2d: Array
    moments: dict[str, Array]

    def __post_init__(self) -> None:
        # The cubic splines of the moments are fitted once, so every call of
        # read and read_traced evaluates the same splines.
        x = np.log(self.r_2d)
        splines = {k: CubicSpline(x, self.moments[k]) for k in GAUSS_HERMITE_NAMES}
        object.__setattr__(self, "_spline_cache", splines)

    def _splines(self) -> dict[str, CubicSpline]:
        return self._spline_cache  # type: ignore[attr-defined]

    def read(self, r_2d: ArrayLike) -> dict[str, Array]:
        """Return ``mean``, ``sigma``, ``h3`` and ``h4`` at projected radii.

        A projected radius inside the first row or beyond the last row
        raises InputError.
        """
        r = np.atleast_1d(np.asarray(r_2d, dtype=np.float64))
        _refuse_below(r, float(self.r_2d[0]), "Gauss-Hermite moments")
        _refuse_beyond(r, float(self.r_2d[-1]), "Gauss-Hermite moments")
        return {
            k: np.asarray(self.read_traced(r, np)[k], dtype=np.float64)
            for k in GAUSS_HERMITE_NAMES
        }

    def read_traced(self, r_2d: Any, xp: Any = np) -> dict[str, Any]:
        """Return ``mean``, ``sigma``, ``h3`` and ``h4`` at projected radii,
        with `r_2d` traceable by JAX.

        The method evaluates the coefficients of the cubic splines in the
        array module `xp` (``numpy`` or ``jax.numpy``). Beyond the last row
        every moment is zero. Inside the first row the cubic of the first
        interval is extended, and the values are not checked.
        """
        log_r = xp.log(xp.maximum(r_2d, TINY))
        last = float(self.r_2d[-1])
        out = {}
        for k, spline in self._splines().items():
            value = spline_value(spline.x, spline.c, log_r, xp)
            # At the last row's radius, the moment is the last row's stored
            # value: the cubic of the last interval gives that value only to
            # rounding.
            value = xp.where(r_2d == last, float(self.moments[k][-1]), value)
            out[k] = xp.where(r_2d > last * (1.0 + END_TOLERANCE), 0.0, value)
        return out


def gauss_hermite_table(table: LosvdTable) -> GaussHermiteTable:
    """Return the Gauss-Hermite moments of a table's rows.

    Each row holds nodes at non-negative velocity only. The function mirrors
    each row to both signs of velocity and takes the row's moments with
    `gauss_hermite_moments`. A row whose integral with its weights is not
    positive has no moments and is left out.
    """
    radii: list[float] = []
    values: dict[str, list[float]] = {k: [] for k in GAUSS_HERMITE_NAMES}
    for row in table.rows:
        v = np.asarray(row.v_nodes, dtype=np.float64)
        w = np.asarray(row.v_weights, dtype=np.float64)
        d = np.asarray(row.losvd, dtype=np.float64)[0]
        if not float(np.dot(w, d)) > 0.0:
            continue
        g = gauss_hermite_moments(
            np.concatenate([-v[::-1], v]),
            np.concatenate([w[::-1], w]),
            np.concatenate([d[::-1], d]),
        )
        radii.append(float(row.r_2d))
        for k in GAUSS_HERMITE_NAMES:
            values[k].append(g[k])
    return GaussHermiteTable(
        r_2d=np.asarray(radii),
        moments={k: np.asarray(values[k]) for k in GAUSS_HERMITE_NAMES},
    )


def validate_table(
    table: LosvdTable, projected: ProjectedDispersion
) -> ValidationReport:
    r"""Return the validation report of a table of rows.

    The report has six checks: ``norm``, ``second_moment``, ``unjudged``,
    ``t_converged``, ``certificate`` and ``layer_floor``.

    Notes
    -----
    A row is judged where both of the row's relative errors,
    ``norm_against_surface_density`` and
    ``m2_against_projected_dispersion``, are finite. ``norm`` and
    ``second_moment`` are the largest absolute relative errors over the
    judged rows, each at the projected radius of its row.

    ``unjudged`` is infinite, at the first such row, where a row is not
    judged while the surface density at the row is positive. A row is not
    judged when the row's integral is zero or the line-of-sight dispersion
    at the row is not positive. A row where the surface density is zero has
    nothing to judge. ``t_converged`` is infinite, at the first such row,
    where the quadrature along the line of sight of any row did not converge
    (the row's ``t_converged`` is false). An infinite error puts the report
    in the tier ``"refused"``.

    ``certificate`` is the table's last estimate of the error of the
    interpolation between rows (``est_last``), in units of the table's
    tolerances, multiplied by the larger tolerance `ROW_TOL_PW`
    (:math:`10^{-7}`). The product bounds both the pointwise estimate and
    the second-moment estimate.

    ``layer_floor`` is nonzero where the quadrature layers toward the point
    :math:`L = 0`, with :math:`L` the angular momentum, stop converging. The
    layers stop converging where the anisotropy's end-point power is
    negative, for :math:`\beta_{\rm c}` between 1/2 and 1. Every graded
    layer then carries the quadrature rule's own error. ``layer_floor`` is
    that error floor, the larger of the floors at :math:`V > 0` and at
    :math:`V = 0`. The layer count is the least count whose error is within
    :math:`10^{-8}` of the floor. `k_end` raises ResultError for a floor
    above :math:`10^{-6}`. ``layer_floor`` is zero where the layers converge.
    """
    r = np.array([row.r_2d for row in table.rows])
    norm = np.array([row.record["norm_against_surface_density"] for row in table.rows])
    m2 = np.array([row.record["m2_against_projected_dispersion"] for row in table.rows])
    surface = projected.read(r)[0][0]
    judged = np.isfinite(norm) & np.isfinite(m2)
    refused = ~judged & (surface > 0.0)

    def worst(values: Array) -> Check:
        a = np.where(judged, np.abs(values), -1.0)
        i = int(np.argmax(a))
        return Check(error=float(max(a[i], 0.0)), where=float(r[i]))

    unjudged = (
        Check(error=float("inf"), where=float(r[np.flatnonzero(refused)[0]]))
        if refused.any()
        else Check(error=0.0)
    )
    open_t = np.array([not row.record["t_converged"] for row in table.rows])
    t_converged = (
        Check(error=float("inf"), where=float(r[np.flatnonzero(open_t)[0]]))
        if open_t.any()
        else Check(error=0.0)
    )
    estimate = float(table.certificate.get("est_last", float("nan")))
    return ValidationReport(
        keys=(
            "norm",
            "second_moment",
            "unjudged",
            "t_converged",
            "certificate",
            "layer_floor",
        ),
        checks={
            "norm": worst(norm),
            "second_moment": worst(m2),
            "unjudged": unjudged,
            "t_converged": t_converged,
            "certificate": Check(error=estimate * ROW_TOL_PW),
            "layer_floor": Check(
                error=float(table.certificate.get("layer_floor", 0.0))
            ),
        },
    )


def validate_aperture(result: ApertureLosvd, sigma_los: Array) -> ValidationReport:
    r"""Return the validation report of an aperture distribution.

    The report has two checks, ``norm`` and ``second_moment``. The tier of
    the report of the table that the distribution was read from enters as
    the report's parent level.

    Notes
    -----
    ``norm`` is the absolute difference between 1 and the distribution's
    integral over both signs of velocity with its own weights
    (``record["norm_file"]``). ``second_moment`` is the absolute relative
    difference between the distribution's second moment, divided by that
    integral, and :math:`\sigma_{\rm los}^2`, with :math:`\sigma_{\rm los}`
    the first entry of `sigma_los`. `sigma_los` comes from
    `aperture_dispersion`, which reads no velocity distribution. The parent
    level is ``record["table_level"]``, the level of the tier of the
    table's report (`validate_table`).
    """
    norm = float(result.record["norm_file"])
    m2 = float(result.record["m2_file"]) / norm
    sigma2 = float(np.asarray(sigma_los, dtype=np.float64).ravel()[0]) ** 2
    return ValidationReport(
        keys=("norm", "second_moment"),
        checks={
            "norm": Check(error=abs(norm - 1.0)),
            "second_moment": Check(error=abs(m2 / sigma2 - 1.0)),
        },
        parents_level=int(result.record["table_level"]),
    )


# The largest number of times the velocity distribution's angle rule doubles its
# nodes per panel while the change of the distribution between the rule and the
# doubled rule is above velocity_tolerance (1e-6 in the default settings, 1e-4 in
# the quick settings). After the last doubling the change is recorded as it is.
VELOCITY_DOUBLINGS = 3
# The relative distance from V* within which velocities are left out of the
# halving check. On a DF that diverges at its floor, the velocity distribution
# itself diverges at V*, so the change between rules is not defined there.
VELOCITY_KINK_GAP = 1e-9


@dataclass(frozen=True)
class VelocityDistribution:
    r"""The local velocity distribution of a model at one 3D radius.

    `velocity_distribution` builds the object. Every quantity is in internal
    units, whatever the unit system of the model.

    Parameters
    ----------
    r : float
        3D radius :math:`R`, in internal units.
    v_nodes : array, shape (n,)
        Velocity nodes, in internal units: Gauss-Legendre nodes on panels
        graded geometrically toward both ends of each interval between
        neighboring entries of `panel_edges`.
    v_weights : array, shape (n,)
        Quadrature weights of `v_nodes`. The checks in `record` integrate
        over velocity with these weights.
    panel_edges : array
        The velocities at which the distribution is not smooth, in increasing
        order: 0; :math:`V_*`, the velocity above which the DF's support
        excludes part of the directions of the velocity; :math:`V_{\rm esc}`,
        where :math:`V_{\rm esc}` is below :math:`V_{\rm max}`; and
        :math:`V_{\rm max}`. :math:`V_*` is included only where it lies
        strictly between 0 and :math:`V_{\rm max}`.
    pdf : array, shape (n,)
        The distribution :math:`p(V\,|\,R)` at `v_nodes`, per unit internal
        velocity.
    v_esc : float
        Escape velocity :math:`V_{\rm esc} = \sqrt{2P(R)}`, with :math:`P` the
        relative potential.
    v_max : float
        The largest velocity in the DF's support, :math:`V_{\rm max}`. With
        :math:`K_{\rm c}` the DF's floor energy (0 where the DF has no
        floor), :math:`V_{\rm max} = \sqrt{2(P - K_{\rm c})}` for an
        isotropic model, a constant-anisotropy model and a Type I model, and
        :math:`V_{\rm max} = \sqrt{2(P - K_{\rm c})/(1 - R^2/R_{\rm a}^2)}`
        for a Type II model.
    unbound_fraction : float
        The integral of `pdf` over velocities above :math:`V_{\rm esc}`: the
        share of particles on unbound orbits, which only a Type II DF
        holds.
    record : dict
        ``norm`` is the integral of `pdf` less 1. ``second_moment`` is the
        mean squared velocity over :math:`\sigma_{\rm r}^2(3 - 2\beta_\sigma)`
        less 1, with :math:`\sigma_{\rm r}` and :math:`\beta_\sigma` from
        `dispersion_profile`. ``halving`` is the largest change of `pdf` at
        `v_nodes` when the angle rule doubles its nodes, over the largest
        value of the doubled rule's distribution. ``n_angle_nodes`` and
        ``n_angle_panels`` define the angle rule of `pdf`. ``report`` is the
        `ValidationReport` of the three checks, and ``tier`` is the report's
        tier.
    model : Model
        The model.
    """

    r: float
    v_nodes: Array
    v_weights: Array
    panel_edges: Array
    pdf: Array
    v_esc: float
    v_max: float
    unbound_fraction: float
    record: dict[str, Any]
    model: Model

    def read(self, v: ArrayLike) -> Array:
        """Return :math:`p(V\\,|\\,R)` at velocities `v`, in internal units.

        The method evaluates the angle integral with the rule that `record`
        gives (``n_angle_nodes`` and ``n_angle_panels``), the rule whose
        change ``halving`` measures. The result has the shape of `v`. The
        result is zero at negative velocities and at velocities above `v_max`.
        """
        velocities = np.asarray(v, dtype=np.float64)
        sm = velocity_model(self.model)
        psi, xi, scale = _velocity_point(self.model, sm, self.r)
        inside = (velocities >= 0.0) & (velocities <= self.v_max)
        out = np.zeros(velocities.shape)
        u = velocities[inside]
        n_nodes = int(self.record["n_angle_nodes"])
        n_panels = int(self.record["n_angle_panels"])
        out[inside] = (
            scale
            * u ** (2.0 - 2.0 * sm.b)
            * angle_integral(sm, psi, xi, u, n_nodes, n_panels)
        )
        return out


def _velocity_point(
    model: Model, sm: VelocityModel, r: float
) -> tuple[float, float, float]:
    """Return three values at the 3D radius `r`: the relative potential
    :math:`P(r)`; :math:`\\xi = (r/r_{\\rm a})^2`, or 0 for a model without
    an anisotropy radius; and the factor :math:`4\\pi r^{-2b}/\\varrho(r)`,
    with :math:`b` the exponent ``sm.b`` and :math:`\\varrho` the density."""
    rr = np.array([r])
    psi = float(np.asarray(model.profile.potential(rr))[0])
    rho = float(np.atleast_2d(np.asarray(model.profile.density(rr)))[0, 0])
    xi = 0.0 if sm.sign == 0.0 else (r / sm.r_a) ** 2
    return psi, xi, 4.0 * np.pi * r ** (-2.0 * sm.b) / rho


def velocity_distribution(
    model: Model,
    r: float,
    settings: Settings,
    profile: DispersionProfile | None = None,
) -> VelocityDistribution:
    r"""Return the local velocity distribution of a model at one 3D radius.

    The distribution :math:`p(V\,|\,R)` is the probability density of the
    velocity :math:`V` of the model's particles at the 3D radius :math:`R`. The
    call returns the distribution at one radius, together with the checks
    of the distribution and their validation report. `pdf_v` evaluates the
    same distribution at many radii and velocities, in the model's unit system.

    Parameters
    ----------
    model : Model
        The model. The model has one component.
    r : float
        3D radius :math:`R`. The radius is in internal units in all three
        unit systems: a model set to the model system or to the physical
        system (`Model.to`) still takes `r` in internal units. An Astropy
        quantity is not accepted. The radius must lie at or beyond the
        innermost radius of the model's tabulated profile and below the
        outermost radius of the model's tabulated profile. For a Type II
        model, the radius must also lie below the anisotropy radius
        :math:`R_{\rm a}`. A radius outside that range raises InputError.
    settings : Settings
        The numerical settings. The call reads three fields of the
        ``observables`` group: ``n_velocity_angle_nodes``,
        ``n_velocity_angle_panels`` and ``velocity_tolerance``. The settings table
        (:doc:`/reference/settings`) describes the three fields.
    profile : DispersionProfile, optional
        The intrinsic velocity dispersions of `model`, as the attribute
        ``profile`` of ``pp.LosTable(model)`` holds them. The second moment
        of the distribution is checked against these dispersions. Default:
        ``None``, which means the call computes the intrinsic velocity
        dispersions of `model`.

    Returns
    -------
    distribution : VelocityDistribution
        The distribution at `r`. Every value is in internal units in all
        three unit systems of the model: velocities in the internal unit of
        velocity, and the distribution per unit internal velocity. The
        object holds the 3D radius ``r``; the velocity nodes ``v_nodes`` and
        their quadrature weights ``v_weights``; the distribution ``pdf`` at
        the nodes; the escape velocity ``v_esc``; the largest velocity of the
        distribution's support ``v_max``; and the share of the distribution
        above the escape velocity ``unbound_fraction``. The method
        ``read(v)`` returns the distribution at any velocities `v`, and zero at
        negative velocities and at velocities above ``v_max``. The integral of the
        distribution over velocity is 1 to within ``record["norm"]``.
        ``record`` holds the relative errors of three checks (``"norm"``,
        ``"second_moment"`` and ``"halving"``), their validation report
        ``record["report"]`` and the report's tier ``record["tier"]``. The
        call raises no error for a large check error: a check error above
        :math:`10^{-2}` gives the tier ``"refused"`` in ``record["tier"]``.

    Raises
    ------
    InputError
        If `model` has more than one component.
    InputError
        If `r` lies inside the innermost radius or at or beyond the
        outermost radius of the model's tabulated profile.
    InputError
        If `model` is a Type II model and `r` lies at or beyond the
        anisotropy radius :math:`R_{\rm a}`.
    InputError
        If `r` lies outside the radii at which the intrinsic velocity
        dispersions, given as `profile` or computed by the call, are
        tabulated.
    ResultError
        If `profile` is not given and the call cannot form the model's
        radial velocity dispersion: the contribution of the DF below the
        tabulated potential is not finite, or the dispersion is positive at
        no tabulated radius.

    See Also
    --------
    pdf_v : The local velocity distribution at many radii and velocities, in the
        model's unit system.
    sigma_r : The radial velocity dispersion, which the second-moment check
        reads.
    LosTable : The tables of a model, whose attribute ``profile`` can be
        passed as `profile`.

    Notes
    -----
    The distribution is

    .. math::

        p(V\,|\,R) = \frac{4\pi}{\varrho(R)}\, R^{-2b} V^{2-2b}
        \int_0^1 (1-t^2)^{-b} F(\mathcal{Q})\,{\rm d}t,

    with :math:`\varrho` the density, :math:`F` the distribution function
    (DF), :math:`t = \cos\eta` and :math:`\eta` the angle between the
    velocity and the radius vector. The exponent :math:`b` is the central
    anisotropy :math:`\beta_{\rm c}` for a constant-anisotropy model and for
    a Cuddeford-Osipkov-Merritt (COM) model, and :math:`b = 0` for an
    isotropic model and for an Osipkov-Merritt model. The argument of the
    DF is

    .. math::

        \mathcal{Q} = P(R) - \frac{V^2}{2}(1 + s\xi)
        + \frac{s\xi V^2}{2}\,t^2,

    with :math:`P` the relative potential, :math:`\xi = R^2/R_{\rm a}^2`,
    and :math:`s = +1` for a Type I model, :math:`s = -1` for a Type II
    model and :math:`s = 0` for a model without an anisotropy radius. The
    integral runs over the DF's support, :math:`\mathcal{Q} > K_{\rm c}`,
    with :math:`K_{\rm c}` the DF's floor energy (0 where the DF has no
    floor).

    The support excludes part of the directions above the velocity
    :math:`V_*`: :math:`V_* = \sqrt{2(P - K_{\rm c})/(1 + \xi)}` for a
    Type I model, and :math:`V_* = \sqrt{2(P - K_{\rm c})}` for a Type II
    model. The support ends at
    :math:`V_{\rm max} = \sqrt{2(P - K_{\rm c})/(1 - \xi)}` for a Type II
    model and at :math:`V_{\rm max} = \sqrt{2(P - K_{\rm c})}` for every
    other model. The escape velocity is :math:`V_{\rm esc} = \sqrt{2P}`. For a
    Type II model, :math:`V_{\rm max}` is above :math:`V_{\rm esc}` where
    :math:`K_{\rm c}` is below :math:`\xi P(R)`. `unbound_fraction` is the
    share of the distribution between :math:`V_{\rm esc}` and
    :math:`V_{\rm max}`.

    The angle integral starts with ``n_velocity_angle_nodes`` nodes per
    interval. The nodes double, at most three times, until the largest
    change of the distribution between the integral and the integral with
    doubled nodes is at most ``velocity_tolerance`` times the largest absolute
    value of the distribution with doubled nodes. ``record["halving"]`` is
    the ratio of the two maxima for the kept distribution.
    ``velocity_tolerance`` is :math:`10^{-6}` in the default settings and
    :math:`10^{-4}` in the quick settings. Velocities within a
    relative distance of :math:`10^{-9}` from :math:`V_*` are left out of
    that comparison. On a DF that diverges at its floor energy, the
    distribution diverges at :math:`V_*`, and the divergence is integrable.

    References
    ----------
    Chiang+(2026d), Eq. (B38). Chiang+(2026d), Eq. (B39). Chiang+(2026d),
    Eq. (B40). Notes Eq. (T:6.4).

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.observables.stages import velocity_distribution
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> distribution = velocity_distribution(model, 1.0, model.settings)
    >>> distribution.pdf.shape == distribution.v_nodes.shape
    True
    """
    if len(model.components) != 1:
        raise InputError(
            f"The velocity distribution is computed for a model of one component; "
            f"the model given has {len(model.components)} components."
        )
    observables = settings.observables
    n_nodes = int(observables["n_velocity_angle_nodes"])
    n_panels = int(observables["n_velocity_angle_panels"])
    tolerance = float(observables["velocity_tolerance"])
    with timed("velocity_distribution"):
        sm = velocity_model(model)
        r = float(r)
        r_lo, r_hi = float(model.profile.r[0]), float(model.profile.r_table_max)
        if sm.sign < 0.0:
            r_hi = min(r_hi, sm.r_a)
        if not r_lo <= r < r_hi:
            raise InputError(
                f"The velocity distribution is read at a radius in [{r_lo:.6e}, "
                f"{r_hi:.6e}), the range of the profile's tabulated radii, cut at "
                f"the anisotropy radius for a Type II model; the radius given is "
                f"{r!r}."
            )
        psi, xi, scale = _velocity_point(model, sm, r)
        v_esc, v_kink, v_max = velocity_breaks(sm, psi, xi)
        edges = np.array(
            sorted({0.0, v_max} | {x for x in (v_kink, v_esc) if 0.0 < x < v_max})
        )
        parts = [
            graded_rule(a, b, n_panels, n_nodes)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        v = np.concatenate([p[0] for p in parts])
        w = np.concatenate([p[1] for p in parts])
        factor = scale * v ** (2.0 - 2.0 * sm.b)
        away = np.abs(v / v_kink - 1.0) > VELOCITY_KINK_GAP

        def pdf_at(n: int) -> Array:
            return factor * angle_integral(sm, psi, xi, v, n, n_panels)

        # The distribution kept is that of the coarser rule of the last
        # compared pair, and halving is the change between the two rules of
        # that pair.
        pdf = pdf_at(n_nodes)
        for doubling in range(VELOCITY_DOUBLINGS + 1):
            finer = pdf_at(2 * n_nodes)
            peak = float(np.max(np.abs(finer[away])))
            halving = float(np.max(np.abs(finer - pdf)[away])) / peak
            if halving <= tolerance or doubling == VELOCITY_DOUBLINGS:
                break
            n_nodes, pdf = 2 * n_nodes, finer
        jeans = dispersion_profile(model, settings) if profile is None else profile
        sigma_r, _, beta = (
            float(np.asarray(x).ravel()[0]) for x in jeans.read(np.array([r]))
        )
        norm = float(np.dot(w, pdf))
        m2 = float(np.dot(w, v * v * pdf))
        report = ValidationReport(
            keys=("norm", "second_moment", "halving"),
            checks={
                "norm": Check(error=abs(norm - 1.0), where=r),
                "second_moment": Check(
                    error=abs(m2 / (sigma_r**2 * (3.0 - 2.0 * beta)) - 1.0), where=r
                ),
                "halving": Check(error=halving, where=r),
            },
        )
        return VelocityDistribution(
            r=r,
            v_nodes=v,
            v_weights=w,
            panel_edges=edges,
            pdf=pdf,
            v_esc=v_esc,
            v_max=v_max,
            unbound_fraction=float(np.dot(w[v > v_esc], pdf[v > v_esc])),
            record=dict(
                norm=norm - 1.0,
                second_moment=m2 / (sigma_r**2 * (3.0 - 2.0 * beta)) - 1.0,
                halving=halving,
                n_angle_nodes=n_nodes,
                n_angle_panels=n_panels,
                report=report,
                tier=report.tier,
            ),
            model=model,
        )
