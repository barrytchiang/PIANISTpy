"""Stages of the observables and their results.

Second moments follow from the density and the potential of a model::

    dispersion_profile -> projected_dispersion -> aperture_dispersion

Line-of-sight velocity distributions follow from its DF::

    kept_ranges -> losvd_row -> losvd_table -> aperture_losvd

The two routes share nothing but the model, so each checks the other: the
norm of a row against the surface density, its second moment against the
projected dispersion, and the second moment of an aperture distribution
against the aperture dispersion.

A request for one observable computes only what it needs; the prepared
`LosTable` of a model holds the tables and reads from them. The two ways
are held to the accuracy goal against each other by the tests.
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
    "SpeedDistribution",
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
    "speed_distribution",
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
from ..exceptions import InputError
from ..runtime import get_workers, timed
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..validation import GOAL, Check, ValidationReport
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
    dispersion_table_reader,
    project_tangent,
    spline_value,
    surface_density_reader,
)
from ._speed import SpeedModel, angle_integral, graded_rule, speed_breaks, speed_model
from ._tables import projected_table, radial_table
from .aperture import Aperture

# The relative tolerance of a projected table's reading between rows: a hundredth
# of the accuracy goal, so that the reading adds nothing measurable.
CERTIFICATE_TOLERANCE = 0.01 * GOAL
# The largest estimated share of an aperture's weight inside the first row of
# the projected table: the accuracy goal, since that disk is not computed.
INNER_DISK_TOLERANCE = GOAL


@dataclass(frozen=True)
class DispersionProfile:
    r"""Intrinsic second moments of a model on a grid of radius.

    Parameters
    ----------
    r : array, shape (n,)
        Radius, in internal units.
    sigma_r : array, shape (n_components, n)
        Radial velocity dispersion.
    sigma_t : array, shape (n_components, n)
        Tangential velocity dispersion, of both tangential components
        together: :math:`\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2`, so that
        :math:`\beta_\sigma = 1 - \sigma_{\rm t}^2 / (2 \sigma_r^2)`.
    beta_sigma : array, shape (n_components, n)
        Anisotropy profile :math:`\beta_\sigma(R)`.
    edge_limit : tuple of float
        For a spatially finite model, the limit of
        :math:`\sigma_r^2 / \Psi` at the edge of each component, which the
        reading between grid points uses there; NaN otherwise.
    certificate : dict
        How the rows were placed (`radial_table`): the tolerance, how they
        ended, the rounds, the rows, the first and last worst reading at an
        interval's midpoint and the intervals left above the tolerance.
    model : Model
        The model the profile belongs to: its potential is the variable of
        the reading between grid points, and its anisotropies give
        `beta_sigma` at any radius.
    tangents : dict of str to array, shape (n_components, n)
        With ``tangents`` asked of `dispersion_profile`, the derivative of
        :math:`\sigma_r^2` on the same rows in each anisotropy parameter
        named, on the model's fixed profile, in internal units; empty
        otherwise.

    Notes
    -----
    The rule that reads :math:`\sigma_r^2` between grid points is part of
    the result (`read`): every check of a velocity distribution reads the
    second moment through it. It is the cubic Hermite in :math:`P` of the
    Jeans integral :math:`\varrho_{\rm aux}\sigma_r^2` with its exact slopes
    :math:`\varrho_{\rm aux}` (`radial_integral_reader`), held inside the
    innermost row; on a spatially finite model it reads
    :math:`h = \sigma_r^2 / P` from the outermost row to `edge_limit` at the
    edge.
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
        r""":math:`\varrho_{\rm aux} = \varrho / g` of one component."""
        profile = self.model.profile
        anisotropy = self.model.components[component].anisotropy

        def rho_aux(r: Array) -> Array:
            return profile.density(r)[component] / anisotropy.factor(r)

        return rho_aux

    def _read_tangent(
        self, parameter: str, component: int = 0
    ) -> Callable[[ArrayLike], Array]:
        r"""The derivative of the reading of :math:`\sigma_r^2` of one
        component in one anisotropy parameter (`tangents`)."""
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
        r"""The reading of :math:`\sigma_r^2` of one component."""
        return radial_integral_reader(
            self.r,
            np.asarray(self.sigma_r)[component] ** 2,
            self.model.profile.potential,
            self._rho_aux(component),
            self.model.spatially_finite,
            self.edge_limit[component],
        )

    def read(self, r: ArrayLike) -> tuple[Array, Array, Array]:
        """Return `sigma_r`, `sigma_t` and `beta_sigma` at any radius."""
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
    """Surface density and line-of-sight dispersion over projected radius.

    Both come from one quadrature with one density, on rows added until the
    table reads itself between rows within the tolerance of the settings.

    Parameters
    ----------
    r_2d : array, shape (n,)
        Projected radius, in internal units.
    surface_density : array, shape (n_components, n)
        Surface density.
    sigma_los : array, shape (n_components, n)
        Line-of-sight velocity dispersion.
    certificate : dict
        How the rows ended: the reason, the rows added and the worst reading
        between rows.
    model : Model
        The model: where its lines of sight end (the last row of its profile)
        and whether it is spatially finite fix the reading between rows.
    tangents : dict of str to array, shape (n_components, n)
        With ``tangents`` asked of `projected_dispersion`, the derivative of
        the squared line-of-sight dispersion on the same rows in each
        anisotropy parameter named, on the model's fixed profile, in internal
        units; empty otherwise. The surface density does not move.

    Notes
    -----
    Between rows the surface density is read through
    ``h = Sigma / (R_t^2 - R_2D^2)^(1/2)``, a not-a-knot cubic in
    ``ln(R / (1 - R / R_t))`` on a spatially finite model and in ln R
    otherwise, and the squared dispersion
    by a not-a-knot cubic in ln R: the readings the table checks its rows
    for. Both are zero beyond the last row and refused inside the first.
    """

    r_2d: Array
    surface_density: Array
    sigma_los: Array
    certificate: dict[str, float | int | str]
    model: Model
    tangents: dict[str, Array] = field(default_factory=dict)

    def read(self, r_2d: ArrayLike) -> tuple[Array, Array]:
        """Return the surface density and `sigma_los` at any radius."""
        r_t = float(self.model.profile.r_table_max)
        finite = self.model.spatially_finite
        surface, sigma_los = [], []
        for i in range(np.shape(self.surface_density)[0]):
            surface.append(
                surface_density_reader(self.r_2d, self.surface_density[i], r_t, finite)(
                    r_2d
                )
            )
            sigma_los.append(
                np.sqrt(
                    dispersion_table_reader(
                        self.r_2d, np.asarray(self.sigma_los[i]) ** 2
                    )(r_2d)
                )
            )
        return np.stack(surface), np.stack(sigma_los)

    def read_traced(self, r_2d: Any, xp: Any = np) -> tuple[Any, Any]:
        """Return the surface density and the squared `sigma_los` at radii,
        with `r_2d` traceable under ``jax.numpy``.

        The same numbers as `read` (squared) inside the table, to rounding:
        the readers' own cubics evaluated in the array module `xp`. Beyond
        the last row both are zero. Inside the first row the first rows'
        cubics continue; a caller refuses such radii before it traces, by
        `END_TOLERANCE` as `read` does.
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
    """Where two breaks of the velocity panels are kept, over projected radius.

    Decided once for a model, on the base rows of its table. Every reader of
    a row or of a table lays out its panels through them, so a row computed
    alone agrees with the same row of the table.

    Parameters
    ----------
    v_rad : array, shape (n_ranges, 2)
        Ranges of projected radius in which the break at the radial turning
        velocity is kept.
    v_k : array, shape (n_ranges, 2)
        Ranges in which the break at the kink velocity is kept.
    """

    v_rad: Array
    v_k: Array


@dataclass(frozen=True)
class LosvdRow:
    """The line-of-sight velocity distribution at one projected radius.

    Parameters
    ----------
    r_2d : float
        Projected radius, in internal units.
    layout : str
        Code of the sequence of panels of the row, the record of how the row
        was made. Two rows are read together only where their layouts agree.
        The code is a decimal integer naming the order of the row's interior
        breaks: ``"0"`` none; ``"1"`` to ``"15"`` the orders of V_k, V_rad
        and V_edge; ``"16"`` to ``"64"`` the orders that include V_c; and a
        code with V_k plus 100 where a kept V_k lies inside the grading toward
        zero velocity.
    v_breaks : dict of str to float
        The break velocities of the row by name (``"v_k"``, ``"v_rad"``,
        ``"v_edge"``, ``"v_c"``, ``"v_end"``), all five as the row computed
        them, NaN where the row has none; the layout says which are breaks.
    v_escape : float
        Escape velocity at `r_2d` along the line of sight.
    v_nodes, v_weights : array, shape (n,)
        Velocity nodes of the panels, for non-negative velocity (the
        distribution is even), and their quadrature weights.
    panel_edges : array
        Edges of the sub-panels; between its nodes a row is read by the
        polynomial of each sub-panel through its own nodes.
    losvd : array, shape (n_components, n)
        The distribution at the nodes, divided by its own integral under
        `v_weights`, so that a row integrates to exactly 1 with its own
        weights. The surface density that weights a row in an aperture is
        that of `ProjectedDispersion`, never the row's own integral; the
        ratio of the two is the row's norm check in `record`.
    record : dict
        The row's own account: the levels of its inner quadrature, whether
        it converged and by which test, its norm against the surface
        density, its second moment against the projected dispersion, the
        level of that check, and the fraction beyond the escape velocity.
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

    One table per component: components in one potential do not share their
    breaks. An aperture sums the components with their surface densities.

    Parameters
    ----------
    rows : tuple of LosvdRow
        The rows, in increasing projected radius.
    r_base : array
        Projected radii the layout rule placed, before any row was added.
    r_breaks : array
        Projected radii at which the layout of the rows changes; no row is
        read across one.
    stretch_ends : array of bool, shape (n_stretches, 2)
        Whether each end of each stretch between breaks is a true end (a
        break, the edge of a finite model, or the last radius of the table),
        which fixes the variable the rows are read in.
    kept : KeptRanges
        The kept ranges the rows were laid out with.
    order : int
        Order of the reading between rows at which the table was checked.
        A reader at another order reads an unchecked table.
    r_uncertified : array, shape (n_gaps, 2)
        Intervals of projected radius in which reading between rows is not
        checked. A request inside one is refused by name.
    why_uncertified : tuple of str
        The reason of each such interval, quoted by the refusal.
    certificate : dict
        How the rows ended: the reason, the rows added, the rounds and the
        tolerances.
    identity : dict
        The model the table belongs to (its digest, whether it is spatially
        finite, its edge and the last radius of the table).
    extrema : dict
        The extrema of the line of sight (`EllipseRoute.f_extrema`) the
        apertures read from this table computed, by the model's identity and
        the rules, then by projected radius: a second aperture on the same
        table reads them again instead of forming them. Each is the
        one its radius forms alone, whatever else was asked with it.
        Not part of the table's value.
    pools : dict
        The pools of processes the apertures read from this table keep open
        between calls (`table_pool`), by the model's identity, the rules and
        the number of cores: started at the first aperture that needs one,
        closed by `close_pools` or when the table goes. Not part of the
        table's value.
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

    def close_pools(self) -> None:
        """Close the pools of processes the table's apertures keep open."""
        while self.pools:
            _, pool = self.pools.popitem()
            pool.terminate()
            pool.join()


def table_pool(table: LosvdTable, route: Any, key: Any) -> Any:
    """Return the table's pool for `key`, starting it at the first call.

    The processes take a copy of the route without its extrema cache: a
    process forms the extrema of the radii it is handed, each the one its
    radius forms alone, and the cache stays with the caller. The pool is
    terminated with the table if `LosvdTable.close_pools` was not called.
    """
    if key not in table.pools:
        cache, route._fext = route._fext, {}
        try:
            pool = new_pool(get_workers(), route)
        finally:
            route._fext = cache
        table.pools[key] = pool
        weakref.finalize(table, pool.terminate)
    return table.pools[key]


@dataclass(frozen=True)
class ApertureLosvd:
    """The line-of-sight velocity distribution inside an aperture.

    Parameters
    ----------
    aperture : Aperture
        The aperture.
    v_nodes, v_weights : array, shape (n,)
        Velocity nodes of the panels of the aperture distribution and their
        quadrature weights, in internal units. Moments are taken with these
        weights.
    v_breaks : array
        Edges of the panels.
    losvd : array, shape (n_components, n)
        The distribution at the nodes, weighted by the surface density over
        the aperture and normalized to unit integral over velocity.
    mass : tuple of float
        Projected mass of each component inside the aperture.
    record : dict
        The norm by its own weights, the second moment against the rows and
        against the aperture dispersion, the estimate of the reading between
        rows over the aperture, how the refinement ended, and the status:
        the worse of its own and the table's.
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
    """The surface density and the squared projected dispersion at radii, of
    the first component, the references of a row's checks."""

    def read(r_2d: ArrayLike) -> tuple[Array, Array]:
        surface, sigma_los = projected.read(np.atleast_1d(r_2d))
        return surface[0], sigma_los[0] ** 2

    return read


def _tangent_parameters(model: Model, tangents: Sequence[str]) -> tuple[str, ...]:
    """The anisotropy parameters asked for, refused where the model's profile
    is not fixed or the family does not vary in one."""
    names = tuple(tangents)
    if not names:
        return names
    component = model.components[0]
    if not model.spatially_finite:
        raise InputError(
            f"The tangent rows of {component.name!r} are not taken: it is "
            "untruncated, and its radial dispersion's part below the table "
            "reads its DF, which moves with the anisotropy."
        )
    rule = model.form.below_floor[0]
    if rule != "edge_law":
        raise InputError(
            f"The tangent rows of {component.name!r} are not taken: it is not "
            f"a Method II model (its DF ends below its table by the rule "
            f"{rule!r}, not the edge law), and a truncated model's profile "
            "follows its own DF, so it moves with the anisotropy."
        )
    family = component.anisotropy.family
    for name in names:
        if name not in TANGENT_PARAMETERS[family]:
            raise InputError(
                f"A {family} model's tangent rows are taken in "
                f"{list(TANGENT_PARAMETERS[family])}; got {name!r}."
            )
    return names


def dispersion_profile(
    model: Model, settings: Settings, tangents: Sequence[str] = ()
) -> DispersionProfile:
    r"""Return the intrinsic second moments of a model.

    Parameters
    ----------
    model : Model
        The model.
    settings : Settings
        Numerical settings.
    tangents : sequence of {"beta_c", "r_a"}, optional
        Anisotropy parameters to take the tangent rows in: the derivative of
        :math:`\sigma_r^2` on the same rows, on the model's fixed profile
        (`DispersionProfile.tangents`). The family's own
        (`TANGENT_PARAMETERS`); none by default.

    Returns
    -------
    profile : DispersionProfile

    Raises
    ------
    InputError
        With `tangents`, where the model is not a Method II model on a
        truncated profile (an untruncated model, or a truncated one whose
        profile follows its own DF), or the family does not vary in a
        parameter named.

    Notes
    -----
    :math:`\varrho_{\rm aux}\sigma_r^2(R) = \int_0^{P(R)}\varrho_{\rm aux}\,{\rm d}K`
    on rows from the profile's first row to its last, placed
    until every interval reads within ``profile_tolerance`` at its midpoint
    (`radial_table`), the part below the table's lowest potential over the
    model's DF (zero on a spatially finite model);
    :math:`\sigma_t^2 = 2(1 - \beta_\sigma)\sigma_r^2`.

    The tangent rows are the derivative of the same rule on the same rows
    and nodes (`radial_dispersion_tangent`): the rows' placement is held, and
    the edge reading's limit does not move, since a Method II model's edge
    law moves with :math:`\beta_{\rm c}` as its class factor does. Without
    `tangents` the result is the same, bit for bit.

    References
    ----------
    Eq. (T:6.1) [jeans]. Chiang+(2026d), Eq. (B42) [eqn:linearity_2].
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
    """Return the surface density and the line-of-sight dispersion.

    Parameters
    ----------
    model : Model
        The model.
    profile : DispersionProfile
        Its intrinsic second moments.
    settings : Settings
        Numerical settings.
    tangents : sequence of {"beta_c", "r_a"}, optional
        Anisotropy parameters to take the tangent rows in: the derivative of
        the squared line-of-sight dispersion on the same rows
        (`ProjectedDispersion.tangents`), from the profile's tangent rows in
        the same parameters. None by default.

    Returns
    -------
    projected : ProjectedDispersion

    Raises
    ------
    InputError
        With `tangents`, as `dispersion_profile`, or where the profile holds
        no tangent rows in a parameter named.

    Notes
    -----
    Both from one line-of-sight rule with one density (`project`), on
    ``n_radii`` projected radii to 0.9999 of the last row, with rows added
    until the table reads itself between rows to `CERTIFICATE_TOLERANCE`.

    The tangent rows are the derivative of the same rule on the same rows
    and nodes (`project_tangent`), through the derivative of the profile's
    reading (`radial_integral_tangent_reader`); zero where the squared
    dispersion is clipped to zero. Without `tangents` the result is the
    same, bit for bit.

    References
    ----------
    Chiang+(2026d), Eqs. (B45), (B46) [eqn:los_sigma_1].
    """
    names = _tangent_parameters(model, tangents)
    missing = [name for name in names if name not in profile.tangents]
    if missing:
        raise InputError(
            f"The profile holds no tangent rows in {missing}: compute it with "
            "dispersion_profile(..., tangents=...) in the same parameters."
        )
    with timed("projected_dispersion"):
        table = projected_table(
            model, profile._read_sigma_r2(0), settings, CERTIFICATE_TOLERANCE
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
            tangent_rows[name] = np.where(table["sigma_los2"] > 0.0, rows, 0.0)[
                np.newaxis, :
            ]
        return ProjectedDispersion(
            r_2d=table["r_2d"],
            surface_density=table["surface_density"][np.newaxis, :],
            sigma_los=np.sqrt(np.maximum(table["sigma_los2"], 0.0))[np.newaxis, :],
            certificate=record,
            model=model,
            tangents=tangent_rows,
        )


def aperture_dispersion(
    projected: ProjectedDispersion, aperture: Aperture, settings: Settings
) -> Array:
    """Return the line-of-sight dispersion averaged over an aperture.

    The average of the squared dispersion weighted by the surface density.
    It reads no velocity distribution, and is the independent reference of
    `aperture_losvd`.

    Parameters
    ----------
    projected : ProjectedDispersion
        Surface density and dispersion over projected radius.
    aperture : Aperture
        The aperture.
    settings : Settings
        Numerical settings.

    Returns
    -------
    sigma_los : array, shape (n_components,)

    Raises
    ------
    InputError
        Where the aperture encloses no tabulated surface density, or where the
        disk inside the table's first row carries more than
        `INNER_DISK_TOLERANCE` of its weight.

    Notes
    -----
    The integral over the aperture is one-dimensional in projected radius,
    weighted by the angular measure of each circle inside the aperture
    (`Aperture.arc`), on panels between the radii where that measure is not
    smooth (`Aperture.breaks`); each panel has ``n_aperture_nodes``
    Gauss-Legendre nodes, or ``aperture_nodes_per_decade`` a decade of its
    span where that is more. The table is read as `ProjectedDispersion.read`
    reads it.

    References
    ----------
    Chiang+(2026d) [eqn:los_sigma_rect] (a rectangle), [eqn:los_sigma_2] (a centered
    circle) and [eqn:los_sigma_3] (an offset circle).
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
    r"""Return where the breaks of the velocity panels are kept.

    A break is kept where each of the two panels next to it carries more
    than a bound of the row's content.

    Parameters
    ----------
    model : Model
        The model.
    projected : ProjectedDispersion
        Its surface density and projected dispersion.
    settings : Settings
        Numerical settings.

    Returns
    -------
    kept : KeptRanges

    Notes
    -----
    The bound of V_rad is :math:`2D(V_{\rm rad})(V_{\rm end} - V_{\rm rad})`
    over the surface density, or :math:`V_{\rm end}^2` times that over the
    surface density times :math:`\sigma_{\rm los}^2` where larger, with
    :math:`D(V_{\rm rad})` from one row at that velocity; that of V_k the
    smaller of the same over its two panels. Each is taken on the base rows
    and bisected in ln R where it crosses `VRAD_KEEP`.
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
                    route, route_df, base, _surface_and_dispersion(projected), pool.map
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

    Parameters
    ----------
    model : Model
        The model.
    r_2d : float
        Projected radius, in internal units.
    kept : KeptRanges
        The kept ranges of the model.
    projected : ProjectedDispersion
        Surface density and projected dispersion, which the row is checked
        against.
    settings : Settings
        Numerical settings.

    Returns
    -------
    row : LosvdRow

    Raises
    ------
    ResultError
        Where the radius's own scan puts a break on or past its V_end.
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
    """Return the table of rows that checks its own reading between rows.

    Rows are added until the distribution read between two neighboring
    rows agrees with a row computed there, to the tolerance of the settings.
    Intervals that cannot be closed are recorded with their reason, not
    hidden.

    Parameters
    ----------
    model : Model
        The model.
    kept : KeptRanges
        The kept ranges of the model.
    projected : ProjectedDispersion
        Surface density and projected dispersion.
    settings : Settings
        Numerical settings.
    Returns
    -------
    table : LosvdTable

    Notes
    -----
    The base rows are ``rows_per_decade`` a decade from the profile's first row
    to 0.9999 of where the rows end: its last row, or the radius where the
    potential reaches the DF's floor where the DF is zero below it
    (`floor_radius`), read before any row is built; rows are added toward
    each R-break, toward that end and where a stretch is short
    (`rows_with_r_breaks`). The rows kept run from the last at or inside
    `row_start` (``innermost_read_radius``) to the first at or beyond
    `row_stop` (``outer_mass_share``), each the full grid's own; a table cut
    inside its edge is read to its last row only, the gap beyond unread. Each row
    is computed in a pool of `get_workers` processes started by spawn (each
    process imports the package once), and checked against the surface
    density and the projected dispersion of `projected`.
    """
    rules = ellipse_rules(settings)
    with timed("losvd_table"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        route_df = ellipse_route(model, rules)
        base = base_rows(model, settings)
        r_full, r_breaks = route.rows_with_r_breaks(base)
        # the rows the inputs read: each the full grid's own
        r_rows = cut_rows(
            r_full,
            row_start(model, projected, float(settings.losvd["innermost_read_radius"])),
            row_stop(model, projected, float(settings.losvd["outer_mass_share"])),
        )
        base = base[(base >= r_rows[0]) & (base <= r_rows[-1])]
        read = _surface_and_dispersion(projected)
        surface, sigma_los2 = read(r_rows)
        r_edge = float(route.R_t)
        # a table cut inside the edge is read to its last row only, one cut
        # inside from its first; its R-breaks are the full table's
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
) -> ApertureLosvd:
    """Return the line-of-sight velocity distribution inside an aperture.

    The rows of the table are read at the radii of the aperture, with the
    breaks at each radius computed from the model and the weights from the
    surface density. No row is computed here.

    Parameters
    ----------
    model : Model
        The model.
    table : LosvdTable
        Its table of rows, covering the aperture.
    projected : ProjectedDispersion
        Surface density and projected dispersion.
    aperture : Aperture
        The aperture.
    settings : Settings
        Numerical settings.

    Returns
    -------
    result : ApertureLosvd

    Raises
    ------
    InputError
        Where the aperture encloses no tabulated surface density, or the disk
        inside the table's first row carries more than `INNER_DISK_TOLERANCE`
        of its weight.
    """
    rules = ellipse_rules(settings)
    observables = settings.observables
    n_nodes = int(observables["n_aperture_nodes"])
    per_decade = int(observables["aperture_nodes_per_decade"])
    with timed("aperture_losvd"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
        # the extrema the table's apertures computed before, kept on it
        route._fext = table.extrema.setdefault((model.identity, rules), {})
        stencil = table_stencil(route, table)
        # on a table cut inside, the integral starts at its first row
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
                f"The table's rows start at {r_first:.4e}, which "
                f"losvd.innermost_read_radius sets; the smallest centered circle "
                f"it serves at the goal has radius "
                f"{served_radius(model, projected, r_first):.4e}."
            )
        n_workers = get_workers()

        def pool() -> Any:
            # the table's pool, started where a radius first needs it and
            # kept for the table's next apertures
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
    v, weights : array, shape (n,)
        Velocity nodes over the whole distribution, both signs, and their
        quadrature weights. A row or an aperture distribution holds
        ``V >= 0`` only: mirror it first.
    losvd : array, shape (n,)
        The distribution at `v`; it is normalized here to unit integral.

    Returns
    -------
    moments : dict
        ``mean`` and ``sigma``, the distribution's own mean and dispersion,
        which are the Gaussian's :math:`V_*` and :math:`\sigma_*`; ``h3`` and
        ``h4``; and the standardized moments ``s3`` and ``s4`` by direct sums.

    Notes
    -----
    With :math:`y = (V - V_*)/\sigma_*`,
    :math:`h_n = 2\sqrt{\pi}\int D(V)\,H_n(y)\,e^{-y^2/2}/\sqrt{2\pi}
    \,{\rm d}V`, the Hermite functions of van der Marel and Franx (1993):
    :math:`H_3 = (2y^3 - 3y)/\sqrt{3}`,
    :math:`H_4 = (4y^4 - 12y^2 + 3)/(2\sqrt{6})`. To first order in
    :math:`h_3, h_4`, :math:`s_3 = 4\sqrt{3}\,h_3` and
    :math:`s_4 = 3 + 8\sqrt{6}\,h_4`.

    References
    ----------
    Chiang+(2026d) [eqn:B_hn_def], [eqn:Hermit_polys].
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
    """The Gauss-Hermite moments of a table's rows, read at any projected radius.

    Parameters
    ----------
    r_2d : array, shape (n,)
        Projected radius of the rows that carry a distribution, increasing.
    moments : dict of str to array, shape (n,)
        ``mean``, ``sigma``, ``h3`` and ``h4`` of each row (`gauss_hermite_moments`).

    Notes
    -----
    Between rows each moment is read as the table's squared dispersion is
    (`dispersion_table_reader`): a not-a-knot cubic in ln R, zero beyond the
    last row; a radius inside the first row is refused by `read`, and a
    caller refuses it before it traces `read_traced`. At a row's own radius
    both readings return the row's moments to the bit.
    """

    r_2d: Array
    moments: dict[str, Array]

    def _splines(self) -> dict[str, CubicSpline]:
        x = np.log(self.r_2d)
        return {k: CubicSpline(x, self.moments[k]) for k in GAUSS_HERMITE_NAMES}

    def read(self, r_2d: ArrayLike) -> dict[str, Array]:
        """Return the four moments at radii."""
        r = np.atleast_1d(np.asarray(r_2d, dtype=np.float64))
        _refuse_below(r, float(self.r_2d[0]), "Gauss-Hermite moments")
        return {
            k: np.asarray(self.read_traced(r, np)[k], dtype=np.float64)
            for k in GAUSS_HERMITE_NAMES
        }

    def read_traced(self, r_2d: Any, xp: Any = np) -> dict[str, Any]:
        """Return the four moments at radii, `r_2d` traceable under
        ``jax.numpy``: the cubics' own coefficients evaluated in `xp`."""
        log_r = xp.log(xp.maximum(r_2d, TINY))
        last = float(self.r_2d[-1])
        out = {}
        for k, spline in self._splines().items():
            value = spline_value(spline.x, spline.c, log_r, xp)
            # the last row's own value, which its interval's cubic gives only
            # to rounding
            value = xp.where(r_2d == last, float(self.moments[k][-1]), value)
            out[k] = xp.where(r_2d > last * (1.0 + END_TOLERANCE), 0.0, value)
        return out


def gauss_hermite_table(table: LosvdTable) -> GaussHermiteTable:
    """Return the Gauss-Hermite moments of a table's rows.

    Each row (``V >= 0``) is mirrored to both signs and its moments taken by
    `gauss_hermite_moments`; a row without a distribution (its integral not
    positive) carries none and is left out.
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
    r"""Return the report of a table: its essential checks.

    The norm of every row against the surface density, its second moment
    against the projected dispersion, the rows that could not be judged, the
    rows whose rule along the line of sight did not converge, the record of
    the check between rows, and the floor of the layers toward the L = 0 point.

    Notes
    -----
    ``norm`` and ``second_moment`` are the worst over the rows judged, at
    their radius. A row is judged only where both checks are; one that is not
    while the surface density there is positive (a row integrating to zero, or
    a projected dispersion not positive) is ``unjudged``, beyond the hard
    limit; a row without surface density has nothing to judge.
    ``t_converged`` is beyond the hard limit where any row's rule along the
    line of sight ran out of doublings before its tests passed (the row's
    ``t_converged``), at the first such row. The
    ``certificate`` is the table's last estimate of its reading between rows,
    in units of its tolerances, times the larger tolerance (`ROW_TOL_PW`): a
    bound on both the pointwise and the second-moment estimate. The
    ``layer_floor`` is the relative error at which the layers toward the
    L = 0 point stop converging, where the anisotropy's endpoint power is
    negative (:math:`\beta_{\rm c}` between 1/2 and 1): every graded layer
    then carries the rule's own error, and the count is the least within a
    hundredth of the goal of the floor (`k_end`, which refuses a floor above
    the goal); zero where the layers converge.
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
    r"""Return the report of an aperture distribution: its essential checks.

    Its norm by its own weights and its second moment against the aperture
    dispersion `sigma_los`, with the status of the table it was read from.

    Notes
    -----
    ``norm`` is the integral of the distribution by its own weights less one,
    ``second_moment`` its second moment over that integral against
    :math:`\sigma_{\rm los}^2` from `aperture_dispersion`, which reads no
    distribution. The table's tier (`validate_table`, held in the record as
    ``table_level``) is the report's parent.
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


# Doublings of the speed distribution's angle rule, at most, before its change
# is reported as it stands.
SPEED_DOUBLINGS = 3
# Speeds within this fraction of V* are left out of the halving check: on a DF
# that diverges at its floor the distribution itself is infinite at V*.
SPEED_KINK_GAP = 1e-9


@dataclass(frozen=True)
class SpeedDistribution:
    r"""The local speed distribution of a model at one 3D radius.

    Parameters
    ----------
    r : float
        Radius, in internal units.
    v_nodes, v_weights : array, shape (n,)
        The speed rule the record integrates on: Gauss-Legendre panels
        graded toward each break of `panel_edges`.
    panel_edges : array
        The breaks of the distribution in speed: 0, :math:`V_*` where the
        support starts to cut the directions, :math:`V_{\rm esc}` where it
        lies inside, and :math:`V_{\rm max}`.
    pdf : array, shape (n,)
        :math:`p(V\,|\,r)` at `v_nodes`, per unit internal velocity.
    v_esc : float
        Escape speed :math:`\sqrt{2\mathcal{P}(r)}`.
    v_max : float
        Where the DF's support ends: :math:`V_{\rm esc}` of an isotropic,
        constant or Type I model whose DF reaches zero energy; for Type II,
        :math:`\sqrt{2(\mathcal{P} - K_c)/(1 - r^2/r_a^2)}`.
    unbound_fraction : float
        The share of the distribution beyond :math:`V_{\rm esc}` (orbits of
        negative energy that a Type II DF holds).
    record : dict
        ``norm`` (the integral less one, against the profile's density),
        ``second_moment`` (against :math:`\sigma_r^2 (3 - 2\beta_\sigma)` of
        `dispersion_profile`), ``halving`` (the largest change of the
        distribution on the rule's nodes when the angle rule's nodes
        double, over its largest value), ``n_angle_nodes`` (the rule
        checked) and ``tier``.
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
        """Return :math:`p(V\\,|\\,r)` at any speeds, by the rule the record
        checked; zero beyond `v_max` and at negative speeds."""
        speeds = np.asarray(v, dtype=np.float64)
        sm = speed_model(self.model)
        psi, xi, scale = _speed_point(self.model, sm, self.r)
        inside = (speeds >= 0.0) & (speeds <= self.v_max)
        out = np.zeros(speeds.shape)
        u = speeds[inside]
        n_nodes = int(self.record["n_angle_nodes"])
        n_panels = int(self.record["n_angle_panels"])
        out[inside] = (
            scale
            * u ** (2.0 - 2.0 * sm.b)
            * angle_integral(sm, psi, xi, u, n_nodes, n_panels)
        )
        return out


def _speed_point(model: Model, sm: SpeedModel, r: float) -> tuple[float, float, float]:
    """The potential, :math:`(r/r_a)^2` and the factor
    :math:`4\\pi r^{-2b}/\\varrho` at a radius."""
    rr = np.array([r])
    psi = float(np.asarray(model.profile.potential(rr))[0])
    rho = float(np.atleast_2d(np.asarray(model.profile.density(rr)))[0, 0])
    xi = 0.0 if sm.sign == 0.0 else (r / sm.r_a) ** 2
    return psi, xi, 4.0 * np.pi * r ** (-2.0 * sm.b) / rho


def speed_distribution(
    model: Model,
    r: float,
    settings: Settings,
    profile: DispersionProfile | None = None,
) -> SpeedDistribution:
    r"""Return the local speed distribution of a model at a 3D radius.

    Parameters
    ----------
    model : Model
        A model of one component.
    r : float
        Radius, in internal units, inside the profile and, for Type II,
        inside the anisotropy radius.
    settings : Settings
        Numerical settings (the ``observables`` group's
        ``n_speed_angle_nodes``, ``n_speed_angle_panels`` and
        ``speed_tolerance``).
    profile : DispersionProfile, optional
        The model's intrinsic second moments, which the second moment is
        checked against; computed when not given.

    Returns
    -------
    distribution : SpeedDistribution

    Raises
    ------
    InputError
        For a model of more than one component, or a radius outside the
        profile or, for Type II, at or beyond the anisotropy radius.

    Notes
    -----
    :math:`p(V\,|\,r) = (4\pi/\varrho)\, r^{-2b} V^{2-2b} \int_0^1
    (1-t^2)^{-b} f(Q)\,{\rm d}t` over the DF's own support (`_speed`), with
    :math:`t = \cos\eta`, the angle between the velocity and the radius,
    the variable of the sampler's angle law. The DF module states the
    support (`AsymptoticForm.below_floor`): F is read through its lookup,
    zero below its first energy where its rule is ``"zero"``. A Type II DF
    is a function of :math:`Q_+ = Z + L^2/(2r_a^2)`, so its speeds reach
    :math:`V_{\rm max} = \sqrt{2(\mathcal{P} - K_c)/(1 - r^2/r_a^2)}`,
    beyond the escape speed where its floor :math:`K_c` is below
    :math:`\xi\mathcal{P}(r)`, :math:`\xi = r^2/r_a^2`, and
    `unbound_fraction` is the share there. A Type II model is truncated at
    :math:`K_t \geq \mathcal{P}(r_a)` (`truncate`'s domain gate), so its
    floor in the truncated potential is zero and its unbound share is not.
    A floor at :math:`\mathcal{P}(r_a)` itself, in the potential that
    vanishes at infinity, would leave no orbit unbound:
    :math:`Z \geq (\mathcal{P}(r_a) - \xi\mathcal{P}(r))/(1 - \xi) > 0`, since
    :math:`r^2\mathcal{P}(r)` increases with :math:`r` (its derivative
    :math:`2r\mathcal{P} - M(r) \geq M(r) > 0`, as
    :math:`\mathcal{P} \geq M(r)/r` there).

    The angle rule's nodes double, at most `SPEED_DOUBLINGS` times, until
    the distribution on the speed rule changes by less than
    ``speed_tolerance`` of its largest value. On an untruncated Type I DF
    that diverges at its floor (a tail :math:`F \propto K^{-1/2}` with a
    logarithmic factor) the distribution is infinite at exactly
    :math:`V_*`, an integrable singularity; speeds within
    `SPEED_KINK_GAP` of it are left out of that check.

    References
    ----------
    Chiang+(2026d) [eqn:rho_f_anisotropic_scale], [eqn:varphi_prob], [eqn:V_prob].
    """
    if len(model.components) != 1:
        raise InputError(
            f"The speed distribution is computed for a model of one component; "
            f"this one has {len(model.components)}."
        )
    observables = settings.observables
    n_nodes = int(observables["n_speed_angle_nodes"])
    n_panels = int(observables["n_speed_angle_panels"])
    tolerance = float(observables["speed_tolerance"])
    with timed("speed_distribution"):
        sm = speed_model(model)
        r = float(r)
        r_lo, r_hi = float(model.profile.r[0]), float(model.profile.r_table_max)
        if sm.sign < 0.0:
            r_hi = min(r_hi, sm.r_a)
        if not r_lo <= r < r_hi:
            raise InputError(
                f"The speed distribution is read at a radius in [{r_lo:.6e}, "
                f"{r_hi:.6e}) (the profile's rows, and a Type II model's "
                f"anisotropy radius); got {r!r}."
            )
        psi, xi, scale = _speed_point(model, sm, r)
        v_esc, v_kink, v_max = speed_breaks(sm, psi, xi)
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
        away = np.abs(v / v_kink - 1.0) > SPEED_KINK_GAP

        def pdf_at(n: int) -> Array:
            return factor * angle_integral(sm, psi, xi, v, n, n_panels)

        # the rule kept is the coarser of the last pair, its change the pair's
        pdf = pdf_at(n_nodes)
        for doubling in range(SPEED_DOUBLINGS + 1):
            finer = pdf_at(2 * n_nodes)
            peak = float(np.max(np.abs(finer[away])))
            halving = float(np.max(np.abs(finer - pdf)[away])) / peak
            if halving <= tolerance or doubling == SPEED_DOUBLINGS:
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
        return SpeedDistribution(
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
