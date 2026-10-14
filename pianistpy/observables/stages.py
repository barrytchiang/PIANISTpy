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
    "aperture_dispersion",
    "aperture_losvd",
    "dispersion_profile",
    "gauss_hermite_moments",
    "gauss_hermite_table",
    "kept_ranges",
    "losvd_row",
    "losvd_table",
    "projected_dispersion",
    "validate_aperture",
    "validate_table",
]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from ..df import Model
from ..runtime import timed
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..validation import GOAL, Check, ValidationReport
from ._aperture import aperture_mean_square, aperture_nodes
from ._aperture_losvd import aperture_losvd_panels
from ._dispersion import beta_sigma, radial_integral_reader
from ._ellipse_model import ellipse_route
from ._ellipse_rules import ROW_TOL_PW
from ._kept import vk_kept_ranges, vrad_kept_ranges
from ._losvd_stages import (
    BREAK_NAMES,
    base_rows,
    ellipse_rules,
    panel_table_of,
    rows_of,
    start_pool,
)
from ._losvd_table import losvd_rows
from ._projection import (
    END_TOLERANCE,
    TINY,
    _refuse_below,
    dispersion_table_reader,
    spline_value,
    surface_density_reader,
)
from ._tables import projected_table, radial_table
from .aperture import Aperture

# The relative tolerance of a projected table's reading between rows: a hundredth
# of the accuracy goal, so that the reading adds nothing measurable.
CERTIFICATE_TOLERANCE = 0.01 * GOAL
# The largest estimated share of an aperture's weight inside the first row of
# the projected table: the accuracy goal, since that disc is not computed.
INNER_DISC_TOLERANCE = GOAL


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
        together: :math:`\sigma_{\rm t}^2 = \sigma_\theta^2 +
        \sigma_\phi^2`, so that :math:`\beta_\sigma = 1 -
        \sigma_{\rm t}^2 / (2 \sigma_r^2)`.
    beta_sigma : array, shape (n_components, n)
        Anisotropy profile :math:`\beta_\sigma(R)`.
    edge_limit : tuple of float
        For a spatially finite model, the limit of
        :math:`\sigma_r^2 / \Psi` at the edge of each component, which the
        reading between grid points uses there; NaN otherwise.
    certificate : dict
        How the rows were placed (`radial_table`): the tolerance, how they
        ended, the rounds, the rows, the first and last worst reading at an
        interval's midpoint and the intervals left uncertified.
    model : Model
        The model the profile belongs to: its potential is the variable of
        the reading between grid points, and its anisotropies give
        `beta_sigma` at any radius.

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

    def _read_sigma_r2(self, component: int = 0) -> Callable[[ArrayLike], Array]:
        r"""The reading of :math:`\sigma_r^2` of one component."""
        profile = self.model.profile
        anisotropy = self.model.components[component].anisotropy

        def rho_aux(r: Array) -> Array:
            return profile.density(r)[component] / anisotropy.factor(r)

        return radial_integral_reader(
            self.r,
            np.asarray(self.sigma_r)[component] ** 2,
            profile.potential,
            rho_aux,
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

    Notes
    -----
    Between rows the surface density is read through ``h = Sigma / (R_t^2 -
    R_2D^2)^(1/2)``, a not-a-knot cubic in ``ln(R / (1 - R / R_t))`` on a
    spatially finite model and in ln R otherwise, and the squared dispersion
    by a not-a-knot cubic in ln R: the readings the table certifies its rows
    for. Both are zero beyond the last row and refused inside the first.
    """

    r_2d: Array
    surface_density: Array
    sigma_los: Array
    certificate: dict[str, float | int | str]
    model: Model

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
        Order of the reading between rows at which the table was certified.
        A reader at another order reads an uncertified table.
    r_uncertified : array, shape (n_gaps, 2)
        Intervals of projected radius in which reading between rows is not
        certified. A request inside one is refused by name.
    why_uncertified : tuple of str
        The reason of each such interval, quoted by the refusal.
    certificate : dict
        How the rows ended: the reason, the rows added, the rounds and the
        tolerances.
    identity : dict
        The model the table belongs to (its digest, whether it is spatially
        finite, its edge and the last radius of the table).
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


def dispersion_profile(model: Model, settings: Settings) -> DispersionProfile:
    r"""Return the intrinsic second moments of a model.

    Parameters
    ----------
    model : Model
        The model.
    settings : Settings
        Numerical settings.

    Returns
    -------
    profile : DispersionProfile

    Notes
    -----
    :math:`\varrho_{\rm aux}\sigma_r^2(R) = \int_0^{P(R)}\varrho_{\rm aux}
    \,{\rm d}K` on rows from the profile's first row to its last, placed
    until every interval reads within ``profile_tolerance`` at its midpoint
    (`radial_table`), the part below the table's lowest potential over the
    model's DF (zero on a spatially finite model); :math:`\sigma_t^2 = 2(1 -
    \beta_\sigma)\sigma_r^2`.

    References
    ----------
    Notes Eq. (II.6) [jeans]. Paper Eq. (B42) [eqn:linearity_2].
    """
    with timed("dispersion_profile"):
        radial = radial_table(model, settings)
        r = radial["r"]
        sigma_r2 = radial["sigma_r2"][np.newaxis, :]
        beta = beta_sigma(r, model.components[0].anisotropy)[np.newaxis, :]
        return DispersionProfile(
            r=r,
            sigma_r=np.sqrt(sigma_r2),
            sigma_t=np.sqrt(2.0 * (1.0 - beta) * sigma_r2),
            beta_sigma=beta,
            edge_limit=(float(radial["edge_limit"]),),
            certificate=radial["certificate"],
            model=model,
        )


def projected_dispersion(
    model: Model, profile: DispersionProfile, settings: Settings
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

    Returns
    -------
    projected : ProjectedDispersion

    Notes
    -----
    Both from one line-of-sight rule with one density (`project`), on
    ``n_radii`` projected radii to 0.9999 of the last row, with rows added
    until the table reads itself between rows to `CERTIFICATE_TOLERANCE`.

    References
    ----------
    Paper Eqs. (B45), (B46) [eqn:los_sigma_1].
    """
    with timed("projected_dispersion"):
        table = projected_table(
            model, profile._read_sigma_r2(0), settings, CERTIFICATE_TOLERANCE
        )
        record = dict(table["certificate"])
        worst_sigma, worst_los = record.pop("worst_first")
        record["worst_first_surface_density"] = worst_sigma
        record["worst_first_sigma_los2"] = worst_los
        return ProjectedDispersion(
            r_2d=table["r_2d"],
            surface_density=table["surface_density"][np.newaxis, :],
            sigma_los=np.sqrt(np.maximum(table["sigma_los2"], 0.0))[np.newaxis, :],
            certificate=record,
            model=model,
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
        disc inside the table's first row carries more than
        `INNER_DISC_TOLERANCE` of its weight.

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
    Paper [eqn:los_sigma_rect] (a rectangle), [eqn:los_sigma_2] (a centerd
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
                INNER_DISC_TOLERANCE,
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
        with start_pool(
            route_df,
            key=(model.identity, rules),
            probe_r2d=float(np.median(base)),
        ) as pool:
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
    """Return the table of rows that certifies its own reading between rows.

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
    to 0.9999 of its last; rows are added toward each R-break, toward the
    table's end and where a stretch is short (`rows_with_r_breaks`). Each row
    is computed in a pool of `get_workers` processes started by spawn (each
    worker imports the package once), and checked against the surface
    density and the projected dispersion of `projected`.
    """
    rules = ellipse_rules(settings)
    with timed("losvd_table"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        route_df = ellipse_route(model, rules)
        base = base_rows(model, settings)
        r_rows, r_breaks = route.rows_with_r_breaks(base)
        read = _surface_and_dispersion(projected)
        surface, sigma_los2 = read(r_rows)
        r_edge = float(model.profile.r_table_max)
        with start_pool(
            route_df,
            key=(model.identity, rules),
            probe_r2d=float(np.median(r_rows)),
        ) as pool:
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
        Where the aperture encloses no tabulated surface density, or the disc
        inside the table's first row carries more than `INNER_DISC_TOLERANCE`
        of its weight.
    """
    rules = ellipse_rules(settings)
    observables = settings.observables
    n_nodes = int(observables["n_aperture_nodes"])
    per_decade = int(observables["aperture_nodes_per_decade"])
    with timed("aperture_losvd"):
        route = ellipse_route(model, rules, with_df=False)
        route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
        stencil = panel_table_of(
            route.lattice,
            np.array([row.r_2d for row in table.rows]),
            [
                (
                    row.v_nodes,
                    row.v_weights,
                    row.losvd[0],
                    row.v_escape,
                    row.v_breaks["v_k"],
                    row.v_breaks["v_end"],
                    row.v_breaks["v_rad"],
                    len(row.panel_edges) - 1,
                    int(row.layout),
                    row.v_breaks["v_edge"],
                    row.v_breaks["v_c"],
                )
                for row in table.rows
            ],
            table.r_breaks,
            float(table.identity["r_edge"]),
            table.order,
            table.r_uncertified,
            list(table.why_uncertified),
        )
        surface = surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        )
        v, w, d, breaks, info, _, _ = aperture_losvd_panels(
            surface,
            stencil,
            route,
            aperture,
            n_nodes,
            per_decade,
            INNER_DISC_TOLERANCE,
        )
        r, weights, sigma = aperture_nodes(
            surface, aperture, n_nodes, per_decade, INNER_DISC_TOLERANCE
        )
        centerd = aperture.shape == "circle" and aperture.r_off == 0.0
        mass = float(np.dot(weights, sigma)) * (2.0 * np.pi if centerd else 1.0)
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
        quadrature weights. A row or an aperture distribution holds V >= 0
        only: mirror it first.
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
    Paper [eqn:B_hn_def], [eqn:Hermit_polys].
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

    Each row (V >= 0) is mirrored to both signs and its moments taken by
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
    """Return the report of a table: its essential checks.

    The norm of every row against the surface density, its second moment
    against the projected dispersion, the rows that could not be judged, and
    the record of the certificate.

    Notes
    -----
    ``norm`` and ``second_moment`` are the worst over the rows judged, at
    their radius. A row is judged only where both checks are; one that is not
    while the surface density there is positive (a row integrating to zero, or
    a projected dispersion not positive) is ``unjudged``, beyond the hard
    limit; a row without surface density has nothing to judge. The
    ``certificate`` is the table's last estimate of its reading between rows,
    in units of its tolerances, times the larger tolerance (`ROW_TOL_PW`): a
    bound on both the pointwise and the second-moment estimate.
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
    estimate = float(table.certificate.get("est_last", float("nan")))
    return ValidationReport(
        keys=("norm", "second_moment", "unjudged", "certificate"),
        checks={
            "norm": worst(norm),
            "second_moment": worst(m2),
            "unjudged": unjudged,
            "certificate": Check(error=estimate * ROW_TOL_PW),
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
