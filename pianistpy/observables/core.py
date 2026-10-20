"""Observables of a model: dispersions, velocity distributions and apertures.

Each observable is one function. Given a model, it computes only what the
request needs, in memory. Given the prepared `LosTable` of a model, it
reads the tables that object holds. The two routes are held to the accuracy
goal against each other by the tests.

A superposed model (`superpose`) is projected one component at a time, each
on its own tables, and the components are summed with their weights: a
measurement is the ratio of the weighted sum of each component's
surface-density weighted second moment to that of its surface density.
"""

__all__ = [
    "LosTable",
    "losvd",
    "pdf_v",
    "sigma_los_aperture",
    "sigma_los",
    "sigma_r",
    "sigma_t",
]

import copy
import dataclasses
import logging
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import AccuracyError, InputError
from ..particles.core import _component_view, _is_superposition
from ..superposition import mixture
from ..typing import Array, ArrayLike
from ..validation import HARD_LIMIT, ValidationReport, announce, refuse_unless_asked
from ._aperture import aperture_nodes, outside_table
from ._ellipse_model import ellipse_route
from ._losvd_stages import ellipse_rules, panel_table_of, shared_pool
from ._panels import panels, read_panels
from ._projection import dispersion_table_reader, surface_density_reader
from .aperture import Aperture
from .stages import (
    INNER_DISK_TOLERANCE,
    DispersionProfile,
    KeptRanges,
    LosvdTable,
    ProjectedDispersion,
    aperture_dispersion,
    dispersion_profile,
    kept_ranges,
    losvd_table,
    projected_dispersion,
    speed_distribution,
    validate_aperture,
    validate_table,
)
from .stages import aperture_losvd as _aperture_losvd


class LosTable:
    """The line-of-sight velocity distributions of a model, built once.

    The object holds the table of line-of-sight velocity distributions over
    projected radius that the aperture quantities read. It is built when the
    object is created and reused by every observable computed from it.

    Parameters
    ----------
    model : Model
        The model to project: of one component, or a superposition
        (`superpose`).

    Attributes
    ----------
    model : Model
        The model, whose unit system the observables read from this object
        are given in.
    profile, projected, kept, table
        The results of the stages `dispersion_profile`,
        `projected_dispersion`, `kept_ranges` and `losvd_table`, at the
        model's settings; of the first component for a superposition.
    validation : ValidationReport
        The checks of the table (those of `validate_table`) and the worst
        tier of the model; for a superposition, each check at its worst over
        the components. A table above the goal says so in a warning.
    parts : tuple of LosTable
        One table per component of a superposition, each of the component
        as a model of one component; ``(self,)`` for a model of one
        component.
    """

    model: Model
    parts: tuple["LosTable", ...]
    profile: DispersionProfile
    projected: ProjectedDispersion
    kept: KeptRanges
    table: LosvdTable
    validation: ValidationReport

    def __init__(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise InputError("A LosTable is built from a Model.")
        views = _views(model)
        if len(views) > 1:
            self.model = model
            self.parts = tuple(LosTable(view) for view in views)
            first = self.parts[0]
            self.profile, self.projected = first.profile, first.projected
            self.kept, self.table = first.kept, first.table
            self.validation = _worst([part.validation for part in self.parts])
            return
        self.parts = (self,)
        settings = model.settings
        self.model = model
        self.profile = dispersion_profile(model, settings)
        self.projected = projected_dispersion(model, self.profile, settings)
        # the two stages read one route: its workers start once
        with shared_pool():
            self.kept = kept_ranges(model, self.projected, settings)
            self.table = losvd_table(model, self.kept, self.projected, settings)
        self.validation = _with_parent(
            validate_table(self.table, self.projected), model.validation.level
        )
        what = f"The LosTable of {model.components[0].name!r}"
        _announce(what, self.validation)
        # a table beyond the hard limit is refused unless the settings ask for it
        refuse_unless_asked(what, self.validation, settings)

    def to(self, system: str, **kwargs: Any) -> "LosTable":
        """Return the same tables presented in another unit system.

        Nothing is recomputed: the tables stay in internal units, and the
        observables read from the returned object are given in the units of
        ``self.model.to(system, **kwargs)`` (`Model.to`).
        """
        other = copy.copy(self)
        other.model = self.model.to(system, **kwargs)
        return other

    def close(self) -> None:
        """Close the parallel processes the table's aperture calls keep.

        The first aperture read from a table that needs them starts a pool
        of them, and the table's later apertures reuse it; the pool closes
        here, or when the table is gone. A table presented in another unit
        system (`to`) shares the pool. An aperture read after `close` starts
        a new one.
        """
        for part in self.parts:
            part.table.close_pools()


def _with_parent(report: ValidationReport, level: int) -> ValidationReport:
    """Return `report` with `level` among the worst tiers of its parents."""
    return ValidationReport(
        keys=report.keys,
        checks=report.checks,
        parents_level=max(report.parents_level, level),
    )


log = logging.getLogger("pianistpy")
# losvd and aperture_losvd build a Model's table at every call: said once
_TABLE_HINT = {"given": False}


def _table_of(model: Model, caller: str) -> LosTable:
    """Return the `LosTable` of a model `caller` was given, saying once in
    the process that the table is worth keeping for more calls."""
    if not _TABLE_HINT["given"]:
        _TABLE_HINT["given"] = True
        log.warning(
            "%s on a Model: building the LosTable of this model; keep it with "
            "pp.LosTable(model) for more calls.",
            caller,
        )
    return LosTable(model)


def _announce(what: str, report: ValidationReport) -> None:
    """Announce `report` above the goal (`validation.announce`), naming its
    checks there."""
    announce(
        what,
        report,
        ", ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at R = {check.where:.6e}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        or "the tier of its parents",
    )


def _views(model: Model) -> tuple[Model, ...]:
    """The models the observables are computed on: the model itself, or each
    component of a superposition as a model of one component."""
    if len(model.components) == 1:
        _refuse_unless_asked(model)
        return (model,)
    if _is_superposition(model):
        _refuse_unless_asked(model)
        return tuple(_component_view(model, k) for k in range(len(model.components)))
    raise InputError(
        f"The observables are computed for a model of one component or a "
        f"superposition (`superpose`); this one has {len(model.components)} "
        f"components in one potential."
    )


def _refuse_unless_asked(model: Model) -> None:
    # a model beyond the hard limit is an input only under the request that
    # made it (Settings(write_anyway=True)); the one statement of validation;
    # a superposition's report holds each check at its worst over its members
    refuse_unless_asked(
        f"The model {model.components[0].name!r}", model.validation, model.settings
    )


def _worst(reports: list[ValidationReport]) -> ValidationReport:
    """Each check at its worst over the components' reports, and the worst
    tier of their parents."""
    keys = reports[0].keys
    checks = {}
    for key in keys:
        present = [r.checks[key] for r in reports if key in r.checks]
        if present:
            checks[key] = max(present, key=lambda check: check.error)
    return ValidationReport(
        keys=keys,
        checks=checks,
        parents_level=max(r.parents_level for r in reports),
    )


def _weights(model: Model) -> Array:
    """The components' weights, as `superpose` set them (normalized by
    `mixture`)."""
    return np.array([float(c.mass) for c in model.components])


def _sums(projected: ProjectedDispersion, r_2d: Array) -> tuple[Array, Array]:
    """A component's surface density times its squared dispersion, and its
    surface density, at radii, read as `ProjectedDispersion.read` reads
    them."""
    model = projected.model
    surface = surface_density_reader(
        projected.r_2d,
        projected.surface_density[0],
        float(model.profile.r_table_max),
        model.spatially_finite,
    )(r_2d)
    sigma2 = dispersion_table_reader(
        projected.r_2d, np.asarray(projected.sigma_los[0]) ** 2
    )(r_2d)
    return surface * sigma2, surface


def _projected(view: Model) -> ProjectedDispersion:
    settings = view.settings
    return projected_dispersion(view, dispersion_profile(view, settings), settings)


def _length_factor(model: Model) -> float:
    """The model's unit of length in internal units: one in the internal and
    the model systems, the scale radius in parsecs in the physical system."""
    units = model.units
    if units.system != "physical":
        return 1.0
    factor = units.factors(model.components[0].unit_profile).physical_r_pc
    if factor is None:
        raise InputError("The physical system of this model states no length unit.")
    return float(factor)


def _internal_length(model: Model, value: Any) -> Array:
    """A length in the model's unit system (a float in parsecs, or an Astropy
    quantity, in the physical system) in internal units."""
    if hasattr(value, "unit"):
        value = value.to_value("pc")
    return np.asarray(value, dtype=np.float64) / _length_factor(model)


def _internal_aperture(model: Model, aperture: Aperture) -> Aperture:
    """An aperture whose lengths are in the model's unit system, in internal
    units."""
    factor = _length_factor(model)
    if factor == 1.0:
        return aperture

    def scaled(x: float | None) -> float | None:
        return None if x is None else float(x) / factor

    return dataclasses.replace(
        aperture,
        r_off=float(aperture.r_off) / factor,
        r_ape=scaled(aperture.r_ape),
        side_x=scaled(aperture.side_x),
        side_y=scaled(aperture.side_y),
    )


def _velocity(model: Model, value: ArrayLike) -> Any:
    """A velocity in internal units, in the model's unit system."""
    return model.units.convert("velocity", value, model.components[0].unit_profile)


def sigma_los(source: Model | LosTable, r_2d: ArrayLike) -> Array:
    r"""Return the line-of-sight velocity dispersion at projected radii.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`.
    r_2d : array-like
        Projected radius :math:`R_{\rm 2D}` in the model's unit system
        (`Model.to`): internal units by default, parsecs (a float or an
        Astropy quantity) in the physical system.

    Returns
    -------
    sigma_los : array
        Line-of-sight velocity dispersion in the model's unit system, with
        the shape of `r_2d`.

    Notes
    -----
    Of a superposition, :math:`\sigma_{\rm los}^2 = \sum_k w_k \Sigma_k
    \sigma_{{\rm los},k}^2 / \sum_k w_k \Sigma_k`, each component read from
    its own table; zero beyond the last row.
    """
    if isinstance(source, LosTable):
        model = source.model
        projected = [part.projected for part in source.parts]
    elif isinstance(source, Model):
        model = source
        projected = [_projected(view) for view in _views(source)]
    else:
        raise InputError("sigma_los reads a Model or a LosTable.")
    x = _internal_length(model, r_2d)
    if len(projected) == 1:
        value = projected[0].read(x.ravel())[1][0].reshape(x.shape)
        return _velocity(model, value)
    sums = [_sums(p, x.ravel()) for p in projected]
    weights = _weights(model)
    moment = mixture(weights, [a for a, _ in sums])
    surface = mixture(weights, [b for _, b in sums])
    with np.errstate(invalid="ignore", divide="ignore"):
        square = np.where(surface > 0.0, moment / surface, 0.0)
    return _velocity(model, np.sqrt(square).reshape(x.shape))


def sigma_los_aperture(source: Model | LosTable, aperture: Aperture) -> float:
    r"""Return the velocity dispersion inside an aperture.

    Given a model, only the projected radii the aperture covers are computed.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`.
    aperture : Aperture
        The aperture, built with `Aperture.circle` or `Aperture.rectangle`,
        its lengths in the model's unit system (parsecs in the physical
        system).

    Returns
    -------
    sigma_ape : float
        Velocity dispersion inside the aperture, in the model's unit system.

    Notes
    -----
    Given a model, the second moment of the line-of-sight dispersion averaged
    over the aperture with the surface density (`aperture_dispersion`), which
    reads no velocity distribution. Given a `LosTable`, the second moment of
    the aperture's velocity distribution (`aperture_losvd`) by its own
    weights. The two agree to the accuracy goal (`validate_aperture`); a
    distribution above the goal, or read from a table above it, says so in a
    warning.

    Of a superposition, the components are summed with their weights under
    the aperture: given a model, the surface-density weighted sums of
    :math:`\sigma_{\rm los}^2` and of the surface density, each component's
    own, on the aperture's nodes of the first component (the components
    share one density), mixed as for `sigma_los`; given a `LosTable`, the
    second moment of the aperture distributions summed with the weights
    times each component's projected mass inside the aperture, each
    distribution checked against its own component's aperture dispersion.
    """
    if isinstance(source, LosTable):
        model = source.model
        inside = _internal_aperture(model, aperture)
        moments, masses = [], []
        for part in source.parts:
            settings = part.model.settings
            result = _aperture_losvd(
                part.model, part.table, part.projected, inside, settings
            )
            moments.append(
                float(result.record["m2_file"]) / float(result.record["norm_file"])
            )
            masses.append(result.mass[0])
            reference = aperture_dispersion(part.projected, inside, settings)
            what = f"The aperture distribution of {part.model.components[0].name!r}"
            report = _with_parent(
                validate_aperture(result, reference), part.validation.level
            )
            _announce(what, report)
            refuse_unless_asked(what, report, settings)
        if len(source.parts) == 1:
            value = np.sqrt(moments[0])
        else:
            weights = _weights(model)
            value = np.sqrt(
                float(mixture(weights, np.multiply(masses, moments)))
                / float(mixture(weights, masses))
            )
    elif isinstance(source, Model):
        model, settings = source, source.settings
        inside = _internal_aperture(model, aperture)
        projected = [_projected(view) for view in _views(source)]
        if len(projected) == 1:
            value = float(aperture_dispersion(projected[0], inside, settings)[0])
        else:
            value = _mixed_aperture(model, projected, inside)
    else:
        raise InputError("sigma_los_aperture reads a Model or a LosTable.")
    converted = _velocity(model, value)
    return converted if hasattr(converted, "unit") else float(converted)


def _velocity_factor(model: Model) -> float:
    """The model's unit of velocity in internal units: one in the internal
    system, the factor of `Model.to` otherwise."""
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return float(getattr(one, "value", one))


def _internal_velocity(model: Model, value: Any) -> Array:
    """A velocity in the model's unit system (a float in km/s, or an Astropy
    quantity, in the physical system) in internal units."""
    if hasattr(value, "unit"):
        value = value.to_value("km / s")
    return np.asarray(value, dtype=np.float64) / _velocity_factor(model)


def _profiles(
    source: Model | LosTable, what: str
) -> tuple[Model, list[Model], list[DispersionProfile]]:
    """The model, its views and the intrinsic second moments of each view."""
    if isinstance(source, LosTable):
        parts = source.parts
        return source.model, [p.model for p in parts], [p.profile for p in parts]
    if isinstance(source, Model):
        views = list(_views(source))
        return source, views, [dispersion_profile(v, v.settings) for v in views]
    raise InputError(f"{what} reads a Model or a LosTable.")


def _intrinsic(source: Model | LosTable, r: ArrayLike, which: int, what: str) -> Any:
    """`sigma_r` (which 0) or `sigma_t` (which 1) at radii, in the model's
    unit system. A superposition's second moment at a radius is the mixture
    of its components' weighted by their density there (the local weight of
    an intrinsic quantity, as the surface density weights a projected one)."""
    model, views, profiles = _profiles(source, what)
    x = _internal_length(model, r)
    flat = x.ravel()
    reads = [np.asarray(profile.read(flat)[which][0]) for profile in profiles]
    if len(reads) == 1:
        value = reads[0]
    else:
        weights = _weights(model)
        densities = [np.asarray(view.profile.density(flat)[0]) for view in views]
        moment = mixture(
            weights, [d * s**2 for d, s in zip(densities, reads, strict=True)]
        )
        density = mixture(weights, densities)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.sqrt(np.where(density > 0.0, moment / density, 0.0))
    return _velocity(model, value.reshape(x.shape))


def sigma_r(source: Model | LosTable, r: ArrayLike) -> Array:
    r"""Return the radial velocity dispersion at radii.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`, whose intrinsic second
        moments are read.
    r : array-like
        Radius :math:`r` in the model's unit system (`Model.to`): internal
        units by default, parsecs (a float or an Astropy quantity) in the
        physical system.

    Returns
    -------
    sigma_r : array
        Radial velocity dispersion :math:`\sigma_r` in the model's unit
        system, with the shape of `r`.

    Notes
    -----
    The reading between the grid points of the intrinsic profile is that of
    `DispersionProfile.read`. Of a superposition, whose components share one
    density, :math:`\sigma_r^2 = \sum_k w_k \sigma_{r,k}^2`.
    """
    return _intrinsic(source, r, 0, "sigma_r")


def sigma_t(source: Model | LosTable, r: ArrayLike) -> Array:
    r"""Return the tangential velocity dispersion at radii.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`.
    r : array-like
        Radius :math:`r` in the model's unit system, as for `sigma_r`.

    Returns
    -------
    sigma_t : array
        Tangential velocity dispersion of both tangential components
        together, :math:`\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2`
        :math:`= 2 (1 - \beta_\sigma) \sigma_r^2`, in the model's unit system,
        with the shape of `r`.

    Notes
    -----
    Read as `sigma_r` is; of a superposition, :math:`\sigma_{\rm t}^2 =
    \sum_k w_k \sigma_{{\rm t},k}^2`.
    """
    return _intrinsic(source, r, 1, "sigma_t")


def _read_losvd(part: LosTable, r_2d: Array, v: Array) -> Array:
    """The distribution of one component at the pairs ``(r_2d[j], v[j])``, read
    through the table's own stencil between rows, as an aperture reads it;
    the density over all velocity, of unit integral."""
    route, stencil = _stencil_of(part)
    return np.asarray(stencil.at_pairs(route, r_2d, v), dtype=np.float64)


def _stencil_of(part: LosTable) -> tuple[Any, Any]:
    """The route of one component and its table's stencil between rows."""
    model, table = part.model, part.table
    rules = ellipse_rules(model.settings)
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
        float(table.identity.get("r_end", float("inf"))),
    )
    return route, stencil


def losvd(source: Model | LosTable, r_2d: ArrayLike, v: ArrayLike) -> Any:
    r"""Return the line-of-sight velocity distribution at projected radii.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`. Given a model, its table is
        built first (`LosTable`), at every call: where a model is read more
        than once, build its table once with ``pp.LosTable(model)`` and pass
        the table.
    r_2d : array-like
        Projected radius :math:`R_{\rm 2D}` in the model's unit system
        (`Model.to`): internal units by default, parsecs (a float or an
        Astropy quantity) in the physical system.
    v : array-like
        Line-of-sight velocity in the model's unit system (km/s, a float or
        an Astropy quantity, in the physical system). `r_2d` and `v`
        broadcast against each other.

    Returns
    -------
    losvd : array
        The probability density of the line-of-sight velocity at each pair
        :math:`(R_{\rm 2D}, v)`, per unit velocity of the model's unit
        system (an Astropy quantity in the physical system), with the
        broadcast shape of `r_2d` and `v`. It is even in :math:`v`, integrates
        to one over all :math:`v`, and is zero beyond the escape velocity
        along the line of sight.

    Notes
    -----
    Between the rows of the table the distribution is read through the
    table's own stencil, the reading the table checked its rows for; a
    projected radius inside an unchecked interval is refused by name.

    Of a superposition, each component's distribution is read from its own
    table and the components are summed with their weights times their
    surface densities: :math:`D = \sum_k w_k \Sigma_k D_k / \sum_k w_k
    \Sigma_k`.
    """
    if isinstance(source, LosTable):
        table = source
    elif isinstance(source, Model):
        table = _table_of(source, "losvd")
    else:
        raise InputError("losvd reads a Model or a LosTable.")
    model = table.model
    x, u = np.broadcast_arrays(
        _internal_length(model, r_2d), _internal_velocity(model, v)
    )
    flat_x, flat_u = x.ravel(), u.ravel()
    if len(table.parts) == 1:
        value = _read_losvd(table.parts[0], flat_x, flat_u)
    else:
        weights = _weights(model)
        surfaces = [_sums(part.projected, flat_x)[1] for part in table.parts]
        reads = [_read_losvd(part, flat_x, flat_u) for part in table.parts]
        moment = mixture(weights, [s * d for s, d in zip(surfaces, reads, strict=True)])
        surface = mixture(weights, surfaces)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(surface > 0.0, moment / surface, 0.0)
    # a row's distribution is the density over all velocity: it integrates
    # to one on a mirror grid (the test of this function)
    density = np.asarray(value, dtype=np.float64).reshape(x.shape)
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return density / one


def aperture_losvd(source: Model | LosTable, aperture: Aperture, v: ArrayLike) -> Any:
    r"""Return the line-of-sight velocity distribution inside an aperture.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable`. Given a model, its table is
        built first (`LosTable`), at every call: where a model is read more
        than once, build its table once with ``pp.LosTable(model)`` and pass
        the table.
    aperture : Aperture
        The aperture, built with `Aperture.circle` or `Aperture.rectangle`,
        its lengths in the model's unit system (parsecs in the physical
        system).
    v : array-like
        Line-of-sight velocity in the model's unit system (km/s, a float or
        an Astropy quantity, in the physical system).

    Returns
    -------
    losvd : array
        The probability density of the line-of-sight velocity of the
        tracers inside the aperture, weighted by the surface density over
        it, at each `v`, per unit velocity of the model's unit system (an
        Astropy quantity in the physical system), with the shape of `v`. It
        is even in :math:`v`, integrates to one over all :math:`v`, and is
        zero beyond the escape velocity along the line of sight.

    Notes
    -----
    The distribution is the stage `aperture_losvd`, on Gauss-Legendre
    panels in velocity refined until its norm and second moment agree with
    the table's rows; at any `v` it is read by each panel's polynomial
    through its own nodes, the polynomial its quadrature integrates. Its
    second moment is `sigma_los_aperture`'s from the same `LosTable`, and
    it is checked against the aperture's dispersion (`validate_aperture`):
    a distribution above the goal, or read from a table above it, says so
    in a warning.

    Of a superposition, each component's distribution inside the aperture
    is summed with its weight times its projected mass inside the
    aperture: :math:`D = \sum_k w_k M_k D_k / \sum_k w_k M_k`.
    """
    if isinstance(source, LosTable):
        table = source
    elif isinstance(source, Model):
        table = _table_of(source, "aperture_losvd")
    else:
        raise InputError("aperture_losvd reads a Model or a LosTable.")
    model = table.model
    inside = _internal_aperture(model, aperture)
    u = np.abs(np.asarray(_internal_velocity(model, v), dtype=np.float64))
    reads, masses = [], []
    for part in table.parts:
        settings = part.model.settings
        result = _aperture_losvd(
            part.model, part.table, part.projected, inside, settings
        )
        reference = aperture_dispersion(part.projected, inside, settings)
        what = f"The aperture distribution of {part.model.components[0].name!r}"
        report = _with_parent(
            validate_aperture(result, reference), part.validation.level
        )
        _announce(what, report)
        refuse_unless_asked(what, report, settings)
        rule = panels(result.v_nodes, result.v_weights, result.losvd[0])
        reads.append(read_panels(rule, u))
        masses.append(result.mass[0])
    if len(reads) == 1:
        value = reads[0]
    else:
        weights = _weights(model)
        value = np.asarray(
            mixture(weights, [m * d for m, d in zip(masses, reads, strict=True)])
        ) / float(mixture(weights, masses))
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return value / one


def pdf_v(source: Model | LosTable, r: ArrayLike, v: ArrayLike) -> Any:
    r"""Return the local speed distribution at 3D radii.

    Parameters
    ----------
    source : Model or LosTable
        The model, or its prepared `LosTable` (whose intrinsic second
        moments the distribution is checked against).
    r : array-like
        Radius :math:`r` in the model's unit system (`Model.to`): internal
        units by default, parsecs (a float or an Astropy quantity) in the
        physical system.
    v : array-like
        Speed in the model's unit system (km/s, a float or an Astropy
        quantity, in the physical system). `r` and `v` broadcast against
        each other.

    Returns
    -------
    pdf_v : array
        The probability density of the speed :math:`V = |\mathbf{v}|` at
        each pair :math:`(r, V)`, per unit velocity of the model's unit
        system (an Astropy quantity in the physical system), with the
        broadcast shape of `r` and `v`. It integrates to one over
        :math:`V \geq 0` and is zero beyond the end of the DF's support,
        the escape speed but for a Type II model, whose speeds reach
        :math:`V_{\rm esc}/\sqrt{1 - r^2/r_a^2}` (its unbound share is
        ``SpeedDistribution.unbound_fraction``).

    Raises
    ------
    AccuracyError
        Where the distribution at a radius is beyond the hard limit, unless
        the model's settings ask for it anyway (``write_anyway``).

    Notes
    -----
    Each radius is the stage `speed_distribution`, checked by its norm
    against the density, its second moment against the Jeans dispersion
    and the change of its angle rule; the distribution at the speeds asked
    is read by the rule it checked. Of a superposition, the components are
    mixed by their density at the radius: :math:`p = \sum_k w_k \varrho_k
    p_k / \sum_k w_k \varrho_k`.

    References
    ----------
    Chiang+(2026d), Sect. 5.1 and [eqn:rho_f_anisotropic_scale].
    """
    model, views, profiles = _profiles(source, "pdf_v")
    x, u = np.broadcast_arrays(_internal_length(model, r), _internal_velocity(model, v))
    flat_x, flat_u = x.ravel(), u.ravel()
    value = np.zeros(flat_x.shape)
    weights = _weights(model)
    # each distinct radius once, its pairs found by one sort (no scan of every
    # pair per radius); each keeps its pairs in their order
    radii, inverse, counts = np.unique(flat_x, return_inverse=True, return_counts=True)
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    for radius, at in zip(radii, groups, strict=True):
        reads, densities, reports = [], [], []
        for view, profile in zip(views, profiles, strict=True):
            result = speed_distribution(view, float(radius), view.settings, profile)
            reports.append(result.record["report"])
            reads.append(result.read(flat_u[at]))
            densities.append(
                float(np.atleast_2d(view.profile.density(np.array([radius])))[0, 0])
            )
        report = _worst(reports)
        name = f"The speed distribution of {model.components[0].name!r}"
        if report.level >= 3 and not model.settings.write_anyway:
            raise AccuracyError(
                f"{name} at r = {radius:.6e} is beyond the hard limit "
                f"{HARD_LIMIT:g}: "
                + ", ".join(f"{k} {c.error:.2e}" for k, c in report.checks.items())
                + "."
            )
        _announce(name, report)
        if len(reads) == 1:
            value[at] = reads[0]
        else:
            moment = mixture(
                weights, [d * p for d, p in zip(densities, reads, strict=True)]
            )
            density = float(mixture(weights, densities))
            value[at] = moment / density if density > 0.0 else 0.0
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return value.reshape(x.shape) / one


def _mixed_aperture(
    model: Model, projected: list[ProjectedDispersion], aperture: Aperture
) -> float:
    """The dispersion of a superposition inside an aperture, from each
    component's two sums on the nodes of the first component."""
    observables = model.settings.observables
    first = projected[0]
    reader = surface_density_reader(
        first.r_2d,
        first.surface_density[0],
        float(first.model.profile.r_table_max),
        first.model.spatially_finite,
    )
    r, q, _ = aperture_nodes(
        reader,
        aperture,
        int(observables["n_aperture_nodes"]),
        int(observables["aperture_nodes_per_decade"]),
        INNER_DISK_TOLERANCE,
    )
    sums = [_sums(p, r) for p in projected]
    weights = _weights(model)
    moment = float(mixture(weights, [q @ a for a, _ in sums]))
    surface = float(mixture(weights, [q @ b for _, b in sums]))
    if not (surface > 0.0):
        raise outside_table(reader.r_min, reader.r_max)
    return float(np.sqrt(max(moment / surface, 0.0)))
