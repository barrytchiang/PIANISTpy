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

__all__ = ["LosTable", "sigma_los_aperture", "sigma_los"]

import copy
import dataclasses
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import InputError
from ..particles.core import _component_view, _is_superposition
from ..superposition import mixture
from ..typing import Array, ArrayLike
from ..validation import ValidationReport, announce, refuse_unless_asked
from ._aperture import aperture_nodes, outside_table
from ._losvd_stages import shared_pool
from ._projection import dispersion_table_reader, surface_density_reader
from .aperture import Aperture
from .stages import (
    INNER_DISC_TOLERANCE,
    DispersionProfile,
    KeptRanges,
    LosvdTable,
    ProjectedDispersion,
    aperture_dispersion,
    aperture_losvd,
    dispersion_profile,
    kept_ranges,
    losvd_table,
    projected_dispersion,
    validate_aperture,
    validate_table,
)


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
        _announce(f"The LosTable of {model.components[0].name!r}", self.validation)

    def to(self, system: str, **kwargs: Any) -> "LosTable":
        """Return the same tables presented in another unit system.

        Nothing is recomputed: the tables stay in internal units, and the
        observables read from the returned object are given in the units of
        ``self.model.to(system, **kwargs)`` (`Model.to`).
        """
        other = copy.copy(self)
        other.model = self.model.to(system, **kwargs)
        return other


def _with_parent(report: ValidationReport, level: int) -> ValidationReport:
    """Return `report` with `level` among the worst tiers of its parents."""
    return ValidationReport(
        keys=report.keys,
        checks=report.checks,
        parents_level=max(report.parents_level, level),
    )


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
    """The components' weights, as `superpose` set them (normalised by
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
            result = aperture_losvd(
                part.model, part.table, part.projected, inside, settings
            )
            moments.append(
                float(result.record["m2_file"]) / float(result.record["norm_file"])
            )
            masses.append(result.mass[0])
            reference = aperture_dispersion(part.projected, inside, settings)
            _announce(
                f"The aperture distribution of {part.model.components[0].name!r}",
                _with_parent(
                    validate_aperture(result, reference), part.validation.level
                ),
            )
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
        INNER_DISC_TOLERANCE,
    )
    sums = [_sums(p, r) for p in projected]
    weights = _weights(model)
    moment = float(mixture(weights, [q @ a for a, _ in sums]))
    surface = float(mixture(weights, [q @ b for _, b in sums]))
    if not (surface > 0.0):
        raise outside_table(reader.r_min, reader.r_max)
    return float(np.sqrt(max(moment / surface, 0.0)))
