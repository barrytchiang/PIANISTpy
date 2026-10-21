r"""The velocity dispersions and velocity distributions of a model.

A user calls `sigma_r`, `sigma_t`, `sigma_los`, `sigma_los_aperture`,
`losvd`, `aperture_losvd` and `pdf_v`. Each observable function takes a
`Model` or the `LosTable` of a `Model`. Given a `Model`, an observable
function computes only the quantities that the call needs. Given a
`LosTable`, an observable function reads the tables that the `LosTable`
holds.

For a superposition (`superpose`) of components :math:`k` with weights
:math:`w_k`, each observable function computes the observable of each component and
combines the components. A projected second moment combines as

.. math::

    \sigma_{\rm los}^2 = \frac{\sum_k w_k \Sigma_k \sigma_{{\rm los},k}^2}
    {\sum_k w_k \Sigma_k},

with :math:`\Sigma_k` the surface density and
:math:`\sigma_{{\rm los},k}` the line-of-sight velocity dispersion of
component :math:`k`. An intrinsic second moment combines in the same form,
with the density :math:`\rho_k` of component :math:`k` in place of
:math:`\Sigma_k`.

A model, a `LosTable` and an aperture distribution each carry a validation
report. A result whose relative error is above :math:`10^{-2}` raises
AccuracyError, unless the settings set ``write_anyway=True``. A result
whose relative error is above :math:`10^{-4}` issues AccuracyWarning.
The observables are the subject of Notes Section T:6.
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
from ..validation import ValidationReport, announce, refuse_unless_asked
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
    validate_aperture,
    validate_table,
    velocity_distribution,
)
from .stages import aperture_losvd as _aperture_losvd


class LosTable:
    """The projected tables of a model, built once for many calls.

    A LosTable holds the intrinsic velocity dispersions of a model, the
    projected velocity dispersion of the model, and the model's table of
    line-of-sight velocity distributions over projected radius. Every
    observable function given a LosTable reads the tables of the LosTable.
    A model passed to many observable calls through one LosTable is
    therefore projected once. Building a LosTable starts parallel
    processes, one per core set by `set_cores`.

    Parameters
    ----------
    model : Model
        The model to project. The model has one component, or the model is
        a superposition of components (`superpose`).

    Attributes
    ----------
    model : Model
        The projected model. The observable functions given the LosTable
        return values in the unit system of `model`.
    profile : DispersionProfile
        The intrinsic second moments of the velocity of the model, computed
        at the model's settings. For a superposition, `profile` belongs to
        the first component.
    projected : ProjectedDispersion
        The surface density and the line-of-sight velocity dispersion over
        projected radius. For a superposition, `projected` belongs to the
        first component.
    kept : KeptRanges
        The ranges of projected radius over which the velocity quadrature
        of the table keeps two of the table's break velocities. For a
        superposition, `kept` belongs to the first component.
    table : LosvdTable
        The line-of-sight velocity distributions over projected radius. For
        a superposition, `table` belongs to the first component.
    validation : ValidationReport
        The checks of the table. Each check holds its relative error and
        the projected radius where the check's relative error is largest. The
        report's tier (``validation.tier``) is the band of the worst
        relative error among the table's checks and the model's own
        validation report. For a superposition, each check holds its
        largest relative error over the components.
    parts : tuple of LosTable
        For a superposition, one LosTable per component, each built from
        the component as a model of one component. For a model of one
        component, `parts` is ``(self,)``.

    Raises
    ------
    InputError
        If `model` is not a `Model`, or `model` has several components in
        one potential.
    InputError
        If a setting of the ``observables`` group is outside its valid
        range: ``n_projection_nodes`` below 20, or ``rows_route`` other than
        ``"kernel"`` or ``"numpy"``.
    AccuracyError
        If the model's or the table's relative error is above
        :math:`10^{-2}` and the settings do not set ``write_anyway=True``.
    ResultError
        If an intermediate result fails a check that the next step of the
        computation needs. Examples are a degenerate model, and a ``losvd``
        setting that leaves the table no valid quadrature
        (``energy_layer_ratio`` of one or more).
    WorkerError
        If a parallel process of the build ends before finishing its task.
        The usual cause is a lack of memory.

    Warns
    -----
    AccuracyWarning
        If the table's relative error is above :math:`10^{-4}` (tier
        ``"warned"``).

    See Also
    --------
    sigma_los : Line-of-sight velocity dispersion at projected radii.
    losvd : Line-of-sight velocity distribution at projected radii.
    aperture_losvd : Line-of-sight velocity distribution inside an aperture.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> table = pp.LosTable(model)
    >>> pp.sigma_los(table, [0.5, 1.0, 2.0]).shape
    (3,)
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
            raise InputError(
                "A LosTable is built from a Model; the value given is not a Model."
            )
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
        self.profile = _held_profile(model)
        self.projected = projected_dispersion(model, self.profile, settings)
        # kept_ranges and losvd_table read one route. Inside shared_pool,
        # the route's parallel processes start once for both calls.
        with shared_pool():
            self.kept = kept_ranges(model, self.projected, settings)
            self.table = losvd_table(model, self.kept, self.projected, settings)
        self.validation = _with_parent(
            validate_table(self.table, self.projected), model.validation.level
        )
        what = f"The LosTable of {model.components[0].name!r}"
        _announce(what, self.validation)
        # A table whose relative error is above 1e-2 raises AccuracyError
        # unless the settings set write_anyway=True.
        refuse_unless_asked(what, self.validation, settings)

    def to(self, system: str, **kwargs: Any) -> "LosTable":
        """Return the same tables presented in another unit system.

        The call computes no table again. The tables stay in internal
        units. The observable functions given the returned LosTable return
        values in the unit system of ``self.model.to(system, **kwargs)``.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            The target unit system.
        **kwargs
            The scales of the target unit system, as `Model.to` takes them:
            ``concentration``, ``mass_0``, and ``r_0`` or ``r_vir``.

        Returns
        -------
        table : LosTable
            A LosTable with the same tables, whose `model` is presented in
            `system`. The returned LosTable shares its parallel processes
            with the original LosTable (`LosTable.close`).

        Raises
        ------
        InputError
            If `system` or the scales in `kwargs` are not valid, under the
            rules of `Model.to`.

        See Also
        --------
        Model.to : Present a model in another unit system.
        """
        other = copy.copy(self)
        other.model = self.model.to(system, **kwargs)
        return other

    def close(self) -> None:
        """Close the parallel processes kept open by the table's aperture calls.

        With more than one core set by `set_cores`, the first aperture call
        on a LosTable starts parallel processes. Later aperture calls on the
        same LosTable reuse the parallel processes of the first aperture
        call. The parallel processes end at `LosTable.close` or when the
        LosTable is deleted. A LosTable returned by `LosTable.to` shares the
        parallel processes of the original LosTable. An aperture call after
        `LosTable.close` starts new parallel processes.

        See Also
        --------
        sigma_los_aperture : Line-of-sight velocity dispersion inside an
            aperture.
        aperture_losvd : Line-of-sight velocity distribution inside an
            aperture.
        """
        for part in self.parts:
            part.table.close_pools()


def _with_parent(report: ValidationReport, level: int) -> ValidationReport:
    """Return a copy of `report` whose parents' tier level is the larger of
    the report's own parents' tier level and `level`."""
    return ValidationReport(
        keys=report.keys,
        checks=report.checks,
        parents_level=max(report.parents_level, level),
    )


log = logging.getLogger("pianistpy")
# losvd and aperture_losvd build the LosTable of a Model at every call on a
# Model. _TABLE_HINT records whether the log message that suggests keeping
# the LosTable has been written in this process, so that the message is
# written once per process.
_TABLE_HINT = {"given": False}


def _table_of(model: Model, caller: str) -> LosTable:
    """Return the `LosTable` of `model` for the public function `caller`.

    At the first call in the process, a log message at level WARNING
    suggests keeping the LosTable with ``pp.LosTable(model)`` for further
    calls."""
    if not _TABLE_HINT["given"]:
        _TABLE_HINT["given"] = True
        log.warning(
            "%s was called on a Model, so the call builds the model's LosTable. A "
            "LosTable kept with pp.LosTable(model) and passed to later calls is "
            "built once.",
            caller,
        )
    return LosTable(model)


def _announce(what: str, report: ValidationReport) -> None:
    """Pass `report` to `validation.announce` with the checks whose
    relative error is above :math:`10^{-6}`, each with its error and its projected
    radius."""
    announce(
        what,
        report,
        ", ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at R = {check.where:.6e}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        or "the tier of the model that the result is computed from",
    )


def _views(model: Model) -> tuple[Model, ...]:
    """Return the models the observables are computed on.

    For a model of one component, the tuple holds `model` itself. For a
    superposition, the tuple holds each component as a model of one
    component. Each returned model passes `_refuse_unless_asked`. A model
    with several components in one potential raises InputError."""
    if len(model.components) == 1:
        _refuse_unless_asked(model)
        return (model,)
    if _is_superposition(model):
        _refuse_unless_asked(model)
        return tuple(_component_view(model, k) for k in range(len(model.components)))
    raise InputError(
        f"The observables are computed for a model of one component or for a "
        f"superposition made by superpose; the model given has "
        f"{len(model.components)} components in one potential."
    )


def _held_profile(view: Model) -> DispersionProfile:
    """Return the intrinsic second moments of `view` at the view's settings.

    The first call on a view builds the profile and stores the profile on
    the view. Later calls on the same view return the stored profile, so
    every read of one view uses the identical profile. `_views` makes the
    views of a superposition anew at each call, so the profile of a
    superposition's component is built at each call."""
    held = view.__dict__.get("_dispersion_profile")
    if held is None:
        held = dispersion_profile(view, view.settings)
        object.__setattr__(view, "_dispersion_profile", held)
    return held


def _refuse_unless_asked(model: Model) -> None:
    # A model whose relative error is above 1e-2 is accepted as an input
    # only when the model's settings set write_anyway=True. The validation
    # report of a superposition holds each check at its largest relative
    # error over the components.
    refuse_unless_asked(
        f"The model {model.components[0].name!r}", model.validation, model.settings
    )


def _worst(reports: list[ValidationReport]) -> ValidationReport:
    """Return one report from the reports of the components.

    Each check of the returned report is the component check with the
    largest relative error. The parents' tier level of the returned report
    is the largest parents' tier level among `reports`."""
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
    """Return the weights of the model's components, as `superpose` set
    them. `mixture` divides the weights by their sum."""
    return np.array([float(c.mass) for c in model.components])


def _sums(projected: ProjectedDispersion, r_2d: Array) -> tuple[Array, Array]:
    """Return two arrays of one component at the projected radii `r_2d`.

    The first array is the component's surface density times the
    component's squared line-of-sight velocity dispersion. The second array
    is the component's surface density. The readers interpolate the
    component's tables as `ProjectedDispersion.read` interpolates them."""
    model = projected.model
    read_surface = surface_density_reader(
        projected.r_2d,
        projected.surface_density[0],
        float(model.profile.r_table_max),
        model.spatially_finite,
    )
    read_los2 = dispersion_table_reader(
        projected.r_2d, np.asarray(projected.sigma_los[0]) ** 2
    )
    inner, outer = projected.notes()
    for reader in (read_surface, read_los2):
        reader.inner_note, reader.outer_note = inner, outer
    surface = read_surface(r_2d)
    sigma2 = read_los2(r_2d)
    return surface * sigma2, surface


def _projected(view: Model) -> ProjectedDispersion:
    settings = view.settings
    return projected_dispersion(view, _held_profile(view), settings)


def _length_factor(model: Model) -> float:
    """Return the model's unit of length in internal units.

    The returned factor is one in the internal and the model systems. In
    the physical system, the returned factor is the scale radius in parsecs.
    A length in parsecs divided by the returned factor is in internal units."""
    units = model.units
    if units.system != "physical":
        return 1.0
    factor = units.factors(model.components[0].unit_profile).physical_r_pc
    if factor is None:
        raise InputError(
            "The model's physical unit system gives no length unit, so a length "
            "in parsecs cannot be converted to internal units."
        )
    return float(factor)


def _internal_length(model: Model, value: Any) -> Array:
    """Return a length given in the model's unit system in internal units.

    In the physical system, `value` is a float in parsecs or an Astropy
    quantity of any length unit."""
    if hasattr(value, "unit"):
        value = value.to_value("pc")
    return np.asarray(value, dtype=np.float64) / _length_factor(model)


def _internal_aperture(model: Model, aperture: Aperture) -> Aperture:
    """Return `aperture` with its lengths converted from the model's unit
    system to internal units. The conversion takes `float` of each length,
    so in the physical system a length must be a float in parsecs."""
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
    """Return a velocity given in internal units in the model's unit
    system."""
    return model.units.convert("velocity", value, model.components[0].unit_profile)


def sigma_los(source: Model | LosTable, r_2d: ArrayLike) -> Array:
    r"""Return the line-of-sight velocity dispersion at projected radii.

    The line-of-sight velocity dispersion
    :math:`\sigma_{\rm los}(R_{\rm 2D})` at projected radius
    :math:`R_{\rm 2D}` is the square root of the second moment of the
    line-of-sight velocity, weighted by the density along the line of
    sight.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. A `Model` and the
        `LosTable` of the same model return the same values.
    r_2d : float, array-like or Quantity
        User-specified projected radii :math:`R_{\rm 2D}`. The unit is the
        scale radius :math:`r_0` in the internal and the model systems, and
        the parsec in the physical system (`Model.to`). In the physical
        system, an Astropy quantity of any length unit is also accepted.

    Returns
    -------
    sigma_los : array or Quantity
        Line-of-sight velocity dispersion :math:`\sigma_{\rm los}`, with
        the shape of `r_2d`. In the internal system, a plain array in
        internal units. In the model system, a plain array in units of
        :math:`(G M_0/r_0)^{1/2}`. In the physical system, an Astropy
        quantity in km/s.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` with several components in one potential.
        A model of one component and a superposition (`superpose`) are
        accepted.
    AccuracyError
        If `source` is a `Model` whose relative error is above
        :math:`10^{-2}` and whose settings do not set
        ``write_anyway=True``.
    InputError
        If a radius lies inside the innermost or beyond the outermost
        projected radius that the model tabulates. The call does not
        extrapolate.
    ResultError
        If `source` is a `Model` and an intermediate result fails a check
        that the next step of the computation needs, such as values that
        are not finite for a degenerate model.

    See Also
    --------
    sigma_los_aperture : Line-of-sight velocity dispersion inside an
        aperture.
    losvd : Line-of-sight velocity distribution at projected radii.
    LosTable : The projected tables of a model, built once for many calls.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    surface densities :math:`\Sigma_k` and line-of-sight velocity
    dispersions :math:`\sigma_{{\rm los},k}`,

    .. math::

        \sigma_{\rm los}^2 = \frac{\sum_k w_k \Sigma_k
        \sigma_{{\rm los},k}^2}{\sum_k w_k \Sigma_k}.

    References
    ----------
    Notes Section T:6.2 (projection) and Notes Section T:6.9 (superposed
    models).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> pp.sigma_los(model, [0.1, 1.0, 10.0]).shape
    (3,)
    """
    if isinstance(source, LosTable):
        model = source.model
        projected = [part.projected for part in source.parts]
    elif isinstance(source, Model):
        model = source
        projected = [_projected(view) for view in _views(source)]
    else:
        raise InputError(
            "The source of sigma_los must be a Model or a LosTable; the "
            "source given is neither."
        )
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
    r"""Return the line-of-sight velocity dispersion inside an aperture.

    The line-of-sight velocity dispersion :math:`\sigma_{\rm ape}` inside
    an aperture is the square root of the second moment of the
    line-of-sight velocity over the aperture, weighted by the surface
    density. Given a `Model`, the call computes the projected dispersion
    only at the projected radii that the aperture covers. Given a
    `LosTable`, the call computes :math:`\sigma_{\rm ape}` as the square
    root of the second moment of the distribution that `aperture_losvd`
    returns. The call then compares the second moment of the aperture
    distribution with :math:`\sigma_{\rm ape}^2` computed from the
    projected dispersion. The relative difference enters the validation
    report of the aperture distribution.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model.
    aperture : Aperture
        The aperture, built by `Aperture.circle` or `Aperture.rectangle`.
        The unit of the aperture's lengths is the scale radius :math:`r_0`
        in the internal and the model systems, and the parsec in the
        physical system (`Model.to`). Each length is a float.

    Returns
    -------
    sigma_ape : float or Quantity
        Line-of-sight velocity dispersion :math:`\sigma_{\rm ape}` inside
        the aperture. In the internal system, a float in internal units. In
        the model system, a float in units of :math:`(G M_0/r_0)^{1/2}`. In
        the physical system, an Astropy quantity in km/s.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` with several components in one potential.
        A model of one component and a superposition (`superpose`) are
        accepted.
    AccuracyError
        If `source` is a `Model` whose relative error is above
        :math:`10^{-2}` and whose settings do not set
        ``write_anyway=True``.
    InputError
        If the aperture fails one of four conditions: the aperture lies
        wholly outside the tabulated projected radii; a centered circle's
        radius does not exceed the innermost tabulated projected radius; a
        centered circle reaches beyond the outermost tabulated projected
        radius; or the part of the aperture inside the innermost tabulated
        projected radius carries more than :math:`10^{-6}` of the
        aperture's weight.
    AccuracyError
        If `source` is a `LosTable`, the aperture distribution's relative
        error is above :math:`10^{-2}`, and the settings do not set
        ``write_anyway=True``.
    ResultError
        If `source` is a `Model` and an intermediate result fails a check
        that the next step of the computation needs, such as values that
        are not finite for a degenerate model.
    WorkerError
        If `source` is a `LosTable`, more than one core is set by
        `set_cores`, and a parallel process of the call ends before
        finishing its task. The usual cause is a lack of memory.

    Warns
    -----
    AccuracyWarning
        If `source` is a `LosTable` and the relative error of the aperture
        distribution, or of the `LosTable`, is above :math:`10^{-4}` (tier
        ``"warned"``).

    See Also
    --------
    sigma_los : Line-of-sight velocity dispersion at projected radii.
    aperture_losvd : Line-of-sight velocity distribution inside an aperture.
    Aperture : A circular or rectangular aperture.

    Notes
    -----
    Given a `Model`, the call evaluates

    .. math::

        \sigma_{\rm ape}^2 = \frac{\int_A \Sigma\,\sigma_{\rm los}^2\,{\rm d}A}
        {\int_A \Sigma\,{\rm d}A},

    with :math:`A` the aperture, :math:`\Sigma` the surface density and
    :math:`\sigma_{\rm los}` the line-of-sight velocity dispersion. For a
    superposition of components :math:`k` with weights :math:`w_k`, the
    numerator integrates :math:`\sum_k w_k \Sigma_k \sigma_{{\rm los},k}^2`
    and the denominator integrates :math:`\sum_k w_k \Sigma_k`. Given a
    `LosTable`, the call sums the second moments of the components'
    aperture distributions with the weights :math:`w_k M_k`, with
    :math:`M_k` the projected mass of component :math:`k` inside the
    aperture.

    References
    ----------
    Notes Section T:6.3 (apertures) and Notes Section T:6.6 (the
    distribution in an aperture).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> sigma = pp.sigma_los_aperture(model, pp.Aperture.circle(1.0))
    >>> bool(sigma > 0.0)
    True
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
        raise InputError(
            "The source of sigma_los_aperture must be a Model or a LosTable; the "
            "source given is neither."
        )
    converted = _velocity(model, value)
    return converted if hasattr(converted, "unit") else float(converted)


def _velocity_factor(model: Model) -> float:
    """Return the value of one internal unit of velocity in the model's unit
    system: one in the internal system, the velocity factor of `Model.to` in
    the model and the physical systems."""
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return float(getattr(one, "value", one))


def _internal_velocity(model: Model, value: Any) -> Array:
    """Return a velocity given in the model's unit system in internal units.

    In the physical system, `value` is a float in km/s or an Astropy
    quantity of any velocity unit."""
    if hasattr(value, "unit"):
        value = value.to_value("km / s")
    return np.asarray(value, dtype=np.float64) / _velocity_factor(model)


def _profiles(
    source: Model | LosTable, what: str
) -> tuple[Model, list[Model], list[DispersionProfile]]:
    """Return the model of `source`, the model's views (`_views`) and the
    intrinsic second moments of each view. The InputError raised for a
    `source` of another type gives `what`, the name of the public function."""
    if isinstance(source, LosTable):
        parts = source.parts
        return source.model, [p.model for p in parts], [p.profile for p in parts]
    if isinstance(source, Model):
        views = list(_views(source))
        return source, views, [_held_profile(v) for v in views]
    raise InputError(
        f"The source of {what} must be a Model or a LosTable; the "
        "source given is neither."
    )


def _intrinsic(source: Model | LosTable, r: ArrayLike, which: int, what: str) -> Any:
    """Return `sigma_r` (`which` 0) or `sigma_t` (`which` 1) at the 3D radii
    `r`, in the model's unit system.

    For a superposition, the squared dispersion at a radius is the weighted
    mean of the components' squared dispersions, each component weighted by
    its weight times its density at the radius. The density weights an
    intrinsic quantity as the surface density weights a projected
    quantity."""
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
    r"""Return the radial velocity dispersion at 3D radii.

    The radial velocity dispersion :math:`\sigma_r(r)` is the square root
    of the second moment of the radial velocity at 3D radius :math:`r`.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. A `Model` and the
        `LosTable` of the same model return the same values.
    r : float, array-like or Quantity
        User-specified 3D radii :math:`r`. The unit is the scale radius
        :math:`r_0` in the internal and the model systems, and the parsec
        in the physical system (`Model.to`). In the physical system, an
        Astropy quantity of any length unit is also accepted.

    Returns
    -------
    sigma_r : array or Quantity
        Radial velocity dispersion :math:`\sigma_r`, with the shape of `r`.
        In the internal system, a plain array in internal units. In the
        model system, a plain array in units of :math:`(G M_0/r_0)^{1/2}`.
        In the physical system, an Astropy quantity in km/s.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` with several components in one potential.
        A model of one component and a superposition (`superpose`) are
        accepted.
    AccuracyError
        If `source` is a `Model` whose relative error is above
        :math:`10^{-2}` and whose settings do not set
        ``write_anyway=True``.
    InputError
        If a radius lies inside the innermost or beyond the outermost
        radius that the model tabulates, or beyond the edge of a spatially
        finite model. The call does not extrapolate.
    ResultError
        If `source` is a `Model` and an intermediate result fails a check
        that the next step of the computation needs, such as values that
        are not finite for a degenerate model.

    See Also
    --------
    sigma_t : Tangential velocity dispersion at 3D radii.
    sigma_los : Line-of-sight velocity dispersion at projected radii.
    pdf_v : Local velocity distribution at 3D radii.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    densities :math:`\rho_k` and radial velocity dispersions
    :math:`\sigma_{r,k}`,

    .. math::

        \sigma_r^2 = \frac{\sum_k w_k \rho_k \sigma_{r,k}^2}
        {\sum_k w_k \rho_k}.

    References
    ----------
    Notes Section T:6.1 (intrinsic moments) and Notes Section T:6.9
    (superposed models).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> pp.sigma_r(model, [0.1, 1.0, 10.0]).shape
    (3,)
    """
    return _intrinsic(source, r, 0, "sigma_r")


def sigma_t(source: Model | LosTable, r: ArrayLike) -> Array:
    r"""Return the tangential velocity dispersion at 3D radii.

    The tangential velocity dispersion :math:`\sigma_{\rm t}(r)` combines
    both tangential components of the velocity at 3D radius :math:`r`:
    :math:`\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2`.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. A `Model` and the
        `LosTable` of the same model return the same values.
    r : float, array-like or Quantity
        User-specified 3D radii :math:`r`. The unit is the scale radius
        :math:`r_0` in the internal and the model systems, and the parsec
        in the physical system (`Model.to`). In the physical system, an
        Astropy quantity of any length unit is also accepted.

    Returns
    -------
    sigma_t : array or Quantity
        Tangential velocity dispersion :math:`\sigma_{\rm t}`, with the
        shape of `r`. The square :math:`\sigma_{\rm t}^2` equals
        :math:`2 (1 - \beta_\sigma) \sigma_r^2`, with :math:`\beta_\sigma`
        the velocity anisotropy and :math:`\sigma_r` the radial velocity
        dispersion. In the internal system, a plain array in internal
        units. In the model system, a plain array in units of
        :math:`(G M_0/r_0)^{1/2}`. In the physical system, an Astropy
        quantity in km/s.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` with several components in one potential.
        A model of one component and a superposition (`superpose`) are
        accepted.
    AccuracyError
        If `source` is a `Model` whose relative error is above
        :math:`10^{-2}` and whose settings do not set
        ``write_anyway=True``.
    InputError
        If a radius lies inside the innermost or beyond the outermost
        radius that the model tabulates, or beyond the edge of a spatially
        finite model. The call does not extrapolate.
    ResultError
        If `source` is a `Model` and an intermediate result fails a check
        that the next step of the computation needs, such as values that
        are not finite for a degenerate model.

    See Also
    --------
    sigma_r : Radial velocity dispersion at 3D radii.
    sigma_los : Line-of-sight velocity dispersion at projected radii.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    densities :math:`\rho_k` and tangential velocity dispersions
    :math:`\sigma_{{\rm t},k}`,

    .. math::

        \sigma_{\rm t}^2 = \frac{\sum_k w_k \rho_k \sigma_{{\rm t},k}^2}
        {\sum_k w_k \rho_k}.

    References
    ----------
    Notes Section T:6.1 (intrinsic moments) and Notes Section T:6.9
    (superposed models).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> pp.sigma_t(model, [0.1, 1.0, 10.0]).shape
    (3,)
    """
    return _intrinsic(source, r, 1, "sigma_t")


def _read_losvd(part: LosTable, r_2d: Array, v: Array) -> Array:
    """Return the line-of-sight velocity distribution of one component at the
    pairs ``(r_2d[j], v[j])``, in internal units.

    The table's stencil interpolates between the table's rows, as an
    aperture reads the table. The returned values are a probability density
    over all line-of-sight velocities, with unit integral."""
    route, stencil = _reader_of(part)
    return np.asarray(stencil.at_pairs(route, r_2d, v), dtype=np.float64)


def _reader_of(part: LosTable) -> tuple[Any, Any]:
    """Return the route and the stencil of one component, from `_stencil_of`.

    The first read of a table makes the route and the stencil and stores
    both in the table's `LosvdTable.readers`. Later reads of the table and
    the transpose of the read (`losvd_read_vjp`) reuse the stored route and
    stencil. The route's cache of line-of-sight extrema is the table's
    `LosvdTable.extrema`. Each cached extremum at a radius equals the
    extremum that the radius forms when computed alone. A table without a
    ``readers`` attribute gets a new route and stencil at every read."""
    model, table = part.model, part.table
    readers = getattr(table, "readers", None)
    if readers is None:
        return _stencil_of(part)
    key = (model.identity, ellipse_rules(model.settings))
    if key not in readers:
        route, stencil = _stencil_of(part)
        route._fext = table.extrema.setdefault(key, {})
        readers[key] = (route, stencil)
    return readers[key]


def _stencil_of(part: LosTable) -> tuple[Any, Any]:
    """Return a new route of one component, without the DF, and a new
    stencil that interpolates the component's table between its rows."""
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

    The line-of-sight velocity distribution :math:`D(R_{\rm 2D}, v)` is the
    probability density of the line-of-sight velocity :math:`v` of the
    tracers at projected radius :math:`R_{\rm 2D}`.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. Given a `Model`, the call
        builds the model's `LosTable` at every call. For many calls on one
        model, pass ``pp.LosTable(model)``.
    r_2d : float, array-like or Quantity
        User-specified projected radii :math:`R_{\rm 2D}`. The unit is the
        scale radius :math:`r_0` in the internal and the model systems, and
        the parsec in the physical system (`Model.to`). In the physical
        system, an Astropy quantity of any length unit is also accepted.
    v : float, array-like or Quantity
        Line-of-sight velocity :math:`v`. The unit is the internal unit in
        the internal system, :math:`(G M_0/r_0)^{1/2}` in the model system,
        and km/s in the physical system. In the physical system, an Astropy
        quantity of any velocity unit is also accepted. The arrays `r_2d`
        and `v` broadcast against each other.

    Returns
    -------
    losvd : array or Quantity
        The probability density :math:`D` at each pair
        :math:`(R_{\rm 2D}, v)`, with the broadcast shape of `r_2d` and `v`.
        :math:`D` is even in :math:`v`, integrates to one over all
        :math:`v`, and is zero beyond the support of the distribution
        function along the line of sight (the escape velocity, except for a
        Type II model, whose support extends above it). In the internal and
        the model systems, a plain array per unit velocity of the system. In
        the physical system, an Astropy quantity in s/km.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` and building the model's `LosTable` raises
        InputError (see `LosTable`).
    AccuracyError
        If `source` is a `Model` and building the model's `LosTable` raises
        AccuracyError (see `LosTable`).
    ResultError
        If `source` is a `Model` and building the model's `LosTable` raises
        ResultError (see `LosTable`).
    WorkerError
        If `source` is a `Model` and building the model's `LosTable` raises
        WorkerError (see `LosTable`).
    InputError
        If a radius lies inside the innermost or beyond the outermost
        projected radius of the table. The call does not extrapolate.
    ResultError
        If a radius lies inside an interval of projected radius where the
        table does not meet its tolerance
        (the attribute ``r_uncertified`` of ``LosTable.table``). The message gives the
        interval and the reason.

    See Also
    --------
    aperture_losvd : Line-of-sight velocity distribution inside an aperture.
    sigma_los : Line-of-sight velocity dispersion at projected radii.
    LosTable : The projected tables of a model, built once for many calls.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    surface densities :math:`\Sigma_k` and line-of-sight velocity
    distributions :math:`D_k`,

    .. math::

        D = \frac{\sum_k w_k \Sigma_k D_k}{\sum_k w_k \Sigma_k}.

    References
    ----------
    Notes Section T:6.4 (the line-of-sight velocity distribution) and
    Notes Section T:6.9 (superposed models).

    Examples
    --------
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> table = pp.LosTable(model)
    >>> v = np.linspace(-1.0, 1.0, 5)
    >>> pp.losvd(table, 1.0, v).shape
    (5,)
    """
    if isinstance(source, LosTable):
        table = source
    elif isinstance(source, Model):
        table = _table_of(source, "losvd")
    else:
        raise InputError(
            "The source of losvd must be a Model or a LosTable; the "
            "source given is neither."
        )
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
    # Each row's distribution is a probability density over all line-of-sight
    # velocities, negative and positive, with unit integral. Dividing by one
    # internal unit of velocity in the model's system gives the density per
    # unit velocity of the model's system.
    density = np.asarray(value, dtype=np.float64).reshape(x.shape)
    one = model.units.convert("velocity", 1.0, model.components[0].unit_profile)
    return density / one


def aperture_losvd(source: Model | LosTable, aperture: Aperture, v: ArrayLike) -> Any:
    r"""Return the line-of-sight velocity distribution inside an aperture.

    The aperture distribution is the probability density of the
    line-of-sight velocity of the model's mass inside the aperture, weighted by
    the surface density over the aperture. The second moment of the
    aperture distribution is the square of `sigma_los_aperture` given the
    same `LosTable`. The call compares the second moment of the aperture distribution
    with the squared
    aperture dispersion computed from the projected dispersion, without the
    distribution. The relative difference enters the validation report of
    the aperture distribution.

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. Given a `Model`, the call
        builds the model's `LosTable` at every call. For many calls on one
        model, pass ``pp.LosTable(model)``.
    aperture : Aperture
        The aperture, built by `Aperture.circle` or `Aperture.rectangle`.
        The unit of the aperture's lengths is the scale radius :math:`r_0`
        in the internal and the model systems, and the parsec in the
        physical system (`Model.to`). Each length is a float.
    v : float, array-like or Quantity
        Line-of-sight velocity :math:`v`. The unit is the internal unit in
        the internal system, :math:`(G M_0/r_0)^{1/2}` in the model system,
        and km/s in the physical system. In the physical system, an Astropy
        quantity of any velocity unit is also accepted.

    Returns
    -------
    losvd : array or Quantity
        The aperture distribution at each `v`, with the shape of `v`. The
        aperture distribution is even in :math:`v`, integrates to one over
        all :math:`v`, and is zero beyond the support of the distribution
        function along the line of sight (the escape velocity, except for a
        Type II model, whose support extends above it). In the internal and
        the model systems, a plain array per unit velocity of the system. In
        the physical system, an Astropy quantity in s/km.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` and building the model's `LosTable` raises
        InputError (see `LosTable`).
    AccuracyError
        If `source` is a `Model` and building the model's `LosTable` raises
        AccuracyError (see `LosTable`).
    ResultError
        If `source` is a `Model` and building the model's `LosTable` raises
        ResultError (see `LosTable`).
    WorkerError
        If `source` is a `Model` and building the model's `LosTable` raises
        WorkerError (see `LosTable`).
    InputError
        If the aperture fails one of four conditions: the aperture lies
        wholly outside the tabulated projected radii; a centered circle's
        radius does not exceed the innermost tabulated projected radius; a
        centered circle reaches beyond the outermost tabulated projected
        radius; or the part of the aperture inside the innermost tabulated
        projected radius carries more than :math:`10^{-6}` of the
        aperture's weight.
    AccuracyError
        If the aperture distribution's relative error is above
        :math:`10^{-2}` and the settings do not set ``write_anyway=True``.
    WorkerError
        If more than one core is set by `set_cores` and a parallel process
        of the call ends before finishing its task. The usual cause is a
        lack of memory.

    Warns
    -----
    AccuracyWarning
        If the relative error of the aperture distribution, or of the
        `LosTable`, is above :math:`10^{-4}` (tier ``"warned"``).

    See Also
    --------
    sigma_los_aperture : Line-of-sight velocity dispersion inside an
        aperture.
    losvd : Line-of-sight velocity distribution at projected radii.
    Aperture : A circular or rectangular aperture.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    aperture distributions :math:`D_k` and projected masses :math:`M_k`
    inside the aperture,

    .. math::

        D = \frac{\sum_k w_k M_k D_k}{\sum_k w_k M_k}.

    References
    ----------
    Notes Section T:6.6 (the distribution in an aperture) and
    Notes Section T:6.9 (superposed models).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> table = pp.LosTable(model)
    >>> pp.aperture_losvd(table, pp.Aperture.circle(1.0), [0.0, 0.2, 0.4]).shape
    (3,)
    """
    if isinstance(source, LosTable):
        table = source
    elif isinstance(source, Model):
        table = _table_of(source, "aperture_losvd")
    else:
        raise InputError(
            "The source of aperture_losvd must be a Model or a LosTable; the "
            "source given is neither."
        )
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
    r"""Return the local velocity distribution at 3D radii.

    The local velocity distribution :math:`p(V \,|\, r)` is the probability
    density of the velocity :math:`V = |\mathbf{v}|` of the model's mass at 3D
    radius :math:`r`. The call checks the distribution at each radius in
    three ways: the norm of the distribution against the density; the
    second moment of the distribution against the velocity dispersion from
    the Jeans equation; and the change of the distribution when the angular
    quadrature is refined (`pianistpy.observables.stages.velocity_distribution`).

    Parameters
    ----------
    source : Model or LosTable
        The model, or the `LosTable` of the model. A `Model` and the
        `LosTable` of the same model return the same values.
    r : float, array-like or Quantity
        User-specified 3D radii :math:`r`. The unit is the scale radius
        :math:`r_0` in the internal and the model systems, and the parsec
        in the physical system (`Model.to`). In the physical system, an
        Astropy quantity of any length unit is also accepted.
    v : float, array-like or Quantity
        Velocity :math:`V`. The unit is the internal unit in the internal
        system, :math:`(G M_0/r_0)^{1/2}` in the model system, and km/s in
        the physical system. In the physical system, an Astropy quantity of
        any velocity unit is also accepted. The arrays `r` and `v` broadcast
        against each other.

    Returns
    -------
    pdf_v : array or Quantity
        The probability density :math:`p(V \,|\, r)` at each pair
        :math:`(r, V)`, with the broadcast shape of `r` and `v`. The
        density integrates to one over :math:`V \geq 0`. The density is zero
        beyond the end of the support of the distribution function (DF).
        For a Type II model, the support ends at
        :math:`V_{\rm esc}/\sqrt{1 - r^2/r_{\rm a}^2}`, with
        :math:`V_{\rm esc}` the escape velocity and :math:`r_{\rm a}` the
        anisotropy radius. For every other model, the support ends at the
        escape velocity :math:`V_{\rm esc}`. In the internal and the model
        systems, a plain array per unit velocity of the system. In the
        physical system, an Astropy quantity in s/km.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError
        If `source` is a `Model` with several components in one potential.
        A model of one component and a superposition (`superpose`) are
        accepted.
    AccuracyError
        If `source` is a `Model` whose relative error is above
        :math:`10^{-2}` and whose settings do not set
        ``write_anyway=True``.
    InputError
        If a radius lies outside the radii that the model tabulates. For a
        Type II model, also if a radius lies at or beyond the anisotropy
        radius :math:`r_{\rm a}`.
    AccuracyError
        If the relative error of the distribution at a radius is above
        :math:`10^{-2}` and the model's settings do not set
        ``write_anyway=True``.
    ResultError
        If `source` is a `Model` and an intermediate result fails a check
        that the next step of the computation needs, such as values that
        are not finite for a degenerate model.

    Warns
    -----
    AccuracyWarning
        If the relative error of the distribution at a radius is above
        :math:`10^{-4}` (tier ``"warned"``).

    See Also
    --------
    sigma_r : Radial velocity dispersion at 3D radii.
    sigma_t : Tangential velocity dispersion at 3D radii.

    Notes
    -----
    For a superposition of components :math:`k` with weights :math:`w_k`,
    densities :math:`\varrho_k` and velocity distributions :math:`p_k`,

    .. math::

        p = \frac{\sum_k w_k \varrho_k p_k}{\sum_k w_k \varrho_k}.

    References
    ----------
    Chiang+(2026d), Section 5.1 and Eq. (B38).
    Notes Section T:6.8 (the local velocity distribution).

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> pp.pdf_v(model, 1.0, [0.1, 0.3, 0.5]).shape
    (3,)
    """
    model, views, profiles = _profiles(source, "pdf_v")
    x, u = np.broadcast_arrays(_internal_length(model, r), _internal_velocity(model, v))
    flat_x, flat_u = x.ravel(), u.ravel()
    value = np.zeros(flat_x.shape)
    weights = _weights(model)
    # The loop computes each distinct radius once. One stable sort groups the
    # indices of the pairs at each distinct radius, so the loop does not scan
    # every pair for every radius. Each group keeps the pairs in input order.
    radii, inverse, counts = np.unique(flat_x, return_inverse=True, return_counts=True)
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    for radius, at in zip(radii, groups, strict=True):
        reads, densities, reports = [], [], []
        for view, profile in zip(views, profiles, strict=True):
            result = velocity_distribution(view, float(radius), view.settings, profile)
            reports.append(result.record["report"])
            reads.append(result.read(flat_u[at]))
            densities.append(
                float(np.atleast_2d(view.profile.density(np.array([radius])))[0, 0])
            )
        report = _worst(reports)
        name = f"The velocity distribution of {model.components[0].name!r}"
        if report.level >= 3 and not model.settings.write_anyway:
            raise AccuracyError(
                f"{name} at r = {radius:.6e} is beyond 0.01 (tier 'refused'); the "
                "checks give the relative errors "
                + ", ".join(f"{k} {c.error:.2e}" for k, c in report.checks.items())
                + ". The call does not return a velocity distribution beyond 0.01 "
                "unless the model's settings set write_anyway=True."
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
    """Return the line-of-sight velocity dispersion of a superposition inside
    `aperture`, in internal units.

    The aperture nodes and weights come from the surface density of the
    first component. At the aperture nodes of the first component, `_sums`
    reads each component's surface density times its squared dispersion,
    and each component's surface density. The squared dispersion of the
    superposition is the weighted mixture of the aperture integrals of
    surface density times squared dispersion, divided by the weighted
    mixture of the aperture integrals of surface density."""
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
