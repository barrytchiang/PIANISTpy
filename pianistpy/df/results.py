"""The results of the DF computation and their files.

A result is frozen, holds the stage results it was built from, and is the
same thing as its file: writing it and reading it back returns an equal
result.
"""

__all__ = ["DistributionFunction", "Model"]

import dataclasses
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..io import StoredResult, load_result, register_reader, write_result, write_table
from ..numerics.pytrees import register_result
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..settings import Settings
from ..typing import Array
from ..units import Units
from ..validation import ValidationReport
from .inputs import Component, Truncation
from .stages import (
    AsymptoticForm,
    DfTable,
    EnergyGrid,
    InversionRecord,
    InversionRows,
)

DF_KIND = "distribution_function"
MODEL_KIND = "model"


@dataclasses.dataclass(frozen=True, eq=False)
class DistributionFunction:
    """Untruncated distribution functions of the components of a model.

    Returned by `compute_df`. It is a mathematical object: it may be negative or
    non-monotonic, and states where. Only a `Model` can be sampled or
    projected.

    Parameters
    ----------
    components : tuple of Component
        The components the DFs belong to.
    settings : Settings
        The settings the DFs were computed with.
    identity : str
        Identity string of the result.
    grid : EnergyGrid
        The energies of the table and their radii.
    table : DfTable
        The DFs on the grid.
    form : AsymptoticForm
        The analytic form of each DF beyond the ends of its table.
    k_t_min : float
        Lowest admissible truncation energy; zero where every DF is
        non-negative and non-decreasing.
    validation : ValidationReport
        The report of the inversion.
    timing : dict of str to float
        Wall time of each stage, in seconds.
    inversion : InversionRecord or None
        How the table was made; None where it was not made by an inversion
        (a closed form). Every solved DF states it.
    rows : InversionRows or None
        The inversion's rows as made on its whole grid, which the truncation
        reads; None where the table was not made by an inversion.
    """

    components: tuple[Component, ...]
    settings: Settings
    identity: str
    grid: EnergyGrid
    table: DfTable
    form: AsymptoticForm
    k_t_min: float
    validation: ValidationReport
    timing: dict[str, float]
    inversion: InversionRecord | None
    units: Units = Units()
    rows: InversionRows | None = None

    @property
    def is_physical(self) -> bool:
        """Whether the untruncated DFs are a valid equilibrium as they stand."""
        return self.k_t_min == 0.0

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        """Whether the DF of each component diverges as the energy goes to zero.

        See `AsymptoticForm.diverges_at_zero`.
        """
        return self.form.diverges_at_zero

    def to(
        self,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "DistributionFunction":
        """Return the same result presented in another unit system.

        The numbers of the result are not recomputed: they stay in internal
        units, and what is computed from the result inherits its units.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            The target unit system.
        concentration : float, optional
            Concentration that defines M_0, the mass inside that many scale
            radii. Taken from the truncation where it was given there.
        mass_0 : float or astropy Quantity, optional
            M_0, for the physical system; a float is in solar masses.
        r_0, r_vir : float or astropy Quantity, optional
            The scale radius, or the concentration radius in its place, for
            the physical system; a float is in parsecs.

        Returns
        -------
        df : DistributionFunction
        """
        units = Units.build(system, concentration, mass_0, r_0, r_vir)
        return dataclasses.replace(self, units=units)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the result to a file; `pianistpy.read` reads it back."""
        _write_df(self, path, overwrite)

    def write_text(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the DF table as fixed-width text, an export beside the file.

        One row per node of each component's support: the energy ``k`` (in
        units of the central potential, internal units), ``log10_df`` and
        the slope ``dlog10_df_dlnk`` of the table, and with more than one
        component its index in a ``component`` column. The header names the
        result, its components and the units. The file is read back and
        compared with what was written, to the digits written
        (`pianistpy.io.write_table`).
        """
        _write_df_table_text(self, path, overwrite)


@dataclasses.dataclass(frozen=True, eq=False)
class Model:
    """A valid equilibrium: truncated or untruncated, of one or more components.

    Returned by `truncate` and `solve`. A model always has non-negative,
    non-decreasing DFs. Whether its density ends at a finite radius is a
    property of the model, `spatially_finite`.

    Parameters
    ----------
    components : tuple of Component
        The components of the model.
    truncation : Truncation or None
        The truncation, or None for an untruncated model.
    settings : Settings
        The settings the model was computed with.
    identity : str
        Identity string of the result.
    parent : str
        Identity string of the `DistributionFunction` the model was built from.
    table : DfTable
        The DFs of the model.
    form : AsymptoticForm
        The analytic form of each DF beyond the ends of its table.
    profile : TruncatedProfile
        The self-consistent profile of the model.
    validation : ValidationReport
        The report of the model.
    timing : dict of str to float
        Wall time of each stage, in seconds.
    r_trust : mapping of str to float
        The radius within which the model holds each accuracy tier, keyed by
        the report's tier names ("goal", "ceiling"; "limit" where it is
        computed): where the result is valid, not an error of it. Every tier
        or none: the radii are measured on an untruncated model, from the
        offset of its potential at large radii, and the mapping is empty on
        a truncated model (a spatially finite one holds its table to its
        edge) and where that offset is undefined. A tier the table's first
        outer row already misses has radius 0; infinite on a closed form.
    inversion : InversionRecord or None
        How the model's table was made; None where it was not made by an
        inversion (a closed form). Every solved model states it.
    """

    components: tuple[Component, ...]
    truncation: Truncation | None
    settings: Settings
    identity: str
    parent: str
    table: DfTable
    form: AsymptoticForm
    profile: TruncatedProfile
    validation: ValidationReport
    timing: dict[str, float]
    r_trust: Mapping[str, float]
    inversion: InversionRecord | None
    units: Units = Units()

    @property
    def spatially_finite(self) -> bool:
        """Whether the density reaches zero at a finite radius and mass."""
        return self.profile.spatially_finite

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        """Whether the DF of each component diverges as the energy goes to zero.

        See `AsymptoticForm.diverges_at_zero`. Such a model is a valid
        equilibrium with a finite density; its DF is not monotonic.
        """
        return self.form.diverges_at_zero

    def to(
        self,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "Model":
        """Return the same model presented in another unit system.

        The numbers of the model are not recomputed: they stay in internal
        units, and what is computed from the model inherits its units.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            The target unit system.
        concentration : float, optional
            Concentration that defines M_0, the mass inside that many scale
            radii. Taken from the truncation where it was given there.
        mass_0 : float or astropy Quantity, optional
            M_0, for the physical system; a float is in solar masses.
        r_0, r_vir : float or astropy Quantity, optional
            The scale radius, or the concentration radius in its place, for
            the physical system; a float is in parsecs.

        Returns
        -------
        model : Model
        """
        truncation = self.truncation
        given = None if truncation is None else truncation.concentration
        if concentration is None:
            concentration = given
        elif given is not None and concentration != given:
            raise InputError(
                f"The model was truncated with concentration {given}; its units "
                f"cannot use {concentration}."
            )
        units = Units.build(system, concentration, mass_0, r_0, r_vir)
        return dataclasses.replace(self, units=units)

    def radial_profiles(self) -> dict[str, Any]:
        """Return the radial profiles of the model in its unit system.

        Returns
        -------
        profiles : dict
            ``"r"``, ``"potential"`` (the relative potential), ``"density"``
            (one row per component) and ``"mass"`` (the total enclosed
            mass): plain arrays in the internal and the model systems,
            Astropy quantities in the physical system.
        """
        profile = self.components[0].unit_profile
        units = self.units
        return {
            "r": units.convert("length", self.profile.r, profile),
            "potential": units.convert("potential", self.profile.p_trunc, profile),
            "density": units.convert("density", self.profile.rho_trunc, profile),
            "mass": units.convert("mass", self.profile.mass_trunc, profile),
        }

    @property
    def mass_t(self) -> Array:
        """Total mass of the model, in its unit system."""
        profile = self.components[0].unit_profile
        return self.units.convert("mass", self.profile.mass_t, profile)

    @property
    def r_t(self) -> Array:
        """Outer radius of the model, in its unit system."""
        profile = self.components[0].unit_profile
        return self.units.convert("length", self.profile.r_t, profile)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the result to a file; `pianistpy.read` reads it back."""
        _write_model(self, path, overwrite)

    def write_text(
        self, path: str | os.PathLike, table: str = "profile", overwrite: bool = False
    ) -> None:
        """Write a table of the model as fixed-width text, an export beside the file.

        Parameters
        ----------
        path : path
            File to write.
        table : {"profile", "df"}
            ``"profile"``: the radial profiles in the model's unit system
            (`radial_profiles`): ``r``, ``potential``, the ``density`` of
            each component (``density_<name>`` with more than one) and the
            enclosed ``mass``; the header states the unit of each column.
            ``"df"``: the DF table, as `DistributionFunction.write_text`
            writes it.
        overwrite : bool, optional
            Replace an existing file.

        Raises
        ------
        InputError
            For a table other than the two, or an existing file without
            ``overwrite``.
        ResultError
            If the file does not read back as written.
        """
        if table == "profile":
            _write_profile_text(self, path, overwrite)
        elif table == "df":
            _write_df_table_text(self, path, overwrite)
        else:
            raise InputError(f"table must be 'profile' or 'df'; got {table!r}.")


def _pack(stage: Any, prefix: str, arrays: dict, record: dict) -> None:
    """Split a stage result into its arrays and its other fields."""
    fields: dict[str, Any] = {}
    for item in dataclasses.fields(stage):
        value = getattr(stage, item.name)
        if isinstance(value, np.ndarray):
            arrays[f"{prefix}/{item.name}"] = value
        elif isinstance(value, (DoublePowerLaw, Anisotropy)):
            fields[item.name] = value.to_dict()
        else:
            fields[item.name] = value
    record[prefix] = fields


def _tuples(value: Any) -> Any:
    """Return a value read from a file with its lists turned back into tuples."""
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _unpack(cls: type, prefix: str, stored: StoredResult) -> Any:
    """Rebuild a stage result from the arrays and the record of a file."""
    values = {key: _tuples(value) for key, value in stored.record[prefix].items()}
    for item in dataclasses.fields(cls):
        key = f"{prefix}/{item.name}"
        # a field a file written before it existed does not hold keeps its default
        if item.name not in values and key in stored.arrays:
            values[item.name] = stored.arrays[key]
    if cls in (DfTable, EnergyGrid) and np.ndim(values.get("k", 0.0)) == 1:
        # the layout before the energy grid was per component (2026-10-08)
        raise ResultError(
            f"The file's {prefix} holds its energies without the component axis: "
            "a file of the previous layout; compute the result again."
        )
    if cls is TruncatedProfile:
        values["source"] = DoublePowerLaw.from_dict(values["source"])
        values["source_anisotropy"] = Anisotropy.from_dict(values["source_anisotropy"])
    return cls(**values)


def _component_prefix(index: int) -> str:
    """Where a file holds the truncated profile a component is built on."""
    return f"component_profile_{index}"


def _pack_components(
    components: tuple[Component, ...], arrays: dict, record: dict
) -> None:
    """Add the truncated profiles components are built on to a file's content."""
    for index, component in enumerate(components):
        if isinstance(component.profile, TruncatedProfile):
            _pack(component.profile, _component_prefix(index), arrays, record)


def _unpack_components(stored: StoredResult) -> tuple[Component, ...]:
    """Rebuild the components of a result, with the profiles they are built on."""
    components = []
    for index, item in enumerate(stored.record["components"]):
        profile = None
        if _component_prefix(index) in stored.record:
            profile = _unpack(TruncatedProfile, _component_prefix(index), stored)
        components.append(Component.from_dict(item, profile=profile))
    return tuple(components)


def _unpack_inversion(stored: StoredResult) -> InversionRecord | None:
    """The inversion record of a file; None where the result is a closed form."""
    if stored.record["inversion"] is None:
        return None
    return _unpack(InversionRecord, "inversion", stored)


def _pack_inversion(
    result: DistributionFunction | Model, arrays: dict, record: dict
) -> None:
    if result.inversion is None:
        record["inversion"] = None
    else:
        _pack(result.inversion, "inversion", arrays, record)


def _common_record(result: DistributionFunction | Model) -> dict[str, Any]:
    return {
        "components": [component.to_dict() for component in result.components],
        "settings": result.settings.to_dict(),
        "validation": result.validation.to_dict(),
        "timing": dict(result.timing),
        "units": result.units.to_dict(),
    }


def _write_df(
    df: DistributionFunction, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write a distribution function to a file.

    Parameters
    ----------
    df : DistributionFunction
        The result to write.
    path : path
        File to write.
    overwrite : bool, optional
        Replace an existing file.
    """
    arrays: dict[str, np.ndarray] = {}
    record = _common_record(df)
    record["k_t_min"] = df.k_t_min
    _pack_components(df.components, arrays, record)
    _pack(df.grid, "grid", arrays, record)
    _pack(df.table, "table", arrays, record)
    _pack(df.form, "form", arrays, record)
    _pack_inversion(df, arrays, record)
    if df.rows is None:
        record["rows"] = None
    else:
        _pack(df.rows, "rows", arrays, record)
    write_result(path, DF_KIND, df.identity, record, arrays, overwrite=overwrite)


def _write_model(
    model: Model, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write a model to a file.

    Parameters
    ----------
    model : Model
        The result to write.
    path : path
        File to write.
    overwrite : bool, optional
        Replace an existing file.
    """
    arrays: dict[str, np.ndarray] = {}
    record = _common_record(model)
    truncation = model.truncation
    record["truncation"] = None if truncation is None else truncation.to_dict()
    record["r_trust"] = {tier: float(r) for tier, r in model.r_trust.items()}
    _pack_components(model.components, arrays, record)
    _pack(model.table, "table", arrays, record)
    _pack(model.form, "form", arrays, record)
    _pack(model.profile, "profile", arrays, record)
    _pack_inversion(model, arrays, record)
    write_result(
        path,
        MODEL_KIND,
        model.identity,
        record,
        arrays,
        parents=(model.parent,),
        overwrite=overwrite,
    )


def _plain(value: Any) -> np.ndarray:
    """The numbers of a converted quantity: the array of an Astropy quantity."""
    return np.asarray(getattr(value, "value", value), dtype=np.float64)


def _unit_label(value: Any, units: Units) -> str:
    """The unit of a converted quantity, as the sampler labels it."""
    return str(getattr(value, "unit", units.system))


def _text_head(result: DistributionFunction | Model, what: str) -> list[str]:
    head = [f"PIANISTpy {what}", f"identity: {result.identity}"]
    parent = getattr(result, "parent", None)
    if parent is not None:
        head.append(f"parent: {parent}")
    head.append("components: " + ", ".join(c.name for c in result.components))
    return head


def _write_df_table_text(
    result: DistributionFunction | Model, path: str | os.PathLike, overwrite: bool
) -> None:
    table = result.table
    multi = len(result.components) > 1
    parts: dict[str, list[np.ndarray]] = {
        "k": [],
        "log10_df": [],
        "dlog10_df_dlnk": [],
        "component": [],
    }
    for i in range(len(result.components)):
        support = slice(int(table.i_first[i]), int(table.i_last[i]) + 1)
        parts["k"].append(np.asarray(table.k[i, support], dtype=np.float64))
        parts["log10_df"].append(np.asarray(table.log_df[i, support], dtype=np.float64))
        parts["dlog10_df_dlnk"].append(
            np.asarray(table.log_df_slope[i, support], dtype=np.float64)
        )
        parts["component"].append(np.full(support.stop - support.start, i))
    columns = {name: np.concatenate(parts[name]) for name in parts}
    if not multi:
        del columns["component"]
    head = _text_head(result, "distribution function table") + [
        "units: internal; k is the energy-like variable in units of the central "
        "potential, log10_df the base-10 logarithm of the DF, dlog10_df_dlnk its "
        "slope d log10 F / d ln k; minus infinity where the DF is zero"
    ]
    if multi:
        head.append("component: index into the components above")
    write_table(path, columns, head, overwrite=overwrite)


def _write_profile_text(model: Model, path: str | os.PathLike, overwrite: bool) -> None:
    profiles = model.radial_profiles()
    units = model.units
    density = profiles["density"]
    columns: dict[str, np.ndarray] = {
        "r": _plain(profiles["r"]),
        "potential": _plain(profiles["potential"]),
    }
    labels = {
        "r": _unit_label(profiles["r"], units),
        "potential": _unit_label(profiles["potential"], units),
    }
    rows = _plain(density)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    if len(model.components) == 1:
        columns["density"] = rows[0]
        labels["density"] = _unit_label(density, units)
    else:
        for component, row in zip(model.components, rows, strict=True):
            columns[f"density_{component.name}"] = row
            labels[f"density_{component.name}"] = _unit_label(density, units)
    columns["mass"] = _plain(profiles["mass"])
    labels["mass"] = _unit_label(profiles["mass"], units)
    head = _text_head(model, "radial profile") + [
        f"unit_system: {units.system}",
        "units: " + " | ".join(f"{name}: {labels[name]}" for name in columns),
    ]
    write_table(path, columns, head, overwrite=overwrite)


def _check_kind(stored: StoredResult, kind: str) -> None:
    if stored.kind != kind:
        raise ResultError(f"The result is of kind {stored.kind!r}, not {kind!r}.")


def _read_df(source: StoredResult | str | os.PathLike) -> DistributionFunction:
    """Return a distribution function from a file or a stored result.

    Parameters
    ----------
    source : StoredResult or path
        A stored result, or the path of a file written by `_write_df`.

    Returns
    -------
    df : DistributionFunction
    """
    stored = load_result(source)
    _check_kind(stored, DF_KIND)
    record = stored.record
    return DistributionFunction(
        components=_unpack_components(stored),
        settings=Settings.from_dict(record["settings"]),
        identity=stored.identity,
        grid=_unpack(EnergyGrid, "grid", stored),
        table=_unpack(DfTable, "table", stored),
        form=_unpack(AsymptoticForm, "form", stored),
        k_t_min=float(record["k_t_min"]),
        validation=ValidationReport.from_dict(record["validation"]),
        timing=dict(record["timing"]),
        units=Units.from_dict(record.get("units")),
        inversion=_unpack_inversion(stored),
        rows=(
            None
            if record.get("rows") is None
            else _unpack(InversionRows, "rows", stored)
        ),
    )


def _read_model(source: StoredResult | str | os.PathLike) -> Model:
    """Return a model from a file or a stored result.

    Parameters
    ----------
    source : StoredResult or path
        A stored result, or the path of a file written by `_write_model`.

    Returns
    -------
    model : Model
    """
    stored = load_result(source)
    _check_kind(stored, MODEL_KIND)
    record = stored.record
    truncation = record["truncation"]
    return Model(
        components=_unpack_components(stored),
        truncation=None if truncation is None else Truncation.from_dict(truncation),
        settings=Settings.from_dict(record["settings"]),
        identity=stored.identity,
        parent=stored.parents[0],
        table=_unpack(DfTable, "table", stored),
        form=_unpack(AsymptoticForm, "form", stored),
        profile=_unpack(TruncatedProfile, "profile", stored),
        validation=ValidationReport.from_dict(record["validation"]),
        timing=dict(record["timing"]),
        units=Units.from_dict(record.get("units")),
        r_trust=dict(record["r_trust"]),
        inversion=_unpack_inversion(stored),
    )


register_reader(DF_KIND, _read_df)
register_reader(MODEL_KIND, _read_model)


# The results as pytrees: the inputs and the arrays are the data (so that a
# function of a model can be traced, mapped and differentiated in them); the
# settings, the identities, the reports and the records are static.
register_result(
    DistributionFunction, ("components", "grid", "table", "form", "k_t_min", "rows")
)
register_result(Model, ("components", "truncation", "table", "form", "profile"))
