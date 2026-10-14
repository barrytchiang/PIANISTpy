"""Numerical settings: the tested tiers and expert overrides."""

__all__ = ["FIELDS", "GROUPS", "TIERS", "Field", "Settings"]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .exceptions import InputError
from .validation import CEILING, GOAL

TIERS = ("default", "quick")
GROUPS = ("inversion", "truncation", "sampling", "observables", "losvd")


@dataclass(frozen=True)
class Field:
    """One numerical setting: its value in each tier and what it controls.

    Parameters
    ----------
    default : bool, int, float or str
        Value in the default tier. Its type is the type of the field.
    doc : str
        What the field controls.
    quick : bool, int, float or str, optional
        Value in the quick tier. If omitted, the quick tier keeps the default
        value: a cheaper value enters only once it has passed the tests.
    """

    default: Any
    doc: str
    quick: Any = None

    def value(self, tier: str) -> Any:
        """Return the value of the field in a tier."""
        if tier == "quick" and self.quick is not None:
            return self.quick
        return self.default


# Every field of every group. A stage reads its numerical settings from here
# and from nowhere else; a field is added when the stage that reads it is
# brought in.
FIELDS: dict[str, dict[str, Field]] = {
    "inversion": {
        "n_nodes": Field(500, "Gauss-Legendre nodes of the inversion integral."),
        "n_grid_rows": Field(
            1000, "Rows of the linear middle zone of the inversion's energy grid."
        ),
        "n_grid_tail_rows": Field(
            3000,
            "Rows of each logarithmic end zone of the inversion's energy grid, "
            "toward the floor and toward the top.",
        ),
        "grid_tail_extent": Field(
            0.03,
            "Extent of the energy grid's end zones, as a fraction of the central "
            "potential.",
        ),
        "table_nodes_per_decade": Field(
            50,
            "Nodes per decade of the energy grid below its floor, down to the "
            "table's floor.",
        ),
        "n_floor_nodes": Field(
            240,
            "Gauss-Legendre nodes in ln P of the completion below the table's "
            "floor (a third of them, at least 40, near the floor).",
        ),
    },
    "truncation": {
        "n_source_nodes": Field(
            500,
            "Gauss-Legendre nodes of each of the two intervals of the quadrature "
            "of the auxiliary density on the source grid.",
        ),
        "n_source_rows": Field(12000, "Rows of the source grid in the potential."),
        "n_source_tail_rows": Field(
            3000,
            "Rows of each logarithmic tail of the source grid, toward the floor "
            "and toward the top (at most a third of the rows).",
        ),
        "source_tail_extent": Field(
            0.03,
            "Largest extent of the source grid's tails, as a fraction of the "
            "grid's top (the tails take at most a hundredth of the top).",
        ),
        "source_tail_floor": Field(
            1.204013866384912e-08,
            "Smallest start of the source grid's lower tail and smallest extent "
            "of its upper tail, as fractions of the grid's top.",
        ),
        "quadrature_split": Field(
            0.9,
            "Fraction of the potential where the first interval of the source "
            "quadrature ends.",
        ),
        "quadrature_stretch": Field(
            2.0,
            "Power k of the second interval's map Q = P - u^k, where the kernel's "
            "exponent is not negative.",
        ),
        "ode_rtol": Field(1e-11, "Relative tolerance of the Poisson integration."),
        "ode_atol": Field(1e-11, "Absolute tolerance of the Poisson integration."),
        "n_rows_inner": Field(
            500, "Rows of the Poisson solve's grid inside the inner zone."
        ),
        "n_rows_outer": Field(
            59500, "Rows of the Poisson solve's grid outside the inner zone."
        ),
        "log10_r_min": Field(
            -9.0, "Logarithm of the Poisson solve's innermost radius."
        ),
        "log10_r_max": Field(
            12.0,
            "Logarithm of the Poisson solve's outermost radius; a run raises it "
            "until the potential there is below half the floor, and the result "
            "states the value it used.",
        ),
        "continuation_start": Field(
            1e-2,
            "Fraction of the central potential below which a truncated model's "
            "last rows are re-solved on a grid geometric in R_t - r.",
        ),
        "continuation_rows_per_decade": Field(
            1200, "Rows per decade of R_t - r of the edge continuation."
        ),
        "continuation_floor_fraction": Field(
            2e-2, "The edge continuation's last row, as a fraction of the floor."
        ),
        "profile_bin_decades": Field(
            0.1, "Width in decades of the potential of the profile's row-count bins."
        ),
        "profile_rows_per_decade": Field(
            1000,
            "Fewest rows per decade of the potential (and of the depth) that the "
            "profile keeps at its two ends; the fills supply the rest.",
        ),
    },
    "sampling": {
        "n_rows": Field(
            512,
            "Rows in potential of the speed table to start from: the rows are "
            "refined until the table's own error across rows is under the goal.",
        ),
        "n_rows_max": Field(
            4075,
            "Most rows of the speed table: no refinement passes this count.",
        ),
        "n_nodes": Field(2048, "Nodes of each row of the speed table.", quick=512),
        "n_speed_intervals": Field(
            200,
            "Intervals in the scaled speed of the distribution integrated on each "
            "row of the speed table.",
        ),
        "n_gauss_legendre": Field(
            16,
            "Gauss-Legendre nodes of each panel of the speed integrals.",
            quick=8,
        ),
        "panels_per_decade": Field(
            8,
            "Panels per decade of the local variable where the speed integrals "
            "refine toward the two ends of a row and inside each interval.",
        ),
        "speed_low_refined": Field(
            1e-9,
            "Smallest scaled speed refined in panels; below it the density is its "
            "power law in the speed, integrated in closed form.",
        ),
        "edge_refined": Field(
            1e-12,
            "Distance from the edge of the speed support, relative to the edge, "
            "down to which panels refine; below it the DF's rule below its table "
            "is integrated in closed form.",
        ),
        "radius_edge_refined": Field(
            1e-10,
            "Smallest enclosed-mass fraction beyond a radius that the radius table "
            "of a model with a sharp edge resolves in nodes; beyond it the mass "
            "follows its power law in the distance from the edge.",
        ),
        "n_radius_nodes": Field(2048, "Nodes of the radius table."),
        "n_angle_nodes": Field(
            1024,
            "Nodes of each half of the angle table, uniform in the logarithm of the "
            "cumulative probability from its end.",
        ),
        "draw_block": Field(
            262144,
            "Particles drawn at once: a draw runs over blocks of this many, so "
            "its working memory is one block's (about 0.75 kB a particle, 2**18 "
            "particles about 0.2 GB) above its outputs, whatever the count; the "
            "particles are the same at any block size.",
        ),
        "row_error_goal": Field(
            GOAL,
            "Error across rows at which the speed table's refinement stops (its "
            "rows double until the table's own error is under it, or n_rows_max "
            "is reached).",
            quick=CEILING,
        ),
        "tail_mass_min": Field(
            1e-18,
            "Smallest tabulated cumulative probability at either end of a table: a "
            "uniform double reaches 2**-53, so a draw never reads beyond it.",
        ),
    },
    "observables": {
        "n_radii": Field(
            10000,
            "Projected radii of the projection table before its certificate adds "
            "rows, logarithmic from the innermost row to the edge.",
            quick=500,
        ),
        "profile_tolerance": Field(
            GOAL / 10,
            "Relative error of sigma_r at which the radial dispersion's rows stop "
            "being added: every interval's midpoint reads within it. A tenth of the "
            "goal, the profile's share of the error of what is projected from it.",
            quick=CEILING / 100,
        ),
        "n_profile_start": Field(
            100,
            "Logarithmic radii the radial dispersion starts from, from the innermost "
            "row to the edge, before rows are added to its tolerance.",
        ),
        "n_jeans_nodes": Field(
            500,
            "Gauss-Legendre nodes of a row's own Jeans integral of the radial "
            "dispersion (the superposition's); the observables' profile sums "
            "panels between its rows instead, their nodes doubled until "
            "sigma_r moves by less than profile_tolerance.",
        ),
        "n_projection_nodes": Field(
            500,
            "Gauss-Legendre nodes of each line-of-sight integral of the surface "
            "density and the projected dispersion.",
        ),
        "n_aperture_nodes": Field(
            100,
            "Gauss-Legendre nodes of each panel of an aperture's radial integral, "
            "at least.",
        ),
        "aperture_nodes_per_decade": Field(
            8,
            "Nodes per decade of a panel's span in the logarithm of the projected "
            "radius where that exceeds the minimum, rounded up to a multiple of 20.",
        ),
        "rows_route": Field(
            "kernel",
            "The angle rule of the line-of-sight velocity distributions' rows: "
            '"kernel", the JAX kernel, or "numpy", the reference route.',
        ),
    },
    "losvd": {
        "rows_per_decade": Field(
            55,
            "Base rows of the table of line-of-sight velocity distributions per "
            "decade of projected radius, from the innermost row of the profile.",
            quick=20,
        ),
        "n_energy_nodes": Field(
            8, "Gauss-Legendre nodes of each panel of the energy integral of a node."
        ),
        "n_floor_layers": Field(
            4,
            "Panels of the energy integral graded toward the energy of the DF's floor.",
            quick=2,
        ),
        "energy_layer_ratio": Field(
            0.2, "Ratio of successive graded panels of the energy integral."
        ),
        "n_angle_nodes": Field(
            64,
            "Gauss-Legendre nodes of the angle integral of an L factor, the most a "
            "node takes.",
        ),
        "n_sightline_nodes": Field(
            16,
            "Gauss-Legendre nodes of each panel of the line-of-sight integral.",
            quick=8,
        ),
        "n_sightline_panels": Field(
            8, "Panels of each interval of the support along the line of sight."
        ),
        "sightline_panels_refined": Field(
            6,
            "Doublings of the line-of-sight panels allowed before a row is "
            "recorded as not converged.",
        ),
        "sightline_layer_ratio": Field(
            0.3, "Ratio of the line-of-sight panels graded toward the axis."
        ),
        "n_sightline_layers": Field(
            4, "Line-of-sight panels graded toward the axis added at each doubling."
        ),
        "n_crossing_scan": Field(
            2001, "Points of the scan for where the angular momentum vanishes."
        ),
        "n_cumulative_nodes": Field(
            200001,
            "Nodes of each of the three branches of the table of the DF's cumulative "
            "integral.",
        ),
        "n_velocity_nodes": Field(
            16, "Gauss-Legendre nodes of each velocity sub-panel of a row.", quick=8
        ),
        "velocity_layer_ratio": Field(
            0.3, "Ratio of the velocity sub-panels graded toward a break."
        ),
        "n_velocity_layers_zero": Field(
            12, "Velocity sub-panels graded toward zero velocity."
        ),
        "n_velocity_layers_break": Field(
            8, "Velocity sub-panels graded toward each break of a row."
        ),
        "n_break_scan": Field(
            4001, "Points of the coarse scan for a velocity break.", quick=1001
        ),
        "n_break_zooms": Field(4, "Zooms of the scan for a velocity break."),
        "n_bisections": Field(
            60,
            "Bisections of an edge of the support or of a root along a line of sight.",
        ),
        "n_radius_break_layers": Field(
            16, "Rows added toward each radius where the layout of the rows changes."
        ),
        "n_radius_break_bisections": Field(
            40, "Bisections in projected radius of a change of the layout."
        ),
        "n_radius_break_samples": Field(
            3, "Samples inside each interval of base rows for a change of the layout."
        ),
        "n_edge_samples": Field(
            10, "Samples beyond the last base row toward the end of the profile."
        ),
        "n_edge_layers": Field(16, "Rows added toward the end of the table."),
        "n_stencil_rows": Field(
            8, "Rows of the stencil that reads the table between its rows."
        ),
        "n_stencil_test_rows": Field(
            8, "Rows of the stencil whose estimate certifies the table."
        ),
        "n_block_elements": Field(
            250000,
            "Elements of one block of node pairs, energy nodes and angle nodes, which "
            "bounds the memory of a row.",
        ),
    },
}


def _checked(group: str, name: str, value: Any) -> Any:
    """Return an override in the type of its field, or refuse it."""
    fields = FIELDS[group]
    if name not in fields:
        raise InputError(
            f"Unknown field {name!r} in settings group {group!r}; "
            f"its fields are {sorted(fields)}."
        )
    kind = type(fields[name].default)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not kind:
        raise InputError(
            f"{group}.{name} takes a value of type {kind.__name__}; got {value!r}."
        )
    if kind in (int, float) and not name.startswith("log10_") and not value > 0:
        raise InputError(f"{group}.{name} must be positive; got {value!r}.")
    return value


class Settings:
    """Numerical settings of a computation.

    A tier is a complete set of numerical settings that has been run through
    the test battery and for which the package states the accuracy it
    delivers. An expert may override single fields; the result then records
    that it was not made with a tested tier and names the fields that differ.

    Parameters
    ----------
    quick : bool, optional
        Use the quick tier instead of the default tier.
    write_anyway : bool, optional
        Return a result whose own check lies beyond the hard limit instead of
        refusing it. The result carries the check's verdict in its validation
        report (tier ``"refused"``) and an `AccuracyWarning` is issued in
        place of the refusal. For debugging a case the package cannot serve
        at its stated accuracy; off by default. A physically invalid model
        is refused regardless.
    **overrides : mapping
        One mapping per group (``inversion``, ``truncation``, ``sampling``,
        ``observables``) of the fields to change within that group.

    Attributes
    ----------
    tier : str
        Name of the tier the settings start from.
    write_anyway : bool
        Whether a result beyond the hard limit is returned instead of refused.
    overrides : mapping
        The fields changed from the tier, by group. Read only.

    Examples
    --------
    >>> Settings().tier
    'default'
    >>> Settings(quick=True).is_tested_tier
    True
    >>> Settings(inversion={"n_nodes": 256}).is_tested_tier
    False
    >>> Settings(inversion={"n_nodes": 256}).inversion["n_nodes"]
    256
    >>> Settings().truncation["ode_rtol"]
    1e-11
    """

    __slots__ = ("_overrides", "_tier", "_write_anyway")
    _overrides: Mapping[str, Mapping[str, Any]]
    _tier: str
    _write_anyway: bool

    def __init__(
        self,
        quick: bool = False,
        write_anyway: bool = False,
        **overrides: Mapping[str, Any],
    ) -> None:
        if not isinstance(write_anyway, bool):
            raise InputError(f"write_anyway is True or False; got {write_anyway!r}.")
        unknown = sorted(set(overrides) - set(GROUPS))
        if unknown:
            raise InputError(
                f"Unknown settings group(s) {unknown}; the groups are {list(GROUPS)}."
            )
        frozen = {
            group: MappingProxyType(
                {name: _checked(group, name, value) for name, value in fields.items()}
            )
            for group, fields in sorted(overrides.items())
            if fields
        }
        object.__setattr__(self, "_tier", "quick" if quick else "default")
        object.__setattr__(self, "_write_anyway", write_anyway)
        object.__setattr__(self, "_overrides", MappingProxyType(frozen))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Settings are immutable; build a new Settings object.")

    @property
    def tier(self) -> str:
        """Name of the tier the settings start from."""
        return self._tier

    @property
    def write_anyway(self) -> bool:
        """Whether a result beyond the hard limit is returned instead of refused."""
        return self._write_anyway

    @property
    def overrides(self) -> Mapping[str, Mapping[str, Any]]:
        """The fields changed from the tier, by group."""
        return self._overrides

    @property
    def is_tested_tier(self) -> bool:
        """Whether the settings are a tested tier with no field changed."""
        return not self._overrides

    def group(self, name: str) -> Mapping[str, Any]:
        """Return the value of every field of one group.

        Parameters
        ----------
        name : str
            One of the groups ``inversion``, ``truncation``, ``sampling`` and
            ``observables``.

        Returns
        -------
        values : mapping
            The fields of the group with the values of the tier, each
            replaced by its override where one was given. Read only.
        """
        if name not in GROUPS:
            raise InputError(
                f"Unknown settings group {name!r}; the groups are {list(GROUPS)}."
            )
        values = {key: field.value(self._tier) for key, field in FIELDS[name].items()}
        values.update(self._overrides.get(name, {}))
        return MappingProxyType(values)

    @property
    def inversion(self) -> Mapping[str, Any]:
        """Settings of the inversion."""
        return self.group("inversion")

    @property
    def truncation(self) -> Mapping[str, Any]:
        """Settings of the truncation."""
        return self.group("truncation")

    @property
    def sampling(self) -> Mapping[str, Any]:
        """Settings of the particle sampling."""
        return self.group("sampling")

    @property
    def observables(self) -> Mapping[str, Any]:
        """Settings of the observables."""
        return self.group("observables")

    @property
    def losvd(self) -> Mapping[str, Any]:
        """The fields of the LOSVD table and the aperture LOSVD."""
        return self.group("losvd")

    def resolved(self) -> dict[str, dict[str, Any]]:
        """Return every field of every group with the value in force."""
        return {name: dict(self.group(name)) for name in GROUPS}

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dictionary.

        Returns
        -------
        record : dict
            The tier and the overrides, in the form written to files, with
            ``"write_anyway": True`` where that was asked. The numerical
            settings alone determine a result, so `numerical` is what the
            identity hashes.
        """
        record = self.numerical()
        if self._write_anyway:
            record["write_anyway"] = True
        return record

    def numerical(self) -> dict[str, Any]:
        """Return the tier and the overrides, the settings that determine a result."""
        return {
            "tier": self._tier,
            "overrides": {
                group: dict(fields) for group, fields in self._overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Settings":
        """Build settings from their dictionary form.

        Parameters
        ----------
        record : mapping
            A dictionary as returned by `to_dict`.

        Returns
        -------
        settings : Settings
        """
        tier = record.get("tier", "default")
        if tier not in TIERS:
            raise InputError(f"Unknown tier {tier!r}; the tiers are {list(TIERS)}.")
        return cls(
            quick=(tier == "quick"),
            write_anyway=bool(record.get("write_anyway", False)),
            **record.get("overrides", {}),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Settings):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        parts = ["quick=True"] if self._tier == "quick" else []
        parts += ["write_anyway=True"] if self._write_anyway else []
        parts += [
            f"{group}={dict(fields)!r}" for group, fields in self._overrides.items()
        ]
        return f"Settings({', '.join(parts)})"
