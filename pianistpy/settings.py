"""The numerical settings of a computation: the default settings, the quick
settings, and single fields changed by an expert.

A user passes a `Settings` object to a computation, for example
``pp.solve(component, settings=pp.Settings(quick=True))``. The default
settings and the quick settings are the two tested settings levels: the
package states the accuracy of each level from its test suite. The quick
settings use fewer nodes and rows than the default settings and hold every
check of a result to the same bounds, :math:`10^{-6}`, :math:`10^{-4}` and
:math:`10^{-2}`. The manual's settings tables list every field with its
value at each level.
"""

__all__ = ["FIELDS", "GROUPS", "LEVELS", "Field", "Settings"]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .exceptions import InputError
from .validation import TIER_BOUNDS

# The names of the two tested settings levels.
LEVELS = ("default", "quick")
# The names of the groups of fields, one group for each part of a computation.
GROUPS = ("inversion", "truncation", "sampling", "observables", "losvd")


@dataclass(frozen=True)
class Field:
    """One numerical setting: its value at each settings level and a
    description of what the setting controls.

    Parameters
    ----------
    default : bool, int, float or str
        The field's value in the default settings. The type of `default` is
        the type every value of the field must have.
    doc : str
        A description of what the field controls, as the manual's settings
        tables print it.
    quick : bool, int, float or str, optional
        The field's value in the quick settings. Default: ``None``, which
        means the quick settings use the value `default`.

    See Also
    --------
    Settings : The settings of a computation.
    FIELDS : Every field of every group.

    Examples
    --------
    >>> field = FIELDS["sampling"]["n_nodes"]
    >>> field.value("default"), field.value("quick")
    (2048, 512)
    """

    default: Any
    doc: str
    quick: Any = None

    def value(self, level: str) -> Any:
        """Return the field's value at a settings level.

        Parameters
        ----------
        level : {"default", "quick"}
            The settings level.

        Returns
        -------
        value : bool, int, float or str
            The field's value in the quick settings if `level` is
            ``"quick"`` and the field has a quick value; the field's value
            in the default settings otherwise.
        """
        if level == "quick" and self.quick is not None:
            return self.quick
        return self.default


# Every field of every group, keyed by group name and then by field name.
# Every numerical setting of the package is a field of FIELDS, so that the
# manual's settings tables, which are generated from FIELDS, list every
# setting a computation reads.
FIELDS: dict[str, dict[str, Field]] = {
    "inversion": {
        "n_nodes": Field(
            500,
            (
                "The number of Gauss-Legendre nodes of the integral that inverts "
                "the density into the distribution function."
            ),
        ),
        "n_grid_rows": Field(
            1000,
            (
                "The number of energies in the middle zone of the inversion's "
                "energy grid, where the energies are spaced linearly."
            ),
        ),
        "n_grid_tail_rows": Field(
            3000,
            (
                "The number of energies in each of the two end zones of the "
                "inversion's energy grid, where the energies are spaced "
                "logarithmically toward the grid's floor and toward the grid's top."
            ),
        ),
        "grid_tail_extent": Field(
            0.03,
            (
                "The width of each end zone of the inversion's energy grid, as a "
                "fraction of the central potential."
            ),
        ),
        "table_nodes_per_decade": Field(
            50,
            (
                "The number of energies per decade below the floor of the "
                "inversion's energy grid, down to the floor of the distribution "
                "function's table."
            ),
        ),
        "n_floor_nodes": Field(
            240,
            (
                "The number of Gauss-Legendre nodes, in ln P, of the integral that "
                "completes the distribution function below the table's floor; a "
                "third of the nodes, and at least 40 nodes, lie near the floor."
            ),
        ),
    },
    "truncation": {
        "n_source_nodes": Field(
            500,
            (
                "The number of Gauss-Legendre nodes in each of the two intervals of"
                " the quadrature that computes the auxiliary density on the source "
                "grid."
            ),
        ),
        "n_source_rows": Field(
            12000, "The number of potential values of the source grid."
        ),
        "n_source_tail_rows": Field(
            3000,
            (
                "The number of potential values in each logarithmic tail of the "
                "source grid, toward the grid's floor and toward the grid's top; "
                "each tail holds at most a third of the source grid's potential "
                "values."
            ),
        ),
        "source_tail_extent": Field(
            0.03,
            (
                "The largest width of each tail of the source grid, as a fraction "
                "of the grid's top; the width used is at most a hundredth of the "
                "grid's top."
            ),
        ),
        "source_tail_floor": Field(
            1.204013866384912e-08,
            (
                "The smallest start of the source grid's lower tail and the "
                "smallest width of the source grid's upper tail, as fractions of "
                "the grid's top."
            ),
        ),
        "quadrature_split": Field(
            0.9,
            (
                "The fraction of the potential at which the first interval of the "
                "source quadrature ends and the second interval begins."
            ),
        ),
        "quadrature_stretch": Field(
            2.0,
            (
                "The power k of the map Q = P - u^k on the second interval of the "
                "source quadrature, where the exponent of the quadrature's kernel "
                "is not negative."
            ),
        ),
        "ode_rtol": Field(
            1e-11, "The relative tolerance of the integration of Poisson's equation."
        ),
        "ode_atol": Field(
            1e-11, "The absolute tolerance of the integration of Poisson's equation."
        ),
        "n_rows_inner": Field(
            500,
            ("The number of radii of the Poisson solve's grid inside the inner zone."),
        ),
        "n_rows_outer": Field(
            59500,
            ("The number of radii of the Poisson solve's grid outside the inner zone."),
        ),
        "log10_r_min": Field(
            -9.0,
            (
                "The base-10 logarithm of the innermost radius of the Poisson "
                "solve, in internal length units."
            ),
        ),
        "log10_r_max": Field(
            12.0,
            (
                "The base-10 logarithm of the outermost radius of the Poisson "
                "solve, in internal length units. A computation raises the "
                "outermost radius until the potential at the outermost radius is "
                "below half the floor, and the result records the value used."
            ),
        ),
        "continuation_start": Field(
            1e-2,
            (
                "The fraction of the central potential below which the outermost "
                "radii of a truncated model are solved again on a grid geometric in"
                " R_t - r, where R_t is the truncation radius."
            ),
        ),
        "continuation_rows_per_decade": Field(
            1200,
            (
                "The number of radii per decade of R_t - r in the solve toward the "
                "truncation radius R_t."
            ),
        ),
        "continuation_floor_fraction": Field(
            2e-2,
            (
                "The potential at the outermost radius of the solve toward the "
                "truncation radius, as a fraction of the floor."
            ),
        ),
        "profile_bin_decades": Field(
            0.1,
            (
                "The width, in decades of the potential, of each bin in which the "
                "profile's radii are counted."
            ),
        ),
        "profile_rows_per_decade": Field(
            1000,
            (
                "The smallest number of radii per decade of the potential, and per "
                "decade of the depth, that the profile keeps at the profile's two "
                "ends; the profile's fills supply the other radii."
            ),
        ),
    },
    "sampling": {
        "n_rows": Field(
            512,
            (
                "The number of potential values of the velocity table at the start of "
                "the table's refinement; the number doubles until the table's own "
                "error across potential values is below row_error_goal or until the"
                " number reaches n_rows_max."
            ),
        ),
        "n_rows_max": Field(
            4075,
            (
                "The largest number of potential values of the velocity table; the "
                "table's refinement stops at n_rows_max potential values."
            ),
        ),
        "n_nodes": Field(
            2048,
            "The number of nodes at each potential value of the velocity table.",
            quick=512,
        ),
        "n_velocity_intervals": Field(
            200,
            (
                "The number of intervals in the scaled velocity over which the "
                "velocity distribution is integrated at each potential value of the "
                "velocity "
                "table."
            ),
        ),
        "n_gauss_legendre": Field(
            16,
            (
                "The number of Gauss-Legendre nodes of each panel of the velocity "
                "integrals."
            ),
            quick=8,
        ),
        "panels_per_decade": Field(
            8,
            (
                "The number of panels per decade of the local variable where the "
                "velocity integrals refine toward the two ends of the velocity range "
                "at "
                "one potential value and inside each interval."
            ),
        ),
        "velocity_low_refined": Field(
            1e-9,
            (
                "The smallest scaled velocity that the velocity integrals refine in "
                "panels; below that velocity, the density follows the density's power "
                "law in the velocity and is integrated in closed form."
            ),
        ),
        "edge_refined": Field(
            1e-12,
            (
                "The distance from the edge of the velocity range, relative to the "
                "edge, down to which the velocity integrals refine in panels; closer "
                "to the edge, the distribution function's form below the "
                "distribution function's table is integrated in closed form."
            ),
        ),
        "radius_edge_refined": Field(
            1e-10,
            (
                "The smallest fraction of the mass beyond a radius that the radius "
                "table of a model with a sharp edge resolves in nodes; for smaller "
                "fractions, the mass follows the mass's power law in the distance "
                "from the edge."
            ),
        ),
        "n_radius_nodes": Field(2048, "The number of nodes of the radius table."),
        "n_angle_nodes": Field(
            1024,
            (
                "The number of nodes of each half of the angle table, spaced "
                "uniformly in the logarithm of the cumulative probability from the "
                "half's end."
            ),
        ),
        "draw_block": Field(
            262144,
            (
                "The number of particles drawn at once. A draw runs over blocks of "
                "draw_block particles, so the draw's working memory, above the "
                "memory of the drawn particles, is the memory of one block: about "
                "0.75 kB per particle, or about 0.2 GB for :math:`2^{18}` "
                "particles. The drawn particles do not depend on draw_block."
            ),
        ),
        "row_error_goal": Field(
            TIER_BOUNDS[0],
            (
                "The error across the potential values of the velocity table at which "
                "the table's refinement stops: the number of potential values "
                "doubles until the table's own error is below row_error_goal or "
                "until the number reaches n_rows_max."
            ),
            quick=TIER_BOUNDS[1],
        ),
        "tail_mass_min": Field(
            1e-18,
            (
                "The smallest cumulative probability tabulated at either end of a "
                "table. The smallest nonzero value of a uniform random double is "
                ":math:`2^{-53}`, which is above tail_mass_min, so a draw never "
                "reads beyond the tabulated range."
            ),
        ),
    },
    "observables": {
        "n_radii": Field(
            10000,
            (
                "The number of projected radii of the projection table before the "
                "table's check between projected radii adds projected radii; the "
                "projected radii are spaced logarithmically from the innermost "
                "radius of the profile to the edge."
            ),
            quick=500,
        ),
        "profile_tolerance": Field(
            TIER_BOUNDS[0] / 10,
            (
                "The relative error of sigma_r at which radii stop being added to "
                "the radial dispersion's profile: sigma_r at the midpoint of every "
                "interval between the profile's radii is read within "
                "profile_tolerance. The default value, :math:`10^{-7}`, is a tenth "
                "of :math:`10^{-6}`: the profile's share of the error of the "
                "quantities projected from the profile."
            ),
            quick=TIER_BOUNDS[1] / 100,
        ),
        "n_profile_start": Field(
            100,
            (
                "The number of radii, spaced logarithmically from the innermost "
                "radius of the profile to the edge, with which the radial "
                "dispersion's profile starts before radii are added to meet "
                "profile_tolerance."
            ),
        ),
        "n_jeans_nodes": Field(
            500,
            (
                "The number of Gauss-Legendre nodes of the Jeans integral of the "
                "radial dispersion at one radius, as the superposition of "
                "components computes the radial dispersion. The observables' "
                "profile does not use n_jeans_nodes: the profile sums panels "
                "between the profile's radii and doubles each panel's nodes until "
                "sigma_r changes by less than profile_tolerance."
            ),
        ),
        "n_projection_nodes": Field(
            500,
            (
                "The number of Gauss-Legendre nodes of each line-of-sight integral "
                "of the surface density and of the projected dispersion."
            ),
        ),
        "n_aperture_nodes": Field(
            100,
            (
                "The smallest number of Gauss-Legendre nodes of each panel of an "
                "aperture's radial integral."
            ),
        ),
        "aperture_nodes_per_decade": Field(
            8,
            (
                "The number of nodes per decade of projected radius of a panel of "
                "an aperture's radial integral: a panel spanning d decades takes "
                "aperture_nodes_per_decade times d nodes, rounded up to a multiple "
                "of 20, if that number exceeds n_aperture_nodes."
            ),
        ),
        "rows_route": Field(
            "kernel",
            (
                "The computation of the angle rule of the line-of-sight velocity "
                'distributions: "kernel" computes the angle rule with JAX, and '
                '"numpy" computes the angle rule with NumPy, as a reference.'
            ),
        ),
        "n_velocity_angle_nodes": Field(
            16,
            (
                "The number of Gauss-Legendre nodes of each panel of the angle "
                "integral of the local velocity distribution (pdf_v), in t = cos(eta)."
            ),
        ),
        "n_velocity_angle_panels": Field(
            16,
            (
                "The number of panels of the angle integral of the local velocity "
                "distribution (pdf_v) graded toward each end of each of the "
                "integral's intervals, and the number of panels of the velocity "
                "integral of the distribution's norm and second moment graded "
                "toward each break of that velocity integral."
            ),
            quick=8,
        ),
        "velocity_tolerance": Field(
            TIER_BOUNDS[0],
            (
                "The bound on the change of the local velocity distribution (pdf_v) "
                "when the nodes of the distribution's angle rule double: the nodes "
                "double, at most three times, until the change is below "
                "velocity_tolerance."
            ),
            quick=TIER_BOUNDS[1],
        ),
    },
    "losvd": {
        "rows_per_decade": Field(
            55,
            (
                "The number of base projected radii per decade of the table of "
                "line-of-sight velocity distributions, from the innermost radius of"
                " the profile."
            ),
            quick=20,
        ),
        "n_energy_nodes": Field(
            8,
            (
                "The number of Gauss-Legendre nodes of each panel of the energy "
                "integral at one node of the table."
            ),
        ),
        "n_floor_layers": Field(
            4,
            (
                "The number of Gauss-Legendre panels of the energy integral graded "
                "toward K = 0, where the DF's rule below its table takes over."
            ),
            quick=2,
        ),
        "energy_layer_ratio": Field(
            0.2,
            (
                "The ratio of the widths of successive graded panels of the energy "
                "integral."
            ),
        ),
        "n_angle_nodes": Field(
            64,
            (
                "The largest number of Gauss-Legendre nodes of the angle integral "
                "of an L factor at one node of the table."
            ),
        ),
        "n_sightline_nodes": Field(
            12,
            (
                "The number of Gauss-Legendre nodes of each panel of the line-of-"
                "sight integral."
            ),
            quick=8,
        ),
        "n_sightline_panels": Field(
            8,
            (
                "The number of panels of each interval of the support along the "
                "line of sight."
            ),
        ),
        "sightline_panels_refined": Field(
            6,
            (
                "The number of doublings of the line-of-sight panels allowed before"
                " a projected radius is recorded as not converged."
            ),
        ),
        "sightline_layer_ratio": Field(
            0.3,
            (
                "The ratio of the widths of successive line-of-sight panels graded "
                "toward the axis."
            ),
        ),
        "n_sightline_layers": Field(
            4,
            (
                "The number of line-of-sight panels graded toward the axis that "
                "each doubling adds."
            ),
        ),
        "n_crossing_scan": Field(
            2001,
            (
                "The number of points of the scan for the positions where the "
                "angular momentum vanishes."
            ),
        ),
        "n_cumulative_nodes": Field(
            200001,
            (
                "The number of nodes of each of the three branches of the table of "
                "the distribution function's cumulative integral."
            ),
        ),
        "n_velocity_nodes": Field(
            16,
            (
                "The number of Gauss-Legendre nodes of each velocity sub-panel at "
                "one projected radius."
            ),
            quick=8,
        ),
        "velocity_layer_ratio": Field(
            0.3,
            (
                "The ratio of the widths of successive velocity sub-panels graded "
                "toward a break of the distribution."
            ),
        ),
        "n_velocity_layers_zero": Field(
            12, "The number of velocity sub-panels graded toward zero velocity."
        ),
        "n_velocity_layers_break": Field(
            8,
            (
                "The number of velocity sub-panels graded toward each break of the "
                "distribution at one projected radius."
            ),
        ),
        "n_break_scan": Field(
            4001,
            (
                "The number of points of the coarse scan for a break of the "
                "distribution in velocity."
            ),
            quick=1001,
        ),
        "n_break_zooms": Field(
            4,
            (
                "The number of successive refinements of the scan for a break of "
                "the distribution in velocity."
            ),
        ),
        "n_bisections": Field(
            60,
            (
                "The number of bisections that locate an edge of the support, or a "
                "root, along a line of sight."
            ),
        ),
        "n_radius_break_layers": Field(
            16,
            (
                "The number of projected radii added toward each projected radius "
                "at which the table's layout changes."
            ),
        ),
        "n_radius_break_bisections": Field(
            40,
            (
                "The number of bisections in projected radius that locate a change "
                "of the table's layout."
            ),
        ),
        "n_radius_break_samples": Field(
            3,
            (
                "The number of samples inside each interval between base projected "
                "radii at which a change of the table's layout is sought."
            ),
        ),
        "n_edge_samples": Field(
            10,
            (
                "The number of samples beyond the last base projected radius toward"
                " the end of the profile."
            ),
        ),
        "n_edge_layers": Field(
            16, "The number of projected radii added toward the end of the table."
        ),
        "n_stencil_rows": Field(
            8,
            (
                "The number of tabulated projected radii that the interpolation "
                "between the table's projected radii reads."
            ),
        ),
        "n_stencil_test_rows": Field(
            8,
            (
                "The number of tabulated projected radii of the second "
                "interpolation, whose estimate checks the table's interpolation."
            ),
        ),
        "n_block_elements": Field(
            25000,
            (
                "The number of elements (node pairs times energy nodes times angle "
                "nodes) computed in one block, which bounds the memory of the "
                "computation at one projected radius."
            ),
        ),
        "innermost_read_radius": Field(
            0.0,
            (
                "The smallest projected radius at which the table will be read, in "
                "the model's internal length unit. The table starts at the last "
                "base projected radius inside the radius whose disk holds "
                ":math:`10^{-6}` times the projected mass inside "
                "innermost_read_radius, and a read inside the table's first "
                "projected radius raises InputError. The value 0 keeps the table's "
                "full depth, from the innermost radius of the profile."
            ),
        ),
        "outer_mass_share": Field(
            TIER_BOUNDS[0],
            (
                "The share of the total projected mass that may lie beyond the "
                "table's last projected radius. The table ends at the first base "
                "projected radius beyond the radius outside which outer_mass_share "
                "of the projected mass lies, and a read beyond the table's last "
                "projected radius raises InputError. The value 0 keeps the table "
                "out to the model's edge."
            ),
        ),
    },
}


# The fields that accept the value 0, which turns the field's limit off:
# innermost_read_radius = 0 keeps the projected table's full depth, and
# outer_mass_share = 0 keeps the projected table out to the model's edge.
OFF_AT_ZERO = ("innermost_read_radius", "outer_mass_share")


def _checked(group: str, name: str, value: Any) -> Any:
    """Return an override converted to the type of its field.

    An int override of a float field is converted to float. An override of
    a number field must be positive, except for a field whose name starts
    with ``log10_`` and for the value 0 of a field in OFF_AT_ZERO.

    Raises
    ------
    InputError
        If `name` is not a field of `group`, if the override's type is not
        the field's type, or if a number override is not allowed.
    """
    fields = FIELDS[group]
    if name not in fields:
        raise InputError(
            f"The settings group {group!r} has no field {name!r}; the fields of "
            f"the settings group {group!r} are {sorted(fields)}."
        )
    kind = type(fields[name].default)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not kind:
        raise InputError(
            f"The field {group}.{name} takes a value of type {kind.__name__}; "
            f"the value given is {value!r}."
        )
    zero = name in OFF_AT_ZERO and value == 0
    if (
        kind in (int, float)
        and not name.startswith("log10_")
        and not (value > 0 or zero)
    ):
        raise InputError(
            f"The field {group}.{name} does not take the value {value!r}, which "
            "is not positive."
        )
    return value


class Settings:
    r"""The numerical settings of a computation.

    A `Settings` object starts from one of the two tested settings levels,
    the default settings or the quick settings, and may change single
    fields of a group. A result records its settings; a result made with a
    changed field records that its settings are not a tested settings level
    and lists each changed field.

    Parameters
    ----------
    quick : bool, optional
        Whether the settings start from the quick settings. Default:
        ``False``, which means the settings start from the default settings.
    write_anyway : bool, optional
        Whether a computation returns a result whose own check is above
        :math:`10^{-2}` instead of raising `AccuracyError`. The returned
        result's validation report has the band ``"refused"``, and an
        `AccuracyWarning` is issued in place of the error. A model that
        is not a physical equilibrium raises `UnphysicalModelError`
        whatever the value of `write_anyway`. Default: ``False``, which
        means a result above :math:`10^{-2}` raises `AccuracyError`.
    check_rows : bool, optional
        Whether a gradient through a `LosTable` recomputes each
        line-of-sight velocity distribution the gradient reads and compares
        the recomputed distribution with the stored distribution. A
        recomputed distribution must equal the stored distribution bit for
        bit, and a JAX copy of the distribution's computation (the twin
        that carries the derivative) must agree with the stored
        distribution to :math:`10^{-12}` of the distribution's largest
        value; otherwise the gradient raises `ResultError`. The gradient's value is
        the same with and without the comparison; the comparison computes
        each distribution read a second time. Default: ``False``, which
        means no distribution is recomputed.
    **overrides : mapping
        For each group to change (``inversion``, ``truncation``,
        ``sampling``, ``observables`` or ``losvd``), a mapping from field
        name to new value. The manual's settings tables list every field.
        Default: no override, which means every field has its value at the
        settings level.

    Attributes
    ----------
    level : {"default", "quick"}
        The settings level the settings start from.
    write_anyway : bool
        Whether a result above :math:`10^{-2}` is returned instead of
        raising `AccuracyError`.
    check_rows : bool
        Whether a gradient through a `LosTable` recomputes and compares
        each distribution the gradient reads.
    overrides : mapping
        For each changed group, the changed fields and their values. Read
        only.
    is_tested_level : bool
        Whether no field is changed from the settings level.

    Raises
    ------
    InputError
        If `write_anyway` or `check_rows` is not a bool.
    InputError
        If a keyword of `overrides` is not a group name.
    InputError
        If a field of an override is not a field of the override's group.
    InputError
        If an override's value does not have the type of the field's
        default value (an int is accepted for a float field).
    InputError
        If a number override is zero or negative, except a field whose name
        starts with ``log10_`` and the value 0 of ``innermost_read_radius``
        and ``outer_mass_share``.

    See Also
    --------
    FIELDS : Every field of every group with its value at each level.
    LEVELS : The names of the two tested settings levels.

    Examples
    --------
    >>> Settings().level
    'default'
    >>> Settings(quick=True).is_tested_level
    True
    >>> Settings(inversion={"n_nodes": 256}).is_tested_level
    False
    >>> Settings(inversion={"n_nodes": 256}).inversion["n_nodes"]
    256
    >>> Settings().truncation["ode_rtol"]
    1e-11
    """

    __slots__ = ("_check_rows", "_overrides", "_level", "_write_anyway")
    _overrides: Mapping[str, Mapping[str, Any]]
    _level: str
    _write_anyway: bool
    _check_rows: bool

    def __init__(
        self,
        quick: bool = False,
        write_anyway: bool = False,
        check_rows: bool = False,
        **overrides: Mapping[str, Any],
    ) -> None:
        if not isinstance(write_anyway, bool):
            raise InputError(
                "Settings takes write_anyway as True or False; the value given is "
                f"{write_anyway!r}."
            )
        if not isinstance(check_rows, bool):
            raise InputError(
                "Settings takes check_rows as True or False; the value given is "
                f"{check_rows!r}."
            )
        unknown = sorted(set(overrides) - set(GROUPS))
        if unknown:
            raise InputError(
                f"The names {unknown} are not settings groups; the settings groups "
                f"are {list(GROUPS)}."
            )
        frozen = {
            group: MappingProxyType(
                {name: _checked(group, name, value) for name, value in fields.items()}
            )
            for group, fields in sorted(overrides.items())
            if fields
        }
        object.__setattr__(self, "_level", "quick" if quick else "default")
        object.__setattr__(self, "_write_anyway", write_anyway)
        object.__setattr__(self, "_check_rows", check_rows)
        object.__setattr__(self, "_overrides", MappingProxyType(frozen))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "A Settings object cannot be changed after it is built; build a new "
            "Settings object with the changed values."
        )

    @property
    def level(self) -> str:
        """The settings level the settings start from, ``"default"`` or
        ``"quick"``."""
        return self._level

    @property
    def write_anyway(self) -> bool:
        """Whether a result above :math:`10^{-2}` is returned instead of
        raising `AccuracyError`."""
        return self._write_anyway

    @property
    def check_rows(self) -> bool:
        """Whether a gradient through a `LosTable` recomputes and compares
        each distribution the gradient reads."""
        return self._check_rows

    @property
    def overrides(self) -> Mapping[str, Mapping[str, Any]]:
        """For each changed group, the changed fields and their values."""
        return self._overrides

    @property
    def is_tested_level(self) -> bool:
        """Whether no field is changed from the settings level."""
        return not self._overrides

    def group(self, name: str) -> Mapping[str, Any]:
        """Return the value in force of every field of one group.

        Parameters
        ----------
        name : {"inversion", "truncation", "sampling", "observables", "losvd"}
            The group's name.

        Returns
        -------
        values : mapping
            Each field of the group mapped to the field's override if the
            settings change the field, and to the field's value at the
            settings level otherwise. Read only.

        Raises
        ------
        InputError
            If `name` is not a group name.

        See Also
        --------
        resolved : The values in force of every group.

        Examples
        --------
        >>> Settings(quick=True).group("sampling")["n_nodes"]
        512
        """
        if name not in GROUPS:
            raise InputError(
                f"{name!r} is not a settings group; the settings groups are "
                f"{list(GROUPS)}."
            )
        values = {key: field.value(self._level) for key, field in FIELDS[name].items()}
        values.update(self._overrides.get(name, {}))
        return MappingProxyType(values)

    @property
    def inversion(self) -> Mapping[str, Any]:
        """The values in force of the group ``inversion``."""
        return self.group("inversion")

    @property
    def truncation(self) -> Mapping[str, Any]:
        """The values in force of the group ``truncation``."""
        return self.group("truncation")

    @property
    def sampling(self) -> Mapping[str, Any]:
        """The values in force of the group ``sampling``."""
        return self.group("sampling")

    @property
    def observables(self) -> Mapping[str, Any]:
        """The values in force of the group ``observables``."""
        return self.group("observables")

    @property
    def losvd(self) -> Mapping[str, Any]:
        """The values in force of the group ``losvd``."""
        return self.group("losvd")

    def resolved(self) -> dict[str, dict[str, Any]]:
        """Return the value in force of every field of every group.

        Returns
        -------
        values : dict
            Each group name mapped to a dictionary of the group's fields and
            their values in force.

        See Also
        --------
        group : The values in force of one group.

        Examples
        --------
        >>> sorted(Settings().resolved())
        ['inversion', 'losvd', 'observables', 'sampling', 'truncation']
        """
        return {name: dict(self.group(name)) for name in GROUPS}

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as the dictionary written to result files.

        Returns
        -------
        record : dict
            The entries of `numerical` (``"level"`` and ``"overrides"``),
            and ``"write_anyway": True`` and ``"check_rows": True`` if the
            settings set those options.

        See Also
        --------
        from_dict : The settings built from the dictionary.
        numerical : The entries that determine a result's values.

        Examples
        --------
        >>> Settings(quick=True, write_anyway=True).to_dict()
        {'level': 'quick', 'overrides': {}, 'write_anyway': True}
        """
        record = self.numerical()
        if self._write_anyway:
            record["write_anyway"] = True
        if self._check_rows:
            record["check_rows"] = True
        return record

    def numerical(self) -> dict[str, Any]:
        """Return the settings that determine a result's values: the
        settings level and the overrides.

        A result's identity is computed from this dictionary, so that two
        results computed with the same level and overrides share an
        identity whatever their `write_anyway` and `check_rows`.

        Returns
        -------
        record : dict
            The entries ``"level"`` and ``"overrides"`` (each changed group
            mapped to its changed fields and their values).

        Examples
        --------
        >>> Settings(inversion={"n_nodes": 256}).numerical()
        {'level': 'default', 'overrides': {'inversion': {'n_nodes': 256}}}
        """
        return {
            "level": self._level,
            "overrides": {
                group: dict(fields) for group, fields in self._overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Settings":
        """Build settings from the dictionary that `to_dict` writes.

        Parameters
        ----------
        record : mapping
            A dictionary in the form `to_dict` returns. A missing
            ``"level"`` entry is read from the entry ``"tier"``, the
            entry's former name, and else as ``"default"``.

        Returns
        -------
        settings : Settings
            The settings the dictionary describes.

        Raises
        ------
        InputError
            If the dictionary's settings level is not one of `LEVELS`.
        InputError
            If an override in the dictionary is not accepted by `Settings`.

        See Also
        --------
        to_dict : The dictionary form of settings.

        Examples
        --------
        >>> settings = Settings(sampling={"n_rows_max": 1000})
        >>> Settings.from_dict(settings.to_dict()) == settings
        True
        """
        # Result files written with the former entry name "tier" for the
        # settings level are read through that entry.
        level = record.get("level", record.get("tier", "default"))
        if level not in LEVELS:
            raise InputError(
                f"{level!r} is not a settings level; the settings levels are "
                f"{list(LEVELS)}."
            )
        return cls(
            quick=(level == "quick"),
            write_anyway=bool(record.get("write_anyway", False)),
            check_rows=bool(record.get("check_rows", False)),
            **record.get("overrides", {}),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Settings):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        parts = ["quick=True"] if self._level == "quick" else []
        parts += ["write_anyway=True"] if self._write_anyway else []
        parts += ["check_rows=True"] if self._check_rows else []
        parts += [
            f"{group}={dict(fields)!r}" for group, fields in self._overrides.items()
        ]
        return f"Settings({', '.join(parts)})"
