"""Particles, their writers, and the prepared sampler."""

__all__ = [
    "PARTICLE_WRITERS",
    "Particles",
    "Sampler",
    "component_view",
    "is_superposition",
    "register_particle_writer",
    "write_particles",
]

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .._version import __version__
from ..df import Model
from ..df.core import N_COMPONENTS_MAX
from ..exceptions import InputError, ResultError
from ..identity import identity_string
from ..io import StoredResult, load_result, register_reader, write_result
from ..profiles import TruncatedProfile
from ..typing import Array
from ..validation import ValidationReport, announce, refuse_unless_asked
from . import stages

RESULT_KIND = "particles"
UNIT_SYSTEMS = ("internal", "model", "physical")


@dataclass(frozen=True, eq=False)
class Particles:
    """A particle realization of a model.

    Parameters
    ----------
    position : array, shape (n, 3)
        Cartesian positions.
    velocity : array, shape (n, 3)
        Cartesian velocities.
    mass : array, shape (n,)
        Particle masses.
    component : array of int, shape (n,)
        Index, into `component_names`, of the component each particle
        belongs to.
    component_names : tuple of str
        Names of the components of the model.
    unit_system : {"internal", "model", "physical"}
        The unit system of the arrays. Internal by default everywhere.
    units : dict of str to str
        Unit of the position, the velocity and the mass, as text.
    identity : str
        Identity string of the realization.
    parent : str
        Identity string of the sampler or model the particles were drawn from.

    Notes
    -----
    The arrays are read-only. Two realizations are compared by their
    identity strings.
    """

    position: Array
    velocity: Array
    mass: Array
    component: Array
    component_names: tuple[str, ...]
    unit_system: str
    units: dict[str, str]
    identity: str
    parent: str

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        mass = np.asarray(self.mass, dtype=np.float64)
        component = np.asarray(self.component, dtype=np.int64)
        n_particles = len(mass)
        if position.shape != (n_particles, 3) or velocity.shape != (n_particles, 3):
            raise InputError(
                f"position and velocity must have the shape ({n_particles}, 3); got "
                f"{position.shape} and {velocity.shape}."
            )
        if component.shape != (n_particles,):
            raise InputError(
                f"component must have the shape ({n_particles},); "
                f"got {component.shape}."
            )
        if n_particles and not (
            0 <= component.min() and component.max() < len(self.component_names)
        ):
            raise InputError("component holds an index with no name.")
        if self.unit_system not in UNIT_SYSTEMS:
            raise InputError(
                f"unit_system must be one of {list(UNIT_SYSTEMS)}; "
                f"got {self.unit_system!r}."
            )
        for name, values in (
            ("position", position),
            ("velocity", velocity),
            ("mass", mass),
            ("component", component),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "component_names", tuple(self.component_names))
        object.__setattr__(self, "units", dict(self.units))

    def __len__(self) -> int:
        return len(self.mass)

    def write(
        self, path: str | os.PathLike, layout: str = "native", **options: Any
    ) -> None:
        """Write the particles to a file; see `write_particles`."""
        write_particles(self, path, layout=layout, **options)


ParticleWriter = Callable[..., None]
PARTICLE_WRITERS: dict[str, ParticleWriter] = {}


def register_particle_writer(layout: str) -> Callable[[ParticleWriter], ParticleWriter]:
    """Register a function that writes particles in a given file layout.

    A writer takes the particles, the path and its own keyword options. The
    layouts of simulation codes are added this way, without a change to the
    sampling or to `Particles`.

    Parameters
    ----------
    layout : str
        Name of the layout, as passed to `write_particles`.

    Returns
    -------
    decorator : callable
        Registers the decorated function and returns it unchanged.
    """

    def decorator(writer: ParticleWriter) -> ParticleWriter:
        if layout in PARTICLE_WRITERS:
            raise InputError(f"A particle writer named {layout!r} already exists.")
        PARTICLE_WRITERS[layout] = writer
        return writer

    return decorator


def write_particles(
    particles: Particles,
    path: str | os.PathLike,
    layout: str = "native",
    **options: Any,
) -> None:
    """Write particles to a file in the requested layout.

    Parameters
    ----------
    particles : Particles
        The particles to write.
    path : path
        File to write.
    layout : str, optional
        Name of a registered layout. ``"native"`` is the package's own HDF5
        result file, which `_read_particles` reads back.
    **options
        Options of the layout's writer.
    """
    if layout not in PARTICLE_WRITERS:
        raise InputError(
            f"Unknown particle layout {layout!r}; the layouts are "
            f"{sorted(PARTICLE_WRITERS)}."
        )
    PARTICLE_WRITERS[layout](particles, path, **options)


@register_particle_writer("native")
def _write_native(
    particles: Particles, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write particles as a result file of the package."""
    record = {
        "component_names": list(particles.component_names),
        "unit_system": particles.unit_system,
        "units": particles.units,
    }
    arrays = {
        "position": particles.position,
        "velocity": particles.velocity,
        "mass": particles.mass,
        "component": particles.component,
    }
    write_result(
        path,
        RESULT_KIND,
        particles.identity,
        record,
        arrays,
        parents=(particles.parent,),
        overwrite=overwrite,
    )


def _read_particles(source: StoredResult | str | os.PathLike) -> Particles:
    """Return particles from a result file or from a stored result.

    Parameters
    ----------
    source : StoredResult or path
        A stored result, or the path of a file written in the native layout.

    Returns
    -------
    particles : Particles

    Raises
    ------
    ResultError
        If the file holds a result of another kind.
    """
    stored = load_result(source)
    if stored.kind != RESULT_KIND:
        raise ResultError(
            f"The result is of kind {stored.kind!r}, not {RESULT_KIND!r}."
        )
    return Particles(
        position=stored.arrays["position"],
        velocity=stored.arrays["velocity"],
        mass=stored.arrays["mass"],
        component=stored.arrays["component"],
        component_names=tuple(stored.record["component_names"]),
        unit_system=stored.record["unit_system"],
        units=stored.record["units"],
        identity=stored.identity,
        parent=stored.parents[0],
    )


register_reader(RESULT_KIND, _read_particles)


@dataclass(frozen=True)
class _Part:
    """One component's tables, and the one-component model they are built on."""

    model: Model
    radius_table: Any
    speed_table: Any
    angle_table: Any


def _is_superposition(model: Model) -> bool:
    """Whether every component is built on one truncated profile (`superpose`)."""
    first = model.components[0].profile
    return isinstance(first, TruncatedProfile) and all(
        c.profile == first for c in model.components[1:]
    )


def _component_view(model: Model, k: int) -> Model:
    """Component `k` of a superposed model as a model of one component: its own
    table on its own grid, its forms, and the shared truncated profile."""
    t = model.table
    row = t.k[k]
    n_k = (
        int(np.argmax(row == row[-1])) + 1
    )  # where the padding by the last energy starts
    table = dataclasses.replace(
        t,
        k=t.k[k : k + 1, :n_k],
        log_df=t.log_df[k : k + 1, :n_k],
        log_df_slope=t.log_df_slope[k : k + 1, :n_k],
        i_first=(t.i_first[k],),
        i_last=(t.i_last[k],),
        slopes_exact=(t.slopes_exact[k],),
        k_below=t.k_below[k : k + 1] if t.k_below.size else t.k_below,
        log_df_below=t.log_df_below[k : k + 1]
        if t.log_df_below.size
        else t.log_df_below,
        log_df_slope_below=(
            t.log_df_slope_below[k : k + 1]
            if t.log_df_slope_below.size
            else t.log_df_slope_below
        ),
        n_below=(t.n_below[k],) if t.n_below else (),
    )
    n = len(model.components)

    def one(record: Any) -> Any:
        values = {}
        for f in dataclasses.fields(record):
            v = getattr(record, f.name)
            values[f.name] = (v[k],) if isinstance(v, tuple) and len(v) == n else v
        return type(record)(**values)

    component = model.components[k]
    return dataclasses.replace(
        model,
        components=(dataclasses.replace(component, mass=1.0),),
        table=table,
        form=one(model.form),
        inversion=None if model.inversion is None else one(model.inversion),
        profile=cast(TruncatedProfile, component.profile),
    )


def is_superposition(model: Model) -> bool:
    """Return whether a model is a superposition (`superpose`).

    Parameters
    ----------
    model : Model
        The model.

    Returns
    -------
    superposed : bool
        Whether every component is built on one truncated profile, as
        `superpose` builds them.
    """
    return _is_superposition(model)


def component_view(model: Model, k: int) -> Model:
    """Return component `k` of a superposed model as a model of one component.

    Parameters
    ----------
    model : Model
        A superposed model (`is_superposition`).
    k : int
        Index of the component.

    Returns
    -------
    view : Model
        The component on its own: its table on its own grid, its forms, the
        shared truncated profile, and mass 1. The sampler and the observables
        read each component of a superposition through it.
    """
    return _component_view(model, k)


class Sampler:
    """Everything needed to draw particles from a model, built once.

    The sampler holds the tables that sampling needs. They are built when the
    sampler is created and reused by every call of `sample`, so a user keeps
    one object however many tables there are.

    Parameters
    ----------
    model : Model
        The model to draw from.
    r_sample : float, optional
        Outer radius of the sampling, in internal units: the radius table
        depends on it. Required for a model that is not spatially finite;
        the truncation radius of one that is.

    Attributes
    ----------
    validation : ValidationReport
        The checks of the tables (those of `SpeedTable.validation`) and the
        worst tier of the model. A sampler above the goal says so in a
        warning; one beyond the hard limit is refused.

    Raises
    ------
    InputError
        If `model` is not a Model, or has more components than this release
        samples.
    AccuracyError
        If the model's report reads ``"refused"`` and its settings do not ask
        for results from it anyway (`Settings`, ``write_anyway``).
    ResultError
        If a speed table lies beyond the hard limit and the settings do not ask
        for it anyway.

    Notes
    -----
    A physical model is not thereby samplable at every tier. The speed
    table's rows are refined until the error across them is under the goal
    or their count reaches ``n_rows_max``; close to the smallest anisotropy
    radius at which an Osipkov-Merritt or Cuddeford Type I DF is non-negative,
    the rows reach ``n_rows_max`` first and the error grows steeply as r_a
    falls toward that radius. Measured on a truncated NFW density of
    concentration 10 at the default tier, the error crosses the hard limit at
    1.02 to 1.2 times that radius and the ceiling at 1.09 to 1.84 times it
    (up to beta_c = 0.4), the widest for the most negative beta_c; at
    beta_c = 1/2, whose DF is non-negative at every r_a, the table is beyond
    the hard limit at every r_a up to 0.32. A constant-anisotropy model is
    within the ceiling up to beta_c = 0.4. The bound is measured, not
    imposed: the report of each table is the verdict (the user guide,
    Particles, gives the table per family).
    """

    def __init__(self, model: Model, r_sample: float | None = None) -> None:
        if not isinstance(model, Model):
            raise InputError("A Sampler is built from a Model.")
        superposed = len(model.components) > 1 and _is_superposition(model)
        if len(model.components) > N_COMPONENTS_MAX and not superposed:
            raise InputError(
                "This release samples models of one component; got "
                f"{len(model.components)}. Several components in one potential "
                "are not available yet."
            )
        settings = model.settings
        # a model beyond the hard limit is an input only under the request
        # that made it (Settings(write_anyway=True)); the one statement
        refuse_unless_asked(
            f"The model {model.components[0].name!r}", model.validation, settings
        )
        self.model = model
        self.r_sample = r_sample
        # one part per component: a superposed model's components share the
        # density, and each is drawn from its own tables
        views = [_component_view(model, k) for k in range(len(model.components))]
        if not superposed:
            views = [model]
        self.parts = [
            _Part(
                view,
                stages.radius_table(view, settings, r_sample),
                stages.speed_table(view, settings),
                stages.angle_table(view, settings),
            )
            for view in views
        ]
        self.radius_table = self.parts[0].radius_table
        self.speed_table = self.parts[0].speed_table
        self.angle_table = self.parts[0].angle_table
        masses = np.array([c.mass for c in model.components])
        self.weights = masses / masses.sum()
        worst = max(
            (part.speed_table.validation for part in self.parts),
            key=lambda report: report.level,
        )
        table_checks = ValidationReport(
            keys=worst.keys,
            checks={
                key: max(
                    (p.speed_table.validation.checks[key] for p in self.parts),
                    key=lambda check: check.error,
                )
                for key in worst.checks
            },
        )
        self.validation = ValidationReport(
            keys=table_checks.keys,
            checks=table_checks.checks,
            parents_level=model.validation.level,
        )
        if self.validation.level > 0:
            announce(
                f"The sampler of {model.components[0].name!r}",
                self.validation,
                ", ".join(
                    f"{key} {check.error:.2e} at Psi = {check.where:.6e}"
                    for key, check in table_checks.checks.items()
                    if check.level > 0
                )
                or "the tier of its model",
            )
        self.identity = identity_string(
            {"model": model.identity, "r_sample": r_sample},
            settings.to_dict(),
            __version__,
        )

    def sample(self, n: int, seed: int = 0) -> Particles:
        """Draw particles from the model.

        Parameters
        ----------
        n : int
            Number of particles.
        seed : int, optional
            Seed of the random streams.

        Returns
        -------
        particles : Particles
            The particles, in the model's units (`Model.units`). Every draw is
            made in internal units and converted at the output only.

        Raises
        ------
        ResultError
            If a particle lies beyond the phase-space bound of its component.
        """
        if not (isinstance(n, int) and n > 0):
            raise InputError(f"n must be a positive integer; got {n!r}.")
        component = 0
        # blocks of one size (the last padded and cut), so the working memory is
        # one block's and one compilation serves every block
        block = min(int(self.model.settings.sampling["draw_block"]), n)
        units = self.model.units
        # the analytic profile that fixes the units (a truncated model's source)
        profile = self.model.components[component].unit_profile
        prepared = [
            stages.speed_draw_tables(part.speed_table, part.model, component)
            for part in self.parts
        ]
        arrays = {"position": np.empty((n, 3)), "velocity": np.empty((n, 3))}
        labels = np.zeros(n, dtype=np.int64)
        edges = np.cumsum(self.weights)[:-1]
        unit_text = {}
        for start in range(0, n, block):
            size = min(block, n - start)
            # the particle's numbers are those of a one-component draw; a
            # superposed model's particle takes its component by the weights from
            # a channel of its own, and its draw from that component's tables
            numbers = stages.random_numbers_block(seed, component, start, block)
            chosen = np.zeros(block, dtype=np.int64)
            if len(self.parts) > 1:
                u = stages.component_numbers(seed, start, block)
                chosen = np.asarray(
                    np.searchsorted(edges, u, side="right"), dtype=np.int64
                )
            pos = np.empty((block, 3))
            vel = np.empty((block, 3))
            for k, part in enumerate(self.parts):
                mine = chosen[:size] == k
                if len(self.parts) > 1 and not mine.any():
                    continue
                r = stages.draw_radius(part.radius_table, component, numbers)
                cos_phi, sin_phi, factor = stages.draw_angle(
                    part.angle_table, part.model, component, r, numbers
                )
                v, x, d, psi = stages.draw_speed(
                    part.speed_table, part.model, component, r, factor, numbers,
                    prepared[k],
                )  # fmt: skip
                pos_k, vel_k = stages.draw_directions(r, v, cos_phi, sin_phi, numbers)
                if len(self.parts) == 1:
                    pos, vel = pos_k, vel_k
                    stages.validate_particles(
                        part.model, component, pos[:size], vel[:size], x[:size],
                        d[:size], psi[:size],
                    )  # fmt: skip
                    continue
                pos[:size][mine], vel[:size][mine] = (
                    pos_k[:size][mine],
                    vel_k[:size][mine],
                )
                stages.validate_particles(
                    part.model, component, pos_k[:size][mine], vel_k[:size][mine],
                    x[:size][mine], d[:size][mine], psi[:size][mine],
                )  # fmt: skip
            labels[start : start + size] = chosen[:size]
            # converted block by block (an elementwise product) into the output
            for name, kind, values in (
                ("position", "length", pos[:size]),
                ("velocity", "velocity", vel[:size]),
            ):
                converted = units.convert(kind, values, profile)
                unit_text[name] = str(getattr(converted, "unit", units.system))
                arrays[name][start : start + size] = getattr(
                    converted, "value", converted
                )
        converted = units.convert(
            "mass", np.full(n, self.radius_table.mass_sample[component] / n), profile
        )
        unit_text["mass"] = str(getattr(converted, "unit", units.system))
        arrays["mass"] = np.asarray(getattr(converted, "value", converted))
        return Particles(
            position=arrays["position"],
            velocity=arrays["velocity"],
            mass=arrays["mass"],
            component=labels,
            component_names=tuple(c.name for c in self.model.components),
            unit_system=units.system,
            units=unit_text,
            identity=identity_string(
                {"sampler": self.identity, "n": n, "seed": seed}, {}, __version__
            ),
            parent=self.identity,
        )
