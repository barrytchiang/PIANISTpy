r"""The particle result, its file writers, and the sampler that draws particles.

A user builds a `Sampler` from a model once and calls `Sampler.sample` for
particles, or `Sampler.velocities` for one velocity at each user-specified 3D radius.
`Sampler.sample` returns a `Particles` object, which `Particles.write` or
`write_particles` writes to a file in one of the layouts registered in
``PARTICLE_WRITERS``.

Every particle is drawn by inversion, with no rejection step. The radius
:math:`r` of a particle solves :math:`M(r)/M(r_{\rm s}) = u_r`, where
:math:`M(r)` is the mass of the model's component enclosed within :math:`r`,
:math:`r_{\rm s}` is the outer radius of the sampling and :math:`u_r` is a
uniform random number between 0 and 1. The velocity of a particle is drawn from
the model's distribution of velocity at the potential :math:`\Psi(r)` of the
particle's radius. For a constant anisotropy :math:`\beta_{\rm c}`, the angle
:math:`\eta` between the velocity and the radius vector follows from
:math:`y = \sin^2\eta`, which has the beta distribution with parameters
:math:`1 - \beta_{\rm c}` and :math:`1/2`. The random numbers of a particle
depend only on the seed, the particle's component and the particle's index.
The derivation is in Notes Section T:5.
"""

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
# The largest number of particles in one draw, 2^31 - 1: the random keys of a
# draw are indexed by int32 integers, and the Gadget writer stores particle IDs
# as uint32 integers.
MAX_PARTICLES = 2**31 - 1


@dataclass(frozen=True, eq=False)
class Particles:
    """A particle realization of a model: positions, velocities, masses and
    component labels.

    `Sampler.sample` returns a `Particles` object. `Particles.write` writes the
    particles to a file, and `read` reads a file of the native layout back
    into a `Particles` object. Each parameter below is kept as an attribute
    with the same spelling as the parameter, and the four arrays are
    read-only.

    Parameters
    ----------
    position : array-like, shape (n, 3)
        Cartesian positions of the n particles, in the unit that
        ``units["position"]`` gives.
    velocity : array-like, shape (n, 3)
        Cartesian velocities of the n particles, in the unit that
        ``units["velocity"]`` gives.
    mass : array-like, shape (n,)
        Masses of the n particles, in the unit that ``units["mass"]`` gives.
        The length of `mass` sets the number of particles n.
    component : array-like of int, shape (n,)
        Index into `component_names` of the component each particle belongs
        to, from 0 to the number of names minus one.
    component_names : tuple of str
        Names of the model's components.
    unit_system : {"internal", "model", "physical"}
        Unit system of `position`, `velocity` and `mass`. `Sampler.sample`
        writes the particles in the unit system of the model that the
        sampler draws from (`Model.to`).
    units : dict of str to str
        Units of the positions, the velocities and the masses as text, under
        the keys ``"position"``, ``"velocity"`` and ``"mass"``.
    identity : str
        Identity string of the realization. `Sampler.sample` derives the
        identity string from the sampler's identity string, the number of
        particles and the seed.
    parent : str
        Identity string of the sampler that the particles were drawn from.

    Raises
    ------
    InputError
        If `position` or `velocity` does not have the shape (n, 3), with n
        the length of `mass`.
    InputError
        If `component` does not have the shape (n,), or holds an index that
        is negative or not below the number of `component_names`.
    InputError
        If `unit_system` is not ``"internal"``, ``"model"`` or
        ``"physical"``.

    See Also
    --------
    Sampler.sample : Draw a `Particles` object from a model.
    write_particles : Write particles to a file in a registered layout.
    read : Read a file of the native layout back into a `Particles` object.

    Notes
    -----
    Two realizations hold identical particles when the two identity strings
    are equal.

    Examples
    --------
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> particles = pp.Particles(
    ...     position=np.zeros((2, 3)),
    ...     velocity=np.ones((2, 3)),
    ...     mass=np.full(2, 0.5),
    ...     component=np.zeros(2, dtype=int),
    ...     component_names=("halo",),
    ...     unit_system="internal",
    ...     units={
    ...         "position": "internal", "velocity": "internal", "mass": "internal"
    ...     },
    ...     identity="a" * 64,
    ...     parent="b" * 64,
    ... )
    >>> len(particles), particles.position.shape
    (2, (2, 3))
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
                f"position and velocity must have the shape ({n_particles}, 3); the "
                f"shapes given are {position.shape} for position and "
                f"{velocity.shape} for velocity."
            )
        if component.shape != (n_particles,):
            raise InputError(
                f"component must have the shape ({n_particles},); the shape "
                f"given is {component.shape}."
            )
        if n_particles and not (
            0 <= np.min(component) and np.max(component) < len(self.component_names)
        ):
            raise InputError(
                "component holds an index that has no entry in component_names."
            )
        if self.unit_system not in UNIT_SYSTEMS:
            raise InputError(
                f"unit_system must be one of {list(UNIT_SYSTEMS)}; the value "
                f"given is {self.unit_system!r}."
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
        """Write the particles to a file in a registered layout.

        ``particles.write(path, layout, **options)`` is the call
        ``write_particles(particles, path, layout, **options)``.

        Parameters
        ----------
        path : str or path-like
            Path of the file to write.
        layout : str, optional
            Name of a layout registered in ``PARTICLE_WRITERS``: ``"native"``,
            ``"gamer"``, ``"gadget"``, ``"text"``, or a name added by
            `register_particle_writer`. Default: ``"native"``, which means
            the package's own HDF5 result file.
        **options
            Keyword options of the layout's writer. `write_particles` lists
            the options of every layout with their defaults.

        Returns
        -------
        None
            The file is written, read back and compared with the particles
            before the call returns.

        Raises
        ------
        InputError
            If `layout` is not registered in ``PARTICLE_WRITERS``.
        InputError
            If an option's value is outside the values the layout's writer
            takes (`write_particles` lists the conditions).
        InputError
            If the file exists and the option ``overwrite`` is not true.
        ResultError
            If the file read back differs from the particles written.

        See Also
        --------
        write_particles : The function that `Particles.write` calls.
        read : Read a file of the native layout back into a `Particles`
            object.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> particles = pp.Particles(
        ...     position=np.zeros((2, 3)),
        ...     velocity=np.ones((2, 3)),
        ...     mass=np.full(2, 0.5),
        ...     component=np.zeros(2, dtype=int),
        ...     component_names=("halo",),
        ...     unit_system="internal",
        ...     units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
        ...     identity="a" * 64,
        ...     parent="b" * 64,
        ... )
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "halo.txt")
        ...     particles.write(path, layout="text")
        ...     with open(path) as file:
        ...         n_lines = len(file.readlines())
        >>> n_lines
        9
        """
        write_particles(self, path, layout=layout, **options)


ParticleWriter = Callable[..., None]
PARTICLE_WRITERS: dict[str, ParticleWriter] = {}


def _block_size(draw_block: int, n: int) -> int:
    """Return the number of particles drawn per block in a draw of n particles.

    The block size is the smaller of `draw_block` and the smallest power of
    two at or above n. Rounding n up to a power of two keeps the number of
    distinct block sizes, and so the number of compilations, small for any n.
    """
    return min(draw_block, 1 << max(int(n) - 1, 0).bit_length())


def register_particle_writer(layout: str) -> Callable[[ParticleWriter], ParticleWriter]:
    """Return a decorator that registers a function as the writer of a layout.

    The decorated function becomes the writer that `write_particles` and
    `Particles.write` call for the name `layout`. A writer takes the
    particles, the path of the file and the writer's own keyword options. A
    file layout of another simulation code is added by one registered
    function, with no change to `Sampler` or to `Particles`.

    Parameters
    ----------
    layout : str
        Name of the layout, as passed to `write_particles`.

    Returns
    -------
    decorator : callable
        A decorator that adds the decorated function to ``PARTICLE_WRITERS``
        under the name `layout` and returns the function unchanged.

    Raises
    ------
    InputError
        If ``PARTICLE_WRITERS`` already holds a writer under the name
        `layout`. The decorator raises the error when the decorator is
        applied to a function.

    See Also
    --------
    write_particles : Write particles in a registered layout.

    Examples
    --------
    >>> from pianistpy.particles import PARTICLE_WRITERS, register_particle_writer
    >>> @register_particle_writer("count_only")
    ... def write_count(particles, path, overwrite=False):
    ...     with open(path, "w") as file:
    ...         file.write(str(len(particles)))
    >>> "count_only" in PARTICLE_WRITERS
    True
    >>> del PARTICLE_WRITERS["count_only"]
    """

    def decorator(writer: ParticleWriter) -> ParticleWriter:
        if layout in PARTICLE_WRITERS:
            raise InputError(
                f"A particle writer is already registered for the layout {layout!r}."
            )
        PARTICLE_WRITERS[layout] = writer
        return writer

    return decorator


def write_particles(
    particles: Particles,
    path: str | os.PathLike,
    layout: str = "native",
    **options: Any,
) -> None:
    """Write particles to a file in a registered layout.

    Every writer reads the file back after writing the file and compares the
    values read with the particles. The comparison is bit for bit after the
    cast to the file's number type, and to the 14 digits written after the
    decimal point for the ``"text"`` layout.

    Parameters
    ----------
    particles : Particles
        The particles to write.
    path : str or path-like
        Path of the file to write.
    layout : {"native", "gamer", "gadget", "text"} or str, optional
        Name of a layout registered in ``PARTICLE_WRITERS``.

        - ``"native"``: the package's own HDF5 result file, which `read`
          reads back into a `Particles` object.
        - ``"gamer"``: the particle file that GAMER reads at its
          initialization from file (``PAR_INIT 3``), a stream of the
          particle attributes with no header.
        - ``"gadget"``: a Gadget-derivative HDF5 initial condition in
          snapshot format 3, with the groups and attributes of the
          OpenGadget3 reference example.
        - ``"text"``: fixed-width text with a header that states the unit
          of every column.
        - any other name: a layout added by `register_particle_writer`.

        Default: ``"native"``, which means the package's own HDF5 result
        file.
    **options
        Keyword options of the layout's writer. Every option is optional.

        - Every layout takes ``overwrite`` (bool). Default: ``False``, which
          means an existing file at `path` is not replaced.
        - ``"gamer"`` takes ``order`` (``"ATT_ID"`` or ``"ID_ATT"``): one
          block per attribute (every particle's mass, then every particle's
          x, and so on) for ``"ATT_ID"``, or one record per particle for
          ``"ID_ATT"``, the two values that GAMER's parameter
          ``PAR_IC_FORMAT`` takes. Default: ``"ATT_ID"``.
        - ``"gamer"`` takes ``double_precision`` (bool): float64 values for a
          GAMER build with ``PAR_IC_FLOAT8``, float32 values otherwise.
          Default: ``True``.
        - ``"gamer"`` takes ``include_mass`` (bool): false for a GAMER run
          that sets the particle mass itself (``PAR_IC_MASS`` at or above
          zero). Default: ``True``.
        - ``"gamer"`` takes ``offset`` (array-like, shape (3,)): a vector
          added to every position, because GAMER's simulation box has its
          corner at the origin while the sampler centers the model at the
          origin. Default: ``None``, which means no shift.
        - ``"gamer"`` takes ``attributes`` (``"standard"`` or ``"legacy"``):
          ``"standard"`` writes the mass, the position and the velocity.
          ``"legacy"`` writes 13 float32 blocks, one per attribute, in the
          layout of the GAMER build used for the runs of Chiang+(2026d): the
          mass, the position, the velocity, the type 1, the tag 1, the
          energy 0, the angular momentum 0, the energy backup 0 and the
          identifiers 0 to :math:`n - 1`. Default: ``"standard"``.
        - ``"gadget"`` takes ``double_precision`` (bool): float64 arrays and
          the header flag ``Flag_DoublePrecision`` set to 1, or float32
          arrays and the flag set to 0. Default: ``False``.
        - ``"gadget"`` takes ``part_type`` (int): the particle type slot,
          from 0 to 5. Default: ``1``.
        - ``"gadget"`` takes ``hubble_param`` (float): the value written to
          the header's ``HubbleParam``, which an isolated model does not
          use. Default: ``0.7``.
        - ``"gadget"`` takes ``offset`` (array-like, shape (3,)): a vector
          added to every position. Default: ``None``, which means no shift.
        - ``"native"`` and ``"text"`` take ``overwrite`` only.

        The ``"gadget"`` writer puts equal particle masses into the header's
        mass table and writes unequal masses per particle. The ``"text"``
        header states the unit system, the unit of every column, the
        identity strings and the column names. Each line of the ``"text"``
        file holds the mass, the position and the velocity of one particle,
        followed by the particle's component index when the particles have
        more than one component.

    Returns
    -------
    None
        The file is written, read back and compared with the particles
        before the call returns.

    Raises
    ------
    InputError
        If `layout` is not registered in ``PARTICLE_WRITERS``.
    InputError
        If the file exists and the option ``overwrite`` is not true.
    InputError
        For ``"gamer"``, if ``order`` is not ``"ATT_ID"`` or ``"ID_ATT"``,
        or ``attributes`` is not ``"standard"`` or ``"legacy"``.
    InputError
        For ``"gamer"`` or ``"gadget"``, if ``offset`` does not have the
        shape (3,).
    InputError
        For ``"gamer"`` with ``attributes="legacy"``, if ``order``,
        ``double_precision`` or ``include_mass`` differs from its default.
    InputError
        For ``"gamer"`` with ``attributes="legacy"``, if the number of
        particles exceeds :math:`2^{24}`, the largest integer that a float32
        identifier holds exactly.
    InputError
        For ``"gadget"``, if ``part_type`` is not an integer from 0 to 5.
    ResultError
        If the file read back differs from the particles written.

    See Also
    --------
    Particles.write : `write_particles` as a method of `Particles`.
    register_particle_writer : Add a layout to ``PARTICLE_WRITERS``.
    read : Read a file of the native layout back into a `Particles` object.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.particles import write_particles
    >>> particles = pp.Particles(
    ...     position=np.zeros((2, 3)),
    ...     velocity=np.ones((2, 3)),
    ...     mass=np.full(2, 0.5),
    ...     component=np.zeros(2, dtype=int),
    ...     component_names=("halo",),
    ...     unit_system="internal",
    ...     units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
    ...     identity="a" * 64,
    ...     parent="b" * 64,
    ... )
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "halo.txt")
    ...     write_particles(particles, path, layout="text")
    ...     with open(path) as file:
    ...         first_line = file.readline().strip()
    >>> first_line
    '# PIANISTpy particles'
    """
    if layout not in PARTICLE_WRITERS:
        raise InputError(
            f"The particle layout {layout!r} is not a registered layout; the "
            f"registered layouts are {sorted(PARTICLE_WRITERS)}."
        )
    PARTICLE_WRITERS[layout](particles, path, **options)


@register_particle_writer("native")
def _write_native(
    particles: Particles, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write the particles as the package's own HDF5 result file, the
    ``"native"`` layout."""
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
    """Return the particles held by a stored result or by a native-layout file.

    `read` calls `_read_particles` for a file whose result kind is
    ``"particles"``.

    Parameters
    ----------
    source : StoredResult or path
        A stored result, or the path of a file written in the native layout.

    Returns
    -------
    particles : Particles
        The particles, with the identity string and the parent identity
        string stored in the file.

    Raises
    ------
    ResultError
        If the stored result's kind is not ``"particles"``.
    """
    stored = load_result(source)
    if stored.kind != RESULT_KIND:
        raise ResultError(
            f"The stored result is of kind {stored.kind!r}; a stored result of "
            f"particles is of kind {RESULT_KIND!r}."
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
    """The radius, velocity and angle tables of one component, with the
    one-component model that the three tables are built on."""

    model: Model
    radius_table: Any
    velocity_table: Any
    angle_table: Any


def _is_superposition(model: Model) -> bool:
    """Return whether the profile of every component of `model` equals the
    first component's profile and the first component's profile is a
    `TruncatedProfile`."""
    first = model.components[0].profile
    return isinstance(first, TruncatedProfile) and all(
        c.profile == first for c in model.components[1:]
    )


def _component_view(model: Model, k: int) -> Model:
    """Return component `k` of a superposed model as a model of one component.

    The returned model holds component k's DF table on component k's own
    energy grid, component k's entries of the asymptotic forms and of the
    inversion record, the shared truncated profile, and mass 1.
    """
    t = model.table
    row = t.k[k]
    n_k = (
        int(np.argmax(row == row[-1])) + 1
    )  # The superposed table pads row k with row k's last energy from n_k on.
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
    """Return whether every component of a model shares one truncated profile.

    A model returned by `superpose` shares one truncated profile across the
    model's components. A model of one component on a truncated profile also
    passes the test; `Sampler` treats a model as a superposition only when
    the model has more than one component.

    Parameters
    ----------
    model : Model
        The model to test.

    Returns
    -------
    superposed : bool
        True if the first component's profile is a truncated profile and
        every other component's profile equals the first component's
        profile; False otherwise. `is_superposition` raises no package
        exception.

    See Also
    --------
    superpose : Build a superposition from models on one truncated profile.
    component_view : One component of a superposition as a model.

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.particles import is_superposition
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> is_superposition(model)
    False
    """
    return _is_superposition(model)


def component_view(model: Model, k: int) -> Model:
    """Return component `k` of a superposed model as a model of one component.

    `Sampler` and the observables read each component of a superposition
    through `component_view`.

    Parameters
    ----------
    model : Model
        A superposed model, for which `is_superposition` returns True.
    k : int
        Index of the component, from 0 to the number of components of
        `model` minus one.

    Returns
    -------
    view : Model
        A model of one component: component k with mass 1, component k's DF
        table on component k's own energy grid, and the truncated profile
        that the components of `model` share. `component_view` raises no
        package exception for a superposed `model` and a valid `k`.

    See Also
    --------
    is_superposition : Test whether a model is a superposition.
    superpose : Build a superposition.

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.particles import component_view
    >>> parent = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()), pp.Truncation(k_t=0.05)
    ... )
    >>> children = [
    ...     pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=beta_c)))
    ...     for beta_c in (0.0, 0.2)
    ... ]
    >>> composite = pp.superpose({children[0]: 0.5, children[1]: 0.5})
    >>> view = component_view(composite, 1)
    >>> len(composite.components), len(view.components)
    (2, 1)
    """
    return _component_view(model, k)


def _radius_range(r_range: Any) -> tuple[float, float]:
    """Return the radius range (r_lo, r_hi) as two floats.

    Raise `InputError` unless r_lo and r_hi are both finite and
    ``0 <= r_lo < r_hi``.
    """
    try:
        r_lo, r_hi = (float(r) for r in r_range)
    except (TypeError, ValueError):
        raise InputError(
            f"r_range must be two radii (r_lo, r_hi); the value given is {r_range!r}."
        ) from None
    if not (np.isfinite(r_lo) and np.isfinite(r_hi) and 0.0 <= r_lo < r_hi):
        raise InputError(
            "r_range must satisfy 0 <= r_lo < r_hi with r_lo and r_hi finite; the "
            f"value given is {r_range!r}."
        )
    return r_lo, r_hi


def _velocity_range(v_range: Any) -> tuple[float, float]:
    """Return the user-specified range of velocities (v_lo, v_hi) as two floats.

    Raise `InputError` unless v_lo is finite, v_hi is not NaN and
    ``0 <= v_lo < v_hi``. The upper bound v_hi may be infinite.
    """
    try:
        v_lo, v_hi = (float(v) for v in v_range)
    except (TypeError, ValueError):
        raise InputError(
            "v_range must be two velocities (v_lo, v_hi); the value given is "
            f"{v_range!r}."
        ) from None
    if not (np.isfinite(v_lo) and not np.isnan(v_hi) and 0.0 <= v_lo < v_hi):
        raise InputError(
            "v_range must satisfy 0 <= v_lo < v_hi with v_lo finite; the value "
            f"given is {v_range!r}."
        )
    return v_lo, v_hi


def _velocity_windows(v_range: Any, r: Array) -> tuple[Any, Any]:
    """Return the user-specified range of velocities (v_lo, v_hi) at each radius of `r`.

    When v_lo and v_hi are both scalars, the return is the two floats of
    `_velocity_range`. Otherwise v_lo and v_hi are broadcast to the shape of `r`.
    Raise `InputError` unless v_lo is finite, v_hi is not NaN and
    ``0 <= v_lo < v_hi`` at every radius of `r`.
    """
    try:
        v_lo, v_hi = v_range
    except (TypeError, ValueError):
        raise InputError(
            "v_range must be two velocities (v_lo, v_hi); the value given is "
            f"{v_range!r}."
        ) from None
    if np.ndim(v_lo) == 0 and np.ndim(v_hi) == 0:
        return _velocity_range(v_range)
    try:
        lo = np.broadcast_to(np.asarray(v_lo, dtype=np.float64), r.shape)
        hi = np.broadcast_to(np.asarray(v_hi, dtype=np.float64), r.shape)
    except ValueError:
        raise InputError(
            f"The bounds of v_range must broadcast against r, of shape {r.shape}; "
            f"the bounds given are v_lo of shape {np.shape(v_lo)} and v_hi of "
            f"shape {np.shape(v_hi)}."
        ) from None
    if not (
        np.all(np.isfinite(lo))
        and not np.any(np.isnan(hi))
        and np.all((lo >= 0.0) & (lo < hi))
    ):
        raise InputError(
            "v_range must satisfy 0 <= v_lo < v_hi at every radius of r, with v_lo "
            "finite."
        )
    return lo, hi


class Sampler:
    r"""The sampling tables of a model, built once, from which particles are drawn.

    A `Sampler` builds the radius table, the velocity table and the angle table
    of a model when the `Sampler` is created. Every call of `Sampler.sample`
    and of `Sampler.velocities` reuses the three tables. For many draws from one
    model, build one `Sampler` and call `Sampler.sample` on the one `Sampler`
    for each draw.

    Parameters
    ----------
    model : Model
        The model to draw from: a model of one component, or a superposition
        returned by `superpose`.
    r_sample : float, optional
        Outer radius of the sampling, in internal units whatever the model's
        unit system. `r_sample` must lie inside the radii of the model's
        profile. A model that is not spatially finite needs `r_sample`, and a
        spatially finite model takes no `r_sample`. Default: ``None``, which
        means the truncation radius of a spatially finite model.
    r_range : tuple of float, optional
        Radii (r_lo, r_hi) between which every particle is drawn, in internal
        units, with r_lo and r_hi finite, ``0 <= r_lo < r_hi`` and r_hi at
        most the outer radius of the sampling. Each particle's radius is
        drawn from the model's enclosed mass restricted to the range, so
        every particle lies in the range and the particles carry the mass of
        the model between r_lo and r_hi. Each particle's velocity and
        directions take the random numbers of the draw without `r_range`.
        Default: ``None``, which means radii from the center to the outer
        radius of the sampling.

    Attributes
    ----------
    validation : ValidationReport
        The report of the sampler: the velocity table's checks ``row_error`` and
        ``polish_residual``, and the model's own report as the parent
        report. The report's tier is the band of the largest relative error:
        ``"silent"`` at or below :math:`10^{-6}`, ``"recorded"`` above
        :math:`10^{-6}` and at or below :math:`10^{-4}`, ``"warned"`` above
        :math:`10^{-4}` and at or below :math:`10^{-2}`, and ``"refused"``
        above :math:`10^{-2}`. A sampler whose tier is ``"recorded"`` writes
        one line at level INFO to the package's log. A sampler whose tier is
        ``"warned"`` or ``"refused"`` also issues an `AccuracyWarning`.
    identity : str
        Identity string of the sampler, derived from the model's identity
        string, `r_sample`, `r_range`, the settings and the package version.

    Raises
    ------
    InputError
        If `model` is not a `Model`.
    InputError
        If `model` has more than one component and is not a superposition.
    InputError
        If `r_sample` is given for a spatially finite model, is missing for a
        model that is not spatially finite, or lies outside the radii of the
        model's profile.
    InputError
        If `r_range` is not two finite radii with ``0 <= r_lo < r_hi``, if r_hi
        is beyond the outer radius of the sampling, if r_hi is at or below the
        innermost radius of the radius table, or if r_lo is at or above the
        outermost radius of the radius table.
    AccuracyError
        If the model's tier is ``"refused"`` and the model's settings do not
        set ``write_anyway=True`` (`Settings`).
    AccuracyError
        If a check of the velocity table exceeds :math:`10^{-2}` and the settings
        do not set ``write_anyway=True``. With ``write_anyway=True``, the
        `Sampler` is built with the tier ``"refused"`` and issues an
        `AccuracyWarning`.
    AccuracyError
        If the velocity table cannot resolve the distribution of velocity at a
        potential with the resolution that the settings allow.
    UnphysicalModelError
        If the DF below the DF table's lowest energy falls so steeply toward
        :math:`\mathcal{K} = 0` that the mass at energies below the DF table's
        lowest energy is not finite.
    ResultError
        If a sampling table fails the table's own consistency check: an
        inverse that is not finite and increasing, or a cumulative
        distribution that is not monotone.

    See Also
    --------
    Sampler.sample : Draw particles from the model.
    Sampler.velocities : Draw one velocity at each user-specified 3D radius.
    Particles : The result of `Sampler.sample`.
    superpose : Build a superposition, which a `Sampler` also takes.

    Notes
    -----
    A model with a non-negative DF is not necessarily sampled within
    :math:`10^{-4}`. The velocity table doubles the velocity table's resolution
    until the velocity table's own error is below the setting
    ``sampling["row_error_goal"]`` (:math:`10^{-6}` with the default
    settings, :math:`10^{-4}` with the quick settings) or until the
    resolution reaches the setting ``sampling["n_rows_max"]``. For an
    Osipkov-Merritt or Cuddeford-Osipkov-Merritt (COM) Type I DF whose
    anisotropy radius :math:`R_{\rm a}` is close to the smallest anisotropy
    radius at which the DF is non-negative, the resolution reaches
    ``sampling["n_rows_max"]`` first, and the velocity table's error grows as
    :math:`R_{\rm a}` decreases toward the smallest anisotropy radius with a
    non-negative DF. The report in
    `validation` gives the velocity table's error for each model. The page on
    particles in the user guide lists, for each anisotropy model, the
    anisotropy radius at which the velocity table's error exceeds
    :math:`10^{-4}` and :math:`10^{-2}`.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> sampler = pp.Sampler(model, r_sample=100.0)
    >>> particles = sampler.sample(1000, seed=1)
    >>> particles.position.shape, particles.unit_system
    ((1000, 3), 'internal')
    """

    def __init__(
        self,
        model: Model,
        r_sample: float | None = None,
        r_range: tuple[float, float] | None = None,
    ) -> None:
        if not isinstance(model, Model):
            raise InputError("Sampler takes a Model as its first argument.")
        superposed = len(model.components) > 1 and _is_superposition(model)
        if len(model.components) > N_COMPONENTS_MAX and not superposed:
            raise InputError(
                "PIANISTpy samples models of one component; got "
                f"{len(model.components)} components."
            )
        settings = model.settings
        # A model whose tier is "refused" (an error above 10^-2) is sampled only
        # when the model's settings set write_anyway=True, the request that
        # built the model in the first place.
        refuse_unless_asked(
            f"The model {model.components[0].name!r}", model.validation, settings
        )
        self.model = model
        self.r_sample = r_sample
        # A superposed model gets one set of tables per component, because each
        # component has its own DF on the shared density; any other model gets
        # one set of tables.
        views = [_component_view(model, k) for k in range(len(model.components))]
        if not superposed:
            views = [model]
        self.parts = [
            _Part(
                view,
                stages.radius_table(view, settings, r_sample),
                stages.velocity_table(view, settings),
                stages.angle_table(view, settings),
            )
            for view in views
        ]
        self.radius_table = self.parts[0].radius_table
        self.r_range = None if r_range is None else _radius_range(r_range)
        r_edge = self.radius_table.r_sample[0]
        if self.r_range is not None and self.r_range[1] > r_edge:
            raise InputError(
                f"The upper radius of r_range, {self.r_range[1]!r}, lies beyond the "
                f"outer radius of the sampling, {r_edge!r}."
            )
        # The enclosed-mass fractions at r_lo and r_hi, per component. The
        # components of a superposition share the density, so every component
        # has identical fractions at r_lo and r_hi.
        self.bounds = [
            (0.0, 1.0)
            if self.r_range is None
            else stages.radius_bounds(part.radius_table, 0, self.r_range)
            for part in self.parts
        ]
        self.velocity_table = self.parts[0].velocity_table
        self.angle_table = self.parts[0].angle_table
        masses = np.array([c.mass for c in model.components])
        self.weights = masses / masses.sum()
        worst = max(
            (part.velocity_table.validation for part in self.parts),
            key=lambda report: report.level,
        )
        owners = {
            key: max(
                self.parts,
                key=lambda p: p.velocity_table.validation.checks[key].error,
            )
            for key in worst.checks
        }
        table_checks = ValidationReport(
            keys=worst.keys,
            checks={
                key: owners[key].velocity_table.validation.checks[key]
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
                    + (
                        f" ({owners[key].velocity_table.row_cap})"
                        if key == "row_error" and owners[key].velocity_table.row_cap
                        else ""
                    )
                    for key, check in table_checks.checks.items()
                    if check.level > 0
                )
                or "the tier of its model",
            )
        record: dict[str, Any] = {"model": model.identity, "r_sample": r_sample}
        if self.r_range is not None:
            record["r_range"] = list(self.r_range)
        self.identity = identity_string(
            record,
            settings.to_dict(),
            __version__,
        )

    def sample(self, n: int, seed: int = 0) -> Particles:
        """Return n equal-mass particles drawn from the model.

        Each particle is determined by the seed, the particle's component and
        the particle's index, and not by the number of particles requested.

        Parameters
        ----------
        n : int
            Number of particles, from 1 to :math:`2^{31} - 1`. A larger
            number of particles is drawn in several calls with distinct seeds.
        seed : int, optional
            Seed of the random numbers. One seed gives identical particles for
            every value of the setting ``sampling["draw_block"]``
            (`Settings`). Default: ``0``.

        Returns
        -------
        particles : Particles
            The n particles, in the unit system of the model (`Model.to`).
            Every particle has the mass :math:`M_{\\rm s}/n`, where
            :math:`M_{\\rm s}` is the model's mass inside the outer radius of
            the sampling, or between r_lo and r_hi when the `Sampler` has an
            `r_range`. The draw is made in internal units, and the positions,
            velocities and masses are converted to the model's unit system
            at the output.

        Raises
        ------
        InputError
            If `n` is not a positive integer.
        InputError
            If `n` is :math:`2^{31}` or more.
        AccuracyError
            If a particle lies outside the phase-space support of the
            particle's component, or if the velocity rebuilt from the
            particle's arrays misses the support's bound by more than the
            rounding of the particle's own variables.

        See Also
        --------
        Sampler.velocities : Draw one velocity at each user-specified 3D radius.
        write_particles : Write the particles to a file.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> model = pp.solve(
        ...     pp.Component(pp.DoublePowerLaw.hernquist()),
        ...     settings=pp.Settings(quick=True),
        ... )
        >>> particles = pp.Sampler(model, r_sample=100.0).sample(1000, seed=1)
        >>> len(particles), bool(np.all(particles.mass == particles.mass[0]))
        (1000, True)
        """
        if not (isinstance(n, int) and n > 0):
            raise InputError(f"n must be a positive integer; the value given is {n!r}.")
        if n > MAX_PARTICLES:
            raise InputError(
                f"n = {n} is at least 2^31, and one draw holds at most 2^31 - 1 "
                "particles, because the particles' random keys are drawn on int32 "
                "indices and the Gadget writer's identifiers are uint32; draw the "
                "particles in several calls of sample with distinct seeds."
            )
        component = 0
        # Every block has one size (the last block is padded, then cut), so the
        # working memory is one block's and one compilation serves every block.
        block = _block_size(int(self.model.settings.sampling["draw_block"]), n)
        units = self.model.units
        # The analytic profile that fixes the units: for a truncated model, the
        # untruncated profile that the truncated profile was built from.
        profile = self.model.components[component].unit_profile
        prepared = [
            stages.velocity_draw_tables(part.velocity_table, part.model, component)
            for part in self.parts
        ]
        arrays = {"position": np.empty((n, 3)), "velocity": np.empty((n, 3))}
        labels = np.zeros(n, dtype=np.int64)
        edges = np.cumsum(self.weights)[:-1]
        unit_text = {}
        for start in range(0, n, block):
            size = min(block, n - start)
            # A particle's random numbers are the numbers of a one-component
            # draw. A particle of a superposed model takes its component from a
            # separate random channel, with probabilities equal to the weights,
            # and is drawn from the chosen component's tables.
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
                r = stages.draw_radius(
                    part.radius_table, component, numbers, self.bounds[k]
                )
                cos_phi, sin_phi, factor = stages.draw_angle(
                    part.angle_table, part.model, component, r, numbers
                )
                v, x, d, psi = stages.draw_velocity(
                    part.velocity_table, part.model, component, r, factor, numbers,
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
            # The unit conversion is an elementwise product, so each block is
            # converted on its own and written into the output arrays.
            for name, kind, values in (
                ("position", "length", pos[:size]),
                ("velocity", "velocity", vel[:size]),
            ):
                converted = units.convert(kind, values, profile)
                unit_text[name] = str(getattr(converted, "unit", units.system))
                arrays[name][start : start + size] = getattr(
                    converted, "value", converted
                )
        u_lo, u_hi = self.bounds[0]
        mass_drawn = self.radius_table.mass_sample[component] * (u_hi - u_lo)
        converted = units.convert("mass", np.full(n, mass_drawn / n), profile)
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

    def velocities(
        self, r: Any, seed: int = 0, v_range: tuple[Any, Any] | None = None
    ) -> tuple[Array, Array]:
        """Return one velocity drawn at each user-specified 3D radius, within a
        user-specified range of velocities.

        Parameters
        ----------
        r : array-like, shape (n,)
            User-specified 3D radii, in internal units whatever the model's
            unit system. Every radius is finite, positive and at most the outer
            radius of the sampling.
        seed : int, optional
            Seed of the random numbers. The velocity at the i-th radius takes the
            velocity's random number of particle i of `Sampler.sample` at the
            same seed, mapped into the user-specified range. For a
            superposition, the component of the i-th velocity is chosen with
            probabilities proportional to each component's weight times the
            component's fraction of velocities in the range. Default: ``0``.
        v_range : tuple of float or of array-like, optional
            The user-specified range of velocities (v_lo, v_hi), in internal
            units, with v_lo finite, v_hi finite or infinite, and
            ``0 <= v_lo < v_hi``. Each bound is a float, which applies at
            every radius, or an array that broadcasts against `r`, which
            gives a bound at each radius (for example, a bound in units of
            each radius's escape velocity). Default: ``None``, which means the
            velocities from 0 to infinity, the whole distribution of velocity.

        Returns
        -------
        v : array, shape (n,)
            The velocity at each radius of `r`, in internal units, drawn from the
            model's distribution of velocity at the radius, restricted to the
            user-specified range. The velocity is NaN at a radius where the
            user-specified range holds no part of the distribution of velocity.
        fraction : array, shape (n,)
            The fraction of the distribution of velocity at each radius that
            lies in the user-specified range. For a superposition, the
            fraction is the sum over components of each component's weight
            times the component's fraction.

        Raises
        ------
        InputError
            If a component has a finite anisotropy radius (Osipkov-Merritt
            or Cuddeford-Osipkov-Merritt). For such a component the velocity is
            the scaled velocity times a factor of the drawn angle, so the
            user-specified range of velocities depends on the angle, and no table
            holds the distribution of velocity at a radius with the angle
            integrated out.
        InputError
            If `r` is not one-dimensional, or a radius in `r` is not finite,
            not positive or beyond the outer radius of the sampling.
        InputError
            If `v_range` is not two bounds, or if v_lo is not finite, v_hi is
            NaN or ``0 <= v_lo < v_hi`` fails at a radius.
        InputError
            If the bounds of `v_range` do not broadcast against `r`.

        See Also
        --------
        Sampler.sample : Draw particles, positions and velocities included.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> model = pp.solve(
        ...     pp.Component(pp.DoublePowerLaw.hernquist()),
        ...     settings=pp.Settings(quick=True),
        ... )
        >>> sampler = pp.Sampler(model, r_sample=100.0)
        >>> v, frac = sampler.velocities([0.5, 1.0, 2.0], seed=1, v_range=(0.1, 0.3))
        >>> v.shape, bool(np.all((frac > 0.0) & (frac < 1.0)))
        ((3,), True)
        """
        coupled = [
            c.name for c in self.model.components if np.isfinite(c.anisotropy.r_a)
        ]
        if coupled:
            raise InputError(
                "Sampler.velocities draws velocities in a user-specified range only "
                f"for components of constant anisotropy; the components {coupled!r} "
                "have a finite anisotropy radius. With a finite anisotropy radius, "
                "the velocity magnitude is the scaled velocity times a factor of the "
                "drawn angle, so "
                "the range of velocities couples to the angle, and no table holds the "
                "velocity's distribution at a radius with the angle integrated out."
            )
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        r_edge = float(self.radius_table.r_sample[0])
        if (
            r.ndim != 1
            or not np.all(np.isfinite(r))
            or np.any(r <= 0.0)
            or np.any(r > r_edge)
        ):
            raise InputError(
                f"r must hold finite radii with 0 < r <= {r_edge!r}, the outer "
                "radius of the sampling."
            )
        v_lo, v_hi = (0.0, np.inf) if v_range is None else _velocity_windows(v_range, r)
        n = r.size
        # The velocities are drawn in blocks of the size `sample` uses, the last
        # block padded with its last radius and then cut, so one compilation
        # serves every block.
        block = _block_size(int(self.model.settings.sampling["draw_block"]), n)
        prepared = [
            stages.velocity_draw_tables(part.velocity_table, part.model, 0)
            for part in self.parts
        ]
        v = np.empty(n)
        fraction = np.empty(n)
        for start in range(0, n, block):
            size = min(block, n - start)

            def padded(x: Any, start: int = start, size: int = size) -> Any:
                if np.ndim(x) == 0:
                    return x
                x = np.broadcast_to(x, r.shape)[start : start + size]
                return np.concatenate([x, np.full(block - size, x[-1])])

            r_b = padded(r)
            v_lo_b, v_hi_b = padded(v_lo), padded(v_hi)
            numbers = stages.random_numbers_block(seed, 0, start, block)
            bounds, draws = [], []
            for part, prep in zip(self.parts, prepared, strict=True):
                u_lo, u_hi = stages.velocity_bounds(
                    part.velocity_table, part.model, 0, r_b, (v_lo_b, v_hi_b), prep
                )
                mapped = dataclasses.replace(
                    numbers,
                    velocity_magnitude=u_lo
                    + (u_hi - u_lo) * np.asarray(numbers.velocity_magnitude),
                )
                v_k = stages.draw_velocity(
                    part.velocity_table,
                    part.model,
                    0,
                    r_b,
                    np.ones(block),
                    mapped,
                    prep,
                )[0]
                bounds.append(u_hi - u_lo)
                draws.append(v_k)
            weights = self.weights if len(self.parts) > 1 else np.ones(1)
            shares = np.asarray(weights)[:, None] * np.array(bounds)
            fraction_b = shares.sum(axis=0)
            chosen = np.zeros(block, dtype=np.int64)
            if len(self.parts) > 1:
                # Each velocity's component is chosen from the component channel of
                # the particle's random key, with probabilities proportional to
                # each component's share of the user-specified range.
                u = stages.component_numbers(seed, start, block)
                with np.errstate(invalid="ignore", divide="ignore"):
                    edges = np.cumsum(shares, axis=0)[:-1] / fraction_b
                chosen = np.sum(u[None, :] >= edges, axis=0)
            v_b = np.choose(chosen, np.array(draws))
            v[start : start + size] = np.where(fraction_b > 0.0, v_b, np.nan)[:size]
            fraction[start : start + size] = fraction_b[:size]
        return v, fraction
