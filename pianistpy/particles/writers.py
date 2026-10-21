"""The particle writers for the file layouts of simulation codes and for text.

Importing `pianistpy.particles` registers `write_gamer`, `write_gadget` and
`write_text` with `register_particle_writer`. A user reaches each writer through
`Particles.write` or `write_particles` with the layout's name:

* ``"gamer"``: the particle file that GAMER reads at its initialization from
  file (``PAR_INIT 3``), a stream of the particle attributes with no header.
* ``"gadget"``: a Gadget-derivative HDF5 initial condition in snapshot
  format 3, with the groups and attributes of the OpenGadget3 reference
  example.
* ``"text"``: fixed-width text with a header that states the unit of every
  column.

Every writer reads the file back and compares the values read with the
values written, bit for bit after the cast to the file's number type, and
raises `ResultError` when any value differs. Every writer handles the
particles in slices of ``slice_size()`` particles, the setting
``sampling["draw_block"]`` of the default settings, so the writer's working
memory beyond the particles themselves is one slice's for any number of
particles.
"""

__all__ = [
    "GAMER_ATTRIBUTES",
    "GAMER_LEGACY_ATTRIBUTES",
    "ID_LIMIT_FLOAT32",
    "TEXT_COLUMNS",
]

import itertools
import os
from typing import Any

import h5py
import numpy as np

from ..exceptions import InputError, ResultError
from ..settings import Settings
from .core import Particles, register_particle_writer

# GAMER's particle attributes in the order that GAMER's Par_Init_ByFile.cpp
# reads them: the mass (when the file holds the mass), then the Cartesian
# position and velocity.
GAMER_ATTRIBUTES = ("mass", "x", "y", "z", "vx", "vy", "vz")
# The 13 float32 attribute blocks of the GAMER build used for the runs of
# Chiang+(2026d): the 7 attributes of GAMER_ATTRIBUTES, then the type, the tag,
# the energy, the angular momentum, the energy backup and the identifier.
GAMER_LEGACY_ATTRIBUTES = GAMER_ATTRIBUTES + (
    "type",
    "tag",
    "energy",
    "angular_momentum",
    "energy_backup",
    "id",
)
# The largest integer up to which every integer is exact in float32, 2^24: an
# identifier above 2^24 cannot be written exactly in the legacy layout.
ID_LIMIT_FLOAT32 = 2**24
TEXT_COLUMNS = ("mass", "x", "y", "z", "vx", "vy", "vz")
TEXT_FIELD = "{:<25.14e}"


def slice_size() -> int:
    """Return the number of particles a writer handles at once, the setting
    ``sampling["draw_block"]`` of the default settings."""
    return int(Settings().sampling["draw_block"])


def _slices(n: int) -> list[slice]:
    step = slice_size()
    return [slice(a, min(a + step, n)) for a in range(0, n, step)]


def _refused(path: str) -> ResultError:
    return ResultError(f"The file {path!r} does not read back as written.")


def _target(path: str | os.PathLike, overwrite: bool) -> str:
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise InputError(
            f"The file {path!r} exists; pass overwrite=True to replace the file."
        )
    return path


def _offset(offset) -> np.ndarray | None:
    if offset is None:
        return None
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,):
        raise InputError(
            f"offset must have the shape (3,); the shape given is {offset.shape}."
        )
    return offset.reshape(1, 3)


def _shifted(
    particles: Particles, offset: np.ndarray | None, part: slice
) -> np.ndarray:
    position = np.asarray(particles.position[part], dtype=np.float64)
    return position if offset is None else position + offset


def _size_check(path: str, expected: int) -> None:
    if os.path.getsize(path) != expected:
        raise ResultError(
            f"The file {path!r} holds {os.path.getsize(path)} bytes; the writer "
            f"wrote {expected} bytes."
        )


@register_particle_writer("gamer")
def write_gamer(
    particles: Particles,
    path: str | os.PathLike,
    order: str = "ATT_ID",
    double_precision: bool = True,
    include_mass: bool = True,
    offset=None,
    attributes: str = "standard",
    overwrite: bool = False,
) -> None:
    """Write the particles as GAMER's particle file, a stream of the particle
    attributes with no header.

    Parameters
    ----------
    particles : Particles
        The particles to write.
    path : str or path-like
        Path of the file to write.
    order : {"ATT_ID", "ID_ATT"}, optional
        ``"ATT_ID"`` writes one block per attribute (every particle's mass,
        then every particle's x, and so on). ``"ID_ATT"`` writes one record
        per particle. GAMER's parameter ``PAR_IC_FORMAT`` takes the same
        two values. Default: ``"ATT_ID"``.
    double_precision : bool, optional
        Write float64 values, for a GAMER build with ``PAR_IC_FLOAT8``, or
        float32 values. Default: ``True``.
    include_mass : bool, optional
        Write the particle mass. False suits a GAMER run that sets the
        particle mass itself (``PAR_IC_MASS`` at or above zero). GAMER's
        parameter ``PAR_IC_TYPE`` sets the particle type for either value.
        Default: ``True``.
    offset : array-like, shape (3,), optional
        A vector added to every position, because GAMER's simulation box has
        its corner at the origin while the sampler centers the model at the
        origin. Default: ``None``, which means no shift.
    attributes : {"standard", "legacy"}, optional
        ``"standard"`` writes the mass, the position and the velocity.
        ``"legacy"`` writes the 13 float32 blocks of
        `GAMER_LEGACY_ATTRIBUTES`, the layout of the GAMER build used for the
        runs of Chiang+(2026d): the type 1, the tag 1, the energy 0, the
        angular momentum 0, the energy backup 0 and the identifiers 0 to
        n - 1 after the mass, the position and the velocity. ``"legacy"``
        requires `order`, `double_precision` and `include_mass` at their
        defaults. Default: ``"standard"``.
    overwrite : bool, optional
        Replace an existing file at `path`. Default: ``False``, which means
        an existing file raises `InputError`.

    Raises
    ------
    InputError
        If `order` is not ``"ATT_ID"`` or ``"ID_ATT"``, or `attributes` is
        not ``"standard"`` or ``"legacy"``.
    InputError
        If the file exists and `overwrite` is false.
    InputError
        If `offset` does not have the shape (3,).
    InputError
        If `attributes` is ``"legacy"`` and `order`, `double_precision` or
        `include_mass` differs from its default.
    InputError
        If `attributes` is ``"legacy"`` and the number of particles exceeds
        ``ID_LIMIT_FLOAT32`` (:math:`2^{24}`), above which a float32
        identifier is not exact.
    ResultError
        If the file read back differs from the values written, or the file's
        size differs from the number of bytes written.
    """
    if order not in ("ATT_ID", "ID_ATT"):
        raise InputError(
            f"order must be 'ATT_ID' or 'ID_ATT'; the value given is {order!r}."
        )
    if attributes not in ("standard", "legacy"):
        raise InputError(
            "attributes must be 'standard' or 'legacy'; the value given is "
            f"{attributes!r}."
        )
    path = _target(path, overwrite)
    n = len(particles)
    shift = _offset(offset)
    if attributes == "legacy":
        if not (order == "ATT_ID" and double_precision and include_mass):
            raise InputError(
                "The legacy layout is attribute-major float32 with the mass, so "
                "order, double_precision and include_mass must be left at their "
                "defaults."
            )
        if n > ID_LIMIT_FLOAT32:
            raise InputError(
                f"The legacy layout stores identifiers as float32, which carries "
                f"integers exactly up to {ID_LIMIT_FLOAT32}; the {n} particles "
                "need more identifiers than float32 carries exactly."
            )
        dtype: np.dtype[Any] = np.dtype(np.float32)
        n_columns = len(GAMER_LEGACY_ATTRIBUTES)
    else:
        dtype = np.dtype(np.float64 if double_precision else np.float32)
        n_columns = len(GAMER_ATTRIBUTES) - (0 if include_mass else 1)

    def columns(part: slice) -> np.ndarray:
        # The attributes of the particles in the slice `part` as float64, one
        # row per attribute.
        position = _shifted(particles, shift, part)
        velocity = np.asarray(particles.velocity[part], dtype=np.float64)
        mass = np.asarray(particles.mass[part], dtype=np.float64)
        size = position.shape[0]
        rows = [mass] if include_mass else []
        rows += [position[:, 0], position[:, 1], position[:, 2]]
        rows += [velocity[:, 0], velocity[:, 1], velocity[:, 2]]
        if attributes == "legacy":
            rows += [np.ones(size), np.ones(size), np.zeros(size), np.zeros(size)]
            rows += [np.zeros(size), np.arange(part.start, part.stop, dtype=np.float64)]
        return np.stack(rows)

    parts = _slices(n)
    # Each slice is written once. In the order "ATT_ID", a slice fills the
    # slice's stretch of every attribute's block; in the order "ID_ATT", a slice
    # appends the slice's records. Either way the file's bytes equal the bytes
    # of the whole array cast to the file's number type.
    item = dtype.itemsize
    with open(path, "wb") as file:
        for part in parts:
            block = columns(part)
            if order == "ATT_ID":
                for c in range(n_columns):
                    file.seek((c * n + part.start) * item)
                    np.ascontiguousarray(block[c], dtype=dtype).tofile(file)
            else:
                np.ascontiguousarray(block.T, dtype=dtype).tofile(file)
    with open(path, "rb") as reader:
        for part in parts:
            want = np.ascontiguousarray(columns(part), dtype=dtype)
            size = part.stop - part.start
            if order == "ATT_ID":
                pieces = []
                for c in range(n_columns):
                    reader.seek((c * n + part.start) * item)
                    pieces.append(np.fromfile(reader, dtype=dtype, count=size))
                same = all(piece.shape == (size,) for piece in pieces) and (
                    np.array_equal(np.stack(pieces), want)
                )
            else:
                reader.seek(part.start * n_columns * item)
                back = np.fromfile(reader, dtype=dtype, count=size * n_columns)
                same = back.shape == (size * n_columns,) and np.array_equal(
                    back.reshape(size, n_columns).T, want
                )
            if not same:
                raise _refused(path)
    _size_check(path, n * n_columns * item)


@register_particle_writer("gadget")
def write_gadget(
    particles: Particles,
    path: str | os.PathLike,
    double_precision: bool = False,
    part_type: int = 1,
    hubble_param: float = 0.7,
    offset=None,
    overwrite: bool = False,
) -> None:
    """Write the particles as a Gadget-derivative HDF5 initial condition in
    snapshot format 3.

    The file has the groups of the OpenGadget3 reference example. The group
    ``/Header`` holds the particle counts, the mass table and cosmological
    parameters set to zero for an isolated model. The group
    ``/PartType<n>`` holds ``Coordinates``, ``Velocities`` and
    ``ParticleIDs`` numbered from 1. When all particle masses are equal, the
    mass is written to the header's mass table only. When the masses differ,
    the masses are written per particle in ``Masses`` and the mass table's
    entry for the particle type is 0.

    Parameters
    ----------
    particles : Particles
        The particles to write.
    path : str or path-like
        Path of the file to write.
    double_precision : bool, optional
        Write float64 arrays with the header flag ``Flag_DoublePrecision``
        set to 1, or float32 arrays with the flag set to 0. Default:
        ``False``.
    part_type : int, optional
        The particle type slot, an integer from 0 to 5. Default: ``1``.
    hubble_param : float, optional
        The value written to the header's ``HubbleParam``, which an isolated
        model does not use. Default: ``0.7``.
    offset : array-like, shape (3,), optional
        A vector added to every position. Default: ``None``, which means no
        shift.
    overwrite : bool, optional
        Replace an existing file at `path`. Default: ``False``, which means
        an existing file raises `InputError`.

    Raises
    ------
    InputError
        If `part_type` is not an integer from 0 to 5.
    InputError
        If the file exists and `overwrite` is false.
    InputError
        If `offset` does not have the shape (3,).
    ResultError
        If the file read back differs from the values written.
    """
    if not 0 <= int(part_type) < 6 or int(part_type) != part_type:
        raise InputError(
            "part_type must be an integer from 0 to 5; the value given is "
            f"{part_type!r}."
        )
    path = _target(path, overwrite)
    n = len(particles)
    dtype = np.float64 if double_precision else np.float32
    shift = _offset(offset)
    mass = np.asarray(particles.mass, dtype=np.float64)
    equal = n > 0 and bool(np.all(mass == mass[0]))
    counts = np.zeros(6, dtype=np.int32)
    counts[part_type] = n
    table = np.zeros(6, dtype=np.float64)
    if equal:
        table[part_type] = mass[0]
    parts = _slices(n)

    def arrays(part: slice) -> dict[str, np.ndarray]:
        out = {
            "Coordinates": _shifted(particles, shift, part).astype(dtype),
            "Velocities": np.asarray(particles.velocity[part], dtype=np.float64).astype(
                dtype
            ),
            "ParticleIDs": np.arange(part.start + 1, part.stop + 1, dtype=np.uint32),
        }
        if not equal:
            out["Masses"] = mass[part].astype(dtype)
        return out

    shapes = {
        "Coordinates": ((n, 3), dtype),
        "Velocities": ((n, 3), dtype),
        "ParticleIDs": ((n,), np.uint32),
    }
    if not equal:
        shapes["Masses"] = ((n,), dtype)
    with h5py.File(path, "w") as file:
        header = file.create_group("Header")
        header.attrs["BoxSize"] = np.float64(0.0)
        for name in (
            "Flag_Cooling",
            "Flag_DoublePrecision",
            "Flag_Feedback",
            "Flag_IC_Info",
            "Flag_Metals",
            "Flag_Sfr",
            "Flag_StellarAge",
        ):
            on = name == "Flag_DoublePrecision" and double_precision
            header.attrs[name] = np.int32(1 if on else 0)
        header.attrs["HubbleParam"] = np.float64(hubble_param)
        header.attrs["MassTable"] = table
        header.attrs["NumFilesPerSnapshot"] = np.int32(1)
        header.attrs["NumPart_ThisFile"] = counts
        header.attrs["NumPart_Total"] = counts.astype(np.uint32)
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        for name in ("Omega0", "OmegaLambda", "Redshift", "Time"):
            header.attrs[name] = np.float64(0.0)
        group = file.create_group(f"PartType{part_type}")
        # Each dataset is created at its full shape and then filled slice by
        # slice, so the file holds the dataset as one write of the whole array
        # would hold the dataset.
        for name, (shape, kind) in shapes.items():
            dataset = group.create_dataset(name, shape=shape, dtype=kind)
            for part in parts:
                dataset[part] = arrays(part)[name]
    with h5py.File(path, "r") as file:
        group = file[f"PartType{part_type}"]
        same = int(file["Header"].attrs["NumPart_ThisFile"][part_type]) == n and (
            not equal or float(file["Header"].attrs["MassTable"][part_type]) == mass[0]
        )
        same = same and all(
            group[name].shape == shape and group[name].dtype == kind
            for name, (shape, kind) in shapes.items()
        )
        for part in parts:
            if not same:
                break
            same = all(
                np.array_equal(group[name][part], values)
                for name, values in arrays(part).items()
            )
    if not same:
        raise _refused(path)


@register_particle_writer("text")
def write_text(
    particles: Particles, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write the particles as fixed-width text with a header.

    Each header line starts with ``#``. The header states the unit system,
    the identity string of the realization, the identity string of the
    realization's parent, the component names, the column names and the unit
    of every column. Each following line holds the mass, the position and
    the velocity of one particle, every value in a field 25 characters wide
    with 14 digits after the decimal point. When the particles have more
    than one component, the particle's component index follows the velocity.

    Parameters
    ----------
    particles : Particles
        The particles to write.
    path : str or path-like
        Path of the file to write.
    overwrite : bool, optional
        Replace an existing file at `path`. Default: ``False``, which means
        an existing file raises `InputError`.

    Raises
    ------
    InputError
        If the file exists and `overwrite` is false.
    ResultError
        If a value read back differs from the value written, rounded to the
        14 digits after the decimal point, or if the file holds a different
        number of lines from the number of particles.
    """
    path = _target(path, overwrite)
    n = len(particles)
    multi = len(particles.component_names) > 1
    units = particles.units
    columns = list(TEXT_COLUMNS) + (["component"] if multi else [])
    column_units = (
        [units.get("mass", "")]
        + [units.get("position", "")] * 3
        + [units.get("velocity", "")] * 3
        + (["index into the components"] if multi else [])
    )
    head = [
        "# PIANISTpy particles",
        f"# unit_system: {particles.unit_system}",
        f"# identity: {particles.identity}",
        f"# parent: {particles.parent}",
        "# components: " + ", ".join(particles.component_names),
        "# columns: " + " ".join(columns),
        "# units: " + " | ".join(column_units),
    ]
    parts = _slices(n)

    def values(part: slice) -> np.ndarray:
        return np.column_stack(
            [particles.mass[part], particles.position[part], particles.velocity[part]]
        ).astype(np.float64)

    component = np.asarray(particles.component, dtype=np.int64)
    with open(path, "w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(head) + "\n")
        for part in parts:
            block = values(part)
            for i, row_values in enumerate(block):
                row = "".join(TEXT_FIELD.format(v) for v in row_values)
                if multi:
                    row += f"{int(component[part.start + i]):<6d}"
                file.write(row.rstrip() + "\n")
    same = True
    width = len(TEXT_COLUMNS)
    with open(path, encoding="ascii") as file:
        lines = (line for line in file if not line.startswith("#"))
        for part in parts:
            block = values(part)
            rows = list(itertools.islice(lines, part.stop - part.start))
            back = np.loadtxt(rows, ndmin=2) if rows else np.zeros((0, width))
            rounded = np.array(
                [[float(TEXT_FIELD.format(v).strip()) for v in r] for r in block]
            )
            same = (
                back.shape[0] == block.shape[0]
                and np.array_equal(back[:, :width], rounded)
                and (
                    not multi
                    or np.array_equal(back[:, -1].astype(np.int64), component[part])
                )
            )
            if not same:
                break
        same = same and next(lines, None) is None
    if not same:
        raise _refused(path)
