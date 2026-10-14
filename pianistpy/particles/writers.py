"""Particles in the layouts of simulation codes, and as text.

Each layout is one registered writer (`register_particle_writer`), reached
through `Particles.write(path, layout=...)` or `write_particles`:

* ``"gamer"``: GAMER's particle file, read by its initialization from file
  (``PAR_INIT 3``): a headerless stream of the attributes, as the working
  reader's exporter writes it.
* ``"gadget"``: a Gadget-derivative HDF5 initial condition (snapshot format
  3), as the OpenGadget3 reference example lays it out.
* ``"text"``: fixed-width text with the units in the file's own header.

Every writer reads its file back and compares it with what it wrote, bit for
bit after the cast to the file's type, and refuses the file otherwise.
"""

__all__ = [
    "GAMER_ATTRIBUTES",
    "GAMER_LEGACY_ATTRIBUTES",
    "ID_LIMIT_FLOAT32",
    "TEXT_COLUMNS",
]

import os

import h5py
import numpy as np

from ..exceptions import InputError, ResultError
from .core import Particles, register_particle_writer

# GAMER's attributes in the order of Par_Init_ByFile.cpp: the mass (where it
# is in the file) and the Cartesian position and velocity.
GAMER_ATTRIBUTES = ("mass", "x", "y", "z", "vx", "vy", "vz")
# The thirteen float32 blocks of the build the paper's runs were made with:
# the seven above, then the type, the tag, the energy, the angular momentum,
# the energy backup and the identifier.
GAMER_LEGACY_ATTRIBUTES = GAMER_ATTRIBUTES + (
    "type",
    "tag",
    "energy",
    "angular_momentum",
    "energy_backup",
    "id",
)
# The largest integer a float32 carries exactly: an identifier above it is
# not representable in the legacy layout.
ID_LIMIT_FLOAT32 = 2**24
TEXT_COLUMNS = ("mass", "x", "y", "z", "vx", "vy", "vz")
TEXT_FIELD = "{:<25.14e}"


def _target(path: str | os.PathLike, overwrite: bool) -> str:
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise InputError(f"{path!r} exists; pass overwrite=True to replace it.")
    return path


def _shifted(particles: Particles, offset) -> np.ndarray:
    position = np.asarray(particles.position, dtype=np.float64)
    if offset is None:
        return position
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,):
        raise InputError(f"offset must have the shape (3,); got {offset.shape}.")
    return position + offset.reshape(1, 3)


def _round_trip_raw(path: str, written: np.ndarray) -> None:
    back = np.fromfile(path, dtype=written.dtype).reshape(written.shape)
    if not np.array_equal(back, written):
        raise ResultError(
            f"{path!r} does not read back as written; the file is refused."
        )
    expected = written.size * written.dtype.itemsize
    if os.path.getsize(path) != expected:
        raise ResultError(
            f"{path!r} has {os.path.getsize(path)} bytes, not the {expected} written."
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
    """Write GAMER's particle file: a headerless stream of the attributes.

    Parameters
    ----------
    order : {"ATT_ID", "ID_ATT"}
        Attribute-major blocks (every particle's mass, then every x, ...) or
        particle-major rows, as GAMER's ``PAR_IC_FORMAT`` names them.
    double_precision : bool
        Doubles where GAMER runs with ``PAR_IC_FLOAT8``, floats otherwise.
    include_mass : bool
        False where GAMER sets the mass itself (``PAR_IC_MASS`` at or above
        zero); the particle type is GAMER's ``PAR_IC_TYPE`` in either case.
    offset : array-like of shape (3,), optional
        Added to the positions: GAMER's box has its corner at the origin, the
        sampler centers the model there.
    attributes : {"standard", "legacy"}
        ``"standard"`` writes the mass, the position and the velocity.
        ``"legacy"`` writes the thirteen float32 attribute-major blocks of the
        build the paper's runs used (`GAMER_LEGACY_ATTRIBUTES`): the type and
        the tag 1, the energies and the angular momentum 0, the identifiers
        0 to n - 1, which a float32 carries exactly only below 2**24; more
        particles are refused. The other options must be at their defaults.

    Raises
    ------
    InputError
        For an unknown order or attribute set, an option that the legacy
        layout cannot take, or more particles than the legacy identifiers
        can carry.
    ResultError
        If the file does not read back as written.
    """
    if order not in ("ATT_ID", "ID_ATT"):
        raise InputError(f"order must be 'ATT_ID' or 'ID_ATT'; got {order!r}.")
    if attributes not in ("standard", "legacy"):
        raise InputError(
            f"attributes must be 'standard' or 'legacy'; got {attributes!r}."
        )
    path = _target(path, overwrite)
    n = len(particles)
    position = _shifted(particles, offset)
    velocity = np.asarray(particles.velocity, dtype=np.float64)
    mass = np.asarray(particles.mass, dtype=np.float64)
    if attributes == "legacy":
        if not (order == "ATT_ID" and double_precision and include_mass):
            raise InputError(
                "the legacy layout is attribute-major float32 with the mass; "
                "order, double_precision and include_mass must be left at their "
                "defaults."
            )
        if n > ID_LIMIT_FLOAT32:
            raise InputError(
                f"{n} particles exceed the {ID_LIMIT_FLOAT32} identifiers a "
                "float32 carries exactly; the legacy layout is refused."
            )
        columns = [
            mass,
            position[:, 0],
            position[:, 1],
            position[:, 2],
            velocity[:, 0],
            velocity[:, 1],
            velocity[:, 2],
            np.ones(n),
            np.ones(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.arange(n, dtype=np.float64),
        ]
        written = np.ascontiguousarray(np.stack(columns), dtype=np.float32)
    else:
        columns = ([mass] if include_mass else []) + [
            position[:, 0],
            position[:, 1],
            position[:, 2],
            velocity[:, 0],
            velocity[:, 1],
            velocity[:, 2],
        ]
        by_attribute = np.stack(columns)
        arranged = by_attribute if order == "ATT_ID" else by_attribute.T
        written = np.ascontiguousarray(
            arranged, dtype=np.float64 if double_precision else np.float32
        )
    written.tofile(path)
    _round_trip_raw(path, written)


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
    """Write a Gadget-derivative HDF5 initial condition (snapshot format 3).

    The layout is the OpenGadget3 reference's: ``/Header`` with the particle
    counts, the mass table and the cosmology of an isolated model (zeros),
    and ``/PartType<n>`` with ``Coordinates``, ``Velocities`` and
    ``ParticleIDs`` from 1. Equal masses go into the mass table alone; unequal
    masses are written per particle in ``Masses`` with the table's slot 0.

    Parameters
    ----------
    double_precision : bool
        Float64 arrays and ``Flag_DoublePrecision`` 1; float32 otherwise.
    part_type : int
        The particle type slot, 0 to 5.
    hubble_param : float
        Written to the header, unused by an isolated model.
    offset : array-like of shape (3,), optional
        Added to the positions.

    Raises
    ------
    InputError
        For a part_type outside 0 to 5.
    ResultError
        If the file does not read back as written.
    """
    if not 0 <= int(part_type) < 6 or int(part_type) != part_type:
        raise InputError(
            f"part_type must be an integer from 0 to 5; got {part_type!r}."
        )
    path = _target(path, overwrite)
    n = len(particles)
    dtype = np.float64 if double_precision else np.float32
    position = _shifted(particles, offset).astype(dtype)
    velocity = np.asarray(particles.velocity, dtype=np.float64).astype(dtype)
    mass = np.asarray(particles.mass, dtype=np.float64)
    equal = n > 0 and bool(np.all(mass == mass[0]))
    counts = np.zeros(6, dtype=np.int32)
    counts[part_type] = n
    table = np.zeros(6, dtype=np.float64)
    if equal:
        table[part_type] = mass[0]
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
        group.create_dataset("Coordinates", data=position)
        group.create_dataset("Velocities", data=velocity)
        group.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        if not equal:
            group.create_dataset("Masses", data=mass.astype(dtype))
    with h5py.File(path, "r") as file:
        group = file[f"PartType{part_type}"]
        same = (
            np.array_equal(group["Coordinates"][...], position)
            and np.array_equal(group["Velocities"][...], velocity)
            and int(file["Header"].attrs["NumPart_ThisFile"][part_type]) == n
            and (
                float(file["Header"].attrs["MassTable"][part_type]) == mass[0]
                if equal
                else np.array_equal(group["Masses"][...], mass.astype(dtype))
            )
        )
    if not same:
        raise ResultError(
            f"{path!r} does not read back as written; the file is refused."
        )


@register_particle_writer("text")
def write_text(
    particles: Particles, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write the particles as fixed-width text with a header.

    The header lines, each starting with ``#``, state the unit system, the
    unit of every column, the identity of the realization and of its parent,
    and the column names. Each row holds the mass, the position and the
    velocity, every field in 25 characters with 14 digits after the point;
    with more than one component the component's index follows them.

    Raises
    ------
    ResultError
        If the file does not read back as written to the digits written.
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
    values = np.column_stack(
        [particles.mass, particles.position, particles.velocity]
    ).astype(np.float64)
    component = np.asarray(particles.component, dtype=np.int64)
    with open(path, "w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(head) + "\n")
        for i in range(n):
            row = "".join(TEXT_FIELD.format(v) for v in values[i])
            if multi:
                row += f"{int(component[i]):<6d}"
            file.write(row.rstrip() + "\n")
    back = np.loadtxt(path, ndmin=2) if n else np.zeros((0, values.shape[1]))
    rounded = np.array(
        [[float(TEXT_FIELD.format(v).strip()) for v in r] for r in values]
    )
    same = back.shape[0] == n and (
        n == 0
        or (
            np.array_equal(back[:, : values.shape[1]], rounded)
            and (not multi or np.array_equal(back[:, -1].astype(np.int64), component))
        )
    )
    if not same:
        raise ResultError(
            f"{path!r} does not read back as written; the file is refused."
        )
