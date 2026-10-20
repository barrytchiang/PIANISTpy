"""Writing and reading results as HDF5 files.

A file holds one result: its arrays, the record that determines it (inputs
and settings in dictionary form), its identity string, the digest of its
arrays, the identities of its parents, and the provenance of the run.
"""

__all__ = [
    "StoredResult",
    "backend_provenance",
    "data_digest",
    "load_result",
    "read",
    "read_result",
    "register_reader",
    "write_result",
]

import datetime
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .._version import __version__
from ..exceptions import InputError, ResultError
from ..identity import FORMAT_VERSION

DATA_GROUP = "data"
PROVENANCE_GROUP = "provenance"
REQUIRED_ATTRIBUTES = (
    "format_version",
    "result_kind",
    "identity",
    "data_sha256",
    "record",
    "parents",
)


@dataclass(frozen=True)
class StoredResult:
    """A result as it is stored in a file.

    Parameters
    ----------
    kind : str
        Kind of result, for example ``"model"``.
    identity : str
        Identity string of the result.
    data_sha256 : str
        Digest of the arrays.
    record : dict
        The inputs and settings that determine the result, in dictionary form.
    arrays : dict of str to array
        The arrays of the result, by name.
    parents : tuple of str
        Identity strings of the results this one was computed from.
    provenance : dict
        Date, versions and backend of the run that wrote the file.
    """

    kind: str
    identity: str
    data_sha256: str
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    parents: tuple[str, ...]
    provenance: dict[str, str]


def data_digest(source: h5py.Group | str | os.PathLike) -> str:
    """Return the SHA-256 digest of every dataset of a file.

    The digest covers, for each dataset in the sorted order of its path, the
    path, the type, the shape and the raw bytes. Attributes are excluded, so
    the digest can be stored in the file it describes.

    Parameters
    ----------
    source : h5py.Group or path
        An open file or group, or the path of a file.

    Returns
    -------
    digest : str
        Hexadecimal digest, 64 characters.
    """
    if not isinstance(source, h5py.Group):
        with h5py.File(source, "r") as file:
            return data_digest(file)
    names: list[str] = []
    source.visititems(
        lambda name, item: (
            names.append(name) if isinstance(item, h5py.Dataset) else None
        )
    )
    digest = hashlib.sha256()
    for name in sorted(names):
        dataset = source[name]
        digest.update(f"{name}|{dataset.dtype.str}|{dataset.shape}|".encode())
        values = dataset[...]
        if dataset.dtype.kind == "O":
            # Variable-length strings have no fixed byte layout.
            digest.update(repr(np.asarray(values).tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def backend_provenance() -> dict[str, str]:
    """Return the versions and the numerical backend of the running session.

    Returns
    -------
    provenance : dict of str to str
        The Python, NumPy and SciPy versions, the platform JAX was pinned to
        by the environment, and the backend and devices JAX reports. A
        session without JAX says so by name.
    """
    import scipy

    provenance = {
        "python_version": sys.version.split()[0],
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS", "unset"),
        "pianistpy_platform_env": os.environ.get("PIANISTPY_PLATFORM", "unset"),
    }
    try:
        import jax
    except ImportError:
        provenance.update(
            jax_version="unavailable", jax_backend="none", jax_devices="unavailable"
        )
        return provenance
    devices = sorted({str(getattr(d, "device_kind", d)) for d in jax.devices()})
    provenance.update(
        jax_version=str(jax.__version__),
        jax_backend=str(jax.default_backend()),
        jax_devices=",".join(devices),
    )
    return provenance


def write_result(
    path: str | os.PathLike,
    kind: str,
    identity: str,
    record: Mapping[str, Any],
    arrays: Mapping[str, Any],
    parents: Sequence[str] = (),
    overwrite: bool = False,
) -> str:
    """Write one result to an HDF5 file.

    Parameters
    ----------
    path : path
        File to write.
    kind : str
        Kind of result.
    identity : str
        Identity string of the result.
    record : mapping
        The inputs and settings that determine the result, in dictionary form.
    arrays : mapping of str to array-like
        The arrays of the result, by name. A name may contain ``/`` to place
        the array in a group.
    parents : sequence of str, optional
        Identity strings of the results this one was computed from.
    overwrite : bool, optional
        Replace an existing file. By default an existing file is refused.

    Returns
    -------
    data_sha256 : str
        Digest of the arrays as written.

    Raises
    ------
    InputError
        If the file exists and `overwrite` is false.
    ResultError
        If the file does not read back as written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise InputError(f"{str(path)!r} exists; pass overwrite=True to replace it.")
    now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017 (any Python)
    created = now.isoformat(timespec="seconds")
    with h5py.File(path, "w") as file:
        group = file.create_group(DATA_GROUP)
        for name, values in arrays.items():
            group.create_dataset(name, data=np.asarray(values))
        digest = data_digest(file)
        file.attrs["format_version"] = FORMAT_VERSION
        file.attrs["pianistpy_version"] = __version__
        file.attrs["result_kind"] = kind
        file.attrs["identity"] = identity
        file.attrs["data_sha256"] = digest
        file.attrs["record"] = json.dumps(dict(record), sort_keys=True)
        file.attrs["parents"] = json.dumps(list(parents))
        provenance = file.create_group(PROVENANCE_GROUP)
        provenance.attrs["created"] = created
        for key, value in backend_provenance().items():
            provenance.attrs[key] = value
    _check_read_back(path, kind, identity, record, arrays, parents, digest)
    return digest


def _check_read_back(
    path: Path,
    kind: str,
    identity: str,
    record: Mapping[str, Any],
    arrays: Mapping[str, Any],
    parents: Sequence[str],
    digest: str,
) -> None:
    """Read the file just written and refuse it unless it holds what was
    asked to be written: the arrays to the bit, the record, the identity.
    The arrays are compared one at a time, so the check holds at most one of
    them beside the caller's."""
    with h5py.File(path, "r") as file:
        group = file[DATA_GROUP]
        stored: list[str] = []
        group.visititems(
            lambda name, item: (
                stored.append(name) if isinstance(item, h5py.Dataset) else None
            )
        )
        same = (
            str(file.attrs["result_kind"]) == kind
            and str(file.attrs["identity"]) == identity
            and str(file.attrs["data_sha256"]) == digest
            and data_digest(file) == digest
            and str(file.attrs["record"]) == json.dumps(dict(record), sort_keys=True)
            and tuple(json.loads(file.attrs["parents"])) == tuple(parents)
            and sorted(stored) == sorted(arrays)
        )
        for name, values in arrays.items():
            if not same:
                break
            expected = np.asarray(values)
            found = group[name][...]
            same = found.dtype == expected.dtype and np.array_equal(
                found, expected, equal_nan=expected.dtype.kind in "fc"
            )
    if not same:
        raise ResultError(f"{path} does not read back as written; the file is refused.")


def read_result(path: str | os.PathLike) -> StoredResult:
    """Read one result from an HDF5 file and check it.

    Parameters
    ----------
    path : path
        File to read.

    Returns
    -------
    stored : StoredResult

    Raises
    ------
    ResultError
        If the file is not a result of this format, or if its arrays no
        longer match the digest it records.
    """
    path = Path(path)
    with h5py.File(path, "r") as file:
        missing = [key for key in REQUIRED_ATTRIBUTES if key not in file.attrs]
        if missing:
            raise ResultError(f"The file {path} lacks the attribute(s) {missing}.")
        version = str(file.attrs["format_version"])
        if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ResultError(
                f"The file {path} has format version {version}; this version of "
                f"the package reads format version {FORMAT_VERSION}."
            )
        stored_digest = str(file.attrs["data_sha256"])
        digest = data_digest(file)
        if digest != stored_digest:
            raise ResultError(
                f"The arrays of {path} do not match the digest the file records "
                f"(stored {stored_digest[:16]}, recomputed {digest[:16]})."
            )
        arrays: dict[str, np.ndarray] = {}
        file[DATA_GROUP].visititems(
            lambda name, item: (
                arrays.update({name: item[...]})
                if isinstance(item, h5py.Dataset)
                else None
            )
        )
        provenance = {
            key: str(value) for key, value in file[PROVENANCE_GROUP].attrs.items()
        }
        return StoredResult(
            kind=str(file.attrs["result_kind"]),
            identity=str(file.attrs["identity"]),
            data_sha256=stored_digest,
            record=json.loads(file.attrs["record"]),
            arrays=arrays,
            parents=tuple(json.loads(file.attrs["parents"])),
            provenance=provenance,
        )


def load_result(source: StoredResult | str | os.PathLike) -> StoredResult:
    """Return a stored result from a live object or from a path.

    Every function that takes a result accepts either form and calls this
    once, so that the two forms follow the same code afterwards.

    Parameters
    ----------
    source : StoredResult or path
        A stored result, or the path of a file that holds one.

    Returns
    -------
    stored : StoredResult
    """
    if isinstance(source, StoredResult):
        return source
    return read_result(source)


READERS: dict[str, Callable[[StoredResult], Any]] = {}


def register_reader(kind: str, reader: Callable[[StoredResult], Any]) -> None:
    """Make `reader` the function that rebuilds the results of one kind from a file.

    Parameters
    ----------
    kind : str
        Kind of result, as stored in the file.
    reader : callable
        Takes the stored result and returns the result object.
    """
    READERS[kind] = reader


def read(path: str | os.PathLike) -> Any:
    """Read a result from a file, whatever kind of result it holds.

    Parameters
    ----------
    path : path
        A file written by the ``write`` method of a result.

    Returns
    -------
    result : object
        The distribution function, model, particles or other result the file
        holds, equal to the one that was written.

    Raises
    ------
    ResultError
        If the file is not a result of this package, if its arrays no longer
        match the digest it records, or if its kind is unknown.
    """
    stored = read_result(path)
    if stored.kind not in READERS:
        raise ResultError(
            f"The file {path} holds a result of the unknown kind {stored.kind!r}."
        )
    return READERS[stored.kind](stored)
