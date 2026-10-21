"""Writing results to HDF5 files and reading results back from them.

A user writes a result with the result's ``write`` method and reads any
result file back with `read`. The functions of the module implement the file
format that the ``write`` methods and `read` share: `write_result` writes one
result to one file, `read_result` reads the file back as a `StoredResult`, and
`register_reader` sets the function that rebuilds the result object of each
kind of result. One file
holds one result: the result's arrays, the record of inputs and settings that
determines the result, the result's identity string, the SHA-256 digest of
the arrays (`data_digest`), the identity strings of the parent results, and
the versions and numerical backend of the session that wrote the file
(`backend_provenance`). `write_result` reads every file back and compares the
file with the request before returning; `read_result` recomputes the digest
and compares the recomputed digest with the digest the file records.
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
    """A result as one file stores the result.

    `read_result` returns a `StoredResult`; the reader registered for the
    result's kind (`register_reader`) rebuilds the result object from the
    `StoredResult`.

    Parameters
    ----------
    kind : str
        The kind of result, for example ``"model"``.
    identity : str
        The identity string of the result.
    data_sha256 : str
        The SHA-256 digest of the file's datasets (`data_digest`), as a
        hexadecimal string of 64 characters.
    record : dict
        The inputs and settings that determine the result, in dictionary form.
    arrays : dict of str to array
        The arrays of the result, by name.
    parents : tuple of str
        The identity strings of the results from which the result was
        computed.
    provenance : dict of str to str
        The creation date of the file and the versions and numerical backend
        of the session that wrote the file (`backend_provenance`).

    See Also
    --------
    read_result : The function that reads a file into a `StoredResult`.
    write_result : The function that writes a result to a file.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import read_result, write_result
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     arrays = {"x": np.arange(3.0)}
    ...     _ = write_result(path, "example", "example-1", {"n": 3}, arrays)
    ...     stored = read_result(path)
    >>> stored.kind, stored.identity, stored.record, stored.parents
    ('example', 'example-1', {'n': 3}, ())
    >>> stored.arrays["x"]
    array([0., 1., 2.])
    """

    kind: str
    identity: str
    data_sha256: str
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    parents: tuple[str, ...]
    provenance: dict[str, str]


def data_digest(source: h5py.Group | str | os.PathLike) -> str:
    """Return the SHA-256 digest of every dataset of an HDF5 file or group.

    The digest covers every dataset, in the sorted order of the datasets'
    paths: each dataset's path, data type, shape and raw bytes. The digest
    excludes the attributes, so a file can store the digest of its own
    datasets as an attribute.

    Parameters
    ----------
    source : h5py.Group or path
        An open HDF5 file or group, or the path of an HDF5 file.

    Returns
    -------
    digest : str
        The hexadecimal digest, 64 characters.

    See Also
    --------
    write_result : The function that records the digest in each file the
        function writes.
    read_result : The function that compares the recorded digest with a
        recomputed digest.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import data_digest, write_result
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     written = write_result(path, "example", "example-1", {}, {"x": np.ones(2)})
    ...     recomputed = data_digest(path)
    >>> recomputed == written, len(recomputed)
    (True, 64)
    """
    if not isinstance(source, h5py.Group):
        with h5py.File(source, "r") as file:
            return data_digest(file)
    digest = hashlib.sha256()
    for name in sorted(_dataset_names(source)):
        dataset = source[name]
        _digest_dataset(digest, name, dataset.dtype, dataset.shape, dataset[...])
    return digest.hexdigest()


def _dataset_names(source: h5py.Group) -> list[str]:
    """Return the path of every dataset under an HDF5 group, in h5py's order."""
    names: list[str] = []
    source.visititems(
        lambda name, item: (
            names.append(name) if isinstance(item, h5py.Dataset) else None
        )
    )
    return names


def _digest_dataset(
    digest: Any, name: str, dtype: np.dtype[Any], shape: tuple[int, ...], values: Any
) -> None:
    """Add one dataset's path, data type, shape and raw bytes to the digest."""
    digest.update(f"{name}|{dtype.str}|{shape}|".encode())
    if dtype.kind == "O":
        # A dataset of variable-length strings has no fixed byte layout, so the
        # digest covers the repr of the dataset's values instead of raw bytes.
        digest.update(repr(np.asarray(values).tolist()).encode())
    else:
        digest.update(np.ascontiguousarray(values).tobytes())


def backend_provenance() -> dict[str, str]:
    """Return the software versions and the numerical backend of the running session.

    `write_result` stores the returned dictionary in every file the function
    writes.

    Returns
    -------
    provenance : dict of str to str
        The keys ``python_version``, ``numpy_version`` and ``scipy_version``;
        ``jax_platforms_env`` and ``pianistpy_platform_env``, the values of
        the environment variables ``JAX_PLATFORMS`` and
        ``PIANISTPY_PLATFORM``, each ``"unset"`` where the variable is not
        set; and ``jax_version``, ``jax_backend`` and ``jax_devices``. A
        session in which JAX cannot be imported records ``"unavailable"`` for
        ``jax_version`` and ``jax_devices`` and ``"none"`` for
        ``jax_backend``.

    See Also
    --------
    write_result : The function that stores the provenance in each file.

    Examples
    --------
    >>> from pianistpy.io import backend_provenance
    >>> sorted(backend_provenance())[:3]
    ['jax_backend', 'jax_devices', 'jax_platforms_env']
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
    """Write one result to an HDF5 file and check the file against the request.

    After writing, the call reads the file back and compares the file's
    arrays, bit for bit, and the file's record, identity string, kind and
    parents with the values passed. The call stores the SHA-256 digest of the
    datasets (`data_digest`) in the file only when every comparison holds.

    Parameters
    ----------
    path : str or path-like
        The file to write.
    kind : str
        The kind of result, the key under which `register_reader` finds the
        function that rebuilds the result.
    identity : str
        The identity string of the result.
    record : mapping
        The inputs and settings that determine the result, in dictionary
        form. Every value is one that ``json.dumps`` writes.
    arrays : mapping of str to array-like
        The arrays of the result, by name. A name may contain ``/`` to place
        the array in an HDF5 group.
    parents : sequence of str, optional
        The identity strings of the results from which the result was
        computed. Default: ``()``, which means the result has no parent
        results.
    overwrite : bool, optional
        Whether to replace an existing file at `path`. Default: ``False``,
        which means an existing file is kept and the call raises
        `InputError`.

    Returns
    -------
    data_sha256 : str
        The SHA-256 digest of the datasets as written (`data_digest`), the
        same digest the file stores.

    Raises
    ------
    InputError
        If a file exists at `path` and `overwrite` is false.
    ResultError
        If the file read back differs from the request in an array, the
        record, the identity string, the kind or the parents.

    See Also
    --------
    read_result : The function that reads the file back.
    data_digest : The digest of the datasets.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import write_result
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     digest = write_result(path, "example", "example-1", {}, {"x": np.ones(2)})
    >>> len(digest)
    64
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise InputError(
            f"The file {str(path)!r} exists; pass overwrite=True to replace the file."
        )
    now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017 (any Python)
    created = now.isoformat(timespec="seconds")
    with h5py.File(path, "w") as file:
        group = file.create_group(DATA_GROUP)
        for name, values in arrays.items():
            group.create_dataset(name, data=np.asarray(values))
        file.attrs["format_version"] = FORMAT_VERSION
        file.attrs["pianistpy_version"] = __version__
        file.attrs["result_kind"] = kind
        file.attrs["identity"] = identity
        file.attrs["record"] = json.dumps(dict(record), sort_keys=True)
        file.attrs["parents"] = json.dumps(list(parents))
        provenance = file.create_group(PROVENANCE_GROUP)
        provenance.attrs["created"] = created
        for key, value in backend_provenance().items():
            provenance.attrs[key] = value
    return _check_read_back(path, kind, identity, record, arrays, parents)


def _check_read_back(
    path: Path,
    kind: str,
    identity: str,
    record: Mapping[str, Any],
    arrays: Mapping[str, Any],
    parents: Sequence[str],
) -> str:
    """Read the file at `path` back, compare the file with the request, and
    store and return the digest of the file's datasets.

    The comparison covers the kind, the identity string, the record, the
    parents, the names of the arrays and every array bit for bit. The digest
    (`data_digest`) is stored in the file only when every comparison holds;
    otherwise the call raises `ResultError`. Each dataset is read once and
    serves both the comparison and the digest, and the datasets are read one
    at a time, so the check holds at most one stored dataset in memory beside
    the arrays the caller passed."""
    with h5py.File(path, "r+") as file:
        names = _dataset_names(file)
        prefix = DATA_GROUP + "/"
        stored = [n[len(prefix) :] for n in names if n.startswith(prefix)]
        same = (
            str(file.attrs["result_kind"]) == kind
            and str(file.attrs["identity"]) == identity
            and str(file.attrs["record"]) == json.dumps(dict(record), sort_keys=True)
            and tuple(json.loads(file.attrs["parents"])) == tuple(parents)
            and len(stored) == len(names)
            and sorted(stored) == sorted(arrays)
        )
        digest = hashlib.sha256()
        for name in sorted(names):
            if not same:
                break
            dataset = file[name]
            found = dataset[...]
            _digest_dataset(digest, name, dataset.dtype, dataset.shape, found)
            expected = np.asarray(arrays[name[len(prefix) :]])
            same = found.dtype == expected.dtype and np.array_equal(
                found, expected, equal_nan=expected.dtype.kind in "fc"
            )
        if same:
            file.attrs["data_sha256"] = digest.hexdigest()
    if not same:
        raise ResultError(
            f"The file {path} does not read back as written, so the file is not a "
            "faithful copy of the result."
        )
    return digest.hexdigest()


def read_result(path: str | os.PathLike) -> StoredResult:
    """Read one result from an HDF5 file and check the file's digest.

    The call recomputes the SHA-256 digest of the file's datasets
    (`data_digest`) and compares the recomputed digest with the digest the
    file records.

    Parameters
    ----------
    path : str or path-like
        The file to read.

    Returns
    -------
    stored : StoredResult
        The arrays, record, identity string, digest, parents and provenance
        the file holds.

    Raises
    ------
    ResultError
        If the file lacks one of the attributes ``format_version``,
        ``result_kind``, ``identity``, ``data_sha256``, ``record`` and
        ``parents``.
    ResultError
        If the major number of the file's format version differs from the
        major number of the format version the installed package reads.
    ResultError
        If the recomputed digest differs from the digest the file records.

    See Also
    --------
    read : The function that rebuilds the result object from the file.
    write_result : The function that writes the file.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import read_result, write_result
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     arrays = {"x": np.arange(3.0)}
    ...     _ = write_result(path, "example", "example-1", {"n": 3}, arrays)
    ...     stored = read_result(path)
    >>> stored.record, stored.arrays["x"]
    ({'n': 3}, array([0., 1., 2.]))
    """
    path = Path(path)
    with h5py.File(path, "r") as file:
        missing = [key for key in REQUIRED_ATTRIBUTES if key not in file.attrs]
        if missing:
            raise ResultError(
                f"The file {path} lacks the attributes {missing}, which every result "
                "file holds."
            )
        version = str(file.attrs["format_version"])
        if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ResultError(
                f"The file {path} has format version {version}; the installed package "
                f"reads only files of format version {FORMAT_VERSION} or of the "
                f"same major version as {FORMAT_VERSION}."
            )
        stored_digest = str(file.attrs["data_sha256"])
        # Each dataset is read once and serves two purposes: the digest, which
        # covers the datasets in the sorted order of their paths
        # (`data_digest`), and the returned arrays, which follow the order in
        # which h5py visits the data group.
        names = _dataset_names(file)
        values = {name: file[name][...] for name in names}
        hashed = hashlib.sha256()
        for name in sorted(names):
            dataset = file[name]
            _digest_dataset(hashed, name, dataset.dtype, dataset.shape, values[name])
        digest = hashed.hexdigest()
        if digest != stored_digest:
            raise ResultError(
                f"The arrays of the file {path} do not match the digest that the file "
                f"records: the stored digest begins {stored_digest[:16]}, and the "
                f"digest recomputed from the arrays begins {digest[:16]}."
            )
        arrays: dict[str, np.ndarray] = {}
        file[DATA_GROUP].visititems(
            lambda name, item: (
                arrays.update({name: values[f"{DATA_GROUP}/{name}"]})
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
    """Return a `StoredResult`, given the `StoredResult` or the path of a result file.

    A `StoredResult` passed as `source` is returned unchanged. A path is read
    and checked by `read_result`.

    Parameters
    ----------
    source : StoredResult or str or path-like
        A `StoredResult`, or the path of a file that holds one result.

    Returns
    -------
    stored : StoredResult
        `source` itself when `source` is a `StoredResult`; otherwise the
        `StoredResult` that `read_result` reads from the file.

    Raises
    ------
    ResultError
        If `source` is a path and `read_result` raises `ResultError`: the file
        lacks an attribute of a result file, the major number of the file's
        format version differs from the major number the package reads, or
        the recomputed digest differs from the recorded digest.

    See Also
    --------
    read_result : The function that reads and checks a file.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import load_result, read_result, write_result
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     _ = write_result(path, "example", "example-1", {}, {"x": np.ones(2)})
    ...     stored = load_result(path)
    >>> load_result(stored) is stored
    True
    """
    if isinstance(source, StoredResult):
        return source
    return read_result(source)


READERS: dict[str, Callable[[StoredResult], Any]] = {}


def register_reader(kind: str, reader: Callable[[StoredResult], Any]) -> None:
    """Make `reader` the function that rebuilds every result of one kind.

    `read` calls the function registered for the kind a file records. A
    second registration for the same kind replaces the first registration.

    Parameters
    ----------
    kind : str
        The kind of result, as `write_result` stores the kind in the file.
    reader : callable
        A function that takes the `StoredResult` and returns the result
        object.

    See Also
    --------
    read : The function that calls the registered reader.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import read, register_reader, write_result
    >>> register_reader("doc_example", lambda stored: float(stored.arrays["x"].sum()))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     arrays = {"x": np.arange(3.0)}
    ...     _ = write_result(path, "doc_example", "example-1", {}, arrays)
    ...     total = read(path)
    >>> total
    3.0
    """
    READERS[kind] = reader


def read(path: str | os.PathLike) -> Any:
    """Read a result from a file, whatever kind of result the file holds.

    The call reads and checks the file with `read_result` and passes the
    `StoredResult` to the reader registered for the file's kind
    (`register_reader`).

    Parameters
    ----------
    path : str or path-like
        A file written by the ``write`` method of a result.

    Returns
    -------
    result : object
        The distribution function, model, particles or other result the file
        holds, equal to the result that was written.

    Raises
    ------
    ResultError
        If the file lacks one of the attributes ``format_version``,
        ``result_kind``, ``identity``, ``data_sha256``, ``record`` and
        ``parents``.
    ResultError
        If the major number of the file's format version differs from the
        major number of the format version the installed package reads.
    ResultError
        If the recomputed digest of the datasets differs from the digest the
        file records.
    ResultError
        If no reader is registered for the kind the file records.

    See Also
    --------
    register_reader : The registration of the function that rebuilds a kind.
    read_result : The read and the check of the file.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import read, register_reader, write_result
    >>> register_reader("doc_example", lambda stored: float(stored.arrays["x"].sum()))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "example.h5")
    ...     arrays = {"x": np.arange(3.0)}
    ...     _ = write_result(path, "doc_example", "example-1", {}, arrays)
    ...     result = read(path)
    >>> result
    3.0
    """
    stored = read_result(path)
    if stored.kind not in READERS:
        raise ResultError(
            f"The file {path} holds a result of the kind {stored.kind!r}, and no "
            f"reader is registered for the kind {stored.kind!r}."
        )
    return READERS[stored.kind](stored)
