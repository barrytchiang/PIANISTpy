"""Tables of a result as fixed-width text.

A text table is an export, not a record: the HDF5 file of `write_result`
holds the result, and `write_table` writes named columns of it in a form
any program reads. Every file is read back and compared with what was
written, to the digits written, and refused otherwise.
"""

__all__ = ["TEXT_FIELD", "write_table"]

import os
from collections.abc import Mapping, Sequence

import numpy as np

from ..exceptions import InputError, ResultError

# 14 digits after the point, as the particle text layout writes them.
TEXT_FIELD = "{:<25.14e}"
INTEGER_FIELD = "{:<25d}"


def _columns(columns: Mapping[str, object]) -> tuple[list[str], list[np.ndarray]]:
    if not columns:
        raise InputError("write_table needs at least one column.")
    names = [str(name) for name in columns]
    for name in names:
        if not name or any(c.isspace() for c in name):
            raise InputError(f"a column name holds no space; got {name!r}.")
    arrays = []
    for name in names:
        array = np.asarray(columns[name])
        if array.ndim != 1:
            raise InputError(
                f"column {name!r} must be one-dimensional; got shape {array.shape}."
            )
        if array.dtype.kind not in "iuf":
            raise InputError(
                f"column {name!r} must be numeric; got dtype {array.dtype}."
            )
        arrays.append(array)
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise InputError(
            "every column must have the same length; got "
            + ", ".join(f"{n}: {len(a)}" for n, a in zip(names, arrays, strict=True))
            + "."
        )
    return names, arrays


def _field(array: np.ndarray) -> str:
    return INTEGER_FIELD if array.dtype.kind in "iu" else TEXT_FIELD


def write_table(
    path: str | os.PathLike,
    columns: Mapping[str, object],
    header: Sequence[str] = (),
    overwrite: bool = False,
) -> None:
    """Write named columns as fixed-width text with a header.

    The header lines, each starting with ``#``, are written as given,
    followed by ``# columns:`` and the column names in order. Each row holds
    one value of every column, 25 characters a field: a float with 14 digits
    after the point in exponent form, an integer as it is. The file is read
    back with ``numpy.loadtxt`` and compared with the values written, to the
    digits written; a file that does not read back is refused.

    Parameters
    ----------
    path : path
        File to write.
    columns : mapping of str to array-like
        The columns, in order; one-dimensional, numeric, of one length. A
        name holds no space.
    header : sequence of str, optional
        Lines for the head of the file, without the leading ``#``.
    overwrite : bool, optional
        Replace an existing file.

    Raises
    ------
    InputError
        For a column that is not one-dimensional and numeric, columns of
        different lengths, a name with a space, or an existing file without
        ``overwrite``.
    ResultError
        If the file does not read back as written.
    """
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise InputError(f"{path!r} exists; pass overwrite=True to replace it.")
    names, arrays = _columns(columns)
    n = len(arrays[0])
    fields = [_field(array) for array in arrays]
    head = ["# " + str(line) for line in header] + ["# columns: " + " ".join(names)]
    with open(path, "w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(head) + "\n")
        for i in range(n):
            row = "".join(f.format(a[i]) for f, a in zip(fields, arrays, strict=True))
            file.write(row.rstrip() + "\n")
    back = np.loadtxt(path, ndmin=2) if n else np.zeros((0, len(arrays)))
    same = back.shape == (n, len(arrays))
    for j, (field, array) in enumerate(zip(fields, arrays, strict=True)):
        if not same:
            break
        if array.dtype.kind in "iu":
            same = np.array_equal(back[:, j].astype(np.int64), array.astype(np.int64))
        else:
            rounded = np.array([float(field.format(v).strip()) for v in array])
            same = np.array_equal(back[:, j], rounded, equal_nan=True)
    if not same:
        raise ResultError(
            f"{path!r} does not read back as written; the file is refused."
        )
