"""Columns of a result, each with a column name, written as a fixed-width text table.

A user calls `write_table` to export columns of a result in a form any program
reads. A text table is an export and not a record of the result: the HDF5 file
that `write_result` writes holds the result. `write_table` reads every text
file back and compares the values read with the values written, rounded to the
14 digits after the decimal point that the file holds; a file whose values
differ raises `ResultError`.
"""

__all__ = ["TEXT_FIELD", "write_table"]

import os
from collections.abc import Mapping, Sequence

import numpy as np

from ..exceptions import InputError, ResultError

# A float is written in exponent form with 14 digits after the decimal point,
# the precision of the particle text layout, left-aligned in a field of 25
# characters so that the columns line up.
TEXT_FIELD = "{:<25.14e}"
INTEGER_FIELD = "{:<25d}"


def _columns(columns: Mapping[str, object]) -> tuple[list[str], list[np.ndarray]]:
    """Return the column names and the column arrays, after checking both.

    Each name must be non-empty and free of whitespace; each array must be
    one-dimensional and of integer or float type; every array must have the
    length of the first array. A column that fails a check raises `InputError`.
    """
    if not columns:
        raise InputError("write_table needs at least one column.")
    names = [str(name) for name in columns]
    for name in names:
        if not name or any(c.isspace() for c in name):
            raise InputError(
                f"Every column name must be a non-empty string with no whitespace; "
                f"the name given is {name!r}."
            )
    arrays = []
    for name in names:
        array = np.asarray(columns[name])
        if array.ndim != 1:
            raise InputError(
                f"Column {name!r} must be one-dimensional; the column given has "
                f"shape {array.shape}."
            )
        if array.dtype.kind not in "iuf":
            raise InputError(
                f"Column {name!r} must hold integers or floats; the column given has "
                f"dtype {array.dtype}."
            )
        arrays.append(array)
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise InputError(
            "Every column must have the same length; the lengths given are "
            + ", ".join(f"{n}: {len(a)}" for n, a in zip(names, arrays, strict=True))
            + "."
        )
    return names, arrays


def _field(array: np.ndarray) -> str:
    """Return the format of one field of `array`: integer or float."""
    return INTEGER_FIELD if array.dtype.kind in "iu" else TEXT_FIELD


def write_table(
    path: str | os.PathLike,
    columns: Mapping[str, object],
    header: Sequence[str] = (),
    overwrite: bool = False,
) -> None:
    """Write columns, each with a column name, as a fixed-width text table.

    The file opens with the lines of `header`, each prefixed with ``# ``,
    followed by the line ``# columns:`` and the column names in order. Each
    following row holds one value of every column, each value in a field of
    25 characters: a float in exponent form with 14 digits after the decimal
    point, an integer as the integer. After writing, the call reads the file
    back with ``numpy.loadtxt`` and compares the values read with the values
    written, rounded to the 14 digits the file holds.

    Parameters
    ----------
    path : str or path-like
        The file to write.
    columns : mapping of str to array-like
        The columns, in order. Each column is one-dimensional, of integer or
        float type, and of the length of every other column. Each column name
        is non-empty and holds no whitespace.
    header : sequence of str, optional
        The lines at the head of the file, each without the leading ``#``.
        Default: ``()``, which means the head holds only the line of column
        names.
    overwrite : bool, optional
        Whether to replace an existing file at `path`. Default: ``False``,
        which means an existing file is kept and the call raises
        `InputError`.

    Raises
    ------
    InputError
        If a file exists at `path` and `overwrite` is false.
    InputError
        If `columns` is empty.
    InputError
        If a column name is empty or holds whitespace.
    InputError
        If a column is not one-dimensional.
    InputError
        If a column is not of integer or float type.
    InputError
        If two columns differ in length.
    ResultError
        If a value read back differs from the value written, rounded to 14
        digits after the decimal point.

    See Also
    --------
    write_result : The HDF5 file that records a result.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> from pianistpy.io import write_table
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "table.txt")
    ...     columns = {"r": np.array([1.0, 2.0]), "n": np.array([3, 4])}
    ...     write_table(path, columns, header=["radii and counts"])
    ...     with open(path) as file:
    ...         print(file.read(), end="")
    # radii and counts
    # columns: r n
    1.00000000000000e+00     3
    2.00000000000000e+00     4
    """
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise InputError(
            f"The file {path!r} exists; pass overwrite=True to replace the file."
        )
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
            f"The file {path!r} does not read back as written, so the file is not "
            "a faithful copy of the columns."
        )
