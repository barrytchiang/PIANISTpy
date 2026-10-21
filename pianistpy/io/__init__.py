"""Writing results to files and reading results back from files.

A result is written by the result's ``write`` method and read back by `read`.
`write_result` and `read_result` implement the HDF5 file format that the
``write`` methods and `read` share. `write_table` exports columns of a result,
each with a column name, as a text table.
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
    "write_table",
]

from .core import (
    StoredResult,
    backend_provenance,
    data_digest,
    load_result,
    read,
    read_result,
    register_reader,
    write_result,
)
from .text import write_table
