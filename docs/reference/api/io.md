# Results and files

Source: https://pianistpy.com/docs/reference/api/io

Module `pianistpy.io`

Writing results to files and reading results back from files.

A result is written by the result’s `write` method and read back by read.
write_result and read_result implement the HDF5 file format that the
`write` methods and read share. write_table exports columns of a result,
each with a column name, as a text table.

class pianistpy.io.StoredResult(*kind*, *identity*, *data_sha256*, *record*, *arrays*, *parents*, *provenance*)

A result as one file stores the result.

read_result returns a StoredResult; the reader registered for the
result’s kind (register_reader) rebuilds the result object from the
StoredResult.

Parameters:

- **kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind of result, for example `"model"`.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The identity string of the result.

- **data_sha256** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The SHA-256 digest of the file’s datasets (data_digest), as a
hexadecimal string of 64 characters.

- **record** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The inputs and settings that determine the result, in dictionary form.

- **arrays** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to array*) – The arrays of the result, by name.

- **parents** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The identity strings of the results from which the result was
computed.

- **provenance** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to str*) – The creation date of the file and the versions and numerical backend
of the session that wrote the file (backend_provenance).

See also

[`read_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read_result)
The function that reads a file into a StoredResult.

[`write_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.write_result)
The function that writes a result to a file.

Examples

```python
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
```

pianistpy.io.backend_provenance()

Return the software versions and the numerical backend of the running session.

write_result stores the returned dictionary in every file the function
writes.

Returns:

**provenance** – The keys `python_version`, `numpy_version` and `scipy_version`;
`jax_platforms_env` and `pianistpy_platform_env`, the values of
the environment variables `JAX_PLATFORMS` and
`PIANISTPY_PLATFORM`, each `"unset"` where the variable is not
set; and `jax_version`, `jax_backend` and `jax_devices`. A
session in which JAX cannot be imported records `"unavailable"` for
`jax_version` and `jax_devices` and `"none"` for
`jax_backend`.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict) of str to str

See also

[`write_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.write_result)
The function that stores the provenance in each file.

Examples

```python
>>> from pianistpy.io import backend_provenance
>>> sorted(backend_provenance())[:3]
['jax_backend', 'jax_devices', 'jax_platforms_env']
```

pianistpy.io.data_digest(*source*)

Return the SHA-256 digest of every dataset of an HDF5 file or group.

The digest covers every dataset, in the sorted order of the datasets’
paths: each dataset’s path, data type, shape and raw bytes. The digest
excludes the attributes, so a file can store the digest of its own
datasets as an attribute.

Parameters:

**source** (*h5py.Group** or **path*) – An open HDF5 file or group, or the path of an HDF5 file.

Returns:

**digest** – The hexadecimal digest, 64 characters.

Return type:

[str](https://docs.python.org/3/builtins/stdtypes.html#str)

See also

[`write_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.write_result)
The function that records the digest in each file the function writes.

[`read_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read_result)
The function that compares the recorded digest with a recomputed digest.

Examples

```python
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
```

pianistpy.io.load_result(*source*)

Return a StoredResult, given the StoredResult or the path of a result file.

A StoredResult passed as source is returned unchanged. A path is read
and checked by read_result.

Parameters:

**source** ([*StoredResult*](https://pianistpy.com/docs/reference/api/io#pianistpy.io.StoredResult)* or *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – A StoredResult, or the path of a file that holds one result.

Returns:

**stored** – source itself when source is a StoredResult; otherwise the
StoredResult that read_result reads from the file.

Return type:

[StoredResult](https://pianistpy.com/docs/reference/api/io#pianistpy.io.StoredResult)

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a path and read_result raises ResultError: the file
    lacks an attribute of a result file, the major number of the file’s
    format version differs from the major number the package reads, or
    the recomputed digest differs from the recorded digest.

See also

[`read_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read_result)
The function that reads and checks a file.

Examples

```python
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
```

pianistpy.io.read(*path*)

Read a result from a file, whatever kind of result the file holds.

The call reads and checks the file with read_result and passes the
StoredResult to the reader registered for the file’s kind
(register_reader).

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – A file written by the `write` method of a result.

Returns:

**result** – The distribution function, model, particles or other result the file
holds, equal to the result that was written.

Return type:

[object](https://docs.python.org/3/builtins/functions.html#object)

Raises:

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the file lacks one of the attributes `format_version`,
    `result_kind`, `identity`, `data_sha256`, `record` and
    `parents`.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the major number of the file’s format version differs from the
    major number of the format version the installed package reads.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the recomputed digest of the datasets differs from the digest the
    file records.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If no reader is registered for the kind the file records.

See also

[`register_reader`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.register_reader)
The registration of the function that rebuilds a kind.

[`read_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read_result)
The read and the check of the file.

Examples

```python
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
```

pianistpy.io.read_result(*path*)

Read one result from an HDF5 file and check the file’s digest.

The call recomputes the SHA-256 digest of the file’s datasets
(data_digest) and compares the recomputed digest with the digest the
file records.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – The file to read.

Returns:

**stored** – The arrays, record, identity string, digest, parents and provenance
the file holds.

Return type:

[StoredResult](https://pianistpy.com/docs/reference/api/io#pianistpy.io.StoredResult)

Raises:

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the file lacks one of the attributes `format_version`,
    `result_kind`, `identity`, `data_sha256`, `record` and
    `parents`.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the major number of the file’s format version differs from the
    major number of the format version the installed package reads.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the recomputed digest differs from the digest the file records.

See also

[`read`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read)
The function that rebuilds the result object from the file.

[`write_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.write_result)
The function that writes the file.

Examples

```python
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
```

pianistpy.io.register_reader(*kind*, *reader*)

Make reader the function that rebuilds every result of one kind.

read calls the function registered for the kind a file records. A
second registration for the same kind replaces the first registration.

Parameters:

- **kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind of result, as write_result stores the kind in the file.

- **reader** (*callable*) – A function that takes the StoredResult and returns the result
object.

Return type:

None

See also

[`read`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read)
The function that calls the registered reader.

Examples

```python
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
```

pianistpy.io.write_result(*path*, *kind*, *identity*, *record*, *arrays*, *parents=()*, *overwrite=False*)

Write one result to an HDF5 file and check the file against the request.

After writing, the call reads the file back and compares the file’s
arrays, bit for bit, and the file’s record, identity string, kind and
parents with the values passed. The call stores the SHA-256 digest of the
datasets (data_digest) in the file only when every comparison holds.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – The file to write.

- **kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind of result, the key under which register_reader finds the
function that rebuilds the result.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The identity string of the result.

- **record** (*mapping*) – The inputs and settings that determine the result, in dictionary
form. Every value is one that `json.dumps` writes.

- **arrays** (*mapping** of **str to array-like*) – The arrays of the result, by name. A name may contain `/` to place
the array in an HDF5 group.

- **parents** (*sequence** of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – The identity strings of the results from which the result was
computed. Default: `()`, which means the result has no parent
results.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default: `False`,
which means an existing file is kept and the call raises
InputError.

Returns:

**data_sha256** – The SHA-256 digest of the datasets as written (data_digest), the
same digest the file stores.

Return type:

[str](https://docs.python.org/3/builtins/stdtypes.html#str)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the file read back differs from the request in an array, the
    record, the identity string, the kind or the parents.

See also

[`read_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.read_result)
The function that reads the file back.

[`data_digest`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.data_digest)
The digest of the datasets.

Examples

```python
>>> import os
>>> import tempfile
>>> import numpy as np
>>> from pianistpy.io import write_result
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "example.h5")
...     digest = write_result(path, "example", "example-1", {}, {"x": np.ones(2)})
>>> len(digest)
64
```

pianistpy.io.write_table(*path*, *columns*, *header=()*, *overwrite=False*)

Write columns, each with a column name, as a fixed-width text table.

The file opens with the lines of header, each prefixed with `# ``,
followed by the line ``# columns:` and the column names in order. Each
following row holds one value of every column, each value in a field of
25 characters: a float in exponent form with 14 digits after the decimal
point, an integer as the integer. After writing, the call reads the file
back with `numpy.loadtxt` and compares the values read with the values
written, rounded to the 14 digits the file holds.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – The file to write.

- **columns** (*mapping** of **str to array-like*) – The columns, in order. Each column is one-dimensional, of integer or
float type, and of the length of every other column. Each column name
is non-empty and holds no whitespace.

- **header** (*sequence** of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – The lines at the head of the file, each without the leading `#`.
Default: `()`, which means the head holds only the line of column
names.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default: `False`,
which means an existing file is kept and the call raises
InputError.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If columns is empty.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a column name is empty or holds whitespace.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a column is not one-dimensional.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a column is not of integer or float type.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If two columns differ in length.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a value read back differs from the value written, rounded to 14
    digits after the decimal point.

Return type:

None

See also

[`write_result`](https://pianistpy.com/docs/reference/api/io#pianistpy.io.write_result)
The HDF5 file that records a result.

Examples

```python
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
```
