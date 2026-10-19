# Results and files (pianistpy.io)

Source: https://pianistpy.com/docs/reference/api/io

# Results and files (pianistpy.io)

Reading and writing results.

class pianistpy.io.StoredResult(kind, identity, data_sha256, record, arrays, parents, provenance)

A result as it is stored in a file.

Parameters:

kind (str) – Kind of result, for example "model".

identity (str) – Identity string of the result.

data_sha256 (str) – Digest of the arrays.

record (dict) – The inputs and settings that determine the result, in dictionary form.

arrays (dict of str to array) – The arrays of the result, by name.

parents (tuple of str) – Identity strings of the results this one was computed from.

provenance (dict) – Date, versions and backend of the run that wrote the file.

pianistpy.io.backend_provenance()

Return the versions and the numerical backend of the running session.

Returns:

provenance – The Python, NumPy and SciPy versions, the platform JAX was pinned to
by the environment, and the backend and devices JAX reports. A
session without JAX says so by name.

Return type:

dict of str to str

pianistpy.io.data_digest(source)

Return the SHA-256 digest of every dataset of a file.

The digest covers, for each dataset in the sorted order of its path, the
path, the type, the shape and the raw bytes. Attributes are excluded, so
the digest can be stored in the file it describes.

Parameters:

source (h5py.Group or path) – An open file or group, or the path of a file.

Returns:

digest – Hexadecimal digest, 64 characters.

Return type:

str

pianistpy.io.load_result(source)

Return a stored result from a live object or from a path.

Every function that takes a result accepts either form and calls this
once, so that the two forms follow the same code afterwards.

Parameters:

source (StoredResult or path) – A stored result, or the path of a file that holds one.

Returns:

stored

Return type:

StoredResult

pianistpy.io.read(path)

Read a result from a file, whatever kind of result it holds.

Parameters:

path (path) – A file written by the write method of a result.

Returns:

result – The distribution function, model, particles or other result the file
holds, equal to the one that was written.

Return type:

object

Raises:

ResultError – If the file is not a result of this package, if its arrays no longer
    match the digest it records, or if its kind is unknown.

pianistpy.io.read_result(path)

Read one result from an HDF5 file and check it.

Parameters:

path (path) – File to read.

Returns:

stored

Return type:

StoredResult

Raises:

ResultError – If the file is not a result of this format, or if its arrays no
    longer match the digest it records.

pianistpy.io.register_reader(kind, reader)

Make reader the function that rebuilds the results of one kind from a file.

Parameters:

kind (str) – Kind of result, as stored in the file.

reader (callable) – Takes the stored result and returns the result object.

Return type:

None

pianistpy.io.write_result(path, kind, identity, record, arrays, parents=(), overwrite=False)

Write one result to an HDF5 file.

Parameters:

path (path) – File to write.

kind (str) – Kind of result.

identity (str) – Identity string of the result.

record (mapping) – The inputs and settings that determine the result, in dictionary form.

arrays (mapping of str to array-like) – The arrays of the result, by name. A name may contain / to place
the array in a group.

parents (sequence of str, optional) – Identity strings of the results this one was computed from.

overwrite (bool, optional) – Replace an existing file. By default an existing file is refused.

Returns:

data_sha256 – Digest of the arrays as written.

Return type:

str

Raises:

InputError – If the file exists and overwrite is false.

ResultError – If the file does not read back as written.

pianistpy.io.write_table(path, columns, header=(), overwrite=False)

Write named columns as fixed-width text with a header.

The header lines, each starting with #, are written as given,
followed by # columns: and the column names in order. Each row holds
one value of every column, 25 characters a field: a float with 14 digits
after the point in exponent form, an integer as it is. The file is read
back with numpy.loadtxt and compared with the values written, to the
digits written; a file that does not read back is refused.

Parameters:

path (path) – File to write.

columns (mapping of str to array-like) – The columns, in order; one-dimensional, numeric, of one length. A
name holds no space.

header (sequence of str, optional) – Lines for the head of the file, without the leading #.

overwrite (bool, optional) – Replace an existing file.

Raises:

InputError – For a column that is not one-dimensional and numeric, columns of
    different lengths, a name with a space, or an existing file without
    overwrite.

ResultError – If the file does not read back as written.

Return type:

None

     On this page
