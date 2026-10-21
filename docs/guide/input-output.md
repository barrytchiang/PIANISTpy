# Input and output

Source: https://pianistpy.com/docs/guide/input-output

Every result (a distribution function, a model, a set of particles) is an immutable object. Each result holds the
inputs, the numerical settings and the code version from which the result was computed. The `write` method of a
result stores the result in a single HDF5 file, for particles in the default layout `"native"`, and
`pp.read(path)` returns an equal object from every such file. The package writes a result file only when a write
method of the result is called.

The `write` method of a distribution function, a model or a set of particles raises `InputError`
for an existing file. The argument `overwrite=True` replaces the existing file. `read` raises
`ResultError` if the file is not a result of the package or if its arrays no longer match the
checksum the file records.
