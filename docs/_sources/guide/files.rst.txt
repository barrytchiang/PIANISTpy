Input and output
================

Every result (a distribution function, a model, a set of particles) is an immutable object that holds what it
was computed from: the inputs, the numerical settings and the version of the code. Its ``write`` method stores
it in a single HDF5 file, and ``pp.read(path)`` returns an equal object, whatever kind of result the file holds.
Nothing is written unless requested.

A ``Model`` is always a valid equilibrium. A request for which none exists raises ``UnphysicalModelError``, and
the exception carries the distribution function for inspection.
