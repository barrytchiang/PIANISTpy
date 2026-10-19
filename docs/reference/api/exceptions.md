# Exceptions (pianistpy.exceptions)

Source: https://pianistpy.com/docs/reference/api/exceptions

# Exceptions (pianistpy.exceptions)

Exceptions and warnings raised by PIANISTpy.

exception pianistpy.exceptions.AccuracyError(*args)

Raised when a result misses the hard accuracy limit.

Parameters:

args (object)

Return type:

None

exception pianistpy.exceptions.AccuracyWarning

Issued when a result lies between the stated accuracy and the hard limit.

exception pianistpy.exceptions.InputError(*args)

Raised when an input parameter is outside its admissible range.

Parameters:

args (object)

Return type:

None

exception pianistpy.exceptions.PianistpyError(*args)

Base class of every error raised by PIANISTpy.

Variables:

recent (tuple of str) – The most recent messages of the package when the error was raised,
oldest first. They are kept whatever the verbosity, so that a quiet
run that fails can still say what it was doing.

Parameters:

args (object)

Return type:

None

exception pianistpy.exceptions.PianistpyWarning

Base class of every warning issued by PIANISTpy.

exception pianistpy.exceptions.ResultError(*args)

Raised when a file does not hold the result it is taken for.

This covers an unknown format version, a data digest that does not match
the stored arrays, and a parent whose identity differs from the one the
file records.

Parameters:

args (object)

Return type:

None

exception pianistpy.exceptions.UnphysicalModelError(message, result=None)

Raised when no valid equilibrium exists for the requested model.

Parameters:

message (str) – The condition that failed and the values involved.

result (object, optional) – The distribution function that was computed before the model was
refused, so that it can be inspected without a second computation.

Variables:

result (object or None) – The distribution function attached to the refusal.

Return type:

None

     On this page
