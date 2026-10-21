# Exceptions

Source: https://pianistpy.com/docs/reference/api/exceptions

Module `pianistpy.exceptions`

The exception classes and warning classes of PIANISTpy.

Every error the package raises is an instance of PianistpyError. Every
warning the package issues is an instance of PianistpyWarning.

exception pianistpy.exceptions.AccuracyError(*message*, *result=None*)

Raised when a result’s relative error is above \(10^{-2}\).

AccuracyError is also raised when the resolution of the settings cannot
resolve the result. An AccuracyError raised after the result is built
carries the result. With `Settings(write_anyway=True)`, the call
returns such a result, with its validation report, and raises no
AccuracyError.

Parameters:

- **message** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The check the result fails, the relative error of the failed check,
and the position where the failed check’s relative error occurs.

- **result** ([*object*](https://docs.python.org/3/builtins/functions.html#object)*, **optional*) – The result, with its validation report. The user can inspect the
attached result without computing the result a second time.
Default: `None`, which means no result is attached.

Variables:

**result** ([*object*](https://docs.python.org/3/builtins/functions.html#object)* or **None*) – The result passed as result. None where the AccuracyError is
raised before the result exists.

Return type:

None

See also

[`AccuracyWarning`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyWarning)
Issued when a result’s relative error is above \(10^{-4}\).

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.AccuracyError, pp.PianistpyError)
True
```

exception pianistpy.exceptions.AccuracyWarning

Issued when a result’s relative error is above \(10^{-4}\).

A result with a relative error above \(10^{-4}\) and at most
\(10^{-2}\) has the validation report tier `"warned"`. A result
with a relative error above \(10^{-2}\) also issues AccuracyWarning
when `Settings(write_anyway=True)` lets the call return the result.

See also

[`AccuracyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError)
Raised when a result’s relative error is above \(10^{-2}\).

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.AccuracyWarning, pp.PianistpyWarning)
True
```

exception pianistpy.exceptions.InputError(**args*)

Raised when a call does not accept one of its inputs.

A call raises InputError for an input of the wrong type, an input
outside its valid range, or an input inconsistent with another input of
the same call. The message of an InputError identifies the input and the rule
the input breaks. InputError is a subclass of `ValueError`.

See also

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

Examples

```python
>>> import pianistpy as pp
>>> try:
...     pp.Aperture.circle(-1.0)
... except pp.InputError as error:
...     print(type(error).__name__, isinstance(error, ValueError))
InputError True
```

Parameters:

**args** ([*object*](https://docs.python.org/3/builtins/functions.html#object))

Return type:

None

exception pianistpy.exceptions.PianistpyError(**args*)

Base class of every error raised by PIANISTpy.

Variables:

**recent** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The most recent messages of the package at the moment the error is
raised, oldest first. The package keeps the recent messages at every
verbosity setting. A run with its messages silenced therefore still
reports, in recent, the steps the run took before the error.

Parameters:

**args** ([*object*](https://docs.python.org/3/builtins/functions.html#object))

Return type:

None

See also

[`PianistpyWarning`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyWarning)
Base class of every warning issued by PIANISTpy.

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.InputError, pp.PianistpyError)
True
```

exception pianistpy.exceptions.PianistpyWarning

Base class of every warning issued by PIANISTpy.

See also

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.PianistpyWarning, UserWarning)
True
```

exception pianistpy.exceptions.ResultError(**args*)

Raised when a result fails a check of its integrity.

Reading a file raises ResultError in three cases: the file’s format
version is unknown; the file’s data digest does not match the arrays
stored in the file; or the parent of the stored result differs from the
parent the file records. Writing a file raises ResultError when the
written file does not read back as written. A computation raises
ResultError when an intermediate result fails a check that the next
step of the computation needs, such as values that are not finite or
radii that do not increase.

See also

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.ResultError, pp.PianistpyError)
True
```

Parameters:

**args** ([*object*](https://docs.python.org/3/builtins/functions.html#object))

Return type:

None

exception pianistpy.exceptions.UnphysicalModelError(*message*, *result=None*)

Raised when the requested model has no valid equilibrium.

Parameters:

- **message** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The condition the model fails and the values of the quantities in
the failed condition.

- **result** ([*object*](https://docs.python.org/3/builtins/functions.html#object)*, **optional*) – The distribution function computed before the error is raised. The
user can inspect the attached distribution function without
computing the distribution function a second time. Default:
`None`, which means no distribution function is attached.

Variables:

**result** ([*object*](https://docs.python.org/3/builtins/functions.html#object)* or **None*) – The distribution function passed as result, or None.

Return type:

None

See also

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

[`AccuracyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError)
Raised when a result’s relative error is above \(10^{-2}\).

Examples

```python
>>> import pianistpy as pp
>>> issubclass(pp.UnphysicalModelError, pp.PianistpyError)
True
```

exception pianistpy.exceptions.WorkerError(**args*)

Raised when a parallel process ends before finishing its task.

A call raises WorkerError as soon as the call finds one of its parallel
processes ended. The call does not wait for the lost task’s result. The
usual cause of a WorkerError is a lack of memory. The operating system
ends a process when the machine runs out of memory. Fewer cores lower
the memory a call uses. The number of cores is set by set_cores or by
the environment variable
`PIANISTPY_CORES`. WorkerError is a subclass of `RuntimeError`.

Under a JAX transformation such as `jax.grad`, the caller receives
`jax.errors.JaxRuntimeError` with the message of the WorkerError.

See also

[`PianistpyError`](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.PianistpyError)
Base class of every error raised by PIANISTpy.

Examples

```python
>>> from pianistpy.exceptions import PianistpyError, WorkerError
>>> issubclass(WorkerError, PianistpyError), issubclass(WorkerError, RuntimeError)
(True, True)
```

Parameters:

**args** ([*object*](https://docs.python.org/3/builtins/functions.html#object))

Return type:

None
