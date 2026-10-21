"""The exception classes and warning classes of PIANISTpy.

Every error the package raises is an instance of `PianistpyError`. Every
warning the package issues is an instance of `PianistpyWarning`.
"""

__all__ = [
    "AccuracyError",
    "AccuracyWarning",
    "InputError",
    "PianistpyError",
    "PianistpyWarning",
    "ResultError",
    "UnphysicalModelError",
    "WorkerError",
]

from typing import Any


class PianistpyError(Exception):
    """Base class of every error raised by PIANISTpy.

    Attributes
    ----------
    recent : tuple of str
        The most recent messages of the package at the moment the error is
        raised, oldest first. The package keeps the recent messages at every
        verbosity setting. A run with its messages silenced therefore still
        reports, in `recent`, the steps the run took before the error.

    See Also
    --------
    PianistpyWarning : Base class of every warning issued by PIANISTpy.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.InputError, pp.PianistpyError)
    True
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        # The import of pianistpy.runtime sits inside __init__ because
        # pianistpy.runtime imports this module to raise these errors.
        from .runtime import recent_messages

        self.recent = recent_messages()


class InputError(PianistpyError, ValueError):
    """Raised when a call does not accept one of its inputs.

    A call raises InputError for an input of the wrong type, an input
    outside its valid range, or an input inconsistent with another input of
    the same call. The message of an InputError identifies the input and the rule
    the input breaks. InputError is a subclass of ``ValueError``.

    See Also
    --------
    PianistpyError : Base class of every error raised by PIANISTpy.

    Examples
    --------
    >>> import pianistpy as pp
    >>> try:
    ...     pp.Aperture.circle(-1.0)
    ... except pp.InputError as error:
    ...     print(type(error).__name__, isinstance(error, ValueError))
    InputError True
    """


class UnphysicalModelError(PianistpyError):
    """Raised when the requested model has no valid equilibrium.

    Parameters
    ----------
    message : str
        The condition the model fails and the values of the quantities in
        the failed condition.
    result : object, optional
        The distribution function computed before the error is raised. The
        user can inspect the attached distribution function without
        computing the distribution function a second time. Default:
        ``None``, which means no distribution function is attached.

    Attributes
    ----------
    result : object or None
        The distribution function passed as `result`, or None.

    See Also
    --------
    PianistpyError : Base class of every error raised by PIANISTpy.
    AccuracyError : Raised when a result's relative error is above
        :math:`10^{-2}`.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.UnphysicalModelError, pp.PianistpyError)
    True
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class AccuracyError(PianistpyError):
    """Raised when a result's relative error is above :math:`10^{-2}`.

    AccuracyError is also raised when the resolution of the settings cannot
    resolve the result. An AccuracyError raised after the result is built
    carries the result. With ``Settings(write_anyway=True)``, the call
    returns such a result, with its validation report, and raises no
    AccuracyError.

    Parameters
    ----------
    message : str
        The check the result fails, the relative error of the failed check,
        and the position where the failed check's relative error occurs.
    result : object, optional
        The result, with its validation report. The user can inspect the
        attached result without computing the result a second time.
        Default: ``None``, which means no result is attached.

    Attributes
    ----------
    result : object or None
        The result passed as `result`. None where the AccuracyError is
        raised before the result exists.

    See Also
    --------
    AccuracyWarning : Issued when a result's relative error is above
        :math:`10^{-4}`.
    PianistpyError : Base class of every error raised by PIANISTpy.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.AccuracyError, pp.PianistpyError)
    True
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class ResultError(PianistpyError):
    """Raised when a result fails a check of its integrity.

    Reading a file raises ResultError in three cases: the file's format
    version is unknown; the file's data digest does not match the arrays
    stored in the file; or the parent of the stored result differs from the
    parent the file records. Writing a file raises ResultError when the
    written file does not read back as written. A computation raises
    ResultError when an intermediate result fails a check that the next
    step of the computation needs, such as values that are not finite or
    radii that do not increase.

    See Also
    --------
    PianistpyError : Base class of every error raised by PIANISTpy.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.ResultError, pp.PianistpyError)
    True
    """


class WorkerError(PianistpyError, RuntimeError):
    """Raised when a parallel process ends before finishing its task.

    A call raises WorkerError as soon as the call finds one of its parallel
    processes ended. The call does not wait for the lost task's result. The
    usual cause of a WorkerError is a lack of memory. The operating system
    ends a process when the machine runs out of memory. Fewer cores lower
    the memory a call uses. The number of cores is set by `set_cores` or by
    the environment variable
    ``PIANISTPY_CORES``. WorkerError is a subclass of ``RuntimeError``.

    Under a JAX transformation such as ``jax.grad``, the caller receives
    ``jax.errors.JaxRuntimeError`` with the message of the WorkerError.

    See Also
    --------
    PianistpyError : Base class of every error raised by PIANISTpy.

    Examples
    --------
    >>> from pianistpy.exceptions import PianistpyError, WorkerError
    >>> issubclass(WorkerError, PianistpyError), issubclass(WorkerError, RuntimeError)
    (True, True)
    """


class PianistpyWarning(UserWarning):
    """Base class of every warning issued by PIANISTpy.

    See Also
    --------
    PianistpyError : Base class of every error raised by PIANISTpy.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.PianistpyWarning, UserWarning)
    True
    """


class AccuracyWarning(PianistpyWarning):
    """Issued when a result's relative error is above :math:`10^{-4}`.

    A result with a relative error above :math:`10^{-4}` and at most
    :math:`10^{-2}` has the validation report tier ``"warned"``. A result
    with a relative error above :math:`10^{-2}` also issues AccuracyWarning
    when ``Settings(write_anyway=True)`` lets the call return the result.

    See Also
    --------
    AccuracyError : Raised when a result's relative error is above
        :math:`10^{-2}`.

    Examples
    --------
    >>> import pianistpy as pp
    >>> issubclass(pp.AccuracyWarning, pp.PianistpyWarning)
    True
    """
