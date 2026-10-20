"""Exceptions and warnings raised by PIANISTpy."""

__all__ = [
    "AccuracyError",
    "AccuracyWarning",
    "InputError",
    "PianistpyError",
    "PianistpyWarning",
    "ResultError",
    "UnphysicalModelError",
]

from typing import Any


class PianistpyError(Exception):
    """Base class of every error raised by PIANISTpy.

    Attributes
    ----------
    recent : tuple of str
        The most recent messages of the package when the error was raised,
        oldest first. They are kept whatever the verbosity, so that a quiet
        run that fails can still say what it was doing.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        # Imported here because the runtime module raises these errors itself.
        from .runtime import recent_messages

        self.recent = recent_messages()


class InputError(PianistpyError, ValueError):
    """Raised when an input parameter is outside its admissible range."""


class UnphysicalModelError(PianistpyError):
    """Raised when no valid equilibrium exists for the requested model.

    Parameters
    ----------
    message : str
        The condition that failed and the values involved.
    result : object, optional
        The distribution function that was computed before the model was
        refused, so that it can be inspected without a second computation.

    Attributes
    ----------
    result : object or None
        The distribution function attached to the refusal.
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class AccuracyError(PianistpyError):
    """Raised when a result misses the hard accuracy limit, or cannot be
    resolved at the settings' resolution.

    Parameters
    ----------
    message : str
        The check that failed, its error and where.
    result : object, optional
        The result that was refused, complete with its validation report,
        where the refusal came after it was built, so that it can be inspected
        without a second computation (``Settings(write_anyway=True)`` returns
        it instead).

    Attributes
    ----------
    result : object or None
        The refused result, or None where the refusal came inside a stage.
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class ResultError(PianistpyError):
    """Raised when a result is not what it is taken for.

    A file that does not hold the result it is taken for: an unknown format
    version, a data digest that does not match the stored arrays, a parent
    whose identity differs from the one the file records, a file that does
    not read back as written. Or a stage's output that fails a structural
    check the next stage needs: rows that are not finite or not increasing.
    """


class PianistpyWarning(UserWarning):
    """Base class of every warning issued by PIANISTpy."""


class AccuracyWarning(PianistpyWarning):
    """Issued when a result lies between the stated accuracy and the hard limit."""
