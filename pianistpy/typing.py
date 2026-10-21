"""The type aliases of the package's annotations.

Both aliases stand for ``Any``, so that an annotation documents the kind of
value without restricting the array library."""

__all__ = ["Array", "ArrayLike"]

from typing import Any, TypeAlias

# An array of JAX or of NumPy.
Array: TypeAlias = Any
# A value that converts to an array: a number, a sequence of numbers, or an
# array.
ArrayLike: TypeAlias = Any
