"""Type aliases used in annotations."""

__all__ = ["Array", "ArrayLike"]

from typing import Any, TypeAlias

# An array of the numerical backend (JAX or NumPy).
Array: TypeAlias = Any
# Anything that converts to an array: a number, a sequence, an array.
ArrayLike: TypeAlias = Any
