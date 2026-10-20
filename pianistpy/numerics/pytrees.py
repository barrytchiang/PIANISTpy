"""Result objects as pytrees: traced data, static configuration.

A frozen result dataclass is registered with the names of its data fields
(arrays and numbers, the leaves JAX traces, maps and differentiates); every
other field is static configuration carried beside the tree, compared by
value where it is hashable and by identity otherwise. Unflattening bypasses
the constructor, so no validation meets a traced value (the gradient
contract: a Python branch never reads a tracer).
"""

__all__ = ["register_result"]

import dataclasses
from typing import Any

import jax.tree_util


class _Static:
    """A static field's value, hashable for the tree definition."""

    __slots__ = ("key", "value")

    def __init__(self, value: Any) -> None:
        self.value = value
        try:
            hash(value)
            self.key: Any = ("value", value)
        except TypeError:
            self.key = ("identity", id(value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Static) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def register_result(cls: type, data_fields: tuple[str, ...]) -> type:
    """Make a frozen dataclass a pytree.

    Parameters
    ----------
    cls : type
        The dataclass.
    data_fields : tuple of str
        The fields that are leaves or subtrees (arrays, numbers, registered
        objects). The remaining fields are static.

    Returns
    -------
    cls : type
        The same class, registered.
    """
    names = tuple(f.name for f in dataclasses.fields(cls))
    missing = set(data_fields) - set(names)
    if missing:
        raise ValueError(f"{cls.__name__} has no field {sorted(missing)}.")
    meta_fields = tuple(n for n in names if n not in data_fields)

    def flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(obj, n) for n in data_fields)
        aux = tuple(_Static(getattr(obj, n)) for n in meta_fields)
        return children, aux

    def unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        obj: Any = object.__new__(cls)
        for name, value in zip(data_fields, children, strict=True):
            object.__setattr__(obj, name, value)
        for name, static in zip(meta_fields, aux, strict=True):
            object.__setattr__(obj, name, static.value)
        return obj

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls
