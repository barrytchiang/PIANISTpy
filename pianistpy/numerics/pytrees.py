"""Registration of frozen result dataclasses as JAX pytrees (`register_result`).

`register_result` registers a frozen dataclass as a pytree with the names of
its data fields. The data fields hold arrays, numbers and registered objects:
these fields are the leaves that JAX traces, maps and differentiates. Every
other field is static configuration that the pytree carries beside the
leaves. A static field is compared by value when the field's value is
hashable, and by identity otherwise. Unflattening sets the fields directly
and does not call the constructor, so that no input check of the
constructor compares a traced value in a Python branch.
"""

__all__ = ["register_result"]

import dataclasses
from typing import Any

import jax.tree_util


class _Static:
    """Wrapper of a static field's value, hashable for the pytree definition.

    A hashable value is compared by value. An unhashable value is compared by
    the identity of the object.
    """

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
    """Declare a frozen dataclass as a JAX pytree and return the class.

    Parameters
    ----------
    cls : type
        The frozen dataclass to declare as a JAX pytree.
    data_fields : tuple of str
        Names of the fields that are leaves or subtrees of the pytree:
        arrays, numbers and registered objects. Every other field of `cls`
        is static.

    Returns
    -------
    cls : type
        The class `cls` itself, now registered.

    Raises
    ------
    ValueError
        If a name in `data_fields` is not a field of `cls`.
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
