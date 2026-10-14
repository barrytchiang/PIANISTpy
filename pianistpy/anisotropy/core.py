"""The anisotropy model: isotropic, constant, OM and COM as one family."""

__all__ = ["Anisotropy"]

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.tree_util
import numpy as np

from ..exceptions import InputError

KINDS = ("I", "II")
BETA_C_MAX = 1.0


@dataclass(frozen=True)
class Anisotropy:
    r"""Velocity anisotropy of one component.

    The isotropic, constant-anisotropy, Osipkov-Merritt (OM) and
    Cuddeford-Osipkov-Merritt (COM) models are regions of the parameters of
    this one class, since the COM model contains the other three.

    Parameters
    ----------
    beta_c : float, optional
        Central anisotropy :math:`\beta_{\rm c}`, smaller than 1. The default,
        exactly 0, gives an isotropic or an OM model.
    r_a : float, optional
        Anisotropy radius :math:`R_{\rm a}` in internal units, positive. The
        default, infinity, gives an isotropic or a constant-anisotropy model.
    kind : {"I", "II"}, optional
        Type of the OM or COM model. Ignored when `r_a` is infinite.

    Notes
    -----
    The model is identified by exact comparison of the parameters as given:
    ``beta_c == 0`` and an infinite `r_a`. A value such as ``1e-12`` is a
    constant-anisotropy model and is not rounded to zero. The package is
    tested to give results that are continuous through these special values.

    Examples
    --------
    >>> Anisotropy().model
    'isotropic'
    >>> Anisotropy(beta_c=0.3).model
    'constant'
    >>> Anisotropy(r_a=10.0).model
    'om'
    >>> Anisotropy(beta_c=-0.25, r_a=10.0, kind="II").model
    'com'
    """

    beta_c: float = 0.0
    r_a: float = math.inf
    kind: str = "I"
    family: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        if not self.beta_c < BETA_C_MAX:
            raise InputError(
                f"beta_c must be smaller than {BETA_C_MAX}; got {self.beta_c}."
            )
        if not self.r_a > 0.0:
            raise InputError(f"r_a must be positive; got {self.r_a}.")
        if self.kind not in KINDS:
            raise InputError(f"kind must be one of {list(KINDS)}; got {self.kind!r}.")
        # The family is decided here, on concrete values, and stored: no
        # method compares the parameters again, so that a traced beta_c or
        # r_a (a fit) never meets a Python branch (the gradient contract).
        if math.isinf(self.r_a):
            family = "isotropic" if self.beta_c == 0.0 else "constant"
        else:
            family = ("OM" if self.beta_c == 0.0 else "COM") + "_" + self.kind
        object.__setattr__(self, "family", family)

    def factor(self, r: Any, xp: Any = np) -> Any:
        r"""Return the factor :math:`g(R)` of the augmented density.

        The one definition of :math:`g` in the package: every consumer of
        :math:`\varrho_{\rm aug} = g(R)\,\varrho_{\rm aux}` reads it here, in
        this order of operations.

        Parameters
        ----------
        r : array-like
            Radius, in internal units.
        xp : module, optional
            The array namespace, NumPy by default; JAX's inside a kernel.

        Returns
        -------
        g : array
            :math:`g(R)`: 1 (isotropic), :math:`R^{-2\beta_{\rm c}}`
            (constant), :math:`(1 \pm R^2/R_{\rm a}^2)^{-1}` (OM) and
            :math:`R^{-2\beta_{\rm c}} (1 \pm R^2/R_{\rm a}^2)^{-(1-\beta_{\rm c})}`
            (COM), the upper sign for Type I. A Type II model is defined
            inside :math:`R_{\rm a}` only; beyond it the value is not
            clamped and not meaningful.

        References
        ----------
        Paper Eq. (B43) [eqn:linearity_beta_1].
        """
        r = xp.asarray(r, dtype=xp.float64)
        family = self.family
        if family == "isotropic":
            return xp.ones_like(r)
        if family == "constant":
            return r ** (-2.0 * self.beta_c)
        base = (
            1.0 + (r / self.r_a) ** 2
            if family.endswith("_I")
            else 1.0 - (r / self.r_a) ** 2
        )
        if family.startswith("OM"):
            return 1.0 / base
        return r ** (-2.0 * self.beta_c) * base ** (-(1.0 - self.beta_c))

    @property
    def model(self) -> str:
        """Label of the model in the paper: isotropic, constant, om or com."""
        return self.family.split("_")[0].lower()

    @property
    def domain_edge(self) -> float:
        """The outer edge of the radii where the anisotropy is defined: R_a
        for a Type II model (its factor is real, finite and positive only
        inside it), infinite otherwise."""
        if self.kind == "II" and math.isfinite(self.r_a):
            return float(self.r_a)
        return math.inf

    def to_dict(self) -> dict[str, Any]:
        """Return the anisotropy as a plain dictionary."""
        # An infinite radius is written as None, since JSON has no infinity.
        r_a = None if math.isinf(self.r_a) else self.r_a
        return {
            "model": self.model,
            "beta_c": self.beta_c,
            "r_a": r_a,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Anisotropy":
        """Build the anisotropy from its dictionary form."""
        r_a = record.get("r_a")
        return cls(
            beta_c=float(record.get("beta_c", 0.0)),
            r_a=math.inf if r_a is None else float(r_a),
            kind=str(record.get("kind", "I")),
        )


def _flatten_anisotropy(a: Anisotropy):
    # beta_c and r_a are the traced parameters; the kind and the family the
    # static configuration (the gradient contract, R-47 item 9).
    return (a.beta_c, a.r_a), (a.kind, a.family)


def _unflatten_anisotropy(aux, children) -> Anisotropy:
    kind, family = aux
    beta_c, r_a = children
    a = object.__new__(Anisotropy)
    object.__setattr__(a, "beta_c", beta_c)
    object.__setattr__(a, "r_a", r_a)
    object.__setattr__(a, "kind", kind)
    object.__setattr__(a, "family", family)
    return a


jax.tree_util.register_pytree_node(
    Anisotropy, _flatten_anisotropy, _unflatten_anisotropy
)
