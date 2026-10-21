r"""The class `Anisotropy`, which covers four velocity anisotropy models.

The four models are the isotropic model, the constant-anisotropy model, the
Osipkov-Merritt (OM) model and the Cuddeford-Osipkov-Merritt (COM) model.
`Anisotropy` holds the central anisotropy :math:`\beta_{\rm c}`, the
anisotropy radius :math:`R_{\rm a}` and the type of an OM or COM model, and
returns the factor :math:`g(R)` of Chiang+(2026d), Eq. (B44).
"""

__all__ = ["Anisotropy"]

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.tree_util
import numpy as np

from ..exceptions import InputError
from .factor import inverse_factor

KINDS = ("I", "II")
BETA_C_MAX = 1.0


@dataclass(frozen=True)
class Anisotropy:
    r"""Velocity anisotropy of one component.

    The isotropic, constant-anisotropy, Osipkov-Merritt (OM) and
    Cuddeford-Osipkov-Merritt (COM) models are all instances of
    `Anisotropy`, since the COM model contains the other three models as
    special cases. The anisotropy profile of the COM model is

    .. math::

        \beta(R) = \frac{\beta_{\rm c} \pm R^2/R_{\rm a}^2}
        {1 \pm R^2/R_{\rm a}^2},

    with the upper sign for Type I and the lower sign for Type II.

    Parameters
    ----------
    beta_c : float, optional
        Central anisotropy :math:`\beta_{\rm c}`. `beta_c` must be smaller
        than 1. Default: ``0.0``, which means an isotropic model or an OM
        model.
    r_a : float, optional
        Anisotropy radius :math:`R_{\rm a}`, in units of :math:`r_0`, the
        unit of length of the internal and the model unit systems. `r_a`
        is a plain float, and an Astropy quantity is not accepted. `r_a`
        must be positive. Default: ``math.inf``, which means an isotropic model or a
        constant-anisotropy model.
    kind : {"I", "II"}, optional
        Type of the OM or COM model. The anisotropy of a Type I model rises
        outward to :math:`\beta = 1` (radial orbits) as :math:`R` grows. The
        anisotropy of a Type II model falls to :math:`\beta \to -\infty`
        (tangential orbits) as :math:`R` approaches :math:`R_{\rm a}`, and a
        Type II model is defined only at :math:`R < R_{\rm a}`. When `r_a`
        is infinite, the type does not enter the model, and the type is
        stored as ``"I"`` so that each model has one identity. Default:
        ``"I"``, which means a Type I model.

    Attributes
    ----------
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`.
    r_a : float
        Anisotropy radius :math:`R_{\rm a}`, in units of :math:`r_0`;
        ``math.inf`` for an isotropic or a constant-anisotropy model.
    kind : str
        Type of the model, ``"I"`` or ``"II"``.
    family : str
        Model and type together: ``"isotropic"``, ``"constant"``,
        ``"OM_I"``, ``"OM_II"``, ``"COM_I"`` or ``"COM_II"``.

    Raises
    ------
    InputError
        If `beta_c` is 1 or larger.
    InputError
        If `r_a` is zero or negative.
    InputError
        If `kind` is neither ``"I"`` nor ``"II"``.

    See Also
    --------
    Component : A density profile with its anisotropy.
    Anisotropy.factor : The factor :math:`g(R)` of the augmented density.

    Notes
    -----
    `Anisotropy` decides the model by exact comparison of the given
    parameters with ``beta_c == 0`` and with an infinite `r_a`. A value such
    as
    :math:`\beta_{\rm c} = 10^{-12}` gives a constant-anisotropy model and
    is not rounded to zero. The results of the package are continuous
    across :math:`\beta_{\rm c} = 0` and across an infinite
    :math:`R_{\rm a}`.

    References
    ----------
    Chiang+(2026d), Eq. (B44).

    Examples
    --------
    >>> from pianistpy import Anisotropy
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
        # The family is decided once, here, from the concrete parameter values,
        # and stored. No method compares beta_c or r_a again, so that a beta_c
        # or an r_a traced by JAX (as in a fit) never reaches a Python branch.
        if math.isinf(self.r_a):
            family = "isotropic" if self.beta_c == 0.0 else "constant"
            object.__setattr__(self, "kind", "I")
        else:
            family = ("OM" if self.beta_c == 0.0 else "COM") + "_" + self.kind
        object.__setattr__(self, "family", family)

    def factor(self, r: Any, xp: Any = np) -> Any:
        r"""Return the factor :math:`g(R)` of the augmented density.

        The factor :math:`g(R)` relates the augmented density to the
        auxiliary density, :math:`\varrho_{\rm aug} = g(R)\,\varrho_{\rm aux}`.
        The package uses the factor :math:`g(R)` in the inversion of the
        density, at the truncation edge and in the solution of the Poisson
        equation.

        Parameters
        ----------
        r : float or array-like
            User-specified 3D radius :math:`R`, in internal units (units of
            :math:`r_0`). `r` is a plain float or array, and an Astropy
            quantity is not accepted.
        xp : module, optional
            The array namespace, ``numpy`` or ``jax.numpy``. Default:
            ``numpy``, which means the factor is a NumPy array.

        Returns
        -------
        g : array
            The factor :math:`g(R)`, dimensionless, with the shape of `r`:
            :math:`1` for the isotropic model,
            :math:`R^{-2\beta_{\rm c}}` for the constant-anisotropy model,
            :math:`(1 \pm R^2/R_{\rm a}^2)^{-1}` for the OM model and
            :math:`R^{-2\beta_{\rm c}} (1 \pm R^2/R_{\rm a}^2)^{-(1-\beta_{\rm c})}`
            for the COM model, with the upper sign for Type I and the lower
            sign for Type II.

        Raises
        ------
        InputError
            If the model is Type II and a radius in `r` is equal to or larger
            than :math:`R_{\rm a}`, where the factor is not defined. Inside a
            JAX transformation such as ``jax.jit``, the radius is not
            checked.

        See Also
        --------
        Anisotropy.domain_edge : The largest radius where the factor is
            defined.

        References
        ----------
        Chiang+(2026d), Eq. (B44).

        Examples
        --------
        >>> from pianistpy import Anisotropy
        >>> float(Anisotropy(beta_c=0.5).factor(4.0))
        0.25
        >>> Anisotropy(r_a=1.0).factor([1.0]).tolist()
        [0.5]
        """
        r = xp.asarray(r, dtype=xp.float64)
        if self.family == "isotropic":
            return xp.ones_like(r)
        (h,) = inverse_factor(r, self.family, self.beta_c, self.r_a)
        return 1.0 / h

    @property
    def model(self) -> str:
        """Name of the model: ``"isotropic"``, ``"constant"``, ``"om"`` or ``"com"``.

        The names are those of Chiang+(2026d). The name does not include the
        type, which is in `kind`.

        See Also
        --------
        Anisotropy.to_dict : The dictionary form, which records the name.

        Examples
        --------
        >>> from pianistpy import Anisotropy
        >>> Anisotropy(beta_c=0.25, r_a=10.0).model
        'com'
        """
        return self.family.split("_")[0].lower()

    @property
    def domain_edge(self) -> float:
        r"""Largest radius of the anisotropy's domain, in units of :math:`r_0`.

        For a Type II model with a finite :math:`R_{\rm a}`, the domain edge
        is :math:`R_{\rm a}`. The factor :math:`g(R)` of a Type II model is
        real, finite and positive only at :math:`R < R_{\rm a}`. For every
        other model, the
        domain edge is infinite.

        See Also
        --------
        Anisotropy.factor : The factor :math:`g(R)`.

        Examples
        --------
        >>> from pianistpy import Anisotropy
        >>> Anisotropy(r_a=10.0, kind="II").domain_edge
        10.0
        >>> Anisotropy(r_a=10.0).domain_edge
        inf
        """
        if self.kind == "II" and math.isfinite(self.r_a):
            return float(self.r_a)
        return math.inf

    def to_dict(self) -> dict[str, Any]:
        """Return the anisotropy as a plain dictionary.

        The dictionary holds only JSON types, so that a model file can store
        the anisotropy as text.

        Returns
        -------
        record : dict
            The keys ``"model"`` (as `model`), ``"beta_c"``, ``"r_a"`` and
            ``"kind"``. The value of ``"r_a"`` is None for an infinite
            anisotropy radius.

        See Also
        --------
        Anisotropy.from_dict : The anisotropy built back from the dictionary.

        Examples
        --------
        >>> from pianistpy import Anisotropy
        >>> Anisotropy(beta_c=0.3).to_dict()
        {'model': 'constant', 'beta_c': 0.3, 'r_a': None, 'kind': 'I'}
        """
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
        """Return the anisotropy that a dictionary describes.

        Parameters
        ----------
        record : mapping
            The dictionary form of the anisotropy, as `to_dict` returns it.
            The key ``"model"`` is not read. A missing key takes the default
            of the matching parameter of `Anisotropy`, and an ``"r_a"`` of
            None means an infinite anisotropy radius.

        Returns
        -------
        anisotropy : Anisotropy
            The anisotropy that `record` describes.

        Raises
        ------
        InputError
            If ``"beta_c"`` is 1 or larger, ``"r_a"`` is zero or negative,
            or ``"kind"`` is neither ``"I"`` nor ``"II"``.

        See Also
        --------
        Anisotropy.to_dict : The dictionary form of the anisotropy.

        Examples
        --------
        >>> from pianistpy import Anisotropy
        >>> Anisotropy.from_dict({"beta_c": 0.3}) == Anisotropy(beta_c=0.3)
        True
        """
        r_a = record.get("r_a")
        return cls(
            beta_c=float(record.get("beta_c", 0.0)),
            r_a=math.inf if r_a is None else float(r_a),
            kind=str(record.get("kind", "I")),
        )


def _flatten_anisotropy(a: Anisotropy):
    # In the pytree form of Anisotropy, beta_c and r_a are the leaves that JAX
    # traces and differentiates. The kind and the family are static data.
    return (a.beta_c, a.r_a), (a.kind, a.family)


def _unflatten_anisotropy(aux, children) -> Anisotropy:
    """Rebuild an Anisotropy from its pytree form without running its checks.

    The checks of ``__post_init__`` would compare traced values in Python
    branches, so the fields are set directly.
    """
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
