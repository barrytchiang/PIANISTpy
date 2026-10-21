"""The inputs of a distribution-function computation: components and truncations.

`Component` holds a density profile with its velocity anisotropy, and
`Truncation` holds the truncation energy, target mass or target radius of
Method I. Both are frozen dataclasses registered as JAX pytrees.
"""

__all__ = ["Component", "Truncation"]

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.tree_util

from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..profiles.core import RHO_BETA_FINITE_MASS, RHO_BETA_MIN
from .edge import check_edge_existence


@dataclass(frozen=True)
class Component:
    r"""One component of a model: a density profile with its velocity anisotropy.

    A model is a sequence of components that share one total potential. A
    model of one component is a sequence of length one. `compute_df` and
    `solve` accept a model of one component.

    A component whose profile is the `TruncatedProfile` of another model
    (``parent.profile``) is Method II of Chiang+(2026d). The Method II
    component has the density of the parent model in the parent model's
    potential, with the component's own anisotropy.

    Parameters
    ----------
    profile : DoublePowerLaw or TruncatedProfile
        Density profile of the component. A `DoublePowerLaw` is an analytic
        profile and needs an outer slope :math:`\beta \ge 3`. A
        `TruncatedProfile` must be the profile of a model of one component.
    anisotropy : Anisotropy, optional
        Velocity anisotropy of the component. Default: ``Anisotropy()``,
        which means an isotropic velocity distribution.
    mass : float, optional
        Mass of the component in units of the mass of the first component.
        Must be positive. Default: ``1.0``, which means the mass of the
        first component.
    name : str, optional
        Name of the component, carried by the component's results and
        particles. Default: ``"component"``, which means the results and
        particles carry the name ``"component"``.

    Raises
    ------
    InputError
        If `mass` is not positive.
    InputError
        If the outer slope :math:`\beta` of the analytic profile lies above
        2 and below 3. The inversion assumes a relative potential
        :math:`P \propto 1/R` at large radius :math:`R`, which holds for
        :math:`\beta \ge 3` only.
    InputError
        If `profile` is the `TruncatedProfile` of a model of more than one
        component.
    UnphysicalModelError
        If `profile` is a `TruncatedProfile` with an edge and the central
        anisotropy :math:`\beta_{\rm c}` is not above :math:`1/2 - q`, with
        :math:`q` the exponent of the density at the edge,
        :math:`\varrho \propto \hat{P}^{\,q}`. No distribution function of
        the given `anisotropy` exists at the edge of such a profile.

    See Also
    --------
    Truncation : The truncation of a model's distribution function
        (Method I).
    solve : The equilibrium model of a component.
    Anisotropy : The velocity anisotropy of a component.

    References
    ----------
    Chiang+(2026d), Section 3.1 and Appendix B2.2.

    Examples
    --------
    >>> import pianistpy as pp
    >>> halo = pp.Component(
    ...     pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25), name="halo"
    ... )
    >>> halo.anisotropy.model, halo.mass, halo.name
    ('constant', 1.0, 'halo')
    """

    profile: DoublePowerLaw | TruncatedProfile
    anisotropy: Anisotropy = field(default_factory=Anisotropy)
    mass: float = 1.0
    name: str = "component"

    def __post_init__(self) -> None:
        if not self.mass > 0.0:
            raise InputError(f"mass must be positive; got {self.mass}.")
        beta = float(self.unit_profile.rho_beta)
        if RHO_BETA_MIN < beta < RHO_BETA_FINITE_MASS:
            raise InputError(
                f"OUTER SLOPE: rho_beta = {beta:g} lies between {RHO_BETA_MIN:g} and "
                f"{RHO_BETA_FINITE_MASS:g}, where PIANISTpy does not compute the "
                "DF: the outer end of the inversion and the tail exponent "
                "assume Psi ~ 1/R, which holds for rho_beta >= 3 only. Use rho_beta "
                ">= 3."
            )
        profile = self.profile
        if isinstance(profile, TruncatedProfile):
            n_components = profile.rho_trunc.shape[0]
            if n_components != 1:
                raise InputError(
                    "A component takes the truncated profile of a model of one "
                    f"component; this one has {n_components}."
                )
            if profile.spatially_finite:
                check_edge_existence(profile.edge_exponent[0], self.anisotropy.beta_c)

    @property
    def unit_profile(self) -> DoublePowerLaw:
        """Analytic profile that defines the internal units of the component."""
        profile = self.profile
        return profile.source if isinstance(profile, TruncatedProfile) else profile

    def to_dict(self) -> dict[str, Any]:
        """Return the component as a plain dictionary.

        Returns
        -------
        record : dict
            Dictionary with the keys ``"name"``, ``"mass"``, ``"profile"``
            (the dictionary form of the profile) and ``"anisotropy"`` (the
            dictionary form of the anisotropy). `Component.from_dict`
            builds the component back from the dictionary.

        See Also
        --------
        Component.from_dict : The component built from its dictionary.

        Examples
        --------
        >>> import pianistpy as pp
        >>> sorted(pp.Component(pp.DoublePowerLaw.nfw()).to_dict())
        ['anisotropy', 'mass', 'name', 'profile']
        """
        return {
            "name": self.name,
            "mass": self.mass,
            "profile": self.profile.to_dict(),
            "anisotropy": self.anisotropy.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, record: Mapping[str, Any], profile: TruncatedProfile | None = None
    ) -> "Component":
        """Return the component that a dictionary describes.

        Parameters
        ----------
        record : mapping
            Dictionary form of the component, as `Component.to_dict`
            returns it.
        profile : TruncatedProfile, optional
            The `TruncatedProfile` of the component, where `record` states a
            `TruncatedProfile`. The dictionary of a `TruncatedProfile` identifies
            the profile's model and does not hold the profile's arrays.
            Default: ``None``, which means that `record` states an analytic
            profile.

        Returns
        -------
        component : Component
            The component that `record` describes.

        Raises
        ------
        InputError
            If `record` states a `TruncatedProfile` and `profile` is not
            given or belongs to a model other than the one `record` identifies.
        InputError
            If the component that `record` describes is not admissible. The
            Raises section of `Component` lists the conditions.

        See Also
        --------
        Component.to_dict : The dictionary of a component.

        Examples
        --------
        >>> import pianistpy as pp
        >>> halo = pp.Component(pp.DoublePowerLaw.nfw(), name="halo")
        >>> pp.Component.from_dict(halo.to_dict()) == halo
        True
        """
        if record["profile"].get("profile") == "truncated":
            if profile is None or profile.model != record["profile"]["model"]:
                raise InputError(
                    "The component is built on the truncated profile of model "
                    f"{record['profile']['model']}; its arrays are needed to "
                    "rebuild it."
                )
            built: DoublePowerLaw | TruncatedProfile = profile
        else:
            built = DoublePowerLaw.from_dict(record["profile"])
        return cls(
            profile=built,
            anisotropy=Anisotropy.from_dict(record.get("anisotropy", {})),
            mass=float(record.get("mass", 1.0)),
            name=str(record.get("name", "component")),
        )


@dataclass(frozen=True)
class Truncation:
    r"""Truncation of a model's distribution function (Method I).

    A truncation shifts the distribution function (DF) in energy by the
    truncation energy :math:`\mathcal{K}_{\rm t}` and lowers the DF by the
    DF's value at :math:`\mathcal{K}_{\rm t}`, which gives a model of finite
    mass and finite radius (Method I of Chiang+(2026d)). The truncation is given
    by exactly one of three inputs: the truncation energy `k_t`, a target
    mass `mass_t` or a target radius `r_t`. A target mass is given together
    with the `concentration` that defines the unit of the target mass.
    Method II needs no `Truncation`: a Method II model is a `Component`
    whose profile is the `TruncatedProfile` of another model.

    Parameters
    ----------
    k_t : float, optional
        Truncation energy :math:`\mathcal{K}_{\rm t}`, positive, in internal
        units, in which the central relative potential of the untruncated
        profile is 1. Default: ``None``, which means that `mass_t` or `r_t`
        is given.
    mass_t : float, optional
        Target total mass of the truncated model, positive, in units of
        :math:`M_0`, the mass of the untruncated profile inside the radius
        :math:`c\,r_0` (model units). A value of 1 is the virial mass.
        `mass_t` is the one input of `Truncation` that is not in internal
        units. The truncation energy is solved for. `mass_t` requires
        `concentration`. Default: ``None``, which means that `k_t` or `r_t`
        is given.
    r_t : float, optional
        Target truncation radius :math:`R_{\rm t}`, positive, in units of
        the scale radius :math:`r_0`, the unit of length of the internal
        and the model systems. The truncation energy is solved for.
        Default: ``None``, which means that `k_t` or `mass_t` is given.
    concentration : float, optional
        Concentration :math:`c` of the first component, positive. The mass
        unit :math:`M_0` of `mass_t` is the mass of the untruncated profile
        inside :math:`c\,r_0`. `concentration` is given with `mass_t` only,
        since `k_t` and `r_t` do not depend on the concentration. Default:
        ``None``, which means that no target mass is given.

    Raises
    ------
    InputError
        If not exactly one of `k_t`, `mass_t` and `r_t` is given, or if the
        one given is not positive.
    InputError
        If `mass_t` is given without `concentration`, if `concentration` is
        given without `mass_t`, or if `concentration` is not positive.

    See Also
    --------
    truncate : The model that a truncation gives.
    solve : The model of a component, with or without a truncation.
    Component : A component on the `TruncatedProfile` of another model
        (Method II).

    Notes
    -----
    The truncated DF is
    :math:`\hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
    F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
    - F(\mathcal{K}_{\rm t}, \mathcal{L})` for
    :math:`\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0`, and
    :math:`\hat{F} = 0` for :math:`\hat{\mathcal{K}} < 0`, with :math:`F`
    the untruncated DF, :math:`\mathcal{K}` the energy-like variable and
    :math:`\mathcal{L}` the angular momentum.

    References
    ----------
    Chiang+(2026d), Eq. (9). Notes Section T:3.6.

    Examples
    --------
    >>> import pianistpy as pp
    >>> pp.Truncation(mass_t=1.0, concentration=10.0).to_dict()
    {'k_t': None, 'mass_t': 1.0, 'r_t': None, 'concentration': 10.0}
    """

    k_t: float | None = None
    mass_t: float | None = None
    r_t: float | None = None
    concentration: float | None = None

    def __post_init__(self) -> None:
        given = [
            name for name in ("k_t", "mass_t", "r_t") if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise InputError(
                f"Exactly one of k_t, mass_t and r_t must be given; got {given}."
            )
        value = getattr(self, given[0])
        if not value > 0.0:
            raise InputError(f"{given[0]} must be positive; got {value}.")
        if self.mass_t is not None and self.concentration is None:
            raise InputError(
                "mass_t is in units of M_0, the mass inside the concentration "
                "radius; give the concentration with it."
            )
        if self.mass_t is None and self.concentration is not None:
            raise InputError(
                f"concentration is used only with mass_t; {given[0]} does not "
                "depend on it."
            )
        if self.concentration is not None and not self.concentration > 0.0:
            raise InputError(
                f"concentration must be positive; got {self.concentration}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the truncation as a plain dictionary.

        Returns
        -------
        record : dict
            Dictionary with the keys ``"k_t"``, ``"mass_t"``, ``"r_t"`` and
            ``"concentration"``, each None where the field is not given.
            `Truncation.from_dict` builds the truncation back from the
            dictionary.

        See Also
        --------
        Truncation.from_dict : The truncation built from its dictionary.

        Examples
        --------
        >>> import pianistpy as pp
        >>> pp.Truncation(k_t=0.05).to_dict()
        {'k_t': 0.05, 'mass_t': None, 'r_t': None, 'concentration': None}
        """
        return {
            "k_t": self.k_t,
            "mass_t": self.mass_t,
            "r_t": self.r_t,
            "concentration": self.concentration,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Truncation":
        """Return the truncation that a dictionary describes.

        Parameters
        ----------
        record : mapping
            Dictionary form of the truncation, as `Truncation.to_dict`
            returns it. A key missing from `record` means None for that
            field.

        Returns
        -------
        truncation : Truncation
            The truncation that `record` describes.

        Raises
        ------
        InputError
            If the fields of `record` do not make an admissible truncation.
            The Raises section of `Truncation` lists the conditions.

        See Also
        --------
        Truncation.to_dict : The dictionary of a truncation.

        Examples
        --------
        >>> import pianistpy as pp
        >>> pp.Truncation.from_dict({"r_t": 20.0})
        Truncation(k_t=None, mass_t=None, r_t=20.0, concentration=None)
        """
        return cls(
            k_t=record.get("k_t"),
            mass_t=record.get("mass_t"),
            r_t=record.get("r_t"),
            concentration=record.get("concentration"),
        )


def _flatten_component(c: Component):
    # The anisotropy (itself a pytree) and the mass are the pytree's
    # children, so that JAX traces and differentiates a function of a
    # Component in them. The profile and the name are static auxiliary data,
    # whether the profile is a DoublePowerLaw or a TruncatedProfile.
    return (c.anisotropy, c.mass), (c.profile, c.name)


def _unflatten_component(aux, children) -> Component:
    """Rebuild a Component from its pytree parts, without `__post_init__`.

    The children may be JAX tracers, which the checks of `__post_init__`
    cannot compare, so the fields are set directly.
    """
    profile, name = aux
    anisotropy, mass = children
    c = object.__new__(Component)
    object.__setattr__(c, "profile", profile)
    object.__setattr__(c, "anisotropy", anisotropy)
    object.__setattr__(c, "mass", mass)
    object.__setattr__(c, "name", name)
    return c


def _flatten_truncation(t: Truncation):
    # The four fields are the pytree's children; None fields are empty
    # subtrees in JAX.
    return (t.k_t, t.mass_t, t.r_t, t.concentration), ()


def _unflatten_truncation(aux, children) -> Truncation:
    """Rebuild a Truncation from its pytree parts, without `__post_init__`.

    The children may be JAX tracers, which the checks of `__post_init__`
    cannot compare, so the fields are set directly.
    """
    k_t, mass_t, r_t, concentration = children
    t = object.__new__(Truncation)
    object.__setattr__(t, "k_t", k_t)
    object.__setattr__(t, "mass_t", mass_t)
    object.__setattr__(t, "r_t", r_t)
    object.__setattr__(t, "concentration", concentration)
    return t


jax.tree_util.register_pytree_node(Component, _flatten_component, _unflatten_component)
jax.tree_util.register_pytree_node(
    Truncation, _flatten_truncation, _unflatten_truncation
)
