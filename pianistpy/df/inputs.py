"""Inputs of the distribution-function computation."""

__all__ = ["Component", "Truncation"]

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.tree_util

from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..profiles import DoublePowerLaw, TruncatedProfile
from .edge import check_edge_existence


@dataclass(frozen=True)
class Component:
    """One component of a model: a density profile with its anisotropy.

    A model is a sequence of components that share one total potential. A
    single component is a sequence of length one.

    Given the truncated profile of another model (``parent.profile``) in
    place of an analytic profile, the component is Method II of the paper:
    the density of the parent in the parent's potential, with a new
    anisotropy. It is refused here where no distribution function exists
    for that anisotropy at the edge of the density.

    Parameters
    ----------
    profile : DoublePowerLaw or TruncatedProfile
        Density profile of the component: an analytic profile, or the
        truncated profile of a model of one component.
    anisotropy : Anisotropy, optional
        Velocity anisotropy of the component. Isotropic by default.
    mass : float, optional
        Mass of the component relative to the first component.
    name : str, optional
        Label carried by the particles and the results of the component.
    """

    profile: DoublePowerLaw | TruncatedProfile
    anisotropy: Anisotropy = field(default_factory=Anisotropy)
    mass: float = 1.0
    name: str = "component"

    def __post_init__(self) -> None:
        if not self.mass > 0.0:
            raise InputError(f"mass must be positive; got {self.mass}.")
        profile = self.profile
        if isinstance(profile, TruncatedProfile):
            n_components = profile.rho_trunc.shape[0]
            if n_components != 1:
                raise InputError(
                    "A component takes the truncated profile of a model of one "
                    f"component; this one has {n_components}. Several components "
                    "in one potential are not available yet."
                )
            if profile.spatially_finite:
                check_edge_existence(profile.edge_exponent[0], self.anisotropy.beta_c)

    @property
    def unit_profile(self) -> DoublePowerLaw:
        """The analytic profile that fixes the internal units of the component."""
        profile = self.profile
        return profile.source if isinstance(profile, TruncatedProfile) else profile

    def to_dict(self) -> dict[str, Any]:
        """Return the component as a plain dictionary."""
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
        """Build the component from its dictionary form.

        Parameters
        ----------
        record : mapping
            The dictionary form.
        profile : TruncatedProfile, optional
            The profile, where the record states a truncated one: the
            record identifies it and does not hold its arrays.
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
    r"""How the distribution function of a model is truncated.

    This is Method I of the paper. Method II needs no truncation: it is a
    component whose profile is the truncated profile of another model, given
    a new anisotropy.

    Exactly one of `k_t`, `mass_t` and `r_t` is given. A target mass is given
    together with the `concentration` that defines its unit.

    Parameters
    ----------
    k_t : float, optional
        Truncation energy :math:`\mathcal{K}_{\rm t}`, in internal units (the
        central potential of the untruncated profile is 1).
    mass_t : float, optional
        Target total mass of the truncated model, in MODEL units: in units of
        :math:`M_0`, the mass of the untruncated profile inside the
        concentration radius, so that a value of 1 is the virial mass. This
        is the one input that is not in internal units. The truncation energy
        is solved for. Requires `concentration`.
    r_t : float, optional
        Target truncation radius :math:`R_{\rm t}`, in internal units (the
        scale radius :math:`r_0` is their unit of length). The truncation
        energy is solved for.
    concentration : float, optional
        Concentration :math:`c` of the first component: :math:`M_0` is the
        mass of its untruncated profile inside :math:`c\,r_0`. Required
        with `mass_t`, and refused with `k_t` and `r_t`, which do not depend
        on it.
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
        """Return the truncation as a plain dictionary."""
        return {
            "k_t": self.k_t,
            "mass_t": self.mass_t,
            "r_t": self.r_t,
            "concentration": self.concentration,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Truncation":
        """Build the truncation from its dictionary form."""
        return cls(
            k_t=record.get("k_t"),
            mass_t=record.get("mass_t"),
            r_t=record.get("r_t"),
            concentration=record.get("concentration"),
        )


def _flatten_component(c: Component):
    # The anisotropy (a pytree) and the mass are traced; the profile and the
    # name are configuration. A truncated profile is held as static data
    # until it is registered itself.
    return (c.anisotropy, c.mass), (c.profile, c.name)


def _unflatten_component(aux, children) -> Component:
    profile, name = aux
    anisotropy, mass = children
    c = object.__new__(Component)
    object.__setattr__(c, "profile", profile)
    object.__setattr__(c, "anisotropy", anisotropy)
    object.__setattr__(c, "mass", mass)
    object.__setattr__(c, "name", name)
    return c


def _flatten_truncation(t: Truncation):
    return (t.k_t, t.mass_t, t.r_t, t.concentration), ()


def _unflatten_truncation(aux, children) -> Truncation:
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
