"""The unit system a result is presented in.

Every computation runs in internal units. A result carries a `Units` object
that says in which system its quantities are handed to the user, and what is
computed from a result inherits that object.
"""

__all__ = ["KINDS", "SYSTEMS", "Units"]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import InputError
from ..profiles import DoublePowerLaw
from ..typing import ArrayLike
from .core import UnitFactors, unit_factors

SYSTEMS = ("internal", "model", "physical")
KINDS = ("length", "velocity", "mass", "density", "potential")

_MODEL_FACTOR = {
    "velocity": "model_v",
    "mass": "model_mass",
    "density": "model_rho",
    "potential": "model_p",
}
_PHYSICAL_FACTOR = {
    "length": ("physical_r_pc", "pc"),
    "velocity": ("physical_v_kms", "km / s"),
    "mass": ("physical_mass_msun", "Msun"),
    "density": ("physical_rho_msun_pc3", "Msun / pc3"),
    "potential": ("physical_p_kms2", "km2 / s2"),
}


def _plain(value: Any, unit: str, name: str) -> float | None:
    """Return a scale as a float in `unit`, from a float or an Astropy quantity."""
    if value is None:
        return None
    if hasattr(value, "to_value"):
        import astropy.units as u

        value = value.to_value(u.Unit(unit))
    value = float(value)
    if not value > 0.0:
        raise InputError(f"{name} must be positive; got {value}.")
    return value


@dataclass(frozen=True)
class Units:
    r"""The unit system of a result and the scales that define it.

    Parameters
    ----------
    system : {"internal", "model", "physical"}
        ``"internal"``: :math:`G = \rho_0/C = r_0 = 1`. ``"model"``:
        :math:`G = M_0 = r_0 = 1`. ``"physical"``: Astropy quantities.
    concentration : float, optional
        Concentration :math:`c` that defines :math:`M_0` as the mass inside
        :math:`c\,r_0`. Needed by the model and the physical systems.
    mass_0_msun : float, optional
        :math:`M_0` in solar masses. Needed by the physical system.
    r_0_pc : float, optional
        Scale radius :math:`r_0` in parsecs. Needed by the physical system.

    Examples
    --------
    >>> Units().system
    'internal'
    >>> Units.build("model", concentration=10.0).concentration
    10.0
    """

    system: str = "internal"
    concentration: float | None = None
    mass_0_msun: float | None = None
    r_0_pc: float | None = None

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise InputError(f"system must be one of {SYSTEMS}; got {self.system!r}.")
        if self.system != "internal" and self.concentration is None:
            raise InputError(
                f"The {self.system} units need the concentration that defines M_0."
            )
        if self.system == "physical" and None in (self.mass_0_msun, self.r_0_pc):
            raise InputError("The physical units need mass_0 and r_0 (or r_vir).")

    @classmethod
    def build(
        cls,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "Units":
        r"""Return the units of a system from the scales a user gives.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            The target system.
        concentration : float, optional
            Concentration that defines :math:`M_0`.
        mass_0 : float or astropy Quantity, optional
            :math:`M_0`; a float is in solar masses.
        r_0, r_vir : float or astropy Quantity, optional
            The scale radius, or the radius :math:`c\,r_0` in its place; a
            float is in parsecs. One of the two for the physical system.

        Returns
        -------
        units : Units
        """
        if system == "internal":
            return cls()
        if r_0 is not None and r_vir is not None:
            raise InputError("Give r_0 or r_vir, not both.")
        concentration = _plain(concentration, "", "concentration")
        r_0_pc = _plain(r_0, "pc", "r_0")
        if r_vir is not None:
            if concentration is None:
                raise InputError("r_vir needs the concentration: r_vir = c r_0.")
            r_0_pc = _plain(r_vir, "pc", "r_vir")
            assert r_0_pc is not None
            r_0_pc = r_0_pc / concentration
        if system == "model":
            return cls(system="model", concentration=concentration)
        return cls(
            system=system,
            concentration=concentration,
            mass_0_msun=_plain(mass_0, "Msun", "mass_0"),
            r_0_pc=r_0_pc,
        )

    def factors(self, profile: DoublePowerLaw) -> UnitFactors:
        """Return the conversion factors of these units for a density profile."""
        if self.concentration is None:
            raise InputError("Internal units have no conversion factors.")
        return unit_factors(profile, self.concentration, self.mass_0_msun, self.r_0_pc)

    def convert(self, kind: str, value: ArrayLike, profile: DoublePowerLaw) -> Any:
        """Return a quantity in internal units in the units of this object.

        Parameters
        ----------
        kind : {"length", "velocity", "mass", "density", "potential"}
            What the quantity is.
        value : array-like
            The quantity, in internal units.
        profile : DoublePowerLaw
            The density profile that fixes the normalization.

        Returns
        -------
        converted : array or astropy Quantity
            A plain array for the internal and the model systems, an Astropy
            quantity for the physical system.
        """
        if kind not in KINDS:
            raise InputError(f"kind must be one of {KINDS}; got {kind!r}.")
        value = np.asarray(value, dtype=np.float64)
        if self.system == "internal":
            return value
        factors = self.factors(profile)
        if self.system == "model":
            name = _MODEL_FACTOR.get(kind)
            return value if name is None else value * getattr(factors, name)
        import astropy.units as u

        name, unit = _PHYSICAL_FACTOR[kind]
        return value * getattr(factors, name) * u.Unit(unit)

    def to_dict(self) -> dict[str, Any]:
        """Return the units as a plain dictionary."""
        return {
            "system": self.system,
            "concentration": self.concentration,
            "mass_0_msun": self.mass_0_msun,
            "r_0_pc": self.r_0_pc,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any] | None) -> "Units":
        """Build the units from their dictionary form; internal if absent."""
        if not record:
            return cls()
        return cls(
            system=str(record["system"]),
            concentration=record.get("concentration"),
            mass_0_msun=record.get("mass_0_msun"),
            r_0_pc=record.get("r_0_pc"),
        )
