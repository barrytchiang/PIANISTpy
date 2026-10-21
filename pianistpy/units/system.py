r"""The unit system in which a result presents the result's quantities.

A user selects the unit system of a result with the result's ``to`` method
(`Model.to`), which builds a `Units` object with `Units.build`. Every
computation of the package runs in internal units, in which
:math:`G = \rho_0/C = r_0 = 1`; the `Units` object of a result only decides
how the result hands the result's quantities over. A quantity computed from a result
carries the result's `Units` object. The three unit systems are the internal
units, the model units (:math:`G = M_0 = r_0 = 1`, with :math:`M_0` the mass
inside the radius :math:`c\,r_0` and :math:`c` the concentration), and the
physical units (Astropy quantities in parsecs, kilometers per second and
solar masses). `unit_factors` returns the conversion factors.
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
    """Return a positive scale as a float in `unit`, from a float or a quantity.

    A float is taken to be in `unit` already. None is returned as None. A
    scale that is not positive raises `InputError`, whose message gives the
    scale's name, `name`.
    """
    if value is None:
        return None
    if hasattr(value, "to_value"):
        import astropy.units as u

        value = value.to_value(u.Unit(unit))
    value = float(value)
    if not value > 0.0:
        raise InputError(f"{name} must be positive; the value given is {value}.")
    return value


@dataclass(frozen=True)
class Units:
    r"""The unit system of a result and the scales that define the unit system.

    A result holds the result's numbers in internal units and hands the
    numbers over in the unit system that the result's `Units` object sets.
    `Units.build` builds a `Units` object from scales in any unit; the
    constructor takes the scales in solar masses and parsecs.

    Parameters
    ----------
    system : {"internal", "model", "physical"}, optional
        The unit system. ``"internal"``: :math:`G = \rho_0/C = r_0 = 1`, with
        :math:`\rho_0` the scale density and :math:`C` the normalization of
        the density profile. ``"model"``: :math:`G = M_0 = r_0 = 1`, with
        :math:`M_0` the mass inside the radius :math:`c\,r_0`.
        ``"physical"``: Astropy quantities in parsecs, kilometers per second,
        solar masses, solar masses per cubic parsec and
        :math:`{\rm km^2\,s^{-2}}`. Default: ``"internal"``, which means the
        result hands the result's quantities over in internal units.
    concentration : float, optional
        The concentration :math:`c`, which defines :math:`M_0` as the mass
        inside the radius :math:`c\,r_0`. Required by the model units and the
        physical units. Default: ``None``, which means no concentration, as
        the internal units require.
    mass_0_msun : float, optional
        The mass :math:`M_0` in solar masses. Required by the physical units.
        Default: ``None``, which means no physical mass scale.
    r_0_pc : float, optional
        The scale radius :math:`r_0` in parsecs. Required by the physical
        units. Default: ``None``, which means no physical length scale.

    Raises
    ------
    InputError
        If `system` is not ``"internal"``, ``"model"`` or ``"physical"``.
    InputError
        If `system` is ``"model"`` or ``"physical"`` and `concentration` is
        None.
    InputError
        If `system` is ``"physical"`` and `mass_0_msun` or `r_0_pc` is None.

    See Also
    --------
    Units.build : The units of a unit system from scales in any unit.
    unit_factors : The conversion factors out of internal units.
    UnitFactors : The conversion factors and their definitions.

    Examples
    --------
    >>> from pianistpy.units import Units
    >>> Units().system
    'internal'
    >>> Units.build("model", concentration=10.0).concentration
    10.0
    >>> units = Units.build("physical", concentration=10.0, mass_0=1e12, r_vir=2e5)
    >>> units.r_0_pc
    20000.0
    """

    system: str = "internal"
    concentration: float | None = None
    mass_0_msun: float | None = None
    r_0_pc: float | None = None

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise InputError(
                f"The unit system must be one of {SYSTEMS}; the value given is "
                f"{self.system!r}."
            )
        if self.system != "internal" and self.concentration is None:
            raise InputError(
                f"The {self.system} unit system needs the concentration that "
                "defines M_0; no concentration is given."
            )
        if self.system == "physical" and None in (self.mass_0_msun, self.r_0_pc):
            raise InputError(
                "The physical unit system needs mass_0 and r_0 (or r_vir in place of "
                "r_0); at least one of the two scales is not given."
            )

    @classmethod
    def build(
        cls,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "Units":
        r"""Return the units of a unit system from user-specified scales in any unit.

        The internal units ignore every scale. The model units keep the
        concentration and ignore the physical scales. The physical units keep
        the concentration, :math:`M_0` in solar masses and :math:`r_0` in
        parsecs.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            The unit system of the returned units.
        concentration : float, optional
            The concentration :math:`c`, which defines :math:`M_0` as the mass
            inside the radius :math:`c\,r_0`. Positive. Required by the model
            units and the physical units. Default: ``None``, which means no
            concentration.
        mass_0 : float or astropy Quantity, optional
            The mass :math:`M_0`. A float is in solar masses; a quantity of any
            mass unit is converted to solar masses. Positive. Required by the
            physical units. Default: ``None``, which means no mass scale.
        r_0 : float or astropy Quantity, optional
            The scale radius :math:`r_0`. A float is in parsecs; a quantity of
            any length unit is converted to parsecs. Positive. Default:
            ``None``, which means the scale radius is taken from `r_vir`.
        r_vir : float or astropy Quantity, optional
            The radius :math:`c\,r_0`, given in place of `r_0`. A float is in
            parsecs; a quantity of any length unit is converted to parsecs.
            Positive. Default: ``None``, which means the scale radius is
            taken from `r_0`.

        Returns
        -------
        units : Units
            The units of `system`, with :math:`M_0` in solar masses and
            :math:`r_0` in parsecs.

        Raises
        ------
        InputError
            If both `r_0` and `r_vir` are given.
        InputError
            If `concentration`, `r_0`, `r_vir` or, for the physical units,
            `mass_0` is given and is not positive.
        InputError
            If `r_vir` is given and `concentration` is not.
        InputError
            If `system` is not ``"internal"``, ``"model"`` or ``"physical"``.
        InputError
            If `system` is ``"model"`` or ``"physical"`` and `concentration`
            is None.
        InputError
            If `system` is ``"physical"`` and `mass_0` is None, or neither
            `r_0` nor `r_vir` is given.

        See Also
        --------
        Model.to : The model presented in the unit system that a `Units` object sets.

        Examples
        --------
        >>> from pianistpy.units import Units
        >>> Units.build("internal", concentration=10.0)
        Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None)
        >>> Units.build("physical", concentration=10.0, mass_0=1e12, r_0=2e4).r_0_pc
        20000.0
        """
        if system == "internal":
            return cls()
        if r_0 is not None and r_vir is not None:
            raise InputError(
                "Units.build takes r_0 or r_vir, not both; both are given."
            )
        concentration = _plain(concentration, "", "concentration")
        r_0_pc = _plain(r_0, "pc", "r_0")
        if r_vir is not None:
            if concentration is None:
                raise InputError(
                    "r_vir needs the concentration c, because r_vir = c r_0; no "
                    "concentration is given."
                )
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
        """Return the conversion factors of the units' scales for a density profile.

        Parameters
        ----------
        profile : DoublePowerLaw
            The density profile, which fixes the normalization :math:`C` of
            the density.

        Returns
        -------
        factors : UnitFactors
            The factors `unit_factors` returns for the profile, the units'
            concentration, and the units' `mass_0_msun` and `r_0_pc`.

        Raises
        ------
        InputError
            If the units' concentration is None, as in the internal units.
        InputError
            If the units' concentration is not positive.
        InputError
            If one of `mass_0_msun` and `r_0_pc` is set and the other is None.

        See Also
        --------
        unit_factors : The function that builds the factors.

        Examples
        --------
        >>> from pianistpy import DoublePowerLaw
        >>> from pianistpy.units import Units
        >>> units = Units.build("model", concentration=10.0)
        >>> round(units.factors(DoublePowerLaw.nfw()).model_mass, 6)
        0.67168
        """
        if self.concentration is None:
            raise InputError(
                "The internal unit system has no conversion factors, because the "
                "internal unit system has no concentration."
            )
        return unit_factors(profile, self.concentration, self.mass_0_msun, self.r_0_pc)

    def convert(self, kind: str, value: ArrayLike, profile: DoublePowerLaw) -> Any:
        """Return a quantity given in internal units in the unit system of the units.

        Parameters
        ----------
        kind : {"length", "velocity", "mass", "density", "potential"}
            The physical dimension of the quantity.
        value : array-like
            The quantity, in internal units.
        profile : DoublePowerLaw
            The density profile, which fixes the normalization :math:`C` of
            the density.

        Returns
        -------
        converted : array or astropy Quantity
            The quantity, with the shape of `value`. A float64 array in the
            internal units and in the model units; an Astropy quantity in the
            physical units. A length has the same value in internal units and
            in model units, because both unit systems measure a length in
            units of :math:`r_0`.

        Raises
        ------
        InputError
            If `kind` is not ``"length"``, ``"velocity"``, ``"mass"``,
            ``"density"`` or ``"potential"``.
        InputError
            If the unit system is not the internal one and `factors` raises
            `InputError` for the units' scales.

        See Also
        --------
        Units.factors : The conversion factors the conversion multiplies by.

        Examples
        --------
        >>> from pianistpy import DoublePowerLaw
        >>> from pianistpy.units import Units
        >>> units = Units.build("model", concentration=10.0)
        >>> units.convert("length", [1.0, 2.0], DoublePowerLaw.nfw())
        array([1., 2.])
        >>> round(float(units.convert("mass", 1.0, DoublePowerLaw.nfw())), 6)
        0.67168
        """
        if kind not in KINDS:
            raise InputError(
                f"The kind of quantity must be one of {KINDS}; the value given is "
                f"{kind!r}."
            )
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
        """Return the units as a dictionary of plain values.

        Returns
        -------
        record : dict
            The keys ``system``, ``concentration``, ``mass_0_msun`` and
            ``r_0_pc``, with the values of the units' fields, in the form
            `from_dict` reads.

        See Also
        --------
        Units.from_dict : The units rebuilt from the dictionary.

        Examples
        --------
        >>> from pianistpy.units import Units
        >>> record = Units.build("model", concentration=10.0).to_dict()
        >>> record["system"], record["concentration"], record["r_0_pc"]
        ('model', 10.0, None)
        """
        return {
            "system": self.system,
            "concentration": self.concentration,
            "mass_0_msun": self.mass_0_msun,
            "r_0_pc": self.r_0_pc,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any] | None) -> "Units":
        """Return the units that a dictionary written by `to_dict` records.

        Parameters
        ----------
        record : mapping or None
            The dictionary `to_dict` writes. None or an empty mapping stands
            for the internal units.

        Returns
        -------
        units : Units
            The units the dictionary records.

        Raises
        ------
        InputError
            If the recorded unit system and scales raise `InputError` in the
            `Units` constructor.

        See Also
        --------
        Units.to_dict : The dictionary of the units.

        Examples
        --------
        >>> from pianistpy.units import Units
        >>> units = Units.build("model", concentration=10.0)
        >>> Units.from_dict(units.to_dict()) == units
        True
        >>> Units.from_dict(None).system
        'internal'
        """
        if not record:
            return cls()
        return cls(
            system=str(record["system"]),
            concentration=record.get("concentration"),
            mass_0_msun=record.get("mass_0_msun"),
            r_0_pc=record.get("r_0_pc"),
        )
