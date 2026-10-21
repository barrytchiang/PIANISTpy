r"""Conversion factors from internal units to model units and to physical units.

A user calls `unit_factors` to obtain the `UnitFactors` of a density profile;
`Units.factors` calls `unit_factors` for the scales a result carries. Every
computation of the package runs in internal units, in which
:math:`G = \rho_{\rm w} = r_0 = 1` with :math:`\rho_{\rm w} = \rho_0/C`; the
central potential of the untruncated profile is 1 in internal units. Here
:math:`G` is the gravitational constant, :math:`\rho_0` the scale density,
:math:`r_0` the scale radius and :math:`C` the normalization of the density
profile (`DoublePowerLaw.rho_norm`). The model units have
:math:`G = M_0 = r_0 = 1`, with :math:`M_0` the mass inside the radius
:math:`c\,r_0` and :math:`c` the concentration. A quantity in internal units is
multiplied by the factor of the quantity's dimension to obtain the quantity in
the target units:

.. math::

    v_{\rm model} = (C q)^{-1/2}\, v, \qquad
    M_{\rm model} = (C q)^{-1}\, M, \qquad
    \Phi_{\rm model} = (C q)^{-1}\, \Phi, \qquad
    \rho_{\rm model} = (4\pi C q)^{-1}\, \rho,

with :math:`q = \mathcal{M}(c)/C`, the mass inside the radius :math:`c\,r_0` in
units of :math:`4\pi\rho_0 r_0^3`. A length has the same value in internal
units and in model units, because both unit systems measure a length in units
of :math:`r_0`. The definitions follow Chiang+(2026d), Section 2.1 and Eq. (6).
"""

__all__ = ["UnitFactors", "unit_factors"]

import math
from dataclasses import dataclass

import numpy as np

from ..exceptions import InputError
from ..profiles import DoublePowerLaw


@dataclass(frozen=True)
class UnitFactors:
    r"""Factors that convert quantities from internal units to model and physical units.

    A quantity in internal units is multiplied by the factor of the
    quantity's dimension to obtain the quantity in the target units. The
    model units have :math:`G = M_0 = r_0 = 1`, with :math:`M_0` the mass
    inside the radius :math:`c\,r_0` and :math:`c` the concentration given to
    `unit_factors`. A length has no model factor, because internal units and
    model units both measure a length in units of :math:`r_0`.

    Parameters
    ----------
    rho_norm : float
        Normalization :math:`C` of the density profile
        (`DoublePowerLaw.rho_norm`).
    mass_q : float
        The number :math:`q = \mathcal{M}(c)/C`: the mass inside the radius
        :math:`c\,r_0` in units of :math:`4\pi\rho_0 r_0^3`.
    model_v : float
        Factor that converts a velocity to model units, :math:`(C q)^{-1/2}`.
    model_mass : float
        Factor that converts a mass to model units, :math:`(C q)^{-1}`.
    model_rho : float
        Factor that converts a density to model units,
        :math:`(4\pi C q)^{-1}`.
    model_p : float
        Factor that converts a potential to model units, :math:`(C q)^{-1}`.
    physical_r_pc : float or None, optional
        Factor that converts a length to parsecs, equal to :math:`r_0` in
        parsecs. Default: ``None``, which means `unit_factors` received no
        physical scales.
    physical_v_kms : float or None, optional
        Factor that converts a velocity to kilometers per second. Default:
        ``None``, which means `unit_factors` received no physical scales.
    physical_mass_msun : float or None, optional
        Factor that converts a mass to solar masses. Default: ``None``, which
        means `unit_factors` received no physical scales.
    physical_rho_msun_pc3 : float or None, optional
        Factor that converts a density to solar masses per cubic parsec.
        Default: ``None``, which means `unit_factors` received no physical
        scales.
    physical_p_kms2 : float or None, optional
        Factor that converts a potential to the square of kilometers per
        second. Default: ``None``, which means `unit_factors` received no
        physical scales.
    physical_rho_0_msun_pc3 : float or None, optional
        The scale density :math:`\rho_0` in solar masses per cubic parsec.
        Default: ``None``, which means `unit_factors` received no physical
        scales.

    See Also
    --------
    unit_factors : The function that builds the factors of a density profile.
    Units : The unit system a result carries.

    Notes
    -----
    Every physical factor is None unless `unit_factors` received both
    physical scales, :math:`M_0` in solar masses and :math:`r_0` in parsecs.

    Examples
    --------
    >>> from pianistpy import DoublePowerLaw
    >>> from pianistpy.units import unit_factors
    >>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
    >>> round(factors.model_mass, 6)
    0.67168
    >>> factors.physical_r_pc is None
    True
    """

    rho_norm: float
    mass_q: float
    model_v: float
    model_mass: float
    model_rho: float
    model_p: float
    physical_r_pc: float | None = None
    physical_v_kms: float | None = None
    physical_mass_msun: float | None = None
    physical_rho_msun_pc3: float | None = None
    physical_p_kms2: float | None = None
    physical_rho_0_msun_pc3: float | None = None


def unit_factors(
    profile: DoublePowerLaw,
    concentration: float,
    mass_0_msun: float | None = None,
    r_0_pc: float | None = None,
) -> UnitFactors:
    r"""Return the factors that convert a profile's quantities out of internal units.

    The model-unit factors depend on the density profile and the
    concentration :math:`c` alone. The physical factors also need the two
    physical scales, :math:`M_0` in solar masses and :math:`r_0` in parsecs.

    Parameters
    ----------
    profile : DoublePowerLaw
        The density profile. The profile fixes the normalization :math:`C`
        and the mass :math:`\mathcal{M}(c)` inside the radius :math:`c\,r_0`.
    concentration : float
        The concentration :math:`c`: the radius, in units of the scale radius
        :math:`r_0`, inside which the mass :math:`M_0` is defined. Positive.
    mass_0_msun : float, optional
        The mass :math:`M_0` in solar masses. Positive. Given together with
        `r_0_pc` or not at all. Default: ``None``, which means only the
        model-unit factors are formed.
    r_0_pc : float, optional
        The scale radius :math:`r_0` in parsecs. Positive. Given together with
        `mass_0_msun` or not at all. Default: ``None``, which means only the
        model-unit factors are formed.

    Returns
    -------
    factors : UnitFactors
        The model-unit factors, and the physical factors when both
        `mass_0_msun` and `r_0_pc` are given. Every physical factor is None
        when neither `mass_0_msun` nor `r_0_pc` is given.

    Raises
    ------
    InputError
        If `concentration` is not positive.
    InputError
        If one of `mass_0_msun` and `r_0_pc` is given and the other is not.

    See Also
    --------
    UnitFactors : The factors and their definitions.
    Units.factors : The factors for the scales a result carries.

    Notes
    -----
    With :math:`\rho_{\rm w} = \rho_0/C`, the internal unit of mass is
    :math:`M_{\rm W} = 4\pi\rho_{\rm w} r_0^3`, the internal unit of velocity is
    :math:`v_{\rm W} = (4\pi G\rho_{\rm w} r_0^2)^{1/2}`, and the mass inside
    the radius :math:`c\,r_0` is :math:`M_0 = 4\pi\rho_0 r_0^3\,q`. A velocity in
    internal units is therefore multiplied by :math:`(C q)^{-1/2}` to reach
    model units, a mass and a potential by :math:`(C q)^{-1}`, and a density
    by :math:`(4\pi C q)^{-1}`.

    References
    ----------
    Chiang+(2026d), Section 2.1 and Eq. (6).

    Examples
    --------
    >>> import math
    >>> from pianistpy import DoublePowerLaw
    >>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
    >>> factors.rho_norm
    1.0
    >>> round(factors.mass_q, 12) == round(math.log(11.0) - 10.0 / 11.0, 12)
    True
    >>> physical = unit_factors(
    ...     DoublePowerLaw.nfw(), 10.0, mass_0_msun=1e12, r_0_pc=2e4
    ... )
    >>> physical.physical_r_pc
    20000.0
    """
    if not concentration > 0.0:
        raise InputError(
            f"The concentration must be positive; the value given is {concentration}."
        )
    if (mass_0_msun is None) != (r_0_pc is None):
        raise InputError(
            "mass_0_msun and r_0_pc must be given together or not at all; one of "
            "the two is given without the other."
        )
    rho_norm = float(profile.rho_norm)
    mass_q = float(np.asarray(profile.mass(np.array([float(concentration)])))[0])
    mass_q = mass_q / rho_norm
    model = {
        "model_v": math.sqrt(1.0 / (rho_norm * mass_q)),
        "model_mass": 1.0 / (rho_norm * mass_q),
        "model_rho": 1.0 / (4.0 * math.pi * rho_norm * mass_q),
        "model_p": 1.0 / (rho_norm * mass_q),
    }
    if mass_0_msun is None:
        return UnitFactors(rho_norm=rho_norm, mass_q=mass_q, **model)

    if r_0_pc is None:
        raise InputError(
            "The physical unit system needs r_0_pc together with mass_0_msun; "
            "r_0_pc is not given."
        )

    import astropy.units as u
    from astropy.constants import G

    gravity = G.to(u.pc**3 / (u.Msun * u.Myr**2)).value
    pc_per_myr_in_kms = (1.0 * u.pc / u.Myr).to(u.km / u.s).value
    rho_0 = mass_0_msun / (4.0 * math.pi * r_0_pc**3 * mass_q)
    rho_w = rho_0 / rho_norm
    physical_v_kms = (
        math.sqrt(4.0 * math.pi * gravity * rho_w * r_0_pc**2) * pc_per_myr_in_kms
    )
    return UnitFactors(
        rho_norm=rho_norm,
        mass_q=mass_q,
        **model,
        physical_r_pc=r_0_pc,
        physical_v_kms=physical_v_kms,
        physical_mass_msun=1.0 * 4.0 * math.pi * rho_w * r_0_pc**3,
        physical_rho_msun_pc3=rho_w,
        physical_p_kms2=physical_v_kms**2,
        physical_rho_0_msun_pc3=rho_0,
    )
