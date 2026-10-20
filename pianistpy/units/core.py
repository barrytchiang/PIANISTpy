"""Conversion factors from internal units to model and physical units.

Every computation runs in the internal units of Chiang+(2026d), in which the
central potential of the untruncated profile is 1. This module holds the one
place where results leave those units.
"""

__all__ = ["UnitFactors", "unit_factors"]

import math
from dataclasses import dataclass

import numpy as np

from ..exceptions import InputError
from ..profiles import DoublePowerLaw


@dataclass(frozen=True)
class UnitFactors:
    r"""Factors that convert internal units to model and to physical units.

    A quantity in internal units is multiplied by the factor to obtain it in
    the target units. The model units have :math:`G = M_0 = r_0 = 1`, with
    :math:`M_0` the mass inside the radius `concentration` times
    :math:`r_0`.

    Parameters
    ----------
    rho_norm : float
        Normalization :math:`C` of the density profile.
    mass_q : float
        :math:`q = \mathcal{M}(c)/C`, the mass inside the concentration
        radius in units of :math:`4\pi\rho_0 r_0^3`.
    model_v, model_mass, model_rho, model_p : float
        Factors to the model units, for a speed, a mass, a density and a
        potential.
    physical_r_pc, physical_v_kms, physical_mass_msun : float or None
        Factors to parsecs, kilometers per second and solar masses.
    physical_rho_msun_pc3, physical_p_kms2 : float or None
        Factors to solar masses per cubic parsec, and to the square of
        kilometers per second.
    physical_rho_0_msun_pc3 : float or None
        The scale density :math:`\rho_0` in solar masses per cubic parsec.

    Notes
    -----
    The physical factors are None unless both physical scales were given to
    `unit_factors`.
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
    r"""Return the conversion factors of a profile.

    Parameters
    ----------
    profile : DoublePowerLaw
        The density profile, which fixes the normalization :math:`C`.
    concentration : float
        Radius, in units of the scale radius :math:`r_0`, inside which the
        mass :math:`M_0` is defined. Positive.
    mass_0_msun : float, optional
        :math:`M_0` in solar masses.
    r_0_pc : float, optional
        Scale radius :math:`r_0` in parsecs.

    Returns
    -------
    factors : UnitFactors
        The model-unit factors, and the physical factors if both `mass_0_msun`
        and `r_0_pc` are given.

    Notes
    -----
    With :math:`\rho_{\rm w} = \rho_0/C` the internal units are
    :math:`M_{\rm W} = 4\pi\rho_{\rm w} r_0^3` for mass and
    :math:`v_{\rm W} = \sqrt{4\pi G\rho_{\rm w} r_0^2}` for speed, and
    :math:`M_0 = 4\pi\rho_0 r_0^3\,q`. Hence a speed is multiplied by
    :math:`(C q)^{-1/2}`, a mass and a potential by :math:`(C q)^{-1}`, and a
    density by :math:`(4\pi C q)^{-1}` to reach the model units.

    References
    ----------
    Chiang+(2026d), Section on units (labels ``ssec:Units``, ``eqn:M_nor_1``).

    Examples
    --------
    >>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
    >>> factors.rho_norm
    1.0
    >>> round(factors.mass_q, 12) == round(math.log(11.0) - 10.0 / 11.0, 12)
    True
    """
    if not concentration > 0.0:
        raise InputError(f"concentration must be positive; got {concentration}.")
    if (mass_0_msun is None) != (r_0_pc is None):
        raise InputError("mass_0_msun and r_0_pc are given together or not at all.")
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
        raise InputError("Physical units need r_0_pc together with mass_0_msun.")

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
