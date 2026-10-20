"""The profile near the center: its series in radius and where it is trusted."""

__all__ = [
    "inner_density",
    "inner_depth",
    "inner_mass",
    "inner_potential",
    "series_join",
]

import numpy as np

from ..exceptions import AccuracyError
from ..typing import Array, ArrayLike
from ..validation import GOAL
from .core import DoublePowerLaw

# Number of terms of the series in powers of r**rho_alpha.
N_TERMS = 4
# The radii, in internal units, on which the depth of the join is searched.
JOIN_LOG10_R_MIN = -14.0
JOIN_LOG10_R_MAX = 0.5
JOIN_N_R = 5801
# The series is trusted to where its next term reaches this fraction of the
# accuracy goal.
JOIN_GOAL_FRACTION = 0.1


def _binomial_sum(
    profile: DoublePowerLaw, r: Array, power_offset: float, with_second_factor: bool
) -> tuple[Array, float]:
    """Return the sum of the first terms of an inner series and its last binomial.

    A density term ``binomial(e, k) r**(-gamma + k alpha)``, with
    ``e = (gamma - beta) / alpha``, contributes ``r**(q + k alpha)`` divided
    by ``q + k alpha`` to the mass (``q = 3 - gamma``) and by
    ``(q + k alpha)(q + 1 + k alpha)`` to the depth (``q = 2 - gamma``).
    """
    alpha, beta, gamma = profile.rho_alpha, profile.rho_beta, profile.rho_gamma
    exponent = (gamma - beta) / alpha
    total = np.zeros_like(r)
    binomial = 1.0
    for k in range(N_TERMS):
        if k > 0:
            binomial *= (exponent - (k - 1)) / k
        power = power_offset - gamma + k * alpha
        divisor = power * (power + 1.0) if with_second_factor else power
        total = total + binomial * r**power / divisor
    return total, binomial


def inner_density(
    profile: DoublePowerLaw, r: ArrayLike, rho_norm: float | None = None
) -> Array:
    r"""Return the density near the center for a given amplitude.

    The shape of the double power law is closed form, so no series is
    truncated; only the amplitude may differ from the profile's own, as it
    does for a truncated model.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile, which fixes the shape.
    r : array-like
        Radius, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C`; the profile's own by default.

    Returns
    -------
    rho : array

    References
    ----------
    Chiang+(2026d), label ``eqn:inner_density_expansion``.
    """
    r = np.asarray(r, dtype=np.float64)
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    exponent = (profile.rho_gamma - profile.rho_beta) / profile.rho_alpha
    correction = (1.0 + r**profile.rho_alpha) ** exponent
    return amplitude * (r ** (-profile.rho_gamma)) * correction


def inner_mass(
    profile: DoublePowerLaw, r: ArrayLike, rho_norm: float | None = None
) -> Array:
    r"""Return the enclosed mass near the center from its series.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile.
    r : array-like
        Radius, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C`; the profile's own by default.

    Returns
    -------
    mass : array

    References
    ----------
    Chiang+(2026d), label ``eqn:inner_mass_expansion``.
    """
    r = np.asarray(r, dtype=np.float64)
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    total, _ = _binomial_sum(profile, r, 3.0, with_second_factor=False)
    return amplitude * total


def inner_depth(
    profile: DoublePowerLaw, r: ArrayLike, rho_norm: float | None = None
) -> Array:
    r"""Return the depth :math:`P_0 - P(R)` near the center from its series.

    The depth is summed term by term, with no subtraction from the central
    potential.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile.
    r : array-like
        Radius, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C`; the profile's own by default.

    Returns
    -------
    depth : array

    References
    ----------
    Chiang+(2026d), label ``eqn:inner_potential_expansion``.
    """
    r = np.asarray(r, dtype=np.float64)
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    total, _ = _binomial_sum(profile, r, 2.0, with_second_factor=True)
    return amplitude * total


def inner_potential(
    profile: DoublePowerLaw,
    r: ArrayLike,
    p_0: float,
    rho_norm: float | None = None,
) -> Array:
    """Return the potential near the center from its series.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile.
    r : array-like
        Radius, in internal units.
    p_0 : float
        Central potential.
    rho_norm : float, optional
        Amplitude of the density; the profile's own by default.

    Returns
    -------
    p : array
    """
    return p_0 - inner_depth(profile, r, rho_norm)


def series_join(
    profile: DoublePowerLaw,
    rho_norm: float | None = None,
    tolerance: float | None = None,
) -> tuple[float, float, float]:
    """Return the depth to which the inner series is trusted.

    The series is trusted to the largest radius at which its next term,
    relative to the depth, stays within the tolerance. The bound follows
    from the profile's parameters alone; no fixed depth is assumed.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile.
    rho_norm : float, optional
        Amplitude of the density; the profile's own by default.
    tolerance : float, optional
        Largest relative size of the next term. A tenth of the accuracy goal
        by default.

    Returns
    -------
    depth_join : float
        Depth at the join.
    r_join : float
        Radius of the join.
    bound_join : float
        Relative size of the next term there.

    Raises
    ------
    AccuracyError
        If the series holds the tolerance at no radius.
    """
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    tolerance = float(GOAL) * JOIN_GOAL_FRACTION if tolerance is None else tolerance
    alpha, beta, gamma = profile.rho_alpha, profile.rho_beta, profile.rho_gamma
    r = np.logspace(JOIN_LOG10_R_MIN, JOIN_LOG10_R_MAX, JOIN_N_R)
    total, binomial = _binomial_sum(profile, r, 2.0, with_second_factor=True)
    exponent = (gamma - beta) / alpha
    binomial *= (exponent - (N_TERMS - 1)) / N_TERMS
    power = 2.0 - gamma + N_TERMS * alpha
    next_term = np.abs(binomial) * r**power / (power * (power + 1.0))
    bound = np.where(total > 0.0, next_term / np.maximum(total, 1e-300), np.inf)
    within = np.nonzero(np.isfinite(bound) & (bound <= tolerance))[0]
    if within.size == 0:
        raise AccuracyError(
            f"The inner series of {profile.name} holds a relative bound of "
            f"{tolerance:.1e} at no radius."
        )
    i_join = int(within.max())
    return (
        float(amplitude) * float(total[i_join]),
        float(r[i_join]),
        float(bound[i_join]),
    )
