r"""The series of a double-power-law profile near the center.

The enclosed mass, the depth :math:`P_0 - P(R)` and the relative potential
near the center are sums of the first `N_TERMS` terms of their series in
:math:`R^{\alpha}`, with :math:`R` the radius, :math:`\alpha` the profile's
transition sharpness and :math:`P_0` the central relative potential. The
depth's sum takes no difference from :math:`P_0`. `inner_density` returns the
closed-form density, not a series. `series_join` returns the largest radius
at which the absolute value of the next term of the depth's series is at
most :math:`10^{-7}` times the sum of the first `N_TERMS` terms, by default
(Notes Section T:1.2.2).
"""

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
from ..validation import TIER_BOUNDS
from .core import DoublePowerLaw

# Number of terms summed in each series in powers of R**rho_alpha.
N_TERMS = 4
# series_join searches the join radius on JOIN_N_R radii uniform in log10 R
# from JOIN_LOG10_R_MIN to JOIN_LOG10_R_MAX, R in internal units.
JOIN_LOG10_R_MIN = -14.0
JOIN_LOG10_R_MAX = 0.5
JOIN_N_R = 5801
# The default tolerance of series_join is JOIN_GOAL_FRACTION times the 1e-6
# bound of TIER_BOUNDS[0], that is 1e-7, so that the series' truncation error
# uses a tenth of the 1e-6 bound.
JOIN_GOAL_FRACTION = 0.1


def _binomial_sum(
    profile: DoublePowerLaw, r: Array, power_offset: float, with_second_factor: bool
) -> tuple[Array, float]:
    """Return a partial sum of the mass or depth series and its last binomial.

    The partial sum holds the first N_TERMS terms of the series, without the
    amplitude C of the density. The density term
    ``binomial(e, k) * r**(-gamma + k * alpha)``, with
    ``e = (gamma - beta) / alpha``, contributes to the mass the term
    ``binomial(e, k) * r**(q + k * alpha) / (q + k * alpha)`` with
    ``q = 3 - gamma`` (``power_offset = 3`` and
    ``with_second_factor=False``), and contributes to the depth the term
    ``binomial(e, k) * r**(q + k * alpha) / ((q + k * alpha) * (q + 1 + k *
    alpha))`` with ``q = 2 - gamma`` (``power_offset = 2`` and
    ``with_second_factor=True``). The returned binomial coefficient is
    ``binomial(e, N_TERMS - 1)``, from which `series_join` forms the next
    term.
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
    r"""Return the density of a double-power-law profile with a given amplitude.

    The density is the closed form
    :math:`\varrho(R) = C R^{-\gamma} (1 + R^{\alpha})^{(\gamma-\beta)/\alpha}`,
    not a truncated series. Only the amplitude :math:`C` may differ from the
    profile's own normalization `DoublePowerLaw.rho_norm`.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile whose slopes :math:`\alpha`, :math:`\beta` and
        :math:`\gamma` fix the shape of the density.
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C` of the density, in internal units. Default:
        ``None``, which means the profile's own `DoublePowerLaw.rho_norm`.

    Returns
    -------
    rho : array
        Density :math:`\varrho(R)` in internal units, with the shape of `r`.

    References
    ----------
    Chiang+(2026d), Eq. (A9), gives the first two terms of the density's
    series in :math:`R^{\alpha}`.
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

    The enclosed mass :math:`\mathcal{M}(R)` is the sum of the first
    `N_TERMS` terms of its series in :math:`R^{\alpha}`, each term the
    integral of one term of the binomial series of the density. The series
    converges for :math:`R < 1` only.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile whose slopes fix the series.
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C` of the density, in internal units. Default:
        ``None``, which means the profile's own `DoublePowerLaw.rho_norm`.

    Returns
    -------
    mass : array
        Enclosed mass :math:`\mathcal{M}(R)` in internal units, with the
        shape of `r`.

    References
    ----------
    Chiang+(2026d), Eq. (A12), gives the first two terms of the series.
    """
    r = np.asarray(r, dtype=np.float64)
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    total, _ = _binomial_sum(profile, r, 3.0, with_second_factor=False)
    return amplitude * total


def inner_depth(
    profile: DoublePowerLaw, r: ArrayLike, rho_norm: float | None = None
) -> Array:
    r"""Return the depth :math:`P_0 - P(R)` near the center from its series.

    The depth is the sum of the first `N_TERMS` terms of its series in
    :math:`R^{\alpha}`, term by term, with no subtraction from the central
    potential :math:`P_0`. The sum therefore keeps its relative precision
    where the depth is small compared with :math:`P_0`. The series converges
    for :math:`R < 1` only.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile whose slopes fix the series.
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C` of the density, in internal units. Default:
        ``None``, which means the profile's own `DoublePowerLaw.rho_norm`.

    Returns
    -------
    depth : array
        Depth :math:`P_0 - P(R)` in internal units, with the shape of `r`.

    References
    ----------
    Chiang+(2026d), Eq. (A11), gives the first two terms of the series.
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
    r"""Return the relative potential near the center from its series.

    The relative potential is :math:`P(R) = P_0 - u(R)`, with the depth
    :math:`u(R)` from `inner_depth`.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile whose slopes fix the series.
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    p_0 : float
        Central relative potential :math:`P_0`, in internal units.
    rho_norm : float, optional
        Amplitude :math:`C` of the density, in internal units. Default:
        ``None``, which means the profile's own `DoublePowerLaw.rho_norm`.

    Returns
    -------
    p : array
        Relative potential :math:`P(R)` in internal units, with the shape
        of `r`.
    """
    return p_0 - inner_depth(profile, r, rho_norm)


def series_join(
    profile: DoublePowerLaw,
    rho_norm: float | None = None,
    tolerance: float | None = None,
) -> tuple[float, float, float]:
    r"""Return the depth, the radius and the bound at the join of the series.

    The join is the largest radius, among `JOIN_N_R` radii uniform in
    :math:`\log_{10} R` from `JOIN_LOG10_R_MIN` to `JOIN_LOG10_R_MAX`, at
    which the absolute value of the next term of the depth's series, divided
    by the sum of the first `N_TERMS` terms, is at most `tolerance`. The join
    radius depends on the profile's slopes and on `tolerance`, and not on
    `rho_norm`.

    Parameters
    ----------
    profile : DoublePowerLaw
        The profile whose slopes fix the series.
    rho_norm : float, optional
        Amplitude :math:`C` of the density, in internal units. Default:
        ``None``, which means the profile's own `DoublePowerLaw.rho_norm`.
    tolerance : float, optional
        Largest ratio of the absolute value of the next term to the sum of
        the first `N_TERMS` terms. Default: ``None``, which means
        `JOIN_GOAL_FRACTION` (0.1) times ``TIER_BOUNDS[0]``
        (:math:`10^{-6}`), that is :math:`10^{-7}`.

    Returns
    -------
    depth_join : float
        Depth :math:`P_0 - P(R)` at the join radius, in internal units.
    r_join : float
        Join radius, in internal units.
    bound_join : float
        Ratio of the absolute value of the next term to the sum of the
        first `N_TERMS` terms at the join radius.

    Raises
    ------
    AccuracyError
        If, at every radius searched, the sum of the first `N_TERMS` terms
        is not positive or the ratio of the absolute value of the next term
        to that sum exceeds `tolerance`.
    """
    amplitude = profile.rho_norm if rho_norm is None else rho_norm
    tolerance = (
        float(TIER_BOUNDS[0]) * JOIN_GOAL_FRACTION if tolerance is None else tolerance
    )
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
    i_join = int(np.max(within))
    return (
        float(amplitude) * float(total[i_join]),
        float(r[i_join]),
        float(bound[i_join]),
    )
