r"""The DF at the edge of a truncated density, and whether it exists.

A density that ends as :math:`\mathcal{P}^q` at its edge has a DF of
central anisotropy :math:`\beta_{\rm c}` if and only if
:math:`\beta_{\rm c} > 1/2 - q`. The DF then goes as
:math:`\mathcal{K}^{\,q - 3/2 + \beta_{\rm c}}` there. The statement holds
for every profile and every anisotropy model of the package; the functions
here are that statement and its two readings.
"""

__all__ = [
    "check_edge_existence",
    "density_edge_exponent",
    "df_edge_amplitude",
    "df_edge_exponent",
    "lowest_beta_c",
]

import math

from scipy.special import beta as beta_function

from ..exceptions import InputError, UnphysicalModelError
from .inversion import forward_prefactor

ORDER_OFFSET = 1.5


def density_edge_exponent(df_edge_exponent: float, beta_c: float) -> float:
    r"""Return the exponent of a density at its edge from that of its DF.

    Parameters
    ----------
    df_edge_exponent : float
        Exponent :math:`p` of the DF at its edge,
        :math:`F \propto \mathcal{K}^p`, above -1. A truncated DF with a
        positive slope at the truncation energy has :math:`p = 1`; one
        truncated at a stationary point has :math:`p = 2`.
    beta_c : float
        Central anisotropy of the DF.

    Returns
    -------
    density_edge_exponent : float
        :math:`q = p + 3/2 - \beta_{\rm c}`, with the density going as
        :math:`\mathcal{P}^q` at the edge.

    Examples
    --------
    >>> density_edge_exponent(1.0, 0.0)
    2.5
    """
    if not df_edge_exponent > -1.0:
        raise InputError(
            f"df_edge_exponent must be above -1 for a finite density; "
            f"got {df_edge_exponent}."
        )
    return df_edge_exponent + ORDER_OFFSET - beta_c


def df_edge_exponent(density_edge_exponent: float, beta_c: float) -> float:
    r"""Return the exponent of a DF at the edge of its density.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at its edge.
    beta_c : float
        Central anisotropy of the DF.

    Returns
    -------
    df_edge_exponent : float
        :math:`p = q - 3/2 + \beta_{\rm c}`. The DF exists only where this
        is above -1 (`check_edge_existence`). It diverges at the edge,
        integrably, where :math:`-1 < p < 0`.

    Examples
    --------
    >>> df_edge_exponent(2.5, -1.5)
    -0.5
    """
    return density_edge_exponent - ORDER_OFFSET + beta_c


def lowest_beta_c(density_edge_exponent: float) -> float:
    r"""Return the central anisotropy below which a density has no DF.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at its edge.

    Returns
    -------
    lowest_beta_c : float
        :math:`1/2 - q`. A DF exists for every central anisotropy strictly
        above it, and for none at or below it.

    Examples
    --------
    >>> lowest_beta_c(2.5)
    -2.0
    """
    return 0.5 - density_edge_exponent


def check_edge_existence(density_edge_exponent: float, beta_c: float) -> None:
    r"""Refuse a central anisotropy for which a truncated density has no DF.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at its edge.
    beta_c : float
        Central anisotropy requested.

    Raises
    ------
    UnphysicalModelError
        If :math:`\beta_{\rm c} \leq 1/2 - q`.

    Notes
    -----
    The condition is necessary and sufficient for the DF to exist near the
    edge. Whether the DF is non-negative over its whole range is a separate
    check of the computed DF.
    """
    lowest = lowest_beta_c(density_edge_exponent)
    if not beta_c > lowest:
        raise UnphysicalModelError(
            f"No distribution function exists for beta_c = {beta_c:g}: a "
            f"density that ends as P^{density_edge_exponent:g} at its edge "
            f"needs beta_c > 1/2 - {density_edge_exponent:g} = {lowest:g}."
        )


def df_edge_amplitude(
    density_edge_amplitude: float, density_edge_exponent: float, beta_c: float
) -> float:
    r"""Return the amplitude of a DF at the edge of its density.

    Parameters
    ----------
    density_edge_amplitude : float
        Amplitude :math:`C` of the auxiliary density at the edge,
        :math:`\rho_{\rm aux} = C\,\mathcal{P}^q`.
    density_edge_exponent : float
        Its exponent :math:`q`.
    beta_c : float
        Central anisotropy of the DF.

    Returns
    -------
    df_edge_amplitude : float
        Amplitude :math:`A` of :math:`F = A\,\mathcal{K}^p`,
        :math:`p = q - 3/2 + \beta_{\rm c}`:
        :math:`A = C / [K_f\,B(p + 1,\, 3/2 - \beta_{\rm c})]`, with
        :math:`K_f = 2^{1/2 - \beta_{\rm c}}` times `forward_prefactor`.

    Raises
    ------
    UnphysicalModelError
        If no DF exists (`check_edge_existence`).
    """
    check_edge_existence(density_edge_exponent, beta_c)
    exponent = df_edge_exponent(density_edge_exponent, beta_c)
    forward = math.pow(2.0, 0.5 - beta_c) * forward_prefactor(beta_c)
    return density_edge_amplitude / (
        forward * float(beta_function(exponent + 1.0, ORDER_OFFSET - beta_c))
    )
