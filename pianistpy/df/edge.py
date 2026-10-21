r"""Exponent and amplitude of a DF at the edge of a truncated density.

A truncated density that ends as :math:`\mathcal{P}^q` at its edge, with
:math:`\mathcal{P}` the potential measured from the edge and :math:`q` the
density's edge exponent, has a DF of central anisotropy :math:`\beta_{\rm c}`
if and only if :math:`\beta_{\rm c} > 1/2 - q`. The DF goes as
:math:`\mathcal{K}^{\,p}` at the edge, with
:math:`p = q - 3/2 + \beta_{\rm c}` and :math:`\mathcal{K}` the energy
measured from the edge. The existence condition and the DF's edge exponent
hold for every profile and every anisotropy model of the package.
`df_edge_exponent` and `density_edge_exponent` convert the DF's edge
exponent :math:`p` and the density's edge exponent :math:`q` into each other,
`lowest_beta_c` and `check_edge_existence` apply the existence condition,
and `df_edge_amplitude` returns the amplitude of the DF's edge power.
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

# ORDER_OFFSET is the 3/2 of the exponent relation p = q - 3/2 + beta_c, an
# exact constant of the density integral over the DF.
ORDER_OFFSET = 1.5


def density_edge_exponent(df_edge_exponent: float, beta_c: float) -> float:
    r"""Return the edge exponent of a density from the edge exponent of its DF.

    Parameters
    ----------
    df_edge_exponent : float
        Exponent :math:`p` of the DF at the edge,
        :math:`F \propto \mathcal{K}^p`. The exponent must be above -1, so
        that the density is finite. A truncated DF with a positive slope at
        the truncation energy has :math:`p = 1`. A DF truncated at a
        stationary point of the DF has :math:`p = 2`.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}` of the DF.

    Returns
    -------
    density_edge_exponent : float
        Exponent :math:`q = p + 3/2 - \beta_{\rm c}` of the density at the
        edge, :math:`\rho \propto \mathcal{P}^q`.

    Raises
    ------
    InputError
        If `df_edge_exponent` is not above -1.

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
    r"""Return the edge exponent of a DF from the edge exponent of its density.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at the edge,
        :math:`\rho \propto \mathcal{P}^q`.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}` of the DF.

    Returns
    -------
    df_edge_exponent : float
        Exponent :math:`p = q - 3/2 + \beta_{\rm c}` of the DF at the edge,
        :math:`F \propto \mathcal{K}^p`. The DF exists only where :math:`p`
        is above -1 (`check_edge_existence`). Where :math:`-1 < p < 0`, the
        DF diverges at the edge and the divergence is integrable.

    Examples
    --------
    >>> df_edge_exponent(2.5, -1.5)
    -0.5
    """
    return density_edge_exponent - ORDER_OFFSET + beta_c


def lowest_beta_c(density_edge_exponent: float) -> float:
    r"""Return the central anisotropy at or below which a density has no DF.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at the edge,
        :math:`\rho \propto \mathcal{P}^q`.

    Returns
    -------
    lowest_beta_c : float
        The bound :math:`1/2 - q`. A DF exists for every central anisotropy
        strictly above :math:`1/2 - q`, and for no central anisotropy at or
        below :math:`1/2 - q`.

    Examples
    --------
    >>> lowest_beta_c(2.5)
    -2.0
    """
    return 0.5 - density_edge_exponent


def check_edge_existence(density_edge_exponent: float, beta_c: float) -> None:
    r"""Raise if a truncated density has no DF of the given central anisotropy.

    Parameters
    ----------
    density_edge_exponent : float
        Exponent :math:`q` of the density at the edge,
        :math:`\rho \propto \mathcal{P}^q`.
    beta_c : float
        Requested central anisotropy :math:`\beta_{\rm c}`.

    Raises
    ------
    UnphysicalModelError
        If :math:`\beta_{\rm c}` is not above :math:`1/2 - q`.

    Notes
    -----
    The condition :math:`\beta_{\rm c} > 1/2 - q` is necessary and
    sufficient for the DF to exist near the edge. The condition does not
    decide whether the DF is non-negative over its whole range: a separate
    check of the computed DF decides the sign.
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
    r"""Return the amplitude of a DF's edge power from its density's edge power.

    Parameters
    ----------
    density_edge_amplitude : float
        Amplitude :math:`C` of the auxiliary density at the edge,
        :math:`\rho_{\rm aux} = C\,\mathcal{P}^q`.
    density_edge_exponent : float
        Exponent :math:`q` of the auxiliary density at the edge.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}` of the DF.

    Returns
    -------
    df_edge_amplitude : float
        Amplitude :math:`A` of the DF's edge power
        :math:`F = A\,\mathcal{K}^p`, with
        :math:`p = q - 3/2 + \beta_{\rm c}`. The amplitude is
        :math:`A = C / [K_f\,B(p + 1,\, 3/2 - \beta_{\rm c})]`, with
        :math:`B` the Euler beta function and
        :math:`K_f = 2^{1/2 - \beta_{\rm c}}\,A(\beta_{\rm c})`, with
        :math:`A(\beta_{\rm c})` the prefactor of the density integral over
        the DF (`forward_prefactor`).

    Raises
    ------
    UnphysicalModelError
        If :math:`\beta_{\rm c}` is not above :math:`1/2 - q`
        (`check_edge_existence`).
    """
    check_edge_existence(density_edge_exponent, beta_c)
    exponent = df_edge_exponent(density_edge_exponent, beta_c)
    forward = math.pow(2.0, 0.5 - beta_c) * forward_prefactor(beta_c)
    return density_edge_amplitude / (
        forward * float(beta_function(exponent + 1.0, ORDER_OFFSET - beta_c))
    )
