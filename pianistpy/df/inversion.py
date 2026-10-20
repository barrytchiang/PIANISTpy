"""The order of the inversion, its prefactors and its interior point."""

__all__ = [
    "OrderPair",
    "forward_prefactor",
    "interior_point",
    "inversion_prefactor",
    "order_pair",
]

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_function

from ..anisotropy import Anisotropy
from ..exceptions import InputError, UnphysicalModelError
from ..profiles import DoublePowerLaw

ORDER_OFFSET = 1.5
KERNEL_EXPONENT_MAX = 0.5


@dataclass(frozen=True)
class OrderPair:
    r"""The order of the inversion for one central anisotropy.

    Parameters
    ----------
    order_m : int
        :math:`m = \lfloor 3/2 - \beta_{\rm c} \rfloor`.
    order_n : float
        :math:`n = 3/2 - \beta_{\rm c} - m`, with :math:`0 \leq n < 1`.
    order_j : int
        Derivative order in use: :math:`m + 1` where :math:`n > 1/2`,
        otherwise :math:`m`.
    kernel_exponent : float
        Kernel exponent in use: :math:`n - 1` where :math:`n > 1/2`,
        otherwise :math:`n`. It lies in :math:`(-1/2, 1/2]`.
    """

    order_m: int
    order_n: float
    order_j: int
    kernel_exponent: float


def order_pair(beta_c: float) -> OrderPair:
    r"""Return the order of the inversion for a central anisotropy.

    Parameters
    ----------
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`, smaller than 1. The OM
        models enter with 0.

    Returns
    -------
    pair : OrderPair

    Notes
    -----
    For half-integer :math:`\beta_{\rm c}` the remainder :math:`n` is exactly
    zero: :math:`3/2 - \beta_{\rm c}` is then an integer that double
    precision holds exactly, so the test needs no tolerance.

    References
    ----------
    Eq. (T:0.1) [order], Eq. (T:2.16) [unified-pair].

    Examples
    --------
    >>> order_pair(0.0)
    OrderPair(order_m=1, order_n=0.5, order_j=1, kernel_exponent=0.5)
    >>> order_pair(-0.25)
    OrderPair(order_m=1, order_n=0.75, order_j=2, kernel_exponent=-0.25)
    """
    beta_c = float(beta_c)
    if not beta_c < 1.0:
        raise InputError(f"beta_c must be smaller than 1; got {beta_c}.")
    order = ORDER_OFFSET - beta_c
    order_m = int(np.floor(order))
    order_n = 0.0 if float(order).is_integer() else float(order - order_m)
    if order_n > KERNEL_EXPONENT_MAX:
        return OrderPair(order_m, order_n, order_m + 1, order_n - 1.0)
    return OrderPair(order_m, order_n, order_m, order_n)


def inversion_prefactor(kernel_exponent: float, beta_c: float) -> float:
    r"""Return the prefactor of the inversion.

    Parameters
    ----------
    kernel_exponent : float
        Kernel exponent :math:`n` in use.
    beta_c : float
        Central anisotropy.

    Returns
    -------
    prefactor : float

    Notes
    -----
    .. math::

        C(n) = \frac{2^{\beta_{\rm c}}}
                    {(2\pi)^{3/2}\,\Gamma(1-n)\,\Gamma(1-\beta_{\rm c})} .

    At :math:`n = 0`, that is :math:`\beta_{\rm c} = 1/2 - k` with integer
    :math:`k`, the exact reduction :math:`1/[2\pi^2 (2k-1)!!]` is used.

    References
    ----------
    Chiang+(2026d), label ``eqn:COM_DF_gen``; Eq. (T:2.2) [local].
    """
    if kernel_exponent == 0.0:
        k_half = int(round(0.5 - float(beta_c)))
        double_factorial = int(np.prod(np.arange(1, 2 * k_half, 2), dtype=np.int64))
        return 1.0 / (2.0 * np.pi**2 * double_factorial)
    return float(
        (2.0**beta_c)
        / (
            (2.0 * np.pi) ** 1.5
            * gamma_function(1.0 - kernel_exponent)
            * gamma_function(1.0 - beta_c)
        )
    )


def forward_prefactor(beta_c: float) -> float:
    r"""Return the prefactor of the density integral over the DF.

    Parameters
    ----------
    beta_c : float
        Central anisotropy.

    Returns
    -------
    prefactor : float
        :math:`2\pi^{3/2}\,\Gamma(1-\beta_{\rm c})/\Gamma(3/2-\beta_{\rm c})`;
        :math:`4\pi` for an isotropic model.

    References
    ----------
    Eq. (T:2.3) [volterra].
    """
    return float(
        2.0 * np.pi**1.5 * gamma_function(1.0 - beta_c) / gamma_function(1.5 - beta_c)
    )


def interior_point(
    profile: DoublePowerLaw,
    anisotropy: Anisotropy,
    order_j: int,
    full_domain: bool,
) -> tuple[float, float]:
    r"""Return the interior point of the auxiliary density and its exponent.

    Only a COM Type II model inverted over its full domain has one, at the
    potential of the anisotropy radius. There the derivative of order
    `order_j` of the auxiliary density behaves as
    :math:`|P - P(R_{\rm a})|^{s}` with :math:`s = 1 - \beta_{\rm c} - j`,
    or jumps (:math:`s = 0`) where :math:`1 - \beta_{\rm c}` is an even
    integer; where it is an odd integer the auxiliary density is analytic.

    Parameters
    ----------
    profile : DoublePowerLaw
        Density profile, which gives the potential of the anisotropy radius.
    anisotropy : Anisotropy
        The anisotropy model.
    order_j : int
        Derivative order in use.
    full_domain : bool
        Whether the inversion runs over the untruncated model.

    Returns
    -------
    p_interior : float
        Potential of the interior point; zero where there is none.
    exponent : float
        The exponent :math:`s`; zero where there is no interior point or
        where the derivative jumps.

    Raises
    ------
    UnphysicalModelError
        If the derivative is not integrable at the interior point.

    References
    ----------
    Eq. (T:2.15) [Ra-exponent], Eq. (T:3.1) [continuation].
    """
    is_com_type_2 = anisotropy.model == "com" and anisotropy.kind == "II"
    if not (full_domain and is_com_type_2):
        return 0.0, 0.0
    power = 1.0 - float(anisotropy.beta_c)
    if float(power).is_integer():
        if int(power) % 2 == 1:
            return 0.0, 0.0
        if int(order_j) > int(power):
            raise UnphysicalModelError(
                f"For COM Type II with beta_c = {anisotropy.beta_c:+.4f}, the "
                f"derivative of order {int(order_j)} of the auxiliary density "
                "carries a delta function at the anisotropy radius."
            )
        exponent = 0.0
    else:
        exponent = power - float(order_j)
        if not exponent > -1.0:
            raise UnphysicalModelError(
                f"For COM Type II with beta_c = {anisotropy.beta_c:+.4f}, the "
                f"derivative of order {int(order_j)} of the auxiliary density "
                f"goes as |P - P(R_a)|^{exponent:+.3f} and is not integrable."
            )
    p_interior = float(profile.p(np.array([float(anisotropy.r_a)]))[0])
    return p_interior, exponent
