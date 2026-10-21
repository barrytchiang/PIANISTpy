r"""Order, prefactors and interior point of the DF inversion.

The inversion of a model of central anisotropy :math:`\beta_{\rm c}`
differentiates the auxiliary density :math:`\rho_{\rm aux}(P)` an integer
number of times and integrates the derivative against the kernel
:math:`(Q - P)^{-n}`, with :math:`P` the potential, :math:`Q` the energy
and :math:`n` the kernel exponent. `order_pair` returns the integer order and the kernel
exponent, `inversion_prefactor` the constant in front of the inversion,
`forward_prefactor` the constant in front of the density integral over the
DF, and `interior_point` the potential inside the domain where a COM Type II
auxiliary density is not smooth. The callers pass :math:`\beta_{\rm c} = 0`
for Osipkov-Merritt (OM) models.
"""

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

# The 3/2 of the order 3/2 - beta_c of the inversion: an exact constant of
# the inversion formula.
ORDER_OFFSET = 1.5
# The largest kernel exponent the inversion uses: a remainder n above 1/2 is
# replaced by n - 1 with one more derivative, so that every kernel exponent
# lies in (-1/2, 1/2].
KERNEL_EXPONENT_MAX = 0.5


@dataclass(frozen=True)
class OrderPair:
    r"""The derivative order and the kernel exponent of the inversion.

    Parameters
    ----------
    order_m : int
        Integer part :math:`m = \lfloor 3/2 - \beta_{\rm c} \rfloor`, with
        :math:`\beta_{\rm c}` the central anisotropy.
    order_n : float
        Remainder :math:`n = 3/2 - \beta_{\rm c} - m`, with
        :math:`0 \leq n < 1`.
    order_j : int
        Derivative order :math:`j` the inversion uses: :math:`j = m + 1`
        where :math:`n > 1/2`, and :math:`j = m` where :math:`n \leq 1/2`.
    kernel_exponent : float
        Kernel exponent the inversion uses: :math:`n - 1` where
        :math:`n > 1/2`, and :math:`n` where :math:`n \leq 1/2`. The kernel
        exponent lies in :math:`(-1/2, 1/2]`.
    """

    order_m: int
    order_n: float
    order_j: int
    kernel_exponent: float


def order_pair(beta_c: float) -> OrderPair:
    r"""Return the derivative order and the kernel exponent of the inversion.

    Parameters
    ----------
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`. The value must be below 1.
        OM models pass 0.

    Returns
    -------
    pair : OrderPair
        The integer part :math:`m`, the remainder :math:`n`, the derivative
        order :math:`j` and the kernel exponent for `beta_c`.

    Raises
    ------
    InputError
        If `beta_c` is not below 1.

    Notes
    -----
    For a half-integer :math:`\beta_{\rm c}`, the number
    :math:`3/2 - \beta_{\rm c}` is an integer that double precision holds
    exactly. The remainder :math:`n` is then set to exactly zero, and the
    test for an integer needs no tolerance.

    References
    ----------
    Notes Eq. (T:0.1), Notes Eq. (T:2.16).

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
    r"""Return the constant prefactor :math:`C(n)` of the DF inversion.

    Parameters
    ----------
    kernel_exponent : float
        Kernel exponent :math:`n` the inversion uses
        (`OrderPair.kernel_exponent`).
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`. OM models pass 0.

    Returns
    -------
    prefactor : float
        The prefactor :math:`C(n)`, defined under Notes below.

    Notes
    -----
    .. math::

        C(n) = \frac{2^{\beta_{\rm c}}}
                    {(2\pi)^{3/2}\,\Gamma(1-n)\,\Gamma(1-\beta_{\rm c})} .

    The kernel exponent :math:`n` is exactly zero only for
    :math:`\beta_{\rm c} = 1/2 - k` with a non-negative integer :math:`k`.
    For :math:`n = 0` the call returns the exact reduction
    :math:`C(0) = 1/[2\pi^2 (2k-1)!!]`, with :math:`(-1)!! = 1`.

    References
    ----------
    Chiang+(2026d), Eq. (B22). Notes Eq. (T:2.2).
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
    r"""Return the constant prefactor of the density integral over the DF.

    The density integral is
    :math:`\rho_{\rm aux}(P) = A(\beta_{\rm c}) \int_0^P F(Q)\,
    [2(P - Q)]^{1/2 - \beta_{\rm c}}\,{\rm d}Q`, with :math:`F` the DF,
    :math:`Q` the energy and :math:`P` the potential. The call returns
    :math:`A(\beta_{\rm c})`.

    Parameters
    ----------
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`. OM models pass 0.

    Returns
    -------
    prefactor : float
        :math:`A(\beta_{\rm c}) = 2\pi^{3/2}\,\Gamma(1-\beta_{\rm c})
        /\Gamma(3/2-\beta_{\rm c})`. The prefactor equals :math:`4\pi` for
        :math:`\beta_{\rm c} = 0`.

    References
    ----------
    Notes Eq. (T:2.3).
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

    Only a COM Type II model inverted over its full domain has an interior
    point. The interior point is the potential :math:`P(R_{\rm a})` at the
    anisotropy radius :math:`R_{\rm a}`. Near :math:`P(R_{\rm a})`, the
    derivative of order :math:`j` of the auxiliary density behaves as
    :math:`|P - P(R_{\rm a})|^{s}` with :math:`s = 1 - \beta_{\rm c} - j`.
    Where :math:`1 - \beta_{\rm c}` is an odd integer, the auxiliary density
    is analytic at :math:`P(R_{\rm a})` and the call returns no interior
    point. Where :math:`1 - \beta_{\rm c}` is an even integer, the call
    returns the exponent :math:`s = 0`, which stands for a jump of the
    derivative.

    Parameters
    ----------
    profile : DoublePowerLaw
        Density profile. The profile's potential at :math:`R_{\rm a}` is the
        interior point.
    anisotropy : Anisotropy
        Anisotropy model of the component.
    order_j : int
        Derivative order :math:`j` the inversion uses
        (`OrderPair.order_j`).
    full_domain : bool
        True where the inversion runs over the untruncated model.

    Returns
    -------
    p_interior : float
        Potential :math:`P(R_{\rm a})` of the interior point, in internal
        units. Zero where the model has no interior point.
    exponent : float
        The exponent :math:`s`. Zero where the model has no interior point,
        and zero where the derivative jumps.

    Raises
    ------
    UnphysicalModelError
        If :math:`1 - \beta_{\rm c}` is an even integer below :math:`j`: the
        derivative of order :math:`j` then carries a delta function at
        :math:`P(R_{\rm a})`.
    UnphysicalModelError
        If :math:`1 - \beta_{\rm c}` is not an integer and
        :math:`s \leq -1`: the derivative of order :math:`j` is then not
        integrable at :math:`P(R_{\rm a})`.

    References
    ----------
    Notes Eq. (T:2.15), Notes Eq. (T:3.1).
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
