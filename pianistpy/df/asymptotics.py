r"""The tail of the DF below the first energy of its table, and the tail's integral.

The DF lookup and the table checks call `tail_log10_df` for the DF below
the table's first energy, and `tail_mass_below` for the integral of the
tail from zero up to an energy. The tail is

.. math::

    F(\mathcal{K}) = A\,\mathcal{K}^{-a}\,
        \left[\ln(P_{0}/\mathcal{K})\right]^{-b},

with amplitude :math:`A`, power-law exponent :math:`a`, logarithmic
exponent :math:`b` and central potential :math:`P_0`. The integral needs
the scaled upper incomplete gamma function :math:`e^{x}\Gamma(s, x)`
(`scaled_upper_gamma`).

References
----------
Notes Eq. (T:2.13), Notes Section T:2.5.3.
"""

__all__ = [
    "ln_ratio",
    "scaled_upper_gamma",
    "tail_log10_df",
    "tail_mass_below",
]

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln as jax_gammaln
from scipy.special import exp1, gammaincc, gammaln
from scipy.special import gamma as gamma_function

from ..typing import Array, ArrayLike

# The argument x from which `scaled_upper_gamma` sums the asymptotic series
# in 1/x, for orders s outside (-1, 0). Below SERIES_ARGUMENT_MIN,
# `scaled_upper_gamma` uses the regularized incomplete gamma function for
# s > 0, the exponential integral for s = 0, and the recurrence in s for
# s <= -1. The asymptotic series sums at most SERIES_TERMS_MAX terms, and the
# series about s = 0 at most SERIES_TERMS_MAX - 1 terms.
SERIES_ARGUMENT_MIN = 30.0
SERIES_TERMS_MAX = 80
# For an order -1 < s < 0, the continued fraction is used from the argument
# FRACTION_ARGUMENT_MIN up, and the series about s = 0 below
# FRACTION_ARGUMENT_MIN. The continued fraction takes at most
# FRACTION_TERMS_MAX - 1 modified Lentz steps.
FRACTION_ARGUMENT_MIN = 1.0
FRACTION_TERMS_MAX = 500


def ln_ratio(p_0: float, k: ArrayLike, xp=np) -> Array:
    r"""Return :math:`\ln(P_0/\mathcal{K})`, finite for every positive energy.

    Parameters
    ----------
    p_0 : float
        Central potential :math:`P_0`, in internal units.
    k : array-like
        Energy :math:`\mathcal{K}`, positive, in internal units.
    xp : module, optional
        The array namespace. Default: ``numpy``; ``jax.numpy`` inside a JAX
        kernel.

    Returns
    -------
    ln_ratio : array
        :math:`\ln(P_0/\mathcal{K})`, with the shape of `k`. Where the ratio
        :math:`P_0/\mathcal{K}` is a finite double, the logarithm of the
        ratio; where the ratio overflows, the difference
        :math:`\ln P_0 - \ln\mathcal{K}`.
    """
    if xp is np:
        k = np.asarray(k, dtype=np.float64)
    with (
        np.errstate(divide="ignore", over="ignore", invalid="ignore")
        if xp is np
        else contextlib.nullcontext()
    ):
        ratio = p_0 / k
        return xp.where(xp.isfinite(ratio), xp.log(ratio), xp.log(p_0) - xp.log(k))


def tail_log10_df(
    tail_amplitude: float,
    tail_exponent: float,
    k: ArrayLike,
    tail_log_exponent: float = 0.0,
    p_0: float = 1.0,
) -> Array:
    r"""Return the base-10 logarithm of the tail of the DF.

    Parameters
    ----------
    tail_amplitude : float
        Amplitude :math:`A` of the tail, in internal units.
    tail_exponent : float
        Exponent :math:`a` of the power law.
    k : array-like
        Energy :math:`\mathcal{K}`, positive, in internal units.
    tail_log_exponent : float, optional
        Exponent :math:`b` of the logarithmic factor. Default: ``0.0``, which
        means a pure power law, the tail of a profile with an outer slope
        above 3.
    p_0 : float, optional
        Central potential :math:`P_0`, in internal units. Default: ``1.0``,
        the central potential of an untruncated model in internal units.
        :math:`P_0` enters only where :math:`b` is not zero.

    Returns
    -------
    log10_df : array
        :math:`\log_{10} F(\mathcal{K})`, with the shape of `k`.

    Notes
    -----
    .. math::

        F(\mathcal{K}) = A\,\mathcal{K}^{-a}\,
            \left[\ln(P_{0}/\mathcal{K})\right]^{-b} .

    References
    ----------
    Notes Eq. (T:2.13).
    """
    k = np.asarray(k, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(tail_amplitude) - tail_exponent * np.log10(k)
        if tail_log_exponent != 0.0:
            out = out - tail_log_exponent * np.log10(ln_ratio(p_0, k))
    return out


def scaled_upper_gamma(s: float, x: ArrayLike) -> Array:
    r"""Return :math:`e^{x}\,\Gamma(s, x)` for real `s` and positive `x`.

    Parameters
    ----------
    s : float
        Order :math:`s` of the incomplete gamma function; any real number.
    x : array-like
        Argument :math:`x`, positive.

    Returns
    -------
    value : array
        :math:`e^{x}\,\Gamma(s, x)`, with the shape of `x` and at least one
        dimension.

    Notes
    -----
    The evaluation depends on the order :math:`s` and the argument
    :math:`x`:

    - For :math:`s > 0` and :math:`x` below `SERIES_ARGUMENT_MIN` (30), the
      regularized upper incomplete gamma function times
      :math:`\Gamma(s)\,e^{x}`.
    - For :math:`s = 0` and :math:`x` below 30, the exponential integral
      :math:`e^{x} E_1(x)`.
    - For :math:`-1 < s < 0` and every :math:`x`, the order itself, with no
      recurrence: from `FRACTION_ARGUMENT_MIN` up, the continued fraction
      :math:`e^{x}\Gamma(s, x) = x^{s}/(x+1-s-\,1(1-s)/(x+3-s-\cdots))`;
      below `FRACTION_ARGUMENT_MIN`, the series about :math:`s = 0`,

      .. math::

          \Gamma(s, x) = \frac{\operatorname{expm1}\ln\Gamma(1+s)
              - \operatorname{expm1}(s\ln x)}{s}
              - x^{s}\sum_{n\ge1}\frac{(-x)^{n}}{n!\,(s+n)},

      whose numerator is formed to full relative precision before the
      division by :math:`s`.
    - For :math:`s \le -1` and :math:`x` below 30, the recurrence
      :math:`\Gamma(s, x) = [\Gamma(s+1, x) - x^{s}e^{-x}]/s`. For
      :math:`s \le -1`, the division by :math:`s` does not magnify the
      rounding error of the difference.
    - For :math:`s` outside :math:`(-1, 0)` and :math:`x` at or above 30,
      the asymptotic series :math:`x^{s-1}\sum_n (s-1)\cdots(s-n)/x^{n}`,
      summed for each element until the magnitude of the terms stops
      decreasing, with at most `SERIES_TERMS_MAX` terms.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty(x.shape)
    # For an order -1 < s < 0, the continued fraction serves every argument
    # from FRACTION_ARGUMENT_MIN up, so the asymptotic series is never used
    # at orders -1 < s < 0.
    small = (x < SERIES_ARGUMENT_MIN) | (-1.0 < s < 0.0)
    if np.any(small):
        x_small = x[small]
        if s > 0.0:
            out[small] = gammaincc(s, x_small) * gamma_function(s) * np.exp(x_small)
        elif s == 0.0:
            out[small] = exp1(x_small) * np.exp(x_small)
        elif s > -1.0:
            out[small] = _scaled_upper_gamma_small_order(s, x_small)
        else:
            out[small] = (scaled_upper_gamma(s + 1.0, x_small) - x_small**s) / s
    if np.any(~small):
        x_large = x[~small]
        term = np.ones_like(x_large)
        total = np.ones_like(x_large)
        previous = np.full_like(x_large, np.inf)
        live = np.ones(x_large.shape, dtype=bool)
        for n in range(1, SERIES_TERMS_MAX):
            term = np.where(live, term * (s - n) / x_large, 0.0)
            live &= np.abs(term) < previous
            term = np.where(live, term, 0.0)
            total = total + term
            previous = np.abs(term)
        out[~small] = x_large ** (s - 1.0) * total
    return out


def _masked_loop(term, state, trips, xp):
    """Return the state after ``state = term(i, state)`` for i = 1, 2, ..., trips - 1.

    The last entry of ``state`` is the mask of the lanes not yet converged,
    and ``term`` leaves unchanged, on the converged lanes, the entries of
    ``state`` that the caller reads. The loop stops early once the mask is
    False on every lane, so the early stop gives the caller the same values
    as running every trip. With ``xp`` equal to NumPy the loop is a
    Python loop; otherwise the loop is a `jax.lax.while_loop` inside a JAX
    kernel.
    """
    if xp is np:
        for i in range(1, trips):
            if not np.any(state[-1]):
                break
            state = term(i, state)
        return state

    def more(carry):
        return (carry[0] < trips) & jnp.any(carry[1][-1])

    def step(carry):
        return carry[0] + 1, term(carry[0], carry[1])

    return jax.lax.while_loop(more, step, (jnp.int32(1), state))[1]


def _scaled_upper_gamma_small_order(s: float, x: Array, xp=np) -> Array:
    """Return e^x Gamma(s, x) for an order -1 < s < 0 and a positive x.

    The function is evaluated at the order s itself, with no recurrence.
    From FRACTION_ARGUMENT_MIN up, the continued fraction is evaluated by the
    modified Lentz method; below FRACTION_ARGUMENT_MIN, the series about
    s = 0 of `scaled_upper_gamma` is summed. The function runs in NumPy, or
    inside a JAX kernel with ``xp=jax.numpy`` and a traced ``s``. The
    continued-fraction loop and the series loop each run over every lane,
    with at most FRACTION_TERMS_MAX - 1 and SERIES_TERMS_MAX - 1 steps
    respectively,
    under the mask of the lanes not yet converged.
    """
    log_gamma = gammaln if xp is np else jax_gammaln
    tiny = 1e-300
    far = x >= FRACTION_ARGUMENT_MIN
    xf = xp.where(far, x, 1.0)
    b_0 = xf + 1.0 - s

    def fraction_term(i, state):
        b, c, d, h, live = state
        an = -i * (i - s)
        b = b + 2.0
        d = an * d + b
        d = xp.where(xp.abs(d) < tiny, tiny, d)
        c = b + an / c
        c = xp.where(xp.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h = xp.where(live, h * delta, h)
        return b, c, d, h, live & (xp.abs(delta - 1.0) > np.finfo(np.float64).eps)

    start = (b_0, xp.full(xf.shape, 1.0 / tiny), 1.0 / b_0, 1.0 / b_0, far)
    h = _masked_loop(fraction_term, start, FRACTION_TERMS_MAX, xp)[3]
    xn = xp.where(far, 0.5, x)
    head = (xp.expm1(log_gamma(1.0 + s)) - xp.expm1(s * xp.log(xn))) / s

    def series_term(n, state):
        term, total, live = state
        term = term * (-xn) / n
        new = total + term / (s + n)
        return term, xp.where(live, new, total), live & (new != total)

    start_series = (xp.ones(xn.shape), xp.zeros(xn.shape), ~far)
    total = _masked_loop(series_term, start_series, SERIES_TERMS_MAX, xp)[1]
    return xp.where(far, xf**s * h, xp.exp(xn) * (head - xn**s * total))


def tail_mass_below(
    tail_amplitude: float,
    tail_exponent: float,
    k: ArrayLike,
    tail_log_exponent: float = 0.0,
    p_0: float = 1.0,
    xp=np,
) -> Array:
    r"""Return the integral of the tail of the DF from zero to an energy.

    Parameters
    ----------
    tail_amplitude : float
        Amplitude :math:`A` of the tail, in internal units.
    tail_exponent : float
        Exponent :math:`a` of the power law, below 1.
    k : array-like
        Upper limit :math:`\mathcal{K}` of the integral, positive, in
        internal units.
    tail_log_exponent : float, optional
        Exponent :math:`b` of the logarithmic factor. Default: ``0.0``, which
        means a pure power law.
    p_0 : float, optional
        Central potential :math:`P_0`, in internal units. Default: ``1.0``,
        the central potential of an untruncated model in internal units.
    xp : module, optional
        The array namespace. Default: ``numpy``. With ``jax.numpy``, inside a
        JAX kernel, the numbers may be traced, and the order :math:`1-b`
        must lie in :math:`(-1, 0)`, the only order range the kernel path
        evaluates.

    Returns
    -------
    integral : array
        :math:`\int_0^{\mathcal{K}} F\,\mathrm{d}E`, in internal units. With
        NumPy, the shape of `k` with at least one dimension; with
        ``jax.numpy``, the shape of `k`.

    Notes
    -----
    For :math:`b = 0` the integral is :math:`A\,\mathcal{K}^{1-a}/(1-a)`.
    Otherwise, with :math:`L = \ln(P_0/\mathcal{K})`,

    .. math::

        \int_0^{\mathcal{K}} F\,\mathrm{d}E = A\,\mathcal{K}^{1-a}\,(1-a)^{b-1}\,
            e^{(1-a)L}\,\Gamma\!\left(1-b,\,(1-a)L\right) .
    """
    if xp is np:
        k = np.atleast_1d(np.asarray(k, dtype=np.float64))
        if tail_log_exponent == 0.0:
            return tail_amplitude * k ** (1.0 - tail_exponent) / (1.0 - tail_exponent)
        argument = (1.0 - tail_exponent) * ln_ratio(p_0, k)
        gamma = scaled_upper_gamma(1.0 - tail_log_exponent, argument)
    else:
        argument = (1.0 - tail_exponent) * ln_ratio(p_0, k, xp)
        gamma = _scaled_upper_gamma_small_order(1.0 - tail_log_exponent, argument, xp)
    return (
        tail_amplitude
        * k ** (1.0 - tail_exponent)
        * (1.0 - tail_exponent) ** (tail_log_exponent - 1.0)
        * gamma
    )
