"""The DF below the first energy of its table: the tail and its mass."""

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

# Above this argument the scaled incomplete gamma function is evaluated by
# its asymptotic series, below it by the regularised function.
SERIES_ARGUMENT_MIN = 30.0
SERIES_TERMS_MAX = 80
# For an order -1 < s < 0, the continued fraction from this argument up and
# the series about s = 0 below it; the most terms of either.
FRACTION_ARGUMENT_MIN = 1.0
FRACTION_TERMS_MAX = 500


def ln_ratio(p_0: float, k: ArrayLike, xp=np) -> Array:
    r"""Return :math:`\ln(P_0/\mathcal{K})`, finite for every positive energy.

    Parameters
    ----------
    p_0 : float
        Central potential, in internal units.
    k : array-like
        Energy, positive.
    xp : module, optional
        ``numpy``, or ``jax.numpy`` in a kernel.

    Returns
    -------
    ln_ratio : array
        The logarithm of the ratio where the ratio is a finite double, and
        the difference of the logarithms where the ratio overflows.
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
        Amplitude :math:`A` of the tail.
    tail_exponent : float
        Exponent :math:`a` of the power law.
    k : array-like
        Energy, in internal units.
    tail_log_exponent : float, optional
        Exponent :math:`b` of the logarithmic factor; zero for a pure power
        law, which a profile with an outer slope above 3 has.
    p_0 : float, optional
        Central potential.

    Returns
    -------
    log10_df : array

    Notes
    -----
    .. math::

        F(\mathcal{K}) = A\,\mathcal{K}^{-a}\,
            \left[\ln(P_0/\mathcal{K})\right]^{-b} .

    References
    ----------
    Notes Eq. (I.16) [tail-exponent].
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
        Order of the incomplete gamma function; any real number.
    x : array-like
        Argument, positive.

    Returns
    -------
    value : array, shape of `x` with at least one dimension

    Notes
    -----
    Below an argument of 30 the regularised function is used, with the
    exponential integral at :math:`s = 0`. For :math:`-1 < s < 0` the
    function is evaluated at the order itself: the continued fraction
    :math:`e^{x}\Gamma(s, x) = x^{s}/(x+1-s-\,1(1-s)/(x+3-s-\cdots))` from
    `FRACTION_ARGUMENT_MIN` up, and below it the series about
    :math:`s = 0`,

    .. math::

        \Gamma(s, x) = \frac{\operatorname{expm1}\ln\Gamma(1+s)
            - \operatorname{expm1}(s\ln x)}{s}
            - x^{s}\sum_{n\ge1}\frac{(-x)^{n}}{n!\,(s+n)},

    whose bracket is formed to its own precision before the division by
    `s`. The recurrence :math:`\Gamma(s, x) = [\Gamma(s+1, x) -
    x^{s}e^{-x}]/s` divided a rounded difference by `s` there and read
    1e-11 relative at :math:`s \approx -0.01`; it remains for
    :math:`s \le -1`, where `s` magnifies nothing. Above 30, except at
    :math:`-1 < s < 0`, the asymptotic series
    :math:`x^{s-1}\sum_n (s-1)\cdots(s-n)/x^{n}` is summed, each element
    stopped at its smallest term.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty(x.shape)
    # at an order -1 < s < 0 the continued fraction serves every argument from
    # FRACTION_ARGUMENT_MIN up, where the asymptotic series reads 7e-13 near 30
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
    """`state = term(i, state)` for i = 1, 2, ... below `trips`, the state's last
    entry the mask of the lanes not yet converged, which `term` keeps frozen.
    The loop leaves early once no lane is live: the result is that of every
    trip. A Python loop in NumPy, a `jax.lax.while_loop` in a kernel."""
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
    """e^x Gamma(s, x) for -1 < s < 0 and positive x, at the order itself: the
    continued fraction (modified Lentz) from FRACTION_ARGUMENT_MIN up, the series
    about s = 0 below it. Traceable: in NumPy, or in a kernel with
    `xp=jax.numpy` and `s` traced; each loop runs over every lane, its most
    trips fixed, under the mask of the lanes not yet converged."""
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
        Amplitude :math:`A` of the tail.
    tail_exponent : float
        Exponent :math:`a` of the power law, smaller than 1.
    k : array-like
        Upper limit of the integral, in internal units.
    tail_log_exponent : float, optional
        Exponent :math:`b` of the logarithmic factor.
    p_0 : float, optional
        Central potential.
    xp : module, optional
        ``numpy``, or ``jax.numpy`` in a kernel, where the numbers may be
        traced and the order :math:`1-b` must lie in :math:`(-1, 0)`, the
        only one a kernel reads.

    Returns
    -------
    integral : array, shape of `k` with at least one dimension (in a kernel,
        the shape of `k`)

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
