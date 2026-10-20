"""The double-power-law density profile and its named presets."""

__all__ = ["DoublePowerLaw"]

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import jax.tree_util
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betainc, betaincc, digamma, hyp2f1
from scipy.special import gamma as gamma_function

from ..exceptions import InputError
from ..typing import Array, ArrayLike

RHO_ALPHA_MIN = 0.5
RHO_BETA_MIN = 2.0
RHO_GAMMA_MAX = 2.0
# Outer slope at which the total mass stops converging.
RHO_BETA_FINITE_MASS = 3.0
# Below this radius the closed forms of the named profiles cancel in double
# precision and their power series are used instead.
SERIES_R_MAX = 1e-2
# Working precision, in decimal digits, of the arbitrary-precision path.
MP_DIGITS = 30
# Below this complement the arbitrary-precision path switches from the
# incomplete Beta function in z to its form in the complement.
MP_COMPLEMENT_MIN = "1e-10"

NAMES = {
    (1.0, 3.0, 1.0): "NFW",
    (1.0, 3.0, 0.0): "cored NFW",
    (1.0, 3.0, 0.5): "Widrow II",
    (1.0, 3.0, 1.5): "Widrow IV",
    (2.0, 3.0, 0.0): "Widrow V",
    (2.0, 3.0, 1.0): "Widrow VI",
    (1.0, 4.0, 1.0): "Hernquist",
    (2.0, 5.0, 0.0): "Plummer",
}


@dataclass(frozen=True)
class DoublePowerLaw:
    r"""Double-power-law density profile in internal units.

    Parameters
    ----------
    rho_alpha : float
        Sharpness :math:`\alpha` of the transition between the inner and the
        outer slope. At least 1/2.
    rho_beta : float
        Outer logarithmic slope :math:`\beta`. Greater than 2.
    rho_gamma : float
        Inner logarithmic slope :math:`\gamma`, with
        :math:`0 \leq \gamma < 2`.

    Notes
    -----
    In internal units the scale radius is 1, the central potential is 1 and the
    potential vanishes at infinity:

    .. math::

        \varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha} .

    The lower bound on :math:`\gamma` excludes a density that decreases
    toward the center; the upper bound keeps the central potential finite.
    The bound on :math:`\beta` keeps the potential finite at large radii.

    The named profiles are parameter sets of this one profile, built by the
    class methods `nfw`, `hernquist` and `plummer`. They are not separate
    implementations. Where a profile has a closed form for its mass or its
    potential, that form is used inside `mass` and `p`; the tests hold it to
    the general incomplete-Beta form.

    References
    ----------
    Chiang+(2026d), Appendix A (labels ``eqn:double_power_law_MT``,
    ``eqn:double_power_law_mass``, ``app:P_beta``).

    Examples
    --------
    >>> DoublePowerLaw.nfw()
    DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
    >>> DoublePowerLaw.hernquist().mass_total
    1.0
    >>> float(DoublePowerLaw.plummer().p(0.0))
    1.0
    """

    rho_alpha: float
    rho_beta: float
    rho_gamma: float

    def __post_init__(self) -> None:
        for name in ("rho_alpha", "rho_beta", "rho_gamma"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not self.rho_alpha >= RHO_ALPHA_MIN:
            raise InputError(
                f"rho_alpha must be at least {RHO_ALPHA_MIN}; got {self.rho_alpha}."
            )
        if not 0.0 <= self.rho_gamma < RHO_GAMMA_MAX:
            raise InputError(
                f"rho_gamma must satisfy 0 <= rho_gamma < {RHO_GAMMA_MAX}; "
                f"got {self.rho_gamma}."
            )
        if not self.rho_beta > RHO_BETA_MIN:
            raise InputError(
                f"rho_beta must be greater than {RHO_BETA_MIN}; got {self.rho_beta}."
            )

    @classmethod
    def nfw(cls) -> "DoublePowerLaw":
        """Return the Navarro-Frenk-White profile, ``(1, 3, 1)``."""
        return cls(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)

    @classmethod
    def hernquist(cls) -> "DoublePowerLaw":
        """Return the Hernquist profile, ``(1, 4, 1)``."""
        return cls(rho_alpha=1.0, rho_beta=4.0, rho_gamma=1.0)

    @classmethod
    def plummer(cls) -> "DoublePowerLaw":
        """Return the Plummer profile, ``(2, 5, 0)``."""
        return cls(rho_alpha=2.0, rho_beta=5.0, rho_gamma=0.0)

    @property
    def _key(self) -> tuple[float, float, float]:
        return (self.rho_alpha, self.rho_beta, self.rho_gamma)

    @property
    def name(self) -> str:
        """Name of the profile, or its parameters if it has none."""
        return NAMES.get(
            self._key,
            f"double power law ({self.rho_alpha:g},{self.rho_beta:g},"
            f"{self.rho_gamma:g})",
        )

    @property
    def rho_norm(self) -> float:
        r"""Normalization :math:`C` of the density, for a central potential of 1.

        Notes
        -----
        .. math::

            C = \frac{\alpha\,\Gamma[(\beta-\gamma)/\alpha]}
                     {\Gamma[(\beta-2)/\alpha]\,\Gamma[(2-\gamma)/\alpha]} .
        """
        alpha, beta, gamma = self._key
        return (
            alpha
            * gamma_function((beta - gamma) / alpha)
            / (
                gamma_function((beta - 2.0) / alpha)
                * gamma_function((2.0 - gamma) / alpha)
            )
        )

    @property
    def mass_total(self) -> float:
        """Total mass in internal units; infinite for an outer slope of 3 or less."""
        alpha, beta, gamma = self._key
        if beta > RHO_BETA_FINITE_MASS:
            return float(
                (self.rho_norm / alpha)
                * beta_function((3.0 - gamma) / alpha, (beta - 3.0) / alpha)
            )
        return math.inf

    def rho(self, r: ArrayLike) -> Array:
        r"""Return the density :math:`\varrho(R)`.

        Parameters
        ----------
        r : array-like
            Radius in internal units.

        Returns
        -------
        rho : array
            Density in internal units, with the shape of `r`.
        """
        r = np.asarray(r, dtype=np.float64)
        alpha, beta, gamma = self._key
        return (
            self.rho_norm * r ** (-gamma) * (1.0 + r**alpha) ** ((gamma - beta) / alpha)
        )

    def mass(self, r: ArrayLike) -> Array:
        r"""Return the enclosed mass :math:`\mathcal{M}(R)`.

        Parameters
        ----------
        r : array-like
            Radius in internal units.

        Returns
        -------
        mass : array
            Enclosed mass in internal units, with the shape of `r`.

        Notes
        -----
        With :math:`z = R^\alpha/(1+R^\alpha)`,

        .. math::

            \mathcal{M}(R) = \frac{C}{\alpha}\,
                B\!\left(z;\,\frac{3-\gamma}{\alpha},\,\frac{\beta-3}{\alpha}\right) .
        """
        shape = np.shape(r)
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        value = _mass_closed(self, r)
        if value is None:
            value = _mass_general(self, r)
        return np.reshape(value, shape)

    def p(self, r: ArrayLike) -> Array:
        r"""Return the relative potential :math:`P(R)`.

        Parameters
        ----------
        r : array-like
            Radius in internal units.

        Returns
        -------
        p : array
            Relative potential in internal units, with the shape of `r`. It is 1
            at the center and vanishes at infinity.

        Notes
        -----
        With :math:`z = R^\alpha/(1+R^\alpha)`, :math:`a_2 = (2-\gamma)/\alpha`
        and :math:`b_2 = (\beta-2)/\alpha`,

        .. math::

            P(R) = \frac{\mathcal{M}(R)}{R}
                 + \frac{C}{\alpha}\left[B(a_2, b_2) - B(z;\,a_2, b_2)\right] .
        """
        shape = np.shape(r)
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        value = _p_closed(self, r)
        if value is None:
            value = _p_general(self, r)
        return np.reshape(value, shape)

    def depth_inner(self, r: ArrayLike, terms: int = 4) -> tuple[Array, Array]:
        r"""Return the depth :math:`u = P(0) - P(R)` near the center, with no
        subtraction, and the series' own truncation estimate.

        Parameters
        ----------
        r : array-like
            Radius in internal units.
        terms : int, optional
            Number of terms of the series.

        Returns
        -------
        u : array
            The depth from the first `terms` terms.
        bound : array
            The magnitude of the next term relative to `u`, per point;
            infinite where `u` is not positive.

        Notes
        -----
        A density term :math:`C\binom{e}{k}R^{-\gamma+k\alpha}`,
        :math:`e = (\gamma-\beta)/\alpha`, contributes
        :math:`C\binom{e}{k}R^{2-\gamma+k\alpha}/[(2-\gamma+k\alpha)(3-\gamma+k\alpha)]`
        to the depth, term by term.
        """
        r = np.asarray(r, dtype=np.float64)
        g, a = self.rho_gamma, self.rho_alpha
        e = (g - self.rho_beta) / a
        # the series in R^alpha converges for R < 1 only: it is not formed
        # beyond, where the depth is infinite to it and its bound infinite
        inside = r < 1.0
        r = np.where(inside, r, 0.5)
        acc = np.zeros_like(r)
        bk = 1.0
        for k in range(terms):
            if k > 0:
                bk *= (e - (k - 1)) / k
            acc = acc + bk * r ** (2.0 - g + k * a) / (
                (2.0 - g + k * a) * (3.0 - g + k * a)
            )
        bk *= (e - (terms - 1)) / terms
        nxt = (
            np.abs(bk)
            * r ** (2.0 - g + terms * a)
            / ((2.0 - g + terms * a) * (3.0 - g + terms * a))
        )
        u = np.where(inside, self.rho_norm * acc, np.inf)
        return u, np.where(inside & (u > 0.0), nxt / np.maximum(acc, 1e-300), np.inf)

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a plain dictionary."""
        return {"profile": "double_power_law", **asdict(self)}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DoublePowerLaw":
        """Build the profile from its dictionary form."""
        return cls(
            rho_alpha=float(record["rho_alpha"]),
            rho_beta=float(record["rho_beta"]),
            rho_gamma=float(record["rho_gamma"]),
        )


def _exponents(profile: DoublePowerLaw) -> tuple[float, float, float, float]:
    """Return the arguments of the Beta functions of the mass and the potential."""
    alpha, beta, gamma = profile._key
    return (
        (3.0 - gamma) / alpha,
        (beta - 3.0) / alpha,
        (2.0 - gamma) / alpha,
        (beta - 2.0) / alpha,
    )


def _z_and_complement(profile: DoublePowerLaw, r: Array) -> tuple[Array, Array, Array]:
    """Return z, its complement and where the complement is the smaller one.

    Past the knee, where ``r**alpha > 1``, z rounds to 1 in double precision.
    The complement ``1 / (1 + r**alpha)`` is formed directly there, and the
    incomplete Beta functions are evaluated in it, without cancellation.
    """
    r_alpha = r**profile.rho_alpha
    return r_alpha / (1.0 + r_alpha), 1.0 / (1.0 + r_alpha), r_alpha > 1.0


def _mass_general(profile: DoublePowerLaw, r: Array) -> Array:
    """Return the enclosed mass from the incomplete Beta function."""
    if profile.rho_beta > RHO_BETA_FINITE_MASS:
        a_1, b_1, _, _ = _exponents(profile)
        z, complement, outer = _z_and_complement(profile, r)
        fraction = np.where(
            outer, 1.0 - betainc(b_1, a_1, complement), betainc(a_1, b_1, z)
        )
        return (
            (profile.rho_norm / profile.rho_alpha) * fraction * beta_function(a_1, b_1)
        )
    if profile.rho_beta == RHO_BETA_FINITE_MASS:
        return _mass_beta3(profile, r)
    return _mass_or_p_mpmath(profile, r, "mass")


def _mass_beta3(profile: DoublePowerLaw, r: Array) -> Array:
    """Return the enclosed mass for an outer slope of exactly 3.

    The second argument of the mass's Beta function vanishes there, which
    the incomplete Beta function of SciPy does not take. Inside the knee the
    incomplete Beta function is its hypergeometric series; past it, in the
    complement, it is a logarithm less a series whose remainder bound sets
    where the series stops.
    """
    z, complement, outer = _z_and_complement(profile, r)
    a_1, _, _, _ = _exponents(profile)
    value = np.empty_like(z)
    inside = ~outer
    if inside.any():
        z_in = z[inside]
        value[inside] = z_in**a_1 / a_1 * hyp2f1(a_1, 1.0, a_1 + 1.0, z_in)
    if outer.any():
        e = complement[outer]
        base = -np.log(e) - digamma(a_1) - np.euler_gamma
        series = np.zeros_like(e)
        coefficient, k = 1.0, 0
        while True:
            k += 1
            coefficient *= (a_1 - k) / k
            series = series + coefficient * (-e) ** k / k
            coefficient_next = coefficient * (a_1 - k - 1) / (k + 1)
            # the coefficients do not grow from k = a_1 - 1 on, so the tail
            # after term k is bounded by the next term over 1 - e
            bound = abs(coefficient_next) * e ** (k + 1) / ((k + 1) * (1.0 - e))
            if k >= a_1 - 1.0 and np.all(bound <= 1e-17 * np.abs(base - series)):
                break
            if k > 400:
                raise FloatingPointError(
                    "The series of the mass for an outer slope of 3 did not converge."
                )
        value[outer] = base - series
    return (profile.rho_norm / profile.rho_alpha) * value


def _p_general(profile: DoublePowerLaw, r: Array) -> Array:
    """Return the relative potential from the incomplete Beta function."""
    if profile.rho_beta >= RHO_BETA_FINITE_MASS:
        _, _, a_2, b_2 = _exponents(profile)
        z, complement, outer = _z_and_complement(profile, r)
        mass_term = _mass_general(profile, r) / r
        # The complemented incomplete Beta function is evaluated natively on
        # both sides of the knee: forming it as 1 - I_z cancels at large
        # radii, and I_(1-z) cancels at small radii.
        outer_term = (
            (profile.rho_norm / profile.rho_alpha)
            * beta_function(a_2, b_2)
            * np.where(outer, betainc(b_2, a_2, complement), betaincc(a_2, b_2, z))
        )
        return mass_term + outer_term
    return _mass_or_p_mpmath(profile, r, "p")


def _mass_or_p_mpmath(profile: DoublePowerLaw, r: Array, what: str) -> Array:
    """Return the mass or the potential in arbitrary precision.

    This is the general path for an outer slope below 3, where the second
    argument of the mass's Beta function is negative, and the reference of
    the tests for every profile.
    """
    import mpmath as mp

    with mp.workdps(MP_DIGITS):
        alpha, beta, gamma = map(mp.mpf, profile._key)
        norm = (
            alpha
            * mp.gamma((beta - gamma) / alpha)
            / (mp.gamma((beta - 2) / alpha) * mp.gamma((2 - gamma) / alpha))
        )
        a_1, b_1 = (3 - gamma) / alpha, (beta - 3) / alpha
        a_2, b_2 = (2 - gamma) / alpha, (beta - 2) / alpha
        out = np.empty(len(r))
        for i, r_float in enumerate(r):
            r_i = mp.mpf(float(r_float))
            r_alpha = r_i**alpha
            complement = 1 / (1 + r_alpha)
            z = r_alpha / (1 + r_alpha)
            if complement >= mp.mpf(MP_COMPLEMENT_MIN):
                mass = (norm / alpha) * mp.betainc(a_1, b_1, 0, z, regularized=False)
            elif b_1 != 0:
                # Both signs of b_1: for an outer slope below 3 the Beta
                # functions are their analytic continuations, and the two
                # terms are large and of opposite sign in the working
                # precision of this path.
                mass = (norm / alpha) * (
                    mp.beta(a_1, b_1)
                    - mp.betainc(b_1, a_1, 0, complement, regularized=False)
                )
            else:
                # Outer slope 3: the logarithm and the first terms of its
                # series in the complement, whose next term is negligible
                # where this branch is used.
                mass = (norm / alpha) * (
                    -mp.log(complement)
                    - mp.digamma(a_1)
                    - mp.euler
                    + (a_1 - 1) * complement
                    - (a_1 - 1) * (a_1 - 2) * complement**2 / 4
                    + (a_1 - 1) * (a_1 - 2) * (a_1 - 3) * complement**3 / 18
                )
            if what == "mass":
                out[i] = float(mass)
                continue
            outer_term = (norm / alpha) * mp.betainc(
                b_2, a_2, 0, complement, regularized=False
            )
            out[i] = float(mass / r_i + outer_term)
    return out


def _p_closed(profile: DoublePowerLaw, r: Array) -> Array | None:
    """Return the closed-form potential of a named profile, or None."""
    alpha, beta, gamma = key = profile._key
    small = r < SERIES_R_MAX
    rs = np.where(small, r, 0.0)  # the series are formed where they serve
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == (1.0, 3.0, 1.0):
            series = sum((-1.0) ** k * rs**k / (k + 1.0) for k in range(18))
            return np.where(small, series, np.log1p(r) / r)
        if key == (1.0, 3.0, 0.0):
            series = 2.0 * sum(
                (-1.0) ** k * rs**k / (k + 1.0) for k in range(18)
            ) - 1.0 / (1.0 + rs)
            return np.where(small, series, 2.0 * np.log1p(r) / r - 1.0 / (1.0 + r))
        if key in ((1.0, 3.0, 0.5), (1.0, 3.0, 1.5)):
            # These closed forms cancel at both ends; arbitrary precision there.
            s = 1.0 + 2.0 * r + 2.0 * np.sqrt(r + r * r)
            if gamma == 0.5:
                value = (
                    1.0 - (3.0 + r) / np.sqrt(r + r * r) + 3.0 / (2.0 * r) * np.log(s)
                )
            else:
                value = 1.0 - np.sqrt((1.0 + r) / r) + np.log(s) / (2.0 * r)
            bad = (r < SERIES_R_MAX) | (r > 1e3)
            if bad.any():
                value = value.copy()
                value[bad] = _mass_or_p_mpmath(profile, r[bad], "p")
            return value
        if key == (2.0, 3.0, 0.0):
            series = (
                1.0
                - rs**2 / 6.0
                + 3.0 * rs**4 / 40.0
                - 5.0 * rs**6 / 112.0
                + 35.0 * rs**8 / 1152.0
            )
            return np.where(small, series, np.arcsinh(r) / r)
        if key == (2.0, 3.0, 1.0):
            return 2.0 / np.pi * np.arctan(1.0 / r) + np.log1p(r * r) / (np.pi * r)
        if alpha == 1.0 and beta == 4.0:
            # 1 - (r / (1 + r))**(2 - gamma), with no cancellation at large r.
            return -np.expm1(-(2.0 - gamma) * np.log1p(1.0 / r))
        if key == (2.0, 5.0, 0.0):
            return 1.0 / np.sqrt(1.0 + r * r)
    return None


def _mass_closed(profile: DoublePowerLaw, r: Array) -> Array | None:
    """Return the closed-form enclosed mass of a named profile, or None."""
    alpha, beta, gamma = key = profile._key
    small = r < SERIES_R_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == (1.0, 3.0, 1.0):
            rs = np.where(small, r, 0.0)  # the series is formed where it serves
            series = sum((-1.0) ** k * (k - 1.0) * rs**k / k for k in range(2, 22))
            return np.where(small, series, np.log1p(r) - r / (1.0 + r))
        if key == (1.0, 3.0, 0.0):
            # fmt: off
            coefficients = [0, 0, 0, 2/3, -3/2, 12/5, -10/3, 30/7, -21/4, 56/9,
                            -36/5, 90/11, -55/6, 132/13]
            # fmt: on
            rs = np.where(r < 0.02, r, 0.0)
            series = sum(c * rs**k for k, c in enumerate(coefficients))
            closed = 2.0 * (np.log1p(r) + 2.0 / (1.0 + r) - 0.5 / (1.0 + r) ** 2 - 1.5)
            return np.where(r < 0.02, series, closed)
        if key == (2.0, 3.0, 0.0):
            # fmt: off
            coefficients = [0, 0, 0, 1/3, 0, -3/10, 0, 15/56, 0, -35/144, 0,
                            315/1408, 0, -693/3328]
            # fmt: on
            rs = np.where(r < 0.05, r, 0.0)
            series = sum(c * rs**k for k, c in enumerate(coefficients))
            return np.where(r < 0.05, series, np.arcsinh(r) - r / np.sqrt(1.0 + r * r))
        if key in ((1.0, 3.0, 0.5), (1.0, 3.0, 1.5)):
            if gamma == 0.5:
                value = (
                    3.0 * np.arccosh(np.sqrt(1.0 + r))
                    - 4.0 * np.sqrt((1.0 + r) / r)
                    + 5.0 / np.sqrt(r * (1.0 + r))
                    - 1.0 / (np.sqrt(r) * (1.0 + r) ** 1.5)
                )
            else:
                value = np.arccosh(np.sqrt(1.0 + r)) - np.sqrt(r / (1.0 + r))
            bad = r < 5e-2
            if bad.any():
                value = value.copy()
                value[bad] = _mass_or_p_mpmath(profile, r[bad], "mass")
            return value
        if key == (2.0, 3.0, 1.0):
            return np.log1p(r * r) / np.pi
        if alpha == 1.0 and beta == 4.0:
            return (2.0 - gamma) * (r / (1.0 + r)) ** (3.0 - gamma)
        if key == (2.0, 5.0, 0.0):
            return r**3 / (1.0 + r * r) ** 1.5
    return None


def _flatten_profile(p: DoublePowerLaw):
    # The exponents are configuration, not fit parameters, in the first
    # release: the profile is a pytree node with no leaves, so that a model
    # holding it can be traced in its other parameters.
    return (), (p.rho_alpha, p.rho_beta, p.rho_gamma)


def _unflatten_profile(aux, children) -> DoublePowerLaw:
    return DoublePowerLaw(*aux)


jax.tree_util.register_pytree_node(DoublePowerLaw, _flatten_profile, _unflatten_profile)
