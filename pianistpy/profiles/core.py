r"""The double-power-law density profile and its parameter sets that have names.

`DoublePowerLaw` evaluates the density, the enclosed mass and the relative
potential of the profile in closed form, in internal units. The profiles
with names (NFW, Hernquist, Plummer and others) are parameter sets of the
double-power-law profile. The formulas are in Chiang+(2026d), Appendix A.
"""

__all__ = ["DoublePowerLaw"]

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache, lru_cache
from typing import Any

import jax.tree_util
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betainc, betaincc, digamma, hyp2f1
from scipy.special import gamma as gamma_function

from ..exceptions import InputError
from ..typing import Array, ArrayLike

# The admissible slopes: rho_alpha >= RHO_ALPHA_MIN, rho_beta > RHO_BETA_MIN
# (a finite potential at large radii) and rho_gamma < RHO_GAMMA_MAX (a finite
# central potential).
RHO_ALPHA_MIN = 0.5
RHO_BETA_MIN = 2.0
RHO_GAMMA_MAX = 2.0
# The total mass is finite for an outer slope rho_beta above
# RHO_BETA_FINITE_MASS and infinite otherwise.
RHO_BETA_FINITE_MASS = 3.0
# Below the radius SERIES_R_MAX, the closed forms of the profiles with names lose
# digits to cancellation in double precision, and the power series are used.
# At R = SERIES_R_MAX the first omitted term of each potential series and of
# the NFW mass series is below 1e-21 of the series' leading term, so the
# series hold double precision and serve the 1e-6 bound.
SERIES_R_MAX = 1e-2
# Working precision, in decimal digits, of the arbitrary-precision mass and
# potential. Thirty digits, against the 16 of double precision, leave 14
# digits for the cancellation between the two large terms of opposite sign
# at an outer slope below 3; the values serve as references for the 1e-6
# bound.
MP_DIGITS = 30
# Where the complement 1 / (1 + R**rho_alpha) is below MP_COMPLEMENT_MIN, the
# arbitrary-precision mass is evaluated from the incomplete Beta function in
# the complement instead of the incomplete Beta function in z.
MP_COMPLEMENT_MIN = "1e-10"
# Number of (profile, quantity, radius) values the cache of `_mp_point` keeps.
MP_MEMO_SIZE = 1 << 18

# The names of the parameter sets (rho_alpha, rho_beta, rho_gamma).
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
    r"""Double-power-law density profile, in internal units.

    The profile is the input density of a `Component`. Its methods return
    the density, the enclosed mass and the relative potential at
    user-specified 3D radii, in closed form. The class methods `nfw`,
    `hernquist` and `plummer` return three parameter sets that have names.

    Parameters
    ----------
    rho_alpha : float
        Sharpness :math:`\alpha` of the transition between the inner and the
        outer slope. Must be at least 1/2.
    rho_beta : float
        Outer logarithmic slope :math:`\beta`. Must be greater than 2. A
        `Component` takes :math:`\beta \geq 3` only.
    rho_gamma : float
        Inner logarithmic slope :math:`\gamma`. Must satisfy
        :math:`0 \leq \gamma < 2`.

    Raises
    ------
    InputError
        If :math:`\alpha < 1/2`, :math:`\beta \leq 2`, :math:`\gamma < 0` or
        :math:`\gamma \geq 2`.

    See Also
    --------
    Component : A profile with its anisotropy, the input of a model.
    TruncatedProfile : The self-consistent profile of a model.

    Notes
    -----
    In internal units the scale radius :math:`r_0` is 1, the central
    relative potential :math:`P(0)` is 1, and :math:`P(R)` vanishes as
    :math:`R \to \infty`. The density is

    .. math::

        \varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha},

    with :math:`R` the radius in units of :math:`r_0` and :math:`C` the
    normalization `rho_norm`. The bound :math:`\gamma \geq 0` excludes a
    density that decreases toward the center. The bound :math:`\gamma < 2`
    keeps the central potential finite. The bound :math:`\beta > 2` keeps
    the integral of :math:`\varrho R` over :math:`R` from 0 to infinity
    finite, so that :math:`P(0)` is finite.

    The profiles with names are parameter sets of the double-power-law
    profile. Where a profile with a name has a closed form for the profile's mass or
    potential in elementary functions, `mass` and `p` evaluate the
    elementary closed form.

    References
    ----------
    Chiang+(2026d), Eq. (5), defines the profile. Chiang+(2026d), Eq. (A6),
    gives :math:`C`. Chiang+(2026d), Eq. (A7), gives the enclosed mass.

    Examples
    --------
    >>> import pianistpy as pp
    >>> pp.DoublePowerLaw.nfw()
    DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
    >>> pp.DoublePowerLaw.hernquist().mass_total
    1.0
    >>> float(pp.DoublePowerLaw.plummer().p(0.0))
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
        r"""Return the Navarro-Frenk-White profile.

        Returns
        -------
        profile : DoublePowerLaw
            The profile with
            :math:`(\alpha, \beta, \gamma) = (1, 3, 1)`.

        See Also
        --------
        hernquist : The Hernquist profile.
        plummer : The Plummer profile.

        Examples
        --------
        >>> import pianistpy as pp
        >>> pp.DoublePowerLaw.nfw()
        DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
        """
        return cls(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)

    @classmethod
    def hernquist(cls) -> "DoublePowerLaw":
        r"""Return the Hernquist profile.

        Returns
        -------
        profile : DoublePowerLaw
            The profile with
            :math:`(\alpha, \beta, \gamma) = (1, 4, 1)`.

        See Also
        --------
        nfw : The Navarro-Frenk-White profile.
        plummer : The Plummer profile.

        Examples
        --------
        >>> import pianistpy as pp
        >>> pp.DoublePowerLaw.hernquist()
        DoublePowerLaw(rho_alpha=1.0, rho_beta=4.0, rho_gamma=1.0)
        """
        return cls(rho_alpha=1.0, rho_beta=4.0, rho_gamma=1.0)

    @classmethod
    def plummer(cls) -> "DoublePowerLaw":
        r"""Return the Plummer profile.

        Returns
        -------
        profile : DoublePowerLaw
            The profile with
            :math:`(\alpha, \beta, \gamma) = (2, 5, 0)`.

        See Also
        --------
        nfw : The Navarro-Frenk-White profile.
        hernquist : The Hernquist profile.

        Examples
        --------
        >>> import pianistpy as pp
        >>> pp.DoublePowerLaw.plummer()
        DoublePowerLaw(rho_alpha=2.0, rho_beta=5.0, rho_gamma=0.0)
        """
        return cls(rho_alpha=2.0, rho_beta=5.0, rho_gamma=0.0)

    @property
    def _key(self) -> tuple[float, float, float]:
        """The slopes (rho_alpha, rho_beta, rho_gamma), the key of `NAMES`."""
        return (self.rho_alpha, self.rho_beta, self.rho_gamma)

    @property
    def name(self) -> str:
        """Name of the parameter set, or the three slopes if the set has no name."""
        return NAMES.get(
            self._key,
            f"double power law ({self.rho_alpha:g},{self.rho_beta:g},"
            f"{self.rho_gamma:g})",
        )

    @property
    def rho_norm(self) -> float:
        r"""Normalization :math:`C` of the density that sets :math:`P(0) = 1`.

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
        """Total mass in internal units, infinite for an outer slope of 3 or less."""
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
            User-specified 3D radii :math:`R`, in internal units (units of
            the scale radius :math:`r_0`).

        Returns
        -------
        rho : array
            Density :math:`\varrho(R)` in internal units, with the shape of
            `r`.

        See Also
        --------
        mass : The enclosed mass.
        p : The relative potential.

        Notes
        -----
        .. math::

            \varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha}.

        Examples
        --------
        >>> import pianistpy as pp
        >>> float(pp.DoublePowerLaw.hernquist().rho(1.0))
        0.25
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
            User-specified 3D radii :math:`R`, in internal units (units of
            the scale radius :math:`r_0`).

        Returns
        -------
        mass : array
            Enclosed mass :math:`\mathcal{M}(R)` in internal units, with the
            shape of `r`. The mass is the integral of
            :math:`\varrho R^2` over :math:`R` from 0 to `r`, that is,
            :math:`1/(4\pi)` times the integral of the density over the
            volume.

        See Also
        --------
        mass_total : The total mass.
        rho : The density.
        p : The relative potential.

        Notes
        -----
        With :math:`z = R^\alpha/(1+R^\alpha)` and the incomplete Beta
        function :math:`B(z; a, b)`,

        .. math::

            \mathcal{M}(R) = \frac{C}{\alpha}\,
                B\!\left(z;\,\frac{3-\gamma}{\alpha},\,\frac{\beta-3}{\alpha}\right) .

        For :math:`\beta = 3` the second argument of :math:`B` is zero, and
        the mass is evaluated from a logarithm and a series. For
        :math:`\beta < 3` the second argument is negative, and the analytic
        continuation of the formula is evaluated in arbitrary precision.

        References
        ----------
        Chiang+(2026d), Eq. (A7).

        Examples
        --------
        >>> import pianistpy as pp
        >>> float(pp.DoublePowerLaw.hernquist().mass(1.0))
        0.25
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
            User-specified 3D radii :math:`R`, in internal units (units of
            the scale radius :math:`r_0`).

        Returns
        -------
        p : array
            Relative potential :math:`P(R)` in internal units, with the shape
            of `r`. :math:`P(0) = 1`, and :math:`P(R)` vanishes as
            :math:`R \to \infty`.

        See Also
        --------
        depth_inner : The depth :math:`P(0) - P(R)` near the center.
        mass : The enclosed mass.
        rho : The density.

        Notes
        -----
        With :math:`z = R^\alpha/(1+R^\alpha)`, :math:`a_2 = (2-\gamma)/\alpha`,
        :math:`b_2 = (\beta-2)/\alpha`, the Beta function :math:`B(a, b)` and
        the incomplete Beta function :math:`B(z; a, b)`,

        .. math::

            P(R) = \frac{\mathcal{M}(R)}{R}
                 + \frac{C}{\alpha}\left[B(a_2, b_2) - B(z;\,a_2, b_2)\right] .

        References
        ----------
        Chiang+(2026d), Eq. (A3).

        Examples
        --------
        >>> import pianistpy as pp
        >>> float(pp.DoublePowerLaw.hernquist().p(1.0))
        0.5
        """
        shape = np.shape(r)
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        value = _p_closed(self, r)
        if value is None:
            value = _p_general(self, r)
        return np.reshape(value, shape)

    def depth_inner(self, r: ArrayLike, terms: int = 4) -> tuple[Array, Array]:
        r"""Return the depth :math:`u = P(0) - P(R)` near the center, from its series.

        The depth is summed term by term from its series in
        :math:`R^{\alpha}`, with no subtraction from :math:`P(0)`, so the
        depth keeps its relative precision where :math:`u` is small.
        `depth_inner` also returns the size of the series' next term
        relative to the depth, as an estimate of the truncation error of
        the sum.

        Parameters
        ----------
        r : array-like
            User-specified 3D radii :math:`R`, in internal units (units of
            the scale radius :math:`r_0`).
        terms : int, optional
            Number of terms of the series summed. Must be at least 1.
            Default: ``4``, which means the first four terms.

        Returns
        -------
        u : array
            Depth :math:`u` from the first `terms` terms, in internal units,
            with the shape of `r`. Infinite at :math:`R \geq 1`, where the
            series in :math:`R^{\alpha}` does not converge.
        bound : array
            Magnitude of the series' next term divided by `u`, with the shape
            of `r`. Infinite where `u` is not positive or not finite.

        See Also
        --------
        p : The relative potential in closed form.

        Notes
        -----
        The density term :math:`C\binom{e}{k}R^{-\gamma+k\alpha}`, with
        :math:`e = (\gamma-\beta)/\alpha`, contributes the term
        :math:`C\binom{e}{k}R^{2-\gamma+k\alpha}/[(2-\gamma+k\alpha)(3-\gamma+k\alpha)]`
        to the depth.

        References
        ----------
        Chiang+(2026d), Eq. (A11), gives the first two terms of the series.

        Examples
        --------
        >>> import pianistpy as pp
        >>> u, bound = pp.DoublePowerLaw.hernquist().depth_inner([1e-3, 2.0])
        >>> round(float(u[0]), 9)
        0.000999001
        >>> bool(bound[1] == float("inf"))
        True
        """
        r = np.asarray(r, dtype=np.float64)
        g, a = self.rho_gamma, self.rho_alpha
        e = (g - self.rho_beta) / a
        # The series in R**alpha converges for R < 1 only. At R >= 1 the
        # series is evaluated at the placeholder radius 0.5, and both the
        # returned depth and the returned bound are set to infinity.
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
        """Return the profile as a plain dictionary.

        Returns
        -------
        record : dict
            The key ``"profile"`` with the value ``"double_power_law"``, and
            the keys ``"rho_alpha"``, ``"rho_beta"`` and ``"rho_gamma"`` with
            the three slopes. `from_dict` builds the profile back from the
            dictionary.

        See Also
        --------
        from_dict : The profile from its dictionary.

        Examples
        --------
        >>> import pianistpy as pp
        >>> record = pp.DoublePowerLaw.hernquist().to_dict()
        >>> record["profile"], record["rho_beta"]
        ('double_power_law', 4.0)
        """
        return {"profile": "double_power_law", **asdict(self)}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DoublePowerLaw":
        """Return the profile that a dictionary from `to_dict` describes.

        Parameters
        ----------
        record : mapping
            The dictionary that `to_dict` returns. The keys ``"rho_alpha"``,
            ``"rho_beta"`` and ``"rho_gamma"`` are read; other keys are
            ignored.

        Returns
        -------
        profile : DoublePowerLaw
            The profile with the slopes of `record`.

        Raises
        ------
        InputError
            If a slope in `record` lies outside its admissible range (see
            `DoublePowerLaw`).

        See Also
        --------
        to_dict : The dictionary of a profile.

        Examples
        --------
        >>> import pianistpy as pp
        >>> record = {"rho_alpha": 1.0, "rho_beta": 3.0, "rho_gamma": 1.0}
        >>> pp.DoublePowerLaw.from_dict(record).name
        'NFW'
        """
        return cls(
            rho_alpha=float(record["rho_alpha"]),
            rho_beta=float(record["rho_beta"]),
            rho_gamma=float(record["rho_gamma"]),
        )


def _exponents(profile: DoublePowerLaw) -> tuple[float, float, float, float]:
    """Return the arguments (a_1, b_1, a_2, b_2) of the Beta functions.

    The mass's Beta function takes ``a_1 = (3 - gamma) / alpha`` and
    ``b_1 = (beta - 3) / alpha``; the potential's outer term takes
    ``a_2 = (2 - gamma) / alpha`` and ``b_2 = (beta - 2) / alpha``.
    """
    alpha, beta, gamma = profile._key
    return (
        (3.0 - gamma) / alpha,
        (beta - 3.0) / alpha,
        (2.0 - gamma) / alpha,
        (beta - 2.0) / alpha,
    )


def _z_and_complement(profile: DoublePowerLaw, r: Array) -> tuple[Array, Array, Array]:
    """Return z, the complement 1 - z, and the mask where R**alpha > 1.

    Here ``z = R**alpha / (1 + R**alpha)``. Where ``R**alpha > 1``, the
    complement formed as ``1 - z`` loses digits to cancellation, and all of
    them once ``R**alpha`` exceeds about :math:`10^{16}`. The complement is therefore
    formed directly as ``1 / (1 + R**alpha)``, and on the mask the callers
    evaluate the incomplete Beta functions in the complement.
    """
    r_alpha = r**profile.rho_alpha
    return r_alpha / (1.0 + r_alpha), 1.0 / (1.0 + r_alpha), r_alpha > 1.0


def _mass_general(profile: DoublePowerLaw, r: Array) -> Array:
    """Return the enclosed mass of any profile from the incomplete Beta function.

    For an outer slope above 3 the regularized incomplete Beta function of
    SciPy is used, in z where ``R**alpha <= 1`` and in the complement where
    ``R**alpha > 1``. An outer slope of exactly 3 goes to `_mass_beta3`, and an
    outer slope below 3 goes to the arbitrary-precision `_mass_or_p_mpmath`.
    """
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
    r"""Return the enclosed mass for an outer slope of exactly 3.

    At an outer slope of 3 the second argument b_1 of the mass's Beta
    function is zero, which the incomplete Beta function of SciPy does not
    take. Where ``R**alpha <= 1``, the incomplete Beta function is
    ``z**a_1 / a_1 * hyp2f1(a_1, 1, a_1 + 1, z)``. Where ``R**alpha > 1``, it
    is ``-ln(e) - digamma(a_1) - euler_gamma`` less a series in the
    complement e, and the series stops once a bound on the series' remainder
    is below :math:`10^{-17}` of the result.

    Raises
    ------
    FloatingPointError
        If the series in the complement has not met its stopping condition
        after 400 terms.
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
            # The coefficient binomial(a_1 - 1, k) does not grow in magnitude
            # for k >= a_1 - 1, so the sum of the terms after term k is at most
            # the magnitude of term k + 1 divided by 1 - e. The series stops
            # where that bound is below 1e-17 of the result, under the
            # double-precision rounding of 1.1e-16.
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
    """Return the relative potential of any profile from the incomplete Beta function.

    For an outer slope of 3 or more the potential is ``M(R) / R`` plus the
    outer term in the complemented regularized incomplete Beta function. An
    outer slope below 3 goes to the arbitrary-precision `_mass_or_p_mpmath`.
    """
    if profile.rho_beta >= RHO_BETA_FINITE_MASS:
        _, _, a_2, b_2 = _exponents(profile)
        z, complement, outer = _z_and_complement(profile, r)
        mass_term = _mass_general(profile, r) / r
        # The complemented function 1 - I_z(a_2, b_2) is evaluated by a SciPy
        # function on both sides of R**alpha = 1, never as a difference:
        # 1 - I_z would cancel at large radii, where I_z is close to 1, and
        # forming the complement 1 - z would cancel there as well. Where
        # R**alpha > 1 the function is I_e(b_2, a_2) in the complement e, and
        # where R**alpha <= 1 it is betaincc(a_2, b_2, z).
        outer_term = (
            (profile.rho_norm / profile.rho_alpha)
            * beta_function(a_2, b_2)
            * np.where(outer, betainc(b_2, a_2, complement), betaincc(a_2, b_2, z))
        )
        return mass_term + outer_term
    return _mass_or_p_mpmath(profile, r, "p")


def _mass_or_p_mpmath(profile: DoublePowerLaw, r: Array, what: str) -> Array:
    """Return the mass or the potential in arbitrary precision, per radius.

    ``what`` is ``"mass"`` or ``"p"``. `_mass_or_p_mpmath` evaluates the mass and
    the potential for an outer slope below 3, where the second argument b_1
    of the mass's Beta function is negative, and serves the tests as the
    reference for every profile. `_mp_point` caches each radius's value,
    because a bisection in radius on the potential evaluates the same radii
    repeatedly.
    """
    key = tuple(float(x) for x in profile._key)
    out = np.empty(len(r))
    for i, r_float in enumerate(r):
        out[i] = _mp_point(key, what, float(r_float))
    return out


@cache
def _mp_constants(key: tuple[float, float, float]) -> tuple[Any, ...]:
    """Return the constants of a profile's arbitrary-precision evaluation.

    The constants (alpha, C, a_1, b_1, a_2, b_2) are mpmath numbers formed
    with `MP_DIGITS` decimal digits, for the slopes ``key``.
    """
    import mpmath as mp

    with mp.workdps(MP_DIGITS):
        alpha, beta, gamma = map(mp.mpf, key)
        norm = (
            alpha
            * mp.gamma((beta - gamma) / alpha)
            / (mp.gamma((beta - 2) / alpha) * mp.gamma((2 - gamma) / alpha))
        )
        a_1, b_1 = (3 - gamma) / alpha, (beta - 3) / alpha
        a_2, b_2 = (2 - gamma) / alpha, (beta - 2) / alpha
    return alpha, norm, a_1, b_1, a_2, b_2


@lru_cache(maxsize=MP_MEMO_SIZE)
def _mp_point(key: tuple[float, float, float], what: str, r_float: float) -> float:
    """Return the mass or the potential at one radius in arbitrary precision.

    The value depends on the slopes ``key``, the quantity ``what`` (``"mass"``
    or ``"p"``) and the radius ``r_float`` alone, and is computed with
    `MP_DIGITS` decimal digits; the cache of size `MP_MEMO_SIZE` keeps it.
    """
    import mpmath as mp

    alpha, norm, a_1, b_1, a_2, b_2 = _mp_constants(key)
    with mp.workdps(MP_DIGITS):
        r_i = mp.mpf(r_float)
        r_alpha = r_i**alpha
        complement = 1 / (1 + r_alpha)
        z = r_alpha / (1 + r_alpha)
        if complement >= mp.mpf(MP_COMPLEMENT_MIN):
            mass = (norm / alpha) * mp.betainc(a_1, b_1, 0, z, regularized=False)
        elif b_1 != 0:
            # This branch holds for b_1 > 0 and b_1 < 0. For an outer slope
            # below 3 (b_1 < 0) the Beta functions are their analytic
            # continuations, and the two terms are large and of opposite
            # sign; MP_DIGITS digits keep double precision in their
            # difference.
            mass = (norm / alpha) * (
                mp.beta(a_1, b_1)
                - mp.betainc(b_1, a_1, 0, complement, regularized=False)
            )
        else:
            # Outer slope 3 (b_1 = 0): the logarithm and the first three
            # terms of its series in the complement. The complement is below
            # MP_COMPLEMENT_MIN = 1e-10 on this branch, so the omitted term is
            # of order complement**4, below about 1e-40.
            mass = (norm / alpha) * (
                -mp.log(complement)
                - mp.digamma(a_1)
                - mp.euler
                + (a_1 - 1) * complement
                - (a_1 - 1) * (a_1 - 2) * complement**2 / 4
                + (a_1 - 1) * (a_1 - 2) * (a_1 - 3) * complement**3 / 18
            )
        if what == "mass":
            return float(mass)
        outer_term = (norm / alpha) * mp.betainc(
            b_2, a_2, 0, complement, regularized=False
        )
        return float(mass / r_i + outer_term)


def _p_closed(profile: DoublePowerLaw, r: Array) -> Array | None:
    """Return the closed-form potential of a profile with a name, or None.

    None means the profile has no closed form here, and the caller uses
    `_p_general`.
    """
    alpha, beta, gamma = key = profile._key
    small = r < SERIES_R_MAX
    # The series are evaluated at R = 0 outside R < SERIES_R_MAX, where
    # np.where discards them, so no large power overflows.
    rs = np.where(small, r, 0.0)
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
            # The closed forms of Widrow II and Widrow IV lose digits to
            # cancellation at R < SERIES_R_MAX and at R > 1e3, and those radii
            # are evaluated in arbitrary precision.
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
            # The potential 1 - (R / (1 + R))**(2 - gamma), written with
            # expm1 and log1p so that no cancellation occurs at large R.
            return -np.expm1(-(2.0 - gamma) * np.log1p(1.0 / r))
        if key == (2.0, 5.0, 0.0):
            return 1.0 / np.sqrt(1.0 + r * r)
    return None


def _mass_closed(profile: DoublePowerLaw, r: Array) -> Array | None:
    """Return the closed-form enclosed mass of a profile with a name, or None.

    None means the profile has no closed form here, and the caller uses
    `_mass_general`.
    """
    alpha, beta, gamma = key = profile._key
    small = r < SERIES_R_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == (1.0, 3.0, 1.0):
            # The series is evaluated at R = 0 outside R < SERIES_R_MAX,
            # where np.where discards it.
            rs = np.where(small, r, 0.0)
            series = sum((-1.0) ** k * (k - 1.0) * rs**k / k for k in range(2, 22))
            return np.where(small, series, np.log1p(r) - r / (1.0 + r))
        if key == (1.0, 3.0, 0.0):
            # The series of the cored NFW mass, used at R < 0.02, where its
            # first omitted term is below 1e-17 of its leading term.
            # fmt: off
            coefficients = [0, 0, 0, 2/3, -3/2, 12/5, -10/3, 30/7, -21/4, 56/9,
                            -36/5, 90/11, -55/6, 132/13]
            # fmt: on
            rs = np.where(r < 0.02, r, 0.0)
            series = sum(c * rs**k for k, c in enumerate(coefficients))
            closed = 2.0 * (np.log1p(r) + 2.0 / (1.0 + r) - 0.5 / (1.0 + r) ** 2 - 1.5)
            return np.where(r < 0.02, series, closed)
        if key == (2.0, 3.0, 0.0):
            # The series of the Widrow V mass, used at R < 0.05, where its
            # first omitted term is below 2e-16 of its leading term.
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
            # The closed forms of the Widrow II and Widrow IV masses lose
            # digits to cancellation at R < 0.05, and those radii are
            # evaluated in arbitrary precision.
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
    """Return the JAX pytree form of a profile: no leaves, the slopes as data.

    The three slopes are fixed inputs, not parameters that JAX
    differentiates. The profile is therefore a pytree node with no leaves,
    so that a model holding the profile can be traced in its other
    parameters.
    """
    return (), (p.rho_alpha, p.rho_beta, p.rho_gamma)


def _unflatten_profile(aux, children) -> DoublePowerLaw:
    """Return the profile from its pytree form, the slopes in ``aux``."""
    return DoublePowerLaw(*aux)


jax.tree_util.register_pytree_node(DoublePowerLaw, _flatten_profile, _unflatten_profile)
