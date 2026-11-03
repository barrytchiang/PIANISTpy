r"""The round trip of a distribution function through the package's own machinery.

`round_trip` takes an untruncated distribution function (DF) on an analytic
profile, from `compute_df` or from `DistributionFunction.from_function`,
and returns a `RoundTrip`: the density that the DF gives, the potential
that this density generates, and the relative difference against the
analytic profile at every radius. The three steps are the package's own:
the density by the quadrature of the DF that the validation reads
(`density_from_df`), the potential by Poisson's equation, the comparison
against the analytic density. The method is the one that made Fig. 8 of
Chiang+(2026d).

With :math:`P_{\rm ref}(R)`, :math:`\varrho_{\rm ref}(R)` and
:math:`\mathcal{M}_{\rm ref}(R)` the analytic profile's relative potential,
density and enclosed mass, and :math:`\varrho_F(P, R)` the density that the
DF gives at a potential and a radius, the closure

.. math::

    c(P) = \frac{\varrho_F(P, R_{\rm ref}(P))}
                {\varrho_{\rm ref}(R_{\rm ref}(P))} - 1,

with :math:`R_{\rm ref}(P)` the radius at which the analytic potential
equals :math:`P`, is tabulated at nodes of the depth
:math:`u = P_0 - P` inside and of the potential outside, and interpolated
by cubic splines in :math:`\ln u` and :math:`\ln P`. The potential is
written as :math:`P(R) = P_{\rm ref}(R) + \delta(R)`, and Poisson's
equation for the residual :math:`\delta`,

.. math::

    \frac{1}{R^2}\frac{{\rm d}}{{\rm d}R}
    \Big(R^2 \frac{{\rm d}\delta}{{\rm d}R}\Big)
    = -\big[\varrho_F(P(R), R) - \varrho_{\rm ref}(R)\big],

is integrated outward in :math:`\ln R` from the starting radius, where
:math:`\delta = 0`, in the state :math:`(\delta, Y)` with
:math:`Y = R^2\,{\rm d}\delta/{\rm d}R = \mathcal{M}_{\rm ref} - \mathcal{M}`.
The source keeps its digits: with :math:`R' = R_{\rm ref}(P)` and
:math:`D = \ln \varrho_{\rm ref}(R') - \ln \varrho_{\rm ref}(R)`,

.. math::

    \frac{\varrho_F(P, R)}{\varrho_{\rm ref}(R)} - 1
    = {\rm expm1}(D) + c(P)\,e^{D},

and this quantity at a radius is the relative difference the result
reports. The residual form holds in float64 where a plain outward solve of
:math:`P` loses its digits beyond :math:`R \sim 10^4`. The solve ends where
the potential reaches the floor of the DF table (`K_FLOOR`), which happens
at a finite radius for a DF whose mass exceeds the profile's; the radii
beyond carry no value.

The potential is fixed at the center, :math:`\delta(0) = 0`, as the paper's
round trip fixes :math:`P(0) = 1`. A mass of the DF's density that differs
from the profile's by a fraction :math:`\epsilon` therefore leaves a
constant offset :math:`\delta_\infty \sim \epsilon` of the potential at
large radii, where :math:`P_{\rm ref} \to 0`; read at that offset
potential, the density differs from the analytic one by about
:math:`|{\rm d}\ln\varrho/{\rm d}\ln P|\,\delta_\infty / P_{\rm ref}(R)`,
which grows in proportion to :math:`R`. The relative difference of a DF
whose closure stays at the quadrature's floor inside therefore rises
outward as the radius grows, and the rise measures the DF's total mass
against the profile's; the result's `closure` holds the difference before
Poisson's equation, which does not rise.
"""

__all__ = ["RoundTrip", "round_trip"]

import dataclasses
import hashlib
import os
import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .._version import __version__
from ..exceptions import InputError, ResultError
from ..identity import identity_string
from ..io import StoredResult, load_result, register_reader, write_result
from ..profiles import DoublePowerLaw
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..units import Units
from .derivatives import _r_of_p
from .inputs import Component
from .lookup import prepare_lookup
from .poisson import cuddeford_prefactor
from .profile_checks import _log_aux, _model_type
from .results import DistributionFunction
from .truncation import K_FLOOR

ROUND_TRIP_KIND = "round_trip"
_LN10 = np.log(10.0)
# The depth table and the potential table both end at 0.7, so that the two
# tables overlap on P in [0.3, 0.7] and the switch between them at P = 0.5
# lies inside both.
_TABLE_TOP = 0.7
# The inner series of the depth is read with this many terms, where its
# next term is below this fraction of the depth.
_SERIES_TERMS = 12
_SERIES_BOUND = 1e-17
# The number of cells per table at whose midpoints the spline is checked
# against the direct quadrature (the table budget).
_BUDGET_CELLS = 24
# The second solve for the tolerance budget runs at this multiple of the
# relative tolerance.
_BUDGET_TOLERANCE_FACTOR = 100.0


@dataclasses.dataclass(frozen=True, eq=False)
class RoundTrip:
    r"""The round trip of a distribution function against its analytic profile.

    `round_trip` returns a `RoundTrip`. The arrays are in internal units,
    with the central relative potential :math:`P_0 = 1`; `radial_profiles`
    presents them in the unit system of `units`.

    Parameters
    ----------
    components : tuple of Component
        The component of the DF.
    settings : Settings
        Numerical settings of the round trip (the group ``round_trip``).
    identity : str
        Identity string of the result.
    df_identity : str
        Identity string of the `DistributionFunction` the round trip read.
    df_source : {"inversion", "function"}
        Origin of the DF table's values, as `DistributionFunction.source`.
    function_name : str or None
        The ``__name__`` of the DF function, as
        `DistributionFunction.function_name`.
    r : array
        Radii :math:`R` of the result, in internal units.
    potential : array
        The relative potential :math:`P(R)` that the density of the DF
        generates. NaN beyond the radius `r_floor`.
    delta : array
        The residual :math:`\delta(R) = P(R) - P_{\rm ref}(R)` of the
        potential against the analytic potential. NaN beyond `r_floor`.
    mass : array
        The enclosed mass :math:`\mathcal{M}(R)` of the density of the DF.
        NaN beyond `r_floor`.
    density : array
        The density :math:`\varrho_F(P(R), R)` that the DF gives at the
        potential it generates. NaN beyond `r_floor`.
    error : array
        The relative difference :math:`\varrho_F(P(R), R) /
        \varrho_{\rm ref}(R) - 1` against the analytic density, at every
        radius. NaN beyond `r_floor`.
    closure : array
        The closure :math:`c(P_{\rm ref}(R))`: the relative difference of
        the density of the DF against the analytic density at the ANALYTIC
        potential, before Poisson's equation.
    r_floor : float
        The radius at which the potential reached the floor of the DF table,
        in internal units; infinite where the potential stayed above the
        floor out to the last radius.
    budget_table : float
        The largest difference between the spline of the closure and the
        direct quadrature at the midpoints of sampled cells of the two
        tables: the part of `error` that the tables' interpolation can
        contribute.
    budget_tolerance : float
        The largest difference in `error` between the solve at the
        settings' relative tolerance and a solve at a hundred times that
        tolerance: the part of `error` that the integration can contribute.
    timing : dict of str to float
        Elapsed time of each step, in seconds.
    units : Units, optional
        Unit system in which `radial_profiles` presents the result (`to`).
        Default: ``Units()``, which means internal units.

    See Also
    --------
    round_trip : The function that returns a `RoundTrip`.
    DistributionFunction.evaluate : The DF at given values of K.

    References
    ----------
    Chiang+(2026d), Section 4.1 and Fig. 8.
    """

    components: tuple[Component, ...]
    settings: Settings
    identity: str
    df_identity: str
    df_source: str
    function_name: str | None
    r: Array
    potential: Array
    delta: Array
    mass: Array
    density: Array
    error: Array
    closure: Array
    r_floor: float
    budget_table: float
    budget_tolerance: float
    timing: dict[str, float]
    units: Units = Units()

    def radial_profiles(self) -> dict[str, Any]:
        """Return the radial profiles of the round trip in the unit system of `units`.

        Returns
        -------
        profiles : dict
            The keys ``r``, ``potential``, ``mass`` and ``density``, each
            converted to the unit system of `units`, and the keys ``error``
            and ``closure``, which are relative differences and have no
            unit. Each value has the shape of `r`.

        See Also
        --------
        RoundTrip.to : A round trip presented in another unit system.

        Examples
        --------
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.plummer()))
        >>> trip = pp.round_trip(df, radii=[0.5, 1.0, 2.0])
        >>> sorted(trip.radial_profiles())
        ['closure', 'density', 'error', 'mass', 'potential', 'r']
        """
        profile = self.components[0].unit_profile
        units = self.units
        return {
            "r": units.convert("length", self.r, profile),
            "potential": units.convert("potential", self.potential, profile),
            "mass": units.convert("mass", self.mass, profile),
            "density": units.convert("density", self.density, profile),
            "error": np.array(self.error, dtype=np.float64),
            "closure": np.array(self.closure, dtype=np.float64),
        }

    def to(
        self,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "RoundTrip":
        r"""Return the round trip presented in another unit system.

        The values of the result are not recomputed and stay in internal
        units. `radial_profiles` of the returned result presents them in
        the returned result's unit system.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            Target unit system: ``"internal"`` (:math:`G = \rho_0/C = r_0 =
            1`), ``"model"`` (:math:`G = M_0 = r_0 = 1`) or ``"physical"``
            (Astropy quantities).
        concentration : float, optional
            Concentration :math:`c`, positive. The mass unit :math:`M_0` is
            the mass of the untruncated profile inside :math:`c\,r_0`. The
            model and the physical systems need `concentration`. Default:
            ``None``, which means that no concentration is given.
        mass_0 : float or Quantity, optional
            The mass unit :math:`M_0` in solar masses, for the physical
            system. Default: ``None``.
        r_0 : float or Quantity, optional
            The scale radius :math:`r_0` in parsecs, for the physical
            system. Default: ``None``.
        r_vir : float or Quantity, optional
            The virial radius :math:`c\,r_0` in parsecs, in place of `r_0`,
            for the physical system. Default: ``None``.

        Returns
        -------
        trip : RoundTrip
            The round trip with unchanged values, with the attribute
            `units` set to the target unit system.

        Raises
        ------
        InputError
            If `system` is not one of the three unit systems, or the scales
            the system needs are missing or inconsistent.

        See Also
        --------
        DistributionFunction.to : The same presentation of a DF.

        Examples
        --------
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.plummer()))
        >>> trip = pp.round_trip(df, radii=[1.0])
        >>> trip.to("model", concentration=10.0).units.system
        'model'
        """
        units = Units.build(system, concentration, mass_0, r_0, r_vir)
        return dataclasses.replace(self, units=units)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the round trip to an HDF5 file.

        `pianistpy.read` reads the file back into a `RoundTrip` with the
        same identity and the same arrays. The file records the identity of
        the `DistributionFunction` as the round trip's parent.

        Parameters
        ----------
        path : str or path-like
            Path of the HDF5 file to write.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means that an existing file is kept and
            `InputError` is raised.

        Raises
        ------
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as written.

        Examples
        --------
        >>> import os, tempfile
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.plummer()))
        >>> trip = pp.round_trip(df, radii=[1.0])
        >>> path = os.path.join(tempfile.mkdtemp(), "trip.h5")
        >>> trip.write(path)
        >>> pp.read(path).identity == trip.identity
        True
        """
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in _ARRAY_FIELDS
        }
        record = {
            "components": [component.to_dict() for component in self.components],
            "settings": self.settings.to_dict(),
            "units": self.units.to_dict(),
            "timing": dict(self.timing),
            "df_identity": self.df_identity,
            "df_source": self.df_source,
            "function_name": self.function_name,
            "r_floor": float(self.r_floor),
            "budget_table": float(self.budget_table),
            "budget_tolerance": float(self.budget_tolerance),
        }
        write_result(
            path,
            ROUND_TRIP_KIND,
            self.identity,
            record,
            arrays,
            parents=(self.df_identity,),
            overwrite=overwrite,
        )


_ARRAY_FIELDS = ("r", "potential", "delta", "mass", "density", "error", "closure")


def _read_round_trip(source: StoredResult | str | os.PathLike) -> RoundTrip:
    """Return a round trip from a file or a stored result.

    Raises
    ------
    ResultError
        If the file holds a result of a kind other than a round trip.
    """
    stored = load_result(source)
    if stored.kind != ROUND_TRIP_KIND:
        raise ResultError(
            f"The file holds a result of kind {stored.kind!r}, not a round trip."
        )
    record = stored.record
    return RoundTrip(
        components=tuple(Component.from_dict(item) for item in record["components"]),
        settings=Settings.from_dict(record["settings"]),
        identity=stored.identity,
        df_identity=str(record["df_identity"]),
        df_source=str(record["df_source"]),
        function_name=record["function_name"],
        r_floor=float(record["r_floor"]),
        budget_table=float(record["budget_table"]),
        budget_tolerance=float(record["budget_tolerance"]),
        timing=dict(record["timing"]),
        units=Units.from_dict(record.get("units")),
        **{
            name: np.asarray(stored.arrays[name], dtype=np.float64)
            for name in _ARRAY_FIELDS
        },
    )


register_reader(ROUND_TRIP_KIND, _read_round_trip)


class _DensityOfDf:
    r"""The density that a DF gives, by the package's quadrature.

    The lookup is prepared once. `log_rho(r, phi, depth)` returns the natural
    logarithm of the density at each radius and potential, the depth
    :math:`P_0 - \Phi` given where the caller holds it exactly and NaN
    elsewhere (`profile_checks._log_aux`)."""

    def __init__(self, df: DistributionFunction) -> None:
        component = df.components[0]
        self.anisotropy = component.anisotropy
        self.arrays, self.kinds = prepare_lookup(df.table, df.form, 0)
        self.first = float(df.table.k[0, 0])
        bc = float(self.anisotropy.beta_c)
        model_type = _model_type(self.anisotropy)
        om = model_type in ("OM_I", "OM_II")
        self.n = 0.5 if om else 0.5 - bc
        self.log_prefactor = (
            np.log(4.0 * np.pi) if om else np.log(cuddeford_prefactor(bc))
        )

    def log_rho(self, r: Any, phi: Any, depth: Any) -> Array:
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        depth = np.atleast_1d(np.asarray(depth, dtype=np.float64))
        held = np.where(phi >= 0.5, 1.0 - phi, np.nan)
        held = np.where(np.isfinite(depth), depth, held)
        log_aux = _log_aux(phi, self.first, self.arrays, self.kinds, self.n, held)
        return self.log_prefactor + log_aux + np.log(self.anisotropy.factor(r))


def _depth(profile: DoublePowerLaw, r: Any) -> Array:
    """Return the depth ``P_0 - P(R)`` with its relative digits: the inner series
    where the series is converged, ``1 - P`` elsewhere."""
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    u_series, bound = profile.depth_inner(r, terms=_SERIES_TERMS)
    converged = np.isfinite(u_series) & (bound < _SERIES_BOUND)
    return np.where(converged, u_series, 1.0 - profile.p(r))


def _log_density_ratio(profile: DoublePowerLaw, r: float, e: float) -> float:
    r"""Return :math:`\ln \varrho(R e^e) - \ln \varrho(R)` with its digits.

    For the double power law the ratio is closed,
    :math:`-\gamma e + \frac{\gamma - \beta}{\alpha}
    \ln\frac{1 + R^\alpha e^{\alpha e}}{1 + R^\alpha}`, and the logarithm is
    formed from ``expm1`` and ``log1p``, so a small shift gives a small
    ratio exactly, where the difference of the two logarithms would be
    rounding."""
    alpha, beta, gamma = profile.rho_alpha, profile.rho_beta, profile.rho_gamma
    r_alpha = r**alpha
    return float(
        -gamma * e
        + (gamma - beta)
        / alpha
        * np.log1p(r_alpha * np.expm1(alpha * e) / (1.0 + r_alpha))
    )


def _dlnu_dlnr(profile: DoublePowerLaw, r: Array, u: Array) -> Array:
    """Return d ln u / d ln R = M(R) / (R u), from the mass."""
    return profile.mass(r) / (r * u)


def _radius_of_depth(profile: DoublePowerLaw, u: Array) -> Array:
    """Return the radii at which the depth of the analytic potential equals `u`,
    by Newton's method in ln R on the depth with its digits."""
    gamma = profile.rho_gamma
    c = profile.rho_norm
    # The leading term of the depth series,
    # u = C R^(2 - gamma) / ((2 - gamma)(3 - gamma)).
    ln_r = np.log(u * (2.0 - gamma) * (3.0 - gamma) / c) / (2.0 - gamma)
    target = np.log(u)
    for _ in range(200):
        r = np.exp(ln_r)
        u_r = _depth(profile, r)
        f = np.log(u_r) - target
        step = f / _dlnu_dlnr(profile, r, u_r)
        # Newton on a monotonic function, steps capped at one e-fold.
        step = np.clip(step, -1.0, 1.0)
        ln_r = ln_r - step
        if np.all(np.abs(step) < 1e-15):
            break
    return np.exp(ln_r)


def _radii_default(settings: Settings) -> Array:
    """Return the round trip's grid of radii from the settings."""
    group = settings.round_trip
    lo, hi = float(group["log10_r_min"]), float(group["log10_r_max"])
    n = int(round((hi - lo) * int(group["rows_per_decade"]))) + 1
    return 10.0 ** np.linspace(lo, hi, n)


def round_trip(
    df: DistributionFunction,
    radii: ArrayLike | None = None,
    settings: Settings | None = None,
) -> RoundTrip:
    r"""Return the round trip of a distribution function against its analytic profile.

    The round trip is the density that the DF gives, the potential that
    this density generates by Poisson's equation, and the relative
    difference of the density, at the potential it generates, against the
    analytic density, at every radius. The three steps run through the
    package's own machinery, the same for a DF from `compute_df` and for a
    DF from `DistributionFunction.from_function`, so a fitting function of
    the DF is compared with the inverted DF on equal terms. The module
    docstring of `pianistpy.df.round_trip` states the method.

    Parameters
    ----------
    df : DistributionFunction
        An untruncated DF of one component built on an analytic profile
        (a `DoublePowerLaw`), from `compute_df` or from
        `DistributionFunction.from_function`.
    radii : array-like, optional
        Radii :math:`R` at which the result is reported, in the unit system
        of `df` (`DistributionFunction.units`): internal units by default,
        in the physical system a float in parsecs or an Astropy quantity of
        any length unit. Every radius must lie at or beyond the starting
        radius of the integration (the setting ``log10_r_start``). Default:
        ``None``, which means the grid of the settings: from
        ``log10_r_min`` to ``log10_r_max`` with ``rows_per_decade`` radii
        per decade.
    settings : Settings, optional
        Numerical settings; the round trip reads the group ``round_trip``.
        Default: ``None``, which means the settings of `df`.

    Returns
    -------
    trip : RoundTrip
        The round trip: the potential, the residual of the potential, the
        enclosed mass, the density and the relative difference against the
        analytic density at each radius, the closure before Poisson's
        equation, the radius at which the potential reached the floor of
        the DF table, and the two budgets (the tables' interpolation and
        the integration's tolerance).

    Raises
    ------
    InputError
        If `df` is not a `DistributionFunction`, holds more than one
        component, is built on a `TruncatedProfile` (a Method II DF; the
        round trip compares against an analytic profile), or belongs to a
        Type II anisotropy model (an untruncated Type II DF has no density
        beyond the anisotropy radius, and the package's validation does not
        judge its round trip either).
    InputError
        If a radius of `radii` lies inside the starting radius of the
        integration or beyond the setting ``log10_r_max``.
    ResultError
        If the DF is negative somewhere on its table. The round trip reads
        a non-negative DF; `DistributionFunction.k_t_min` states where
        such a DF is negative.

    See Also
    --------
    DistributionFunction.evaluate : The DF at given values of K.
    DistributionFunction.from_function : A DF given as a function of
        K, for the round trip of a fitting function.
    compute_df : The DF computed by inversion of the density.

    References
    ----------
    Chiang+(2026d), Section 4.1 and Fig. 8.

    Examples
    --------
    The round trip of the package's own Plummer DF at three radii:

    >>> import pianistpy as pp
    >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.plummer()))
    >>> trip = pp.round_trip(df, radii=[0.1, 1.0, 10.0])
    >>> bool(np.all(np.abs(trip.error) < 1e-8))
    True
    """
    if not isinstance(df, DistributionFunction):
        raise InputError(
            "round_trip takes a DistributionFunction from compute_df or from "
            f"DistributionFunction.from_function; got {type(df).__name__}."
        )
    if len(df.components) != 1:
        raise InputError("This release runs the round trip of one component's DF.")
    component = df.components[0]
    if not isinstance(component.profile, DoublePowerLaw):
        raise InputError(
            "The DF is built on a TruncatedProfile (a Method II DF); the round "
            "trip compares against an analytic profile."
        )
    if np.isfinite(component.anisotropy.domain_edge):
        raise InputError(
            "The DF belongs to a Type II anisotropy model, whose untruncated DF "
            "has no density beyond the anisotropy radius; the round trip is not "
            "defined for it."
        )
    settings = df.settings if settings is None else settings
    group = settings.round_trip
    profile = component.unit_profile
    times: dict[str, float] = {}

    t_0 = time.perf_counter()
    try:
        density = _DensityOfDf(df)
    except ResultError as exc:
        raise ResultError(
            "The round trip reads a non-negative DF, and this DF is negative on "
            f"its table (k_t_min = {df.k_t_min!r}): {exc}"
        ) from exc
    r_start = 10.0 ** float(group["log10_r_start"])
    r_max = 10.0 ** float(group["log10_r_max"])
    if radii is None:
        r_out = _radii_default(settings)
        radii_digest = "default"
    else:
        r_given = _internal_length(df.units, radii, profile)
        if r_given.size == 0 or not np.all(np.isfinite(r_given)):
            raise InputError("round_trip needs at least one finite radius.")
        if np.any(r_given < r_start) or np.any(r_given > r_max):
            raise InputError(
                "Every radius of the round trip must lie between the starting "
                f"radius {r_start:.3e} and the outermost radius {r_max:.3e} of the "
                "settings (log10_r_start, log10_r_max); got radii from "
                f"{float(np.min(r_given)):.3e} to {float(np.max(r_given)):.3e}."
            )
        r_out = np.sort(r_given.ravel())
        radii_digest = hashlib.sha256(np.ascontiguousarray(r_out).tobytes()).hexdigest()

    # The closure tables: the depth table from the depth at the starting
    # radius to _TABLE_TOP, the potential table from the floor to _TABLE_TOP.
    u_start = float(_depth(profile, r_start)[0])
    n_in = (
        int(
            np.ceil(
                np.log10(_TABLE_TOP / u_start) * int(group["depth_nodes_per_decade"])
            )
        )
        + 1
    )
    u_nodes = np.logspace(np.log10(u_start), np.log10(_TABLE_TOP), n_in)
    n_out = (
        int(
            np.ceil(
                np.log10(_TABLE_TOP / K_FLOOR)
                * int(group["potential_nodes_per_decade"])
            )
        )
        + 1
    )
    p_nodes = np.logspace(np.log10(K_FLOOR), np.log10(_TABLE_TOP), n_out)

    def closure_at_depth(u: Array) -> Array:
        r = _radius_of_depth(profile, u)
        log_rho = density.log_rho(r, 1.0 - u, u)
        return np.expm1(log_rho - np.log(profile.rho(r)))

    def closure_at_potential(p: Array) -> Array:
        r = np.asarray(_r_of_p(profile, p), dtype=np.float64)
        log_rho = density.log_rho(r, p, np.full(p.shape, np.nan))
        return np.expm1(log_rho - np.log(profile.rho(r)))

    c_in = closure_at_depth(u_nodes)
    c_out = closure_at_potential(p_nodes)
    spline_in = CubicSpline(np.log(u_nodes), c_in)
    spline_out = CubicSpline(np.log(p_nodes), c_out)
    times["tables"] = time.perf_counter() - t_0

    # The table budget: the spline against the direct quadrature at the
    # midpoints of sampled cells.
    t_0 = time.perf_counter()
    i_in = np.linspace(0, n_in - 2, _BUDGET_CELLS).astype(int)
    u_mid = np.sqrt(u_nodes[i_in] * u_nodes[i_in + 1])
    i_out = np.linspace(0, n_out - 2, _BUDGET_CELLS).astype(int)
    p_mid = np.sqrt(p_nodes[i_out] * p_nodes[i_out + 1])
    budget_table = float(
        max(
            np.max(np.abs(spline_in(np.log(u_mid)) - closure_at_depth(u_mid))),
            np.max(np.abs(spline_out(np.log(p_mid)) - closure_at_potential(p_mid))),
        )
    )
    times["table_budget"] = time.perf_counter() - t_0

    def closure(p: float, u: float) -> float:
        """The closure at a potential, read from the depth table where the depth
        is at most 0.5."""
        if u <= 0.5:
            return float(spline_in(np.log(max(u, u_start))))
        return float(spline_out(np.log(p)))

    def shift(r: float, delta: float, u_ref: float, p_ref: float) -> float:
        """Return ln(R'/R) with P_ref(R') = P_ref(R) + delta, by Newton's method
        in the variable that keeps its digits (the depth where the reference
        potential is at least 1/2, the potential outside)."""
        inner = u_ref <= 0.5
        if inner:
            target = np.log1p(-delta / u_ref)

            def f(e: float) -> float:
                return float(np.log(_depth(profile, r * np.exp(e))[0] / u_ref)) - target

            def g(e: float) -> float:
                r_e = r * np.exp(e)
                return float(
                    _dlnu_dlnr(profile, np.array([r_e]), _depth(profile, r_e))[0]
                )

        else:
            target = np.log1p(delta / p_ref)

            def f(e: float) -> float:
                return float(np.log(profile.p(r * np.exp(e)) / p_ref)) - target

            def g(e: float) -> float:
                r_e = r * np.exp(e)
                return -float(profile.mass(r_e) / (r_e * profile.p(r_e)))

        e = target / g(0.0)
        if abs(target) < 1e-6:
            # First order: exact to a relative |target|, and target carries
            # its digits through log1p.
            return float(e)
        for _ in range(60):
            de = f(e) / g(e)
            e -= de
            if abs(de) <= 1e-14 * abs(e) + 1e-16:
                break
        return float(e)

    def source(r: float, delta: float) -> tuple[float, float, float]:
        """Return (rho_F / rho_ref - 1, P, u) at a radius and a residual."""
        u_ref = float(_depth(profile, r)[0])
        p_ref = float(profile.p(r))
        e = shift(r, delta, u_ref, p_ref)
        d = _log_density_ratio(profile, r, e)
        u, p = u_ref - delta, p_ref + delta
        c = closure(p, u)
        return float(np.expm1(d) + c * np.exp(d)), p, u

    def rhs(t: float, y: Any) -> list[float]:
        r = float(np.exp(t))
        rel, _, _ = source(r, float(y[0]))
        return [float(y[1]) / r, -(r**3) * float(profile.rho(r)) * rel]

    def at_floor(t: float, y: Any) -> float:
        _, p, _ = source(float(np.exp(t)), float(y[0]))
        return p - K_FLOOR

    at_floor.terminal = True  # type: ignore[attr-defined]
    at_floor.direction = -1  # type: ignore[attr-defined]

    def solve(rtol: float) -> tuple[Array, Array, Array, float]:
        """Return (error, delta, mass residual Y, r_floor) at the radii r_out."""
        t_eval = np.log(r_out)
        t_end = max(float(t_eval[-1]), np.log(r_start) + 1e-6)
        sol = solve_ivp(
            rhs,
            (np.log(r_start), t_end),
            [0.0, 0.0],
            method="DOP853",
            rtol=rtol,
            atol=1e-60,
            t_eval=t_eval,
            events=at_floor,
        )
        if sol.status < 0:
            raise ResultError(
                f"The integration of Poisson's equation failed: {sol.message}"
            )
        error = np.full(r_out.shape, np.nan)
        delta = np.full(r_out.shape, np.nan)
        y_mass = np.full(r_out.shape, np.nan)
        n = sol.t.size
        for i in range(n):
            error[i] = source(float(r_out[i]), float(sol.y[0, i]))[0]
        delta[:n] = sol.y[0]
        y_mass[:n] = sol.y[1]
        r_floor = float(np.exp(sol.t_events[0][0])) if sol.t_events[0].size else np.inf
        return error, delta, y_mass, r_floor

    t_0 = time.perf_counter()
    rtol = float(group["ode_rtol"])
    error, delta, y_mass, r_floor = solve(rtol)
    times["poisson"] = time.perf_counter() - t_0
    t_0 = time.perf_counter()
    error_coarse = solve(rtol * _BUDGET_TOLERANCE_FACTOR)[0]
    both = np.isfinite(error) & np.isfinite(error_coarse)
    budget_tolerance = (
        float(np.max(np.abs(error[both] - error_coarse[both])))
        if both.any()
        else np.nan
    )
    times["tolerance_budget"] = time.perf_counter() - t_0

    p_ref = np.asarray(profile.p(r_out), dtype=np.float64)
    u_ref = _depth(profile, r_out)
    rho_ref = np.asarray(profile.rho(r_out), dtype=np.float64)
    closure_out = np.array(
        [closure(float(p), float(u)) for p, u in zip(p_ref, u_ref, strict=True)]
    )
    potential = p_ref + delta
    mass = np.asarray(profile.mass(r_out), dtype=np.float64) - y_mass
    density_out = rho_ref * (1.0 + error)

    if radii is not None:
        # Report in the order the caller gave the radii.
        order = np.argsort(r_given.ravel(), kind="stable")
        back = np.empty_like(order)
        back[order] = np.arange(order.size)
        shape = np.shape(r_given)
        r_out, potential, delta, mass, density_out, error, closure_out = (
            np.reshape(a[back], shape)
            for a in (r_out, potential, delta, mass, density_out, error, closure_out)
        )

    identity = identity_string(
        {"round_trip_of": df.identity, "radii": radii_digest},
        settings.numerical(),
        __version__,
    )
    return RoundTrip(
        components=df.components,
        settings=settings,
        identity=identity,
        df_identity=df.identity,
        df_source=df.source,
        function_name=df.function_name,
        r=np.asarray(r_out, dtype=np.float64),
        potential=np.asarray(potential, dtype=np.float64),
        delta=np.asarray(delta, dtype=np.float64),
        mass=np.asarray(mass, dtype=np.float64),
        density=np.asarray(density_out, dtype=np.float64),
        error=np.asarray(error, dtype=np.float64),
        closure=np.asarray(closure_out, dtype=np.float64),
        r_floor=r_floor,
        budget_table=budget_table,
        budget_tolerance=budget_tolerance,
        timing=times,
        units=df.units,
    )


def _internal_length(units: Units, value: Any, profile: DoublePowerLaw) -> Array:
    """Return a length given in the result's unit system in internal units.

    In the physical system, `value` is a float in parsecs or an Astropy
    quantity of any length unit."""
    factor = units.convert("length", 1.0, profile)
    if hasattr(factor, "unit"):
        if hasattr(value, "unit"):
            value = value.to_value(factor.unit)
        factor = factor.value
    return np.asarray(value, dtype=np.float64) / float(factor)
