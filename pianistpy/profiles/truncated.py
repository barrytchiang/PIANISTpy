"""The self-consistent profile of a model, tabulated, and its evaluation."""

__all__ = ["ProfileTables", "TruncatedProfile"]

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..numerics import hermite, hermite_inverse
from ..numerics.pytrees import register_result
from ..typing import Array, ArrayLike
from .core import DoublePowerLaw


@dataclass(frozen=True)
class ProfileTables:
    r"""The arrays a kernel reads a `TruncatedProfile` from.

    They are the numbers `TruncatedProfile.potential`, `density`, `mass` and
    `radius` evaluate, laid out for a kernel that takes them as traced
    arguments: the cubic Hermite in the logarithms on the interior rows and
    the edge's closed forms beyond them.

    Parameters
    ----------
    log_r, log_p, log_p_slope : array, shape (n_interior,)
        ln r, ln P and d ln P / d ln r at the interior rows (every row but
        an edge row).
    log_r_all, log_mass, log_mass_slope : array, shape (n,)
        ln r, ln M and d ln M / d ln r at every row.
    log_rho_coefficients : array, shape (n_components, 4, n_interior - 1)
        The coefficients of the cubic spline of ln rho in ln r on each
        interval, highest power first, in the interval's own ln r offset.
    has_edge_row : bool
        Whether the profile carries its edge as its last row; the fields
        below are read only then.
    r_last, p_last : float
        Radius and potential of the last interior row.
    rho_last : array, shape (n_components,)
        Density of each component there.
    r_t, mass_t : float
        Truncation radius and total mass.
    edge_exponent : array, shape (n_components,)
        Exponent q of each component's density at the edge.
    """

    log_r: Array
    log_p: Array
    log_p_slope: Array
    log_r_all: Array
    log_mass: Array
    log_mass_slope: Array
    log_rho_coefficients: Array
    has_edge_row: bool
    r_last: float
    p_last: float
    rho_last: Array
    r_t: float
    mass_t: float
    edge_exponent: Array


@dataclass(frozen=True, eq=False)
class TruncatedProfile:
    r"""The self-consistent profile of a truncated model.

    It is the profile of a model, and it can be the profile of a component
    of another model: a component given the truncated profile of a parent
    model and a new anisotropy is Method II of Chiang+(2026d).

    Two profiles are equal where they belong to the same model: the identity
    of a model determines its profile.

    The profile evaluates itself: `potential`, `density`, `mass` and
    `radius` are the one reading of the rows that every stage of the
    package takes, in place of an interpolant of its own. Between the rows
    the potential and the mass are cubic Hermite interpolants in the
    logarithms with the exact slopes the rows state (the Poisson equation
    gives :math:`\mathrm{d}\ln\mathcal{P}/\mathrm{d}\ln r = -M/(r\mathcal{P})`
    and :math:`\mathrm{d}\ln M/\mathrm{d}\ln r = r^3\rho/M`); the density
    of each component is a cubic spline in the logarithms, since the rows
    state no density slope. On the last interval of a spatially finite
    profile whose last row is its edge (:math:`\mathcal{P} = 0` at `r_t`),
    the potential is the Kepler form :math:`M_t(1/r - 1/r_t)` and the
    density follows the edge law :math:`\rho \propto \mathcal{P}^q` with
    the stated `edge_exponent`; beyond `r_t` the potential is :math:`M_t/r`
    and the density zero. Inside the first row and, for a profile without
    an edge, beyond the last, the interpolants continue as the power laws
    of the end rows.

    Parameters
    ----------
    r : array, shape (n,)
        Radius, strictly increasing.
    p_trunc : array, shape (n,)
        Total relative potential of the truncated model.
    rho_trunc : array, shape (n_components, n)
        Density of each component.
    mass_trunc : array, shape (n,)
        Total enclosed mass, in the unit of `DoublePowerLaw.mass` (the
        Hernquist profile's total is 1); 4 pi times it is the integral of
        the density over the volume.
    k_t : float
        Truncation energy.
    r_t, mass_t : float
        Truncation radius and total mass; infinite where the model is not
        spatially finite.
    spatially_finite : bool
        Whether the density reaches zero at a finite radius with a finite
        total mass. A spatially finite profile whose last row has
        ``p_trunc == 0`` carries its edge as that row, and then the row's
        density is zero and its radius is `r_t`.
    source : DoublePowerLaw
        The analytic profile of the first component of the model: its scale
        radius and its central potential are the internal units the arrays
        are in.
    source_anisotropy : Anisotropy
        The anisotropy of that component in the model the profile belongs
        to. A component built on this profile (Method II) reads here what
        its parent was.
    model : str
        Identity string of the model the profile belongs to.
    model_level : int or None
        The tier level of that model's report (`ValidationReport.level`),
        written by `truncate`, so that a component built on this profile
        (Method II) takes it as its parents' tier. None where it was not
        recorded (a profile made otherwise, or read from a file written
        before it was): read as level 0.
    edge_exponent : tuple of float
        Exponent :math:`q` of each component's density at the edge of the
        model, :math:`\rho \propto \mathcal{P}^q`; empty where the model
        is not spatially finite and has no edge.
    u_trunc : array, shape (n,)
        The central potential less the potential, p_trunc[0] + u_trunc[0]
        less p_trunc, kept exact where p_trunc is close to it: near the
        center p_trunc alone has lost the digits of the difference that the
        DF's singular center is read in. Empty where not computed.
    central_deficit : tuple of tuple of float
        For each component, (D_0, D_1, ...): the central auxiliary density
        of the orbits the truncation removed as its Taylor series in the
        depth u, D_0 - D_1 u + D_2 u^2 - ... (D_0 to D_4 as made; a record
        of D_0 and D_1 alone reads as the linear deficit). Empty where not
        recorded.
    override_r, override_r_end : float
        Radii between which the closed-form profile replaced the rows of
        the solve; NaN where no row was replaced.
    df_k, df_log_f, df_slope : array, shape (n_components, m)
        The DF table of the model the profile belongs to: the energies,
        log10 F and its slope d log10 F / d ln K (NaN where not exact), as
        `truncate` made them. A component built on this profile (Method II)
        reads its sources from the integral of this DF. Empty where not
        recorded.
    df_taylor : tuple of tuple of float
        For each component, the model's Taylor edge (F_1, F_2, K_switch):
        F = F_1 K + F_2 K^2 / 2 below K_switch. Empty where not recorded.
    override_rows : int
        Number of rows replaced there.
    override_rule : str
        What replaced them, ``"none"`` where nothing did.
    origin : array of int8, shape (n,)
        Where each row comes from, a code of
        `pianistpy.df.truncation.ORIGINS` (the main solve, the edge
        continuation, the floor event, the fills, the closed-form outer rows,
        the edge row): the profile's checks read each row's tolerance from it.
        Empty where not recorded.
    """

    r: Array
    p_trunc: Array
    rho_trunc: Array
    mass_trunc: Array
    k_t: float
    r_t: float
    mass_t: float
    spatially_finite: bool
    source: DoublePowerLaw
    source_anisotropy: Anisotropy
    model: str
    edge_exponent: tuple[float, ...]
    u_trunc: Array = field(default_factory=lambda: np.zeros(0))
    central_deficit: tuple[tuple[float, ...], ...] = ()
    override_r: float = float("nan")
    override_r_end: float = float("nan")
    override_rows: int = 0
    override_rule: str = "none"
    origin: Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    df_k: Array = field(default_factory=lambda: np.zeros((0, 0)))
    df_log_f: Array = field(default_factory=lambda: np.zeros((0, 0)))
    df_slope: Array = field(default_factory=lambda: np.zeros((0, 0)))
    df_taylor: tuple[tuple[float, float, float], ...] = ()
    model_level: int | None = None

    def __post_init__(self) -> None:
        r = np.asarray(self.r, dtype=np.float64)
        if r.ndim != 1 or r.size < 2 or not np.all(np.diff(r) > 0.0) or not r[0] > 0.0:
            raise ResultError("The radii of a profile must be positive and increasing.")
        if np.shape(self.p_trunc) != r.shape or np.shape(self.mass_trunc) != r.shape:
            raise ResultError(
                "The potential and the mass must have one value per radius."
            )
        if np.ndim(self.rho_trunc) != 2 or np.shape(self.rho_trunc)[1] != r.size:
            raise ResultError(
                "The density must have one row per component and one value per radius."
            )
        if (
            self.spatially_finite
            and len(self.edge_exponent) != np.shape(self.rho_trunc)[0]
        ):
            raise ResultError(
                "A spatially finite profile states the edge exponent of every "
                "component."
            )
        if self.has_edge_row:
            if not np.all(np.asarray(self.rho_trunc)[:, -1] == 0.0):
                raise ResultError("The edge row of a profile has zero density.")
            if not float(r[-1]) == float(self.r_t):
                raise ResultError(
                    "The edge row of a profile lies at the truncation radius r_t."
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TruncatedProfile):
            return NotImplemented
        return self.model == other.model

    def __hash__(self) -> int:
        return hash(self.model)

    @property
    def has_edge_row(self) -> bool:
        """Whether the last row is the edge of a spatially finite profile."""
        return bool(self.spatially_finite) and float(self.p_trunc[-1]) == 0.0

    @property
    def r_table_max(self) -> float:
        """The radius of the last row: the outer end of the table, which is
        the edge of a spatially finite model."""
        return float(self.r[-1])

    @property
    def mass_at_r_table_max(self) -> float:
        """`mass_trunc` at the last row."""
        return float(self.mass_trunc[-1])

    # The rows the logarithmic interpolants are built on: every row but an
    # edge row, whose potential and density are zero.
    @cached_property
    def _n_interior(self) -> int:
        return int(self.r.size) - (1 if self.has_edge_row else 0)

    @cached_property
    def _log_r(self) -> Array:
        return np.log(np.asarray(self.r[: self._n_interior], dtype=np.float64))

    @cached_property
    def _log_p(self) -> Array:
        p = np.asarray(self.p_trunc[: self._n_interior], dtype=np.float64)
        if not np.all(p > 0.0):
            raise ResultError("The potential of a profile is positive inside its edge.")
        return np.log(p)

    @cached_property
    def _log_p_slope(self) -> Array:
        # d ln P / d ln r = -M / (r P), the Poisson equation at the rows
        n = self._n_interior
        return -np.asarray(self.mass_trunc[:n]) / (
            np.asarray(self.r[:n]) * np.asarray(self.p_trunc[:n])
        )

    @cached_property
    def _log_mass_slope(self) -> Array:
        # d ln M / d ln r = r^3 rho / M, with rho the total density
        r, m = (
            np.asarray(self.r, dtype=np.float64),
            np.asarray(self.mass_trunc, dtype=np.float64),
        )
        if not np.all(m > 0.0):
            raise ResultError(
                "The enclosed mass of a profile is positive at every row."
            )
        return r**3 * np.sum(np.asarray(self.rho_trunc, dtype=np.float64), axis=0) / m

    @cached_property
    def _log_rho_splines(self) -> tuple[CubicSpline, ...]:
        n = self._n_interior
        rho = np.asarray(self.rho_trunc[:, :n], dtype=np.float64)
        if not np.all(rho > 0.0):
            raise ResultError("The density of a profile is positive inside its edge.")
        return tuple(CubicSpline(self._log_r, np.log(row)) for row in rho)

    def interpolation_tables(self) -> ProfileTables:
        """Return the arrays a kernel reads this profile from.

        A kernel that evaluates them as `potential`, `density`, `mass` and
        `radius` do gives the same numbers; the arrays are traced arguments
        of the kernel, never closed over.
        """
        last = self._n_interior - 1
        finite = self.has_edge_row
        return ProfileTables(
            log_r=self._log_r,
            log_p=self._log_p,
            log_p_slope=self._log_p_slope,
            log_r_all=np.log(np.asarray(self.r, dtype=np.float64)),
            log_mass=np.log(np.asarray(self.mass_trunc, dtype=np.float64)),
            log_mass_slope=self._log_mass_slope,
            log_rho_coefficients=np.stack(
                [
                    np.asarray(spline.c, dtype=np.float64)
                    for spline in self._log_rho_splines
                ]
            ),
            has_edge_row=finite,
            r_last=float(self.r[last]),
            p_last=float(self.p_trunc[last]),
            rho_last=np.asarray(self.rho_trunc[:, last], dtype=np.float64),
            r_t=float(self.r_t) if finite else math.inf,
            mass_t=float(self.mass_t) if finite else math.inf,
            edge_exponent=np.asarray(self.edge_exponent, dtype=np.float64),
        )

    def _on_last_interval(self, r: Array) -> Array:
        """Where `r` lies on the Kepler interval of a profile with an edge row."""
        if not self.has_edge_row:
            return np.zeros(np.shape(r), dtype=bool)
        return r > float(self.r[self._n_interior - 1])

    def potential(self, r: ArrayLike) -> Array:
        """Return the relative potential at radii `r`."""
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p, _ = hermite(np.log(r), self._log_r, self._log_p, self._log_p_slope)
            inside = np.exp(log_p)
        if not self.has_edge_row:
            return inside
        kepler = self.mass_t * np.maximum(1.0 / r - 1.0 / self.r_t, 0.0)
        return np.where(self._on_last_interval(r), kepler, inside)

    def density(self, r: ArrayLike) -> Array:
        """Return the density of each component at radii `r`.

        The result has one row per component and one column per radius.
        """
        r = np.asarray(r, dtype=np.float64)
        log_r = np.log(r)
        inside = np.stack([np.exp(spline(log_r)) for spline in self._log_rho_splines])
        if not self.has_edge_row:
            return inside
        last = self._n_interior - 1
        p_last = float(self.p_trunc[last])
        p = self.potential(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            edge = np.stack(
                [
                    float(self.rho_trunc[i, last]) * (p / p_last) ** float(q)
                    for i, q in enumerate(self.edge_exponent)
                ]
            )
        edge = np.where(p > 0.0, edge, 0.0)
        return np.where(self._on_last_interval(r)[np.newaxis, :], edge, inside)

    def mass(self, r: ArrayLike) -> Array:
        """Return the total enclosed mass at radii `r`."""
        r = np.asarray(r, dtype=np.float64)
        log_mass, _ = hermite(
            np.log(r),
            np.log(np.asarray(self.r, dtype=np.float64)),
            np.log(np.asarray(self.mass_trunc, dtype=np.float64)),
            self._log_mass_slope,
        )
        inside = np.exp(log_mass)
        if not self.spatially_finite:
            return inside
        return np.where(r >= self.r_t, self.mass_t, inside)

    def potential_log_slope(self, r: ArrayLike) -> Array:
        r"""Return :math:`d\ln P / d\ln r` at radii `r`, from the profile's own
        representation.

        The slope is exact in the rows' quantities: :math:`dP/dr = -M/r^2`
        (the Poisson equation in the profile's unit), so
        :math:`d\ln P/d\ln r = -M/(r P)` with `mass` and `potential` read as
        every consumer reads them. The knots of the representation are the
        rows `r`; between two rows the slope is smooth.
        """
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -self.mass(r) / (r * self.potential(r))

    def potential_log_slope_derivative(self, r: ArrayLike) -> Array:
        r"""Return :math:`d(d\ln P/d\ln r)/dr` at radii `r`, exact in the rows'
        quantities.

        With :math:`q = -M/(rP)`, :math:`dM/dr = r^2 \varrho` (the total
        density of the components) and :math:`dP/dr = -M/r^2`:
        :math:`dq/dr = -(r^3 \varrho P - M P + M^2/r) / (r^2 P^2)`. Beyond the
        edge the density is zero and the same form holds on the Kepler
        potential.
        """
        r = np.asarray(r, dtype=np.float64)
        m, p = self.mass(r), self.potential(r)
        rho = np.sum(self.density(r), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -(r**3 * rho * p - m * p + m * m / r) / (r * r * p * p)

    def radius(self, p: ArrayLike) -> Array:
        """Return the radii at which the potential takes the values `p`.

        Values above the central potential are refused; a value of zero or
        below is the truncation radius of a spatially finite profile.
        """
        p = np.asarray(p, dtype=np.float64)
        if np.any(p > float(self.p_trunc[0])):
            raise InputError("A potential above the central value has no radius.")
        p_last = float(self.p_trunc[self._n_interior - 1])
        safe = np.where(p > 0.0, p, p_last)
        # -ln P increases with ln r: the inverse of the Hermite interpolant
        log_r, _ = hermite_inverse(
            -np.log(safe), self._log_r, -self._log_p, -self._log_p_slope
        )
        inside = np.exp(log_r)
        if not self.has_edge_row:
            return inside
        kepler = 1.0 / (1.0 / self.r_t + np.maximum(p, 0.0) / self.mass_t)
        return np.where(p < p_last, kepler, inside)

    def to_dict(self) -> dict[str, Any]:
        """Return what identifies the profile, as a plain dictionary.

        The arrays are not part of it: the identity of the model determines
        them, and a file holds them beside this record.
        """
        return {
            "profile": "truncated",
            "model": self.model,
            "source": self.source.to_dict(),
            "source_anisotropy": self.source_anisotropy.to_dict(),
        }


# The profile as a pytree: its rows and the numbers of its edge are the data;
# the source, the labels and the override record are static (the gradient
# contract).
register_result(
    TruncatedProfile,
    (
        "r",
        "p_trunc",
        "rho_trunc",
        "mass_trunc",
        "k_t",
        "r_t",
        "mass_t",
        "edge_exponent",
        "u_trunc",
        "central_deficit",
        "origin",
        "df_k",
        "df_log_f",
        "df_slope",
        "df_taylor",
    ),
)
register_result(
    ProfileTables,
    (
        "log_r",
        "log_p",
        "log_p_slope",
        "log_r_all",
        "log_mass",
        "log_mass_slope",
        "log_rho_coefficients",
        "r_last",
        "p_last",
        "rho_last",
        "r_t",
        "mass_t",
        "edge_exponent",
    ),
)
