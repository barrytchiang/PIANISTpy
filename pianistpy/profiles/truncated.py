r"""The self-consistent profile of a model, tabulated at nodes, and its evaluation.

`TruncatedProfile` holds the relative potential, the density of each
component and the enclosed mass of a model at a set of radii, the nodes, and
evaluates them at any radius. `ProfileTables` holds the interpolants of
`TruncatedProfile` as arrays that a JAX function takes as arguments. The
evaluation is described in Notes Section T:1.3.
"""

__all__ = ["ProfileTables", "TruncatedProfile"]

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..numerics import hermite, hermite_inverse
from ..numerics.pytrees import register_result
from ..typing import Array, ArrayLike
from ._series import inner_density_series, inner_depth_series, inner_mass_series
from .core import DoublePowerLaw

# The Newton solve of the inner series' depth for the radius (`radius` inside
# the first node): at most this many steps, stopped when a step moves the
# radius by less than INNER_RADIUS_TOLERANCE relative.
INNER_RADIUS_STEPS = 50
INNER_RADIUS_TOLERANCE = 1e-15


@dataclass(frozen=True)
class _InnerSeries:
    """The continuation of a profile inside its first node (`TruncatedProfile`).

    Each component's density is ``amplitude`` times the inner density series
    of the source density less ``deficit`` times ``R ** -deficit_slope``;
    the enclosed mass and the depth below ``p_0`` are the series' first four
    terms integrated, with the amplitude ``total``, less the deficit's term
    integrated. All quantities are in internal units, the mass in the unit of
    `TruncatedProfile.mass_trunc`.
    """

    amplitude: Array
    total: float
    slopes: tuple[float, float, float]
    deficit: Array
    deficit_slope: float
    p_0: float

    def density(self, r: Array) -> Array:
        """Return each component's density at radii `r`, shape (n_components, m)."""
        a, b, g = self.slopes
        shape = inner_density_series(r, 1.0, g, a, b)
        missing = self.deficit[:, np.newaxis] * r[np.newaxis, :] ** (
            -self.deficit_slope
        )
        return self.amplitude[:, np.newaxis] * shape[np.newaxis, :] - missing

    def mass(self, r: Array) -> Array:
        """Return the enclosed mass at radii `r`."""
        a, b, g = self.slopes
        w = self.deficit_slope
        missing = float(np.sum(self.deficit)) * r ** (3.0 - w) / (3.0 - w)
        return inner_mass_series(r, self.total, g, a, b) - missing

    def depth(self, r: Array) -> Array:
        """Return the depth ``P_0 - P`` at radii `r`."""
        a, b, g = self.slopes
        w = self.deficit_slope
        missing = float(np.sum(self.deficit)) * r ** (2.0 - w) / ((3.0 - w) * (2.0 - w))
        return inner_depth_series(r, self.total, g, a, b) - missing


@dataclass(frozen=True)
class ProfileTables:
    r"""Arrays from which a JAX function evaluates a `TruncatedProfile`.

    The arrays are the numbers from which `TruncatedProfile.potential`,
    `TruncatedProfile.density`, `TruncatedProfile.mass` and
    `TruncatedProfile.radius` evaluate the profile, laid out so that a JAX
    function takes them as arguments and is differentiable in them. A node
    is one tabulated radius of the profile. An interior node is any node
    other than the edge node of a spatially finite profile.

    Parameters
    ----------
    log_r : array, shape (n_interior,)
        :math:`\ln R` at the interior nodes, with :math:`R` the radius in
        internal units.
    log_p : array, shape (n_interior,)
        :math:`\ln P` at the interior nodes, with :math:`P` the relative
        potential in internal units.
    log_p_slope : array, shape (n_interior,)
        :math:`\mathrm{d}\ln P/\mathrm{d}\ln R` at the interior nodes.
    log_r_all : array, shape (n,)
        :math:`\ln R` at every node.
    log_mass : array, shape (n,)
        :math:`\ln \mathcal{M}` at every node, with :math:`\mathcal{M}` the
        enclosed mass in internal units.
    log_mass_slope : array, shape (n,)
        :math:`\mathrm{d}\ln \mathcal{M}/\mathrm{d}\ln R` at every node.
    log_rho_coefficients : array, shape (n_components, 4, n_interior - 1)
        Coefficients of the cubic spline of :math:`\ln\varrho` in
        :math:`\ln R` for each component and each interval between interior
        nodes, highest power first, as polynomials in the offset of
        :math:`\ln R` from the interval's lower node.
    has_edge_row : bool
        Whether the last node of the profile is its edge node.
    r_last : float
        Radius of the last interior node, in internal units.
    p_last : float
        Relative potential at the last interior node, in internal units.
    rho_last : array, shape (n_components,)
        Density of each component at the last interior node, in internal
        units.
    r_t : float
        Truncation radius :math:`R_{\rm t}`, in internal units. Infinite
        where `has_edge_row` is False.
    mass_t : float
        Total mass :math:`\mathcal{M}_{\rm t}`, in internal units. Infinite
        where `has_edge_row` is False.
    edge_exponent : array, shape (n_components,)
        Exponent :math:`q` of each component's density at the edge,
        :math:`\varrho \propto P^q`. Empty where the model has no edge.
    inner_amplitude : array, shape (n_components,)
        Amplitude :math:`C` of each component's inner density series, read
        from the first node, with which the density continues inside the
        first node (`TruncatedProfile.density`).
    inner_total_amplitude : float
        Amplitude of the inner series of the total density, read from the
        enclosed mass at the first node, with which the enclosed mass and
        the potential continue inside the first node.
    inner_slopes : array, shape (3,)
        The slopes :math:`(\alpha, \beta, \gamma)` of the source density.
    inner_deficit : array, shape (n_components,)
        First-order central deficit :math:`\delta_1` of each component, the
        amplitude of the density :math:`\delta_1 R^{-w}` missing inside the
        first node (`TruncatedProfile`), in internal units; zero where no
        deficit is carried.
    inner_deficit_slope : float
        The exponent :math:`w = 2\beta_{\rm c}` of the missing density; zero
        where no deficit is carried.
    p_0 : float
        Central relative potential :math:`P_0`, in internal units: the first
        node's potential plus the depth at the first node.

    See Also
    --------
    TruncatedProfile.interpolation_tables : The arrays of a profile.
    TruncatedProfile : The profile the arrays evaluate.

    Examples
    --------
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.profiles import TruncatedProfile
    >>> hernquist = pp.DoublePowerLaw.hernquist()
    >>> r = np.geomspace(1e-3, 1e3, 601)
    >>> profile = TruncatedProfile(
    ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
    ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
    ...     spatially_finite=False, source=hernquist,
    ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
    ... )
    >>> tables = profile.interpolation_tables()
    >>> tables.log_r.shape, tables.log_rho_coefficients.shape
    ((601,), (1, 4, 600))
    >>> tables.has_edge_row
    False
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
    inner_amplitude: Array
    inner_total_amplitude: float
    inner_slopes: Array
    inner_deficit: Array
    inner_deficit_slope: float
    p_0: float


@dataclass(frozen=True, eq=False)
class TruncatedProfile:
    r"""Self-consistent profile of a model, tabulated at radii called nodes.

    The profile of a model is ``model.profile``. A component given the
    profile of a parent model and a new anisotropy is Method II of
    Chiang+(2026d), Appendix B2.2. The methods `potential`, `density`,
    `mass` and `radius` evaluate the profile between the nodes.

    Two profiles are equal if the identity strings `model` of the two
    profiles are equal, because the identity of a model determines its
    profile.

    Parameters
    ----------
    r : array, shape (n,)
        Radius :math:`R` of each node, in internal units. The radii are
        positive and strictly increasing, and there are at least two.
    p_trunc : array, shape (n,)
        Total relative potential :math:`P` of the model at each node, in
        internal units.
    rho_trunc : array, shape (n_components, n)
        Density :math:`\varrho` of each component at each node, in internal
        units.
    mass_trunc : array, shape (n,)
        Total enclosed mass :math:`\mathcal{M}` at each node, in internal
        units, the unit of `DoublePowerLaw.mass`, in which the total mass of
        the Hernquist profile is 1. :math:`4\pi` times `mass_trunc` is the
        integral of the density over the volume.
    k_t : float
        Truncation value :math:`\mathcal{K}_{\rm t}`, in internal units.
    r_t : float
        Truncation radius :math:`R_{\rm t}`, in internal units. Infinite
        where the model is not spatially finite.
    mass_t : float
        Total mass :math:`\mathcal{M}_{\rm t}`, in internal units. Infinite
        where the model is not spatially finite.
    spatially_finite : bool
        Whether the density reaches zero at a finite radius with a finite
        total mass. A spatially finite profile whose last node has
        ``p_trunc == 0`` holds its edge as that last node, the edge node.
        The edge node's density is zero and its radius is `r_t`.
    source : DoublePowerLaw
        The analytic profile of the first component of the model. The scale
        radius and the central potential of `source` define the internal
        units of the arrays.
    source_anisotropy : Anisotropy
        The anisotropy of the first component in the model the profile
        belongs to. A component built on the profile (Method II) reads its
        parent's anisotropy from `source_anisotropy`.
    model : str
        Identity string of the model the profile belongs to.
    edge_exponent : tuple of float
        Exponent :math:`q` of each component's density at the edge of the
        model, :math:`\varrho \propto P^q`. Empty where the model is not
        spatially finite.
    u_trunc : array, shape (n,), optional
        Depth :math:`u`, the central potential less the potential, at each
        node, in internal units. Near the center `p_trunc` has lost the
        digits of the depth, and `u_trunc` keeps them. Default: an empty
        array, which means the depth was not stored.
    central_deficit : tuple of tuple of float, optional
        For each component, the coefficients :math:`(D_0, D_1, \ldots, D_4)`
        of the Taylor series :math:`D_0 - D_1 u + D_2 u^2 - \ldots` in the
        depth :math:`u` of the central augmented density of the orbits that
        the truncation removed. A record that holds only :math:`D_0` and
        :math:`D_1` is the linear form of the deficit. Default: ``()``,
        which means the deficit was not recorded.
    override_r : float, optional
        Inner radius of the range in which a closed form of the profile
        replaced the solution of the Poisson equation, in internal units.
        Default: NaN, which means no node was replaced.
    override_r_end : float, optional
        Outer radius of the range in which a closed form replaced the
        solution of the Poisson equation, in internal units. Default: NaN,
        which means no node was replaced.
    override_rows : int, optional
        Number of nodes that a closed form replaced. Default: ``0``, which
        means no node was replaced.
    override_rule : str, optional
        Name of the closed form that replaced the nodes. Default:
        ``"none"``, which means no node was replaced.
    origin : array of int8, shape (n,), optional
        Source of each node, as a code from 0 to 6. The code is 0 for the
        solution of the Poisson equation, 1 for the continuation of the
        Poisson solution to the edge, 2 for the radius at which the
        potential reaches zero, 3 for a node filled in at the inner end, 4
        for a node filled in at the outer end, 5 for a closed-form outer
        node and 6 for the edge node. Default: an empty array, which means
        the sources were not recorded.
    df_k : array, shape (n_components, m), optional
        Values :math:`\mathcal{K}` of the DF table of the model the
        profile belongs to, in internal units. A component built on the
        profile (Method II) computes its own DF from the parent's DF table.
        Default: an empty array, which means the table was not recorded.
    df_log_f : array, shape (n_components, m), optional
        :math:`\log_{10} F` at the nodes `df_k`, with :math:`F` the DF in
        internal units. Default: an empty array, which means the table was
        not recorded.
    df_slope : array, shape (n_components, m), optional
        :math:`\mathrm{d}\log_{10} F/\mathrm{d}\ln\mathcal{K}` at the
        nodes `df_k`. Default: an empty array, which means the table was
        not recorded.
    df_taylor : tuple of tuple of float, optional
        For each component, the coefficients :math:`(F_1, F_2,
        \mathcal{K}_{\rm s})` of the model's DF near the edge:
        :math:`F = F_1 \mathcal{K} + F_2 \mathcal{K}^2/2` for
        :math:`\mathcal{K}` below :math:`\mathcal{K}_{\rm s}`. Default:
        ``()``, which means the coefficients were not recorded.
    model_level : int or None, optional
        Level of the validation report of the model the profile belongs to
        (`ValidationReport.level`), set by the call that made the model. A
        component built on the profile (Method II) takes `model_level` as
        the level of its parent. Default: ``None``, which means the level
        was not recorded and is read as level 0.

    Raises
    ------
    ResultError
        If the radii are fewer than two, not positive or not strictly
        increasing.
    ResultError
        If `p_trunc` or `mass_trunc` does not have one value per radius, or
        `rho_trunc` does not have the shape (n_components, n).
    ResultError
        If a spatially finite profile does not give the edge exponent of
        every component.
    ResultError
        If the edge node does not have zero density or does not lie at
        `r_t`.

    See Also
    --------
    truncate : The call that makes a model and its profile.
    Component : A component built on a profile (Method II).
    DoublePowerLaw : The analytic profile of a component.

    Notes
    -----
    Between the nodes, :math:`\ln P` and :math:`\ln \mathcal{M}` are cubic
    polynomials in :math:`\ln R` that match the values and the slopes at
    the two nodes of each interval. The slopes are exact, from the Poisson
    equation and the mass integral:

    .. math::

        \frac{\mathrm{d}\ln P}{\mathrm{d}\ln R} = -\frac{\mathcal{M}}{R P},
        \qquad
        \frac{\mathrm{d}\ln \mathcal{M}}{\mathrm{d}\ln R}
        = \frac{R^3\varrho}{\mathcal{M}},

    with :math:`\varrho` the total density. The logarithm of each
    component's density is a cubic spline in :math:`\ln R`. On the last
    interval of a profile with an edge node, the potential is the point-mass
    (Keplerian) form :math:`\mathcal{M}_{\rm t}(1/R - 1/R_{\rm t})`, and each
    component's density is :math:`\varrho_L (P/P_L)^q`, with
    :math:`\varrho_L` and :math:`P_L` the values at the last interior node
    and :math:`q` from `edge_exponent`. At and beyond :math:`R_{\rm t}`, the
    relative potential and the density are zero, and the enclosed mass is
    :math:`\mathcal{M}_{\rm t}`. Beyond the last node of a profile without
    an edge node, :math:`P` and :math:`\mathcal{M}` continue as the power
    laws of the last node.

    Inside the first node, the three columns continue as the inner series
    of the source density, the forms with which the Poisson solver laid the
    innermost nodes (Notes Section T:3.4), lowered by the first-order term
    of the central deficit where the model carries one: each component's
    density is :math:`C R^{-\gamma} (1 + R^{\alpha})^{(\gamma - \beta)/\alpha}
    - \delta_1 R^{-2\beta_{\rm c}}`, with :math:`\delta_1 = D_0 / (1 +
    \kappa)` the component's first-order central deficit (:math:`D_0` from
    `central_deficit`, :math:`\kappa` as `pianistpy.df.center.lowered_center`
    states it) on a cusp with :math:`2\beta_{\rm c} \le \gamma`, zero
    otherwise, and the amplitude :math:`C` read from the component's density
    at the first node; the enclosed mass and the depth :math:`P_0 - P` are
    the first four terms of the series integrated less the deficit's term
    integrated, with the total amplitude read from the enclosed mass at the
    first node; and :math:`P_0` is the first node's potential plus the depth
    at the first node, so that every column is continuous at the first node.
    The deficit's higher orders enter the amplitudes read at the first node
    and, relative to the density, fall inward as :math:`R^{2(\gamma -
    2\beta_{\rm c})}`; they are not carried.

    References
    ----------
    Notes Section T:1.3.

    Examples
    --------
    The example builds the untruncated Hernquist profile on 601 nodes
    directly; a model's profile is ``model.profile``.

    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.profiles import TruncatedProfile
    >>> hernquist = pp.DoublePowerLaw.hernquist()
    >>> r = np.geomspace(1e-3, 1e3, 601)
    >>> profile = TruncatedProfile(
    ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
    ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
    ...     spatially_finite=False, source=hernquist,
    ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
    ... )
    >>> round(float(profile.potential(1.5)), 6)
    0.4
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
        """Return whether two profiles belong to the same model."""
        if not isinstance(other, TruncatedProfile):
            return NotImplemented
        return self.model == other.model

    def __hash__(self) -> int:
        """Return the hash of the model's identity string."""
        return hash(self.model)

    @property
    def has_edge_row(self) -> bool:
        """Whether the last node is the edge node of a spatially finite profile."""
        return bool(self.spatially_finite) and float(self.p_trunc[-1]) == 0.0

    @property
    def r_table_max(self) -> float:
        """Radius of the last node, in internal units; `r_t` at an edge node."""
        return float(self.r[-1])

    @property
    def mass_at_r_table_max(self) -> float:
        """Enclosed mass `mass_trunc` at the last node, in internal units."""
        return float(self.mass_trunc[-1])

    # The interpolants in the logarithms are built on the interior nodes:
    # every node but an edge node, whose potential and density are zero and
    # have no logarithm.
    @cached_property
    def _n_interior(self) -> int:
        """Number of interior nodes."""
        return int(self.r.size) - (1 if self.has_edge_row else 0)

    @cached_property
    def _log_r(self) -> Array:
        """Return ln R at the interior nodes."""
        return np.log(np.asarray(self.r[: self._n_interior], dtype=np.float64))

    @cached_property
    def _log_p(self) -> Array:
        """Return ln P at the interior nodes; ResultError if a P is not positive."""
        p = np.asarray(self.p_trunc[: self._n_interior], dtype=np.float64)
        if not np.all(p > 0.0):
            raise ResultError("The potential of a profile is positive inside its edge.")
        return np.log(p)

    @cached_property
    def _log_p_slope(self) -> Array:
        """Return d ln P / d ln R at the interior nodes."""
        # The Poisson equation dP/dR = -M / R**2 gives
        # d ln P / d ln R = -M / (R P) at each node.
        n = self._n_interior
        return -np.asarray(self.mass_trunc[:n]) / (
            np.asarray(self.r[:n]) * np.asarray(self.p_trunc[:n])
        )

    @cached_property
    def _log_mass_slope(self) -> Array:
        """Return d ln M / d ln R at every node; ResultError if an M is not positive."""
        # The mass integral dM/dR = R**2 rho, with rho the total density of
        # the components, gives d ln M / d ln R = R**3 rho / M at each node.
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
        """Return each component's cubic spline of ln rho in ln R.

        The splines are built on the interior nodes. ResultError if a
        density at an interior node is not positive.
        """
        n = self._n_interior
        rho = np.asarray(self.rho_trunc[:, :n], dtype=np.float64)
        if not np.all(rho > 0.0):
            raise ResultError("The density of a profile is positive inside its edge.")
        return tuple(CubicSpline(self._log_r, np.log(row)) for row in rho)

    @cached_property
    def _inner(self) -> _InnerSeries:
        """Return the inner series' data, read from the first node.

        The data are the amplitude of each component's density series, the
        total amplitude (from the enclosed mass at the first node), the
        slopes (alpha, beta, gamma) of the source density, each component's
        first-order central deficit with its slope, and the central potential
        ``P_0`` (the first node's potential plus the depth at the first
        node). The amplitudes are read from the first node's values with the
        deficit's term restored, so that every column is continuous there.
        ResultError if a density or the enclosed mass at the first node is
        not positive.
        """
        source = self.source
        slopes = (
            float(source.rho_alpha),
            float(source.rho_beta),
            float(source.rho_gamma),
        )
        a, b, g = slopes
        r_1 = float(self.r[0])
        r_1a = np.array([r_1])
        rho_1 = np.asarray(self.rho_trunc[:, 0], dtype=np.float64)
        mass_1 = float(self.mass_trunc[0])
        if not (np.all(rho_1 > 0.0) and mass_1 > 0.0):
            raise ResultError(
                "The density and the enclosed mass of a profile are positive at "
                "its first node."
            )
        deficit, w = self._first_order_deficit(len(rho_1), g)
        missing_mass = float(np.sum(deficit)) * r_1 ** (3.0 - w) / (3.0 - w)
        amplitude = (rho_1 + deficit * r_1 ** (-w)) / float(
            inner_density_series(r_1a, 1.0, g, a, b)[0]
        )
        total = (mass_1 + missing_mass) / float(
            inner_mass_series(r_1a, 1.0, g, a, b)[0]
        )
        inner = _InnerSeries(amplitude, total, slopes, deficit, w, math.nan)
        return replace(inner, p_0=float(self.p_trunc[0]) + float(inner.depth(r_1a)[0]))

    def _first_order_deficit(
        self, n_components: int, gamma: float
    ) -> tuple[Array, float]:
        r"""Return each component's first-order central deficit and its slope.

        The first-order deficit is :math:`\delta_1 = D_0 / (1 + \kappa)`,
        with :math:`D_0` the recorded central deficit at the center
        (`central_deficit`) and :math:`\kappa = (\gamma - 2\beta_{\rm c})
        (3 - \gamma) / [(3 - 2\beta_{\rm c})(2 - 2\beta_{\rm c})]`
        (`pianistpy.df.center.lowered_center`), carried where the model
        records a deficit on a cusp with :math:`2\beta_{\rm c} \le \gamma`,
        as the solve carried it. The slope is :math:`w = 2\beta_{\rm c}`.
        Both are zero where no deficit is carried.
        """
        w = 2.0 * float(self.source_anisotropy.beta_c)
        deficit = np.zeros(n_components)
        recorded = len(self.central_deficit) == n_components
        if not (recorded and gamma > 0.0 and w <= gamma):
            return deficit, 0.0
        kappa = (gamma - w) * (3.0 - gamma) / ((3.0 - w) * (2.0 - w))
        for j, terms in enumerate(self.central_deficit):
            if len(terms) and float(terms[0]) > 0.0:
                deficit[j] = float(terms[0]) / (1.0 + kappa)
        return deficit, w

    def _inner_potential(self, r: Array) -> Array:
        """Return the inner series' potential ``P_0 - u(R)`` at radii `r`."""
        inner = self._inner
        return inner.p_0 - inner.depth(r)

    def _inner_mass(self, r: Array) -> Array:
        """Return the inner series' enclosed mass at radii `r`, in the unit of
        `mass_trunc`."""
        return self._inner.mass(r)

    def _inner_density(self, r: Array) -> Array:
        """Return each component's inner series density at radii `r`, shape
        (n_components, m)."""
        return self._inner.density(r)

    def _inner_radius(self, u: Array) -> Array:
        """Return the radius at which the inner series' depth equals `u`, by
        Newton steps in the radius from the series' leading term."""
        inner = self._inner
        _, _, g = inner.slopes
        u = np.asarray(u, dtype=np.float64)
        leading = inner.total / ((3.0 - g) * (2.0 - g))
        r = (u / leading) ** (1.0 / (2.0 - g))
        for _ in range(INNER_RADIUS_STEPS):
            residual = inner.depth(r) - u
            slope = inner.mass(r) / (r * r)
            step = residual / slope
            move = np.isfinite(step) & (np.abs(step) < 0.5 * r)
            r = np.where(move, r - step, r)
            if not np.any(move & (np.abs(step) > INNER_RADIUS_TOLERANCE * r)):
                break
        return r

    def interpolation_tables(self) -> ProfileTables:
        """Return the arrays from which a JAX function evaluates the profile.

        A JAX function that applies the formulas of `potential`, `density`,
        `mass` and `radius` to the arrays returns the values of those four
        methods. Such a JAX function takes the arrays as arguments and is
        differentiable in the arrays.

        Returns
        -------
        tables : ProfileTables
            The logarithms of the potential and of the mass and their slopes
            at the nodes, the coefficients of each component's density
            spline, the values at the last interior node and at the edge,
            and the inner series' amplitudes, slopes and central potential.

        Raises
        ------
        ResultError
            If the potential or a density at an interior node is not
            positive, or the enclosed mass at a node is not positive.

        See Also
        --------
        ProfileTables : The arrays and their meaning.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> profile.interpolation_tables().log_r.shape
        (601,)
        """
        last = self._n_interior - 1
        finite = self.has_edge_row
        inner = self._inner
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
            inner_amplitude=np.asarray(inner.amplitude, dtype=np.float64),
            inner_total_amplitude=float(inner.total),
            inner_slopes=np.asarray(inner.slopes, dtype=np.float64),
            inner_deficit=np.asarray(inner.deficit, dtype=np.float64),
            inner_deficit_slope=float(inner.deficit_slope),
            p_0=float(inner.p_0),
        )

    def _on_last_interval(self, r: Array) -> Array:
        """Return the mask of the radii on the point-mass interval of the profile.

        The point-mass interval of a profile with an edge node is the range of
        radii beyond the last interior node, where the potential takes the
        point-mass form. The mask is False everywhere for a profile without an
        edge node.
        """
        if not self.has_edge_row:
            return np.zeros(np.shape(r), dtype=bool)
        return r > float(self.r[self._n_interior - 1])

    def potential(self, r: ArrayLike) -> Array:
        r"""Return the relative potential :math:`P(R)`.

        Parameters
        ----------
        r : array-like
            User-specified 3D radii :math:`R`, positive, in internal units.

        Returns
        -------
        p : ndarray
            Relative potential :math:`P(R)` in internal units, with the
            shape of `r`. For a profile with an edge node, :math:`P` is the
            point-mass form :math:`\mathcal{M}_{\rm t}(1/R - 1/R_{\rm t})` beyond
            the last interior node and zero at and beyond :math:`R_{\rm t}`.
            Inside the first node, :math:`P` is :math:`P_0` less the inner
            series' depth (see the class notes).

        Raises
        ------
        ResultError
            If the potential at an interior node is not positive.

        See Also
        --------
        radius : The radius at a given potential.
        potential_log_slope : The logarithmic slope of the potential.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> round(float(profile.potential(1.0)), 6)
        0.5
        """
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p, _ = hermite(np.log(r), self._log_r, self._log_p, self._log_p_slope)
            inside = np.exp(log_p)
        below = r < float(self.r[0])
        if np.any(below):
            r_safe = np.where(below, r, float(self.r[0]))
            inside = np.where(below, self._inner_potential(r_safe), inside)
        if not self.has_edge_row:
            return inside
        point_mass = self.mass_t * np.maximum(1.0 / r - 1.0 / self.r_t, 0.0)
        return np.where(self._on_last_interval(r), point_mass, inside)

    def density(self, r: ArrayLike) -> Array:
        r"""Return the density :math:`\varrho(R)` of each component.

        Parameters
        ----------
        r : array-like, shape (m,)
            User-specified 3D radii :math:`R`, positive, in internal units.

        Returns
        -------
        rho : ndarray, shape (n_components, m)
            Density of each component in internal units. For a profile with
            an edge node, the density is zero at and beyond
            :math:`R_{\rm t}`. Inside the first node, the density is the
            inner series with the amplitude read from the first node (see
            the class notes).

        Raises
        ------
        ResultError
            If a density or, for a profile with an edge node, the potential
            at an interior node is not positive.

        See Also
        --------
        mass : The enclosed mass.
        potential : The relative potential.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> rho = profile.density([0.5, 1.0])
        >>> rho.shape, round(float(rho[0, 1]), 6)
        ((1, 2), 0.25)
        """
        r = np.asarray(r, dtype=np.float64)
        r_1 = float(self.r[0])
        below = r < r_1
        # The splines are read at the first node for the radii below it, where
        # the inner series replaces them: the spline's exponential overflows
        # far inside.
        log_r = np.log(np.maximum(r, r_1))
        inside = np.stack([np.exp(spline(log_r)) for spline in self._log_rho_splines])
        if np.any(below):
            series = self._inner_density(np.minimum(r, r_1))
            inside = np.where(below[np.newaxis, :], series, inside)
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
        r"""Return the total enclosed mass :math:`\mathcal{M}(R)`.

        Parameters
        ----------
        r : array-like
            User-specified 3D radii :math:`R`, positive, in internal units.

        Returns
        -------
        mass : ndarray
            Enclosed mass :math:`\mathcal{M}(R)` in internal units (the unit
            of `mass_trunc`), with the shape of `r`. For a spatially finite
            profile, the mass is `mass_t` at and beyond `r_t`. Inside the
            first node, the mass is the inner series' enclosed mass (see the
            class notes).

        Raises
        ------
        ResultError
            If the enclosed mass at a node is not positive.

        See Also
        --------
        density : The density of each component.
        potential : The relative potential.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> round(float(profile.mass(1.0)), 6)
        0.25
        """
        r = np.asarray(r, dtype=np.float64)
        log_mass, _ = hermite(
            np.log(r),
            np.log(np.asarray(self.r, dtype=np.float64)),
            np.log(np.asarray(self.mass_trunc, dtype=np.float64)),
            self._log_mass_slope,
        )
        inside = np.exp(log_mass)
        below = r < float(self.r[0])
        if np.any(below):
            r_safe = np.where(below, r, float(self.r[0]))
            inside = np.where(below, self._inner_mass(r_safe), inside)
        if not self.spatially_finite:
            return inside
        return np.where(r >= self.r_t, self.mass_t, inside)

    def potential_log_slope(self, r: ArrayLike) -> Array:
        r"""Return the logarithmic slope :math:`\mathrm{d}\ln P/\mathrm{d}\ln R`.

        The Poisson equation :math:`\mathrm{d}P/\mathrm{d}R =
        -\mathcal{M}/R^2` gives :math:`\mathrm{d}\ln P/\mathrm{d}\ln R =
        -\mathcal{M}/(R P)`. The slope is evaluated from `mass` and
        `potential` with no further approximation. The slope is continuous
        in :math:`R` between the nodes.

        Parameters
        ----------
        r : array-like
            User-specified 3D radii :math:`R`, positive, in internal units.

        Returns
        -------
        slope : ndarray
            :math:`\mathrm{d}\ln P/\mathrm{d}\ln R`, dimensionless, with the
            shape of `r`.

        Raises
        ------
        ResultError
            If the potential at an interior node or the enclosed mass at a
            node is not positive.

        See Also
        --------
        potential_log_slope_derivative : The derivative of the slope in
            :math:`R`.
        potential : The relative potential.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> round(float(profile.potential_log_slope(1.0)), 6)
        -0.5
        """
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -self.mass(r) / (r * self.potential(r))

    def potential_log_slope_derivative(self, r: ArrayLike) -> Array:
        r"""Return the derivative in :math:`R` of the potential's logarithmic slope.

        With :math:`s = -\mathcal{M}/(R P)` the logarithmic slope of the
        potential, :math:`\mathrm{d}\mathcal{M}/\mathrm{d}R = R^2 \varrho`
        (:math:`\varrho` the total density of the components) and
        :math:`\mathrm{d}P/\mathrm{d}R = -\mathcal{M}/R^2`, the derivative is

        .. math::

            \frac{\mathrm{d}s}{\mathrm{d}R} = -\frac{R^3 \varrho P
            - \mathcal{M} P + \mathcal{M}^2/R}{R^2 P^2},

        evaluated from `mass`, `potential` and `density` with no further
        approximation.

        Parameters
        ----------
        r : array-like
            User-specified 3D radii :math:`R`, positive, in internal units.

        Returns
        -------
        derivative : ndarray
            :math:`\mathrm{d}s/\mathrm{d}R` in internal units (inverse units
            of :math:`r_0`), with the shape of `r`.

        Raises
        ------
        ResultError
            If the potential or a density at an interior node, or the
            enclosed mass at a node, is not positive.

        See Also
        --------
        potential_log_slope : The logarithmic slope of the potential.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> round(float(profile.potential_log_slope_derivative(1.0)), 6)
        -0.25
        """
        r = np.asarray(r, dtype=np.float64)
        m, p = self.mass(r), self.potential(r)
        rho = np.sum(self.density(r), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -(r**3 * rho * p - m * p + m * m / r) / (r * r * p * p)

    def radius(self, p: ArrayLike) -> Array:
        r"""Return the radius :math:`R` at which the potential takes the value `p`.

        Parameters
        ----------
        p : array-like
            User-specified relative potentials :math:`P`, in internal units,
            below the central potential :math:`P_0` (`ProfileTables.p_0`).
            Above `p_trunc` at the first node, the radius comes from the
            inner series' depth (see the class notes). For a profile with an
            edge node, a value of zero or below gives `r_t`. For a profile
            without an edge node, a value of zero or below gives the radius
            of the last node.

        Returns
        -------
        r : ndarray
            Radius :math:`R` in internal units, with the shape of `p`.

        Raises
        ------
        InputError
            If a value of `p` is at or above the central potential.
        ResultError
            If the potential at an interior node is not positive.

        See Also
        --------
        potential : The potential at a given radius.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> round(float(profile.radius(0.5)), 6)
        1.0
        """
        p = np.asarray(p, dtype=np.float64)
        p_0 = self._inner.p_0
        if np.any(p >= p_0):
            raise InputError(
                "A potential at or above the central potential has no radius."
            )
        p_first = float(self.p_trunc[0])
        p_last = float(self.p_trunc[self._n_interior - 1])
        safe = np.where(p > 0.0, p, p_last)
        safe = np.where(safe > p_first, p_first, safe)
        # The inverse is taken of -ln P, which increases with ln R, as the
        # inverse of the cubic Hermite interpolant of -ln P in ln R.
        log_r, _ = hermite_inverse(
            -np.log(safe), self._log_r, -self._log_p, -self._log_p_slope
        )
        inside = np.exp(log_r)
        above = p > p_first
        if np.any(above):
            u_safe = np.where(above, p_0 - p, p_0 - p_first)
            inside = np.where(above, self._inner_radius(u_safe), inside)
        if not self.has_edge_row:
            return inside
        point_mass = 1.0 / (1.0 / self.r_t + np.maximum(p, 0.0) / self.mass_t)
        return np.where(p < p_last, point_mass, inside)

    def to_dict(self) -> dict[str, Any]:
        """Return the record that identifies the profile, as a plain dictionary.

        The arrays of the profile are not in the record. The identity of
        the model determines the arrays, and a file stores the arrays
        beside the record.

        Returns
        -------
        record : dict
            The key ``"profile"`` with the value ``"truncated"``, the key
            ``"model"`` with the identity string of the model, and the keys
            ``"source"`` and ``"source_anisotropy"`` with the dictionaries of
            `source` and `source_anisotropy`.

        See Also
        --------
        DoublePowerLaw.to_dict : The dictionary of the source profile.

        Examples
        --------
        >>> import numpy as np
        >>> import pianistpy as pp
        >>> from pianistpy.profiles import TruncatedProfile
        >>> hernquist = pp.DoublePowerLaw.hernquist()
        >>> r = np.geomspace(1e-3, 1e3, 601)
        >>> profile = TruncatedProfile(
        ...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
        ...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
        ...     spatially_finite=False, source=hernquist,
        ...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
        ... )
        >>> sorted(profile.to_dict())
        ['model', 'profile', 'source', 'source_anisotropy']
        """
        return {
            "profile": "truncated",
            "model": self.model,
            "source": self.source.to_dict(),
            "source_anisotropy": self.source_anisotropy.to_dict(),
        }


# The profile as a JAX pytree. The fields listed are the pytree's leaves and
# subtrees. The other fields (source, source_anisotropy, model,
# spatially_finite, the override record and model_level) are static.
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
        "inner_amplitude",
        "inner_total_amplitude",
        "inner_slopes",
        "inner_deficit",
        "inner_deficit_slope",
        "p_0",
    ),
)
