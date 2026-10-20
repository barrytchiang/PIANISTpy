"""The intrinsic second moments of a model: the radial dispersion by the Jeans
integral."""

__all__ = [
    "TANGENT_PARAMETERS",
    "beta_sigma",
    "beta_sigma_tangent",
    "log_factor_tangent",
    "radial_dispersion",
    "radial_dispersion_tangent",
    "radial_integral_reader",
    "radial_integral_tangent_reader",
]

import math
from collections.abc import Callable

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from ..anisotropy import Anisotropy
from ..anisotropy.factor import inverse_factor
from ..exceptions import InputError, ResultError
from ..runtime import log
from ..typing import Array, ArrayLike

# The nodes of one chunk of rows of the Jeans integral, which bounds its memory.
_CHUNK_NODES = 500_000


def _inverse_factor(anisotropy: Anisotropy, r: Array) -> Array:
    """h = 1/g at the radii (`inverse_factor`): rho_aux = rho h, with no
    division by a g that underflows; the radii lie on the model's support."""
    (h,) = inverse_factor(r, anisotropy.family, anisotropy.beta_c, anisotropy.r_a)
    return h


# The parameters of each family that tangent rows are taken in: those of the DF
# table's rule (`df.gradient.PARAMETERS`), COM Type II, which the Jeans
# integral serves since it reads no DF, and beta_c of an Osipkov-Merritt model:
# its g(R) is COM's at beta_c = 0, one smooth function through it whatever the
# family's label, so its tangent there is COM's of the same type.
TANGENT_PARAMETERS = {
    "isotropic": ("beta_c",),
    "constant": ("beta_c",),
    "OM_I": ("beta_c", "r_a"),
    "OM_II": ("beta_c", "r_a"),
    "COM_I": ("beta_c", "r_a"),
    "COM_II": ("beta_c", "r_a"),
}


def beta_sigma(r: ArrayLike, anisotropy: Anisotropy) -> Array:
    r"""Return the anisotropy profile :math:`\beta_\sigma(R)`.

    Parameters
    ----------
    r : array-like
        Radius, in internal units.
    anisotropy : Anisotropy
        The anisotropy of the component.

    Returns
    -------
    beta_sigma : array
        :math:`\beta_\sigma(R)`, with the shape of `r`.

    References
    ----------
    Chiang+(2026d), Eqs. (B13), (B17), (B21) [eqn:COM_beta].
    """
    r = np.asarray(r, dtype=np.float64)
    kind = anisotropy.family
    beta_c, r_a = float(anisotropy.beta_c), float(anisotropy.r_a)
    if kind == "isotropic":
        return np.zeros_like(r)
    if kind == "constant":
        return np.full_like(r, beta_c)
    if kind == "OM_I":
        return r**2 / (r_a**2 + r**2)
    if kind == "OM_II":
        return -(r**2) / (r_a**2 - r**2)
    if kind == "COM_I":
        return beta_c + (1.0 - beta_c) * r**2 / (r_a**2 + r**2)
    return beta_c - (1.0 - beta_c) * r**2 / (r_a**2 - r**2)


def _parameter_of(anisotropy: Anisotropy, parameter: str) -> str:
    """The family of `anisotropy`, refusing a parameter it does not vary in."""
    kind = anisotropy.family
    if parameter not in TANGENT_PARAMETERS[kind]:
        raise InputError(
            f"A {kind} model's tangent rows are taken in "
            f"{list(TANGENT_PARAMETERS[kind])}; got {parameter!r}."
        )
    return kind


def log_factor_tangent(r: ArrayLike, anisotropy: Anisotropy, parameter: str) -> Array:
    r"""Return the derivative of :math:`\ln g(R)` in one anisotropy parameter.

    Parameters
    ----------
    r : array-like
        Radius, in internal units.
    anisotropy : Anisotropy
        The anisotropy of the component.
    parameter : {"beta_c", "r_a"}
        The parameter, one of the family's `TANGENT_PARAMETERS`.

    Returns
    -------
    tangent : array
        :math:`\partial \ln g / \partial\theta` at `r`, with the shape of `r`.

    Notes
    -----
    Of the factor of `Anisotropy.factor`, with :math:`x = R^2/R_{\rm a}^2`:
    :math:`\partial_{\beta_{\rm c}} \ln g = -2\ln R + \ln(1 \pm x)` (the
    second term for COM and OM) and :math:`\partial_{R_{\rm a}} \ln g = \pm
    2(1 - \beta_{\rm c})\,x / (R_{\rm a}(1 \pm x))`, the upper sign for Type I.
    An isotropic model is varied as the constant-anisotropy model at
    :math:`\beta_{\rm c} = 0`, and an Osipkov-Merritt model as COM of its
    type there.
    """
    r = np.asarray(r, dtype=np.float64)
    kind = _parameter_of(anisotropy, parameter)
    beta_c, r_a = float(anisotropy.beta_c), float(anisotropy.r_a)
    if parameter == "beta_c":
        tangent = -2.0 * np.log(r)
        if kind in ("COM_I", "OM_I"):
            tangent = tangent + np.log1p((r / r_a) ** 2)
        elif kind in ("COM_II", "OM_II"):
            tangent = tangent + np.log1p(-((r / r_a) ** 2))
        return tangent
    if kind.endswith("_I"):
        return 2.0 * (1.0 - beta_c) * r**2 / (r_a * (r_a**2 + r**2))
    return -2.0 * (1.0 - beta_c) * r**2 / (r_a * (r_a**2 - r**2))


def beta_sigma_tangent(r: ArrayLike, anisotropy: Anisotropy, parameter: str) -> Array:
    r"""Return the derivative of :math:`\beta_\sigma(R)` in one anisotropy
    parameter.

    Parameters
    ----------
    r : array-like
        Radius, in internal units.
    anisotropy : Anisotropy
        The anisotropy of the component.
    parameter : {"beta_c", "r_a"}
        The parameter, one of the family's `TANGENT_PARAMETERS`.

    Returns
    -------
    tangent : array
        :math:`\partial\beta_\sigma / \partial\theta` at `r`, with the shape
        of `r`.

    Notes
    -----
    Of `beta_sigma`, with :math:`s = R^2/(R_{\rm a}^2 \pm R^2)`:
    :math:`\partial_{\beta_{\rm c}}\beta_\sigma = 1 \mp s` (1 without
    :math:`R_{\rm a}`) and :math:`\partial_{R_{\rm a}}\beta_\sigma = \mp
    2(1 - \beta_{\rm c})\,s(1 \mp s)/R_{\rm a}`, the upper sign for Type I.
    """
    r = np.asarray(r, dtype=np.float64)
    kind = _parameter_of(anisotropy, parameter)
    beta_c, r_a = float(anisotropy.beta_c), float(anisotropy.r_a)
    if kind in ("isotropic", "constant"):
        return np.ones_like(r)
    if kind.endswith("_I"):
        s = r**2 / (r_a**2 + r**2)
        if parameter == "beta_c":
            return 1.0 - s
        return -2.0 * (1.0 - beta_c) * s * (1.0 - s) / r_a
    s = r**2 / (r_a**2 - r**2)
    if parameter == "beta_c":
        return 1.0 + s
    return 2.0 * (1.0 - beta_c) * s * (1.0 + s) / r_a


def _depth(k: ArrayLike, p_0: float) -> Array:
    """The Jeans rule's variable ``d = ln(1 - k / P_0)``, by log1p where it keeps
    the digits of k."""
    k = np.asarray(k, dtype=np.float64)
    return np.where(
        k < 0.5 * p_0,
        np.log1p(-k / p_0),
        np.log(np.maximum(p_0 - k, 1e-16 * p_0) / p_0),
    )


def _knot_integrals(
    p_rows: Array,
    density: Callable[[Array], Array],
    radius: Callable[[Array], Array],
    anisotropy: Anisotropy,
    p_0: float,
    p_outer: float,
    n_nodes: int,
    knots: ArrayLike,
    panel_cache: dict[int, tuple[Array, Array]] | None = None,
    weight: Callable[[Array], Array] | None = None,
) -> Array:
    r"""The integral of :math:`\varrho_{\rm aux}` (times `weight` at the same
    radii, where given) in K from `p_outer` to each row's potential, by
    Gauss-Legendre nodes in ``d`` on panels between the knots, each integrated
    once and summed from ``p_outer``; a row adds its own part panel up to the
    next knot, so its value does not depend on the other rows asked for. The
    cache holds the unweighted panels only."""
    xi, weights = np.polynomial.legendre.leggauss(n_nodes)
    t_nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * weights
    d_hi = _depth(p_outer, p_0)

    def panel_sums(a: Array, b: Array) -> Array:
        # the integral over each [a, b] in d by the Gauss-Legendre nodes
        out = np.zeros(a.size)
        per_chunk = max(1, _CHUNK_NODES // n_nodes)
        for start in range(0, a.size, per_chunk):
            sl = slice(start, start + per_chunk)
            d_nodes = a[sl, None] + (b[sl] - a[sl])[:, None] * t_nodes[None, :]
            r_k = radius((-p_0 * np.expm1(d_nodes)).ravel())
            rho_aux_k = density(r_k) * _inverse_factor(anisotropy, r_k)
            if weight is not None:
                rho_aux_k = rho_aux_k * weight(r_k)
            rho_aux_k = np.where(np.isfinite(rho_aux_k), rho_aux_k, 0.0).reshape(
                d_nodes.shape
            )
            out[sl] = (b[sl] - a[sl]) * ((p_0 * np.exp(d_nodes) * rho_aux_k) @ weights)
        return out

    cache = panel_cache if weight is None else None
    if cache is not None and n_nodes in cache:
        d_knots, running = cache[n_nodes]
    else:
        k = np.asarray(knots, dtype=np.float64)
        d_knots = np.unique(np.append(_depth(k[k > p_outer], p_0), d_hi))
        running = np.append(
            np.cumsum(panel_sums(d_knots[:-1], d_knots[1:])[::-1])[::-1], 0.0
        )
        if cache is not None:
            cache[n_nodes] = (d_knots, running)
    d_rows = _depth(p_rows, p_0)
    j = np.minimum(np.searchsorted(d_knots, d_rows), d_knots.size - 1)
    return panel_sums(d_rows, d_knots[j]) + running[j]


def radial_dispersion(
    r: ArrayLike,
    potential: Callable[[Array], Array],
    density: Callable[[Array], Array],
    radius: Callable[[Array], Array],
    anisotropy: Anisotropy,
    p_outer: float,
    boundary_term: float,
    n_nodes: int,
    knots: ArrayLike | None = None,
    panel_cache: dict[int, tuple[Array, Array]] | None = None,
) -> tuple[Array, Array]:
    r"""Return the radial velocity dispersion squared on a grid of radius.

    Parameters
    ----------
    r : array-like, shape (n,)
        Radius, in internal units.
    potential, density, radius : callable
        The profile's potential and density at radii and its radius at
        potentials, in internal units.
    anisotropy : Anisotropy
        The anisotropy of the component.
    p_outer : float
        The lowest potential of the profile's table, where the integral
        starts.
    boundary_term : float
        :math:`\int_0^{P_{\rm outer}}\varrho_{\rm aux}\,{\rm d}K`, zero on a
        model with compact support.
    n_nodes : int
        Gauss-Legendre nodes of the integral, or of each panel with `knots`.
    knots : array-like, optional
        Potentials where the integrand may not be smooth (the profile's own
        table). If given, the integral runs over panels between them, each
        integrated once and summed from ``p_outer``, and each row adds its part
        panel to the next knot; otherwise each row over its own nodes.
    panel_cache : dict, optional
        The knots' running sums by `n_nodes`, kept between calls with the same
        profile and `knots`, since they do not depend on `r`.

    Returns
    -------
    sigma_r2 : array, shape (n,)
        :math:`\sigma_r^2`, in internal units; zero where the density is zero.
    boundary_fraction : array, shape (n,)
        The share of each row's integral that the boundary term carries.

    Raises
    ------
    ResultError
        If the boundary term is not finite, or no row has a positive value.

    Notes
    -----
    .. math::

        \varrho_{\rm aux}\sigma_{r}^2(R) = \int_{P_{\rm outer}}^{P(R)}
        \varrho_{\rm aux}(K)\,{\rm d}K + \int_0^{P_{\rm outer}}
        \varrho_{\rm aux}\,{\rm d}K,

    the first integral by Gauss-Legendre nodes in
    ``d = ln(1 - K / P_0)``, ``P_0`` the central potential, so that a central
    cusp of :math:`\varrho_{\rm aux}` is smooth in the variable and ``K``
    keeps its relative precision at both ends (``log1p`` and ``expm1``).
    :math:`\sigma_r^2` is the integral over :math:`\varrho_{\rm aux} =
    \varrho / g` taken in logarithms.

    References
    ----------
    Chiang+(2026d), Eq. (B42) [eqn:linearity_2].
    """
    r = np.asarray(r, dtype=np.float64)
    xi, weights = np.polynomial.legendre.leggauss(n_nodes)
    t_nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * weights
    # The readers clamp to the innermost row, which gives the central potential.
    p_0 = float(potential(np.array([1e-300]))[0])

    p_grid = potential(r)
    rho_grid = density(r)
    g_grid = anisotropy.factor(r)
    sigma_r2 = np.zeros(len(r))
    boundary_fraction = np.zeros(len(r))

    if not np.isfinite(boundary_term):
        raise ResultError(
            f"The boundary term of the radial dispersion is {boundary_term!r} at "
            f"the table's lowest potential {p_outer:.6e}: it enters every row's "
            f"integral, so no dispersion can be formed."
        )

    # The integral of each row inside the table: the profile's readers are
    # called once a chunk of rows, on all of its nodes at once.
    empty = (rho_grid <= 0.0) | ((p_grid <= p_outer) & (boundary_term == 0.0))
    inside = np.flatnonzero(~empty & (p_grid > p_outer))
    integrals = np.full(len(r), boundary_term)
    d_hi = _depth(p_outer, p_0)

    if knots is not None and inside.size:
        integrals[inside] = (
            _knot_integrals(
                p_grid[inside],
                density,
                radius,
                anisotropy,
                p_0,
                p_outer,
                n_nodes,
                knots,
                panel_cache,
            )
            + boundary_term
        )
    rows_per_chunk = max(1, _CHUNK_NODES // n_nodes)
    for start in range(0, 0 if knots is not None else inside.size, rows_per_chunk):
        rows = inside[start : start + rows_per_chunk]
        d_lo = _depth(p_grid[rows], p_0)
        d_nodes = d_lo[:, None] + (d_hi - d_lo)[:, None] * t_nodes[None, :]
        e_d = p_0 * np.exp(d_nodes)
        k_nodes = -p_0 * np.expm1(d_nodes)
        r_k = radius(k_nodes.ravel())
        rho_k = density(r_k)
        rho_aux_k = rho_k * _inverse_factor(anisotropy, r_k)
        rho_aux_k = np.where(np.isfinite(rho_aux_k), rho_aux_k, 0.0).reshape(
            d_nodes.shape
        )
        for j, i in enumerate(rows):
            integrals[i] = (d_hi - d_lo[j]) * float(
                np.dot(weights, e_d[j] * rho_aux_k[j])
            ) + boundary_term

    n_nonfinite = 0
    for i in np.flatnonzero(~empty):
        integral, rho_i, g_i = float(integrals[i]), rho_grid[i], g_grid[i]
        if integral <= 0.0:
            value = 0.0
        elif g_i > 0.0 and rho_i > 0.0:
            value = math.exp(math.log(integral) - (math.log(rho_i) - math.log(g_i)))
        else:
            value = float("nan")
        boundary_fraction[i] = boundary_term / integral if integral > 0.0 else 0.0

        if not np.isfinite(value):
            n_nonfinite += 1
            sigma_r2[i] = 0.0
        else:
            sigma_r2[i] = value

    if n_nonfinite > 0:
        log.debug(
            "The radial dispersion was not finite on %d rows, set to 0.", n_nonfinite
        )
    if not np.any(sigma_r2 > 0):
        raise ResultError(
            f"No row has a positive radial dispersion ({n_nonfinite} not finite, "
            f"{int(np.sum(sigma_r2 == 0))} of {len(sigma_r2)} zero)."
        )
    return sigma_r2, boundary_fraction


def radial_dispersion_tangent(
    r: ArrayLike,
    sigma_r2: ArrayLike,
    potential: Callable[[Array], Array],
    density: Callable[[Array], Array],
    radius: Callable[[Array], Array],
    anisotropy: Anisotropy,
    p_outer: float,
    n_nodes: int,
    knots: ArrayLike,
    parameter: str,
) -> Array:
    r"""Return the derivative of :math:`\sigma_r^2` on rows in one anisotropy
    parameter, on a fixed profile, at the Jeans rule's nodes.

    Parameters
    ----------
    r : array-like, shape (n,)
        Radius of the rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows, from `radial_dispersion` with the
        same `n_nodes` and `knots`.
    potential, density, radius : callable
        The profile's potential and density at radii and its radius at
        potentials, in internal units.
    anisotropy : Anisotropy
        The anisotropy of the component.
    p_outer : float
        The lowest potential of the profile's table, where the integral
        starts.
    n_nodes : int
        Gauss-Legendre nodes of each panel.
    knots : array-like
        The potentials the panels end at, as given to `radial_dispersion`.
    parameter : {"beta_c", "r_a"}
        The parameter, one of the family's `TANGENT_PARAMETERS`.

    Returns
    -------
    tangent : array, shape (n,)
        :math:`\partial\sigma_r^2/\partial\theta` at the rows, in internal
        units; zero where :math:`\sigma_r^2` is.

    Notes
    -----
    With :math:`I = \varrho_{\rm aux}\sigma_r^2` and :math:`\varrho_{\rm aux}
    = \varrho / g`, on a fixed profile only :math:`g` moves:

    .. math::

        \frac{\partial\sigma_{r}^2}{\partial\theta} = \frac{1}{\varrho_{\rm aux}}
        \frac{\partial I}{\partial\theta} + \sigma_{r}^2\frac{\partial\ln
        g}{\partial\theta}, \qquad \frac{\partial I}{\partial\theta} =
        -\int_{P_{\rm outer}}^{P(R)}\varrho_{\rm aux}\frac{\partial\ln g}
        {\partial\theta}\,{\rm d}K,

    the integral on the panels and nodes of `radial_dispersion` with `knots`.
    The boundary term is held: it is zero on a spatially finite model, the
    only one whose tangent rows are taken (`dispersion_profile`).
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_r2 = np.asarray(sigma_r2, dtype=np.float64)
    p_0 = float(potential(np.array([1e-300]))[0])
    p_rows = potential(r)
    tangent = np.zeros(len(r))
    rows = np.flatnonzero((sigma_r2 > 0.0) & (p_rows > p_outer))
    if rows.size:
        integrals = _knot_integrals(
            p_rows[rows],
            density,
            radius,
            anisotropy,
            p_0,
            p_outer,
            n_nodes,
            knots,
            weight=lambda r_k: -log_factor_tangent(r_k, anisotropy, parameter),
        )
        rho_aux = density(r[rows]) / anisotropy.factor(r[rows])
        tangent[rows] = integrals / rho_aux + sigma_r2[rows] * log_factor_tangent(
            r[rows], anisotropy, parameter
        )
    return tangent


def radial_integral_reader(
    r: ArrayLike,
    sigma_r2: ArrayLike,
    potential: Callable[[Array], Array],
    rho_aux: Callable[[Array], Array],
    spatially_finite: bool,
    edge_limit: float,
) -> Callable[[ArrayLike], Array]:
    r"""Return the reading of a table of :math:`\sigma_r^2` by the Jeans
    integral's own slopes.

    Parameters
    ----------
    r : array-like, shape (n,)
        Radius of the table's rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows.
    potential : callable
        The profile's potential at radii.
    rho_aux : callable
        :math:`\varrho_{\rm aux} = \varrho / g` at radii.
    spatially_finite : bool
        Whether the model reaches its edge at a finite radius.
    edge_limit : float
        On a spatially finite model, the limit of :math:`\sigma_r^2 / P` at
        the edge, or NaN where none is stated.

    Returns
    -------
    read : callable
        :math:`\sigma_r^2` at radii, in internal units.

    Notes
    -----
    Each row holds :math:`I = \varrho_{\rm aux}\sigma_r^2`, the integral of
    `radial_dispersion`, whose slope in :math:`P` is
    :math:`{\rm d}I/{\rm d}P = \varrho_{\rm aux}` at the row, exactly. ``I``
    is read by the cubic Hermite in :math:`P` through the rows' values and
    these slopes, so no row's slope depends on its neighbors, and
    :math:`\sigma_r^2 = I / \varrho_{\rm aux}` at the radius read. The rows
    with a positive :math:`\sigma_r^2` and :math:`\varrho_{\rm aux}` are
    read, except the last row of a spatially finite model; inside the
    innermost of them the reading is held at its value. Outside the
    outermost, on a spatially finite model (the stretch to the edge, where
    ``I`` and :math:`\varrho_{\rm aux}` both go to zero),
    :math:`h = \sigma_r^2 / P` is read linearly in :math:`P` from that row to
    `edge_limit` at :math:`P = 0`, or held at the row's ``h`` where no limit
    is stated; on any other model the reading is held at that row's value.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_r2 = np.asarray(sigma_r2, dtype=np.float64)
    p_rows = potential(r)
    rho_rows = rho_aux(r)
    alive = (sigma_r2 > 0.0) & (rho_rows > 0.0) & np.isfinite(rho_rows)
    if spatially_finite:
        # the edge row is the edge rule's, never a node
        alive[np.argmax(r)] = False
    p_nodes, i_unique = np.unique(p_rows[alive], return_index=True)
    slope = rho_rows[alive][i_unique]
    value = sigma_r2[alive][i_unique] * slope
    spline = CubicHermiteSpline(p_nodes, value, slope)
    sigma_outer = float(value[0] / slope[0])
    sigma_inner = float(value[-1] / slope[-1])
    h_outer = sigma_outer / float(p_nodes[0])

    def read(r_any: ArrayLike) -> Array:
        r_any = np.asarray(r_any, dtype=np.float64)
        p_at = potential(r_any)
        between = (p_at > p_nodes[0]) & (p_at < p_nodes[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            read_between = spline(np.clip(p_at, p_nodes[0], p_nodes[-1])) / rho_aux(
                r_any
            )
        if spatially_finite and np.isfinite(edge_limit):
            outer = p_at * (edge_limit + (h_outer - edge_limit) * p_at / p_nodes[0])
        elif spatially_finite:
            outer = p_at * h_outer
        else:
            outer = np.full(np.shape(p_at), sigma_outer)
        return np.where(
            between,
            read_between,
            np.where(p_at >= p_nodes[-1], sigma_inner, outer),
        )

    return read


def radial_integral_tangent_reader(
    r: ArrayLike,
    sigma_r2: ArrayLike,
    tangent: ArrayLike,
    potential: Callable[[Array], Array],
    rho_aux: Callable[[Array], Array],
    anisotropy: Anisotropy,
    parameter: str,
    spatially_finite: bool,
    edge_limit: float,
) -> Callable[[ArrayLike], Array]:
    r"""Return the derivative of the reading of `radial_integral_reader` in
    one anisotropy parameter, at any radius, on the same rows.

    Parameters
    ----------
    r : array-like, shape (n,)
        Radius of the table's rows, in internal units.
    sigma_r2, tangent : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows and its derivative in the parameter
        (`radial_dispersion_tangent`).
    potential : callable
        The profile's potential at radii.
    rho_aux : callable
        :math:`\varrho_{\rm aux} = \varrho / g` at radii.
    anisotropy : Anisotropy
        The anisotropy of the component.
    parameter : {"beta_c", "r_a"}
        The parameter, one of the family's `TANGENT_PARAMETERS`.
    spatially_finite : bool
        Whether the model reaches its edge at a finite radius.
    edge_limit : float
        On a spatially finite model, the limit of :math:`\sigma_r^2 / P` at
        the edge, or NaN where none is stated; held fixed.

    Returns
    -------
    read : callable
        :math:`\partial\sigma_r^2/\partial\theta` at radii, in internal units.

    Notes
    -----
    The reading is linear in the rows' values ``I`` and slopes
    :math:`\varrho_{\rm aux}` on fixed nodes, so between rows its derivative
    is the same cubic Hermite through :math:`\partial I/\partial\theta`, equal
    to :math:`\varrho_{\rm aux}` times
    :math:`\partial\sigma_r^2/\partial\theta - \sigma_r^2\partial\ln g/\partial\theta`,
    and :math:`\partial\varrho_{\rm aux}/\partial\theta`, equal to
    :math:`-\varrho_{\rm aux}\,\partial\ln g/\partial\theta`, divided by
    :math:`\varrho_{\rm aux}`, plus :math:`\sigma_r^2\,\partial\ln g/\partial\theta`
    at the radius read. Inside the innermost row and toward
    the edge it is the derivative of the held value and of the linear
    reading of ``h`` in :math:`P`, with `edge_limit` fixed: for a Method II
    model it does not depend on the anisotropy (its edge law's exponent
    moves with :math:`\beta_{\rm c}` as the class factor's does).
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_r2 = np.asarray(sigma_r2, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    forward = radial_integral_reader(
        r, sigma_r2, potential, rho_aux, spatially_finite, edge_limit
    )
    p_rows = potential(r)
    rho_rows = rho_aux(r)
    alive = (sigma_r2 > 0.0) & (rho_rows > 0.0) & np.isfinite(rho_rows)
    if spatially_finite:
        alive[np.argmax(r)] = False
    p_nodes, i_unique = np.unique(p_rows[alive], return_index=True)
    slope = rho_rows[alive][i_unique]
    sigma_nodes = sigma_r2[alive][i_unique]
    tangent_nodes = tangent[alive][i_unique]
    d_slope = -slope * log_factor_tangent(r[alive][i_unique], anisotropy, parameter)
    d_value = tangent_nodes * slope + sigma_nodes * d_slope
    spline = CubicHermiteSpline(p_nodes, d_value, d_slope)
    d_outer = float(tangent_nodes[0])
    d_inner = float(tangent_nodes[-1])

    def read(r_any: ArrayLike) -> Array:
        r_any = np.asarray(r_any, dtype=np.float64)
        p_at = potential(r_any)
        between = (p_at > p_nodes[0]) & (p_at < p_nodes[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            read_between = spline(np.clip(p_at, p_nodes[0], p_nodes[-1])) / rho_aux(
                r_any
            ) + forward(r_any) * log_factor_tangent(r_any, anisotropy, parameter)
        if spatially_finite and np.isfinite(edge_limit):
            outer = p_at * (d_outer / float(p_nodes[0])) * p_at / p_nodes[0]
        elif spatially_finite:
            outer = p_at * d_outer / float(p_nodes[0])
        else:
            outer = np.full(np.shape(p_at), d_outer)
        return np.where(
            between,
            read_between,
            np.where(p_at >= p_nodes[-1], d_inner, outer),
        )

    return read
