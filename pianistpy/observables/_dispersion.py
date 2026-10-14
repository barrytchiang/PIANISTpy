"""The intrinsic second moments of a model: the radial dispersion by the Jeans
integral."""

__all__ = [
    "beta_sigma",
    "radial_dispersion",
    "radial_dispersion_reader",
    "radial_integral_reader",
]

import math
from collections.abc import Callable

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from ..anisotropy import Anisotropy
from ..exceptions import ResultError
from ..runtime import log
from ..typing import Array, ArrayLike

# The smallest value a factor is divided by, against an underflow to zero.
_TINY = 1e-300
# The nodes of one chunk of rows of the Jeans integral, which bounds its memory.
_CHUNK_NODES = 500_000


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
    Paper Eqs. (B13), (B17), (B21) [eqn:COM_beta].
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
        panel to the next knot (3-193); otherwise each row over its own nodes.
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

        \varrho_{\rm aux}\sigma_r^2(R) = \int_{P_{\rm outer}}^{P(R)}
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
    Paper Eq. (B42) [eqn:linearity_2].
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

    def depth_of(k: Array) -> Array:
        # d = ln(1 - k / P_0), by log1p where it keeps the digits of k
        k = np.asarray(k, dtype=np.float64)
        return np.where(
            k < 0.5 * p_0,
            np.log1p(-k / p_0),
            np.log(np.maximum(p_0 - k, 1e-16 * p_0) / p_0),
        )

    # The integral of each row inside the table: the profile's readers are
    # called once a chunk of rows, on all of its nodes at once.
    empty = (rho_grid <= 0.0) | ((p_grid <= p_outer) & (boundary_term == 0.0))
    inside = np.flatnonzero(~empty & (p_grid > p_outer))
    integrals = np.full(len(r), boundary_term)
    d_hi = depth_of(p_outer)

    def panel_sums(a: Array, b: Array) -> Array:
        # the integral over each [a, b] in d by the Gauss-Legendre nodes
        out = np.zeros(a.size)
        per_chunk = max(1, _CHUNK_NODES // n_nodes)
        for start in range(0, a.size, per_chunk):
            sl = slice(start, start + per_chunk)
            d_nodes = a[sl, None] + (b[sl] - a[sl])[:, None] * t_nodes[None, :]
            r_k = radius((-p_0 * np.expm1(d_nodes)).ravel())
            rho_aux_k = density(r_k) / np.maximum(anisotropy.factor(r_k), _TINY)
            rho_aux_k = np.where(np.isfinite(rho_aux_k), rho_aux_k, 0.0).reshape(
                d_nodes.shape
            )
            out[sl] = (b[sl] - a[sl]) * ((p_0 * np.exp(d_nodes) * rho_aux_k) @ weights)
        return out

    if knots is not None and inside.size:
        # Panels between the given knots, each integrated once and summed from
        # d_hi; a row adds its own part panel up to the next knot, so its value
        # does not depend on the other rows asked for.
        if panel_cache is not None and n_nodes in panel_cache:
            d_knots, running = panel_cache[n_nodes]
        else:
            k = np.asarray(knots, dtype=np.float64)
            d_knots = np.unique(np.append(depth_of(k[k > p_outer]), d_hi))
            running = np.append(
                np.cumsum(panel_sums(d_knots[:-1], d_knots[1:])[::-1])[::-1], 0.0
            )
            if panel_cache is not None:
                panel_cache[n_nodes] = (d_knots, running)
        d_rows = depth_of(p_grid[inside])
        j = np.minimum(np.searchsorted(d_knots, d_rows), d_knots.size - 1)
        integrals[inside] = panel_sums(d_rows, d_knots[j]) + running[j] + boundary_term
    rows_per_chunk = max(1, _CHUNK_NODES // n_nodes)
    for start in range(0, 0 if knots is not None else inside.size, rows_per_chunk):
        rows = inside[start : start + rows_per_chunk]
        d_lo = depth_of(p_grid[rows])
        d_nodes = d_lo[:, None] + (d_hi - d_lo)[:, None] * t_nodes[None, :]
        e_d = p_0 * np.exp(d_nodes)
        k_nodes = -p_0 * np.expm1(d_nodes)
        r_k = radius(k_nodes.ravel())
        rho_k = density(r_k)
        g_k = anisotropy.factor(r_k)
        rho_aux_k = rho_k / np.maximum(g_k, _TINY)
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


def radial_dispersion_reader(
    r: ArrayLike,
    sigma_r2: ArrayLike,
    potential: Callable[[Array], Array],
    spatially_finite: bool,
    edge_limit: float,
) -> Callable[[ArrayLike], Array]:
    r"""Return the reading of a table of :math:`\sigma_r^2` at any radius.

    Parameters
    ----------
    r : array-like, shape (n,)
        Radius of the table's rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows.
    potential : callable
        The profile's potential at radii.
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
    The table is read by PCHIP in :math:`-P`, held at the table's ends. On a
    spatially finite model :math:`\sigma_r^2 = P\,h(P)` with ``h`` read by
    PCHIP, and the last row's ``h`` is `edge_limit` where it is stated; where
    it is not, that row is left out and PCHIP's end rule reads ``h`` there.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_r2 = np.asarray(sigma_r2, dtype=np.float64)
    minus_p, i_unique = np.unique(-potential(r), return_index=True)
    if spatially_finite:
        h = sigma_r2[i_unique] / (-minus_p)
        keep = np.ones(len(h), dtype=bool)
        if np.isfinite(edge_limit):
            h[-1] = edge_limit
        else:
            keep[-1] = False
        h_pchip = PchipInterpolator(minus_p[keep], h[keep], extrapolate=True)

        def read_finite(r_any: ArrayLike) -> Array:
            minus_p_at = np.clip(-potential(r_any), minus_p[0], minus_p[-1])
            return -minus_p_at * h_pchip(minus_p_at)

        return read_finite

    pchip = PchipInterpolator(minus_p, sigma_r2[i_unique])

    def read(r_any: ArrayLike) -> Array:
        return pchip(np.clip(-potential(r_any), minus_p[0], minus_p[-1]))

    return read


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
