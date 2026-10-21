"""Intrinsic velocity moments of one component: the anisotropy profile and the
radial velocity dispersion.

The stage `stages.dispersion_profile` builds a model's table of the radial
dispersion with the functions of `_dispersion`. The public calls `sigma_r`,
`sigma_t`, `sigma_los` and `LosTable` read the radial dispersion table. The
radial dispersion follows from the Jeans integral
``rho_aux(R) sigma_r^2(R) = integral of rho_aux(K) dK from K = 0 to K = P(R)``.
In the Jeans integral, ``P`` is the relative potential, ``K`` is the
integration variable in units of the relative potential, and
``rho_aux = rho / g`` is the auxiliary density. The auxiliary density divides
the density ``rho`` by the separability factor ``g(R)`` of the anisotropy
model. The functions of `_dispersion` also return the derivatives of the
anisotropy profile and of the radial dispersion in one anisotropy parameter,
``beta_c`` or ``r_a``. The caller sets the Gauss-Legendre node count of the
Jeans integral. The formulas are Chiang+(2026d), Eqs. (B43) and (B44).
"""

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

# The largest number of quadrature nodes evaluated at once in the Jeans
# integral. The rows of one chunk share one array of nodes, so _CHUNK_NODES
# bounds the memory of the array of nodes.
_CHUNK_NODES = 500_000


def _inverse_factor(anisotropy: Anisotropy, r: Array) -> Array:
    """Return ``h = 1 / g`` at radii `r` from `inverse_factor`.

    The caller forms ``rho_aux = rho * h``. The product avoids a division by a
    separability factor ``g`` that underflows to zero. The radii must lie on
    the model's support.
    """
    (h,) = inverse_factor(r, anisotropy.family, anisotropy.beta_c, anisotropy.r_a)
    return h


# The anisotropy parameters in which each family's tangent rows are taken.
# The set holds three groups of parameters. First, the parameters of the DF
# gradient (df.gradient.PARAMETERS). Second, beta_c and r_a of COM Type II:
# the Jeans integral reads no DF, so the Jeans integral differentiates a COM
# Type II model although the DF gradient does not. Third, beta_c of an
# Osipkov-Merritt (OM) model: an OM model's g(R) equals the COM g(R) at
# beta_c = 0, and the COM g(R) is smooth in beta_c at beta_c = 0, so the OM
# derivative in beta_c is the COM derivative of the same type at beta_c = 0.
TANGENT_PARAMETERS = {
    "isotropic": ("beta_c",),
    "constant": ("beta_c",),
    "OM_I": ("beta_c", "r_a"),
    "OM_II": ("beta_c", "r_a"),
    "COM_I": ("beta_c", "r_a"),
    "COM_II": ("beta_c", "r_a"),
}


def beta_sigma(r: ArrayLike, anisotropy: Anisotropy) -> Array:
    r"""Return the velocity anisotropy profile :math:`\beta_\sigma(R)`.

    Parameters
    ----------
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    anisotropy : Anisotropy
        The anisotropy model of the component.

    Returns
    -------
    beta_sigma : array
        :math:`\beta_\sigma(R)`, with the shape of `r`.

    Notes
    -----
    With the central anisotropy :math:`\beta_{\rm c}` and the anisotropy
    radius :math:`R_{\rm a}`, :math:`\beta_\sigma = 0` for an isotropic
    model and :math:`\beta_\sigma = \beta_{\rm c}` for a constant-anisotropy
    model. For the Cuddeford-Osipkov-Merritt (COM) models,

    .. math::

        \beta_\sigma(R) = \beta_{\rm c} \pm (1 - \beta_{\rm c})
        \frac{R^2}{R_{\rm a}^2 \pm R^2},

    with the upper sign for Type I and the lower sign for Type II. An
    Osipkov-Merritt model is the COM model of its type at
    :math:`\beta_{\rm c} = 0`.

    References
    ----------
    Chiang+(2026d), Eqs. (7), (B17), (B21).
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
    """Return the family of `anisotropy`.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.
    """
    kind = anisotropy.family
    if parameter not in TANGENT_PARAMETERS[kind]:
        raise InputError(
            f"The tangents of a model of the anisotropy family {kind} are taken in "
            f"the parameters {list(TANGENT_PARAMETERS[kind])}; the parameter given "
            f"is {parameter!r}."
        )
    return kind


def log_factor_tangent(r: ArrayLike, anisotropy: Anisotropy, parameter: str) -> Array:
    r"""Return the derivative of :math:`\ln g(R)` in one anisotropy parameter.

    The separability factor :math:`g(R)` is the factor of
    `Anisotropy.factor`.

    Parameters
    ----------
    r : array-like
        User-specified 3D radii :math:`R`, in internal units.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    parameter : {"beta_c", "r_a"}
        The anisotropy parameter :math:`\theta`, one of the entries of
        ``TANGENT_PARAMETERS`` for the family of `anisotropy`.

    Returns
    -------
    tangent : array
        :math:`\partial \ln g / \partial\theta` at `r`, with the shape of `r`.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.

    Notes
    -----
    With :math:`x = R^2/R_{\rm a}^2`,

    .. math::

        \frac{\partial \ln g}{\partial \beta_{\rm c}} = -2\ln R
        + \ln(1 \pm x), \qquad
        \frac{\partial \ln g}{\partial R_{\rm a}} = \pm
        \frac{2(1 - \beta_{\rm c})\,x}{R_{\rm a}(1 \pm x)},

    with the upper sign for Type I and the lower sign for Type II. The term
    :math:`\ln(1 \pm x)` is present only for the COM and Osipkov-Merritt
    families. The derivative of an isotropic model is the derivative of the
    constant-anisotropy model at :math:`\beta_{\rm c} = 0`. The derivative
    of an Osipkov-Merritt model is the derivative of the COM model of the
    same type at :math:`\beta_{\rm c} = 0`.
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
        User-specified 3D radii :math:`R`, in internal units.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    parameter : {"beta_c", "r_a"}
        The anisotropy parameter :math:`\theta`, one of the entries of
        ``TANGENT_PARAMETERS`` for the family of `anisotropy`.

    Returns
    -------
    tangent : array
        :math:`\partial\beta_\sigma / \partial\theta` at `r`, with the shape
        of `r`.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.

    Notes
    -----
    The derivative is taken of `beta_sigma`. With
    :math:`s = R^2/(R_{\rm a}^2 \pm R^2)`,

    .. math::

        \frac{\partial \beta_\sigma}{\partial \beta_{\rm c}} = 1 \mp s,
        \qquad
        \frac{\partial \beta_\sigma}{\partial R_{\rm a}} = \mp
        \frac{2(1 - \beta_{\rm c})\,s(1 \mp s)}{R_{\rm a}},

    with the upper sign for Type I and the lower sign for Type II. For the
    isotropic and the constant-anisotropy families,
    :math:`\partial\beta_\sigma/\partial\beta_{\rm c} = 1`.
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
    """Return the quadrature variable ``d = ln(1 - k / P_0)`` of the Jeans
    integral.

    ``P_0`` is the central relative potential. Where ``k`` is below
    ``P_0 / 2``, ``log1p`` keeps the relative precision of ``k``. Where ``k``
    is ``P_0 / 2`` or above, ``P_0 - k`` is floored at ``1e-16 * P_0``.
    """
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
    r"""Return the integral of :math:`\varrho_{\rm aux}\,{\rm d}K` from
    `p_outer` to the potential of each row in `p_rows`.

    Where `weight` is given, the integrand is :math:`\varrho_{\rm aux}`
    times `weight`, both evaluated at the same radius. The integral uses
    `n_nodes` Gauss-Legendre nodes in the variable ``d`` of `_depth` on each
    panel between consecutive knots. Each panel is integrated once, and the
    panel integrals are summed from `p_outer` inward. A row adds the integral
    over the part of a panel between the row's potential and the next knot
    toward `p_outer`.
    So the value of one row does not depend on the other rows of the call.
    `panel_cache` holds the summed panel integrals of the unweighted
    integrand only.
    """
    xi, weights = np.polynomial.legendre.leggauss(n_nodes)
    t_nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * weights
    d_hi = _depth(p_outer, p_0)

    def panel_sums(a: Array, b: Array) -> Array:
        # The integral over each interval [a, b] of d, by the Gauss-Legendre
        # nodes.
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
    r"""Return the squared radial velocity dispersion at user-specified radii.

    Parameters
    ----------
    r : array-like, shape (n,)
        User-specified 3D radii :math:`R`, in internal units.
    potential : callable
        The profile's relative potential :math:`P(R)` at radii, in internal
        units.
    density : callable
        The profile's density :math:`\varrho(R)` at radii, in internal
        units.
    radius : callable
        The profile's radius at relative potentials, the inverse of
        `potential`, in internal units.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    p_outer : float
        The lowest relative potential :math:`P_{\rm outer}` of the profile's
        table. The quadrature of the Jeans integral starts at `p_outer`.
    boundary_term : float
        :math:`\int_0^{P_{\rm outer}}\varrho_{\rm aux}\,{\rm d}K`. The
        boundary term is zero on a model with compact support.
    n_nodes : int
        Gauss-Legendre node count of each row's integral, or of each panel
        where `knots` is given.
    knots : array-like, optional
        Relative potentials where the integrand may not be smooth, such as
        the potentials of the profile's own table. Where `knots` is given,
        the integral runs over panels between consecutive knots
        (`_knot_integrals`). Default: ``None``, which means that each row's
        integral from `p_outer` to the row's potential uses `n_nodes` nodes
        of its own.
    panel_cache : dict, optional
        The summed panel integrals of `_knot_integrals`, keyed by `n_nodes`.
        The summed panel integrals do not depend on `r`, so one cache serves
        every call with the same profile and `knots`. Default: ``None``,
        which means that no cache is kept.

    Returns
    -------
    sigma_r2 : array, shape (n,)
        :math:`\sigma_r^2`, in internal units. :math:`\sigma_r^2` is zero
        where the density is zero, where the row's integral ``I`` (Notes) is
        not positive, and where the quotient ``I / rho_aux`` is not finite.
    boundary_fraction : array, shape (n,)
        The share of each row's integral that the boundary term carries.

    Raises
    ------
    ResultError
        If `boundary_term` is not finite.
    ResultError
        If no row has a positive :math:`\sigma_r^2`.

    Notes
    -----
    .. math::

        \varrho_{\rm aux}\sigma_{r}^2(R) = \int_{P_{\rm outer}}^{P(R)}
        \varrho_{\rm aux}(K)\,{\rm d}K + \int_0^{P_{\rm outer}}
        \varrho_{\rm aux}\,{\rm d}K,

    with the auxiliary density :math:`\varrho_{\rm aux} = \varrho / g` and
    the separability factor :math:`g(R)` of the anisotropy model. The first
    integral uses Gauss-Legendre nodes in ``d = ln(1 - K / P_0)``, with
    ``P_0`` the central relative potential. A central cusp of
    :math:`\varrho_{\rm aux}` is smooth in ``d``. ``log1p`` and ``expm1``
    keep the relative precision of ``K`` at both ends of the integral. The
    quotient :math:`\sigma_r^2 = I / \varrho_{\rm aux}`, with ``I`` the sum
    of the two integrals, is formed in logarithms.

    References
    ----------
    Chiang+(2026d), Eqs. (B43), (B44).
    """
    r = np.asarray(r, dtype=np.float64)
    xi, weights = np.polynomial.legendre.leggauss(n_nodes)
    t_nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * weights
    # The profile's readers clamp a radius inside the innermost row to the
    # innermost row, so the potential at R = 1e-300 is the central relative
    # potential.
    p_0 = float(potential(np.array([1e-300]))[0])

    p_grid = potential(r)
    rho_grid = density(r)
    g_grid = anisotropy.factor(r)
    sigma_r2 = np.zeros(len(r))
    boundary_fraction = np.zeros(len(r))

    if not np.isfinite(boundary_term):
        raise ResultError(
            f"The boundary term of the radial dispersion at the table's lowest "
            f"potential {p_outer:.6e} is {boundary_term!r}, which is not finite. The "
            f"boundary term enters the integral at every tabulated radius, so no "
            f"radial dispersion can be computed."
        )

    # The integral of each row whose potential is above p_outer. The profile's
    # readers are called once per chunk of rows, on all nodes of the chunk.
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
            "The radial dispersion is not finite at %d tabulated radii; the radial "
            "dispersion at each such radius is set to 0.",
            n_nonfinite,
        )
    if not np.any(sigma_r2 > 0):
        raise ResultError(
            f"No tabulated radius has a positive radial dispersion. The radial "
            f"dispersion is zero at {int(np.sum(sigma_r2 == 0))} of the "
            f"{len(sigma_r2)} tabulated radii, and {n_nonfinite} of the zero values "
            f"replace a value that is not finite."
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
    r"""Return the derivative of :math:`\sigma_r^2` at rows in one anisotropy
    parameter, on a fixed profile, with the quadrature nodes of
    `radial_dispersion`.

    Parameters
    ----------
    r : array-like, shape (n,)
        3D radii :math:`R` of the rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows, from `radial_dispersion` with the
        same `n_nodes` and `knots`.
    potential : callable
        The profile's relative potential :math:`P(R)` at radii, in internal
        units.
    density : callable
        The profile's density :math:`\varrho(R)` at radii, in internal
        units.
    radius : callable
        The profile's radius at relative potentials, the inverse of
        `potential`, in internal units.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    p_outer : float
        The lowest relative potential :math:`P_{\rm outer}` of the profile's
        table. The quadrature starts at `p_outer`.
    n_nodes : int
        Gauss-Legendre node count of each panel.
    knots : array-like
        The relative potentials at the panel ends, the `knots` given to
        `radial_dispersion`.
    parameter : {"beta_c", "r_a"}
        The anisotropy parameter :math:`\theta`, one of the entries of
        ``TANGENT_PARAMETERS`` for the family of `anisotropy`.

    Returns
    -------
    tangent : array, shape (n,)
        :math:`\partial\sigma_r^2/\partial\theta` at the rows, in internal
        units. The derivative is zero at a row where :math:`\sigma_r^2` is
        zero and at a row whose potential is at or below `p_outer`.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.

    Notes
    -----
    With :math:`I = \varrho_{\rm aux}\sigma_r^2` and the auxiliary density
    :math:`\varrho_{\rm aux} = \varrho / g`, only the separability factor
    :math:`g` depends on :math:`\theta` on a fixed profile:

    .. math::

        \frac{\partial\sigma_{r}^2}{\partial\theta} = \frac{1}{\varrho_{\rm aux}}
        \frac{\partial I}{\partial\theta} + \sigma_{r}^2\frac{\partial\ln
        g}{\partial\theta}, \qquad \frac{\partial I}{\partial\theta} =
        -\int_{P_{\rm outer}}^{P(R)}\varrho_{\rm aux}\frac{\partial\ln g}
        {\partial\theta}\,{\rm d}K,

    with the integral on the panels and nodes of `radial_dispersion` with
    `knots`. The derivative omits the boundary term of `radial_dispersion`.
    The boundary term is zero on a spatially finite model, and
    `stages.dispersion_profile` takes tangent rows only on a spatially finite
    model.
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
    r"""Return the interpolant of a table of :math:`\sigma_r^2`, built from
    the slopes of the Jeans integral.

    Parameters
    ----------
    r : array-like, shape (n,)
        3D radii :math:`R` of the table's rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows, in internal units.
    potential : callable
        The profile's relative potential :math:`P(R)` at radii.
    rho_aux : callable
        The auxiliary density :math:`\varrho_{\rm aux} = \varrho / g` at
        radii, with the separability factor :math:`g(R)`.
    spatially_finite : bool
        Whether the model's density ends at a finite edge radius.
    edge_limit : float
        On a spatially finite model, the limit of :math:`\sigma_r^2 / P` at
        the edge radius, or NaN where the caller states no limit.

    Returns
    -------
    read : callable
        A function of user-specified 3D radii that returns
        :math:`\sigma_r^2` in internal units. The function raises InputError
        at a radius outside the interval the Notes describe.

    Notes
    -----
    The interpolant reads only the rows where :math:`\sigma_r^2` and
    :math:`\varrho_{\rm aux}` are both positive. The interpolant also skips
    the outermost row of a spatially finite model. At each read row, the
    Jeans integral :math:`I = \varrho_{\rm aux}\sigma_r^2` of
    `radial_dispersion` has the exact slope
    :math:`{\rm d}I/{\rm d}P = \varrho_{\rm aux}`. The interpolant of
    :math:`I` is the cubic Hermite interpolant in :math:`P` through the
    rows' values of :math:`I` and the rows' exact slopes
    :math:`\varrho_{\rm aux}`. So the interpolant's slope at a row does not
    depend on the neighboring rows. At a radius between read rows,
    the interpolant returns :math:`\sigma_r^2 = I / \varrho_{\rm aux}`.

    The function raises InputError at a radius inside the innermost read
    row. Beyond the outermost read row, the behavior depends on the model:

    - On a spatially finite model with a finite `edge_limit`, the function
      interpolates :math:`h = \sigma_r^2 / P` linearly in :math:`P`, from
      the value of :math:`h` at the outermost read row to `edge_limit` at
      :math:`P = 0`. Between the outermost read row and the edge radius,
      :math:`I` and :math:`\varrho_{\rm aux}` both go to zero. The function
      raises InputError at a radius beyond the edge radius.
    - On every other model, the function raises InputError at a radius
      beyond the outermost read row.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_r2 = np.asarray(sigma_r2, dtype=np.float64)
    p_rows = potential(r)
    rho_rows = rho_aux(r)
    alive = (sigma_r2 > 0.0) & (rho_rows > 0.0) & np.isfinite(rho_rows)
    if spatially_finite:
        # The reader takes the outermost row of a spatially finite model as
        # the edge row. The linear edge law reads up to the edge row, so the
        # edge row is never a node of the Hermite interpolant.
        alive[np.argmax(r)] = False
    p_nodes, i_unique = np.unique(p_rows[alive], return_index=True)
    slope = rho_rows[alive][i_unique]
    value = sigma_r2[alive][i_unique] * slope
    spline = CubicHermiteSpline(p_nodes, value, slope)
    sigma_outer = float(value[0] / slope[0])
    sigma_inner = float(value[-1] / slope[-1])
    h_outer = sigma_outer / float(p_nodes[0])
    r_read = r[alive]
    edge_law = spatially_finite and bool(np.isfinite(edge_limit))
    # The radius of the edge row ends the linear edge law. The potential
    # continues beyond the edge row, so the InputError test beyond the edge
    # row compares radii, not potentials.
    r_edge = float(np.max(r))

    def read(r_any: ArrayLike) -> Array:
        r_any = np.asarray(r_any, dtype=np.float64)
        p_at = potential(r_any)
        _refuse_outside_radial(
            r_any,
            p_at,
            p_nodes,
            float(r_read.min()),
            float(r_read.max()),
            r_edge if edge_law else None,
        )
        between = (p_at > p_nodes[0]) & (p_at < p_nodes[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            read_between = spline(np.clip(p_at, p_nodes[0], p_nodes[-1])) / rho_aux(
                r_any
            )
        # Without the edge law, _refuse_outside_radial lets through, below
        # the outermost node, only radii whose potential equals p_nodes[0].
        if edge_law:
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


def _refuse_outside_radial(
    r_any: Array,
    p_at: Array,
    p_nodes: Array,
    r_first: float,
    r_last: float,
    r_edge: float | None,
) -> None:
    r"""Raise InputError at a radius outside the interval that
    `radial_integral_reader` interpolates.

    A radius whose potential is above the innermost node's potential
    ``p_nodes[-1]`` raises InputError. Where `r_edge` is None, a radius whose
    potential is below the outermost node's potential ``p_nodes[0]`` raises
    InputError. Where `r_edge` is given, the linear edge law covers the radii
    up to `r_edge`, and a radius above ``r_edge * (1 + 1e-12)`` raises
    InputError. The potential continues beyond `r_edge`, so the comparison
    with `r_edge` uses radii, not potentials.
    """
    inside = p_at > p_nodes[-1]
    if np.any(inside):
        raise InputError(
            f"sigma_r^2 is asked for at r = {float(np.min(r_any[inside])):.4e}, "
            f"inside the innermost radius {r_first:.4e} of the radial table. The "
            f"innermost radius of the radial table is the profile's innermost "
            f"radius (set by truncation.log10_r_min on an untruncated model, by "
            f"the method's floor on a truncated model) or the first radius with a "
            f"positive sigma_r^2. The radial table is not extrapolated inward."
        )
    outside = (
        (r_any > r_edge * (1.0 + 1e-12)) if r_edge is not None else (p_at < p_nodes[0])
    )
    if np.any(outside):
        where = (
            f"beyond the model's edge {r_edge:.4e}, which the model's truncation sets"
            if r_edge is not None
            else f"beyond the outermost radius {r_last:.4e} of the radial table. The "
            f"outermost radius of the radial table is the profile's outermost "
            f"radius (where the potential reaches the method's floor, or the last "
            f"radius before the edge) or the last radius with a positive "
            f"sigma_r^2. The radial table is not extrapolated outward"
        )
        raise InputError(
            f"sigma_r^2 is asked for at r = {float(np.max(r_any[outside])):.4e}, "
            f"{where}."
        )


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
    r"""Return the derivative in one anisotropy parameter of the interpolant
    of `radial_integral_reader`, on the same rows.

    Parameters
    ----------
    r : array-like, shape (n,)
        3D radii :math:`R` of the table's rows, in internal units.
    sigma_r2 : array-like, shape (n,)
        :math:`\sigma_r^2` at the rows, in internal units.
    tangent : array-like, shape (n,)
        :math:`\partial\sigma_r^2/\partial\theta` at the rows, from
        `radial_dispersion_tangent`.
    potential : callable
        The profile's relative potential :math:`P(R)` at radii.
    rho_aux : callable
        The auxiliary density :math:`\varrho_{\rm aux} = \varrho / g` at
        radii, with the separability factor :math:`g(R)`.
    anisotropy : Anisotropy
        The anisotropy model of the component.
    parameter : {"beta_c", "r_a"}
        The anisotropy parameter :math:`\theta`, one of the entries of
        ``TANGENT_PARAMETERS`` for the family of `anisotropy`.
    spatially_finite : bool
        Whether the model's density ends at a finite edge radius.
    edge_limit : float
        On a spatially finite model, the limit of :math:`\sigma_r^2 / P` at
        the edge radius, or NaN where the caller states no limit. The
        derivative holds `edge_limit` fixed.

    Returns
    -------
    read : callable
        A function of user-specified 3D radii that returns
        :math:`\partial\sigma_r^2/\partial\theta` in internal units. The
        function raises InputError at every radius where the interpolant of
        `radial_integral_reader` raises InputError.

    Raises
    ------
    InputError
        If `parameter` is not in ``TANGENT_PARAMETERS`` of the family.

    Notes
    -----
    On fixed nodes in :math:`P`, the interpolant of `radial_integral_reader`
    is linear in the rows' values :math:`I = \varrho_{\rm aux}\sigma_r^2`
    and in the rows' slopes :math:`\varrho_{\rm aux}`. So between rows, the
    derivative of the interpolant is a cubic Hermite interpolant on the same
    nodes, through the values and slopes

    .. math::

        \frac{\partial I}{\partial\theta} = \varrho_{\rm aux}\Big(
        \frac{\partial\sigma_r^2}{\partial\theta}
        - \sigma_r^2\frac{\partial\ln g}{\partial\theta}\Big), \qquad
        \frac{\partial\varrho_{\rm aux}}{\partial\theta} =
        -\varrho_{\rm aux}\frac{\partial\ln g}{\partial\theta}.

    At the radius read, the function divides the Hermite interpolant of
    :math:`\partial I/\partial\theta` by :math:`\varrho_{\rm aux}` and adds
    :math:`\sigma_r^2\,\partial\ln g/\partial\theta`. Beyond the outermost
    read row of a spatially finite model with a finite `edge_limit`, the
    function returns the
    derivative of the linear interpolant of :math:`h = \sigma_r^2 / P` in
    :math:`P`, with `edge_limit` held fixed. For a Method II model,
    `edge_limit` does not depend on the anisotropy parameters, since the
    exponent of the model's edge law changes with :math:`\beta_{\rm c}` in
    the same way as the exponent of the separability factor.
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
