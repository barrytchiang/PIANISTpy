"""Integrals over an aperture of the quantities of a projected table."""

__all__ = [
    "aperture_mean_square",
    "aperture_nodes",
    "general_aperture_nodes",
    "unit_rule",
]

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

from ..exceptions import InputError
from ..typing import Array
from .aperture import Aperture

# Gauss-Legendre rules on [0, 1] of more nodes than the passed one, by count.
_CORE_RULES: dict[int, tuple[Array, Array]] = {}


def unit_rule(n_nodes: int) -> tuple[Array, Array]:
    """Return the Gauss-Legendre nodes and weights of `n_nodes` on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * (x + 1.0), 0.5 * w


def _core_rule(
    span: float, t: Array, w: Array, nodes_per_decade: int
) -> tuple[Array, Array]:
    """Return the rule of a panel spanning `span` in ln R: the passed rule, or
    `nodes_per_decade` nodes per decade of the span, rounded up to a multiple
    of 20, where that is more."""
    n = 20 * int(np.ceil(nodes_per_decade * span / np.log(10.0) / 20.0))
    if n <= len(t):
        return t, w
    if n not in _CORE_RULES:
        x, wx = np.polynomial.legendre.leggauss(n)
        _CORE_RULES[n] = (0.5 * (x + 1.0), 0.5 * wx)
    return _CORE_RULES[n]


def _edge_split(surface: Any) -> float | None:
    """Return the radius above which a panel is integrated in
    v = (R_t^2 - R^2)^(1/2): R_t / 2^(1/2), where the factors R and v of the
    measure R dR = -v dv are equal; None where the reading has no edge."""
    r_t = surface.r_t
    return None if r_t is None else r_t / np.sqrt(2.0)


def _v_panel(
    lo: float, hi: float, r_t: float, t: Array, w: Array
) -> tuple[Array, Array]:
    """Return the nodes and weights of the integral of f(R) R dR over [lo, hi]
    taken in v = (R_t^2 - R^2)^(1/2), under the endpoint map of the R panels:
    smooth up to R_t for f = Sigma g, which carries the factor v."""
    v_lo = float(np.sqrt(max(r_t * r_t - lo * lo, 0.0)))
    v_hi = float(np.sqrt(max(r_t * r_t - hi * hi, 0.0)))
    th = np.pi * t
    span = v_lo - v_hi
    v = v_hi + span * 0.5 * (1.0 - np.cos(th))
    dv = span * 0.5 * np.pi * np.sin(th) * w
    r = np.sqrt(np.maximum(r_t * r_t - v * v, 0.0))
    return r, v * dv


def _band_arc(r: Array, r_ape: float, r_off: float, x: Array) -> Array:
    """The angular measure of a circle of radius `r` inside an offset circle,
    from its band variable `x`."""
    return np.where(
        r <= r_ape - r_off,
        2.0 * np.pi,
        4.0 * np.arcsin(np.sqrt(np.clip(0.5 * x, 0.0, 1.0))),
    )


def outside_table(r_min: float, r_edge: float) -> InputError:
    """Return the refusal of an aperture that encloses no tabulated surface
    density: one statement for the dispersion and the distribution of every
    shape, however the nodes find it (no tabulated radius inside the
    aperture, or no weight on those inside)."""
    return InputError(
        f"The aperture lies outside the tabulated projected radii "
        f"[{r_min:.4e}, {r_edge:.4e}]: it encloses no surface density."
    )


def general_aperture_nodes(
    surface: Any,
    arc: Callable[[Array], Array],
    breaks: Sequence[float],
    t: Array,
    w: Array,
    nodes_per_decade: int,
    inner_tolerance: float,
    r_edge: float | None = None,
) -> tuple[Array, Array, Array]:
    r"""Return the nodes and weights of :math:`\int f(R)\,{\rm arc}(R)\,R\,{\rm d}R`
    over the table, one panel between consecutive breaks, and the surface
    density at the nodes.

    Parameters
    ----------
    surface : callable
        The reading of the surface density, with its ``r_min``, ``r_max``,
        ``r_t`` and ``inner_disc``.
    arc : callable
        The angular measure of the circle of each radius inside the aperture.
    breaks : sequence of float
        The radii at which `arc` is not smooth.
    t, w : array
        The Gauss-Legendre rule on [0, 1] of each panel at least.
    nodes_per_decade : int
        Nodes per decade of a panel's span in ln R, where that is more.
    inner_tolerance : float
        The largest estimated share of the aperture's weight inside the first
        row of the table.
    r_edge : float, optional
        Where the integral ends, if inside the table's last row.

    Returns
    -------
    r, weights, surface_density : array

    Notes
    -----
    Each panel is graded in ln R by its own span, for the projected cusp at
    the first row, and runs under the map R = lo + (hi - lo)(1 - cos pi t)/2,
    which clusters nodes at both ends and removes the square-root endpoint
    of `arc` where a circle becomes tangent to the aperture. Beyond
    R_t / 2^(1/2) a panel is taken in v = (R_t^2 - R^2)^(1/2).
    """
    r_min = surface.r_min
    r_edge = float(
        surface.r_max if r_edge is None else min(float(r_edge), float(surface.r_max))
    )
    edges = sorted({r_min, r_edge} | {float(b) for b in breaks if r_min < b < r_edge})
    r_list, w_list = [], []
    r_s = _edge_split(surface)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if not (hi > lo):
            continue
        hi_r = hi if (r_s is None or hi <= r_s) else max(lo, r_s)
        if hi_r > lo:
            span = np.log(hi_r / lo)
            t_p, w_p = _core_rule(span, t, w, nodes_per_decade)
            th = np.pi * t_p
            width = hi_r - lo
            r_p = lo + width * 0.5 * (1.0 - np.cos(th))
            dr = width * 0.5 * np.pi * np.sin(th) * w_p
            r_list.append(r_p)
            w_list.append(arc(r_p) * r_p * dr)
        if hi > hi_r:
            r_v, w_v = _v_panel(hi_r, hi, surface.r_t, t, w)
            r_list.append(r_v)
            w_list.append(arc(r_v) * w_v)
    if not r_list:
        raise outside_table(r_min, r_edge)
    r_nodes, w_nodes = np.concatenate(r_list), np.concatenate(w_list)
    sigma_nodes = surface(r_nodes)
    disc = float(arc(np.array([r_min]))[0]) * surface.inner_disc
    total = float(np.dot(w_nodes, sigma_nodes))
    if not (disc + total) > 0.0:
        raise outside_table(r_min, r_edge)
    _refuse_inner_disc(disc / (disc + total), r_min, inner_tolerance)
    return r_nodes, w_nodes, sigma_nodes


def _refuse_inner_disc(share: float, r_min: float, inner_tolerance: float) -> None:
    if share > inner_tolerance:
        raise InputError(
            f"The disc inside the innermost tabulated projected radius "
            f"{r_min:.4e} carries an estimated {share:.2e} of the aperture's "
            f"weight, more than {inner_tolerance:g}; the table is not "
            f"extrapolated inward."
        )


def _centerd_nodes(
    surface: Any,
    r_ape: float,
    t: Array,
    w: Array,
    nodes_per_decade: int,
    inner_tolerance: float,
) -> tuple[Array, Array, Array]:
    """The nodes of a centerd circle: uniform in ln R from the first row, in v
    beyond the edge split."""
    r_min = surface.r_min
    if not (r_ape > r_min):
        raise InputError(
            f"The aperture's radius {r_ape:.4e} does not exceed the innermost "
            f"tabulated projected radius {r_min:.4e}."
        )
    r_s = _edge_split(surface)
    r_hi = r_ape if (r_s is None or r_ape <= r_s) else max(r_min, r_s)
    span = np.log(r_hi / r_min) if r_hi > r_min else 0.0
    t_c, w_c = (
        _core_rule(span, t, w, nodes_per_decade)
        if span > 0.0
        else (np.zeros(0), np.zeros(0))
    )
    r_nodes = r_min * np.exp(span * t_c)
    w_nodes = span * w_c * r_nodes**2
    if r_ape > r_hi:
        r_v, w_v = _v_panel(r_hi, r_ape, surface.r_t, t, w)
        r_nodes, w_nodes = (
            np.concatenate([r_nodes, r_v]),
            np.concatenate([w_nodes, w_v]),
        )
    sigma_nodes = surface(r_nodes)
    share = surface.inner_disc / (
        surface.inner_disc + float(np.dot(w_nodes, sigma_nodes))
    )
    _refuse_inner_disc(share, r_min, inner_tolerance)
    return r_nodes, w_nodes, sigma_nodes


def _offset_nodes(
    surface: Any,
    r_ape: float,
    r_off: float,
    r_edge: float,
    t: Array,
    w: Array,
    nodes_per_decade: int,
    inner_tolerance: float,
) -> tuple[Array, Array, Array]:
    """The nodes of an offset circle: its full-circle core uniform in ln R,
    its band under the endpoint map (in R - R_off where the band is narrow,
    in ln R otherwise), and either in v beyond the edge split."""
    r_min = surface.r_min
    r_list, w_list = [], []
    r_s = _edge_split(surface)
    r_core = min(r_ape - r_off, r_edge)
    if r_core > r_min:
        r_c_hi = r_core if (r_s is None or r_core <= r_s) else max(r_min, r_s)
        if r_c_hi > r_min:
            span = np.log(r_c_hi / r_min)
            t_c, w_c = _core_rule(span, t, w, nodes_per_decade)
            r_c = r_min * np.exp(span * t_c)
            r_list.append(r_c)
            w_list.append(2.0 * np.pi * span * w_c * r_c**2)
        if r_core > r_c_hi:
            r_v, w_v = _v_panel(r_c_hi, r_core, surface.r_t, t, w)
            r_list.append(r_v)
            w_list.append(2.0 * np.pi * w_v)
    lo, hi = max(abs(r_off - r_ape), r_min), min(r_off + r_ape, r_edge)
    narrow = r_ape <= 0.5 * r_off and r_off + r_ape <= r_edge
    hi_r = hi if (narrow or r_s is None or hi <= r_s) else max(lo, r_s)
    # The narrow map needs no hi - lo: a radius below 1e-16 R_off leaves hi == lo.
    if hi_r > lo or narrow:
        th = np.pi * t
        if narrow:
            r_b = r_off - r_ape * np.cos(th)
            dr = r_ape * np.pi * np.sin(th) * w
            x = (r_ape * np.sin(th)) ** 2 / (2.0 * r_b * r_off)
        else:
            s_lo, s_hi = np.log(lo), np.log(hi_r)
            r_b = np.exp(s_lo + (s_hi - s_lo) * 0.5 * (1.0 - np.cos(th)))
            dr = (s_hi - s_lo) * 0.5 * np.pi * np.sin(th) * w * r_b
            t_b = r_b - r_off
            x = (r_ape - t_b) * (r_ape + t_b) / (2.0 * r_b * r_off)
        r_list.append(r_b)
        w_list.append(_band_arc(r_b, r_ape, r_off, x) * r_b * dr)
    if not narrow and hi > hi_r:
        r_v, w_v = _v_panel(hi_r, hi, surface.r_t, t, w)
        t_v = r_v - r_off
        x_v = (r_ape - t_v) * (r_ape + t_v) / (2.0 * r_v * r_off)
        r_list.append(r_v)
        w_list.append(_band_arc(r_v, r_ape, r_off, x_v) * w_v)
    if not r_list:
        raise outside_table(r_min, r_edge)
    r_nodes, w_nodes = np.concatenate(r_list), np.concatenate(w_list)
    sigma_nodes = surface(r_nodes)
    if r_off - r_ape < r_min:
        disc = 2.0 * np.pi * surface.inner_disc
        share = disc / (disc + float(np.dot(w_nodes, sigma_nodes)))
        _refuse_inner_disc(share, r_min, inner_tolerance)
    return r_nodes, w_nodes, sigma_nodes


def aperture_nodes(
    surface: Any,
    aperture: Aperture,
    n_nodes: int,
    nodes_per_decade: int,
    inner_tolerance: float,
    r_edge: float | None = None,
) -> tuple[Array, Array, Array]:
    r"""Return the nodes and weights of :math:`\int_A f\,{\rm d}A` over an
    aperture for an axisymmetric `f`, and the surface density at the nodes.

    The integral is one-dimensional, :math:`\int f(R)\,{\rm arc}(R)\,R\,{\rm
    d}R`: a centerd circle is uniform in ln R from the table's first row, an
    offset circle has a core and a band of its own, and a rectangle takes
    the panels between its breaks (`general_aperture_nodes`). The weights of
    a centerd circle leave out the factor :math:`2\pi` common to its nodes
    (they are those of :math:`\int f R\,{\rm d}R`), as the working code's
    do; a ratio of two integrals is the same either way.

    Parameters
    ----------
    surface : callable
        The reading of the surface density, with its ``r_min``, ``r_max``,
        ``r_t`` and ``inner_disc``.
    aperture : Aperture
        The aperture.
    n_nodes : int
        Gauss-Legendre nodes of each panel at least.
    nodes_per_decade : int
        Nodes per decade of a panel's span in ln R, where that is more.
    inner_tolerance : float
        The largest estimated share of the aperture's weight inside the first
        row of the table; above it the aperture is refused.
    r_edge : float, optional
        Where an offset or rectangular aperture's integral ends, if inside the
        table's last row.

    Returns
    -------
    r, weights, surface_density : array
    """
    t, w = unit_rule(n_nodes)
    if aperture.shape == "circle" and aperture.r_off == 0.0:
        return _centerd_nodes(
            surface,
            float(cast(float, aperture.r_ape)),
            t,
            w,
            nodes_per_decade,
            inner_tolerance,
        )
    if aperture.shape == "circle":
        return _offset_nodes(
            surface,
            float(cast(float, aperture.r_ape)),
            float(aperture.r_off),
            surface.r_max if r_edge is None else r_edge,
            t,
            w,
            nodes_per_decade,
            inner_tolerance,
        )
    return general_aperture_nodes(
        surface,
        aperture.arc,
        aperture.breaks(),
        t,
        w,
        nodes_per_decade,
        inner_tolerance,
        r_edge,
    )


def aperture_mean_square(
    surface: Any,
    sigma_los2: Callable[[Array], Array],
    aperture: Aperture,
    n_nodes: int,
    nodes_per_decade: int,
    inner_tolerance: float,
) -> float:
    """Return the squared line-of-sight dispersion averaged over an aperture,
    weighted by the surface density.

    Parameters
    ----------
    surface : callable
        The reading of the surface density, as for `aperture_nodes`.
    sigma_los2 : callable
        The reading of the squared line-of-sight dispersion.
    aperture : Aperture
        The aperture.
    n_nodes, nodes_per_decade, inner_tolerance
        As for `aperture_nodes`.

    Returns
    -------
    mean_square : float

    Raises
    ------
    InputError
        Where the aperture encloses no tabulated surface density, or too much
        of it lies inside the table's first row.
    """
    r, weights, sigma = aperture_nodes(
        surface, aperture, n_nodes, nodes_per_decade, inner_tolerance
    )
    numerator = float(np.dot(weights, sigma * sigma_los2(r)))
    denominator = float(np.dot(weights, sigma))
    if not (denominator > 0.0):
        raise outside_table(surface.r_min, surface.r_max)
    return numerator / denominator
