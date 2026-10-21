r"""Quadrature nodes and weights of integrals over an aperture.

`sigma_los_aperture` uses this module through `stages.aperture_dispersion`
and `observables.core._mixed_aperture`. `aperture_losvd` uses this module
through `_aperture_losvd.aperture_losvd_panels`. The model is spherical, so
an integral over the aperture :math:`A` of a quantity :math:`f(R)` of
projected radius :math:`R` is

.. math::

    \int_A f\,{\rm d}A = \int f(R)\,{\rm arc}(R)\,R\,{\rm d}R,

with :math:`{\rm arc}(R)` the angle, in radians, of the circle of radius
:math:`R` about the center of the model that lies inside the aperture
(`Aperture.arc`). The integral runs over the tabulated projected radii,
from the table's first row at :math:`R_{\min}` to the table's last row.
The part of the aperture inside :math:`R_{\min}` is not integrated. An
aperture whose part inside :math:`R_{\min}` carries an estimated share of
the aperture's weight above ``inner_tolerance`` raises InputError. The
callers pass ``inner_tolerance`` = `stages.INNER_DISK_TOLERANCE`, which is
:math:`10^{-6}`. The apertures are the subject of Notes Section T:6.3.
"""

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

# Cache of the Gauss-Legendre rules on [0, 1] that _core_rule builds when a
# panel needs more nodes than the passed rule has, keyed by node count.
_CORE_RULES: dict[int, tuple[Array, Array]] = {}


def unit_rule(n_nodes: int) -> tuple[Array, Array]:
    """Return the nodes and weights of the Gauss-Legendre rule of `n_nodes`
    nodes on the interval [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * (x + 1.0), 0.5 * w


def _core_rule(
    span: float, t: Array, w: Array, nodes_per_decade: int
) -> tuple[Array, Array]:
    """Return the Gauss-Legendre rule on [0, 1] of a panel whose span in
    ln R is `span`.

    The node count n is `nodes_per_decade` per decade of `span`, rounded up
    to a multiple of 20. Where n does not exceed the node count of the
    passed rule (`t`, `w`), the passed rule is returned. Otherwise the rule
    of n nodes is returned, from the cache `_CORE_RULES`."""
    n = 20 * int(np.ceil(nodes_per_decade * span / np.log(10.0) / 20.0))
    if n <= len(t):
        return t, w
    if n not in _CORE_RULES:
        x, wx = np.polynomial.legendre.leggauss(n)
        _CORE_RULES[n] = (0.5 * (x + 1.0), 0.5 * wx)
    return _CORE_RULES[n]


def _edge_split(surface: Any) -> float | None:
    """Return the projected radius above which a panel is integrated in the
    variable ``v = (R_t^2 - R^2)^(1/2)``.

    The returned radius is ``R_t / 2^(1/2)``, where the factors R and v of
    the measure ``R dR = -v dv`` are equal. ``R_t`` is ``surface.r_t``, the
    radius where the surface density's edge factor vanishes. The function
    returns None where ``surface.r_t`` is None."""
    r_t = surface.r_t
    return None if r_t is None else r_t / np.sqrt(2.0)


def _v_panel(
    lo: float, hi: float, r_t: float, t: Array, w: Array
) -> tuple[Array, Array]:
    r"""Return the nodes in R and the weights of the integral of
    ``f(R) R dR`` over [`lo`, `hi`], taken in ``v = (R_t^2 - R^2)^(1/2)``.

    The nodes in v follow the same endpoint map as the panels in R,
    ``v = v_hi + (v_lo - v_hi)(1 - cos(pi t))/2``. For ``f = Sigma g``, the
    surface density Sigma carries the factor v, so the integrand in v is
    smooth up to :math:`R_{\rm t}`. The returned weights include the factor
    v of the measure ``R dR = -v dv``."""
    v_lo = float(np.sqrt(max(r_t * r_t - lo * lo, 0.0)))
    v_hi = float(np.sqrt(max(r_t * r_t - hi * hi, 0.0)))
    th = np.pi * t
    span = v_lo - v_hi
    v = v_hi + span * 0.5 * (1.0 - np.cos(th))
    dv = span * 0.5 * np.pi * np.sin(th) * w
    r = np.sqrt(np.maximum(r_t * r_t - v * v, 0.0))
    return r, v * dv


def _band_arc(r: Array, r_ape: float, r_off: float, x: Array) -> Array:
    """Return the angle, in radians, of the circle of radius `r` inside an
    offset circular aperture, from the band variable `x`.

    The angle is 2 pi where `r` does not exceed ``r_ape - r_off``.
    Otherwise the angle is ``4 arcsin((x/2)^(1/2))``, with `x` clipped so
    that ``x/2`` lies in [0, 1]."""
    return np.where(
        r <= r_ape - r_off,
        2.0 * np.pi,
        4.0 * np.arcsin(np.sqrt(np.clip(0.5 * x, 0.0, 1.0))),
    )


def outside_table(r_min: float, r_edge: float) -> InputError:
    """Return the InputError for an aperture that encloses no tabulated
    surface density.

    The aperture dispersion and the aperture distribution of every aperture
    shape raise this one InputError. The node functions raise the
    InputError in two cases: no tabulated projected radius lies inside the
    aperture; or the tabulated projected radii inside the aperture carry no
    weight."""
    return InputError(
        f"The aperture lies outside the tabulated projected radii "
        f"[{r_min:.4e}, {r_edge:.4e}], so the aperture encloses no surface density."
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
    over the tabulated projected radii, and the surface density at the nodes.

    The integral has one panel between each pair of consecutive radii among
    ``surface.r_min``, the radii of `breaks` inside the table, and the
    integral's end radius.

    Parameters
    ----------
    surface : callable
        The reader of the surface density, with the attributes ``r_min``,
        ``r_max``, ``r_t``, ``inner_disk`` and ``inner_note``.
    arc : callable
        The angle, in radians, of the circle of each projected radius
        inside the aperture (`Aperture.arc`).
    breaks : sequence of float
        The projected radii at which `arc` is not smooth.
    t : array
        The nodes of the Gauss-Legendre rule on [0, 1] that each panel uses
        at least.
    w : array
        The weights of the Gauss-Legendre rule on [0, 1] that each panel
        uses at least.
    nodes_per_decade : int
        Nodes per decade of a panel's span in ln R, where this count gives
        a panel more nodes than `t` has.
    inner_tolerance : float
        The largest accepted estimated share of the aperture's weight inside
        ``surface.r_min``, the projected radius of the table's first row.
    r_edge : float, optional
        The projected radius where the integral ends, where `r_edge` is
        below ``surface.r_max``. Default: ``None``, which means the integral
        ends at ``surface.r_max``.

    Returns
    -------
    r : array
        The projected radii of the nodes.
    weights : array
        The weights of the nodes, including ``arc(R) R`` and the Jacobian of
        each panel's map.
    surface_density : array
        The surface density at the nodes.

    Raises
    ------
    InputError
        If the aperture encloses no tabulated surface density
        (`outside_table`).
    InputError
        If the estimated share of the aperture's weight inside
        ``surface.r_min`` is above `inner_tolerance`.

    Notes
    -----
    Each panel's node count follows the panel's own span in ln R
    (`_core_rule`), so that the panels near the projected cusp at the first
    row get enough nodes. Each panel maps the rule on [0, 1] to
    ``R = lo + (hi - lo)(1 - cos(pi t))/2``. The map clusters nodes at both
    ends of the panel and removes the square-root endpoint behavior of
    `arc` where a circle becomes tangent to the aperture's edge. Above
    ``R_t / 2^(1/2)``, a panel is taken in ``v = (R_t^2 - R^2)^(1/2)``
    (`_v_panel`).
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
    # An aperture whose arc is zero at the first row holds none of the disk
    # inside the first row, so the disk's estimate is not read there.
    arc_min = float(arc(np.array([r_min]))[0])
    disk = arc_min * surface.inner_disk if arc_min > 0.0 else 0.0
    total = float(np.dot(w_nodes, sigma_nodes))
    if not (disk + total) > 0.0:
        raise outside_table(r_min, r_edge)
    _refuse_inner_disk(
        disk / (disk + total), r_min, inner_tolerance, surface.inner_note
    )
    return r_nodes, w_nodes, sigma_nodes


def _refuse_inner_disk(
    share: float, r_min: float, inner_tolerance: float, note: str = ""
) -> None:
    if share > inner_tolerance:
        raise InputError(
            f"The disk inside the innermost tabulated projected radius "
            f"{r_min:.4e} carries an estimated fraction {share:.2e} of the "
            f"aperture's weight, more than the tolerance {inner_tolerance:g}; the "
            f"table is not extrapolated inward." + (f" {note}" if note else "")
        )


def _centered_nodes(
    surface: Any,
    r_ape: float,
    t: Array,
    w: Array,
    nodes_per_decade: int,
    inner_tolerance: float,
) -> tuple[Array, Array, Array]:
    """Return the nodes, weights and surface densities of a centered circle
    of radius `r_ape`.

    The nodes are uniform in ln R from ``surface.r_min`` up to the smaller
    of `r_ape` and the edge split (`_edge_split`). Between the edge split
    and `r_ape`, the nodes are taken in ``v = (R_t^2 - R^2)^(1/2)``. The
    weights leave out the factor 2 pi that every node of a centered circle
    shares. A centered circle whose radius does not exceed ``surface.r_min``
    raises InputError."""
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
    share = surface.inner_disk / (
        surface.inner_disk + float(np.dot(w_nodes, sigma_nodes))
    )
    _refuse_inner_disk(share, r_min, inner_tolerance, surface.inner_note)
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
    """Return the nodes, weights and surface densities of an offset circle.

    The offset circle has two parts. The core is the range of R up to
    ``r_ape - r_off``, where the whole circle of radius R lies inside the
    aperture. The core's nodes are uniform in ln R. The band is the range
    of R from ``|r_off - r_ape|`` to ``r_off + r_ape``, where the circle of
    radius R crosses the aperture's edge. The band's nodes follow the
    endpoint map in ``R - r_off`` where the band is narrow (`r_ape` at most
    half of `r_off`, and the band inside `r_edge`), and in ln R otherwise.
    Above the edge split (`_edge_split`), the nodes of the core and of a
    wide band are taken in ``v = (R_t^2 - R^2)^(1/2)``."""
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
    # The narrow band's map does not use hi - lo. An r_ape below 1e-16 r_off
    # rounds hi to lo, and the narrow band still gets its nodes.
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
        disk = 2.0 * np.pi * surface.inner_disk
        share = disk / (disk + float(np.dot(w_nodes, sigma_nodes)))
        _refuse_inner_disk(share, r_min, inner_tolerance, surface.inner_note)
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

    The integral reduces to :math:`\int f(R)\,{\rm arc}(R)\,R\,{\rm d}R`.
    The nodes depend on the aperture's shape. A centered circle has nodes
    uniform in ln R from the table's first row (`_centered_nodes`). An
    offset circle has a core and a band (`_offset_nodes`). A rectangle has
    one panel between each pair of consecutive break radii
    (`general_aperture_nodes`). The weights of a centered circle leave out
    the factor :math:`2\pi` that all nodes of a centered circle share, so
    the weights of a centered circle integrate :math:`f R\,{\rm d}R`. A
    ratio of two integrals with the same weights does not depend on the
    left-out factor.

    Parameters
    ----------
    surface : callable
        The reader of the surface density, with the attributes ``r_min``,
        ``r_max``, ``r_t``, ``inner_disk`` and ``inner_note``.
    aperture : Aperture
        The aperture, with lengths in internal units.
    n_nodes : int
        The least number of Gauss-Legendre nodes of each panel.
    nodes_per_decade : int
        Nodes per decade of a panel's span in ln R, where this count gives
        a panel more nodes than `n_nodes`.
    inner_tolerance : float
        The largest accepted estimated share of the aperture's weight inside
        ``surface.r_min``, the projected radius of the table's first row. A
        larger share raises InputError.
    r_edge : float, optional
        The projected radius where the integral of an offset circle or a
        rectangle ends, where `r_edge` is below ``surface.r_max``. Default:
        ``None``, which means the integral ends at ``surface.r_max``.

    Returns
    -------
    r : array
        The projected radii of the nodes.
    weights : array
        The weights of the nodes.
    surface_density : array
        The surface density at the nodes.
    """
    t, w = unit_rule(n_nodes)
    if aperture.shape == "circle" and aperture.r_off == 0.0:
        return _centered_nodes(
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
    """Return the squared line-of-sight velocity dispersion averaged over an
    aperture, weighted by the surface density.

    Parameters
    ----------
    surface : callable
        The reader of the surface density, as for `aperture_nodes`.
    sigma_los2 : callable
        The reader of the squared line-of-sight velocity dispersion over
        projected radius.
    aperture : Aperture
        The aperture, with lengths in internal units.
    n_nodes : int
        As for `aperture_nodes`.
    nodes_per_decade : int
        As for `aperture_nodes`.
    inner_tolerance : float
        As for `aperture_nodes`.

    Returns
    -------
    mean_square : float
        The integral of the surface density times the squared dispersion
        over the aperture, divided by the integral of the surface density
        over the aperture, in internal units.

    Raises
    ------
    InputError
        If the aperture encloses no tabulated surface density.
    InputError
        If the estimated share of the aperture's weight inside the table's
        first row is above `inner_tolerance`.
    """
    r, weights, sigma = aperture_nodes(
        surface, aperture, n_nodes, nodes_per_decade, inner_tolerance
    )
    numerator = float(np.dot(weights, sigma * sigma_los2(r)))
    denominator = float(np.dot(weights, sigma))
    if not (denominator > 0.0):
        raise outside_table(surface.r_min, surface.r_max)
    return numerator / denominator
