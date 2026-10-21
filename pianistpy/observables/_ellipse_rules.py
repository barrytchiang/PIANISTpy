r"""Hold the numerical rules of the ellipse route and the model constants it reads.

The ellipse route (`EllipseRoute`) computes the rows of the line-of-sight
velocity distribution :math:`D(R_{\rm 2D}, V)` that the public calls
`pianistpy.losvd`, `pianistpy.aperture_losvd` and `pianistpy.LosTable` read.
This module holds the route's node counts and layer counts (`EllipseRules`),
the constants of a model that the route reads (`EllipseModel`), the
tolerances of the route's sub-steps, and the functions that derive layer
counts from the anisotropy :math:`\beta_{\rm c}`.

Every tolerance of the route's sub-steps is :math:`10^{-6}` or a fixed fraction
of :math:`10^{-6}`, the lowest bound that splits the tiers of a validation
report.
The route integrates end-point powers :math:`x^\gamma` on Gauss-Legendre panels
graded toward the end point.
The layer count toward each end point follows from :math:`\beta_{\rm c}` alone
(Notes Section T:6.4).
"""

__all__ = [
    "ANGLE_TOL",
    "BIRTH_SPAN",
    "BIRTH_TOL",
    "BREAK_PAST_END",
    "BREAK_TOL",
    "CUSP_FRAC",
    "DEEP_K",
    "L_BELOW",
    "PANEL_REFUSAL",
    "RB_FILL",
    "RB_MIN_GAP",
    "RB_SAME",
    "ROW_TOL_M2",
    "ROW_TOL_PW",
    "T_TOL",
    "VRAD_KEEP",
    "EllipseModel",
    "EllipseRules",
    "gl01",
    "k_end",
    "k_map",
    "l_max",
    "layer_counts",
    "seg_unit",
]

import functools
from dataclasses import dataclass

import numpy as np

from ..exceptions import ResultError
from ..typing import Array
from ..validation import TIER_BOUNDS

# The tolerances of the route's sub-steps. Each tolerance is 1e-6 or a fixed
# fraction of 1e-6, and each serves the bound 1e-6 on a row.
# ANGLE_TOL (1e-8) bounds the Gauss-Legendre convergence estimate rho^(-2n) of
# each half of the angle rule. A node takes the least listed count n whose
# estimate is at or below ANGLE_TOL (`_angle._need`).
ANGLE_TOL = 1e-2 * TIER_BOUNDS[0]
# T_TOL (1e-6) bounds the change of a row when the panels of its line-of-sight
# rule double. The doubling stops once the change in the row's L1 norm and the
# pointwise change are both at or below T_TOL.
T_TOL = TIER_BOUNDS[0]
# ROW_TOL_M2 (1e-8) and ROW_TOL_PW (1e-7) bound the difference between the
# stencil's reading between two stored rows and a row computed at the radius
# of the reading.
# ROW_TOL_M2 bounds the difference in the second moment. ROW_TOL_PW bounds the
# pointwise difference.
ROW_TOL_M2 = 1e-2 * TIER_BOUNDS[0]
ROW_TOL_PW = 1e-1 * TIER_BOUNDS[0]
# VRAD_KEEP (1e-8) bounds the estimate B of the share of a row's norm or second
# moment that the panels next to a break carry (`_kept.vrad_kept_ranges`,
# `_kept.vk_kept_ranges`). The breaks V_rad and V_k are kept at the projected
# radii where B exceeds VRAD_KEEP.
VRAD_KEEP = 1e-2 * TIER_BOUNDS[0]

# The geometric guards of the velocity breaks and of the rows.
# A break whose distance below V_end is at most BREAK_TOL times V_end coincides
# with V_end and is no interior break of the row.
BREAK_TOL = 1e-9
# The break V_k enters a row where V_end - V_k first exceeds BIRTH_TOL times
# V_end. The search for the entry radius runs over BIRTH_SPAN in ln R_2D below the
# radius where the layout without kept ranges first carries V_k.
BIRTH_TOL = 1e-12
BIRTH_SPAN = 0.05
# Two radii where the layout changes (R-breaks) whose ln R_2D differ by at
# most RB_SAME are merged into one R-break.
RB_SAME = 1e-10
# A stretch between two neighboring R-breaks that holds fewer than RB_FILL
# candidate rows receives RB_FILL more rows.
RB_FILL = 6
# A row is stored only where every gap between neighboring velocity nodes is
# at least RB_MIN_GAP times the value of the upper node.
RB_MIN_GAP = 1e-12
# A row is of the cusp class where the table's top energy Q_top exceeds the
# potential at the row's R_2D by less than CUSP_FRAC times Q_top.
CUSP_FRAC = 1e-3
# The scan for the zero of the angular momentum (L = 0) starts at sinh t equal
# to L_BELOW times V / (2 P(R_2D))^(1/2), the estimate of the first L = 0
# crossing at the least nonzero velocity V of the row.
L_BELOW = 1e-6
# A layout code plus DEEP_K marks a row whose kept V_k lies inside the first
# sub-panel of the grading toward V = 0.
DEEP_K = 100

# Two markers written into the messages of a ResultError. The code that
# catches a ResultError matches a marker in the message to tell the two cases
# apart from every other ResultError.
# BREAK_PAST_END marks a row whose layout puts a break on or above the row's
# own V_end.
BREAK_PAST_END = "[BREAK PAST V_END]"
# PANEL_REFUSAL marks a row, a stretch of rows or a requested radius that the
# panel table cannot read.
PANEL_REFUSAL = "[PANEL TABLE REFUSAL]"


@dataclass(frozen=True)
class EllipseRules:
    r"""The node counts, layer counts and ratios of the ellipse route.

    `_losvd_stages.ellipse_rules` builds an EllipseRules from the settings'
    ``losvd`` group. The field defaults equal the values that
    `_losvd_stages.ellipse_rules` builds from the default settings.

    Parameters
    ----------
    rule_regular : tuple of int
        The rule ``(n_u, k0, ks, kq, m, n_phi)`` of a row of the regular class.
        ``n_u`` is read by no part of the route. ``k0`` is the least layer
        count toward s = 0 of the energy integral. ``ks`` is the least layer
        count toward the split of the energy integral. ``kq`` is the layer
        count toward s = Q_c, where the energy K = Q_c - s of the DF is 0.
        ``m`` is the number of
        Gauss-Legendre nodes of each panel of the energy integral. ``n_phi``
        is the largest number of Gauss-Legendre nodes of the angle rule of a
        node.
    rule_cusp : tuple of int
        The rule of a row of the cusp class (`CUSP_FRAC`), in the form of
        `rule_regular`.
    s_ratio : float
        Ratio of the widths of successive graded panels of the energy
        integral in s.
    angle_counts : tuple of int
        The Gauss-Legendre node counts per half of the angle rule that a node
        may take, smallest first. A node whose need exceeds every listed count
        takes ``n_phi // 2`` nodes per half.
    n_bisect : int
        Bisections of an edge of the support along a line of sight, of an
        L = 0 crossing, and of an interval that may hold an extremum of f in
        `EllipseRoute.f_extrema`.
    chunk : int
        The largest number of elements of one block of node pairs, energy
        nodes and angle nodes, which bounds the memory of a row.
    t_nodes : int
        Gauss-Legendre nodes of each panel of the line-of-sight integral in t.
    t_panels : int
        Panels of each support interval in t at the first level of a row.
    t_levels : int
        The largest number of doublings of the panels in t of a row.
    t_ratio : float
        Ratio of the widths of successive panels in t graded toward the lower
        end of each support interval.
    t_layers : int
        Graded panels in t added at each doubling.
    l_scan : int
        Points of the scan for the L = 0 crossings of a support interval.
    g_nodes : int
        Nodes of each of the three branches of the table of
        :math:`G(Q) = \int_0^Q F\,{\rm d}K` (`EllipseDF`).
    vp_ratio : float
        Ratio of the widths of successive graded velocity sub-panels.
    vp_layers_zero : int
        Velocity sub-panels graded toward V = 0.
    vp_layers_kink : int
        Velocity sub-panels graded toward each velocity break.
    vp_nodes : int
        Gauss-Legendre nodes of each velocity sub-panel.
    brk_scan : int
        Points of the coarse scan in t for V_rad.
    brk_zoom : int
        Zooms of the scan in t for V_rad around the largest value of the scan.
    rb_layers : int
        Rows added on each side of each R-break, a radius where the layout of
        the rows changes.
    rb_bisect : int
        Bisections in :math:`\ln R_{\rm 2D}` of each R-break.
    rb_interior : int
        Samples inside each interval between two base rows in the search for
        R-breaks.
    rb_ext : int
        Samples beyond the last base row toward the end of the lines of sight.
    rt_layers : int
        Rows added toward the end of the table.
    row_order : int
        Rows of the stencil that reads the table between its rows.
    row_order_test : int
        Rows of the stencil whose estimate checks the table.
    kernel : bool
        True takes the angle rule of a node from the JAX kernel
        (`_angle_rule.angle_rule`). False takes the angle rule from the numpy
        rule (`_angle.lam_rule`), the reference.
    """

    rule_regular: tuple[int, ...] = (100, 0, 0, 4, 8, 64)
    rule_cusp: tuple[int, ...] = (200, 0, 0, 4, 8, 64)
    s_ratio: float = 0.2
    angle_counts: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32)
    n_bisect: int = 60
    chunk: int = 25_000
    t_nodes: int = 12
    t_panels: int = 8
    t_levels: int = 6
    t_ratio: float = 0.3
    t_layers: int = 4
    l_scan: int = 2001
    g_nodes: int = 200001
    vp_ratio: float = 0.3
    vp_layers_zero: int = 12
    vp_layers_kink: int = 8
    vp_nodes: int = 16
    brk_scan: int = 4001
    brk_zoom: int = 4
    rb_layers: int = 16
    rb_bisect: int = 40
    rb_interior: int = 3
    rb_ext: int = 10
    rt_layers: int = 16
    row_order: int = 8
    row_order_test: int = 8
    kernel: bool = True


@dataclass(frozen=True)
class EllipseModel:
    r"""The constants of a model that the ellipse route reads.

    `_ellipse_model.ellipse_model` builds an EllipseModel from a model of one
    component.

    Parameters
    ----------
    family : str
        The anisotropy family (`Anisotropy.family`).
    r_a : float
        The anisotropy radius :math:`R_{\rm a}`.
    beta_c : float
        The central anisotropy :math:`\beta_{\rm c}`.
    k_end : int
        Layers of the energy integral toward the L = 0 point at V > 0
        (`layer_counts`).
    k_end_floor : float or None
        The error floor of `k_end` where no layer count reaches the tolerance
        of `k_end`; None where a layer count reaches the tolerance.
    k_end_v0 : int
        Layers of the energy integral toward the L = 0 point at V = 0.
    k_end_v0_floor : float or None
        The error floor of `k_end_v0`, in the form of `k_end_floor`.
    k_map : int
        Layers toward the split that the map of the first panel of piece b
        needs (`k_map`).
    edge_alive : bool
        True where the profile is not spatially finite and the potential at
        the profile's last row of support is above 0. V_edge is then a
        velocity break.
    edge_gamma : float
        The least value of :math:`QF/G` near the potential at the profile's
        last row of support (`_ellipse_model._edge_gamma`); NaN where
        `edge_alive` is False.
    vp_layers_edge : int
        Velocity sub-panels graded toward V_edge from below.
    p0_top : float
        The DF's top energy :math:`P_0`; NaN where the DF's form states none.
    kappa_top : float
        The exponent :math:`\kappa` of :math:`F \sim (P_0 - Q)^\kappa` at the
        top energy; NaN where the DF's form states none.
    vc_p : float
        The exponent p of the turnover of D at V_c; NaN where p is not
        computed.
    vc_alive : bool
        True where V_c is a velocity break.
    vp_layers_c : int
        Velocity sub-panels graded toward V_c from above.
    """

    family: str
    r_a: float
    beta_c: float
    k_end: int
    k_end_floor: float | None
    k_end_v0: int
    k_end_v0_floor: float | None
    k_map: int
    edge_alive: bool
    edge_gamma: float
    vp_layers_edge: int
    p0_top: float
    kappa_top: float
    vc_p: float
    vc_alive: bool
    vp_layers_c: int


@functools.cache
def gl01(m: int) -> tuple[Array, Array]:
    """Return the nodes and weights of the `m`-node Gauss-Legendre rule on [0, 1].

    The arrays are computed once for each `m`, cached, and marked read-only.
    """
    x, w = np.polynomial.legendre.leggauss(m)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def l_max(wv: float, rules: EllipseRules) -> int:
    """Return the most layers L that a graded side of a velocity panel resolves.

    The panel has width w and the graded side ends at the break V. The
    argument `wv` is the ratio w / V. The innermost sub-panel of a side with L
    layers has width ``0.5 r^L w``, with r = ``rules.vp_ratio``. A layer count L
    is resolved where the first Gauss-Legendre node of the innermost
    sub-panel lies at least the spacing of doubles at V away from V. The
    function returns -1 where `wv` is not positive.
    """
    if not float(wv) > 0.0:
        return -1
    x1 = float(gl01(rules.vp_nodes)[0][0])
    return int(
        np.floor(
            np.log(x1 * float(wv) / (2.0 * np.finfo(np.float64).eps))
            / np.log(1.0 / rules.vp_ratio)
        )
    )


def seg_unit(ka: int, kb: int, r: float) -> Array:
    """Return the panel edges on [0, 1] of a rule graded toward both ends.

    The edges on [0, 1/2] are 0 and ``0.5 r^j`` for j from `ka` down to 0. The
    edges on [1/2, 1] mirror the edges of `kb` layers toward 1, ending at 1.
    """
    a = np.concatenate([[0.0], 0.5 * r ** np.arange(ka, -1, -1)])
    b = np.concatenate([[0.0], 0.5 * r ** np.arange(kb, -1, -1)])
    return np.concatenate([a, 1.0 - b[::-1][1:]])


def k_end(
    gam: float,
    log: bool,
    rules: EllipseRules,
    tol: float = 1e-2 * TIER_BOUNDS[0],
    k_max: int = 40,
) -> tuple[int, float | None]:
    r"""Return the layers that integrate an end-point power, and the error floor.

    The test rule on [0, 1] has K layers of ratio ``rules.s_ratio`` on
    [0, 1/2], graded toward 0, and one panel on [1/2, 1]. Each panel takes
    ``m = rules.rule_regular[4]`` Gauss-Legendre nodes. The integrand is
    :math:`x^\gamma` with :math:`\gamma` = `gam`, or :math:`x^\gamma \ln x`
    where `log` is True. The error is the relative error of the test rule
    against the exact integral over [0, 1].

    Returns
    -------
    k : int
        The least K up to `k_max` whose error is at or below `tol`. Where no
        K up to `k_max` reaches `tol`, the least K whose error is at or below
        the error floor plus `tol`.
    floor : float or None
        None where a K up to `k_max` reaches `tol`. Otherwise the error floor,
        the error at K = `k_max`.

    Raises
    ------
    ResultError
        If the error floor is above :math:`10^{-6}`.
    """
    xg, wg = gl01(int(rules.rule_regular[4]))
    f = (lambda x: x**gam * np.log(x)) if log else (lambda x: x**gam)
    ex = -1.0 / (gam + 1.0) ** 2 if log else 1.0 / (gam + 1.0)
    errs = []
    for k in range(0, k_max + 1):
        e = np.concatenate([[0.0], 0.5 * rules.s_ratio ** np.arange(k, -1, -1), [1.0]])
        x = (e[:-1, None] + np.diff(e)[:, None] * xg).ravel()
        w = (np.diff(e)[:, None] * wg).ravel()
        errs.append(abs(float(np.dot(w, f(x))) / ex - 1.0))
        if errs[-1] <= tol:
            return k, None
    floor = errs[-1]
    if floor > TIER_BOUNDS[0]:
        raise ResultError(
            f"No layer count up to {k_max} integrates x^{gam:g}"
            f"{' ln x' if log else ''} to a relative error of 0.000001; the "
            f"relative error at {k_max} layers is {floor:.3e}."
        )
    return next(k for k, err in enumerate(errs) if err <= floor + tol), floor


def k_map(p: float, rules: EllipseRules, tol: float = 1e-2 * TIER_BOUNDS[0]) -> int:
    """Return the layers toward the split that the map of piece b's first panel needs.

    The first panel of piece b, of width d, is mapped by ``s - split = d w^p``
    with w on [0, 1]. The jacobian of the map is ``p w^(p - 1)``. The
    `m`-node Gauss-Legendre rule, with ``m = rules.rule_regular[4]``, has the
    relative error eps on the jacobian over [0, 1]. The function returns the
    least K from 0 to 60 for which ``0.5 s_ratio^K eps`` is at or below `tol`.

    Raises
    ------
    ResultError
        If no K from 0 to 60 brings ``0.5 s_ratio^K eps`` to `tol` or below.
    """
    xg, wg = gl01(int(rules.rule_regular[4]))
    eps = abs(float(np.dot(wg, p * xg ** (p - 1.0))) - 1.0)
    for k in range(0, 61):
        if 0.5 * rules.s_ratio**k * eps <= tol:
            return k
    raise ResultError(
        f"No layer count up to 60 brings the relative error of the mapped first "
        f"panel (p = {p:g}) to the tolerance {tol:g} or below."
    )


def layer_counts(
    beta_c: float, rules: EllipseRules
) -> tuple[int, float | None, int, float | None, int]:
    r"""Return the layer counts at the L = 0 point from :math:`\beta_{\rm c}` alone.

    The function returns the tuple (k_end, the error floor of k_end, k_end_v0,
    the error floor of k_end_v0, k_map) of `EllipseModel`.

    At the L = 0 point :math:`s = s_0`, the non-analytic part of the angle
    integral :math:`\Lambda(s)` behaves as
    :math:`|s - s_0|^{1 - 2\beta_{\rm c}}`. Where :math:`1 - 2\beta_{\rm c}` is
    an integer above -1/2, the non-analytic part carries a factor
    :math:`\ln|s - s_0|`. The count k_end is the `k_end` count of the power
    :math:`1 - 2\beta_{\rm c}`.
    The count k_end is 0 where :math:`\beta_{\rm c} = 0` or
    :math:`\beta_{\rm c} \ge 1`.

    At V = 0 the L = 0 point is s = 0, and :math:`\Lambda \propto
    s^{-\beta_{\rm c}}`. The map of the first panel of piece b carries the
    power :math:`s^{-\beta_{\rm c}}` where :math:`0 < \beta_{\rm c} < 1`.
    The power :math:`s^{-\beta_{\rm c}}` is analytic where
    :math:`\beta_{\rm c}` is an integer. The count k_end_v0 is 0 in both
    cases, and the `k_end` count of :math:`x^{-\beta_{\rm c}}` otherwise.

    The count k_map is the `k_map` count of the map's exponent
    :math:`p = 1/(1 - \beta_{\rm c})` where :math:`0 < \beta_{\rm c} < 1`,
    and 0 otherwise.

    Raises
    ------
    ResultError
        If `k_end` or `k_map` raises ResultError.
    """
    g_end = 1.0 - 2.0 * beta_c
    ke, ke_floor = (
        k_end(g_end, abs(g_end - round(g_end)) < 1e-12 and g_end > -0.5, rules)
        if beta_c != 0.0 and beta_c < 1.0
        else (0, None)
    )
    ke0, ke0_floor = (
        (0, None)
        if 0.0 < beta_c < 1.0 or abs(beta_c - round(beta_c)) < 1e-12
        else k_end(-beta_c, False, rules)
    )
    km = k_map(1.0 / (1.0 - beta_c), rules) if 0.0 < beta_c < 1.0 else 0
    return ke, ke_floor, ke0, ke0_floor, km
