"""The numerical rules of the ellipse route of the line-of-sight velocity
distribution, and the constants of a model that its rules read."""

__all__ = [
    "ANGLE_TOL",
    "BIRTH_SPAN",
    "BIRTH_TOL",
    "BREAK_TOL",
    "CUSP_FRAC",
    "DEEP_K",
    "L_BELOW",
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
from ..validation import GOAL

# The tolerances of the route's sub-steps, each tied to the accuracy goal.
# The angle rule's count per node: Gauss-Legendre's rate on both halves.
ANGLE_TOL = 1e-2 * GOAL
# A row's t rule stops when doubling its panels moves it by less than this.
T_TOL = GOAL
# The stencil's own estimate between rows, in the second moment and pointwise.
ROW_TOL_M2 = 1e-2 * GOAL
ROW_TOL_PW = 1e-1 * GOAL
# The bound above which a panel's break is kept (V_rad, V_k).
VRAD_KEEP = 1e-2 * GOAL

# Geometric guards of the breaks and the rows.
# A break within this of V_end coincides with it.
BREAK_TOL = 1e-9
# V_k is born where V_end - V_k first exceeds this of V_end, looked for within
# this span in ln R below where the floor puts it inside.
BIRTH_TOL = 1e-12
BIRTH_SPAN = 0.05
# Two R-breaks closer than this in ln R are one.
RB_SAME = 1e-10
# Rows given to a stretch the layers do not populate.
RB_FILL = 6
# A row's nodes stay apart by this of their own values.
RB_MIN_GAP = 1e-12
# A row is of the cusp class where the DF's top lies within this of it.
CUSP_FRAC = 1e-3
# The L = 0 scan's lower end, as a fraction of the row's first-root estimate.
L_BELOW = 1e-6
# The code offset of a layout whose kept V_k lies inside the zero grading.
DEEP_K = 100


@dataclass(frozen=True)
class EllipseRules:
    """The node counts and layers of the ellipse route.

    The values are the working code's default tier.

    Parameters
    ----------
    rule_regular, rule_cusp : tuple of int
        ``(n_u, k0, ks, kq, m, n_phi)`` of a regular and a cusp row: layer
        floors toward s = 0 and the split, layers toward the DF's floor, Gauss
        nodes of each s panel and of the angle rule.
    s_ratio : float
        Ratio of the layers of the s rule.
    angle_counts : tuple of int
        Gauss nodes a half the angle rule may take per node.
    n_bisect : int
        Bisections of an edge of the support or of a root.
    chunk : int
        Elements of a (pairs, s nodes, angle nodes) block.
    t_nodes, t_panels, t_levels : int
        Gauss nodes of a t panel, panels of a support interval at level 0,
        doublings allowed.
    t_ratio : float
        Ratio of the t layers toward the axis.
    t_layers : int
        t layers added a level.
    l_scan : int
        Points of the scan for the L = 0 crossing.
    g_nodes : int
        Nodes of each branch of G's table.
    vp_ratio : float
        Ratio of the V panels' layers.
    vp_layers_zero, vp_layers_kink : int
        Layers toward V = 0 and toward a break.
    vp_nodes : int
        Gauss nodes of a V sub-panel.
    brk_scan, brk_zoom : int
        Points of the coarse scan for a break and its zooms.
    rb_layers, rb_bisect, rb_interior, rb_ext, rt_layers : int
        Rows toward an R-break, bisections of one, samples inside a base
        interval, samples beyond the last base row, rows toward the table's
        end.
    row_order, row_order_test : int
        Rows of the stencil between rows, and of its estimate.
    kernel : bool
        Whether the angle rule of a node is the JAX kernel (`angle_rule`) or
        the numpy rule (`lam_rule`), the reference route. The two differ at
        rounding.
    """

    rule_regular: tuple[int, ...] = (100, 0, 0, 4, 8, 64)
    rule_cusp: tuple[int, ...] = (200, 0, 0, 4, 8, 64)
    s_ratio: float = 0.2
    angle_counts: tuple[int, ...] = (4, 6, 8, 12, 16, 24, 32)
    n_bisect: int = 60
    chunk: int = 250_000
    t_nodes: int = 16
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
    """The constants of a model that the ellipse route's rules read.

    Parameters
    ----------
    family : str
        The anisotropy family (`Anisotropy.family`).
    r_a, beta_c : float
        The anisotropy radius and the constant anisotropy.
    k_end, k_end_v0, k_map : int
        Layers toward the L = 0 point, toward it at V = 0, and those the map
        of piece b's first panel asks for.
    k_end_floor, k_end_v0_floor : float or None
        Where the layers stop at the rule's own floor, that floor.
    edge_alive : bool
        Whether the line of sight ends with the potential above zero.
    edge_gamma : float
        The exponent of G at the cut end, NaN where the end is dead.
    vp_layers_edge : int
        Layers toward V_edge from below.
    p0_top, kappa_top : float
        The DF's top P0 and its exponent there, NaN where unstated.
    vc_p : float
        D's turnover exponent at V_c.
    vc_alive : bool
        Whether V_c is a break.
    vp_layers_c : int
        Layers toward V_c from above.
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
    """Return Gauss-Legendre's nodes and weights on [0, 1] at `m` nodes,
    computed once a count and read only."""
    x, w = np.polynomial.legendre.leggauss(m)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def l_max(wv: float, rules: EllipseRules) -> int:
    """Return the most layers a graded side of a V panel can use in doubles:
    its innermost sub-panel, ``0.5 r^L w`` wide, keeps its first node above
    the spacing of doubles at the break `V`; `wv` is ``w / V``. -1 for a panel
    of no width."""
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
    """Return the panel edges on [0, 1]: the first half graded toward 0 (`ka`
    layers of ratio `r`), the second toward 1 (`kb` layers)."""
    a = np.concatenate([[0.0], 0.5 * r ** np.arange(ka, -1, -1)])
    b = np.concatenate([[0.0], 0.5 * r ** np.arange(kb, -1, -1)])
    return np.concatenate([a, 1.0 - b[::-1][1:]])


def k_end(
    gam: float,
    log: bool,
    rules: EllipseRules,
    tol: float = 1e-2 * GOAL,
    k_max: int = 40,
) -> tuple[int, float | None]:
    r"""Return the layers that integrate an endpoint form, and the floor where
    they stop short of `tol`.

    The least K at which the graded half (K layers of ratio ``s_ratio`` toward
    0) and one panel on [1/2, 1], ``m`` Gauss nodes each, integrate
    :math:`x^\gamma` (:math:`x^\gamma \ln x` where `log`) to `tol`, and floor
    None. Where no count up to `k_max` reaches it (a negative power, which is
    self-similar under the layers), the least count within `tol` of the error
    at `k_max`, and that error as the floor.

    Raises
    ------
    ResultError
        Where the floor is above the goal.
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
    if floor > GOAL:
        raise ResultError(
            f"No layer count up to {k_max} integrates x^{gam:g}"
            f"{' ln x' if log else ''} to the goal {GOAL:g}: {floor:.3e} at "
            f"{k_max} layers."
        )
    return next(k for k, err in enumerate(errs) if err <= floor + tol), floor


def k_map(p: float, rules: EllipseRules, tol: float = 1e-2 * GOAL) -> int:
    """Return the least K at which piece b's first panel, mapped
    ``s - split = d w^p``, is narrow enough that its rule's error on the map's
    own jacobian stays under `tol`.

    Raises
    ------
    ResultError
        Where no count up to 60 does.
    """
    xg, wg = gl01(int(rules.rule_regular[4]))
    eps = abs(float(np.dot(wg, p * xg ** (p - 1.0))) - 1.0)
    for k in range(0, 61):
        if 0.5 * rules.s_ratio**k * eps <= tol:
            return k
    raise ResultError(
        f"No layer count up to 60 narrows the mapped first panel (p {p:g}) to {tol:g}."
    )


def layer_counts(
    beta_c: float, rules: EllipseRules
) -> tuple[int, float | None, int, float | None, int]:
    r"""Return the layers the L = 0 point asks for, from :math:`\beta_c`
    alone: (k_end, its floor, k_end at V = 0, its floor, k_map).

    At the L = 0 point :math:`\Lambda`'s non-analytic part goes as
    :math:`|s - s_0|^{1 - 2\beta_c}` (with a logarithm at a non-negative
    integer power); at V = 0 :math:`\Lambda \propto s^{-\beta_c}`, carried by
    piece b's map where :math:`0 < \beta_c < 1` and analytic at an integer
    power. The map's own jacobian asks `k_map` layers.
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
