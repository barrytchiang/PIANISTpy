r"""The edge of a Method II child: the two-term expansion of its augmented
density at the parent's truncation energy, the closed-form DF that follows
from it, and the floor terms the inversion reads from it.

A Method II child is inverted from the self-consistent profile of a
truncated parent. Below the parent's switch energy the parent's DF is its
Taylor form :math:`F_1 \mathcal{K} + F_2 \mathcal{K}^2/2` (`AsymptoticForm`,
rule ``"taylor"``), so the parent's density near the edge is, to the order
kept,

.. math::

    \varrho(P) = 4\pi\sqrt{2}\,\Big[\tfrac{4}{15} F_1 P^{5/2}
                 + \tfrac{8}{105} F_2 P^{7/2}\Big],

and the child's augmented density :math:`\varrho_{\rm aug} = h(R)\,\varrho`,
with :math:`h = 1/g` the inverse of the child's anisotropy factor
(`Anisotropy.factor`) expanded at the edge, :math:`h = h_{\rm t} + H_1 P`
with :math:`H_1 = h'(R_{\rm t})\,R'(P)` and :math:`R' = -R_{\rm t}^2/M_{\rm t}`
from the Poisson equation, is

.. math::

    \varrho_{\rm aug}(P) = a_{5/2} P^{5/2} + a_{7/2} P^{7/2} + O(P^{9/2}).

Each power inverts to a power (`edge.df_edge_amplitude`), so the child's DF
below its switch is the edge law :math:`\mathcal{K}^p (C_0 + C_1 \mathcal{K})`
with :math:`p = 1 + \beta_{\rm c}` (`AsymptoticForm`, rule ``"edge_law"``).
The same coefficients give the inversion's floor terms: the exact value of
:math:`\varrho_{\rm aug}^{(m_{\rm f})}` at the table's floor (the boundary
term the kernel carries) and the input of `rows.method2_floor_tail` (the
exact integral below the floor).
"""

__all__ = [
    "EdgeExpansion",
    "edge_expansion",
    "edge_expansion_at_node",
    "edge_expansion_of_parent",
    "edge_law",
    "floor_tail",
    "floor_value",
]

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gamma as gamma_function

from ..anisotropy import Anisotropy
from ..anisotropy.factor import inverse_factor
from ..exceptions import InputError
from .edge import df_edge_amplitude, df_edge_exponent
from .inversion import inversion_prefactor, order_pair

# 4 pi sqrt 2: the isotropic forward prefactor (4 pi) and (2 (P - K))^(1/2).
_A = 4.0 * np.pi * np.sqrt(2.0)


@dataclass(frozen=True)
class EdgeExpansion:
    r"""The two-term expansion of a child's augmented density at the edge.

    Parameters
    ----------
    coefficients : dict of float to float
        :math:`a_q` for :math:`q = 5/2, 7/2`: :math:`\varrho_{\rm aug}(P) =
        \sum_q a_q P^q`.
    beta_c : float
        The child's central anisotropy (0 for an OM model).
    k_switch : float
        The parent's switch energy: the expansion holds below it.
    h_t, h_1 : float
        :math:`h(R_{\rm t})` and :math:`H_1`, the inverse anisotropy factor
        and its slope in :math:`P` at the edge.
    """

    coefficients: dict[float, float]
    beta_c: float
    k_switch: float
    h_t: float
    h_1: float

    def density(self, p: Any) -> np.ndarray:
        """Return the expansion's augmented density at the potentials ``p``."""
        p = np.asarray(p, dtype=np.float64)
        return sum((a * p**q for q, a in self.coefficients.items()), np.zeros_like(p))


def _inverse_factor_and_slope(
    anisotropy: Anisotropy, r_t: float
) -> tuple[float, float]:
    """Return h = 1/g at the edge and dh/dR there (`inverse_factor`)."""
    if anisotropy.family == "isotropic":
        return 1.0, 0.0
    h, dh = inverse_factor(
        float(r_t),
        anisotropy.family,
        float(anisotropy.beta_c),
        float(anisotropy.r_a),
        1,
    )
    return float(h), float(dh)


def edge_expansion(
    f_1: float,
    f_2: float,
    k_switch: float,
    r_t: float,
    m_t: float,
    anisotropy: Anisotropy,
) -> EdgeExpansion:
    r"""Return the edge expansion of a child's augmented density.

    Parameters
    ----------
    f_1, f_2 : float
        The parent's Taylor coefficients at the edge, :math:`F = F_1
        \mathcal{K} + F_2 \mathcal{K}^2 / 2` below `k_switch`.
    k_switch : float
        The parent's switch energy.
    r_t, m_t : float
        The parent's truncation radius and total mass, in internal units
        (the mass in the unit of `DoublePowerLaw.mass`, so that
        :math:`\mathrm{d}P/\mathrm{d}R = -M/R^2`).
    anisotropy : Anisotropy
        The child's anisotropy.

    Returns
    -------
    expansion : EdgeExpansion

    Raises
    ------
    InputError
        For a non-positive radius, mass or switch energy, or a Type II child
        whose :math:`R_{\rm a}` lies inside the parent's edge.
    """
    r_t, m_t = float(r_t), float(m_t)
    if not (r_t > 0.0 and m_t > 0.0):
        raise InputError(f"r_t and m_t must be positive; got {r_t}, {m_t}.")
    if not float(k_switch) > 0.0:
        raise InputError(f"k_switch must be positive; got {k_switch}.")
    h_t, dh = _inverse_factor_and_slope(anisotropy, r_t)
    r_prime = -(r_t**2) / m_t
    h_1 = dh * r_prime
    f_1, f_2 = float(f_1), float(f_2)
    a_52 = _A * (4.0 / 15.0) * f_1 * h_t
    a_72 = _A * ((8.0 / 105.0) * f_2 * h_t + (4.0 / 15.0) * f_1 * h_1)
    bc = 0.0 if anisotropy.model == "om" else float(anisotropy.beta_c)
    return EdgeExpansion({2.5: a_52, 3.5: a_72}, bc, float(k_switch), h_t, h_1)


def edge_expansion_at_node(
    f_1: float,
    f_2: float,
    k_switch: float,
    p_node: float,
    h_node: float,
    h_slope: float,
    anisotropy: Anisotropy,
    parent_h: tuple[float, float] | None = None,
) -> EdgeExpansion:
    r"""Return the edge expansion with the factor taken at a node near the
    edge, as the inversion's floor node reads it.

    The factor is linear over the node's distance from the edge, :math:`h =
    h_{\rm t} + H_1 P` with :math:`H_1` its slope at the node and
    :math:`h_{\rm t} = h(P_{\rm node}) - H_1 P_{\rm node}`; at
    :math:`P_{\rm node} = 0` this is `edge_expansion`.

    For an Osipkov-Merritt Type I parent the integral of its DF is its own
    augmented density :math:`h_{\rm p}\,\hat\varrho`, :math:`h_{\rm p} = 1 +
    R^2/r_{\rm a,p}^2`, so the factor applied to the parent's Taylor form is
    :math:`h / h_{\rm p}`, with slope :math:`(H_1 h_{\rm p} - h\,H_{\rm
    p,1}) / h_{\rm p}^2`.

    Parameters
    ----------
    f_1, f_2, k_switch : float
        The parent's Taylor coefficients and switch energy.
    p_node : float
        The node's potential.
    h_node, h_slope : float
        The child's factor at the node and its slope in P there.
    anisotropy : Anisotropy
        The child's anisotropy.
    parent_h : tuple of float, optional
        An OM Type I parent's :math:`h_{\rm p}` and its slope in P at the
        node; None for an isotropic parent.
    """
    p_node = float(p_node)
    if parent_h is None:
        h_1 = float(h_slope)
        h_t = float(h_node) - h_1 * p_node
    else:
        hp, hp_1 = (float(v) for v in parent_h)
        h_1 = float((h_slope * hp - h_node * hp_1) / hp**2)
        h_t = float(h_node / hp) - h_1 * p_node
    f_1, f_2 = float(f_1), float(f_2)
    a_52 = _A * (4.0 / 15.0) * f_1 * h_t
    a_72 = _A * ((8.0 / 105.0) * f_2 * h_t + (4.0 / 15.0) * f_1 * h_1)
    bc = 0.0 if anisotropy.model == "om" else float(anisotropy.beta_c)
    return EdgeExpansion({2.5: a_52, 3.5: a_72}, bc, float(k_switch), h_t, h_1)


def edge_expansion_of_parent(
    parent: Any, anisotropy: Anisotropy, component: int = 0
) -> EdgeExpansion:
    """Return the edge expansion from a truncated parent model.

    The parent states its edge as the ``"taylor"`` rule of its
    `AsymptoticForm` (a truncated Method I solve writes it); a parent that
    states another rule is refused, since the expansion would then be a
    guess.

    Parameters
    ----------
    parent : Model
        The truncated parent: its ``form`` (the Taylor coefficients) and its
        ``profile`` (``r_t``, ``mass_t``) are read.
    anisotropy : Anisotropy
        The child's anisotropy.
    component : int, optional
        The parent's component the child is built on.
    """
    form = parent.form
    rule = form.below_floor[component]
    if rule != "taylor":
        raise InputError(
            f"The parent's DF states the rule {rule!r} below its table, not the "
            "Taylor form of a truncated DF: the child's edge expansion needs a "
            "truncated parent."
        )
    f_1, f_2, k_switch = (float(c) for c in form.below_floor_coefficients[component])
    profile = parent.profile
    return edge_expansion(f_1, f_2, k_switch, profile.r_t, profile.mass_t, anisotropy)


def edge_law(expansion: EdgeExpansion) -> tuple[float, float, float]:
    r"""Return the child's edge law (p, C_0, C_1), the ``"edge_law"`` rule.

    :math:`F(\mathcal{K}) = \mathcal{K}^p (C_0 + C_1 \mathcal{K})` with
    :math:`p = 1 + \beta_{\rm c}`; each :math:`C_q` is the amplitude of the
    DF a pure power :math:`a_q P^q` inverts to (`edge.df_edge_amplitude`).
    """
    bc = expansion.beta_c
    c_0 = df_edge_amplitude(expansion.coefficients[2.5], 2.5, bc)
    c_1 = df_edge_amplitude(expansion.coefficients[3.5], 3.5, bc)
    return df_edge_exponent(2.5, bc), float(c_0), float(c_1)


def floor_tail(expansion: EdgeExpansion) -> dict[str, Any]:
    """Return the input of `rows.method2_floor_tail` for the expansion:
    the coefficients per power, the floor pair (m_f, n_f) of the child's
    central anisotropy and the prefactor C(n_f)."""
    bc = expansion.beta_c
    pair = order_pair(bc)
    return {
        "a_q": dict(expansion.coefficients),
        "m_f": pair.order_m,
        "n_f": pair.order_n,
        "pref": inversion_prefactor(pair.order_n, bc),
    }


def floor_value(expansion: EdgeExpansion, order: int, p: float) -> float:
    r"""Return :math:`\varrho_{\rm aug}^{(k)}(P)` from the expansion at a
    potential below the switch: the exact source of the boundary term the
    kernel carries at the table's floor."""
    p = float(p)
    if not 0.0 <= p <= expansion.k_switch:
        raise InputError(
            f"The expansion holds for 0 <= P <= k_switch = {expansion.k_switch}; "
            f"got P = {p}."
        )
    total = 0.0
    for q, a in expansion.coefficients.items():
        total += (
            a
            * gamma_function(q + 1.0)
            / gamma_function(q - order + 1.0)
            * p ** (q - order)
        )
    return float(total)
