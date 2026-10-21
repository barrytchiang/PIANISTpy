r"""Edge expansion of a Method II child's augmented density.

A Method II child is a DF inverted from the self-consistent profile of a
truncated parent model. Near the parent's edge, the child's augmented
density is a sum of two powers of the potential. This module returns the
two-term expansion (`edge_expansion`, `edge_expansion_at_node`,
`edge_expansion_of_parent`), the child's closed-form DF below the parent's
switch energy (`edge_law`), and the terms the child's inversion reads at
the floor of its table (`floor_value`, `floor_tail`).

Below the parent's switch energy, the parent's DF is the Taylor form
:math:`F = F_1 \mathcal{K} + F_2 \mathcal{K}^2/2` (`AsymptoticForm`, rule
``"taylor"``), with :math:`\mathcal{K}` the energy measured from the edge.
To the order kept, the parent's density near the edge is

.. math::

    \varrho(P) = 4\pi\sqrt{2}\,\Big[\tfrac{4}{15} F_1 P^{5/2}
                 + \tfrac{8}{105} F_2 P^{7/2}\Big],

with :math:`P` the potential measured from the edge. The child's augmented
density is :math:`\varrho_{\rm aug} = h(R)\,\varrho`, with :math:`h = 1/g`
the inverse of the child's anisotropy factor :math:`g`
(`Anisotropy.factor`). The expansion takes :math:`h` as linear in
:math:`P` at the edge, :math:`h = h_{\rm t} + H_1 P`, with
:math:`h_{\rm t} = h(R_{\rm t})`, :math:`H_1 = h'(R_{\rm t})\,R'(P)` and
:math:`R'(P) = -R_{\rm t}^2/M_{\rm t}` from the Poisson equation, where
:math:`R_{\rm t}` is the parent's truncation radius and :math:`M_{\rm t}`
the parent's total mass. The augmented density is then

.. math::

    \varrho_{\rm aug}(P) = a_{5/2} P^{5/2} + a_{7/2} P^{7/2} + O(P^{9/2}).

Each power of :math:`P` in :math:`\varrho_{\rm aug}` inverts to a power of
:math:`\mathcal{K}` in the DF (`edge.df_edge_amplitude`). Below the
switch energy, the child's DF is therefore the edge law
:math:`F = \mathcal{K}^p (C_0 + C_1 \mathcal{K})` with
:math:`p = 1 + \beta_{\rm c}` (`AsymptoticForm`, rule ``"edge_law"``).
The coefficients :math:`a_{5/2}` and :math:`a_{7/2}` also give the child's
inversion two exact terms at the floor of its table: the value of the
derivative :math:`\varrho_{\rm aug}^{(m_{\rm f})}` that enters the kernel's
boundary term, and the input of `rows.method2_floor_tail`, the exact
integral below the floor.
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

# The constant 4 pi sqrt(2) is the isotropic forward prefactor 4 pi times the
# factor sqrt(2) of the kernel (2 (P - K))^(1/2) of the density integral.
_A = 4.0 * np.pi * np.sqrt(2.0)


@dataclass(frozen=True)
class EdgeExpansion:
    r"""The two-term expansion of a Method II child's augmented density.

    Parameters
    ----------
    coefficients : dict of float to float
        Coefficients :math:`a_q` keyed by the powers :math:`q = 5/2` and
        :math:`q = 7/2`, with
        :math:`\varrho_{\rm aug}(P) = \sum_q a_q P^q` and :math:`P` the
        potential measured from the parent's edge.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}` of the child. The value is 0
        for an OM child.
    k_switch : float
        Switch energy of the parent. The expansion holds for potentials from
        0 to `k_switch`.
    h_t : float
        Constant term :math:`h_{\rm t}` of the inverse anisotropy factor
        :math:`h = h_{\rm t} + H_1 P`.
    h_1 : float
        Slope :math:`H_1` of the inverse anisotropy factor in :math:`P`.
    """

    coefficients: dict[float, float]
    beta_c: float
    k_switch: float
    h_t: float
    h_1: float

    def density(self, p: Any) -> np.ndarray:
        """Return the expansion's augmented density at the potentials `p`."""
        p = np.asarray(p, dtype=np.float64)
        return sum((a * p**q for q, a in self.coefficients.items()), np.zeros_like(p))


def _inverse_factor_and_slope(
    anisotropy: Anisotropy, r_t: float
) -> tuple[float, float]:
    """Return the inverse factor h = 1/g and its slope dh/dR at the radius ``r_t``.

    An isotropic child has h = 1 and dh/dR = 0. For every other family, the
    function returns the values of `inverse_factor` at ``r_t``.
    """
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
    r"""Return the edge expansion of a Method II child's augmented density.

    Parameters
    ----------
    f_1 : float
        Linear Taylor coefficient :math:`F_1` of the parent's DF,
        :math:`F = F_1 \mathcal{K} + F_2 \mathcal{K}^2 / 2` below
        `k_switch`.
    f_2 : float
        Quadratic Taylor coefficient :math:`F_2` of the parent's DF.
    k_switch : float
        Switch energy of the parent, in internal units. Must be positive.
    r_t : float
        Truncation radius :math:`R_{\rm t}` of the parent, in internal
        units. Must be positive.
    m_t : float
        Total mass :math:`M_{\rm t}` of the parent, in the unit of
        `DoublePowerLaw.mass`, the unit in which
        :math:`\mathrm{d}P/\mathrm{d}R = -M/R^2`. Must be positive.
    anisotropy : Anisotropy
        Anisotropy model of the child.

    Returns
    -------
    expansion : EdgeExpansion
        The coefficients :math:`a_{5/2}` and :math:`a_{7/2}`, the child's
        central anisotropy, `k_switch`, :math:`h_{\rm t} = h(R_{\rm t})`
        and :math:`H_1 = h'(R_{\rm t}) R'(P)`.

    Raises
    ------
    InputError
        If `r_t` or `m_t` is not positive.
    InputError
        If `k_switch` is not positive.
    InputError
        If the child is a Type II model and `r_t` is at or beyond the
        child's anisotropy radius :math:`R_{\rm a}`, where `inverse_factor`
        does not define the factor.
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
    r"""Return the edge expansion with the inverse factor read at a node.

    The child's inversion calls `edge_expansion_at_node` at the floor node
    of the child's table. The inverse factor is taken as linear in
    :math:`P` between the edge and the node, :math:`h = h_{\rm t} + H_1 P`,
    with :math:`H_1` the
    slope of :math:`h` at the node and
    :math:`h_{\rm t} = h(P_{\rm node}) - H_1 P_{\rm node}`. With
    :math:`P_{\rm node} = 0`, `parent_h` set to None, and `h_node` and
    `h_slope` equal to :math:`h(R_{\rm t})` and :math:`H_1` of
    `edge_expansion`, the result equals the result of `edge_expansion`.

    For an Osipkov-Merritt Type I parent, the density integral over the
    parent's DF gives the parent's own augmented density
    :math:`h_{\rm p}\,\hat\varrho`, with :math:`\hat\varrho` the parent's
    density, :math:`h_{\rm p} = 1 + R^2/r_{\rm a,p}^2` and
    :math:`r_{\rm a,p}` the parent's anisotropy radius. The factor applied to
    the density integral of the parent's Taylor form is then
    :math:`h / h_{\rm p}`, with slope
    :math:`(H_1 h_{\rm p} - h\,H_{\rm p,1}) / h_{\rm p}^2` and
    :math:`H_{\rm p,1}` the slope of :math:`h_{\rm p}` in :math:`P`.

    Parameters
    ----------
    f_1 : float
        Linear Taylor coefficient :math:`F_1` of the parent's DF.
    f_2 : float
        Quadratic Taylor coefficient :math:`F_2` of the parent's DF.
    k_switch : float
        Switch energy of the parent, in internal units.
    p_node : float
        Potential :math:`P_{\rm node}` of the node, measured from the edge.
    h_node : float
        The child's inverse factor :math:`h` at the node.
    h_slope : float
        Slope :math:`H_1` of the child's inverse factor in :math:`P` at the
        node.
    anisotropy : Anisotropy
        Anisotropy model of the child.
    parent_h : tuple of float, optional
        The pair (:math:`h_{\rm p}`, :math:`H_{\rm p,1}`) of an OM Type I
        parent at the node. Default: ``None``, which means an isotropic
        parent.

    Returns
    -------
    expansion : EdgeExpansion
        The expansion built on :math:`h_{\rm t}` and :math:`H_1` at the node.
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
    """Return the edge expansion of a Method II child from its parent model.

    The parent's `AsymptoticForm` must state the ``"taylor"`` rule below
    the parent's table. A truncated Method I solve writes the ``"taylor"``
    rule. For a parent whose form states any other rule,
    `edge_expansion_of_parent` raises InputError, because the Taylor
    coefficients are the only input the expansion is derived from.

    Parameters
    ----------
    parent : Model
        The truncated parent model. The call reads the Taylor coefficients
        and the switch energy from ``parent.form``, and ``r_t`` and
        ``mass_t`` from ``parent.profile``.
    anisotropy : Anisotropy
        Anisotropy model of the child.
    component : int, optional
        Index of the parent's component the child is built on. Default:
        ``0``, which means the first component.

    Returns
    -------
    expansion : EdgeExpansion
        The result of `edge_expansion` for the parent's coefficients, switch
        energy, truncation radius and total mass.

    Raises
    ------
    InputError
        If the parent's form states a rule other than ``"taylor"`` below
        the parent's table.
    InputError
        If `edge_expansion` raises InputError for the parent's values.
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
    r"""Return the exponent and the two amplitudes of a child's edge law.

    The edge law is the ``"edge_law"`` rule of `AsymptoticForm`,
    :math:`F(\mathcal{K}) = \mathcal{K}^p (C_0 + C_1 \mathcal{K})` with
    :math:`p = 1 + \beta_{\rm c}`. The amplitude :math:`C_0` is the
    amplitude of the DF that the power :math:`a_{5/2} P^{5/2}` inverts to,
    and :math:`C_1` the amplitude of the DF that :math:`a_{7/2} P^{7/2}`
    inverts to (`edge.df_edge_amplitude`).

    Parameters
    ----------
    expansion : EdgeExpansion
        The child's edge expansion.

    Returns
    -------
    law : tuple of float
        The triple (:math:`p`, :math:`C_0`, :math:`C_1`).

    Raises
    ------
    UnphysicalModelError
        If the child's central anisotropy is not above :math:`-2`, the
        bound below which a density ending as :math:`P^{5/2}` has no DF
        (`edge.check_edge_existence`).
    """
    bc = expansion.beta_c
    c_0 = df_edge_amplitude(expansion.coefficients[2.5], 2.5, bc)
    c_1 = df_edge_amplitude(expansion.coefficients[3.5], 3.5, bc)
    return df_edge_exponent(2.5, bc), float(c_0), float(c_1)


def floor_tail(expansion: EdgeExpansion) -> dict[str, Any]:
    r"""Return the input of `rows.method2_floor_tail` for an edge expansion.

    The returned dict holds four keys: ``"a_q"``, the expansion's
    coefficients keyed by power; ``"m_f"`` and ``"n_f"``, the integer part
    and the remainder of the order :math:`3/2 - \beta_{\rm c}`, with
    :math:`\beta_{\rm c}` the child's central anisotropy (`order_pair`); and
    ``"pref"``, the inversion prefactor :math:`C(n_{\rm f})`
    (`inversion_prefactor`).
    """
    bc = expansion.beta_c
    pair = order_pair(bc)
    return {
        "a_q": dict(expansion.coefficients),
        "m_f": pair.order_m,
        "n_f": pair.order_n,
        "pref": inversion_prefactor(pair.order_n, bc),
    }


def floor_value(expansion: EdgeExpansion, order: int, p: float) -> float:
    r"""Return a derivative of the expansion's augmented density at a potential.

    The child's inversion reads the derivative of order `order` at the
    floor of its table, as the exact value in the kernel's boundary term.

    Parameters
    ----------
    expansion : EdgeExpansion
        The child's edge expansion.
    order : int
        Order :math:`m` of the derivative
        :math:`\varrho_{\rm aug}^{(m)}(P)`.
    p : float
        Potential :math:`P`, measured from the edge. Must lie in
        ``[0, expansion.k_switch]``.

    Returns
    -------
    value : float
        :math:`\sum_q a_q\,\Gamma(q + 1)/\Gamma(q - m + 1)\,P^{q - m}`.

    Raises
    ------
    InputError
        If `p` lies outside ``[0, expansion.k_switch]``.
    """
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
