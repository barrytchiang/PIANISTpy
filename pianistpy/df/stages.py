"""Frozen result classes of the stages that compute and truncate a DF table.

`compute_df` and `truncate` call the stages in turn. Each stage returns one
of the frozen result classes of this module. Each stage is a pure function of
its arguments and reads no module-level state. The stages pass data to one
another only through the result objects of this module.

The inversion of a distribution function (DF) runs the following stages in
order:

- `pair.potential_density_pair` returns the potential-density pair, a
  `PotentialDensityPair`.
- `form.inversion_form` returns the form of the inversion on each row of the
  energy grid, an `InversionForm`.
- `derivatives.aux_derivatives` returns the exact derivatives of the
  auxiliary density that the inversion needs, an `AuxDerivatives`.
- `invert.inversion_df` returns the energy grid (`EnergyGrid`), the DF table
  (`DfTable`), the analytic forms beyond the two ends of the table
  (`AsymptoticForm`), the record of the inversion (`InversionRecord`) and
  the inversion's rows before any row is replaced (`InversionRows`).
- `report.validate_df` returns the validation report of the DF.

The truncation of a model runs the following stages in order:

- `truncation.truncated_source` returns the truncated density and its edge.
- `poisson.run_poisson_ode` solves the Poisson equation for the truncated
  model's potential.
- `truncated.truncate_model` returns the truncated model and its validation
  report.

A Method II model runs the inversion a second time, on the tabulated
potential-density pair of a truncated parent model.
"""

__all__ = [
    "AsymptoticForm",
    "AuxDerivatives",
    "DfTable",
    "EnergyGrid",
    "InversionForm",
    "InversionRecord",
    "InversionRows",
    "PotentialDensityPair",
    "TruncatedProfile",
]

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import InputError
from ..numerics.pytrees import register_result
from ..profiles import TruncatedProfile
from ..typing import Array
from .inputs import Component


@dataclass(frozen=True)
class PotentialDensityPair:
    """The total relative potential and the density of each component.

    A potential-density pair is either analytic or tabulated. An analytic
    pair is given by the components' closed forms, which have exact
    derivatives. A tabulated pair is given by the truncated profile of a
    parent model, for a Method II model.

    Parameters
    ----------
    components : tuple of Component
        The components, in the order of the first axis of `rho`.
    is_analytic : bool
        True if the pair is given by the components' closed forms; False if
        the pair is tabulated.
    p_0 : float
        Total relative potential at the center, in internal units.
    r : array, shape (n,)
        Radii of the tabulated pair, in internal units. Empty for an analytic
        pair, whose closed forms are evaluated at the radii a stage needs.
    p : array, shape (n,)
        Total relative potential at the radii `r`, in internal units. Empty
        for an analytic pair.
    rho : array, shape (n_components, n)
        Density of each component at the radii `r`, in internal units. Empty
        for an analytic pair.
    """

    components: tuple[Component, ...]
    is_analytic: bool
    p_0: float
    r: Array
    p: Array
    rho: Array


@dataclass(frozen=True)
class EnergyGrid:
    r"""The energies at which each component's DF is tabulated, and their radii.

    Parameters
    ----------
    k : array, shape (n_components, n)
        Relative energy :math:`\mathcal{K}` of each node of each component's
        table, increasing along the second axis, in internal units. The
        energies `k` are the grid of `DfTable.k`.
    r_of_k : array, shape (n_components, n)
        Radius at which the total relative potential equals the energy `k`,
        in internal units. NaN where a stage has not computed the radii.
    k_floor : tuple of float
        First (lowest) energy of each component's table, in internal units.
    k_ceiling : tuple of float
        Last (highest) energy of each component's table, in internal units.
    """

    k: Array
    r_of_k: Array
    k_floor: tuple[float, ...]
    k_ceiling: tuple[float, ...]


@dataclass(frozen=True)
class AuxDerivatives:
    """Derivatives of each component's auxiliary density in the potential.

    Parameters
    ----------
    orders : tuple of int
        The derivative orders held, in the order of the second axis of
        `values` and of `at_floor`.
    values : array, shape (n_components, n_orders, n)
        Derivative of each order in `orders` of each component's auxiliary
        density, at the energies of the energy grid, in internal units.
    at_floor : array, shape (n_components, n_orders)
        Derivative of each order in `orders` of each component's auxiliary
        density at the lowest energy of the energy grid, in internal units.
    is_exact : bool
        True if `values` and `at_floor` are evaluated from closed forms;
        False if they are read from a table.
    source : object, optional
        The closed forms from which `values` and `at_floor` are evaluated,
        and which the method `at` evaluates. Default: ``None``, which means
        that the derivatives are read from a table.
    """

    orders: tuple[int, ...]
    values: Array
    at_floor: Array
    is_exact: bool
    source: Any = field(default=None, repr=False, compare=False)

    def at(self, p: Array, order: int) -> Array:
        r"""Return a derivative of the auxiliary density from its closed forms.

        The derivative is
        :math:`\mathrm{d}^{n}\rho_{\rm aux}/\mathrm{d}P^{n}`, of the auxiliary
        density :math:`\rho_{\rm aux}` with respect to the relative potential
        :math:`P`, of order :math:`n` = ``order``.

        Parameters
        ----------
        p : array
            Relative potentials :math:`P` at which the derivative is
            evaluated, in internal units.
        order : int
            Order :math:`n` of the derivative; ``0`` returns the auxiliary
            density itself.

        Returns
        -------
        derivative : array, shape of `p`
            The derivative of order ``order`` at each potential in `p`, in
            internal units. Zero where a potential in `p` lies outside the
            open interval (0, 1).

        Raises
        ------
        InputError
            If the derivatives are read from a table (`source` is None): a
            table holds only its own orders at its own energies.
        """
        if self.source is None:
            raise InputError(
                "These derivatives come from a table: only its own orders and "
                "energies are held."
            )
        p = np.asarray(p, dtype=np.float64)
        out = np.zeros_like(p)
        ok = (p > 0.0) & (p < 1.0)
        if ok.any():
            values = self.source.derivs_at_p(p[ok], max(int(order), 1))
            out[ok] = values[order if order <= 1 else 2]
        return out


@dataclass(frozen=True)
class InversionForm:
    """The form of the inversion on every row of the energy grid.

    `form.inversion_form` chooses the form from the model's closed forms
    alone; no computed number of the inversion enters the choice.

    Parameters
    ----------
    order_j : tuple of int
        Derivative order :math:`j` of each component's inversion, from the
        unified order pair of `inversion.order_pair`.
    kernel_exponent : tuple of float
        Kernel exponent of each component's inversion, from the unified
        order pair of `inversion.order_pair`. The kernel exponent lies in
        :math:`(-1/2, 1/2]`.
    form : array of int, shape (n_components, n)
        Form of the inversion on each row, coded as in `form.FORMS`: ``0``
        for the local closed form, ``1`` for the form with the surface term
        (integrated by parts), ``2`` for the general form (the derivative
        outside the integral).
    k_interior : tuple of float or None
        Energy of the interior point of each component, in internal units;
        None for a component that has no interior point.

    References
    ----------
    Notes Eq. (T:2.16).
    """

    order_j: tuple[int, ...]
    kernel_exponent: tuple[float, ...]
    form: Array
    k_interior: tuple[float | None, ...]


@dataclass(frozen=True)
class DfTable:
    r"""The distribution function (DF) of each component on its energy grid.

    The components of one model may have different energy grids: each
    member of a superposition keeps its own grid.

    Parameters
    ----------
    k : array, shape (n_components, n)
        Relative energies :math:`\mathcal{K}` of the nodes of each
        component's table, in internal units, increasing along the
        component's support. Where one component's grid is shorter than
        another component's grid, the shorter grid's last energy is repeated
        after the index `i_last`, and no code reads the repeated entries.
    log_df : array, shape (n_components, n)
        Base-10 logarithm of the DF :math:`F` at the energies `k`, in
        internal units. NaN outside the component's support; minus infinity
        where the DF is exactly zero.
    log_df_slope : array, shape (n_components, n)
        Slope :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K} =
        \mathcal{K}F'/(F\ln 10)` at the energies `k`, with
        :math:`F' = \mathrm{d}F/\mathrm{d}\mathcal{K}`. The slope is finite
        wherever the DF is positive. The derivative :math:`F'` itself
        exceeds the range of a double at a steep center, so the table holds
        the slope of the logarithm instead.
    i_first : tuple of int
        Index of the first node of each component's support in `k`.
    i_last : tuple of int
        Index of the last node of each component's support in `k`.
    slopes_exact : tuple of bool
        True for a component whose `log_df_slope` and `log_df_slope_below`
        hold the exact slope of the DF on every node of the support. Where
        False, no code reads the component's slope columns, and the lookup
        takes the slopes from a cubic spline through the nodes.
    k_below : array, shape (n_components, m)
        Energies below the first energy of `k` at which the inversion's exact
        completion below the table was evaluated, in internal units. The
        last energy of `k_below` may equal the first energy of `k`, where the
        join of `k_below` and `k` is measured; the lookup reads only the
        energies strictly below the first energy of `k`. The first
        `n_below` entries of a component are its nodes, and the remaining
        entries repeat the last node. Empty for a DF that has no nodes below
        the table, such as a truncated DF, whose continuation below the
        table is the Taylor form of `AsymptoticForm`. Default: an empty
        array.
    log_df_below : array, shape (n_components, m)
        Base-10 logarithm of the DF at the energies `k_below`. Between the
        first energy of `k_below` and the first energy of `k`, the lookup
        reads the nodes of `k_below`; below the first energy of `k_below`, the lookup
        reads the forms of `AsymptoticForm`. Default: an empty array.
    log_df_slope_below : array, shape (n_components, m)
        Slope :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K}` at the
        energies `k_below`. Default: an empty array.
    sign : array of int8, shape (n_components, n)
        Sign of each DF at the energies `k`, held only where some DF of the
        table is negative. A DF with negative values is not a valid
        equilibrium. Where `sign` is not empty, `log_df` holds the logarithm
        of the absolute value of the DF. Default: an empty array, which
        means that every DF of the table is non-negative.
    n_below : tuple of int
        Number of nodes below the table of each component. Default: ``()``,
        which means that every column of `k_below` is a node.
    """

    k: Array
    log_df: Array
    log_df_slope: Array
    i_first: tuple[int, ...]
    i_last: tuple[int, ...]
    slopes_exact: tuple[bool, ...]
    k_below: Array = field(default_factory=lambda: np.zeros((0, 0)))
    log_df_below: Array = field(default_factory=lambda: np.zeros((0, 0)))
    log_df_slope_below: Array = field(default_factory=lambda: np.zeros((0, 0)))
    sign: Array = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    n_below: tuple[int, ...] = ()

    def nodes_below(self, component: int) -> int:
        """Return the number of nodes below the table of one component.

        Parameters
        ----------
        component : int
            Index of the component.

        Returns
        -------
        n : int
            The component's entry of `n_below`. Where `n_below` is empty, the
            number of columns of a two-dimensional `k_below`, or zero where
            `k_below` is not two-dimensional.
        """
        if self.n_below:
            return int(self.n_below[component])
        return int(self.k_below.shape[1]) if self.k_below.ndim == 2 else 0


@dataclass(frozen=True)
class AsymptoticForm:
    r"""The analytic form of each DF beyond the two ends of its table.

    The stage that makes an `AsymptoticForm` states every field explicitly;
    a consumer infers nothing from a field's absence. Every field except
    `p_0` holds one entry per component. The forms above the table are
    written in the depth :math:`u = P_0 - \mathcal{K}`, with :math:`P_0` the
    field `p_0` and :math:`\mathcal{K}` the relative energy.

    Parameters
    ----------
    p_0 : float
        Relative potential :math:`P_0` at the center of the model, in
        internal units; the support of every DF of the model ends at the
        energy :math:`\mathcal{K} = P_0`.
    central_label : tuple of str
        Classification of the center of each component's DF.
    central_exponent : tuple of float
        Exponent of each DF's form above the last energy of the table.
    central_amplitude : tuple of float
        Amplitude of each DF's form above the last energy of the table.
    tail_exponent : tuple of float
        Exponent :math:`a` of each DF's tail below the first energy of the
        table.
    tail_amplitude : tuple of float
        Amplitude :math:`A` of each DF's tail below the first energy of the
        table; zero where the DF has no power-law tail.
    tail_log_exponent : tuple of float
        Exponent :math:`b` of the logarithmic factor of each tail; zero for a
        pure power law.
    central_form : tuple of str
        Form of each DF above the table: ``"power"`` (:math:`A u^s`),
        ``"constant_plus_power"``, ``"log"`` (:math:`L \ln u`),
        ``"analytic"`` (finite at the center), or ``"fitted"`` (the center
        has no closed form, and the exponent was measured on the table).
    central_log_coefficient : tuple of float
        Coefficient :math:`L` of the ``"log"`` form; NaN for every other
        form.
    central_next_form : tuple of str
        Next singular term of each DF above the table: ``"power"``
        (:math:`c u^e`), ``"log_power"`` (:math:`c u^e (-\ln u)`, with
        :math:`e` an integer), or ``"none"``.
    central_next_exponent : tuple of float
        Exponent :math:`e` of the next singular term; NaN where the DF has
        no next singular term.
    central_next_coefficient : tuple of float
        Coefficient :math:`c` of the next singular term; NaN where the DF
        has no next singular term.
    tail_log_scale : tuple of float
        Scale :math:`E_0` of the tail's logarithmic factor, in the tail
        :math:`A\mathcal{K}^{-a}[\ln(E_0/\mathcal{K})]^{-b}`.
    below_floor : tuple of str
        Rule of each DF below the first energy of the table, and below the
        first energy of `DfTable.k_below` where `DfTable.k_below` is not
        empty: ``"tail"`` (the tail of `tail_amplitude`, `tail_exponent`,
        `tail_log_exponent` and `tail_log_scale`), ``"taylor"``
        (:math:`F_1\mathcal{K} + F_2\mathcal{K}^2/2`, for a truncated DF),
        ``"edge_law"`` (:math:`\mathcal{K}^p(C_0 + C_1\mathcal{K})`, for a
        Method II DF), or ``"zero"``.
    below_floor_coefficients : tuple of tuple of float
        Coefficients of the rule in `below_floor`: :math:`(F_1, F_2,
        \mathcal{K}_{\rm switch})` for ``"taylor"``, :math:`(p, C_0, C_1)`
        for ``"edge_law"``, and an empty tuple for ``"tail"`` and
        ``"zero"``.
    central_anchor : tuple of float
        Anchor energy of each DF, in internal units: where the inversion's
        rows end below the ceiling of the energy grid, the table ends at the
        anchor energy and the center's form continues the table above the
        anchor energy. NaN where the inversion's rows reach the ceiling.
        Default: ``()``, which means that no component has an anchor.
    central_anchor_amplitude : tuple of float
        Singular part of the DF at the anchor, :math:`A u_{\rm a}^\kappa`,
        with :math:`u_{\rm a} = P_0 - \mathcal{K}_{\rm anchor}` and
        :math:`\kappa` the field `central_exponent`. Above the anchor the DF
        is :math:`A u_{\rm a}^\kappa (u/u_{\rm a})^\kappa + B + D u`
        (`central_df`). NaN where the DF has no anchor. Default: ``()``.
    central_regular : tuple of tuple of float
        Regular part :math:`(B, D)` of the DF's form above the anchor. Zeros
        where the regular part was not fitted; NaN where the DF has no
        anchor. Default: ``()``.
    """

    p_0: float
    central_label: tuple[str, ...]
    central_exponent: tuple[float, ...]
    central_amplitude: tuple[float, ...]
    tail_exponent: tuple[float, ...]
    tail_amplitude: tuple[float, ...]
    tail_log_exponent: tuple[float, ...]
    central_form: tuple[str, ...]
    central_log_coefficient: tuple[float, ...]
    central_next_form: tuple[str, ...]
    central_next_exponent: tuple[float, ...]
    central_next_coefficient: tuple[float, ...]
    tail_log_scale: tuple[float, ...]
    below_floor: tuple[str, ...]
    below_floor_coefficients: tuple[tuple[float, ...], ...]
    central_anchor: tuple[float, ...] = ()
    central_anchor_amplitude: tuple[float, ...] = ()
    central_regular: tuple[tuple[float, float], ...] = ()

    def central_df(self, k: Array, component: int = 0) -> Array:
        r"""Return the DF of one component above its anchor, from the center's form.

        Parameters
        ----------
        k : array
            Relative energies :math:`\mathcal{K}` above the component's anchor
            energy, in internal units.
        component : int, optional
            Index of the component. Default: ``0``, which means the first
            component.

        Returns
        -------
        f : array, shape of `k`
            The DF :math:`A u_{\rm a}^\kappa (u/u_{\rm a})^\kappa + B + D u`
            at the energies `k`, in internal units, with
            :math:`u = P_0 - \mathcal{K}` and the symbols of
            `central_anchor_amplitude` and `central_regular`. The ratio
            :math:`u/u_{\rm a}` and the depth :math:`u_{\rm a}` are each held
            at or above :math:`10^{-300}` so that the power stays finite.
        """
        u_a = self.p_0 - float(self.central_anchor[component])
        b, d = self.central_regular[component]
        u = self.p_0 - np.asarray(k, dtype=np.float64)
        ratio = np.maximum(u / max(u_a, 1e-300), 1e-300)
        kappa = float(self.central_exponent[component])
        a_anchor = float(self.central_anchor_amplitude[component])
        return a_anchor * ratio**kappa + b + d * u

    def exponent_at_zero(self, component: int = 0) -> float | None:
        r"""Return the exponent of one component's DF as the energy goes to zero.

        Parameters
        ----------
        component : int, optional
            Index of the component. Default: ``0``, which means the first
            component.

        Returns
        -------
        exponent : float or None
            Exponent :math:`p` of :math:`F \propto \mathcal{K}^p` as
            :math:`\mathcal{K} \to 0`. For the rule ``"tail"``, :math:`p` is
            minus the field `tail_exponent`; for the rule ``"edge_law"``,
            :math:`p` is the edge law's own exponent. None for the rules
            ``"taylor"`` and ``"zero"``, under which the DF vanishes as
            :math:`\mathcal{K} \to 0`.
        """
        rule = self.below_floor[component]
        if rule == "tail":
            return -self.tail_exponent[component]
        if rule == "edge_law":
            return self.below_floor_coefficients[component][0]
        return None

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        r"""Whether each component's DF diverges as the energy goes to zero.

        An entry is True where `exponent_at_zero` returns a negative
        exponent :math:`p`. A DF that goes as :math:`\mathcal{K}^p` with
        :math:`-1 < p < 0` as :math:`\mathcal{K} \to 0` is integrable and
        gives a finite density, but such a DF is not monotonic in
        :math:`\mathcal{K}`. Two examples of such a DF are an untruncated
        Osipkov-Merritt model with a shallow outer density slope, and a
        Method II model whose central anisotropy is lower than the central
        anisotropy of its parent model by more than 1.
        """
        exponents = (self.exponent_at_zero(i) for i in range(len(self.below_floor)))
        return tuple(p is not None and p < 0.0 for p in exponents)


@dataclass(frozen=True)
class InversionRecord:
    r"""Record of how each DF's table was made: the inversion and the rows replaced.

    Every field holds one entry per component.

    Parameters
    ----------
    form : tuple of str
        Evaluation form of the inversion: ``"integrated_by_parts"``,
        ``"derivative_outside"`` or ``"closed_form_n0"``. The forms of the
        rows above the interior point or below the floor of the energy grid
        follow after a ``+``.
    order_j : tuple of int
        Derivative order :math:`j` of the unified order pair
        :math:`(j, n)`.
    order_n : tuple of float
        Kernel exponent :math:`n` of the unified order pair :math:`(j, n)`.
    floor_m : tuple of int
        Derivative order :math:`m` of the order pair of the floor rows.
    floor_n : tuple of float
        Kernel exponent :math:`n` of the order pair of the floor rows.
    k_interior : tuple of float
        Energy :math:`P_s` of the interior point; NaN where the component
        has no interior point.
    interior_exponent : tuple of float
        Exponent :math:`s` at the interior point; NaN where the component
        has no interior point.
    outer_power : tuple of float
        Exponent of the auxiliary density as :math:`P \to 0`, recorded where
        the auxiliary density's derivative of the floor order is unbounded
        as :math:`P \to 0`; NaN otherwise.
    outer_end : tuple of str
        ``"unbounded"`` where the auxiliary density's derivative of the
        floor order is unbounded as :math:`P \to 0`, so that the completion
        below the floor takes the general form; ``"bounded"`` otherwise.
    high_k_override_k : tuple of float
        Energy from which an analytic form replaced the table's rows toward
        the center, in internal units; the central potential where no row
        was replaced.
    high_k_override_r : tuple of float
        Radius at which the total relative potential equals
        `high_k_override_k`, in internal units; zero where no row was
        replaced.
    high_k_override_rows : tuple of int
        Number of rows that an analytic form replaced toward the center.
    high_k_override_rule : tuple of str
        Rule of the analytic form that replaced rows toward the center;
        ``"none"`` where no row was replaced.
    low_k_override_k : tuple of float
        Energy below which an analytic form replaced the table's rows toward
        the floor, in internal units; zero where no row was replaced.
    low_k_override_r : tuple of float
        Outer radius of the model, in internal units: the truncation radius
        of a truncated model, and infinity for an untruncated model.
    method2_bands : tuple of dict
        Band record of a Method II child model: the anchor of the top band,
        the width of the floor band, whether the floor term is exact, and
        the number of negative values in each band. Default: ``()``, which
        is the value for every result other than a Method II child model.
    critical_band : tuple of dict
        Closed-form band of a Method II child model on the critical branch,
        or the reason the closed-form band was rejected. Default: ``()``,
        which means that the result has no closed-form band.
    continuation : tuple of str
        Auxiliary density that the inversion integrated beyond the edge of
        the anisotropy's domain. A Type II model has such an edge at the
        anisotropy radius :math:`R_{\rm a}`: the DF above the energy
        :math:`P(R_{\rm a})` is non-local in the auxiliary density beyond
        :math:`R_{\rm a}`. ``"none"`` where the anisotropy's domain has no
        edge. Default: ``()``, which means that the continuation is not
        recorded.
    deficit_degree : tuple of int
        Degree of the interpolant of the central deficit of each component
        of a truncated model; zero where no deficit was fitted. Default:
        ``()``, which is the value for every result other than a truncated
        model.
    """

    form: tuple[str, ...]
    order_j: tuple[int, ...]
    order_n: tuple[float, ...]
    floor_m: tuple[int, ...]
    floor_n: tuple[float, ...]
    k_interior: tuple[float, ...]
    interior_exponent: tuple[float, ...]
    outer_power: tuple[float, ...]
    outer_end: tuple[str, ...]
    high_k_override_k: tuple[float, ...]
    high_k_override_r: tuple[float, ...]
    high_k_override_rows: tuple[int, ...]
    high_k_override_rule: tuple[str, ...]
    low_k_override_k: tuple[float, ...]
    low_k_override_r: tuple[float, ...]
    method2_bands: tuple[dict, ...] = ()
    critical_band: tuple[dict, ...] = ()
    continuation: tuple[str, ...] = ()
    deficit_degree: tuple[int, ...] = ()


@dataclass(frozen=True)
class InversionRows:
    r"""The inversion's rows on its whole energy grid, before any row is replaced.

    The truncation of a model reads these rows.

    Parameters
    ----------
    k : array, shape (n_components, n)
        Relative energies of the inversion's rows, in internal units.
    f : array, shape (n_components, n)
        The DF :math:`F` at the energies `k`, in internal units, with the
        sign the inversion computed.
    ddf : array, shape (n_components, n)
        The derivative :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}` at the
        energies `k`, in internal units, with the sign the inversion
        computed.
    data_top : tuple of float
        Total relative potential at the innermost radius of the inversion's
        data, in internal units: at the radius
        :math:`10^{x}`, with :math:`x` the setting
        ``truncation["log10_r_min"]``, for an analytic pair, and the first
        potential of the parent model's rows for a Method II child model.
        The truncation takes the inversion's rows to reach the ceiling of
        the energy grid where `data_top` is at or above ``1 - K_FLOOR``,
        with ``K_FLOOR`` the potential floor of `truncation.K_FLOOR`.
    """

    k: Array
    f: Array
    ddf: Array
    data_top: tuple[float, ...]


# The stage results are registered as pytrees so that JAX transforms can pass
# the stage results through. The fields listed in each register_result call are
# the pytree's data, and every other field (labels, index bounds, tuples of
# flags) is static. Every
# field of an InversionRecord is static, since an InversionRecord only records
# how a table was made.
register_result(
    DfTable,
    (
        "k",
        "log_df",
        "log_df_slope",
        "k_below",
        "log_df_below",
        "log_df_slope_below",
        "sign",
    ),
)
register_result(EnergyGrid, ("k", "r_of_k"))
register_result(
    AsymptoticForm,
    (
        "central_exponent",
        "central_amplitude",
        "tail_exponent",
        "tail_amplitude",
        "tail_log_exponent",
        "central_log_coefficient",
        "central_next_exponent",
        "central_next_coefficient",
        "tail_log_scale",
        "below_floor_coefficients",
    ),
)
register_result(InversionRecord, ())
register_result(InversionRows, ("k", "f", "ddf"))
