"""Stages of the distribution-function computation and their results.

`compute_df` and `truncate` compose the stages of this module. Each stage is a
pure function of its arguments and returns a frozen result. A stage reads no
module state, and nothing passes between stages except their results.

The stages of the inversion, in order::

    potential_density_pair -> energy_grid -> inversion_form
        -> aux_derivatives -> invert_rows -> asymptotic_form -> validate_df

(`aux_derivatives` takes the `InversionForm`: the form fixes the orders it
needs.)

The stages of the truncation, in order::

    resolve_truncation -> truncated_density -> solve_poisson
        -> validate_model

Method II runs the inversion stages a second time, on the tabulated pair of
a truncated parent model.
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
    "aux_derivatives",
    "asymptotic_form",
    "energy_grid",
    "validate_df",
    "inversion_form",
    "invert_rows",
    "potential_density_pair",
]

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import InputError
from ..numerics.pytrees import register_result
from ..profiles import TruncatedProfile
from ..settings import Settings
from ..typing import Array
from ..validation import ValidationReport
from .inputs import Component


@dataclass(frozen=True)
class PotentialDensityPair:
    """The total potential and the density of each component.

    The pair is either analytic (the components' closed forms, with exact
    derivatives) or tabulated (the truncated profile of a parent model).

    Parameters
    ----------
    components : tuple of Component
        The components, in the order of the arrays below.
    is_analytic : bool
        Whether the pair comes from closed forms.
    p_0 : float
        Central value of the total potential, in internal units.
    r, p : array, shape (n,)
        Radius and total relative potential, for a tabulated pair. Empty for
        an analytic pair, which is evaluated where it is needed.
    rho : array, shape (n_components, n)
        Density of each component at `r`, for a tabulated pair.
    """

    components: tuple[Component, ...]
    is_analytic: bool
    p_0: float
    r: Array
    p: Array
    rho: Array


@dataclass(frozen=True)
class EnergyGrid:
    """The energies at which the DF is tabulated and their radii.

    Parameters
    ----------
    k : array, shape (n_components, n)
        Energy-like variable of the table of each component, increasing, in
        internal units (the grid of `DfTable.k`).
    r_of_k : array, shape (n_components, n)
        Radius at which the total potential equals `k`.
    k_floor, k_ceiling : tuple of float
        First and last energy of the table of each component.
    """

    k: Array
    r_of_k: Array
    k_floor: tuple[float, ...]
    k_ceiling: tuple[float, ...]


@dataclass(frozen=True)
class AuxDerivatives:
    """Derivatives of the auxiliary density with respect to the potential.

    Parameters
    ----------
    orders : tuple of int
        The derivative orders held, in the order of the first axis below.
    values : array, shape (n_components, n_orders, n)
        Derivatives of each component's auxiliary density at the energies of
        the grid.
    at_floor : array, shape (n_components, n_orders)
        The same derivatives at the floor of the grid.
    is_exact : bool
        Whether they come from the closed forms or from a table.
    source : object, optional
        The closed forms the values come from, evaluated by `at`; None for a
        table.
    """

    orders: tuple[int, ...]
    values: Array
    at_floor: Array
    is_exact: bool
    source: Any = field(default=None, repr=False, compare=False)

    def at(self, p: Array, order: int) -> Array:
        """Return d^order rho_aux / dP^order at the energies ``p`` from the
        closed forms (0 outside (0, 1)).

        Raises
        ------
        InputError
            If the derivatives come from a table, which holds its own orders
            only.
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
    """The form of the inversion on every row, chosen analytically.

    Parameters
    ----------
    order_m, order_n : tuple of int, tuple of float
        Order pair in use for each component, after the replacement that
        keeps the kernel exponent within one half.
    form : array of int, shape (n_components, n)
        Form of the inversion on each row: the local form, the form with
        the surface term, or the general form on a split.
    k_interior : tuple of float or None
        Energy of the interior point of each component, where one exists.
    """

    order_m: tuple[int, ...]
    order_n: tuple[float, ...]
    form: Array
    k_interior: tuple[float | None, ...]


@dataclass(frozen=True)
class DfTable:
    """The distribution function of each component on the energy grid.

    Parameters
    ----------
    k : array, shape (n_components, n)
        Energies of the table of each component, increasing along the
        support of the component; beyond `i_last` the last energy is
        repeated where a component's grid is shorter than another's, and
        nothing there is read. Components of one model may have different
        grids (a superposition's members keep their own).
    log_df : array, shape (n_components, n)
        Base-10 logarithm of the DF. NaN outside the support of a component;
        minus infinity where the DF is exactly zero.
    log_df_slope : array, shape (n_components, n)
        Slope of the table in its own variables, d log10 F / d ln K (that is
        K F' / (F ln 10)). It is finite wherever the DF is positive, however
        steep the DF: the derivative F' itself leaves the range of a double
        at a steep center.
    i_first, i_last : tuple of int
        Bounds of the support of each component in the table.
    slopes_exact : tuple of bool
        Whether `log_df_slope` (and `log_df_slope_below`) holds the exact
        slope of each component's DF on every node of its support. Where
        False the column is not read, and a consumer takes its slopes from
        the nodes.
    k_below : array, shape (n_components, m)
        Energies below `k`'s first value at which the inversion's exact
        completion was evaluated (the last may lie on `k`'s first value,
        where the join of the two is measured; the lookup reads the nodes
        strictly below it); empty where the DF has none (a truncated
        DF, whose continuation below the table is the Taylor form of
        `AsymptoticForm`). The first `n_below` entries of a component are
        its nodes; the rest repeat the last one.
    log_df_below, log_df_slope_below : array, shape (n_components, m)
        The DF and its slope at `k_below`. Read below the table's first
        energy before the forms of `AsymptoticForm`, which take over below
        `k_below`'s first value.
    n_below : tuple of int
        Number of nodes below the table of each component; every column of
        `k_below` where the tuple is empty.
    sign : array of int8, shape (n_components, n)
        Sign of each DF where some DF of the table is negative (a DF that is
        no valid equilibrium as it stands): `log_df` is then the logarithm
        of the absolute value. Empty where every DF is non-negative.
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
        """Return the number of nodes below the table of a component."""
        if self.n_below:
            return int(self.n_below[component])
        return int(self.k_below.shape[1]) if self.k_below.ndim == 2 else 0


@dataclass(frozen=True)
class AsymptoticForm:
    """The analytic form of each DF beyond the two ends of its table.

    Every field is stated by whoever makes the form: a consumer infers
    nothing from an absent one.

    Parameters
    ----------
    p_0 : float
        Energy at which the support of every DF ends: the relative potential
        at the center of the model. The forms above the table are written
        in u = p_0 - k.
    central_label : tuple of str
        Classification of the center of each component's DF.
    central_exponent, central_amplitude : tuple of float
        Exponent and amplitude of each DF above the last energy of the table.
    tail_exponent, tail_amplitude : tuple of float
        Exponent and amplitude of each DF below the first energy of the
        table. The amplitude is zero where the DF has no power-law tail.
    tail_log_exponent : tuple of float
        Exponent of the logarithmic factor of each tail; zero for a pure
        power law.
    central_form : tuple of str
        Form of each DF above the table, in u = p_0 - k: ``"power"``
        (A u^s), ``"constant_plus_power"``, ``"log"`` (L ln u), ``"analytic"``
        (finite at the center), or ``"fitted"`` where the center states no
        closed form and the exponent was measured on the table.
    central_log_coefficient : tuple of float
        Coefficient L of the ``"log"`` form; NaN for the others.
    central_next_form : tuple of str
        The next singular term of each DF above the table: ``"power"``
        (c u^e), ``"log_power"`` (c u^e (-ln u), e an integer), or
        ``"none"``.
    central_next_exponent, central_next_coefficient : tuple of float
        Its exponent e and coefficient c; NaN where there is none.
    tail_log_scale : tuple of float
        Scale E_0 of the tail's logarithmic factor,
        A k^-a (ln(E_0 / k))^-b with b `tail_log_exponent`.
    below_floor : tuple of str
        The rule of each DF below the first energy of the table (and below
        `DfTable.k_below` where that is not empty): ``"tail"`` (the tail
        above), ``"taylor"`` (F_1 k + F_2 k^2 / 2, a truncated DF),
        ``"edge_law"`` (k^p (C_0 + C_1 k), a Method II DF), or ``"zero"``.
    below_floor_coefficients : tuple of tuple of float
        The coefficients of that rule: (F_1, F_2, k_switch) for ``"taylor"``,
        (p, C_0, C_1) for ``"edge_law"``, empty for the others.
    central_anchor : tuple of float
        The energy from which the center's form continues the table where
        the inversion's rows end below the ceiling (the table's last
        energy); NaN where the rows reach the ceiling. Empty in a result
        written before the field existed.
    central_anchor_amplitude : tuple of float
        The singular part at the anchor, A u_a^kappa with u_a = p_0 less the
        anchor and kappa `central_exponent`: above the anchor the DF is
        A u_a^kappa (u / u_a)^kappa + B + D u (`central_df`). NaN where there
        is no anchor.
    central_regular : tuple of tuple of float
        The regular part (B, D) of the form above the anchor; zeros where it
        was not fitted, NaN where there is no anchor.
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
        """Return the center's form above a component's anchor.

        Parameters
        ----------
        k : array
            Energies above the anchor.
        component : int, optional
            The component.

        Returns
        -------
        f : array
            A u_a^kappa (u / u_a)^kappa + B + D u, u = p_0 - k.
        """
        u_a = self.p_0 - float(self.central_anchor[component])
        b, d = self.central_regular[component]
        u = self.p_0 - np.asarray(k, dtype=np.float64)
        ratio = np.maximum(u / max(u_a, 1e-300), 1e-300)
        kappa = float(self.central_exponent[component])
        a_anchor = float(self.central_anchor_amplitude[component])
        return a_anchor * ratio**kappa + b + d * u

    def exponent_at_zero(self, component: int = 0) -> float | None:
        r"""Return the exponent of a DF as the energy goes to zero.

        Parameters
        ----------
        component : int, optional
            The component.

        Returns
        -------
        exponent : float or None
            :math:`p` of :math:`F \propto \mathcal{K}^p` as
            :math:`\mathcal{K} \to 0`, where the form states a power there:
            minus the tail's exponent for ``"tail"``, the edge law's own for
            ``"edge_law"``. None for ``"taylor"`` and ``"zero"``, whose DF
            vanishes there.
        """
        rule = self.below_floor[component]
        if rule == "tail":
            return -self.tail_exponent[component]
        if rule == "edge_law":
            return self.below_floor_coefficients[component][0]
        return None

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        r"""Whether each DF diverges as the energy goes to zero.

        A DF that goes as :math:`\mathcal{K}^p` with :math:`-1 < p < 0`
        there is integrable and gives a finite density, but it is not
        monotonic. Examples are an untruncated Osipkov--Merritt model of a
        shallow outer slope, and a Method II model whose central anisotropy
        lies more than one below its parent's.
        """
        exponents = (self.exponent_at_zero(i) for i in range(len(self.below_floor)))
        return tuple(p is not None and p < 0.0 for p in exponents)


@dataclass(frozen=True)
class InversionRecord:
    """How the table of each DF was made: the inversion and the rows replaced.

    One entry per component in every field.

    Parameters
    ----------
    form : tuple of str
        Evaluation form of the inversion: ``"integrated_by_parts"``,
        ``"derivative_outside"`` or ``"closed_form_n0"``, with the forms
        added above the interior point or below the floor after a ``+``.
    order_j, order_n : tuple of int, tuple of float
        The unified order pair (j, n).
    floor_m, floor_n : tuple of int, tuple of float
        The order pair of the floor rows.
    k_interior, interior_exponent : tuple of float
        Energy P_s of the interior point and its exponent s; NaN where the
        component has none.
    outer_power : tuple of float
        Exponent of the auxiliary density at P -> 0 where its floor-order
        derivative is unbounded there; NaN otherwise.
    outer_end : tuple of str
        ``"unbounded"`` where the completion below the floor takes the
        general form for that reason, else ``"bounded"``.
    high_k_override_k, high_k_override_r : tuple of float
        Energy from which an analytic form replaced the rows of the table
        towards the center, and the radius where the potential equals it;
        the central potential and zero where no row was replaced.
    high_k_override_rows : tuple of int
        Number of rows replaced there.
    high_k_override_rule : tuple of str
        The rule that replaced them, ``"none"`` where none did.
    low_k_override_k, low_k_override_r : tuple of float
        Energy below which an analytic form replaced the rows towards the
        floor, zero where none did, and the outer radius of the model.
    method2_bands : tuple of dict
        The band record of a Method II child: the anchor of the top band,
        the width of the floor band, whether the floor term is exact, and
        the negative values counted in each band. Empty for any other
        result.
    critical_band : tuple of dict
        The closed-form band of a Method II child on the critical branch,
        or its refusal; empty where there is none.
    continuation : tuple of str
        For each component, the auxiliary density the inversion integrated
        beyond the anisotropy's domain edge (a Type II model beyond R_a: the
        DF above P(R_a) is non-local in it), ``"none"`` where the domain has
        no edge. Empty where not recorded.
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


@dataclass(frozen=True)
class InversionRows:
    """The inversion's rows as made, on its whole grid, before any row is
    replaced: what the truncation reads.

    Parameters
    ----------
    k : array, shape (n_components, n)
        The inversion's energies.
    f, ddf : array, shape (n_components, n)
        The DF and its derivative dF/dK there, signed as made.
    data_top : tuple of float
        The potential at the innermost radius of the settings, which decides
        whether the rows reach the ceiling.
    """

    k: Array
    f: Array
    ddf: Array
    data_top: tuple[float, ...]


def potential_density_pair(
    components: Sequence[Component],
    parent: TruncatedProfile | None = None,
) -> PotentialDensityPair:
    """Return the total potential and the component densities.

    Parameters
    ----------
    components : sequence of Component
        The components of the model.
    parent : TruncatedProfile, optional
        The truncated profile to invert, for Method II. If omitted, the pair
        is the analytic one of the untruncated components.

    Returns
    -------
    pair : PotentialDensityPair
    """
    raise NotImplementedError


def energy_grid(pair: PotentialDensityPair, settings: Settings) -> EnergyGrid:
    """Return the energies of the DF table and their radii.

    The grid runs from a floor above zero to a ceiling below the central
    potential, with the spacing the settings ask for towards both ends.

    Parameters
    ----------
    pair : PotentialDensityPair
        The potential that maps energies to radii.
    settings : Settings
        Numerical settings.

    Returns
    -------
    grid : EnergyGrid
    """
    raise NotImplementedError


def aux_derivatives(
    pair: PotentialDensityPair,
    grid: EnergyGrid,
    form: InversionForm,
    settings: Settings,
) -> AuxDerivatives:
    """Return the derivatives of each auxiliary density on the grid.

    For an analytic pair the derivatives are exact. For a tabulated pair they
    follow from a spline of the logarithm of the auxiliary density.

    Parameters
    ----------
    pair : PotentialDensityPair
        The potential and the densities.
    grid : EnergyGrid
        The energies at which the derivatives are needed.
    form : InversionForm
        The form, which fixes the orders needed.
    settings : Settings
        Numerical settings.

    Returns
    -------
    derivatives : AuxDerivatives

    References
    ----------
    Notes Section II.2 [num-pair], Notes Eq. (II.1) [dp-operator].
    """
    raise NotImplementedError


def inversion_form(pair: PotentialDensityPair, grid: EnergyGrid) -> InversionForm:
    """Return the form of the inversion on every row.

    The choice is made from the model parameters alone: the order pair, the
    local form for half-integer central anisotropy, the interior point of a
    Type II model, and the behavior of the auxiliary density at the outer
    end. No computed number is inspected.

    Parameters
    ----------
    pair : PotentialDensityPair
        The potential and the densities.
    grid : EnergyGrid
        The energies of the table.

    Returns
    -------
    form : InversionForm

    References
    ----------
    Notes Section II.3 [num-forms], Notes Eq. (II.2) [unified-pair].
    """
    raise NotImplementedError


def invert_rows(
    grid: EnergyGrid,
    derivatives: AuxDerivatives,
    form: InversionForm,
    settings: Settings,
) -> DfTable:
    """Return the DF of each component on the energy grid.

    Parameters
    ----------
    grid : EnergyGrid
        The energies of the table.
    derivatives : AuxDerivatives
        The derivatives of the auxiliary densities.
    form : InversionForm
        The form of the inversion on every row.
    settings : Settings
        Numerical settings.

    Returns
    -------
    table : DfTable

    References
    ----------
    Notes Eq. (II.3) [floor], Notes Eq. (II.4) [split],
    Notes Eq. (II.5) [composite-map].
    """
    raise NotImplementedError


def asymptotic_form(
    pair: PotentialDensityPair, table: DfTable, settings: Settings
) -> AsymptoticForm:
    """Return the analytic form of each DF beyond the two ends of its table.

    At the center the form follows from the classification of the model; at
    the outer end from the tail of a (C)OM Type I model. The table itself is
    not changed: outside its support it holds no value, and these forms say
    how each DF continues.

    Parameters
    ----------
    pair : PotentialDensityPair
        The potential and the densities, which fix the forms.
    table : DfTable
        The table returned by `invert_rows`, whose last and first rows fix
        the parts of the amplitudes that are not closed form.
    settings : Settings
        Numerical settings.

    Returns
    -------
    form : AsymptoticForm

    References
    ----------
    Notes Section I.4 [asymptotics], Notes Eq. (I.16) [tail-exponent].
    """
    raise NotImplementedError


def validate_df(table: DfTable, form: AsymptoticForm) -> tuple[float, ValidationReport]:
    """Return the lowest admissible truncation energy and the DF's report.

    Parameters
    ----------
    table : DfTable
        The DF table.
    form : AsymptoticForm
        The forms beyond its ends.

    Returns
    -------
    k_t_min : float
        The larger of the highest zero crossing and the last minimum of the
        DFs; zero where every DF is non-negative and non-decreasing.
    report : ValidationReport
        The essential checks of the inversion.
    """
    raise NotImplementedError


# The stage results as pytrees: the arrays and numbers are the data, the
# labels and the bounds static (R-47 item 9). An InversionRecord is a record,
# all static.
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
