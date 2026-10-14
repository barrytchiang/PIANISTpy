"""The form of the inversion, chosen from the model alone.

The order pair, the interior point of a Type II model, the behavior of the
auxiliary density at the outer end and, from them, the form every row of the
table is inverted in: the local closed form at half-integer central
anisotropy, the form with the surface term (integrated by parts), or the
general form (the derivative outside the integral). No computed number of the
inversion is inspected; the grid enters only through its first energy, where
the rows of an unbounded outer end change form.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
"""

__all__ = [
    "FORMS",
    "InversionFormRecord",
    "inversion_form",
    "outer_end_power",
    "tail_exponent_analytic",
]

from dataclasses import dataclass

import numpy as np

from ..anisotropy import Anisotropy
from ..profiles import DoublePowerLaw
from .inversion import interior_point, order_pair
from .stages import EnergyGrid, InversionForm, PotentialDensityPair

# the form of a row, `InversionForm.form`
FORMS = {"local": 0, "surface": 1, "general": 2}
# where an unbounded outer end leaves the rows' surface form: this many times
# the first energy of the table
OUTER_FROM_FACTOR = 10.0


@dataclass(frozen=True)
class InversionFormRecord:
    """What the form states of one component, for the inversion's record.

    Parameters
    ----------
    form : str
        The evaluation form: ``"closed_form_n0"``, ``"integrated_by_parts"``
        or ``"derivative_outside"``, with the forms of the rows above the
        interior point or below the floor after a ``+``.
    order_m, order_n : int, float
        The floor pair (m_f, n_f).
    order_j, kernel_exponent : int, float
        The unified pair in use (j, n).
    k_interior, interior_exponent : float
        The interior point P_s and its exponent s; NaN where there is none.
    outer_power : float
        The exponent s of rho_aux ~ P^s at the outer end where the floor
        pair's derivative of rho_aux is unbounded there; NaN otherwise.
    outer_end : str
        ``"unbounded"`` or ``"bounded"``.
    outer_from : float
        The energy from which the rows take the general form for an
        unbounded outer end; zero where none do.
    """

    form: str
    order_m: int
    order_n: float
    order_j: int
    kernel_exponent: float
    k_interior: float
    interior_exponent: float
    outer_power: float
    outer_end: str
    outer_from: float


def outer_end_power(profile: DoublePowerLaw, anisotropy: Anisotropy) -> float:
    """Return the power s of rho_aux ~ P^s at the outer end (P -> 0).

    rho_aux ~ R^(2 - beta) for OM and COM (the beta_c powers cancel), and
    R^(2 beta_c - beta) for a constant (or isotropic) anisotropy, with
    Psi ~ 1/R (at beta = 3, Psi ~ ln R / R and s = 1 with a logarithm).
    """
    if anisotropy.model in ("om", "com"):
        return float(profile.rho_beta) - 2.0
    return float(profile.rho_beta) - 2.0 * float(anisotropy.beta_c)


def tail_exponent_analytic(profile: DoublePowerLaw, anisotropy: Anisotropy) -> float:
    """Return the exponent a of the DF's tail F ~ A K^-a as K -> 0.

    a = 3/2 - beta_c - s with s the power of rho_aux ~ P^s at the outer end
    (`outer_end_power`): 7/2 - beta_c - beta for OM and COM, 3/2 + beta_c -
    beta for a constant (or isotropic) anisotropy. A negative a is a DF that
    vanishes at K -> 0.
    """
    if anisotropy.model in ("om", "com"):
        return 3.5 - float(anisotropy.beta_c) - float(profile.rho_beta)
    return 1.5 + float(anisotropy.beta_c) - float(profile.rho_beta)


def _one(
    profile: DoublePowerLaw, anisotropy: Anisotropy, k: np.ndarray
) -> tuple[np.ndarray, InversionFormRecord]:
    """The rows' forms and the record of one component on its energies ``k``."""
    beta_c = 0.0 if anisotropy.model == "om" else float(anisotropy.beta_c)
    pair = order_pair(beta_c)
    is_half_integer = pair.kernel_exponent == 0.0
    switched = pair.order_j == pair.order_m + 1
    if is_half_integer:
        p_s, s = 0.0, 0.0
    else:
        p_s, s = interior_point(profile, anisotropy, pair.order_j, True)
    s_out = outer_end_power(profile, anisotropy)
    unbounded = bool((not is_half_integer) and s_out < int(pair.order_m))
    if is_half_integer:
        form = "closed_form_n0"
    elif switched:
        form = "derivative_outside" + ("+gauss_jacobi_above_Ps" if p_s > 0.0 else "")
    else:
        form = "integrated_by_parts" + (
            "+derivative_outside_above_Ps" if p_s > 0.0 else ""
        )
    if unbounded:
        form += "+derivative_outside_below_floor"
    outer_from = (
        OUTER_FROM_FACTOR * float(np.min(k)) if unbounded and not switched else 0.0
    )
    if is_half_integer:
        rows = np.full(k.size, FORMS["local"], dtype=np.int8)
    else:
        rows = np.full(
            k.size, FORMS["general"] if switched else FORMS["surface"], dtype=np.int8
        )
        if p_s > 0.0:
            rows[k > p_s] = FORMS["general"]
        if outer_from > 0.0:
            rows[k >= outer_from] = FORMS["general"]
    record = InversionFormRecord(
        form=form,
        order_m=int(pair.order_m),
        order_n=float(pair.order_n),
        order_j=int(pair.order_j),
        kernel_exponent=float(pair.kernel_exponent),
        k_interior=float(p_s) if p_s > 0.0 else float("nan"),
        interior_exponent=float(s) if p_s > 0.0 else float("nan"),
        outer_power=float(s_out) if unbounded else float("nan"),
        outer_end="unbounded" if unbounded else "bounded",
        outer_from=float(outer_from),
    )
    return rows, record


def inversion_form(
    pair: PotentialDensityPair, grid: EnergyGrid
) -> tuple[InversionForm, tuple[InversionFormRecord, ...]]:
    """Return the form of the inversion on every row, and its record.

    Parameters
    ----------
    pair : PotentialDensityPair
        The components (their profile and anisotropy); an analytic pair, the
        inversion of an untruncated model over its full domain.
    grid : EnergyGrid
        The energies of the table of each component.

    Returns
    -------
    form : InversionForm
        The pair in use, the form of every row and the interior point, per
        component.
    records : tuple of InversionFormRecord
        What the form states of each component, for the inversion's record.

    Raises
    ------
    UnphysicalModelError
        If a COM Type II derivative is not integrable at its interior point
        (`interior_point`).

    References
    ----------
    Notes Section II.3 [num-forms], Notes Eq. (II.2) [unified-pair].
    """
    rows, records = [], []
    for i, component in enumerate(pair.components):
        k = np.asarray(grid.k, dtype=np.float64)[i]
        r, record = _one(component.unit_profile, component.anisotropy, k)
        rows.append(r)
        records.append(record)
    form = InversionForm(
        order_m=tuple(rec.order_j for rec in records),
        order_n=tuple(rec.kernel_exponent for rec in records),
        form=np.stack(rows),
        k_interior=tuple(
            None if np.isnan(rec.k_interior) else rec.k_interior for rec in records
        ),
    )
    return form, tuple(records)
