r"""The form in which each row of the DF table is inverted.

The inversion calls `inversion_form` before any integral is evaluated.
`inversion_form` chooses the form of each row from the model's parameters
and the smallest energy of the component's table; no computed value of the
inversion is read. The choice follows from the order pair (`order_pair`),
the interior point :math:`P_s` of a COM Type II model (`interior_point`)
and the power of the auxiliary density at the outer end
(`outer_end_power`). Each row takes one of three forms: the local closed
form where the central anisotropy :math:`\beta_{\rm c}` is a half-integer,
the form with the surface term (integrated by parts), or the general form
(the derivative outside the integral). The rows at energies above
:math:`P_s` take the general form. For an unbounded outer end, the rows at
energies at or above `OUTER_FROM_FACTOR` times the table's smallest energy
take the general form.

References
----------
Notes Section T:2.3.1, Notes Eq. (T:2.16).
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

# The integer code of each row form in `InversionForm.form`.
FORMS = {"local": 0, "surface": 1, "general": 2}
# For an unbounded outer end, the rows with energies at or above
# OUTER_FROM_FACTOR times the table's first energy leave the surface form for
# the general form.
OUTER_FROM_FACTOR = 10.0


@dataclass(frozen=True)
class InversionFormRecord:
    r"""The inversion form of one component, for the inversion's record.

    Parameters
    ----------
    form : str
        The evaluation form: ``"closed_form_n0"``, ``"integrated_by_parts"``
        or ``"derivative_outside"``. Suffixes, each after a ``+``, identify the
        form of the rows above the interior point and the form of the rows
        of an unbounded outer end.
    order_m : int
        Integer part :math:`m = \lfloor 3/2 - \beta_{\rm c} \rfloor`, with
        :math:`\beta_{\rm c}` the central anisotropy (`OrderPair.order_m`).
    order_n : float
        Remainder :math:`n = 3/2 - \beta_{\rm c} - m` (`OrderPair.order_n`).
    order_j : int
        Derivative order :math:`j` the inversion uses (`OrderPair.order_j`),
        Notes Eq. (T:2.16).
    kernel_exponent : float
        Kernel exponent the inversion uses (`OrderPair.kernel_exponent`).
    k_interior : float
        Interior point :math:`P_s` of a COM Type II model, in internal units;
        NaN where the model has no interior point.
    interior_exponent : float
        Exponent :math:`s` in :math:`|P - P_s|^s`, the behavior of the
        auxiliary density's derivative of order :math:`j` near the interior
        point :math:`P_s`; NaN where the model has no interior point.
    outer_power : float
        Power :math:`s` of :math:`\varrho_{\rm aux} \sim P^s` as the
        potential :math:`P \to 0`, where the outer end is unbounded; NaN
        otherwise.
    outer_end : str
        ``"unbounded"`` where the power :math:`s` of
        :math:`\varrho_{\rm aux} \sim P^s` as :math:`P \to 0` is below the
        integer part :math:`m` and :math:`\beta_{\rm c}` is not a
        half-integer; ``"bounded"`` otherwise.
    outer_from : float
        Energy, in internal units, at and above which the rows of an
        unbounded outer end take the general form: `OUTER_FROM_FACTOR` times
        the component's smallest energy. Zero for a bounded outer end, and
        zero for the form ``"derivative_outside"``, in which every row
        already takes the general form.
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
    r"""Return the power of the auxiliary density at the outer end.

    The power :math:`s` is that of :math:`\varrho_{\rm aux} \sim P^s` as
    :math:`P \to 0`, with :math:`\varrho_{\rm aux}` the auxiliary density,
    :math:`P` the potential, :math:`R` the radius, :math:`\beta` the outer
    slope of the density profile and :math:`\beta_{\rm c}` the central
    anisotropy. For an OM or COM model,
    :math:`\varrho_{\rm aux} \sim R^{2 - \beta}` at
    large radius, because the powers of :math:`\beta_{\rm c}` in the
    anisotropy factor cancel. For a constant or isotropic anisotropy,
    :math:`\varrho_{\rm aux} \sim R^{2 \beta_{\rm c} - \beta}` at large
    radius. With the potential :math:`P \sim 1/R` at large radius,
    :math:`s = \beta - 2` for an OM or COM model and
    :math:`s = \beta - 2 \beta_{\rm c}` otherwise. At :math:`\beta = 3`, the
    potential is :math:`P \sim \ln R / R`, and
    :math:`\varrho_{\rm aux} \sim P^s` carries a logarithmic factor.
    """
    if anisotropy.model in ("om", "com"):
        return float(profile.rho_beta) - 2.0
    return float(profile.rho_beta) - 2.0 * float(anisotropy.beta_c)


def tail_exponent_analytic(profile: DoublePowerLaw, anisotropy: Anisotropy) -> float:
    r"""Return the exponent of the DF's tail at the outer end.

    The exponent :math:`a` is that of :math:`F \sim A K^{-a}` as
    :math:`K \to 0`, with :math:`F` the DF, :math:`K` the energy, :math:`A` a
    constant amplitude, :math:`\beta` the outer slope of the density profile and
    :math:`\beta_{\rm c}` the central anisotropy. The exponent is
    :math:`a = 3/2 - \beta_{\rm c} - s`, with :math:`s` the power of
    :math:`\varrho_{\rm aux} \sim P^s` at the outer end (`outer_end_power`).
    For an OM or COM model, :math:`a = 7/2 - \beta_{\rm c} - \beta`; for a
    constant or isotropic anisotropy, :math:`a = 3/2 + \beta_{\rm c} - \beta`.
    A negative exponent :math:`a` means a DF that vanishes as :math:`K \to 0`.

    References
    ----------
    Notes Eq. (T:2.13).
    """
    if anisotropy.model in ("om", "com"):
        return 3.5 - float(anisotropy.beta_c) - float(profile.rho_beta)
    return 1.5 + float(anisotropy.beta_c) - float(profile.rho_beta)


def _one(
    profile: DoublePowerLaw, anisotropy: Anisotropy, k: np.ndarray
) -> tuple[np.ndarray, InversionFormRecord]:
    """Return the form code of each row and the form record of one component.

    The rows are the component's energies ``k``, in internal units. The
    form codes are the codes of `FORMS`.
    """
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
    r"""Return the inversion form of every row of every component, and the form records.

    Parameters
    ----------
    pair : PotentialDensityPair
        The components, read for each component's profile and anisotropy.
        `inversion_form` treats each component as an untruncated model
        inverted over its full domain.
    grid : EnergyGrid
        The energies of each component's table, in internal units.

    Returns
    -------
    form : InversionForm
        Per component: the derivative order :math:`j`, the kernel exponent,
        the form code of every row (`FORMS`), and the interior point
        :math:`P_s`, None where the component has no interior point.
    records : tuple of InversionFormRecord
        The form record of each component, for the inversion's record.

    Raises
    ------
    UnphysicalModelError
        If, for a COM Type II model, the derivative of order :math:`j` of the
        auxiliary density is not integrable at the interior point
        :math:`P_s` or carries a delta function at :math:`P_s`
        (`interior_point`).

    References
    ----------
    Notes Section T:2.3, Notes Eq. (T:2.16).
    """
    rows, records = [], []
    for i, component in enumerate(pair.components):
        k = np.asarray(grid.k, dtype=np.float64)[i]
        r, record = _one(component.unit_profile, component.anisotropy, k)
        rows.append(r)
        records.append(record)
    form = InversionForm(
        order_j=tuple(rec.order_j for rec in records),
        kernel_exponent=tuple(rec.kernel_exponent for rec in records),
        form=np.stack(rows),
        k_interior=tuple(
            None if np.isnan(rec.k_interior) else rec.k_interior for rec in records
        ),
    )
    return form, tuple(records)
