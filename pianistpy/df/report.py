"""Validation report of a DF, measured on the inversion's own table.

`validate_df` measures each check of a DF on the table that the inversion
returns, with the rows as the inversion made them, and on the closed forms
that the asymptotic form states beyond the table's two ends. The module
raises no AccuracyError: `ValidationReport` places each check's value in
its tier, "silent" at or below :math:`10^{-6}`, "recorded" up to
:math:`10^{-4}`, "warned" up to :math:`10^{-2}` and "refused" above
:math:`10^{-2}`. The report holds the DF's checks only; the offset of the
Poisson solve belongs to the model's report.

`validate_df` also returns the gates of a truncation energy, read from the
inversion's table: the lowest admissible truncation energy and the sign
test of the untruncated DF.
"""

__all__ = [
    "DF_REPORT_KEYS",
    "NO_ROUND_TRIP_TYPE_II",
    "NO_SLOPE_AT_TYPE_II_EDGE",
    "NO_SLOPE_COLUMN",
    "NO_SLOPE_COLUMN_CHILD",
    "slope_column_not_judged",
    "stated_forms",
    "validate_df",
]

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..profiles import DoublePowerLaw
from ..validation import Check, ValidationReport
from .central import center_form
from .checks import TableChecks, measure_table, validity_certificate
from .derivatives import _r_of_p
from .form import tail_exponent_analytic
from .gates import TruncationGates, truncation_gates
from .profile_checks import density_from_df
from .stages import AsymptoticForm, DfTable, PotentialDensityPair
from .truncation import CEILING_GAP

# The keys of the checks in a DF's report, one tuple per kind of DF.
DF_REPORT_KEYS = {
    "untruncated_df": (
        "top_form",
        "slope_column",
        "below_floor_join",
        "validity",
        "high_k_slope_mismatch",
        "round_trip",
    ),
    "untruncated_df_with_tail": (
        "top_form",
        "slope_column",
        "below_floor_join",
        "tail_join",
        "validity",
        "high_k_slope_mismatch",
        "round_trip",
    ),
    "flagged_df": ("top_form", "slope_column"),
    "method_ii_df": ("top_form", "slope_column", "below_floor_join"),
}
_NAN = float("nan")
# The reasons a report gives for not judging the slope column of a table
# without a dF/dK column: a DF from a function, which gives F only, and a
# Method II child, whose inversion makes no dF/dK.
NO_SLOPE_COLUMN = (
    "no derivative column: the lookup's slopes are the spline's, not judged"
)
NO_SLOPE_COLUMN_CHILD = (
    "no derivative column: the inversion of a tabulated pair makes no dF/dK, "
    "the lookup's slopes are the spline's, not judged"
)
# The reason a report gives for not judging the density round trip of an
# untruncated Type II DF: the DF is returned over its support K > P(R_a) only.
NO_ROUND_TRIP_TYPE_II = (
    "the support ends at P(R_a); the density inside R_a reads the "
    "continuation below it, which is not part of the product; the roundtrip is "
    "judged on its truncations at K_t > P(R_a)"
)
# The reason a report gives for not judging the slope column of an untruncated
# Type II DF at the support's first node: the polynomial stencils of the slope
# check cannot follow the power (K - P(R_a))^(-1/2) at that node.
NO_SLOPE_AT_TYPE_II_EDGE = (
    "the support's first node, at P(R_a), is not judged: F goes as "
    "(K - P(R_a))^(-1/2) there, which the stencils' polynomials cannot "
    "read; the slope column is judged on the other nodes"
)
# The center's forms whose DF goes as u^kappa at the center, with kappa the
# form's central exponent; `_kappa` gives the other forms the exponent 0 or NaN.
_POWER_FORMS = ("power", "fitted")


def _kappa(form: AsymptoticForm, i: int) -> float:
    """Return the exponent kappa of u = p_0 - k that the DF follows at the center.

    The exponent is the form's central exponent for a "power" or "fitted"
    form, 0 for an "analytic", "log" or "constant_plus_power" form, and NaN
    for any other form.
    """
    if form.central_form[i] in _POWER_FORMS:
        return float(form.central_exponent[i])
    if form.central_form[i] in ("analytic", "log", "constant_plus_power"):
        return 0.0
    return _NAN


def _top(form: AsymptoticForm, i: int) -> dict[str, Any]:
    """Return component i's stated center under the key names `measure_table` reads."""
    top: dict[str, Any] = {"top_kappa": _kappa(form, i)}
    if form.central_form[i] != "fitted":
        top.update(
            top_form=form.central_form[i],
            top_singular_exponent=float(form.central_exponent[i]),
            top_amplitude=float(form.central_amplitude[i]),
            top_log_coefficient=float(form.central_log_coefficient[i]),
        )
    if form.central_next_form[i] != "none":
        top.update(
            top_next_form=form.central_next_form[i],
            top_next_exponent=float(form.central_next_exponent[i]),
            top_next_coefficient=float(form.central_next_coefficient[i]),
        )
    return top


def _slope_mismatch(
    k: np.ndarray,
    log_df: np.ndarray,
    slope: np.ndarray | None,
    p_0: float,
    form: AsymptoticForm,
    i: int,
) -> Check:
    r"""Return the check |kappa_loc - kappa_form| at the table's top energy.

    The check applies where the table's last energy lies below the ceiling
    p_0 - CEILING_GAP, so that the center's closed form covers the energies
    from the table's last energy to the ceiling. The check is zero where the
    table's last energy is at least
    :math:`(p_0 - {\rm CEILING\_GAP})(1 - 10^{-15})`, and zero where the
    center's form is neither "power" nor "fitted" (a power of
    u = p_0 - k).

    kappa_loc is d ln F / d ln u at the table's top: -u F'/F from the exact
    slope column, or, without one, the least-squares slope of ln F against
    ln u over the 7 nodes with the smallest u. kappa_form is the form's own
    d ln F / d ln u at the anchor, including the regular part where the
    form states an anchor, and the central exponent kappa otherwise. The
    check is infinite where fewer than 5 nodes or no spread in ln u are
    available for the fit, or where kappa_loc or kappa_form is not finite.
    """
    reaches_ceiling = float(k[-1]) >= (p_0 - CEILING_GAP) * (1.0 - 1e-15)
    if reaches_ceiling or form.central_form[i] not in _POWER_FORMS:
        return Check(0.0, float(k[-1]))
    kappa = float(form.central_exponent[i])
    u = p_0 - k
    if slope is not None:
        kappa_loc = float(-u[-1] * slope[-1] * np.log(10.0) / k[-1])
    else:
        idx = np.argsort(u)[:7]
        lu = np.log(u[idx])
        if idx.size < 5 or float(np.ptp(lu)) <= 0.0:
            return Check(np.inf, float(k[-1]))
        kappa_loc = float(np.polyfit(lu, log_df[idx] * np.log(10.0), 1)[0])
    if form.central_anchor and np.isfinite(float(form.central_anchor[i])):
        # kappa_form is d ln F / d ln u at u_a of the whole form used above the
        # anchor, A_a (u / u_a)^kappa + B + D u.
        a_a = float(form.central_anchor_amplitude[i])
        b, d = form.central_regular[i]
        u_a = p_0 - float(form.central_anchor[i])
        kappa = (kappa * a_a + d * u_a) / (a_a + b + d * u_a)
    if not (np.isfinite(kappa_loc) and np.isfinite(kappa)):
        return Check(np.inf, float(k[-1]))
    return Check(abs(kappa_loc - kappa), float(k[-1]))


def _forms_below(
    form: AsymptoticForm, i: int
) -> tuple[Mapping[str, float] | None, Any, Any]:
    """Return the tail, the Taylor form and the edge law below the floor.

    Each of the tail, the Taylor form and the edge law is None unless the
    form states the rule of that name ("tail", "taylor" or "edge_law") below
    the floor of component i.
    """
    rule, coefficients = form.below_floor[i], form.below_floor_coefficients[i]
    tail = None
    if rule == "tail":
        tail = {
            "A": float(form.tail_amplitude[i]),
            "a": float(form.tail_exponent[i]),
            "k": float(form.tail_log_exponent[i]),
            "E0": float(form.tail_log_scale[i]),
        }
    taylor = tuple(coefficients) if rule == "taylor" else None
    law = tuple(coefficients) if rule == "edge_law" else None
    return tail, taylor, law


def stated_forms(
    form: AsymptoticForm, i: int
) -> tuple[dict[str, Any], Mapping[str, float] | None, Any, Any]:
    """Return the forms that component i's DF table is measured against.

    Parameters
    ----------
    form : AsymptoticForm
        The forms beyond the table's two ends.
    i : int
        Index of the component.

    Returns
    -------
    top : dict
        The stated center at the table's top, under the key names that
        `measure_table` reads.
    tail : mapping or None
        The tail's parameters ``A``, ``a``, ``k`` and ``E0``; None where the
        form states another rule below the floor.
    taylor : tuple or None
        The Taylor coefficients; None where the form states another rule
        below the floor.
    law : tuple or None
        The edge law's coefficients; None where the form states another
        rule below the floor.
    """
    return (_top(form, i), *_forms_below(form, i))


def validate_df(
    pair: PotentialDensityPair, table: DfTable, form: AsymptoticForm
) -> tuple[float, ValidationReport, TruncationGates]:
    r"""Return the lowest admissible truncation energy, the DF's report and the gates.

    Parameters
    ----------
    pair : PotentialDensityPair
        The potential-density pair of the model. The pair's component fixes
        the validity check and the analytic exponent of the DF's tail.
    table : DfTable
        The inversion's own table: the rows as made, ending where the
        center's closed form takes over.
    form : AsymptoticForm
        The forms beyond the table's two ends.

    Returns
    -------
    k_t_min : float
        The largest of three energies: the highest table energy at which
        the DF is negative (`TruncationGates.k_0`), the energy at which the
        DF's last decrease ends (`TruncationGates.q_min`), and the potential
        :math:`P(R_{\rm a})` at the anisotropy's domain edge for a Type II
        model. Where :math:`P(R_{\rm a})` exceeds the other two energies,
        `k_t_min` is the next double above :math:`P(R_{\rm a})`. Zero where
        the DF has no negative value and no decrease and the anisotropy has
        no domain edge. A truncation energy at or above `k_t_min` is
        necessary for admissibility, not sufficient. A Type II model's
        truncation is admitted only where the truncated solve finds
        :math:`R_{\rm t} < R_{\rm a}`.
    report : ValidationReport
        The DF's checks, each placed in its tier. A DF with a negative
        value gets the "flagged_df" report, whose two checks, ``top_form``
        and ``slope_column``, are NaN. The
        density round trip of an untruncated Type II DF is not computed and
        not judged (`NO_ROUND_TRIP_TYPE_II`). The slope column of an
        untruncated Type II DF is not judged at the support's first node
        (`NO_SLOPE_AT_TYPE_II_EDGE`, in the check's ``excluded``).
    gates : TruncationGates
        The sign record of the DF and the gates that
        :math:`\mathcal{K}_{\rm t} = 0` fails. A truncation energy above
        zero is gated by its comparison with ``gates.k_0`` and `k_t_min`.

    Raises
    ------
    InputError
        If the pair has more than one component.
    InputError
        If the pair is analytic and the component's profile is not a
        `DoublePowerLaw`.
    """
    if len(pair.components) != 1:
        raise InputError("This release validates one component's DF.")
    i = 0
    component = pair.components[i]
    lo, hi = int(table.i_first[i]), int(table.i_last[i]) + 1
    k = np.asarray(table.k, dtype=np.float64)[i, lo:hi]
    log_df = np.asarray(table.log_df, dtype=np.float64)[i, lo:hi]
    slope = (
        np.asarray(table.log_df_slope, dtype=np.float64)[i, lo:hi]
        if table.slopes_exact[i]
        else None
    )
    sign = np.asarray(table.sign)[i, lo:hi] if np.size(table.sign) else None
    f = 10.0**log_df if sign is None else sign * 10.0**log_df
    profile = component.unit_profile
    a_analytic = (
        tail_exponent_analytic(profile, component.anisotropy)
        if isinstance(profile, DoublePowerLaw)
        else _NAN
    )
    # The energy k_edge is P(R_a), the potential at the anisotropy's domain
    # edge R_a, for a Type II component of an analytic pair with a double
    # power law, and 0 otherwise. No truncation energy at or below P(R_a) is
    # admissible.
    edge = component.anisotropy.domain_edge
    k_edge = (
        float(np.asarray(profile.p(np.array([edge])))[0])
        if isinstance(profile, DoublePowerLaw)
        and pair.is_analytic
        and np.isfinite(edge)
        else 0.0
    )
    gates = truncation_gates(k, f, k, f, 0.0, a_analytic, k_edge)
    if sign is not None and np.any(sign < 0):
        report = ValidationReport(
            DF_REPORT_KEYS["flagged_df"],
            {"top_form": Check(_NAN), "slope_column": Check(_NAN)},
        )
        return gates.k_t_min, report, gates
    n_below = table.nodes_below(i)
    below = None
    if n_below:
        below = (
            np.asarray(table.k_below, dtype=np.float64)[i, :n_below],
            np.asarray(table.log_df_below, dtype=np.float64)[i, :n_below],
            np.asarray(table.log_df_slope_below, dtype=np.float64)[i, :n_below]
            if table.slopes_exact[i]
            else None,
        )
    top, tail, taylor, law = stated_forms(form, i)
    p_0 = float(form.p_0)
    # A Type II DF's support starts at P(R_a), where F goes as
    # (K - P(R_a))^(-1/2), so the slope column is not judged at that node.
    edge_first = k_edge > 0.0 and slope is not None
    measured: TableChecks = measure_table(
        k, log_df, slope, p_0, top, below, tail, taylor, law, edge_first
    )
    checks = {
        "top_form": Check(abs(measured.kappa_residual), measured.k_top),
        "slope_column": Check(
            abs(measured.slope_column_excess),
            excluded=NO_SLOPE_AT_TYPE_II_EDGE if edge_first else None,
        ),
        # The join check is zero where the table has no nodes below the floor,
        # because without such nodes the forms below the floor continue the
        # table directly. The join of the tail is the separate check tail_join.
        "below_floor_join": Check(
            abs(measured.below_join) if below is not None else 0.0, float(k[0])
        ),
    }
    if not pair.is_analytic:
        return (
            gates.k_t_min,
            # A Method II child's report takes as parents_level the level of the
            # parent model's report (0 to 3, silent to refused), which the
            # parent's truncation records on the profile as model_level. The
            # parents_level is 0 where the profile records no model_level.
            ValidationReport(
                DF_REPORT_KEYS["method_ii_df"],
                checks,
                int(getattr(component.profile, "model_level", None) or 0),
            ),
            gates,
        )
    if not isinstance(profile, DoublePowerLaw):
        raise InputError("An analytic pair's component has a double power law.")
    # The validity check is evaluated at the table's top row with the center's
    # closed form A u^kappa (`center_form`), the form by which `ceiling_limit`
    # cuts the rows.
    center = center_form(profile, component.anisotropy)
    power = center["form"] == "power" and float(center["amplitude"]) > 0.0
    cert = validity_certificate(
        float(profile.rho_gamma),
        _kappa(form, i),
        p_0 - float(k[-1]),
        math.log10(float(center["amplitude"])) if power else 0.0,
    )
    passed = (
        cert["valid_gamma"]
        and cert["linearF_representable"]
        and cert["R_representable"]
    )
    checks["validity"] = Check(0.0 if passed else np.inf)
    checks["high_k_slope_mismatch"] = _slope_mismatch(k, log_df, slope, p_0, form, i)
    checks["round_trip"] = (
        Check(_NAN, reason=NO_ROUND_TRIP_TYPE_II)
        if k_edge > 0.0
        else _round_trip(profile, component.anisotropy, table, form, k)
    )
    kind = "untruncated_df"
    if tail is not None:
        k_0 = float(below[0][0]) if below is not None else float(k[0])
        checks["tail_join"] = Check(abs(measured.tail_join), k_0)
        kind = "untruncated_df_with_tail"
    return gates.k_t_min, ValidationReport(DF_REPORT_KEYS[kind], checks), gates


def _round_trip(
    profile: DoublePowerLaw,
    anisotropy: Anisotropy,
    table: DfTable,
    form: AsymptoticForm,
    k: np.ndarray,
) -> Check:
    """Return the density round trip of an untruncated DF at the table's radii.

    The check compares the density that the DF gives, through the lookup of
    its table and forms, with the profile's density at the radii r(K) of
    the table's energies, with r(K) from the analytic potential. The DF is
    read by the exact depth u from the profile's inner series where the
    series' error bound is below :math:`10^{-16}`, and by the energy elsewhere. The
    check's value is the largest relative difference, and its location the
    radius of that difference; a non-finite difference counts as infinite.
    """
    r = np.asarray(_r_of_p(profile, k), dtype=np.float64)
    rho = np.asarray(profile.rho(r), dtype=np.float64)
    u, bound = profile.depth_inner(r)
    depth = np.where(np.asarray(bound) < 1e-16, u, np.nan)
    with np.errstate(all="ignore"):
        got = np.asarray(
            density_from_df(r, k, table, form, anisotropy, depth=depth),
            dtype=np.float64,
        ).ravel()
    e = np.abs(got / rho - 1.0)
    e = np.where(np.isfinite(e), e, np.inf)
    j = int(np.argmax(e))
    return Check(float(e[j]), float(r[j]))


def slope_column_not_judged(
    report: ValidationReport, table: DfTable, reason: str
) -> ValidationReport:
    """Return the report with its slope-column check marked as not judged.

    The call applies to a table without a dF/dK column
    (`DfTable.slopes_exact` false): the ``slope_column`` check is replaced
    by a NaN check that carries ``reason``. The report is returned
    unchanged where the table has a dF/dK column, where the table has a
    sign column (the report of a signed DF already holds NaN for both of its
    checks, ``top_form`` and ``slope_column``), or
    where the report has no ``slope_column`` check.
    """
    if table.slopes_exact[0] or table.sign.size or "slope_column" not in report.checks:
        return report
    checks = {**report.checks, "slope_column": Check(math.nan, reason=reason)}
    return dataclasses.replace(report, checks=checks)
