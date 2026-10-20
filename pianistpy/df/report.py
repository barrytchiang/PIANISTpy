"""The DF's report, computed on the inversion's own table and its forms.

Each check is measured on the table `invert_rows` returns, the rows as the
inversion made them, and on the forms `asymptotic_form` states beyond its two
ends; nothing is refused here: every value is judged on the tiers of the
report (silent within the goal, a warning above it, refused beyond the hard
limit). The checks of the DF are the DF's alone: the Poisson solve's offset
eps lives on the model's report, which refers to this one.

The gates of a truncation energy are read from the same table: the lowest
admissible truncation energy and the untruncated DF's sign test.
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
from .central import top_limit
from .checks import TableChecks, measure_table, validity_certificate
from .derivatives import ExactDerivs, host_reachable_radius
from .form import tail_exponent_analytic
from .gates import TruncationGates, truncation_gates
from .profile_checks import density_from_df
from .stages import AsymptoticForm, DfTable, PotentialDensityPair
from .truncation import CEILING_GAP

# The checks of a DF's report, one key set per kind of DF.
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
# why the slope column of a table that states none is not judged: a DF from a
# function (F only), and a Method II child (its table has no slope column)
NO_SLOPE_COLUMN = (
    "no derivative column: the lookup's slopes are the spline's, not judged"
)
NO_SLOPE_COLUMN_CHILD = (
    "no derivative column: the inversion of a tabulated pair makes no dF/dK, "
    "the lookup's slopes are the spline's, not judged"
)
# why an untruncated Type II DF's density round trip is not judged (the DF is
# returned over its support K > P(R_a) only)
NO_ROUND_TRIP_TYPE_II = (
    "the support ends at P(R_a); the density inside R_a reads the "
    "continuation below it, which is not part of the product; the roundtrip is "
    "judged on its truncations at K_t > P(R_a)"
)
# what an untruncated Type II DF's slope column leaves unjudged: the
# polynomial stencils cannot read the edge law at the support's first node
NO_SLOPE_AT_TYPE_II_EDGE = (
    "the support's first node, at P(R_a), is not judged: F goes as "
    "(K - P(R_a))^(-1/2) there, which the stencils' polynomials cannot "
    "read; the slope column is judged on the other nodes"
)
# the center's forms whose exponent is the singular one's; the others follow
# u^0 at the center
_POWER_FORMS = ("power", "fitted")


def _kappa(form: AsymptoticForm, i: int) -> float:
    """The exponent of u = p_0 - k the DF follows at the center."""
    if form.central_form[i] in _POWER_FORMS:
        return float(form.central_exponent[i])
    if form.central_form[i] in ("analytic", "log", "constant_plus_power"):
        return 0.0
    return _NAN


def _top(form: AsymptoticForm, i: int) -> dict[str, Any]:
    """The stated center under the names `measure_table` reads."""
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
    """|kappa_loc - kappa_form| at the table's top where the center's form
    took over below the ceiling (the table ends at an anchor under :math:`P_0` less
    the ceiling gap); zero where the rows reach the ceiling, or the form is
    not a power of u. kappa_form is the form's own d ln F / d ln u at the
    anchor, with its regular part where it states one, else kappa.

    kappa_loc = d ln F / d ln u at the top: -u F'/F from the exact slope, or
    the least-squares slope over the 7 nodes nearest the top.
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
        # the form as used at the anchor, all its terms: d ln F / d ln u of
        # A_a (u / u_a)^kappa + B + D u at u_a
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
    """The tail, the Taylor form and the edge law below the floor."""
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
    """Return the forms a DF's table is measured against, for component i.

    Returns
    -------
    top : dict
        The stated center at the table's top, under the names
        `measure_table` reads.
    tail, taylor, law
        The forms below the floor: the tail's parameters, the Taylor
        coefficients and the edge law's, each None where the form states
        another rule.
    """
    return (_top(form, i), *_forms_below(form, i))


def validate_df(
    pair: PotentialDensityPair, table: DfTable, form: AsymptoticForm
) -> tuple[float, ValidationReport, TruncationGates]:
    r"""Return the lowest admissible truncation energy, the DF's report and
    the gates of the untruncated DF.

    Parameters
    ----------
    pair : PotentialDensityPair
        The model, which fixes the validity check and the analytic
        exponent of the DF's tail.
    table : DfTable
        The inversion's own table: the rows as made, ending where the
        center's form takes over.
    form : AsymptoticForm
        The forms beyond the table's two ends.

    Returns
    -------
    k_t_min : float
        The largest of the highest zero crossing and the last minimum of the
        DF on its support, and the potential at the anisotropy's domain edge
        (:math:`P(R_{\rm a})` for a Type II model); zero where the DF is
        non-negative and non-decreasing and the domain has no edge. A strict
        lower bound: a truncation energy must lie above it, which is
        necessary, not sufficient; a Type II model's admitted truncations are
        those whose solve passes the edge gate (``R_t < R_a``), which only the
        solve decides.
    report : ValidationReport
        The DF's checks, each judged on the tiers. The density round trip
        of an untruncated Type II DF is not judged (`NO_ROUND_TRIP_TYPE_II`)
        and not computed; its slope column is not judged at the support's
        first node (`NO_SLOPE_AT_TYPE_II_EDGE`, in the check's ``excluded``).
    gates : TruncationGates
        The sign of the DF and its gates at :math:`\mathcal{K}_{\rm t} = 0`;
        the gates of a truncation energy above zero are its comparison with
        `k_0` and `k_t_min`.

    Raises
    ------
    InputError
        If the pair has more than one component (this release validates one
        DF).
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
    # the anisotropy's domain edge: P(R_a) for a Type II component of an
    # analytic pair, below which no truncation energy is admissible
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
    # a Type II DF's support starts at P(R_a), where F goes as
    # (K - P(R_a))^(-1/2): the slope column is not judged at that node
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
        # zero where there are no nodes below: nothing is joined there, and the
        # forms continue the table (the tail's join is its own check)
        "below_floor_join": Check(
            abs(measured.below_join) if below is not None else 0.0, float(k[0])
        ),
    }
    if not pair.is_analytic:
        return (
            gates.k_t_min,
            # the parent model's tier, as its truncation recorded it on the
            # profile (level 0 where it did not)
            ValidationReport(
                DF_REPORT_KEYS["method_ii_df"],
                checks,
                int(getattr(component.profile, "model_level", None) or 0),
            ),
            gates,
        )
    if not isinstance(profile, DoublePowerLaw):
        raise InputError("An analytic pair's component has a double power law.")
    # the validity check at the rows' top: the ceiling gap, or where that lies above
    # it the shallower of the depth where the exact derivatives leave float64 and
    # the depth at the host floor (the cut invert._below_limit makes)
    u_floor = float(
        profile.depth_inner(np.array([host_reachable_radius()]), terms=8)[0][0]
    )
    u_top = max(CEILING_GAP, top_limit(profile, component.anisotropy), u_floor)
    cert = validity_certificate(float(profile.rho_gamma), _kappa(form, i), u_top)
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
    """The density the DF gives (its table and its forms, through the lookup)
    against the profile's, at the radii of the table's own rows, r(K) from
    the analytic potential, each read with its exact depth where the
    profile's inner series holds it; the largest relative difference and its
    radius."""
    r = np.asarray(ExactDerivs(profile, max_order=1).r_of_p(k), dtype=np.float64)
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
    """Return the report with its ``slope_column`` stated not judged, for a
    table that states no dF/dK column (`DfTable.slopes_exact` false): the
    check then measures nothing, and says why. A signed DF's report, which
    measures neither of its checks, is returned as it is."""
    if table.slopes_exact[0] or table.sign.size or "slope_column" not in report.checks:
        return report
    checks = {**report.checks, "slope_column": Check(math.nan, reason=reason)}
    return dataclasses.replace(report, checks=checks)
