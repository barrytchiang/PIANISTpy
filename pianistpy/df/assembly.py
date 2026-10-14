"""The results built from the stage outputs of a run.

A run of the solver returns, beside each product it writes, a record of its
stage outputs: the final table and its dF/dK column, the forms beyond the
table's ends, the profile rows, the records of the inversion and of the
rows that analytic forms replaced, the values of the checks and the gates
that fired. `df_from_record` and `model_from_record` assemble the package's
results from that record: the rows of the profile that are written, the
finiteness of the model, the radius of each replacement, the checks the
table states about itself and the validity certificate, and the report.

Nothing is computed here that changes a number of the table or the profile:
those come from their stages. The results equal, field by field, the ones
the legacy mapping reads from the product written by the same run.
"""

__all__ = [
    "REPORT_KEYS",
    "df_from_record",
    "edge_row_of",
    "filled_of",
    "center_form_of",
    "gates_of",
    "grid_of",
    "kernel_rows_of",
    "kernel_sources_of",
    "floor_source_of",
    "pair_of",
    "anchor_of",
    "top_patch_of",
    "inversion_form_of",
    "model_from_record",
    "center_of",
    "poisson_solve_of",
    "profile_rows_of",
    "rows_of",
    "validated_of",
    "with_truncation_source",
]

import dataclasses
import json
import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..anisotropy import Anisotropy
from ..exceptions import ResultError, UnphysicalModelError
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..settings import FIELDS, Settings
from ..typing import Array
from ..validation import CEILING, GOAL, Check, ValidationReport, announce
from .center import CenterAmplitude, center_amplitude
from .central import center_form
from .checks import TableChecks, table_checks, validity_certificate
from .core import model_identity
from .derivatives import aux_derivatives
from .edge import density_edge_exponent
from .fill import FilledProfile, ProfileRows, fill_profile, profile_rows
from .form import InversionFormRecord, inversion_form
from .gates import TruncationGates, truncation_gates
from .grid import (
    BAND_HALFWIDTH,
    BAND_REFINE,
    OUTPUT_FLOOR_FACTOR,
    TABLE_FLOOR,
    GridZones,
    inversion_grid,
)
from .inputs import Component, Truncation
from .inversion import order_pair
from .pair import potential_density_pair
from .poisson import PoissonSolve, profile_density, run_poisson_ode, source_slopes
from .profile_checks import PROFILE_CHECK_KEYS, profile_checks
from .report import validate_df
from .results import DistributionFunction, Model
from .rows import (
    FLOOR_P_MIN,
    below_floor,
    complete_rows,
    floor_source,
    inversion_kernel,
    kernel_rules,
)
from .stages import (
    AsymptoticForm,
    AuxDerivatives,
    DfTable,
    EnergyGrid,
    InversionForm,
    InversionRecord,
    PotentialDensityPair,
)
from .top import Anchor, TopPatch, anchor, top_patch
from .truncation import (
    CEILING_GAP,
    INNER_CEILING,
    INNER_DEPTH,
    K_FLOOR,
    LOG_FLOOR,
    ORIGINS,
    truncated_source,
)
from .validate import ValidatedRows, validate_rows

# The checks of a report, one key set per kind and class of result.
REPORT_KEYS = {
    "untruncated_df": (
        "top_form",
        "slope_column",
        "below_floor_join",
        "eps",
        "validity",
        "high_k_slope_mismatch",
    ),
    "untruncated_df_with_tail": (
        "top_form",
        "slope_column",
        "below_floor_join",
        "tail_join",
        "eps",
        "validity",
        "high_k_slope_mismatch",
    ),
    "flagged_df": ("top_form", "slope_column"),
    # a model's report holds the checks of what its own stages compute and
    # refers to its DF's report (the inversion's: validity, the slope
    # mismatch, the table's checks) as its parent's tier, repeating nothing
    "untruncated_model": ("outer_join", "eps", *PROFILE_CHECK_KEYS),
    "truncated_model_method_i": (
        "top_form",
        "slope_column",
        "edge_join",
        *PROFILE_CHECK_KEYS,
    ),
    "model_method_ii": ("top_form", "edge_join", *PROFILE_CHECK_KEYS),
    "method_ii_df": ("top_form", "slope_column", "below_floor_join"),
}

_KINDS = {"OM_II": "II", "COM_II": "II"}
# the keys of the run's inversion record; NaN in inversion_P_s and inversion_s
# means the component has no interior point, in inversion_outer_power that the
# outer end is bounded
_INVERSION_KEYS = (
    "inversion_form",
    "inversion_j",
    "inversion_n",
    "inversion_m_floor",
    "inversion_n_floor",
    "inversion_P_s",
    "inversion_s",
    "inversion_outer_power",
    "inversion_outer_end",
)
_NAN = float("nan")


def _components(record: Mapping[str, Any]) -> tuple[Component, ...]:
    model = record["model"]
    kind = model["anisotropy_type"]
    r_a = math.inf if model.get("R_a") is None else float(model["R_a"])
    anisotropy = Anisotropy(
        beta_c=0.0 if kind in ("iso", "OM_I", "OM_II") else float(model["beta_c"]),
        r_a=r_a if kind not in ("iso", "constant") else math.inf,
        kind=_KINDS.get(kind, "I"),
    )
    profile = DoublePowerLaw(
        rho_alpha=float(model["alpha"]),
        rho_beta=float(model["beta_r"]),
        rho_gamma=float(model["gamma"]),
    )
    component_profile: DoublePowerLaw | TruncatedProfile = profile
    if record.get("method") == 2:
        component_profile = _parent_profile(record, profile, anisotropy.beta_c)
    return (Component(component_profile, anisotropy, name=str(model["model_name"])),)


def _parent_profile(
    record: Mapping[str, Any], source: DoublePowerLaw, beta_c: float
) -> TruncatedProfile:
    """The truncated profile of a Method II child's parent.

    The rows are the parent's, the central coordinate and the edge exponent
    the child's (a child's density at the edge is its parent's); the parent's
    identity from its anisotropy and settings, which the run records.
    """
    parent = record["parent"]
    kind = parent["anisotropy_type"]
    model = parent["model"]
    r_a = float(model.get("anisotropy", {}).get("R_a", math.inf))
    anisotropy = Anisotropy(
        beta_c=0.0 if kind in ("iso", "OM_I", "OM_II") else float(model["beta_c"]),
        r_a=r_a if kind not in ("iso", "constant") else math.inf,
        kind=_KINDS.get(kind, "I"),
    )
    settings = _settings_of(json.loads(parent["settings_json"]))
    components = (Component(source, anisotropy, name=str(parent["model_name"])),)
    k_t = float(record["model"]["Zt"])
    rows = _profile_rows(record)
    finite, r_t = _finite(record)
    return TruncatedProfile(
        r=rows["R"],
        p_trunc=rows["Psi"],
        rho_trunc=rows["rho"][None, :],
        mass_trunc=rows["M"] / (4.0 * np.pi),
        k_t=k_t,
        r_t=r_t,
        mass_t=float(rows["M"][-1]) / (4.0 * np.pi) if finite else math.inf,
        spatially_finite=finite,
        source=source,
        source_anisotropy=anisotropy,
        model=model_identity(components, Truncation(k_t=k_t), settings),
        edge_exponent=_edge_exponent(record, beta_c),
        u_trunc=rows.get("u", np.zeros(0)),
    )


def _edge_exponent(record: Mapping[str, Any], beta_c: float) -> tuple[float, ...]:
    """The exponent q of the density at the edge, rho ~ P^q; empty with no edge.

    A truncated table's Taylor form: F ~ K^p with p = 1 where k_t lies strictly
    above the lowest admissible truncation energy; at equality p = 2 where that
    energy is the last minimum of F (a stationary edge) and p = 1 where it is the
    last zero crossing, and refused where the run does not say which. A Method II
    child: its edge law's own p.
    """
    if not _finite(record)[0]:
        return ()
    rule, coefficients = _below_floor_rule(record)
    if rule == "edge_law":
        p = coefficients[0]
    elif rule == "taylor":
        accuracy = record["accuracy"]
        k_t = float(record["model"]["Zt"])
        gate = float(accuracy.get("Z_t_min_allowed", _NAN))
        set_by = str(accuracy.get("Z_t_min_set_by", ""))
        if k_t > gate:
            p = 1.0
        elif k_t == gate and set_by in ("K_0", "Q_min"):
            p = 2.0 if set_by == "Q_min" else 1.0
        else:
            raise ResultError(
                f"The edge exponent is not stated: k_t = {k_t!r} is not above the "
                f"lowest admissible truncation energy {gate!r}, and the run does not "
                "say which of the last zero crossing and the last minimum sets it."
            )
    else:
        return ()
    return (density_edge_exponent(p, beta_c),)


def _r_trust(record: Mapping[str, Any]) -> dict[str, float]:
    """The radius within which the model holds each tier.

    The run measures the radii on an untruncated model only, from the offset
    eps of its potential: empty on a truncated model and where eps is
    undefined. A radius the run gives as NaN where eps is defined is a tier
    the first outer row already misses: 0.
    """
    accuracy = record["accuracy"]
    if float(record["model"]["Zt"]) > 0.0 or not math.isfinite(float(accuracy["eps"])):
        return {}
    radii = {}
    for tier, key in (("goal", "R_trust_1em6"), ("ceiling", "R_trust_1em4")):
        value = float(accuracy[key])
        radii[tier] = value if math.isfinite(value) else 0.0
    return radii


def _settings(record: Mapping[str, Any]) -> Settings:
    return _settings_of(record["settings"])


def _settings_of(run: Mapping[str, Any]) -> Settings:
    """The package settings of the fields the run shares with them."""
    shared: dict[str, dict[str, Any]] = {
        "inversion": {
            pkg: kind(run[name])
            for pkg, (name, kind) in _INVERSION_NAMES.items()
            if name in run
        },
        "truncation": {
            "ode_rtol": float(run["ODE_RTOL"]),
            "ode_atol": float(run["ODE_ATOL"]),
        },
    }
    overrides = {
        group: {
            name: value
            for name, value in fields.items()
            if value != FIELDS[group][name].default
        }
        for group, fields in shared.items()
    }
    kwargs: dict[str, Any] = {
        group: fields for group, fields in overrides.items() if fields
    }
    return Settings(**kwargs)


# the run's names of the inversion's settings
_INVERSION_NAMES = {
    "n_nodes": ("ABEL_N", int),
    "n_grid_rows": ("GRID_N_B", int),
    "n_grid_tail_rows": ("GRID_N_A_C", int),
    "grid_tail_extent": ("PSI_EXTENT", float),
    "table_nodes_per_decade": ("G_TABLE_NODES_PER_DECADE", int),
    "n_floor_nodes": ("N9_NODES", int),
}

# the run's names of the truncation's settings
_TRUNCATION_NAMES = {
    "n_source_nodes": ("ABEL_N", int),
    "n_source_rows": ("POISSON_N", int),
    "n_source_tail_rows": ("GRID_N_A_C", int),
    "source_tail_extent": ("PSI_EXTENT", float),
    "source_tail_floor": ("Q_FAST_TAIL_FLOOR", float),
    "quadrature_split": ("GL_SPLIT_FRAC", float),
    "quadrature_stretch": ("K_STRETCH", float),
    "ode_rtol": ("ODE_RTOL", float),
    "ode_atol": ("ODE_ATOL", float),
    "n_rows_inner": ("N_ODE_INNER", int),
    "n_rows_outer": ("N_ODE_OUTER", int),
    "log10_r_min": ("R_MIN_EXP", float),
    "continuation_start": ("EDGE_CONT_PSI_FRAC", float),
    "continuation_rows_per_decade": ("EDGE_CONT_PER_DECADE", int),
    "continuation_floor_fraction": ("EDGE_CONT_FLOOR_FRAC", float),
    "profile_bin_decades": ("PROFILE_BIN_DLOG10", float),
    "profile_rows_per_decade": ("PROFILE_MIN_PER_DEC", int),
}

# The run's values the method has fixed: its own numbers, and the switches whose
# other branch is retired. A run with another value is not one the package makes.
_METHOD = {
    "PSI_ANCHOR_LOW": K_FLOOR,
    "Q_CEIL_GAP": CEILING_GAP,
    "G_TABLE_FLOOR": TABLE_FLOOR,
    "DF_OUTPUT_FLOOR_FACTOR": OUTPUT_FLOOR_FACTOR,
    "T2S_BAND_REFINE": BAND_REFINE,
    "T2S_BAND_HALFWIDTH": BAND_HALFWIDTH,
    "N9_P_MIN": FLOOR_P_MIN,
    "FLOAT32_EPSILON": INNER_DEPTH,
    "FLOAT64_UNDERFLOW_GUARD": LOG_FLOOR,
    "ODE_INNER_PSI_CEIL": INNER_CEILING,
    "ACCURACY_GOAL": CEILING,
    "P_RHS_TAIL_MAP": True,
    "P_RHS_LOG_MAP": True,
    "P_RHS_TAIL_TERM": True,
    "P_RHS_TAIL_SPLIT": "interval",
    "INTERP_ORDER": "cubic",
    "D10_TAYLOR_EDGE": True,
    "D11_ANALYTIC_EDGE": True,
    "RESAMPLE_EXACT_SLOPES": True,
    "EDGE_CONTINUATION": True,
    "GPU_DTYPE": "float64",
    "C_TRUNC_CUSP_LIMIT": True,
    "MASS_CUMULATIVE_SIMPSON": True,
    "APPLY_ODE_LOW_PSI_FILL": True,
    "APPLY_ODE_HIGH_PSI_FILL": True,
}


def _check_method(run: Mapping[str, Any]) -> None:
    """Raise ResultError where the run's value of one of the method's numbers
    or retired switches is not the method's."""
    for name, value in _METHOD.items():
        if name in run and run[name] != value:
            raise ResultError(
                f"The run's {name} = {run[name]!r} is not the method's {value!r}; "
                "the package makes no such run."
            )


def _inversion_settings(record: Mapping[str, Any]) -> Settings:
    """The inversion's settings of a run, by the package's names.

    Raises
    ------
    ResultError
        Where the run's value of one of the method's numbers or retired
        switches is not the method's.
    """
    _check_method(record["settings"])
    return _settings(record)


def _truncation_settings(
    record: Mapping[str, Any], log10_r_max: float | None = None
) -> Mapping[str, Any]:
    """The truncation's settings of a run, by the package's names.

    Raises
    ------
    ResultError
        Where the run's value of one of the method's numbers or retired
        switches is not the method's.
    """
    run = record["settings"]
    _check_method(run)
    fields = {pkg: kind(run[name]) for pkg, (name, kind) in _TRUNCATION_NAMES.items()}
    if log10_r_max is not None:
        fields["log10_r_max"] = float(log10_r_max)
    default = FIELDS["truncation"]
    overrides = {k: v for k, v in fields.items() if v != default[k].default}
    return (
        Settings(truncation=overrides).truncation
        if overrides
        else Settings().truncation
    )


def _truncation(record: Mapping[str, Any]) -> Truncation | None:
    z_t = float(record["model"]["Zt"])
    if z_t == 0.0 or int(record["settings"]["TRUNC_METHOD"]) == 2:
        return None
    return Truncation(k_t=z_t)


def _log_df(df: Mapping[str, Any]) -> Array:
    """The written logF: the record's own, or log10 F as the writer forms it."""
    if df.get("logF") is not None:
        return np.asarray(df["logF"], dtype=np.float64)
    f = np.asarray(df["F"], dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(f > 0.0, np.log10(np.maximum(f, 1e-320)), -np.inf)


def _below(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    below = record.get("below_floor")
    if below is None:
        return None
    return {
        "K": np.asarray(below["K"], dtype=np.float64),
        "logF": np.log10(np.asarray(below["F"], dtype=np.float64)),
        "dFdK": np.asarray(below["dFdK"], dtype=np.float64),
    }


def _table(record: Mapping[str, Any]) -> DfTable:
    df = record["df"]
    k = np.asarray(df["K"], dtype=np.float64)
    log_df = _log_df(df)
    signed: dict[str, Any] = {}
    if record.get("sign") is not None:
        f = np.asarray(df["F"], dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_df = np.log10(np.abs(f))
        signed = dict(sign=np.sign(f).astype(np.int8)[None, :])
    ddf = df.get("dFdK")
    ddf = np.full(k.shape, np.nan) if ddf is None else np.asarray(ddf, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(log_df))
    below: dict[str, Any] = {}
    nodes = _below(record)
    if nodes is not None:
        keep = nodes["K"] < k[0] * (1.0 - 1e-12)
        below = dict(
            k_below=nodes["K"][keep][None, :],
            n_below=(int(keep.sum()),),
            log_df_below=nodes["logF"][keep][None, :],
            log_df_slope_below=_log_slope(
                nodes["K"][keep], nodes["logF"][keep], nodes["dFdK"][keep]
            )[None, :],
        )
    slope = _log_slope(k, log_df, ddf)
    # the run's column is F' itself: exact on every row of the support where it
    # is finite there and not written 0 at the top (a column before 1b-38)
    support = slope[finite[0] : finite[-1] + 1]
    if below:
        support = np.concatenate([below["log_df_slope_below"][0], support])
    slopes_exact = bool(np.all(np.isfinite(support)) and ddf[finite[-1]] != 0.0)
    return DfTable(
        k=k[None, :],
        log_df=log_df[None, :],
        log_df_slope=slope[None, :],
        i_first=(int(finite[0]),),
        i_last=(int(finite[-1]),),
        slopes_exact=(slopes_exact,),
        **below,
        **signed,
    )


def _log_slope(k: Array, log_df: Array, ddf_dk: Array) -> Array:
    """d log10 F / d ln K from dF/dK; NaN where that is not finite."""
    with np.errstate(all="ignore"):
        slope = k * ddf_dk * 10.0 ** (-log_df) / np.log(10.0)
    return np.where(np.isfinite(ddf_dk) & np.isfinite(slope), slope, np.nan)


def _below_floor_rule(record: Mapping[str, Any]) -> tuple[str, tuple[float, ...]]:
    if record.get("edge") is not None:
        return "taylor", tuple(float(v) for v in record["edge"])
    if record.get("edge_law") is not None:
        return "edge_law", tuple(float(v) for v in record["edge_law"])
    tail = record.get("tail")
    if tail is not None and float(tail["A"]) > 0.0:
        return "tail", ()
    return "zero", ()


def _form(record: Mapping[str, Any], checks: TableChecks | None) -> AsymptoticForm:
    top = record["top"]
    tail: Mapping[str, Any] = record.get("tail") or {}
    rule, coefficients = _below_floor_rule(record)
    has_tail = rule == "tail"
    stated = "top_form" in top
    next_stated = "top_next_form" in top
    return AsymptoticForm(
        p_0=1.0 - float(record["model"]["Zt"]),
        central_label=(str(top.get("top_kappa_branch", "absent")),),
        central_exponent=(
            float(top["top_singular_exponent"])
            if stated
            else (float(checks.kappa) if checks else _NAN),
        ),
        central_amplitude=(float(top["top_amplitude"]) if stated else _NAN,),
        tail_exponent=(float(tail["a"]) if has_tail else 0.0,),
        tail_amplitude=(float(tail["A"]) if has_tail else 0.0,),
        tail_log_exponent=(float(tail.get("k", 0.0)) if has_tail else 0.0,),
        central_form=(str(top["top_form"]) if stated else "fitted",),
        central_log_coefficient=(
            float(top.get("top_log_coefficient", _NAN)) if stated else _NAN,
        ),
        central_next_form=(str(top["top_next_form"]) if next_stated else "none",),
        central_next_exponent=(
            float(top["top_next_exponent"]) if next_stated else _NAN,
        ),
        central_next_coefficient=(
            float(top["top_next_coefficient"]) if next_stated else _NAN,
        ),
        tail_log_scale=(float(tail.get("E0", 1.0)) if has_tail else 1.0,),
        below_floor=(rule,),
        below_floor_coefficients=(coefficients,),
        # the record's table holds the patched rows: no anchor until the DF
        # holds the inversion's own rows (invert_rows)
        central_anchor=(_NAN,),
        central_anchor_amplitude=(_NAN,),
        central_regular=((_NAN, _NAN),),
    )


def _checks(record: Mapping[str, Any]) -> TableChecks | None:
    """The table's checks as the working reader makes them; None on a signed table."""
    if record.get("sign") is not None:
        return None
    df = record["df"]
    return table_checks(
        np.asarray(df["K"], dtype=np.float64),
        _log_df(df),
        None if df.get("dFdK") is None else np.asarray(df["dFdK"], dtype=np.float64),
        1.0 - float(record["model"]["Zt"]),
        record["top"],
        below=_below(record),
        tail=record.get("tail"),
        edge_taylor=record.get("edge"),
        edge_law=record.get("edge_law"),
    )


def _check(error: float, where: float | None = None) -> Check:
    return Check(error=abs(float(error)), where=None if where is None else float(where))


def _common_checks(checks: TableChecks | None) -> dict[str, Check]:
    if checks is None:
        return {"top_form": _check(_NAN), "slope_column": _check(_NAN)}
    return {
        "top_form": _check(checks.kappa_residual, checks.k_top),
        "slope_column": _check(checks.slope_column_excess),
    }


def _validity(record: Mapping[str, Any]) -> Check:
    """The certificate: within the goal where every column is representable."""
    cert = validity_certificate(
        float(record["model"]["gamma"]),
        float(record["top"]["top_kappa"]),
        float(record["settings"]["Q_CEIL_GAP"]),
    )
    passed = (
        cert["valid_gamma"]
        and cert["linearF_representable"]
        and cert["R_representable"]
    )
    return _check(0.0 if passed else math.inf)


def _slope_mismatch(record: Mapping[str, Any]) -> Check:
    """The asymptote against the table at the anchor; zero where no row was replaced."""
    override = record["overrides"]["saved"]
    if not override:
        return _check(0.0)
    return _check(override["slope_mismatch"], override["K_anchor"])


def _report(
    keys: tuple[str, ...], checks: dict[str, Check], parents_level: int = 0
) -> ValidationReport:
    return ValidationReport(keys, {key: checks[key] for key in keys}, parents_level)


def _tail_join(record: Mapping[str, Any]) -> Check:
    """The tail against the lowest node it continues, F_tail / F_node - 1."""
    tail = record["tail"]
    nodes = _below(record)
    if nodes is not None:
        order = np.argsort(nodes["K"])
        k_0, lg_0 = float(nodes["K"][order[0]]), float(nodes["logF"][order[0]])
    else:
        df = record["df"]
        k_0, lg_0 = float(df["K"][0]), float(_log_df(df)[0])
    a, amp = float(tail["a"]), float(tail["A"])
    b, e_0 = float(tail.get("k", 0.0)), float(tail.get("E0", 1.0))
    log_tail = np.log10(amp) - a * np.log10(k_0) - b * np.log10(np.log(e_0 / k_0))
    return _check(10.0 ** (log_tail - lg_0) - 1.0, k_0)


def _df_report(
    record: Mapping[str, Any], checks: TableChecks | None
) -> ValidationReport:
    common = _common_checks(checks)
    if checks is None:
        return _report(REPORT_KEYS["flagged_df"], common)
    below = record.get("below_floor")
    common["below_floor_join"] = (
        _check(below["join_rel_at_written_floor"], below["K0_written"])
        if below is not None
        else _check(_NAN)
    )
    if record.get("method") == 2:
        return _report(REPORT_KEYS["method_ii_df"], common)
    common["eps"] = _check(record["accuracy"]["eps"])
    common["validity"] = _validity(record)
    common["high_k_slope_mismatch"] = _slope_mismatch(record)
    if _below_floor_rule(record)[0] == "tail":
        common["tail_join"] = _tail_join(record)
        return _report(REPORT_KEYS["untruncated_df_with_tail"], common)
    return _report(REPORT_KEYS["untruncated_df"], common)


def _model_report(
    record: Mapping[str, Any],
    checks: TableChecks | None,
    profile_checks_: Mapping[str, Check],
    parents_level: int = 0,
) -> ValidationReport:
    common = {**_common_checks(checks), **profile_checks_}
    rule = _below_floor_rule(record)[0]
    if rule == "taylor":
        diag = record["edge_diag"]
        common["edge_join"] = _check(
            max(
                abs(float(diag["splice_rel_diff"])), abs(float(diag["switch_rel_diff"]))
            )
        )
        return _report(REPORT_KEYS["truncated_model_method_i"], common, parents_level)
    if rule == "edge_law":
        if checks is None:
            raise ResultError("A Method II model's record carries no table checks.")
        common["edge_join"] = _check(checks.edge_rel_at_floor)
        return _report(REPORT_KEYS["model_method_ii"], common)
    outer = record["accuracy"].get("exact_outer") or {}
    joins = [
        abs(float(outer[k]))
        for k in (
            "exact_outer_join_rel_rho",
            "exact_outer_join_rel_psi",
            "exact_outer_join_rel_M",
        )
        if k in outer
    ]
    common["outer_join"] = _check(max(joins) if joins else _NAN)
    common["eps"] = _check(record["accuracy"]["eps"])
    return _report(REPORT_KEYS["untruncated_model"], common, parents_level)


def _profile_rows(record: Mapping[str, Any]) -> dict[str, Array]:
    """The profile rows the product holds: those at or above the floor with rho > 0.

    A Method II child holds its parent's rows as they were written.
    """
    profile = record["profile"]
    validated = validated_of(record) if record.get("method") != 2 else None
    if validated is not None:
        profile = dict(
            R=validated.r,
            Psi=validated.phi,
            rho=validated.rho,
            M=validated.mass,
            u_record=(validated.u_r, validated.u),
            origin=validated.origin,
        )
    rows = {
        name: np.asarray(profile[name], dtype=np.float64)
        for name in ("R", "Psi", "rho", "M")
    }
    if record.get("method") == 2:
        if profile.get("u") is not None and np.size(profile["u"]) == rows["R"].size:
            rows["u"] = np.asarray(profile["u"], dtype=np.float64)
        return rows
    floor = float(record["settings"]["PSI_ANCHOR_LOW"])
    mask = (rows["Psi"] >= floor) & (rows["rho"] > 0)
    if validated is not None and math.isfinite(validated.r_t):
        mask[-1] = True  # the edge row, Phi = 0 and rho = 0 at r_t
    rows = {name: column[mask] for name, column in rows.items()}
    if profile.get("origin") is not None:
        rows["origin"] = np.asarray(profile["origin"])[mask]
    u_record = profile.get("u_record")
    if u_record is not None:
        r_solve, u_solve = u_record
        i = np.clip(np.searchsorted(r_solve, rows["R"]), 0, r_solve.size - 1)
        if np.array_equal(r_solve[i], rows["R"]):
            rows["u"] = u_solve[i]
    if validated is not None and math.isfinite(validated.r_t):
        rows["r_t"] = np.array(validated.r_t)
    elif validated is None:
        if _finite(record)[0]:
            rows = edge_row_of(rows)
    return rows


def edge_row_of(rows: Mapping[str, Array]) -> dict[str, Array]:
    """The rows of a spatially finite product, ending on the edge row.

    A spatially finite profile carries its edge as a row: the potential and
    the density zero at r_t, where its last interval reads the Kepler form
    (`TruncatedProfile`), so that no reader takes the logarithm of a
    potential at its floor. A legacy product holds none: its last row is the
    potential's floor (the solve's floor event, whose radius is the header's
    R_t). That floor row is dropped and the edge row appended where the
    Kepler form through the last row above it reaches zero,
    ``r_e = 1 / (1 / r_k - Psi_k / M_t)``, with the floor row's mass as M_t:
    the potential is continuous with the product's last real row and
    Poisson's equation holds on the last interval. The continuation is
    exact where the density beyond r_k is negligible; the mass it neglects
    is at most the floor row's M less M_k. Rows that end on an edge row are
    returned as they are.

    Parameters
    ----------
    rows : mapping
        ``R``, ``Psi``, ``rho`` and ``M`` (the product's columns), and where
        present ``u`` (the central potential less the potential) and
        ``origin``, one value per row.

    Returns
    -------
    rows : dict
        The same columns, the last row the edge row, and ``r_t``, its radius.

    Raises
    ------
    ResultError
        If the Kepler form through the last row above the floor has no zero
        (its potential at or above M_t / r_k).
    """
    out = {name: np.asarray(column) for name, column in rows.items()}
    if float(out["Psi"][-1]) == 0.0:
        return out
    r_k, psi_k, m_t = (
        float(out["R"][-2]),
        float(out["Psi"][-2]),
        float(out["M"][-1]) / (4.0 * np.pi),
    )
    inverse = 1.0 / r_k - psi_k / m_t
    if not inverse > 0.0:
        raise ResultError(
            f"The Kepler form through the last row above the floor (R = {r_k:.10g}, "
            f"Psi = {psi_k:.6g}) does not reach zero."
        )
    r_e = 1.0 / inverse
    edge = {"R": r_e, "Psi": 0.0, "rho": 0.0, "M": float(out["M"][-1])}
    if "u" in out:
        # the central potential less the potential, the potential zero there
        edge["u"] = float(out["Psi"][0]) + float(out["u"][0])
    if "origin" in out:
        edge["origin"] = ORIGINS["edge"]
    n = out["R"].size
    for name, column in out.items():
        if column.ndim == 1 and column.size == n and name not in edge:
            raise ResultError(f"The edge row has no value for the column {name!r}.")
    result = {
        name: np.append(column[:-1], np.asarray(edge[name], dtype=column.dtype))
        if name in edge
        else column
        for name, column in out.items()
    }
    result["r_t"] = np.array(r_e)
    return result


def _finite(record: Mapping[str, Any]) -> tuple[bool, float]:
    """Whether the model is spatially finite, and its outer radius."""
    if record.get("method") == 2:
        parent = record["finite"]
        return bool(parent["spatially_finite"]), float(parent["R_t"])
    event = record["finite"]
    finite = bool(float(record["model"]["Zt"]) > 0.0 and event["hit"])
    return finite, (float(event["R_event"]) if finite else math.inf)


def _profile(
    record: Mapping[str, Any],
    rows: Mapping[str, Array],
    k_t: float,
    component: Component,
    identity: str,
) -> TruncatedProfile:
    finite, r_t = _finite(record)
    if "r_t" in rows:
        r_t = float(rows["r_t"])
    deficit = record.get("central_deficit")
    outer = (record.get("accuracy") or {}).get("exact_outer") or {}
    return TruncatedProfile(
        r=rows["R"],
        p_trunc=rows["Psi"],
        rho_trunc=rows["rho"][None, :],
        mass_trunc=rows["M"] / (4.0 * np.pi),
        k_t=k_t,
        r_t=r_t if finite else math.inf,
        mass_t=float(rows["M"][-1]) / (4.0 * np.pi) if finite else math.inf,
        spatially_finite=finite,
        source=component.unit_profile,
        source_anisotropy=component.anisotropy,
        model=identity,
        edge_exponent=_edge_exponent(record, component.anisotropy.beta_c),
        u_trunc=rows.get("u", np.zeros(0)),
        central_deficit=()
        if deficit is None
        else ((float(deficit[0]), float(deficit[1])),),
        override_r=float(outer.get("exact_outer_R_join", _NAN)),
        override_r_end=float(outer.get("exact_outer_R_end", _NAN)),
        override_rows=int(outer.get("exact_outer_rows", 0)),
        override_rule=str(outer.get("profile_outer_potential", "none")),
        origin=rows.get("origin", np.zeros(0, dtype=np.int8)),
    )


def _overrides(
    record: Mapping[str, Any], rows: Mapping[str, Array] | None
) -> tuple[tuple[float, float, int, str], tuple[float, float]]:
    """The high-K and the low-K override records, with the radius of each.

    The radius of the high-K one is where the written profile's potential
    equals its energy; zero where nothing was replaced.
    """
    over = record["overrides"]
    if over is None:
        if record.get("method") != 2:
            raise ResultError("The run records no overrides for a Method I table.")
        # a Method II child: its rows are patched by the bands it records instead
        return (_NAN, _NAN, 0, "method2_bands"), (_NAN, _NAN)
    saved = over["saved"]
    p_0 = float(over["P0"])
    if rows is None:
        return (p_0, _NAN, 0, "none"), (float(over["low_k"]), _NAN)
    k_c = float(saved["K_anchor"]) if saved else p_0
    r_c = 0.0
    if saved and k_c < float(np.max(rows["Psi"])):
        order = np.argsort(rows["Psi"])
        r_c = float(np.interp(k_c, rows["Psi"][order], rows["R"][order]))
    high = (
        k_c,
        r_c,
        int(saved["rows"]) if saved else 0,
        str(saved["rule"]) if saved else "none",
    )
    finite, r_t = _finite(record)
    if "r_t" in rows:
        r_t = float(rows["r_t"])
    return high, (float(over["low_k"]), r_t if finite else math.inf)


def _inversion(
    record: Mapping[str, Any], rows: Mapping[str, Array] | None
) -> InversionRecord:
    inv = record["inversion"]
    missing = [key for key in _INVERSION_KEYS if key not in inv]
    if missing:
        raise ResultError(f"The run's inversion record lacks {', '.join(missing)}.")
    high, low = _overrides(record, rows)
    bands = record.get("method2_bands")
    band_keys = (
        "m2_patch_anchor_K",
        "m2_floor_band_K",
        "m2_floor_term_exact",
        "m2_top_band_raw_negatives",
        "m2_floor_band_negatives",
        "m2_interior_negatives",
    )
    method2: tuple[dict[str, Any], ...] = ()
    critical: tuple[dict[str, Any], ...] = ()
    if bands is not None:
        method2 = ({key[3:]: bands[key] for key in band_keys},)
        crit = {
            key[3:]: value
            for key, value in bands.items()
            if key.startswith("m2_critical_band")
        }
        critical = (crit,) if crit else ()
    return InversionRecord(
        form=(str(inv["inversion_form"]),),
        order_j=(int(inv["inversion_j"]),),
        order_n=(float(inv["inversion_n"]),),
        floor_m=(int(inv["inversion_m_floor"]),),
        floor_n=(float(inv["inversion_n_floor"]),),
        k_interior=(float(inv["inversion_P_s"]),),
        interior_exponent=(float(inv["inversion_s"]),),
        outer_power=(float(inv["inversion_outer_power"]),),
        outer_end=(str(inv["inversion_outer_end"]),),
        high_k_override_k=(high[0],),
        high_k_override_r=(high[1],),
        high_k_override_rows=(high[2],),
        high_k_override_rule=(high[3],),
        low_k_override_k=(low[0],),
        low_k_override_r=(low[1],),
        method2_bands=method2,
        critical_band=critical,
    )


def _is_physical(record: Mapping[str, Any]) -> bool:
    return bool(record["physical"]) and not record["gates"]


def with_truncation_source(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the record with the truncated density made by the package.

    The truncated DF's edge (its Taylor form, under ``edge``, and the joins
    of the form, under ``edge_diag``) comes from `truncated_source` on the
    inputs the run recorded, and the whole result under ``source``; a record
    of a run that made no truncated density has None in all three.

    Raises
    ------
    ResultError
        If the run's source took a path the package does not make: the
        closed-form Type II edge piece, or an exact density substituted for
        the quadrature (a diagnostic).
    """
    recorded = record.get("truncation_source")
    if recorded is None:
        return {**record, "edge": None, "edge_diag": None, "source": None}
    inputs = recorded["inputs"]
    if inputs["n11"] or inputs["exact_substituted"]:
        raise ResultError(
            "The run's Poisson source took the Type II edge piece or the exact "
            "density; the package makes neither."
        )
    if not inputs["logit"]:
        raise ResultError(
            "The run's source quadrature read the DF against ln Q (a diagnostic); "
            "the package reads it against the logit only."
        )
    source = truncated_source(
        inputs["q"],
        inputs["f"],
        inputs["ddf_dq"],
        inputs["df_has_neg"],
        inputs["k_t"],
        inputs["p_center"],
        inputs["beta_c"],
        inputs["model_type"],
        _truncation_settings(record),
        tail=inputs["tail"],
        below_floor=inputs["below_floor"],
    )
    return {
        **record,
        "edge": source.taylor,
        "edge_diag": source.taylor_check,
        "source": source,
    }


def poisson_solve_of(record: Mapping[str, Any]) -> PoissonSolve | None:
    """Return the Poisson solve the package makes on the inputs the run's
    truncated solve recorded; None where the run made none.

    The source profile's potential places the grid's inner zone: for NFW
    log1p(R) / R, for the others `DoublePowerLaw.p`, as the run evaluates it.
    """
    recorded = record.get("truncation_source")
    if recorded is None or recorded.get("ode") is None:
        return None
    inputs = recorded["ode"]["inputs"]
    if inputs["is_nfw"]:

        def potential(r):
            return jnp.log1p(r) / r

    else:
        source = DoublePowerLaw(*inputs["profile"])

        def potential(r):
            return jnp.asarray(source.p(np.asarray(r, dtype=np.float64)))

    return run_poisson_ode(
        inputs["q"],
        inputs["p_rhs"],
        inputs["p_0"],
        inputs["ra"],
        inputs["bc"],
        inputs["model_type"],
        inputs["gamma_inner"],
        potential,
        _truncation_settings(record, log10_r_max=inputs["R_MAX_EXP"]),
        p_s=inputs["p_s"],
        sentinel_q=inputs["sentinel_q"],
        edge_continuation=float(recorded["inputs"]["k_t"]) > 0.0,
    )


def center_of(
    record: Mapping[str, Any], solve: PoissonSolve | None = None
) -> tuple[Array, CenterAmplitude] | None:
    """Return the density on the Poisson solve's rows and the model's center
    (the inner amplitude and the central deficit) the package makes on the
    inputs the run recorded; None where the run made no truncated solve.

    ``solve`` is `poisson_solve_of(record)` where the caller has it.
    """
    recorded = record.get("truncation_source")
    if recorded is None or recorded.get("centre") is None:
        return None
    if recorded["centre"]["inputs"]["goal"] != GOAL:
        raise ResultError(
            f"the run's goal {recorded['centre']['inputs']['goal']!r} is not the "
            f"package's {GOAL!r}"
        )
    solve = poisson_solve_of(record) if solve is None else solve
    if solve is None:
        raise ResultError("The run recorded no Poisson solve for its center.")
    ode = recorded["ode"]["inputs"]
    inputs = recorded["centre"]["inputs"]
    if inputs["first_order"]:
        raise ResultError(
            "The run kept the lowered center's first order (a diagnostic); the "
            "package solves it at all orders."
        )
    order = "cubic"
    m = source_slopes(ode["q"], ode["p_rhs"], float(ode["p_0"]), ode["p_s"], order)
    rho = profile_density(
        solve.r,
        solve.phi,
        ode["q"],
        ode["p_rhs"],
        ode["ra"],
        ode["bc"],
        ode["model_type"],
        p_0=float(ode["p_0"]),
        m=m,
        order=order,
    )
    center = center_amplitude(
        solve.r,
        solve.phi,
        rho,
        inputs["k_t"],
        inputs["p_0"],
        inputs["pot_center"],
        inputs["q"],
        inputs["df"],
        inputs["f_at_k_t"],
        inputs["bc"],
        inputs["profile"],
        inputs["factor"],
        n_per_decade=inputs["n_per_decade"],
    )
    return rho, center


def filled_of(
    record: Mapping[str, Any],
    solve: PoissonSolve | None = None,
    centerd: tuple[Array, CenterAmplitude] | None = None,
) -> FilledProfile | None:
    """Return the profile after the two fills the package makes on the inputs
    the run recorded; None where the run made no truncated solve.

    ``solve`` and ``centerd`` are `poisson_solve_of(record)` and
    `center_of(record)` where the caller has them.
    """
    recorded = record.get("truncation_source")
    if recorded is None or recorded.get("fill") is None:
        return None
    solve = poisson_solve_of(record) if solve is None else solve
    if solve is None:
        raise ResultError("The run recorded no Poisson solve for its fills.")
    centerd = center_of(record, solve) if centerd is None else centerd
    if centerd is None:
        raise ResultError("The run recorded no center for its fills.")
    rho, center = centerd
    ode = recorded["ode"]["inputs"]
    inputs = recorded["centre"]["inputs"]
    taylor = with_truncation_source(record)["edge"]
    edge = (
        None
        if taylor is None
        else (*taylor, float(ode["bc"]), str(ode["model_type"]), float(ode["ra"]))
    )
    return fill_profile(
        solve,
        rho,
        float(ode["p_0"]),
        center,
        inputs["profile"],
        inputs["bc"],
        inputs["factor"],
        edge,
        ode,
        _truncation_settings(record),
    )


def profile_rows_of(record: Mapping[str, Any]) -> ProfileRows | None:
    """Return the profile's rows, mass and central coordinate, and the central
    deficit, the package makes on the inputs the run recorded; None where the
    run made no truncated solve."""
    recorded = record.get("truncation_source")
    if recorded is None or recorded.get("rows") is None:
        return None
    solve = poisson_solve_of(record)
    centerd = center_of(record, solve)
    filled = filled_of(record, solve, centerd)
    if solve is None or centerd is None or filled is None:
        raise ResultError("The run recorded no solve, center or fill for its rows.")
    ode = recorded["ode"]["inputs"]
    inputs = recorded["centre"]["inputs"]
    center = centerd[1]
    return profile_rows(
        filled,
        solve,
        float(ode["p_0"]),
        center,
        inputs["profile"],
        inputs["bc"],
        inputs["k_t"],
        inputs["pot_center"],
        inputs["q"],
        inputs["df"],
        inputs["f_at_k_t"],
    )


def inversion_form_of(
    record: Mapping[str, Any],
) -> tuple[InversionForm, tuple[InversionFormRecord, ...]] | None:
    """Return the inversion's form the package chooses for the run's model on
    the energies the run inverted on (the sign report's grid); None where the
    run recorded none (a Method II child, whose pair is tabulated)."""
    inputs = record.get("truncation_gates")
    if record.get("method") == 2 or inputs is None or "q_sign" not in inputs:
        return None
    components = _components(record)
    k = np.asarray(inputs["q_sign"], dtype=np.float64)[None, :]
    pair = PotentialDensityPair(
        components=components,
        is_analytic=True,
        p_0=1.0,  # the internal unit of the potential
        r=np.zeros(0),
        p=np.zeros(0),
        rho=np.zeros((len(components), 0)),
    )
    grid = EnergyGrid(
        k=k,
        r_of_k=np.full(k.shape, np.nan),
        k_floor=(float(k[0, 0]),),
        k_ceiling=(float(k[0, -1]),),
    )
    return inversion_form(pair, grid)


# the record's top keys and the closed form's
CENTRE_KEYS = (
    ("top_kappa", "kappa"),
    ("top_kappa_branch", "branch"),
    ("top_form", "form"),
    ("top_singular_exponent", "kappa_sing"),
    ("top_rho_power", "s"),
    ("top_amplitude", "amplitude"),
    ("top_log_coefficient", "log_coef"),
    ("top_next_form", "next_form"),
    ("top_next_exponent", "next_kappa"),
    ("top_next_coefficient", "next_coef"),
    ("top_next_rho_power", "next_s"),
)


def center_form_of(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the DF's center in closed form the package finds for the run's
    model, under the record's top names; None for a Method II child, whose
    pair is tabulated."""
    if record.get("method") == 2:
        return None
    component = _components(record)[0]
    form = center_form(component.unit_profile, component.anisotropy)
    return {key: form[name] for key, name in CENTRE_KEYS}


def _aux_of(
    record: Mapping[str, Any],
) -> tuple[AuxDerivatives, InversionForm, InversionFormRecord] | None:
    """The exact derivatives on the energies the run inverted on, with the
    form; None for a Method II child or a run that recorded no rows."""
    rec = record.get("inversion_rows")
    made = inversion_form_of(record)
    if rec is None or made is None:
        return None
    form, records = made
    q = np.asarray(rec["q"], dtype=np.float64)
    grid = EnergyGrid(
        k=q[None, :],
        r_of_k=np.full((1, q.size), np.nan),
        k_floor=(float(q[0]),),
        k_ceiling=(float(q[-1]),),
    )
    return aux_derivatives(pair_of(record), grid, form), form, records[0]


def floor_source_of(
    record: Mapping[str, Any],
    aux: AuxDerivatives | None = None,
) -> dict[str, Any] | None:
    """Return the rule and the exact source of the completion below the floor
    the package makes for the run's model (`floor_source`), under the record's
    names; None where the model has none, or for a Method II child or a run
    that recorded no rows. ``aux`` is the exact derivatives where already
    made."""
    rec = record.get("inversion_rows")
    made = _aux_of(record) if aux is None else (aux, None, None)
    if rec is None or made is None or record.get("method") == 2:
        return None
    form_record = inversion_form_of(record)
    assert form_record is not None
    component = _components(record)[0]
    profile = component.unit_profile
    return floor_source(
        made[0],
        form_record[1][0],
        profile,
        component.anisotropy,
        float(np.min(np.asarray(rec["q"], dtype=np.float64))),
        int(_inversion_settings(record).inversion["n_floor_nodes"]),
    )


def below_floor_of(
    record: Mapping[str, Any],
    aux: AuxDerivatives | None = None,
    floor: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the DF below the written floor the package makes for the run's
    model (`below_floor`), ending on the first grid node at or above the
    written floor; None where the model has no completion below the floor.
    ``aux`` and ``floor`` are the exact derivatives and the completion's
    source where already made."""
    rec = record.get("inversion_rows")
    made = _aux_of(record) if aux is None else (aux, None, None)
    if rec is None or made is None or record.get("method") == 2:
        return None
    src = floor_source_of(record, made[0]) if floor is None else floor
    if src is None:
        return None
    q = np.asarray(rec["q"], dtype=np.float64)
    k_written = float(np.min(q[q >= OUTPUT_FLOOR_FACTOR * K_FLOOR]))
    return below_floor(made[0], src, k_written)


def kernel_sources_of(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the kernel's sources the package makes for the run's model on the
    energies the run inverted on, from the exact derivatives (aux_derivatives),
    under the record's names; None for a Method II child or a run that
    recorded no rows.

    With j the order in use, the kernel's source ``g1`` is rho_aux^(j+1), or
    rho_aux^(j) where the remainder n is zero; ``g2`` is one order above it;
    ``g0`` is rho_aux^(j); the slope ``g1_slope`` is one order above ``g1``,
    zero where ``g1`` is zero; ``dm_j`` and ``dm_jm1`` are rho_aux^(j) and
    rho_aux^(j-1) at the grid's first energy.
    """
    rec = record.get("inversion_rows")
    made = _aux_of(record)
    if rec is None or made is None:
        return None
    aux, form, _ = made
    by_order = {order: aux.values[0, i] for i, order in enumerate(aux.orders)}
    at_floor = dict(zip(aux.orders, aux.at_floor[0], strict=True))
    j = int(form.order_m[0])
    whole = order_pair(0.0 if rec["is_om"] else float(rec["beta_c"])).order_n == 0.0
    order = j if whole else j + 1
    g1 = by_order[order]
    slope = by_order[order + 1]
    if not np.all(np.isfinite(slope)):
        raise ResultError(
            f"The source's slope rho_aux^({order + 1}) is not finite on the grid: "
            "the kernel would have taken PCHIP slopes of the table instead."
        )
    return {
        "g1": g1,
        "g2": slope,
        "g0": by_order[j],
        "g1_slope": np.where(g1 != 0.0, slope, 0.0),
        "dm_j": float(at_floor[j]),
        "dm_jm1": float(at_floor[j - 1]) if j - 1 in at_floor else 0.0,
    }


def kernel_rows_of(
    record: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
) -> tuple[Array, Array | None] | None:
    """Return the inversion kernel's F rows and dF/dQ rows the package makes on
    the inputs the run recorded (before the completion below the floor); None
    where the run recorded none. dF/dQ is None where the run made none.

    ``sources`` (from `kernel_sources_of`) replaces the record's sources: the
    tables ``g1``, ``g0``, ``g2`` and ``g1_slope`` and the floor values
    ``dm_j`` and ``dm_jm1``; the form's scalars stay the record's."""
    rec = record.get("inversion_rows")
    if rec is None:
        return None
    if sources is not None:
        rec = {
            **rec,
            **{
                key: value
                for key, value in sources.items()
                if not (key in ("g1_slope", "g2") and rec.get(key) is None)
            },
        }
    rules = kernel_rules(_inversion_settings(record))
    common = dict(
        beta_c=float(rec["beta_c"]),
        is_om=bool(rec["is_om"]),
        rules=rules,
        dm_j=float(rec["dm_j"]),
        p_0=float(rec["p_0"]),
        g0=rec["g0"],
        dm_jm1=float(rec["dm_jm1"]),
        switched=bool(rec["switched"]),
        p_s=float(rec["p_s"]),
        s=float(rec["s"]),
        outer_from=float(rec["outer_from"]),
    )
    q = jnp.asarray(rec["q"], dtype=jnp.float64)
    g1 = jnp.asarray(rec["g1"], dtype=jnp.float64)
    slope = None if rec["g1_slope"] is None else jnp.asarray(rec["g1_slope"])
    f = np.asarray(inversion_kernel(q, g1, g1_slope=slope, **common), dtype=np.float64)
    if rec.get("ddf") is None:
        return f, None
    g2 = jnp.asarray(rec["g2"], dtype=jnp.float64)
    half = (
        order_pair(0.0 if rec["is_om"] else float(rec["beta_c"])).kernel_exponent == 0.0
    )
    ddf = inversion_kernel(q, g2 if half else g1, deriv=True, g2=g2, **common)
    return f, np.asarray(ddf, dtype=np.float64)


def rows_of(
    record: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
    floor: Mapping[str, Any] | None = None,
) -> tuple[Array, Array | None] | None:
    """Return the inversion's rows as made (F and dF/dQ, the kernel's rows
    completed below the floor) the package makes on the inputs the run
    recorded, the kernel's sources replaced by ``sources`` and the completion's
    by ``floor`` where given; None where the run recorded none."""
    made = kernel_rows_of(record, sources)
    if made is None:
        return None
    rec = record["inversion_rows"]
    return complete_rows(
        rec["q"],
        made[0],
        made[1],
        edge=rec.get("n11"),
        floor=rec.get("n10") if floor is None else floor,
        m2_tail=rec.get("m2_tail"),
    )


def _center_kappa(record: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """The center's form and the exponent the top patch takes: the singular
    one on the branches whose DF diverges at p_0, else 0."""
    component = _components(record)[0]
    center = center_form(component.unit_profile, component.anisotropy)
    patched = center["branch"] in ("regular", "critical_divergent")
    return center, float(center["kappa_sing"]) if patched else 0.0


def anchor_of(record: Mapping[str, Any]) -> Anchor | None:
    """Return where the package's form takes over from the inversion's own
    rows (the package's rows as made, the center's form, the pair's data top
    at the innermost radius of the run's settings); None where no rule fires,
    for a Type II anisotropy, or for a Method II child."""
    if record.get("method") == 2 or record.get("inversion_rows") is None:
        return None
    if _components(record)[0].anisotropy.kind == "II":
        # rho_aux changes sign at r_a: the inversion takes the linear spline,
        # which carries no form above its rows
        return None
    made = rows_of(record)
    assert made is not None
    f, ddf = made
    q = np.asarray(record["inversion_rows"]["q"], dtype=np.float64)
    center, kappa = _center_kappa(record)
    profile = _components(record)[0].unit_profile
    r_min = 10.0 ** float(_truncation_settings(record)["log10_r_min"])
    data_top = float(np.asarray(profile.p(r_min)))
    return anchor(q, f, 1.0, kappa, center, data_top, ddf)


def top_patch_of(record: Mapping[str, Any]) -> TopPatch | None:
    """Return the form the package fits above its own anchor of the run's
    inversion (`anchor_of`), on the inversion's rows to the anchor; None where
    no rule fires or the form has no row to take over, or for a Method II
    child."""
    at_anchor = anchor_of(record)
    if at_anchor is None:
        return None
    gates = record["truncation_gates"]
    q = np.asarray(gates["q_sign"], dtype=np.float64)
    f = np.asarray(gates["f_sign"], dtype=np.float64)
    center, kappa = _center_kappa(record)
    amplitude = float(center["amplitude"])
    closed = (
        center["form"] == "power"
        and math.isfinite(amplitude)
        and abs(float(center["kappa_sing"]) - kappa) <= 1e-12
    )
    at = q <= at_anchor.k_anchor
    return top_patch(q[at], f[at], 1.0, kappa, amplitude if closed else None, q[~at])


def pair_of(record: Mapping[str, Any]) -> PotentialDensityPair:
    """Return the pair the package inverts for the run's model: the analytic
    one of an untruncated component, a Method II child's parent's rows."""
    components = _components(record)
    parent = components[0].profile if record.get("method") == 2 else None
    assert parent is None or isinstance(parent, TruncatedProfile)
    return potential_density_pair(components, parent)


def grid_of(record: Mapping[str, Any]) -> tuple[Array, GridZones] | None:
    """Return the energies the package lays for the run's inversion and its
    zones, the pilot inversion's feature taken from the run's record; None for
    a Method II child (a tabulated pair) or a run that recorded no zones."""
    zones = record.get("inversion", {}).get("grid_zones")
    if record.get("method") == 2 or zones is None:
        return None
    component = _components(record)[0]
    return inversion_grid(
        component.unit_profile, component.anisotropy, _inversion_settings(record), zones
    )


def gates_of(record: Mapping[str, Any]) -> TruncationGates | None:
    """Return the truncation gates the package finds on the inputs the run
    recorded; None where the run recorded none (a Method II child)."""
    inputs = record.get("truncation_gates")
    if inputs is None or "k_t" not in inputs:
        return None
    return truncation_gates(
        inputs["q_sign"],
        inputs["f_sign"],
        inputs["q"],
        inputs["f"],
        inputs["k_t"],
        inputs["a_analytic"],
    )


def validated_of(record: Mapping[str, Any]) -> ValidatedRows | None:
    """Return the profile's validated rows the package makes on the inputs the
    run recorded; None where the run made no truncated solve."""
    recorded = record.get("truncation_source")
    if recorded is None or recorded.get("rows") is None:
        return None
    solve = poisson_solve_of(record)
    centerd = center_of(record, solve)
    filled = filled_of(record, solve, centerd)
    if solve is None or centerd is None or filled is None:
        raise ResultError("The run recorded no solve, center or fill to validate.")
    ode = recorded["ode"]["inputs"]
    inputs = recorded["centre"]["inputs"]
    rows = profile_rows(
        filled,
        solve,
        float(ode["p_0"]),
        centerd[1],
        inputs["profile"],
        inputs["bc"],
        inputs["k_t"],
        inputs["pot_center"],
        inputs["q"],
        inputs["df"],
        inputs["f_at_k_t"],
    )
    return validate_rows(
        rows,
        solve,
        inputs["k_t"],
        inputs["pot_center"],
        DoublePowerLaw(*ode["profile"]),
        _truncation_settings(record, log10_r_max=ode["R_MAX_EXP"]),
    )


def _inversion_table(record: Mapping[str, Any]) -> DfTable:
    """The inversion's own table: the rows as made (the package's, completed
    below the floor), ending at the anchor of the inversion's own top patch,
    joined below to the inversion's own nodes below the floor."""
    made = rows_of(record)
    assert made is not None
    f, ddf = made
    q = np.asarray(record["inversion_rows"]["q"], dtype=np.float64)
    patch = top_patch_of(record)
    anchor = patch.k_anchor if patch is not None else np.inf
    below = record["truncation_gates"].get("below")
    k_w = float(np.max(below["K"])) if below is not None else -np.inf
    keep = (q <= anchor) & (q >= k_w * (1.0 - 1e-12)) & np.isfinite(f) & (f != 0.0)
    q, f = q[keep], f[keep]
    log_df = np.log10(np.abs(f))
    slope = (
        np.full(q.shape, np.nan)
        if ddf is None
        else q * np.asarray(ddf)[keep] / (f * np.log(10.0))
    )
    extra: dict[str, Any] = {}
    if np.any(f < 0.0):
        extra["sign"] = np.sign(f).astype(np.int8)[None, :]
    exact = bool(np.all(np.isfinite(slope)))
    if below is not None:
        kb = np.asarray(below["K"], dtype=np.float64)
        fb = np.asarray(below["F"], dtype=np.float64)
        order = np.argsort(kb)
        kb, fb = kb[order], fb[order]
        db = below.get("dFdK")
        sb = (
            np.full(kb.shape, np.nan)
            if db is None
            else kb * np.asarray(db, dtype=np.float64)[order] / (fb * np.log(10.0))
        )
        exact = exact and bool(np.all(np.isfinite(sb)))
        extra.update(
            k_below=kb[None, :],
            log_df_below=np.log10(fb)[None, :],
            log_df_slope_below=sb[None, :],
            n_below=(int(kb.size),),
        )
    return DfTable(
        k=q[None, :],
        log_df=log_df[None, :],
        log_df_slope=slope[None, :],
        i_first=(0,),
        i_last=(int(q.size) - 1,),
        slopes_exact=(exact,),
        **extra,
    )


def _inversion_form(record: Mapping[str, Any]) -> AsymptoticForm:
    """The forms beyond the inversion's own table: the center's with the anchor
    of the inversion's own top patch (NaN where the rows reach the ceiling),
    and below the floor the tail where the DF has one, else zero."""
    form = _form(record, _checks(record))
    tail = record.get("tail")
    fields: dict[str, Any] = dict(
        p_0=1.0, below_floor=("zero",), below_floor_coefficients=((),)
    )
    if tail is not None and float(tail["A"]) > 0.0:
        fields.update(
            below_floor=("tail",),
            tail_exponent=(float(tail["a"]),),
            tail_amplitude=(float(tail["A"]),),
            tail_log_exponent=(float(tail.get("k", 0.0)),),
            tail_log_scale=(float(tail.get("E0", 1.0)),),
        )
    patch = top_patch_of(record)
    if patch is not None:
        fields.update(
            central_anchor=(patch.k_anchor,),
            central_anchor_amplitude=(patch.a_anchor,),
            central_regular=((patch.b, patch.d),),
        )
    return dataclasses.replace(form, **fields)


def _inversion_df(
    record: Mapping[str, Any],
) -> tuple[DfTable, AsymptoticForm, float, ValidationReport] | None:
    """The inversion's own DF and its report (validate_df); None where the
    run recorded no rows of its own (a Method II child)."""
    if record.get("method") == 2 or record.get("inversion_rows") is None:
        return None
    table = _inversion_table(record)
    form = _inversion_form(record)
    pair = PotentialDensityPair(
        components=_components(record),
        is_analytic=True,
        p_0=1.0,
        r=np.zeros(0),
        p=np.zeros(0),
        rho=np.zeros((1, 0)),
    )
    k_t_min, report, _ = validate_df(pair, table, form)
    return table, form, k_t_min, report


def _detail(report: ValidationReport) -> str:
    """The checks above the goal, with their errors and where they occur."""
    return "; ".join(
        f"{key} {check.error:.2e}"
        + ("" if check.where is None else f" at {check.where:.6g}")
        for key, check in report.checks.items()
        if check.level > 0
    )


def df_from_record(record: Mapping[str, Any]) -> DistributionFunction:
    """Return the untruncated distribution function a run's record holds.

    Parameters
    ----------
    record : mapping
        The record of a run of an untruncated model, physical or flagged, or
        of a Method II child (the DF of a component on its parent's profile).

    Returns
    -------
    df : DistributionFunction

    Raises
    ------
    ResultError
        If the run is truncated: its table is the truncated model's DF.
    """
    if _truncation(record) is not None:
        raise ResultError(
            "The record is of a truncated run; it holds no untruncated DF."
        )
    record = with_truncation_source(record)
    components, settings = _components(record), _settings(record)
    checks = _checks(record)
    inverted = _inversion_df(record)
    if inverted is not None:
        table, form, k_t_min, report = inverted
        if table.sign.size == 0:  # a signed DF's flagged report is not measured
            announce(
                f"The distribution function of {components[0].name!r}",
                report,
                _detail(report),
            )
    else:
        table, form = _table(record), _form(record, checks)
        k_t_min = 0.0 if record.get("sign") is None else float(record["sign"][0])
        report = _df_report(record, checks)
    k = np.asarray(table.k[0], dtype=np.float64)
    r_of_k = np.full(k.shape, np.nan)
    rows = None
    if record.get("profile"):
        rows = _profile_rows(record)
        pos = rows["Psi"] > 0.0  # an edge row's potential is zero
        order = np.argsort(rows["Psi"][pos])
        r_of_k = np.exp(
            np.interp(
                np.log(k),
                np.log(rows["Psi"][pos][order]),
                np.log(rows["R"][pos][order]),
                left=np.nan,
                right=np.nan,
            )
        )
    return DistributionFunction(
        components=components,
        settings=settings,
        identity=model_identity(components, None, settings),
        grid=EnergyGrid(
            k=k[None, :],
            r_of_k=r_of_k[None, :],
            k_floor=(float(k[0]),),
            k_ceiling=(float(k[-1]),),
        ),
        table=table,
        form=form,
        k_t_min=k_t_min,
        validation=report,
        timing={},
        inversion=_inversion(record, rows),
    )


def model_from_record(record: Mapping[str, Any]) -> Model:
    """Return the model a run's record holds.

    Parameters
    ----------
    record : mapping
        The record of a run: a truncated model (Method I or Method II) or an
        untruncated one.

    Returns
    -------
    model : Model

    Raises
    ------
    UnphysicalModelError
        If a gate fired; it carries the untruncated DF as its result where
        the run is untruncated.
    """
    if not _is_physical(record):
        result = df_from_record(record) if _truncation(record) is None else None
        raise UnphysicalModelError(
            f"The run was flagged (gates {'; '.join(record['gates']) or 'none'}).",
            result=result,
        )
    record = with_truncation_source(record)
    components, settings = _components(record), _settings(record)
    truncation = _truncation(record)
    checks = _checks(record)
    rows = _profile_rows(record)
    identity = model_identity(components, truncation, settings)
    k_t = 0.0 if truncation is None or truncation.k_t is None else float(truncation.k_t)
    profile = _profile(record, rows, k_t, components[0], identity)
    table = _table(record)
    form = _form(record, checks)
    parents_level = 0
    inverted = _inversion_df(record)
    if inverted is not None:
        df_report = inverted[3]
        # a signed DF's report is the flagged one, whose table checks are not
        # measured (NaN): it is not announced and hands its model no tier; what
        # a signed DF passes on is its gates (k_t_min), which the truncation reads
        if inverted[0].sign.size == 0:
            announce(
                f"The distribution function of {components[0].name!r}",
                df_report,
                _detail(df_report),
            )
            parents_level = df_report.level
        if truncation is None:
            # the untruncated model's DF is the inversion's own
            table, form = inverted[0], inverted[1]
    measured = profile_checks(
        profile, table, form, components[0].anisotropy, settings.truncation
    )
    report = _model_report(record, checks, measured, parents_level)
    announce(f"The model of {components[0].name!r}", report, _detail(report))
    return Model(
        components=components,
        truncation=truncation,
        settings=settings,
        identity=identity,
        parent=model_identity(components, None, settings),
        table=table,
        form=form,
        profile=profile,
        validation=report,
        timing={},
        r_trust=_r_trust(record),
        inversion=_inversion(record, rows),
    )
