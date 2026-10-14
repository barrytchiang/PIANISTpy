"""The checks a DF table states about itself, and the validity certificate.

The checks are the working reader's (``pianistpy_io.df_lookup``), made on the
table as it is written: the stated slope column against the table's own F,
the stated center against the table's top decade, and the joins of the forms
below the floor with the table's bottom row. A table that fails one beyond
the hard limit is refused, as the reader refuses it.

The same checks are measured on the inversion's own table by `measure_table`,
which refuses nothing: each value is judged on the tiers by the report that
holds it.

The certificate is closed form: from the model and the exponent of the DF at
the center, whether the table's columns are representable in float64.
"""

__all__ = [
    "TableChecks",
    "measure_table",
    "slope_column_excess",
    "table_checks",
    "validity_certificate",
]

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from ..exceptions import ResultError
from ..typing import Array
from ..validation import GOAL, HARD_LIMIT
from .asymptotics import tail_log10_df


@dataclass(frozen=True)
class TableChecks:
    """What a table states about itself.

    Parameters
    ----------
    kappa : float
        Exponent of u = p_0 - k that the DF follows above the table: the
        stated one, or the slope of the table's top decade where none is
        stated.
    kappa_residual : float
        How far the table's top decade departs from the stated center;
        zero where no center is stated.
    k_top : float
        Last energy of the table.
    slope_column_excess : float
        The worst relative error in F that the stated dF/dK column makes
        beyond what the table's own F can say; zero where the column is not
        used.
    edge_rel_at_floor : float
        The form below the floor against the table's bottom row, F_form /
        F_row - 1; NaN where the table has no such form.
    below_join : float
        The last node below the floor against the table's first row, F_node /
        F_row - 1; NaN where there are no such nodes, or the last does not
        lie on the first row.
    tail_join : float
        The tail against the lowest node it continues, F_tail / F_node - 1;
        NaN where the DF has no tail.
    """

    kappa: float
    kappa_residual: float
    k_top: float
    slope_column_excess: float
    edge_rel_at_floor: float
    below_join: float = float("nan")
    tail_join: float = float("nan")


def slope_column_excess(
    x: Array, log_df: Array, slope: Array, split: int | None = None
) -> Array:
    """Return how far a stated slope column moves the lookup, per node.

    At each node the derivative of the polynomial through the 3 and through
    the 5 nearest nodes; the 5-point value is uncertain by their difference
    (the largest over the node and its two neighbors) plus its rounding. A
    slope beyond that uncertainty, by dm, moves log10 F between the node's
    neighbors by at most dm h 4/27, h the longer adjacent interval.

    Where the nodes change their spacing at ``split`` (the first node of the
    table above the nodes below its floor), a node whose centerd stencils
    would cross the change takes stencils on its own side, one-sided, and h
    is the interval on that side: an exact difference on the nodes' own
    spacing, no node moved.

    Parameters
    ----------
    x : array, shape (n,)
        Logarithm of the energies, increasing.
    log_df : array, shape (n,)
        Base-10 logarithm of the DF.
    slope : array, shape (n,)
        The stated d log10 F / d ln k.
    split : int, optional
        The first node above a change of spacing; none where omitted.

    Returns
    -------
    excess : array, shape (n,)
        The relative error in F the slope makes at each node.
    """
    n = x.size
    if n < 5:
        return np.zeros(n)
    i = np.arange(n)
    near = np.zeros(n, dtype=bool)
    if split is not None and 0 < split < n:
        near = (i >= split - 2) & (i <= split + 1)
    up = near & (i >= split if split is not None else False)
    down = near & ~up

    def deriv(m):
        lo = np.clip(i - m // 2, 0, n - m)
        lo = np.where(up, np.minimum(i, n - m), lo)
        lo = np.where(down, np.maximum(i - m + 1, 0), lo)
        idx = lo[:, None] + np.arange(m)[None, :]
        h = np.maximum(
            np.abs(x[np.minimum(i + 1, n - 1)] - x[np.maximum(i - 1, 0)]), 1e-300
        )
        dx = (x[idx] - x[i][:, None]) / h[:, None]
        a = dx[:, None, :] ** np.arange(m)[None, :, None]
        b = np.zeros((n, m))
        b[:, 1] = 1.0
        w = np.linalg.solve(a, b[:, :, None])[:, :, 0] / h[:, None]
        return (
            np.sum(w * log_df[idx], axis=1),
            np.sum(np.abs(w), axis=1) * np.max(np.abs(log_df[idx]), axis=1),
        )

    c3, _ = deriv(3)
    c5, r5 = deriv(5)
    spread = np.abs(c3 - c5)
    spread = np.maximum.reduce(
        [spread, np.r_[spread[1:], spread[-1]], np.r_[spread[0], spread[:-1]]]
    )
    uncertainty = spread + np.finfo(np.float64).eps * r5
    d = np.diff(x)
    h = np.maximum(np.r_[d, d[-1]], np.r_[d[0], d])
    h = np.where(up, np.r_[d, d[-1]], np.where(down, np.r_[d[0], d], h))
    return (
        np.maximum(np.abs(slope - c5) - uncertainty, 0.0)
        * h
        * (4.0 / 27.0)
        * np.log(10.0)
    )


def _next_term(top: Mapping[str, Any]):
    """The next singular term of the center, c u^e or c u^e (-ln u)."""
    form = top.get("top_next_form")
    if form not in ("power", "log_power") or top.get("top_form") == "analytic":
        return None
    e, c = float(top["top_next_exponent"]), float(top["top_next_coefficient"])
    if not (np.isfinite(e) and np.isfinite(c)):
        raise ResultError(
            f"The table states a next term {form!r} without its finite "
            "exponent and coefficient."
        )
    if form == "power":
        return lambda u: c * np.asarray(u, dtype=np.float64) ** e
    return lambda u: (
        c
        * np.asarray(u, dtype=np.float64) ** e
        * (-np.log(np.asarray(u, dtype=np.float64)))
    )


def _center(top, interp, k, log_df, p_0, refuse=True):
    """The exponent above the top and the residual of the stated center.

    With ``refuse`` False nothing is refused: a residual beyond the hard
    limit is returned as it is, and one that cannot be measured as infinite.
    """
    k_top, k_0 = float(k[-1]), float(k[0])
    lg_top = float(log_df[-1])
    u = p_0 - k
    sel = u <= 10.0 * (p_0 - k_top)
    if int(sel.sum()) < 5:
        if not refuse:
            kappa_attr = top.get("top_kappa")
            return (float("nan") if kappa_attr is None else float(kappa_attr)), np.inf
        raise ResultError(
            f"Only {int(sel.sum())} nodes in the last decade of p_0 - k: the "
            "exponent at the center is not measurable."
        )
    kappa_fit = float(np.polyfit(np.log10(u[sel]), log_df[sel], 1)[0])
    kappa_attr = top.get("top_kappa")
    form = top.get("top_form")
    if form is not None:
        s = float(top.get("top_singular_exponent", np.nan))
        amp = float(top.get("top_amplitude", np.nan))
        log_coef = float(top.get("top_log_coefficient", np.nan))
        need = {
            "power": (s, amp),
            "constant_plus_power": (s, amp),
            "log": (log_coef,),
            "analytic": (),
        }
        if form not in need or not all(np.isfinite(v) for v in need[form]):
            raise ResultError(
                f"The table states the center's form {form!r} without its "
                "finite coefficient."
            )
        kappa = float(kappa_attr)
        next_term = _next_term(top)
        u_top = p_0 - k_top
        u_10 = 10.0 * u_top
        if p_0 - u_10 < k_0:
            return kappa, 0.0
        lg_10 = float(interp(np.log(p_0 - u_10)))

        def g2(uu):
            return float(next_term(uu)) if next_term is not None else 0.0

        if form == "power" and amp > 0.0:
            scale = np.log10(amp) + s * np.log10(u_top)
            f_top, f_10 = 10.0 ** (lg_top - scale), 10.0 ** (lg_10 - scale)
            check = (
                (f_10 - (u_10 / u_top) ** s - g2(u_10) * 10.0 ** (-scale))
                - (f_top - 1.0 - g2(u_top) * 10.0 ** (-scale))
            ) / f_top
        else:
            f_10, f_top = 10.0**lg_10, 10.0**lg_top

            def singular(uu):
                if form in ("power", "constant_plus_power"):
                    return amp * uu**s
                return -log_coef * np.log(uu) if form == "log" else 0.0

            check = (
                (f_10 - singular(u_10) - g2(u_10))
                - (f_top - singular(u_top) - g2(u_top))
            ) / f_top
        residual = abs(check)
        if refuse and not residual <= HARD_LIMIT:
            raise ResultError(
                f"The stated center ({form!r}) does not describe the table: "
                "across the top decade the table less "
                f"its singular part changes by {check:+.3e} of F_top."
            )
        return kappa, residual
    if kappa_attr is not None and np.isfinite(float(kappa_attr)):
        kappa = float(kappa_attr)
        if abs(kappa_fit - kappa) <= GOAL:
            return kappa, 0.0
        gaps = []
        for j in range(3):
            sel = (u >= 10.0**j * (p_0 - k_top) * (1.0 - 1e-12)) & (
                u <= 10.0 ** (j + 1) * (p_0 - k_top) * (1.0 + 1e-12)
            )
            if int(sel.sum()) < 5:
                break
            gaps.append(float(np.polyfit(np.log10(u[sel]), log_df[sel], 1)[0]) - kappa)
        converges = (
            len(gaps) == 3
            and all(g != 0.0 for g in gaps)
            and np.sign(gaps[0]) == np.sign(gaps[1]) == np.sign(gaps[2])
            and abs(gaps[0]) < abs(gaps[1]) < abs(gaps[2])
        )
        reached = len(gaps) >= 1 and any(abs(g) <= GOAL for g in gaps)
        if not (converges or reached):
            if not refuse:
                return kappa, np.inf
            raise ResultError(
                f"The table does not converge to its stated exponent {kappa!r} "
                "at the center."
            )
        if converges:
            return kappa, abs(gaps[0]) / float(np.log10(abs(gaps[1]) / abs(gaps[0])))
        return kappa, float(np.log(10.0) * abs(gaps[0]))
    return kappa_fit, 0.0


def table_checks(
    k: Array,
    log_df: Array,
    ddf_dk: Array | None,
    p_0: float,
    top: Mapping[str, Any],
    below: Mapping[str, Array] | None = None,
    tail: Mapping[str, float] | None = None,
    edge_taylor: tuple[float, float, float] | None = None,
    edge_law: tuple[float, float, float] | None = None,
) -> TableChecks:
    """Return the checks of a DF table, as the working reader makes them.

    Parameters
    ----------
    k, log_df : array, shape (n,)
        Energies and base-10 logarithm of the DF, as written. Leading rows
        with F = 0 are outside the support.
    ddf_dk : array, shape (n,) or None
        The written dF/dK column, if any.
    p_0 : float
        Central potential of the table, 1 - k_t.
    top : mapping
        The stated center: ``top_kappa``, ``top_form``, its exponent,
        amplitude or logarithm's coefficient, and its next term.
    below : mapping, optional
        The exact nodes below the written floor, ``K``, ``logF`` and
        ``dFdK``, ending at the floor.
    tail : mapping, optional
        The tail below the floor, ``A``, ``a``, ``k``, ``E0``.
    edge_taylor : tuple of float, optional
        F_1, F_2 and the switch energy of a truncated table's Taylor form.
    edge_law : tuple of float, optional
        p, C_0 and C_1 of a Method II child's edge law.

    Returns
    -------
    checks : TableChecks

    Raises
    ------
    ResultError
        Where the reader refuses the table.
    """
    order = np.argsort(k)
    k, log_df = (
        np.asarray(k, dtype=np.float64)[order],
        np.asarray(log_df, dtype=np.float64)[order],
    )
    ddf = None if ddf_dk is None else np.asarray(ddf_dk, dtype=np.float64)[order]
    lead = int(np.argmax(~np.isneginf(log_df))) if np.isneginf(log_df[0]) else 0
    if not np.all(np.isfinite(log_df[lead:])):
        raise ResultError("The table's logF is not finite inside its support.")
    k, log_df = k[lead:], log_df[lead:]
    ddf = None if ddf is None else ddf[lead:]
    k_0_written = float(k[0])
    if below is not None:
        kb = np.asarray(below["K"], dtype=np.float64)
        lgb = np.asarray(below["logF"], dtype=np.float64)
        dfb = (
            None
            if below.get("dFdK") is None
            else np.asarray(below["dFdK"], dtype=np.float64)
        )
        ob = np.argsort(kb)
        kb, lgb = kb[ob], lgb[ob]
        dfb = None if dfb is None else dfb[ob]
        if not (np.all(np.isfinite(lgb)) and kb[-1] <= k_0_written * (1.0 + 1e-12)):
            raise ResultError(
                "The nodes below the floor are not a finite table ending at "
                "the written floor."
            )
        join = (
            float(10.0 ** (lgb[-1] - log_df[0]))
            if abs(kb[-1] / k_0_written - 1.0) <= 1e-12
            else float("nan")
        )
        if not abs(join - 1.0) <= GOAL:
            raise ResultError(
                "The nodes below the floor do not join the table at its written "
                f"floor: off by {abs(join - 1.0):.3e}."
            )
        keep = kb < k_0_written * (1.0 - 1e-12)
        k, log_df = np.concatenate([kb[keep], k]), np.concatenate([lgb[keep], log_df])
        ddf = None if (ddf is None or dfb is None) else np.concatenate([dfb[keep], ddf])
    zero_top = 0 if ddf is None else int(np.sum(np.cumprod(ddf[::-1] == 0.0)))
    certified = ddf is not None and np.all(np.isfinite(ddf)) and zero_top == 0
    slope = k * ddf * 10.0 ** (-log_df) / np.log(10.0) if certified else None
    checks = _measure(k, log_df, slope, p_0, top, tail, edge_taylor, edge_law, True)
    return dataclasses.replace(
        checks, below_join=join - 1.0 if below is not None else np.nan
    )


def measure_table(
    k: Array,
    log_df: Array,
    log_df_slope: Array | None,
    p_0: float,
    top: Mapping[str, Any],
    below: tuple[Array, Array, Array | None] | None = None,
    tail: Mapping[str, float] | None = None,
    edge_taylor: tuple[float, float, float] | None = None,
    edge_law: tuple[float, float, float] | None = None,
) -> TableChecks:
    """Return the checks of a DF table, refusing none of them.

    The checks of `table_checks`, measured on a table and its slopes as the
    inversion states them: a value beyond a limit is returned, for the
    report that holds it to judge on the tiers.

    Parameters
    ----------
    k, log_df : array, shape (n,)
        Energies, increasing, and base-10 logarithm of the DF, finite.
    log_df_slope : array, shape (n,) or None
        The exact d log10 F / d ln k on every node; None where the table
        states no exact slope.
    p_0 : float
        Central potential of the table.
    top : mapping
        The stated center, under the names of `table_checks`.
    below : tuple of array, optional
        The nodes below the table: energies, log10 F and its slope (None
        where not exact); the last may lie on the table's first row.
    tail, edge_taylor, edge_law : optional
        The forms below the floor, as for `table_checks`.

    Returns
    -------
    checks : TableChecks
    """
    k = np.asarray(k, dtype=np.float64)
    log_df = np.asarray(log_df, dtype=np.float64)
    slope = None if log_df_slope is None else np.asarray(log_df_slope, dtype=np.float64)
    join = np.nan
    split = None
    if below is not None:
        kb = np.asarray(below[0], dtype=np.float64)
        lgb = np.asarray(below[1], dtype=np.float64)
        sb = None if below[2] is None else np.asarray(below[2], dtype=np.float64)
        if kb.size:
            if abs(kb[-1] / k[0] - 1.0) <= 1e-12:
                join = float(10.0 ** (lgb[-1] - log_df[0])) - 1.0
            keep = kb < k[0] * (1.0 - 1e-12)
            split = int(keep.sum())
            k, log_df = (
                np.concatenate([kb[keep], k]),
                np.concatenate([lgb[keep], log_df]),
            )
            slope = None if (slope is None or sb is None) else np.r_[sb[keep], slope]
    checks = _measure(
        k, log_df, slope, p_0, top, tail, edge_taylor, edge_law, False, split
    )
    return dataclasses.replace(checks, below_join=join)


def _measure(
    k, log_df, slope, p_0, top, tail, edge_taylor, edge_law, refuse, split=None
):
    """The checks on a table joined to its nodes below; the reader's refusals
    where ``refuse``."""
    x = np.log(k)
    excess = 0.0
    if slope is not None:
        per_node = slope_column_excess(x, log_df, slope, split)
        excess = float(np.max(per_node)) if per_node.size else 0.0
        if refuse and np.any(per_node > HARD_LIMIT):
            raise ResultError(
                "The dF/dK column disagrees with the table's own F beyond the "
                "hard limit."
            )
        interp = CubicHermiteSpline(x, log_df, slope)
    else:
        interp = CubicSpline(x, log_df, bc_type="not-a-knot")
    kappa, residual = _center(top, interp, k, log_df, p_0, refuse)
    k_0, lg_0 = float(k[0]), float(log_df[0])
    tail_join = float("nan")
    if tail is not None and float(tail["A"]) > 0.0:
        ratio = (
            float(
                10.0
                ** tail_log10_df(
                    float(tail["A"]),
                    float(tail["a"]),
                    k_0,
                    float(tail.get("k", 0.0)),
                    float(tail.get("E0", 1.0)),
                )
            )
            / 10.0**lg_0
        )
        tail_join = ratio - 1.0
        if refuse and not abs(ratio - 1.0) <= GOAL:
            raise ResultError(
                "The tail does not join the table at its floor: off by "
                f"{abs(ratio - 1.0):.3e}."
            )
    edge_rel = float("nan")
    if edge_taylor is not None:
        f_1, f_2, k_switch = edge_taylor
        if not k_0 <= k_switch:
            raise ResultError("The table's floor is above the Taylor form's switch.")
        edge_rel = float((f_1 * k_0 + 0.5 * f_2 * k_0 * k_0) / 10.0**lg_0 - 1.0)
        if refuse and not abs(edge_rel) <= GOAL:
            raise ResultError(
                "The Taylor form disagrees with the table's bottom row by "
                f"{edge_rel:+.3e}."
            )
    if edge_law is not None:
        if edge_taylor is not None:
            raise ResultError(
                "The table states both a Taylor form and an edge law below its floor."
            )
        p, c_0, c_1 = edge_law
        edge_rel = float((c_0 * k_0**p + c_1 * k_0 ** (p + 1.0)) / 10.0**lg_0 - 1.0)
        if refuse and not abs(edge_rel) <= GOAL:
            raise ResultError(
                "The edge law disagrees with the table's bottom row by "
                f"{edge_rel:+.3e}."
            )
    return TableChecks(
        kappa=kappa,
        kappa_residual=residual,
        k_top=float(k[-1]),
        slope_column_excess=excess,
        edge_rel_at_floor=edge_rel,
        tail_join=tail_join,
    )


def validity_certificate(
    gamma: float, kappa_top: float, u_ceiling: float
) -> dict[str, Any]:
    """Return where float64 holds the table's columns, from closed forms.

    Parameters
    ----------
    gamma : float
        Inner slope of the density.
    kappa_top : float
        Exponent of u that the DF follows at the center; NaN where none.
    u_ceiling : float
        Distance of the table's top from the central potential.

    Returns
    -------
    certificate : dict
        ``valid_gamma``, ``kappa_top``, the projected log10 F and log10 R at
        the ceiling, and whether the linear F and R are representable.
    """
    g = float(gamma)
    kappa = float(kappa_top) if g < 2.0 else float("nan")
    decades = -np.log10(max(float(u_ceiling), 1e-300))
    return {
        "valid_gamma": bool(g < 2.0),
        "kappa_top": kappa,
        "proj_log10F_at_ceiling": float(-kappa * decades)
        if np.isfinite(kappa)
        else float("nan"),
        "linearF_representable": bool(np.isfinite(kappa) and -kappa * decades < 300.0),
        "inner_R_exponent": float(1.0 / (2.0 - g)),
        "proj_log10R_at_ceiling": float(-decades / (2.0 - g)),
        "R_representable": bool(decades / (2.0 - g) < 300.0),
    }
