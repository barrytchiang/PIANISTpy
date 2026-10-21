r"""Checks of what a DF table states about itself, and the validity check.

A table of the distribution function (DF) :math:`F` states more than its
nodes: a column of slopes, a closed form of :math:`F` between the table's
last energy and the central relative potential :math:`P_0` (the stated
center), and forms of :math:`F` below the table's first energy (the
floor). The checks compare each statement with the table's own nodes:

- the stated slope column against the slopes the table's own values of
  :math:`\log_{10}F` imply (`slope_column_excess`);
- the stated center against the table's top decade in the depth
  :math:`u = P_0 - \mathcal{K}`, with :math:`\mathcal{K}` the relative
  energy;
- the forms and nodes below the floor against the table's first node.

`table_checks` makes the checks on a table as written, as a reader of a
written table makes them, and raises ResultError where a check fails: a
slope column whose error exceeds :math:`10^{-2}`, a stated center that
does not describe the table's top decade, or a join below the floor whose
relative mismatch exceeds :math:`10^{-6}`.
`measure_table` makes the checks of `table_checks` on the inversion's own
table and returns every measured value without comparing the value with the
bounds of `table_checks`; the validation report that holds a value assigns
its tier.

`validity_certificate` is the validity check, in closed form: from the inner
density slope and the DF's exponent at the center, whether float64 holds the
table's values of :math:`F` and of the radius at the table's last energy.

References
----------
Notes Section T:2.5.5.
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
from ..validation import TIER_BOUNDS
from .asymptotics import tail_log10_df
from .central import LOG10_DF_CEILING


@dataclass(frozen=True)
class TableChecks:
    r"""The measured results of the checks of what a DF table states about itself.

    Parameters
    ----------
    kappa : float
        Exponent :math:`\kappa` of the depth :math:`u = P_0 - \mathcal{K}`
        that the DF follows above the table: the stated exponent, or, where
        the table states none, the slope of :math:`\log_{10}F` against
        :math:`\log_{10}u` fitted over the table's top decade in :math:`u`;
        NaN where the table states no exponent and `measure_table` finds
        fewer than 5 nodes in the top decade.
    kappa_residual : float
        Departure of the table's top decade from the stated center (see
        `_center`); zero where the table states no center, and
        infinite where `measure_table` cannot measure the departure.
    k_top : float
        Last (highest) energy of the table, in internal units.
    slope_column_excess : float
        Largest relative error in :math:`F` that the stated slope column
        causes beyond the uncertainty of the slope the table's own values of
        :math:`\log_{10}F` imply (`slope_column_excess`); zero where the
        slope column is not used.
    edge_rel_at_floor : float
        Relative mismatch :math:`F_{\rm form}/F_{\rm node} - 1` of the
        Taylor form or the edge law below the floor against the lowest node
        of the table joined to its nodes below the floor; NaN where the
        table states neither the Taylor form nor the edge law.
    below_join : float
        Relative mismatch :math:`F_{\rm below}/F_{\rm node} - 1` of the last
        node below the floor against the table's first node; NaN where the
        table has no nodes below the floor, or where the last node below the
        floor does not lie on the table's first node. Default: NaN.
    tail_join : float
        Relative mismatch :math:`F_{\rm tail}/F_{\rm node} - 1` of the tail
        against the lowest node, the node that the tail continues; NaN where
        the DF has no tail or the tail's amplitude :math:`A` is not above 0.
        Default: NaN.
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
    r"""Return the relative error in F that a stated slope column causes, per node.

    At each node, the table's own values of :math:`\log_{10}F` give two
    slopes: the derivative of the polynomial through the 3 nearest nodes
    (the 3-node derivative) and the derivative of the polynomial through the
    5 nearest nodes (the 5-node derivative). The uncertainty of the 5-node
    derivative is the largest difference between the 3-node and the 5-node
    derivatives over the node and its two neighbors, plus the rounding error
    of the 5-node derivative. The excess :math:`\delta m` is the amount by
    which the stated slope's difference from the 5-node derivative exceeds
    the uncertainty of the 5-node derivative, and zero where the difference
    does not exceed the uncertainty. The excess :math:`\delta m` changes the
    cubic Hermite interpolant of :math:`\log_{10}F` between the node and
    either neighbor by at most :math:`(4/27)\,\delta m\,h`, with
    :math:`h` the longer of the two intervals adjacent to the node. The
    factor :math:`\ln 10` converts the change of :math:`\log_{10}F` into a
    relative error in :math:`F`.

    Where the spacing of the nodes changes at the node ``split`` (the first
    node of the table above the nodes below its floor), a node whose
    centered stencil would cross the change of spacing takes a one-sided
    stencil on its own side of ``split``, and :math:`h` is the node's
    interval on its own side of ``split``. The one-sided derivative uses
    only nodes of one spacing, and no node is moved.

    Parameters
    ----------
    x : array, shape (n,)
        Natural logarithm :math:`\ln\mathcal{K}` of the energies,
        increasing.
    log_df : array, shape (n,)
        Base-10 logarithm of the DF at the energies.
    slope : array, shape (n,)
        The stated slope :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K}`.
    split : int, optional
        Index of the first node above a change of the nodes' spacing.
        Default: ``None``, which means that the spacing has no change to
        respect.

    Returns
    -------
    excess : array, shape (n,)
        Upper bound on the relative error in :math:`F` that the stated slope
        causes beyond the uncertainty of the 5-node derivative, at each
        node. All zeros where `x` has fewer than 5 nodes.
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
    spread = np.maximum(
        np.maximum(spread, np.r_[spread[1:], spread[-1]]), np.r_[spread[0], spread[:-1]]
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
    """Return the next singular term of the stated center as a function of u.

    The next singular term is ``c u**e`` for the form ``"power"`` and
    ``c u**e (-ln u)`` for the form ``"log_power"``, with u the depth below
    the central potential. Returns None where the table states no next
    singular term or states the center's form ``"analytic"``.

    Raises
    ------
    ResultError
        If the table states a next singular term without a finite exponent
        and a finite coefficient.
    """
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
    """Return the DF's exponent above the table and the residual of the stated center.

    The top decade is the set of nodes whose depth u = p_0 - k is at most
    10 times the depth of the table's last node. Three cases follow from
    what the table states in ``top``:

    - A stated form (``top_form``): the residual is the change of F less the
      form's singular part and less the next singular term, from the depth
      10 (p_0 - k[-1]) to the last node, relative to F at the last node.
      Where the top decade extends below the table's first energy, the
      residual is zero.
    - A stated exponent (``top_kappa``) and no form: the residual is zero
      where the exponent fitted over the top decade lies within
      :math:`10^{-6}` (`TIER_BOUNDS[0]`) of the stated exponent. Otherwise
      the fitted exponent less the stated exponent (the gap) is measured
      over up to three successive decades in u, from the last node down.
      Where all three gaps have one sign and shrink toward the center, the
      residual is |g_0| / log10(|g_1| / |g_0|), with g_0 the gap of the top
      decade and g_1 the gap of the next decade. Where the gaps do not
      shrink toward the center but one gap lies within :math:`10^{-6}` of
      zero, the residual is ln(10) times |g_0|.
    - No form and no finite stated exponent: the exponent is the exponent
      fitted over the top decade, and the residual is zero.

    With ``refuse`` True, the function raises ResultError where the top
    decade holds fewer than 5 nodes, where the residual of a stated form
    exceeds :math:`10^{-2}` (`TIER_BOUNDS[2]`), or where the gaps of a
    stated exponent neither shrink toward the center nor come within
    :math:`10^{-6}` of zero. With ``refuse`` False, the function raises none
    of these three errors: a top decade with fewer than 5 nodes returns an
    infinite residual, gaps that neither shrink toward the center nor come
    within :math:`10^{-6}` of zero return an infinite residual, and the
    residual of a stated form is returned as measured. A stated form
    without its finite coefficients raises ResultError for either value of
    ``refuse``.
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
        if refuse and not residual <= TIER_BOUNDS[2]:
            raise ResultError(
                f"The stated center ({form!r}) does not describe the table: "
                "across the top decade the table less "
                f"its singular part changes by {check:+.3e} of F_top."
            )
        return kappa, residual
    if kappa_attr is not None and np.isfinite(float(kappa_attr)):
        kappa = float(kappa_attr)
        if abs(kappa_fit - kappa) <= TIER_BOUNDS[0]:
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
        reached = len(gaps) >= 1 and any(abs(g) <= TIER_BOUNDS[0] for g in gaps)
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
    r"""Return the checks of a written DF table, raising where a check fails.

    The checks are those a reader of a written table makes before using the
    table. The slope column is used only where `ddf_dk` is given, is finite
    on every node, and is not zero on the last node, and, where `below` is
    given, only where `below` holds ``dFdK``.

    Parameters
    ----------
    k : array, shape (n,)
        Relative energies of the table's nodes, as written, in internal
        units; the nodes are sorted by energy before the checks.
    log_df : array, shape (n,)
        Base-10 logarithm of the DF at the energies `k`, as written. Leading
        nodes with :math:`F = 0` (minus infinity) lie outside the support
        and are dropped.
    ddf_dk : array, shape (n,) or None
        The written column :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}` at the
        energies `k`; None where the table has no such column.
    p_0 : float
        Central relative potential of the table, ``1 - k_t`` with ``k_t``
        the truncation energy, in internal units.
    top : mapping
        The stated center: ``top_kappa``, ``top_form``, the form's exponent
        ``top_singular_exponent``, amplitude ``top_amplitude`` or
        logarithm's coefficient ``top_log_coefficient``, and the next
        singular term ``top_next_form``, ``top_next_exponent`` and
        ``top_next_coefficient``. Absent keys mean no statement.
    below : mapping, optional
        The exact nodes below the written floor: energies ``K``, base-10
        logarithm of the DF ``logF`` and ``dFdK``; the last node lies at the
        written floor. Default: ``None``, which means no nodes below the
        floor.
    tail : mapping, optional
        The tail below the floor: amplitude ``A``, exponent ``a``,
        logarithmic exponent ``k`` and scale ``E0``. Default: ``None``,
        which means no tail.
    edge_taylor : tuple of float, optional
        :math:`(F_1, F_2, \mathcal{K}_{\rm switch})` of a truncated table's
        Taylor form below the floor. Default: ``None``, which means no
        Taylor form.
    edge_law : tuple of float, optional
        :math:`(p, C_0, C_1)` of a Method II child table's edge law below
        the floor. Default: ``None``, which means no edge law.

    Returns
    -------
    checks : TableChecks
        The measured result of each check.

    Raises
    ------
    ResultError
        If the table's :math:`\log_{10}F` is not finite inside its support.
    ResultError
        If the nodes below the floor are not finite, do not end at the
        written floor, or differ from the table's first node by a relative
        mismatch above :math:`10^{-6}`.
    ResultError
        If the slope column's error exceeds :math:`10^{-2}` at a node, or
        the stated center fails its check (see `_center`).
    ResultError
        If the tail, the Taylor form or the edge law differs from the
        table's first node by a relative mismatch above :math:`10^{-6}`.
    ResultError
        If the table's first energy lies above the Taylor form's switch
        energy, or the table states both a Taylor form and an edge law.
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
        if not abs(join - 1.0) <= TIER_BOUNDS[0]:
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
    edge_first: bool = False,
) -> TableChecks:
    r"""Return the checks of the inversion's own DF table, without bounds.

    `measure_table` makes the checks of `table_checks` on a table and its
    slopes as the inversion states them. A measured value above a bound of
    `table_checks` is returned, not raised; the validation report that holds
    the value assigns its tier.

    Parameters
    ----------
    k : array, shape (n,)
        Relative energies of the table's nodes, increasing, in internal
        units.
    log_df : array, shape (n,)
        Base-10 logarithm of the DF at the energies `k`, finite on every
        node.
    log_df_slope : array, shape (n,) or None
        The exact slope :math:`\mathrm{d}\log_{10}F/\mathrm{d}\ln\mathcal{K}`
        on every node; None where the table states no exact slope.
    p_0 : float
        Central relative potential of the table, in internal units.
    top : mapping
        The stated center, under the key names of `table_checks`.
    below : tuple of array, optional
        The nodes below the table: their energies, their base-10 logarithm
        of the DF, and their slope (None where the slope is not exact). The
        last node below the table may lie on the table's first node.
        Default: ``None``, which means no nodes below the table.
    tail : mapping, optional
        The tail below the floor, under the key names of `table_checks`.
        Default: ``None``, which means no tail.
    edge_taylor : tuple of float, optional
        The Taylor form below the floor, as for `table_checks`. Default:
        ``None``, which means no Taylor form.
    edge_law : tuple of float, optional
        The edge law below the floor, as for `table_checks`. Default:
        ``None``, which means no edge law.
    edge_first : bool, optional
        True where the support's first node lies on an edge whose known law
        a polynomial stencil cannot represent, as for a Type II DF at the
        energy :math:`P(R_{\rm a})`. Where `edge_first` is true, the slope
        column is not judged at the support's first node. Default:
        ``False``.

    Returns
    -------
    checks : TableChecks
        The measured result of each check. A residual of the stated center
        that cannot be measured is infinite.

    Raises
    ------
    ResultError
        If the stated center or its next singular term lacks its finite
        coefficients, the table's first energy lies above the Taylor form's
        switch energy, or the table states both a Taylor form and an edge
        law.
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
        k,
        log_df,
        slope,
        p_0,
        top,
        tail,
        edge_taylor,
        edge_law,
        False,
        split,
        edge_first,
    )
    return dataclasses.replace(checks, below_join=join)


def _measure(
    k,
    log_df,
    slope,
    p_0,
    top,
    tail,
    edge_taylor,
    edge_law,
    refuse,
    split=None,
    edge_first=False,
):
    """Return the checks of a table already joined to its nodes below the floor.

    With ``refuse`` True, a check that fails raises ResultError, as in
    `table_checks`; with ``refuse`` False, each measured value is returned.
    With ``edge_first`` True, the slope column is not judged at the node of
    index ``split`` (the table's first node above the nodes below the
    floor), or at index 0 where ``split`` is None.
    The table is interpolated in ln k by a cubic Hermite through the stated
    slopes where ``slope`` is given, and by a not-a-knot cubic spline
    otherwise.
    """
    x = np.log(k)
    excess = 0.0
    if slope is not None:
        per_node = slope_column_excess(x, log_df, slope, split)
        if edge_first:
            per_node = np.delete(per_node, split or 0)
        excess = float(np.max(per_node)) if per_node.size else 0.0
        if refuse and np.any(per_node > TIER_BOUNDS[2]):
            raise ResultError(
                "The dF/dK column disagrees with the table's own F beyond 0.01."
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
        if refuse and not abs(ratio - 1.0) <= TIER_BOUNDS[0]:
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
        if refuse and not abs(edge_rel) <= TIER_BOUNDS[0]:
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
        if refuse and not abs(edge_rel) <= TIER_BOUNDS[0]:
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
    gamma: float, kappa_top: float, u_ceiling: float, log10_amplitude: float = 0.0
) -> dict[str, Any]:
    r"""Return whether float64 holds the table's columns, from closed forms.

    Near the center the DF goes as :math:`F = A u^\kappa` in the depth
    :math:`u = P_0 - \mathcal{K}`, and the radius goes as
    :math:`r \propto u^{1/(2-\gamma)}` for an inner density slope
    :math:`\gamma < 2`. At the table's last energy, with
    :math:`d = -\log_{10} u_{\rm ceiling}` decades of depth, the projected
    values are :math:`\log_{10}F = \log_{10}A - \kappa d` and
    :math:`\log_{10}r = -d/(2-\gamma)`.

    Parameters
    ----------
    gamma : float
        Inner slope :math:`\gamma` of the density.
    kappa_top : float
        Exponent :math:`\kappa` of the depth :math:`u` that the DF follows
        at the center; NaN where the DF has no such exponent.
    u_ceiling : float
        Depth :math:`u_{\rm ceiling}` of the table's last energy below the
        central relative potential, in internal units; held at or above
        :math:`10^{-300}`.
    log10_amplitude : float, optional
        Base-10 logarithm of the amplitude :math:`A` of the center's closed
        form :math:`A u^\kappa`. Default: ``0.0``, which is the value for a
        form that has no amplitude.

    Returns
    -------
    validity : dict
        ``valid_gamma``: True where :math:`\gamma < 2`. ``kappa_top``:
        :math:`\kappa`, NaN where :math:`\gamma \ge 2`.
        ``proj_log10F_at_ceiling``: the projected :math:`\log_{10}F`, NaN
        where :math:`\kappa` is NaN. ``linearF_representable``: True where
        the projected :math:`\log_{10}F` lies below
        `central.LOG10_DF_CEILING`, the base-10 logarithm (300) of the
        largest DF value a table holds.
        ``inner_R_exponent``: :math:`1/(2-\gamma)`.
        ``proj_log10R_at_ceiling``: the projected :math:`\log_{10}r`.
        ``R_representable``: True where :math:`d/(2-\gamma)` lies below
        300.
    """
    g = float(gamma)
    kappa = float(kappa_top) if g < 2.0 else float("nan")
    decades = -np.log10(max(float(u_ceiling), 1e-300))
    return {
        "valid_gamma": bool(g < 2.0),
        "kappa_top": kappa,
        "proj_log10F_at_ceiling": float(log10_amplitude - kappa * decades)
        if np.isfinite(kappa)
        else float("nan"),
        "linearF_representable": bool(
            np.isfinite(kappa) and log10_amplitude - kappa * decades < LOG10_DF_CEILING
        ),
        "inner_R_exponent": float(1.0 / (2.0 - g)),
        "proj_log10R_at_ceiling": float(-decades / (2.0 - g)),
        "R_representable": bool(decades / (2.0 - g) < 300.0),
    }
