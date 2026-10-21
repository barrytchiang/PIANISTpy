r"""The form of the DF above the top of an untruncated table: anchor and top patch.

The inversion calls `anchor` for the energy at which a closed form takes
over from the table's rows, and `top_patch` for the closed form. Near the
central potential p_0, the DF is F = A u^kappa + R(u) in the depth
u = p_0 - k: the singular part A u^kappa comes from the cusp, and the
regular part R = B + D u + ... comes from the rest of the potential well.
Where the inversion's rows end below min(1, p_0) - K_FLOOR, the form takes
over from the last row, the anchor. The amplitude A is the center's
closed-form amplitude where `center_form` states one, and is fitted
otherwise. B and D are fitted on the rows with depths from u_anchor to
W u_anchor, where the window W doubles from 2 until the amplitude A_3 of the
three-term fit (A, B, D) and the amplitude A_4 of the four-term fit (A, B, D,
E u^2) satisfy abs(A_3 / A_4 - 1) at most :math:`10^{-7}`, one tenth of
:math:`10^{-6}`, or until W u_anchor exceeds the largest depth of the rows.

References
----------
Notes Section T:2.5.2.
"""

__all__ = ["Anchor", "TopPatch", "anchor", "kappa_loc", "top_patch"]

import math
from dataclasses import dataclass

import numpy as np

from ..typing import Array
from ..validation import TIER_BOUNDS
from .truncation import K_FLOOR

# The number of rows nearest p_0 over which the one-term amplitude and the
# local slope are fitted: the stencil width of the table's spline at the top
# row, k + 2 = 7 knots for a spline of degree k = 5.
_STENCIL = 7
# The central branches whose DF diverges at p_0. Only the listed branches get an
# anchor under the rule "data top" (`anchor`).
_PATCHED_BRANCHES = ("regular", "critical_divergent")


@dataclass(frozen=True)
class Anchor:
    """The energy at which the central form takes over from a table's rows.

    Parameters
    ----------
    k_anchor : float
        The anchor energy: the grid's last energy at or below the target
        energy of the rule, in internal units.
    rule : str
        The rule that placed the anchor. "data top": the rows end at the pair's
        data top, below min(1, p_0) - K_FLOOR. "defect repair": the
        anchor lies below a band of top rows whose DF is not positive, and
        the central form holds at the anchor.
    rows : int
        The number of the grid's energies above the anchor, where the
        central form replaces the rows.
    slope_mismatch : float
        abs(kappa_loc - kappa) at the anchor, with kappa_loc from
        `kappa_loc`; inf where kappa_loc or kappa is not finite.
    """

    k_anchor: float
    rule: str
    rows: int
    slope_mismatch: float


def kappa_loc(
    k: Array, f: Array, p_0: float, k_anchor: float, ddf: Array | None = None
) -> float:
    """Return the local slope d ln F / d ln u at the anchor, with u = p_0 - k.

    Where ``ddf``, the exact dF/dk on the rows, is given with one value per
    row, the slope is -u (dF/dk) / F at the row nearest p_0 among the rows
    at or below ``k_anchor`` with a positive F and a finite dF/dk. Otherwise
    the slope is the least-squares slope of ln F against ln u over the
    `_STENCIL` (7) rows nearest p_0 among the rows at or below ``k_anchor``
    with a positive F. The result is NaN where fewer than 5 such rows exist
    or where the rows span no range of ln u.
    """
    q = np.asarray(k, dtype=np.float64)
    u = float(p_0) - q
    f = np.asarray(f, dtype=np.float64)
    if ddf is not None and len(ddf) == f.size:
        dd = np.asarray(ddf, dtype=np.float64)
        okx = np.isfinite(f) & (f > 0.0) & (u > 0.0) & np.isfinite(dd) & (q <= k_anchor)
        if okx.any():
            ix = np.flatnonzero(okx)[np.argmin(u[okx])]
            return float(-u[ix] * dd[ix] / f[ix])
    ok = np.isfinite(f) & (f > 0.0) & (u > 0.0) & (q <= k_anchor)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(u[idx])][:_STENCIL]
    if idx.size < 5:
        return math.nan
    lu, lf = np.log(u[idx]), np.log(f[idx])
    if float(np.ptp(lu)) <= 0.0:
        return math.nan
    return float(np.polyfit(lu, lf, 1)[0])


def _slope_mismatch(
    k: Array, f: Array, p_0: float, kappa: float, k_anchor: float, ddf: Array | None
) -> float:
    """Return abs(kappa_loc - kappa) at ``k_anchor``.

    The result is inf where kappa_loc or kappa is not finite.
    """
    kl = kappa_loc(k, f, p_0, k_anchor, ddf)
    if not (np.isfinite(kl) and np.isfinite(kappa)):
        return math.inf
    return float(abs(kl - float(kappa)))


def anchor(
    k: Array,
    f: Array,
    p_0: float,
    kappa: float,
    center: dict,
    data_top: float,
    ddf: Array | None = None,
    below_ceiling: bool | None = None,
) -> Anchor | None:
    """Return the anchor where the central form takes over from a table's rows.

    The table is untruncated.

    Two rules can place the anchor.

    - "data top": the central branch is in `_PATCHED_BRANCHES` and the
      rows end below min(1, p_0) - K_FLOOR.
    - "defect repair": kappa is finite, and a band of rows with k above
      p_0 / 2 holds a DF that is NaN or not positive, with a positive row
      above the band. The rule applies only where the central form holds
      below the band: for kappa equal to 0, only where the central form is
      "analytic"; for other kappa, only where abs(kappa_loc - kappa) at the
      largest double below the band's lowest energy is at most
      :math:`10^{-2}`.

    Where the rows end below min(1, p_0) - K_FLOOR, the target energy is
    p_0 less the larger of p_0 - data_top and K_FLOOR; otherwise the target
    energy is the largest double below the band's lowest energy. Under the
    "defect repair" rule, the target energy is the lower of the two target
    energies. The anchor is the highest grid energy at or below the target
    energy, raised to the grid's first energy where the target energy lies
    below the grid's first energy, and lowered to the grid's second-to-last
    energy where the highest grid energy at or below the target energy is
    the grid's last energy.

    Parameters
    ----------
    k : array, shape (n,)
        The energy grid, increasing, in internal units.
    f : array, shape (n,)
        The inversion's DF on the grid.
    p_0 : float
        The central potential, in internal units.
    kappa : float
        The singular exponent of the center.
    center : dict
        The central form (`center_form`), read for its ``branch`` and
        ``form``.
    data_top : float
        The pair's highest potential, at the pair's innermost radius.
    ddf : array or None, optional
        The exact dF/dk on the grid, for `kappa_loc`. Default: ``None``,
        which means `kappa_loc` fits the local slope to the rows.
    below_ceiling : bool or None, optional
        Whether the rows end below min(1, p_0) - K_FLOOR. Default: ``None``,
        which means ``data_top`` is compared with min(1, p_0) - K_FLOOR. A
        truncated table passes the value found for its untruncated rows at
        p_0 = 1.

    Returns
    -------
    anchor : Anchor or None
        The anchor, or None where neither rule applies, in which case the
        rows cover the grid up to the grid's last energy.
    """
    q = np.asarray(k, dtype=np.float64)
    df = np.asarray(f, dtype=np.float64)
    p_0 = float(p_0)
    kappa = float(kappa)
    patched = center["branch"] in _PATCHED_BRANCHES
    if below_ceiling is None:
        below_ceiling = not data_top >= min(1.0, p_0) - K_FLOOR
    bad = (np.isnan(df) | (df <= 0.0)) & (q > 0.5 * p_0)
    pos_at_or_above = np.maximum.accumulate((df > 0.0)[::-1])[::-1]
    pos_above = np.concatenate([pos_at_or_above[1:], [False]])
    defect = bad & pos_above
    repair = bool(np.any(defect)) and np.isfinite(kappa)
    k_below = math.nan
    if repair:
        k_bad = float(q[defect].min())
        k_below = float(np.nextafter(k_bad, 0.0))
        if kappa == 0.0:
            # With kappa = 0 the central form is a constant, and a constant
            # continues the DF only where the center is analytic.
            repair = center["form"] == "analytic"
        else:
            repair = _slope_mismatch(q, df, p_0, kappa, k_below, ddf) <= TIER_BOUNDS[2]
    if not ((patched and below_ceiling) or repair):
        return None
    gap = max(p_0 - float(data_top), K_FLOOR) if below_ceiling else p_0 - k_below
    target = p_0 - gap
    if repair and k_below < target:
        target = k_below
    idx = int(np.searchsorted(q, target, side="right")) - 1
    idx = max(0, min(idx, len(q) - 2))
    k_anchor = float(q[idx])
    return Anchor(
        k_anchor=k_anchor,
        rule="defect repair" if repair else "data top",
        rows=int(np.sum(q > k_anchor)),
        slope_mismatch=_slope_mismatch(q, df, p_0, kappa, k_anchor, ddf),
    )


@dataclass(frozen=True)
class TopPatch:
    """The central form that continues a table above its anchor.

    The form is F = a_anchor (u / u_anchor)^kappa + b + d u in the depth
    u = p_0 - k, as `AsymptoticForm.central_df` evaluates the form.

    Parameters
    ----------
    k_anchor : float
        The anchor energy, the table's last energy, in internal units.
    u_anchor : float
        The anchor depth p_0 - k_anchor, in internal units.
    kappa : float
        The singular exponent.
    a_anchor : float
        The singular part at the anchor, A u_anchor^kappa.
    b : float
        The constant term of the regular part.
    d : float
        The linear coefficient of the regular part; zero where no regular
        part was fitted.
    closed_form : bool
        Whether A is the center's closed-form amplitude.
    fit_window : float
        The fit's window W, in units of u_anchor; NaN where no fit was kept.
    fit_residual : float
        The largest relative residual of the fit on the fit's rows; NaN
        where no fit was kept.
    fit_agreement : float
        The relative difference between the three-term and the four-term
        amplitude, abs(A_3 / A_4 - 1); NaN where no fit was kept.
    """

    k_anchor: float
    u_anchor: float
    kappa: float
    a_anchor: float
    b: float
    d: float
    closed_form: bool
    fit_window: float
    fit_residual: float
    fit_agreement: float

    @property
    def amplitude(self) -> float:
        """The amplitude A of the singular part A u^kappa."""
        return self.a_anchor / self.u_anchor**self.kappa


def top_patch(
    k: Array,
    f: Array,
    p_0: float,
    kappa: float,
    amplitude: float | None,
    k_above: Array,
    terms: int = 3,
) -> TopPatch | None:
    """Return the central form that continues a table above its last row.

    Where the DF at the anchor is positive, the value at the anchor is first
    smoothed: among the rows with a positive DF, the `_STENCIL` (7) rows
    nearest p_0 are selected, and where at least 5 rows are selected, the
    anchor value is the one-term form A u_anchor^kappa, with ln A the mean
    of ln F - kappa ln u over the selected rows.

    With ``terms`` other than 1 and kappa not 0, the regular part B + D u is
    fitted together with the singular amplitude on the rows with depths from
    u_anchor to W u_anchor, with W = 2, 4, 8, ... The rows of the fit are
    the rows with a finite, positive DF. The window W stops widening once
    the three-term amplitude A_3 and the four-term amplitude A_4 satisfy
    abs(A_3 / A_4 - 1) at most :math:`10^{-7}`, one tenth of
    :math:`10^{-6}`, or once W u_anchor exceeds the largest depth of the
    fit's rows. The last window with at least 8 rows is kept. The fit is
    used only where the fitted amplitude is positive and the fitted form is
    finite and positive at every energy of ``k_above``.

    Where ``amplitude`` is a finite closed-form amplitude, A equals
    ``amplitude``. B and D are then refitted with A fixed where a fit was
    kept; where no fit was kept, B joins the form to the anchor value and
    D = 0.

    Parameters
    ----------
    k : array, shape (n,)
        The energies of the inversion's rows, increasing, ending at the
        anchor, in internal units.
    f : array, shape (n,)
        The inversion's DF on the rows.
    p_0 : float
        The central potential, in internal units.
    kappa : float
        The singular exponent of the center.
    amplitude : float or None
        The center's closed-form amplitude A for the exponent ``kappa``, or
        None where the center states no closed-form amplitude.
    k_above : array
        The grid's energies above the anchor, where the form is used.
    terms : int, optional
        The number of terms of the form. Default: ``3``, which means the
        regular part B + D u is fitted. A value of 1 keeps the one-term form
        A u^kappa joined to the anchor value.

    Returns
    -------
    patch : TopPatch or None
        The central form, or None where the anchor value is not positive or
        no energy of the grid lies above the anchor.
    """
    q = np.asarray(k, dtype=np.float64)
    df = np.asarray(f, dtype=np.float64)
    q_anchor = float(q[-1])
    u_all = p_0 - q
    u_anc = p_0 - q_anchor
    f_anchor = float(df[-1])
    if f_anchor > 0.0:
        ok_amp = (u_all >= u_anc) & (df > 0.0)
        idx_amp = np.where(ok_amp)[0]
        idx_amp = idx_amp[np.argsort(u_all[idx_amp])][:_STENCIL]
        bmask = np.zeros_like(ok_amp)
        bmask[idx_amp] = True
        if int(np.sum(bmask)) >= 5:
            ln_c = float(np.mean(np.log(df[bmask]) - kappa * np.log(u_all[bmask])))
            f_anchor = float(np.exp(ln_c + kappa * np.log(u_anc)))
    u_above = p_0 - np.asarray(k_above, dtype=np.float64)
    if f_anchor <= 0.0 or u_above.size == 0:
        return None
    ratio = np.maximum(u_above / max(u_anc, 1e-300), 1e-300)
    a_term, b_term, d_term = f_anchor, 0.0, 0.0
    two_terms = False
    window, residual, agreement = math.nan, math.nan, math.nan
    uq, fq = u_all, df
    ok_fit = np.isfinite(fq) & (fq > 0.0) & (uq >= u_anc)
    if terms != 1 and float(kappa) != 0.0:
        prev = None
        w = 2.0
        while w * u_anc <= float(np.max(uq[ok_fit])) if ok_fit.any() else False:
            m = ok_fit & (uq <= w * u_anc)
            if int(m.sum()) >= 8:
                x = uq[m] / u_anc
                cols = [x ** float(kappa), np.ones_like(x), x, x * x]
                xr = np.vstack(cols).T / fq[m][:, None]
                c3 = np.linalg.lstsq(xr[:, :3], np.ones(x.size), rcond=None)[0]
                c4 = np.linalg.lstsq(xr, np.ones(x.size), rcond=None)[0]
                agree = abs(c3[0] / c4[0] - 1.0) if c4[0] != 0.0 else float("inf")
                res3 = float(np.max(np.abs(xr[:, :3] @ c3 - 1.0)))
                prev = (w, c3, agree, res3)
                if agree <= 0.1 * float(TIER_BOUNDS[0]):
                    break
            w *= 2.0
        if prev is not None:
            w, c3, agree, res3 = prev
            a_try, b_try, d_try = float(c3[0]), float(c3[1]), float(c3[2]) / u_anc
            chk = a_try * (ratio**kappa) + b_try + d_try * u_above
            if a_try > 0.0 and np.all(np.isfinite(chk)) and np.all(chk > 0.0):
                a_term, b_term, d_term, two_terms = a_try, b_try, d_try, True
                window, residual, agreement = w, res3, agree
    closed = amplitude is not None and np.isfinite(float(amplitude))
    if closed:
        assert amplitude is not None
        ua_k = u_anc ** float(kappa)
        a_term = float(amplitude) * ua_k
        if two_terms:
            mb = ok_fit & (uq <= float(window) * u_anc)
            xb = uq[mb] / u_anc
            xbm = np.vstack([np.ones_like(xb), xb]).T / fq[mb][:, None]
            yb = 1.0 - a_term * xb ** float(kappa) / fq[mb]
            cb = np.linalg.lstsq(xbm, yb, rcond=None)[0]
            b_term, d_term = float(cb[0]), float(cb[1]) / u_anc
            residual = float(
                np.max(
                    np.abs(
                        (a_term * xb ** float(kappa) + b_term + d_term * uq[mb])
                        / fq[mb]
                        - 1.0
                    )
                )
            )
        else:
            b_term, d_term = float(f_anchor - a_term), 0.0
    return TopPatch(
        k_anchor=q_anchor,
        u_anchor=float(u_anc),
        kappa=float(kappa),
        a_anchor=float(a_term),
        b=float(b_term),
        d=float(d_term),
        closed_form=bool(closed),
        fit_window=float(window),
        fit_residual=float(residual),
        fit_agreement=float(agreement),
    )
