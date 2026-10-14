"""The center's form at the top of a table: the singular part and the regular one.

Near the central potential the DF is F = A u^kappa + R(u), u = p_0 - k: the
singular part from the cusp and the regular part from the rest of the well,
R = B + D u + ... Where the inversion's rows end below the ceiling (at the
data top), the form takes over from the last row, the anchor: A is the
center's closed-form amplitude where it states one (else fitted), and B and D
are fitted on the rows from the anchor down to W u_anchor, W doubling from 2
until the amplitude of the three-term fit (A, B, D) and of the four-term fit
(+ E u^2) agree to a tenth of the goal.

The arithmetic is the working pipeline's, operation for operation, so that
its results are equal to the bit on the same backend.
"""

__all__ = ["Anchor", "TopPatch", "anchor", "kappa_loc", "top_patch"]

import math
from dataclasses import dataclass

import numpy as np

from ..typing import Array
from ..validation import GOAL, HARD_LIMIT
from .truncation import K_FLOOR

# the spline stencil width at the data top (k = 5 spline support, k + 2 knots):
# the band the one-term amplitude is fitted over
_STENCIL = 7
# the branches of the center whose DF diverges at p_0: the only ones patched
_PATCHED_BRANCHES = ("regular", "critical_divergent")


@dataclass(frozen=True)
class Anchor:
    """Where the form takes over from a table's rows.

    Parameters
    ----------
    k_anchor : float
        The anchor: the grid's last energy at or below the rule's target.
    rule : str
        "data top" (the rows end below the ceiling, at the pair's data top)
        or "defect repair" (the row below a band of rows not positive at the
        top, where the form holds there).
    rows : int
        The grid's energies above the anchor, which the form replaces.
    slope_mismatch : float
        abs(kappa_loc - kappa) at the anchor; inf where kappa_loc is undefined.
    """

    k_anchor: float
    rule: str
    rows: int
    slope_mismatch: float


def kappa_loc(
    k: Array, f: Array, p_0: float, k_anchor: float, ddf: Array | None = None
) -> float:
    """Return d ln F / d ln u at the anchor, u = p_0 - k, signed.

    With the exact dF/dk on the same rows, -u F'/F at the row at or below the
    anchor nearest p_0; else the least-squares slope over the 7 positive rows
    at and below the anchor nearest p_0 (the spline stencil width). NaN where
    neither is defined.
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
    """Return where the form takes over from an untruncated table's rows.

    The rows end at the pair's data top where it lies below the ceiling
    (min(1, p_0) less the table's floor): the form owns the energies above
    it. Where a band of rows at the top (k above p_0 / 2) is not positive or
    NaN with positive rows above it, the anchor moves below the band, but only
    where the form holds there (abs(kappa_loc - kappa) within the hard limit),
    and not on the critical branch of a center that is not analytic. The
    anchor is snapped to the grid's last energy at or below the target.

    Parameters
    ----------
    k, f : array, shape (n,)
        The grid, increasing, and the inversion's rows on it.
    p_0 : float
        The central potential.
    kappa : float
        The singular exponent of the center.
    center : dict
        The center's form (`center_form`): its branch and form.
    data_top : float
        The pair's highest potential (at its innermost radius).
    ddf : array, optional
        The exact dF/dk on the rows, for kappa_loc.
    below_ceiling : bool, optional
        Whether the rows end below the ceiling. By default the data top
        decides it against min(1, p_0); a truncated table takes its
        untruncated rows' answer, made at p_0 = 1.

    Returns
    -------
    anchor : Anchor or None
        None where no rule fires (the rows reach the ceiling and stand).
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
            # the plateau holds only where the center is analytic
            repair = center["form"] == "analytic"
        else:
            repair = _slope_mismatch(q, df, p_0, kappa, k_below, ddf) <= HARD_LIMIT
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
    """The form that continues a table above its anchor.

    F = a_anchor (u / u_anchor)^kappa + b + d u, u = p_0 - k, as
    `AsymptoticForm.central_df` evaluates it.

    Parameters
    ----------
    k_anchor, u_anchor : float
        The anchor (the table's last energy) and p_0 less it.
    kappa : float
        The singular exponent.
    a_anchor : float
        The singular part at the anchor, A u_anchor^kappa.
    b, d : float
        The regular part; zero where it was not fitted.
    closed_form : bool
        Whether A is the center's closed-form amplitude.
    fit_window, fit_residual, fit_agreement : float
        The fit's window W (in u_anchor), its worst relative residual on its
        rows, and the three- against four-term amplitude agreement; NaN where
        there was no fit.
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
        """A of A u^kappa."""
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
    """Return the form that continues a table above its last row.

    Parameters
    ----------
    k, f : array, shape (n,)
        The inversion's rows, increasing in k, ending at the anchor (the
        data top).
    p_0 : float
        The central potential.
    kappa : float
        The singular exponent of the center.
    amplitude : float or None
        The center's closed-form amplitude A, where it states one for this
        exponent.
    k_above : array
        The grid's energies above the anchor, where the form is used: the
        fitted form is kept only where it is positive on every one.
    terms : int, optional
        3 fits the regular part; 1 keeps the one-term form A u^kappa with A
        joining the anchor.

    Returns
    -------
    patch : TopPatch or None
        None where the DF at the anchor is not positive, or no energy of the
        grid lies above the anchor (no form takes over).
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
                if agree <= 0.1 * float(GOAL):
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
