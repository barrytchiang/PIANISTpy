r"""The gates of a truncation energy.

A truncated model ``F(K + K_t) - F(K_t)`` is physical only where the
untruncated DF is non-negative and non-decreasing above
:math:`\mathcal{K}_{\rm t}`: :math:`\mathcal{K}_{\rm t}` must be at least the
last zero crossing K_0 of the DF and the location Q_min of its last minimum.
The lowest admissible truncation energy is the larger of the two. An
untruncated model (:math:`\mathcal{K}_{\rm t} = 0`) is physical where the DF
has no negative value, or where its negatives are rounding: the exact DF
vanishes at K -> 0 (its analytic tail exponent is negative) and every
negative value is within the ceiling of F a decade above the negative zone.

The arithmetic is the working pipeline's, so that the gates and their
values are the same on the same table.
"""

__all__ = ["TruncationGates", "truncation_gates"]

from dataclasses import dataclass

import numpy as np

from ..typing import Array
from ..validation import CEILING


@dataclass(frozen=True)
class TruncationGates:
    r"""The sign of the untruncated DF and the gates a truncation energy meets.

    Parameters
    ----------
    f_min, k_at_f_min : float
        The DF's most negative value and its energy (its smallest value and
        NaN where it has none negative).
    k_0 : float
        The last zero crossing: the highest energy where F < 0; 0 where none.
    q_min : float
        Where the DF's last decrease (by more than :math:`10^{-5}` relative) ends; 0
        where it has none.
    k_t_min : float
        The lowest admissible truncation energy, max(k_0, q_min), or, where
        the anisotropy's domain edge is above both, the next float above
        k_edge (the potential at the edge; DOMAIN is strict). For a Type II
        model it is necessary, not sufficient: the truncated model's edge
        must also lie inside :math:`R_{\rm a}`, which the solve judges
        (TYPE II EDGE).
    set_by : str
        Which sets it: "K_0", "Q_min" (a tie with K_0 included),
        "domain_edge" (above both), or "none".
    fired : tuple of str
        The gates the truncation energy fails: "DOMAIN"
        (:math:`\mathcal{K}_{\rm t}` at or below k_edge, the untruncated
        :math:`\mathcal{K}_{\rm t} = 0` included), ``"0 < K_t < K_0"``,
        "Z_t = 0 GATE" (an untruncated DF negative beyond rounding) and
        "Z GATE" (``0 < K_t < k_t_min``).
    """

    f_min: float
    k_at_f_min: float
    k_0: float
    q_min: float
    k_t_min: float
    set_by: str
    fired: tuple[str, ...]


def truncation_gates(
    q_sign: Array,
    f_sign: Array,
    q: Array,
    f: Array,
    k_t: float,
    a_analytic: float,
    k_edge: float = 0.0,
) -> TruncationGates:
    r"""Return the sign report and the gates of a truncation energy.

    Parameters
    ----------
    q_sign, f_sign : array
        The inversion's table the sign and the last minimum are read from.
    q, f : array
        The table the untruncated rounding test reads F a decade above the
        negative zone from.
    k_t : float
        The truncation energy; 0 for an untruncated model.
    a_analytic : float
        The analytic exponent of the DF's tail A K^-a as K -> 0.
    k_edge : float, optional
        The potential at the anisotropy's domain edge (:math:`P(R_{\rm a})`
        for a Type II model, 0 otherwise): the DF's support is K > k_edge, and
        the sign and the last minimum are read there only.
    """
    f_s = np.asarray(f_sign, dtype=np.float64)
    q_s = np.asarray(q_sign, dtype=np.float64)
    k_edge = float(k_edge)
    if k_edge > 0.0:
        support = q_s > k_edge
        f_s, q_s = f_s[support], q_s[support]
    fin = np.isfinite(f_s)
    if int(np.sum(f_s[fin] < 0)) > 0:
        ineg = np.flatnonzero(fin & (f_s < 0))
        k_0 = float(q_s[ineg[-1]])
        imin = ineg[np.argmin(f_s[ineg])]
        sign = (float(f_s[imin]), float(q_s[imin]), k_0)
    else:
        sign = (
            float(np.min(f_s[fin])) if fin.any() else float("nan"),
            float("nan"),
            0.0,
        )
    ok = np.isfinite(f_s) & (q_s > 0)
    fv, qv = f_s[ok], q_s[ok]
    dec = np.flatnonzero((fv[:-1] > fv[1:] * (1.0 + 1e-5)) & (fv[:-1] > 0))
    q_min = float(qv[dec[-1] + 1]) if dec.size else 0.0
    k_t_min = max(float(sign[2]), q_min, k_edge)
    if k_t_min <= 0.0:
        set_by = "none"
    elif k_edge > max(float(sign[2]), q_min):
        set_by = "domain_edge"
        # DOMAIN is strict (K_t > k_edge): the smallest K_t it admits is the
        # next float above k_edge
        k_t_min = float(np.nextafter(k_edge, np.inf))
    else:
        set_by = "Q_min" if q_min >= float(sign[2]) else "K_0"

    fired = []
    k_t = float(k_t)
    if k_edge > 0.0 and k_t <= k_edge:
        fired.append("DOMAIN")
    if sign[2] > 0.0 and 0.0 < k_t < sign[2]:
        fired.append("0 < K_t < K_0")
    if sign[2] > 0.0 and k_t == 0.0:
        qn = np.asarray(q, dtype=np.float64)
        fn = np.asarray(f, dtype=np.float64)
        ref_in = 10.0 * float(sign[2]) <= float(qn[-1])
        iref = min(int(np.searchsorted(qn, 10.0 * float(sign[2]))), len(fn) - 1)
        f_ref = float(fn[iref])
        rounding = (
            ref_in
            and np.isfinite(a_analytic)
            and a_analytic < 0.0
            and f_ref > 0.0
            and abs(float(sign[0])) <= CEILING * f_ref
        )
        if not rounding:
            fired.append("Z_t = 0 GATE")
    if 0.0 < k_t < k_t_min * (1.0 - 1e-12):
        fired.append("Z GATE")
    return TruncationGates(
        f_min=float(sign[0]),
        k_at_f_min=float(sign[1]),
        k_0=float(sign[2]),
        q_min=q_min,
        k_t_min=float(k_t_min),
        set_by=set_by,
        fired=tuple(fired),
    )
