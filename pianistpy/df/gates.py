r"""Sign checks of an untruncated DF and the gates of a truncation energy.

`truncation_gates` reads the table of an untruncated distribution function
(DF) :math:`F(K)` of the energy :math:`K` and returns a `TruncationGates`
record. The record holds
the DF's most negative value, the DF's last zero crossing :math:`K_0`, the
energy :math:`Q_{\rm min}` where the DF's last decrease ends, the lowest
admissible truncation energy, and the names of the gates that a given
truncation energy :math:`\mathcal{K}_{\rm t}` fails.

The truncated DF is :math:`F(K + \mathcal{K}_{\rm t}) - F(\mathcal{K}_{\rm t})`.
The gates admit a truncation energy :math:`\mathcal{K}_{\rm t} > 0` only at
or above both :math:`K_0` and :math:`Q_{\rm min}`. Above :math:`K_0` the
tabulated DF has no negative value. From :math:`Q_{\rm min}` upward the
tabulated DF has no step between successive table energies that starts at
a positive value :math:`F_i` and ends at a value :math:`F_{i+1}` with
:math:`F_i > (1 + 10^{-5})\,F_{i+1}`.
For a Type II model, the gates also require :math:`\mathcal{K}_{\rm t}` to
exceed the potential at the anisotropy radius :math:`R_{\rm a}`.

An untruncated model (:math:`\mathcal{K}_{\rm t} = 0`) passes the gates when
the DF has no negative value, or when the DF's negative values are rounding.
The negative values count as rounding when three conditions hold: the
analytic exponent :math:`a` of the DF's tail :math:`F \sim A K^{-a}` as
:math:`K \to 0`, with :math:`A` a constant, is negative, so that the exact DF
vanishes as :math:`K \to 0`; the energy :math:`10 K_0` is at or below the
table's highest energy; and the DF's most negative value is, in absolute value, at
most :math:`10^{-4}` times the DF at the first table energy at or above
:math:`10 K_0`, with that DF positive.

The gates are derived in Notes Section T:3.5.
"""

__all__ = ["TruncationGates", "truncation_gates"]

from dataclasses import dataclass

import numpy as np

from ..typing import Array
from ..validation import TIER_BOUNDS


@dataclass(frozen=True)
class TruncationGates:
    r"""Sign record of an untruncated DF and the gates a truncation energy fails.

    All energies are in internal units. The DF's table is read only at
    energies above ``k_edge`` (see `truncation_gates`).

    Parameters
    ----------
    f_min : float
        The DF's most negative tabulated value. Where the DF has no negative
        value, the DF's smallest finite value, or NaN where the table has no
        finite value.
    k_at_f_min : float
        The energy of the DF's most negative value; NaN where the DF has no
        negative value.
    k_0 : float
        The DF's last zero crossing :math:`K_0`: the highest tabulated energy
        where the DF is negative; 0 where the DF has no negative value.
    q_min : float
        The energy :math:`Q_{\rm min}` where the DF's last decrease ends. A
        decrease is a step between successive positive table energies whose
        DF values :math:`F_i` and :math:`F_{i+1}` are finite, with
        :math:`F_i` positive and :math:`F_i > (1 + 10^{-5})\,F_{i+1}`.
        :math:`Q_{\rm min}` is the energy at the end of the last decrease; 0
        where the DF has no decrease.
    k_t_min : float
        The lowest admissible truncation energy: the largest of `k_0` and
        `q_min`, or, where the potential ``k_edge`` at the anisotropy radius
        exceeds both, the next double above ``k_edge`` (the "DOMAIN" gate
        requires :math:`\mathcal{K}_{\rm t}` strictly above ``k_edge``). For
        a Type II model, a truncation energy at or above `k_t_min` is not
        sufficient by itself: the truncated model's edge must also lie
        inside :math:`R_{\rm a}`, which the Poisson solve checks.
    set_by : {"K_0", "Q_min", "domain_edge", "none"}
        The quantity that sets `k_t_min`: "K_0" where `k_0` exceeds `q_min`;
        "Q_min" where `q_min` is at or above `k_0`; "domain_edge" where
        ``k_edge`` exceeds both `k_0` and `q_min`; "none" where `k_t_min` is 0.
    fired : tuple of str
        The names of the gates that the truncation energy
        :math:`\mathcal{K}_{\rm t}` fails, empty where the truncation energy
        fails no gate.
        "DOMAIN": ``k_edge > 0`` and :math:`\mathcal{K}_{\rm t}` is at or
        below ``k_edge`` (which includes :math:`\mathcal{K}_{\rm t} = 0`).
        ``"0 < K_t < K_0"``: :math:`\mathcal{K}_{\rm t}` is positive and
        below `k_0`. "Z_t = 0 GATE": :math:`\mathcal{K}_{\rm t} = 0` and the
        DF has negative values that do not count as rounding. "Z GATE":
        :math:`\mathcal{K}_{\rm t}` is positive and below `k_t_min` by more
        than a relative :math:`10^{-12}`.
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
    r"""Return the DF's sign record and the gates of a truncation energy.

    Parameters
    ----------
    q_sign : array, shape (m,)
        Energies of the inversion's table, increasing, in internal units.
        `k_0`, `q_min` and the DF's most negative value are read on this
        table.
    f_sign : array, shape (m,)
        The untruncated DF at `q_sign`.
    q : array, shape (k,)
        Energies of a table of the untruncated DF, increasing, in internal
        units, from which the rounding test of
        an untruncated model reads the DF at the first energy at or above
        :math:`10 K_0`.
    f : array, shape (k,)
        The untruncated DF at `q`.
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}` in internal units;
        0 for an untruncated model.
    a_analytic : float
        The analytic exponent :math:`a` of the DF's tail :math:`A K^{-a}` as
        :math:`K \to 0`. NaN makes every negative value of an untruncated DF
        fail the rounding test.
    k_edge : float, optional
        The potential at the anisotropy radius, :math:`P(R_{\rm a})`, for a
        Type II model, in internal units. The DF's support is the energies
        above `k_edge`, and `q_sign` and `f_sign` are read only at energies
        above `k_edge`.
        Default: ``0.0``, which means no domain edge (all models other than
        Type II).

    Returns
    -------
    gates : TruncationGates
        The DF's sign record, the lowest admissible truncation energy and
        the gates that `k_t` fails.
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
        # The DOMAIN gate requires K_t strictly above k_edge, so the smallest
        # truncation energy the DOMAIN gate admits is the next double above
        # k_edge.
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
            and abs(float(sign[0])) <= TIER_BOUNDS[1] * f_ref
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
