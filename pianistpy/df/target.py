r"""The truncation energy that gives a target mass or a target radius, Method I.

`solve_target` returns the truncation energy :math:`\mathcal{K}_{\rm t}`
whose Method I truncated model has a target total mass :math:`M_{\rm t}`
(in units of :math:`M_0`) or a target truncation radius :math:`R_{\rm t}`
(in units of :math:`r_0`). Under Method I the untruncated DF does not
depend on the truncation, so the untruncated DF is inverted once, and each
evaluation of a trial :math:`\mathcal{K}_{\rm t}` is one truncated solve
(`truncated.truncation_chain`). The search runs on
:math:`x = \ln \mathcal{K}_{\rm t}` for the value
:math:`v = M_{\rm t}` or :math:`v = R_{\rm t}`, in four steps:

1. From the probe energy, :math:`\mathcal{K}_{\rm t}` increases, by a
   factor of 16 where the trial model is not spatially finite and by a
   factor of 2 otherwise, until the model is admissible.
2. Secant steps in :math:`(\ln \mathcal{K}_{\rm t}, \ln(v / {\rm target}))`
   aim at :math:`\ln(v / {\rm target}) = -0.2` where :math:`v` is above the
   target and at :math:`+0.2` where :math:`v` is below the target, until
   one admissible model lies on each side of the target. Each step changes
   :math:`\mathcal{K}_{\rm t}` by a factor between 1.25 and 16, and by a
   factor of 2 where no secant step exists.
3. Where a step of step 2 toward smaller :math:`\mathcal{K}_{\rm t}` lands
   outside the admissible set, the search narrows the interval between the
   shallowest admissible trial and the deepest inadmissible trial. Each
   step is regula falsi in :math:`\mathcal{K}_{\rm t}` on
   :math:`1 / R_{\rm t} - 1 / R_{\rm root}` where the inadmissible trial
   has a finite model, with :math:`R_{\rm root}` the limit on
   :math:`R_{\rm t}` (the smaller of the cap on :math:`R_{\rm t}` and, for
   a Type II model, :math:`R_{\rm a}`) for a target mass and the target for
   a target radius. Where the inadmissible trial has no finite model, the
   step is a secant step on :math:`1 / R_{\rm t}` through the last two
   admissible trials. The step is a geometric bisection where the regula
   falsi or secant energy is missing, lies outside the interval, or lies
   closer to an end of the interval than :math:`10^{-3}` times the
   interval's width.
4. The Illinois form of regula falsi on
   :math:`(\ln \mathcal{K}_{\rm t}, \ln(v / {\rm target}))` runs until
   :math:`|v / {\rm target} - 1|` is at most the precision, by default
   :math:`10^{-6}`, or until the bracket's width in
   :math:`\ln \mathcal{K}_{\rm t}` falls below :math:`4 \times 10^{-16}`
   times :math:`\max(1, |x_{\rm a}|)`, with :math:`x_{\rm a}` the
   bracket's end at which :math:`v` is above the target. At that width the
   search returns, of the last trial and the two ends of the bracket that
   step 4 started from, the one with the smallest
   :math:`|\ln(v / {\rm target})|`.

The search tries only truncation energies :math:`\mathcal{K}_{\rm t}`
above both the lowest admissible truncation energy of the gates and the
DF's last zero crossing, and at most 0.9 times the central potential. A
tried :math:`\mathcal{K}_{\rm t}` is admissible when four conditions hold:
the truncated solve raises no package error; the model is spatially finite
(the Poisson solve's floor event fires); :math:`R_{\rm t}` is at most the
cap on :math:`R_{\rm t}`; and, for a Type II model, :math:`R_{\rm t}` lies
inside :math:`R_{\rm a}`. On the admissible set, :math:`M_{\rm t}` and
:math:`R_{\rm t}` both decrease as :math:`\mathcal{K}_{\rm t}` increases.
Where no admissible model reaches the target, `solve_target` raises
`TargetRefusal` with the value reached at the limit that binds. Where a
model deeper than an admissible model with :math:`v` above the target is
not admissible, `solve_target` raises `TargetRefusal`. Where a model inside
the final bracket is not admissible (for example, where the
lowered center's fixed point does not converge), `solve_target` raises
`TargetRefusal` with that model's cause and the bracket, and takes no
fallback. The method is in Notes Section T:3.6.
"""

__all__ = ["TargetRefusal", "solve_target"]

import math
import time
from typing import Any

import numpy as np

from ..exceptions import InputError, PianistpyError, UnphysicalModelError
from .results import DistributionFunction
from .truncated import truncation_chain

# The default bound on |v / target - 1| at the root: 1e-6, the upper bound of
# the validation report's "silent" tier.
TARGET_GOAL = 1e-6
# The default largest admissible R_t, in units of r_0. The cap lies below the
# radius out to which the untruncated potential, measured from the potential's
# far-field constant, holds a relative accuracy of 1e-6.
RT_CAP = 1e5
# ZT_MAX_FRAC: the deepest truncation energy searched, as a fraction of the
# central potential. ZT_MIN_FRAC: the shallowest truncation energy searched,
# as a fraction of the central potential, where the gates set no lowest
# admissible truncation energy and the DF has no zero crossing. EDGE_REL: the
# relative margin by which the shallowest truncation energy searched lies
# above the larger of the gates' lowest admissible truncation energy and the
# DF's last zero crossing.
ZT_MAX_FRAC = 0.9
ZT_MIN_FRAC = 1e-12
EDGE_REL = 1e-9
# DEFAULT_PROBE: the first truncation energy tried where the caller gives no
# probe. STEP_MAX: the largest factor by which one step of steps 1 and 2 of
# the search changes K_t.
# MAX_EVAL: the most truncated solves of one search; a search that needs more
# than MAX_EVAL truncated solves raises RuntimeError.
DEFAULT_PROBE = 0.3
STEP_MAX = 16.0
MAX_EVAL = 60


class TargetRefusal(UnphysicalModelError):
    """Error raised where no admissible truncated model reaches the target.

    The error carries the DistributionFunction searched as its ``result``.
    """


def _mass_scale(profile: Any, concentration: float) -> float:
    r"""Return :math:`1 / (4 \pi M(<c))`, the factor from internal mass to model mass.

    :math:`M(<c)` is the profile's mass inside the radius :math:`c` (the
    concentration, in units of :math:`r_0`), and :math:`M(<c)` is the model
    unit of mass :math:`M_0`. For an NFW profile,
    :math:`(\alpha, \beta, \gamma) = (1, 3, 1)`, :math:`M(<c)` is the closed
    form :math:`\ln(1 + c) - c / (1 + c)`.
    """
    c = float(concentration)
    if (profile.rho_alpha, profile.rho_beta, profile.rho_gamma) == (1.0, 3.0, 1.0):
        return float(1.0 / (4.0 * np.pi * (np.log(1.0 + c) - c / (1.0 + c))))
    return 1.0 / (4.0 * np.pi * float(np.asarray(profile.mass(np.array([c])))[0]))


def solve_target(
    df: DistributionFunction,
    kind: str,
    target: float,
    concentration: float | None = None,
    probe: float | None = None,
    precision: float = TARGET_GOAL,
    rt_cap: float = RT_CAP,
) -> tuple[float, dict[str, Any]]:
    r"""Return the truncation energy whose model has the target mass or radius.

    Parameters
    ----------
    df : DistributionFunction
        The result of `compute_df`, carrying the inversion's rows.
    kind : {"mass", "radius"}
        "mass" for a target total mass :math:`M_{\rm t}`; "radius" for a
        target truncation radius :math:`R_{\rm t}`.
    target : float
        The target value: a mass in units of :math:`M_0`, the profile's mass
        inside the concentration, or a radius in units of :math:`r_0`. Must
        be positive and finite.
    concentration : float, optional
        The concentration that defines :math:`M_0`; required where `kind`
        is "mass". Default: ``None``.
    probe : float, optional
        The first truncation energy tried, in internal units, clipped to the
        searched range. Default: ``None``, which means ``0.3``.
    precision : float, optional
        The bound on :math:`|v / {\rm target} - 1|` at the root, in
        :math:`(0, 10^{-2})`. Default: ``1e-6``.
    rt_cap : float, optional
        The largest admissible :math:`R_{\rm t}`, in units of :math:`r_0`.
        Default: ``1e5``.

    Returns
    -------
    k_t : float
        The truncation energy :math:`\mathcal{K}_{\rm t}`, in internal units.
    record : dict
        The search's record: the inputs; every truncated solve with its
        :math:`\mathcal{K}_{\rm t}`, :math:`R_{\rm t}`, :math:`M_{\rm t}`,
        spatial finiteness, admissibility, the cause of inadmissibility and
        wall-clock time; the returned :math:`\mathcal{K}_{\rm t}` with its
        value :math:`v`, :math:`M_{\rm t}`, :math:`R_{\rm t}` and relative
        residual :math:`v / {\rm target} - 1`; and the number of solves.

    Raises
    ------
    InputError
        If `kind` is neither "mass" nor "radius", `target` is not positive
        and finite, `precision` is outside :math:`(0, 10^{-2})`, `kind` is
        "mass" and `concentration` is None, or `df` carries no rows.
    TargetRefusal
        If no admissible model reaches the target, if a model deeper than an
        admissible model with :math:`v` above the target is not admissible,
        or if a model inside the final bracket is not admissible. The error
        carries `df` as its ``result``.
    RuntimeError
        If the search takes more than ``MAX_EVAL`` (60) truncated solves.
    """
    k_t, rec, _ = _solve_target(
        df, kind, target, concentration, probe, precision, rt_cap
    )
    return k_t, rec


def _solve_target(
    df: DistributionFunction,
    kind: str,
    target: float,
    concentration: float | None = None,
    probe: float | None = None,
    precision: float = TARGET_GOAL,
    rt_cap: float = RT_CAP,
) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
    r"""Return `solve_target`'s result and the root's truncation chain.

    The third value is the `truncation_chain` that the search computed at
    the returned truncation energy :math:`\mathcal{K}_{\rm t}`, so that
    `truncate` does not compute that chain again; the third value is None
    where the truncated solve at the returned :math:`\mathcal{K}_{\rm t}`
    raised an error.
    """
    if kind not in ("mass", "radius"):
        raise InputError(f"kind {kind!r}: 'mass' or 'radius'")
    target, precision, rt_cap = float(target), float(precision), float(rt_cap)
    if not (target > 0.0 and math.isfinite(target)):
        raise InputError(
            f"The target {kind} {target!r} is not a positive finite number."
        )
    if not (0.0 < precision < 1e-2):
        raise InputError(f"precision {precision!r}: in (0, 1e-2)")
    if kind == "mass" and concentration is None:
        raise InputError("A target mass needs the concentration that defines its unit.")

    def refuse(message: str) -> TargetRefusal:
        return TargetRefusal(message, result=df)

    component = df.components[0]
    profile, aniso = component.unit_profile, component.anisotropy
    rec: dict[str, Any] = dict(
        kind=kind, target=target, precision=precision, rt_cap=rt_cap, evals=[]
    )
    p_0 = 1.0
    type_ii = math.isfinite(aniso.r_a) and aniso.kind == "II"
    r_a = float(aniso.r_a) if math.isfinite(aniso.r_a) else 1.0
    scale = (
        _mass_scale(profile, float(concentration))
        if kind == "mass" and concentration is not None
        else math.nan
    )

    # The gates read the inversion's rows for the lowest admissible truncation
    # energy and the DF's last zero crossing; the shallowest truncation energy
    # searched lies above both.
    rows = df.rows
    if rows is None:
        raise InputError("The DF carries no rows as made; it was not inverted.")
    from .gates import truncation_gates
    from .truncated import _domain_edge_potential

    probe_gates = truncation_gates(
        np.asarray(rows.k[0]),
        np.asarray(rows.f[0]),
        np.asarray(rows.k[0]),
        np.asarray(rows.f[0]),
        0.0,
        math.nan,
        k_edge=_domain_edge_potential(profile, aniso),
    )
    lim = [
        ("the gate Z_t_min_allowed", float(probe_gates.k_t_min)),
        ("K_0 (the DF's last zero)", float(probe_gates.k_0)),
    ]
    gate_name, z_gate = max(lim, key=lambda t: t[1])
    if z_gate > 0.0:
        z_lo = z_gate * (1.0 + EDGE_REL)
    else:
        gate_name, z_lo = "the search's shallowest truncation", ZT_MIN_FRAC * p_0
    z_hi = ZT_MAX_FRAC * p_0
    r_lim = min(rt_cap, r_a) if type_ii else rt_cap
    r_name = "R_a" if (type_ii and r_a <= rt_cap) else f"the R_t cap {rt_cap:g}"
    r_root = r_lim if kind == "mass" else target
    r_aim = 2.0 * r_lim if kind == "mass" else min(1.2 * target, r_lim)
    if kind == "mass":
        m_0 = (
            profile.mass_total
            / float(np.asarray(profile.mass(np.array([concentration])))[0])
            if math.isfinite(profile.mass_total)
            else math.inf
        )
        if target >= m_0:
            raise refuse(
                f"M_t = {target:.6g} is not reached by any K_t > 0: the untruncated "
                f"model's total mass M_0 = {m_0:.6g} bounds every truncated mass "
                "(binding limit: M_0)"
            )
    if kind == "radius" and (target > r_lim or (type_ii and target >= r_a)):
        raise refuse(
            f"R_t = {target:.6g} is beyond {r_name}"
            + (f" = {r_lim:g}" if r_name == "R_a" else "")
            + f", which bounds every admissible R_t (binding limit: {r_name})"
        )
    cache: dict[float, dict[str, Any]] = {}
    # The chain of each truncated solve that raised no package error, keyed by
    # the solve's K_t; the chain at the root is returned to truncate.
    chains: dict[float, dict[str, Any]] = {}

    def ev(z: float) -> dict[str, Any]:
        r"""Return the truncated solve's evaluation at truncation energy `z`.

        The evaluation is cached by `z`, the truncation energy
        :math:`\mathcal{K}_{\rm t}`. The evaluation states whether the solve
        raised no package error, whether the model is spatially finite, the
        truncation radius :math:`R_{\rm t}`, the truncated model's total mass
        :math:`M_{\rm t}`, the searched value :math:`v`, whether the model is
        admissible, and why the model is not admissible.

        Raises
        ------
        RuntimeError
            If the search has made more than MAX_EVAL truncated solves.
        """
        z = float(z)
        if z in cache:
            return cache[z]
        t0 = time.perf_counter()
        why = ""
        # A failure of the truncated solve, including an early stop of the
        # Poisson ODE (run_poisson_ode), raises a package error and marks K_t
        # inadmissible. Any other error is a fault and propagates.
        try:
            chain = truncation_chain(df, z)
            chains[z] = chain
            ok = True
        except PianistpyError as ex:
            chain, ok = None, False
            why = f"{type(ex).__name__}: {' '.join(str(ex).split())[:160]}"
        wall = time.perf_counter() - t0
        solve = chain["solve"] if chain is not None else None
        finite = ok and solve is not None and bool(solve.event_hit)
        r_t = float(solve.r_event) if finite and solve is not None else math.nan
        m_t = (
            float(np.nanmax(np.asarray(chain["rows"].mass, dtype=np.float64))) * scale
            if finite and chain is not None and kind == "mass"
            else math.nan
        )
        if ok and not finite:
            why = "no edge: the model is not spatially finite"
        elif finite and type_ii and not r_t < r_a:
            why = f"R_t = {r_t:.6g} not inside R_a = {r_a:g}"
        elif finite and r_t > rt_cap:
            why = f"R_t = {r_t:.6g} beyond the R_t cap {rt_cap:g}"
        e = dict(
            z=z,
            ok=ok,
            finite=finite,
            R=r_t,
            M=m_t,
            v=(m_t if kind == "mass" else r_t) if finite else math.nan,
            adm=finite and not why,
            why=why,
        )
        rec["evals"].append(
            dict(
                Z_t=z,
                finite=finite,
                R_t=r_t,
                M_t=m_t,
                admissible=e["adm"],
                why=why,
                wall_s=wall,
            )
        )
        if len(rec["evals"]) > MAX_EVAL:
            raise RuntimeError(
                f"more than {MAX_EVAL} truncated solves without a decision"
            )
        cache[z] = e
        return e

    def y(e: dict[str, Any]) -> float:
        return math.log(e["v"] / target)

    def met(e: dict[str, Any]) -> bool:
        return bool(e["adm"]) and abs(math.expm1(y(e))) <= precision

    def secant(e1: dict[str, Any], e2: dict[str, Any], aim: float) -> float | None:
        x1, x2, y1, y2 = math.log(e1["z"]), math.log(e2["z"]), y(e1), y(e2)
        return None if y1 == y2 else math.exp(x1 + (aim - y1) * (x2 - x1) / (y2 - y1))

    def refuse_above(
        e_max: dict[str, Any], name: str, beyond: dict[str, Any] | None = None
    ) -> TargetRefusal:
        at = f"(Z_t = {e_max['z']:.8g}, M_t = {e_max['M']:.8g}, R_t = {e_max['R']:.6g})"
        if beyond is not None and beyond["finite"]:
            return refuse(
                f"{kind} {target:.8g} is above the reachable range, which ends where "
                f"{name} binds: between {kind} {e_max['v']:.8g}, an admissible model "
                f"{at}, and {kind} {beyond['v']:.8g} at Z_t = {beyond['z']:.8g}, "
                f"already beyond {name}"
            )
        return refuse(
            f"{kind} {target:.8g} is above the reachable range: admissible models "
            f"reach at most {kind} {e_max['v']:.8g} {at}, where {name} binds"
        )

    # Step 1: deepen the truncation from the probe until the model is
    # admissible.
    z = min(max(float(DEFAULT_PROBE if probe is None else probe), z_lo), z_hi)
    e = ev(z)
    while not e["adm"]:
        if z >= z_hi:
            raise refuse(
                f"no admissible model up to the deepest truncation searched, "
                f"Z_t = {z_hi:.6g} ({e['why']})"
            )
        z = min(STEP_MAX * z if not e["finite"] else 2.0 * z, z_hi)
        e = ev(z)
    za: dict[str, Any] | None = None
    zb: dict[str, Any] | None = None
    zs: dict[str, Any] | None = None
    hist = [e]
    if met(e):
        zs = e
    elif y(e) > 0:  # Step 2a: v above the target, so truncate deeper.
        za = e
        while zb is None:
            if za["z"] >= z_hi:
                raise refuse(
                    f"{kind} {target:.8g} is below the reachable range: the deepest "
                    f"truncation searched, Z_t = {z_hi:.6g} ({ZT_MAX_FRAC:g} of the "
                    f"central potential), gives {za['v']:.8g}"
                )
            zp = secant(hist[-2], hist[-1], -0.2) if len(hist) >= 2 else None
            zn = min(
                max(zp if zp else 2.0 * za["z"], 1.25 * za["z"]),
                STEP_MAX * za["z"],
                z_hi,
            )
            e = ev(zn)
            if not e["adm"]:
                raise refuse(
                    f"the model is not admissible at Z_t = {zn:.6g} ({e['why']}), "
                    "deeper than an admissible one"
                )
            hist.append(e)
            if met(e):
                za = zb = e
            elif y(e) > 0:
                za = e
            else:
                zb = e
    else:  # Step 2b: v below the target, so truncate shallower.
        zb, bad = e, None
        while za is None:
            if bad is None:
                if zb["z"] <= z_lo * (1.0 + 1e-15):
                    raise refuse_above(zb, gate_name)
                zp = secant(hist[-2], hist[-1], 0.2) if len(hist) >= 2 else None
                zn = max(
                    min(zp if zp else 0.5 * zb["z"], zb["z"] / 1.25),
                    zb["z"] / STEP_MAX,
                    z_lo,
                )
            else:  # Step 3: narrow the interval between bad and zb.
                zr: float | None = None
                if bad["finite"]:
                    f1, f2 = 1.0 / zb["R"] - 1.0 / r_root, 1.0 / bad["R"] - 1.0 / r_root
                    if f1 != f2:
                        zr = zb["z"] + (0.0 - f1) * (bad["z"] - zb["z"]) / (f2 - f1)
                elif len(hist) >= 2 and hist[-1]["R"] != hist[-2]["R"]:
                    e1, e2 = hist[-2], hist[-1]
                    zr = e2["z"] + (1.0 / r_aim - 1.0 / e2["R"]) * (
                        e1["z"] - e2["z"]
                    ) / (1.0 / e1["R"] - 1.0 / e2["R"])
                lo_, hi_ = bad["z"], zb["z"]
                if (
                    zr is None
                    or not (lo_ < zr < hi_)
                    or min(zr - lo_, hi_ - zr) < 1e-3 * (hi_ - lo_)
                ):
                    zr = math.sqrt(lo_ * hi_)
                if hi_ - lo_ <= 1e-15 * hi_:
                    raise refuse_above(
                        zb,
                        r_name if bad["finite"] else "the edge (no finite model below)",
                        bad,
                    )
                zn = zr
            e2 = ev(zn)
            if e2["adm"]:
                hist.append(e2)
                if met(e2) or y(e2) > 0:
                    za = e2
                else:
                    zb = e2
                    if bad is not None and zb["R"] >= r_lim * (1.0 - precision):
                        raise refuse_above(zb, r_name, bad)
            else:
                if kind == "mass" and e2["finite"] and e2["v"] < target:
                    raise refuse_above(zb, r_name, e2)
                if bad is None or e2["z"] > bad["z"]:
                    bad = e2
    # Step 4: the Illinois regula falsi finds the root between za, where v is
    # above the target or within the precision of the target, and zb, where v
    # is below the target.
    if za is not None and zb is not None and za is not zb:
        if met(za):
            zs = za
        else:
            xa, xb, ya, yb, side = math.log(za["z"]), math.log(zb["z"]), y(za), y(zb), 0
            for _ in range(MAX_EVAL):
                x = (xa * yb - xb * ya) / (yb - ya)
                if not (min(xa, xb) < x < max(xa, xb)):
                    x = 0.5 * (xa + xb)
                e = ev(math.exp(x))
                if not e["adm"]:
                    lo, hi = sorted((math.exp(xa), math.exp(xb)))
                    raise refuse(
                        f"the model is not admissible at Z_t = {e['z']!r} inside the "
                        f"bracket [{lo!r}, {hi!r}] of admissible models on either side "
                        f"of the target ({e['why']}); no fallback is taken"
                    )
                if met(e):
                    zs = e
                    break
                if y(e) > 0:
                    xa, ya = x, y(e)
                    if side == 1:
                        yb *= 0.5
                    side = 1
                else:
                    xb, yb = x, y(e)
                    if side == -1:
                        ya *= 0.5
                    side = -1
                if abs(xb - xa) < 4e-16 * max(1.0, abs(xa)):
                    zs = min((za, zb, e), key=lambda t: abs(y(t)))
                    break
            if zs is None:
                raise RuntimeError(f"no root within {MAX_EVAL} truncated solves")
    elif za is not None:
        zs = za
    assert zs is not None
    rec.update(
        Z_t=zs["z"],
        achieved=zs["v"],
        M_t=zs["M"],
        R_t=zs["R"],
        residual=zs["v"] / target - 1.0,
        n_solves=len(rec["evals"]),
    )
    return float(zs["z"]), rec, chains.get(float(zs["z"]))
