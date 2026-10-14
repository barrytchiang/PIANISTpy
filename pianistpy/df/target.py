"""The truncation energy of a target mass or a target radius, Method I.

Under Method I the untruncated DF does not depend on the truncation, so it is
inverted once and every evaluation is one truncated solve
(`truncate.truncation_chain`). The search runs on x = ln K_t for v = M_t or
R_t:

1. from the probe, deeper truncation until the model is admissible;
2. a bracket: secant steps in (ln K_t, ln v) toward the target, at most a
   factor of 16 per step, until one admissible model lies on each side;
3. where a step lands outside the admissible set, its boundary: the K_t where
   R_t reaches its limit (the cap, or r_a), by regula falsi on 1 / R_t, with a
   bisection step where a probe has no finite model;
4. the Illinois form of regula falsi on (ln K_t, ln(v / target)) until
   |v / target - 1| is within the precision.

A K_t is admissible where its truncated solve succeeds, the model is
spatially finite (the solve's edge event fires), R_t is within the cap and,
for Type II, inside r_a; and K_t lies at or above the gate and the DF's last
zero, and at or below 0.9 of the central potential. On the admissible set
K_t -> M_t and K_t -> R_t decrease. No admissible model at the target is
refused by name, with the value reached at the limit that binds. A model
inside the bracket that a stage refuses (the lowered center's fixed point not
converging) refuses the target, with its cause and the
bracket: the solve takes no fallback.
"""

__all__ = ["TargetRefusal", "solve_target"]

import math
import time
from typing import Any

import numpy as np

from ..exceptions import InputError, UnphysicalModelError
from .results import DistributionFunction
from .truncated import truncation_chain

# |v / target - 1| at the root: the goal of the accuracy tiers
TARGET_GOAL = 1e-6
# the largest admissible R_t, in r_0: below the radius out to which the
# untruncated potential holds the goal against its far-field constant
RT_CAP = 1e5
# the deepest and the shallowest truncation searched, as fractions of the
# central potential; how far above the gate the admissible set starts
ZT_MAX_FRAC = 0.9
ZT_MIN_FRAC = 1e-12
EDGE_REL = 1e-9
# the probe where none is given; the largest factor of one bracketing step; a
# cap on the truncated solves, whose reaching is an error
DEFAULT_PROBE = 0.3
STEP_MAX = 16.0
MAX_EVAL = 60


class TargetRefusal(UnphysicalModelError):
    """No admissible model reaches the target."""


def _mass_scale(profile: Any, concentration: float) -> float:
    """1 / (4 pi M(<c)): the model unit of mass, M(<c) = 1; NFW's closed form
    where the profile is NFW."""
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
    """Return the truncation energy whose model has the target mass or radius.

    Parameters
    ----------
    df : DistributionFunction
        Result of `compute_df`, with its rows as made.
    kind : {"mass", "radius"}
        The target's kind.
    target : float
        The target: a mass in units of the mass inside the concentration
        radius, or a radius in r_0.
    concentration : float, optional
        The concentration that defines the mass unit; required for a mass.
    probe : float, optional
        The first K_t tried; 0.3 by default.
    precision : float, optional
        |v / target - 1| at the root.
    rt_cap : float, optional
        The largest admissible R_t.

    Returns
    -------
    k_t, record
        The truncation energy, and the search's record: every evaluation, the
        value reached and its residual.

    Raises
    ------
    TargetRefusal
        If no admissible model reaches the target; it carries the DF.
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

    # the gates on the rows: the lowest admissible truncation and the last zero
    rows = df.rows
    if rows is None:
        raise InputError("The DF carries no rows as made; it was not inverted.")
    from .gates import truncation_gates

    probe_gates = truncation_gates(
        np.asarray(rows.k[0]),
        np.asarray(rows.f[0]),
        np.asarray(rows.k[0]),
        np.asarray(rows.f[0]),
        0.0,
        math.nan,
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

    def ev(z: float) -> dict[str, Any]:
        """One truncated solve at z: whether it ran, is finite, R_t, M_t, the
        value, whether it is admissible and why not."""
        z = float(z)
        if z in cache:
            return cache[z]
        t0 = time.perf_counter()
        why = ""
        try:
            chain = truncation_chain(df, z)
            ok = True
        except Exception as ex:  # noqa: BLE001 - the solve's own failure
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

    # 1. an admissible model to start from
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
    elif y(e) > 0:  # 2a. too large a model: deeper truncation
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
    else:  # 2b. too small a model: shallower truncation
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
            else:  # 3. the boundary of the admissible set, between zb and bad
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
    # 4. the root, between za (v >= target) and zb (v < target)
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
    return float(zs["z"]), rec
