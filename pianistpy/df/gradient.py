r"""The inversion's gradient in the anisotropy parameters at a fixed profile.

The inversion's rows F = T_b[g](q) (`invert._rows`: the kernel and the
completion below the floor) on a fixed grid q. At a fixed central anisotropy
b the operator T_b is linear in its source g, the rows of d^N rho_aux / dP^N,
so, by that linearity:

    dF/dR_a    = T_b[ d_Ra g ]
    dF/dbeta_c = T_b[ d_b g ] + E_b[g]

with d_theta g the mixed derivatives at fixed P (`ParameterDerivs`) and E_b
the derivative of the operator itself in b with g held: the prefactor C(n, b)
and the kernel's exponent, computed by the operator at b +- d (central,
Richardson over d and d/2) or, at the corners where the inversion's form
changes on one side (half-integer :math:`\beta_{\rm c}`, the pair's switch),
by the one-sided Richardson combinations (third order) of both sides and
their mean; the step d
balances the kernel's measured accuracy over d against the quotient's
truncation term (`_exponent_part`).

The table (`invert._table_pieces`) is a function of the rows and of closed
forms in theta, so its derivative is the chain rule through each piece, with
every discrete choice of the forward held (the grid, the anchor, the fit's
window, the nodes below the written floor), `table_jacobian`:

- the rows' log10 F and their log slope, from dF and d(dF/dK), the rows'
  part as the operator's own derivative along the mixed rows (`_along`);
- the form above the anchor (`top_patch`): its least-squares fit linearized
  in the rows, and the center's exponent kappa(b) and amplitude A(b) in
  closed form (the regular branch);
- the nodes below the written floor (`below_floor`): the integral linear in
  its source, the source's cubic in (ln P, ln ``|g|``) linearized, and in b the
  floor pair's exponent and prefactor by the same difference as E_b;
- the tail (A, a(b), k), from the first node below the floor or the first
  written row.

On a component on a truncated profile (Method II, `tabulated.tabulated_df`)
the parent is held and the child's sources are a Leibniz sum of the parent's
density derivatives and the factor ``g(R(P); beta_c, R_a)``, so their derivative
is the same sum with d g / d theta in closed form; the rows follow as above
on the grid held at theta, and the leaves are log10 ``|F|`` on the table's nodes,
the edge law (p, C_0, C_1) and the center the form states, each closed form
with its derivative written out.

Served: constant beta and Cuddeford-Osipkov-Merritt (COM) Type I (:math:`\beta_{\rm c}`,
:math:`R_{\rm a}`), Osipkov-Merritt Type I and II (:math:`R_{\rm a}`).
Refused: COM Type II, whose interior point at :math:`P(R_{\rm a})` moves
with :math:`R_{\rm a}` (and its exponent with :math:`\beta_{\rm c}`), so the
rows of the mixed derivative are not integrable there; and a table whose
center is not on its regular branch (the critical band, where the branch
changes with :math:`\beta_{\rm c}`).
"""

__all__ = [
    "inversion_nodes",
    "make_inversion_rows",
    "make_inversion_table",
    "rows_jacobian",
    "table_jacobian",
]

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import digamma

from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..profiles import TruncatedProfile
from ..settings import Settings
from .central import _gamma_ratio, center_form
from .derivatives import ExactDerivs, ParameterDerivs, source_rows
from .grid import inversion_grid
from .inputs import Component
from .inversion import inversion_prefactor, order_pair
from .pair import potential_density_pair
from .rows import below_integral
from .stages import AuxDerivatives

# the offset at which the form just beside a central anisotropy is read, and
# the parameters of each family
BESIDE = 2.0**-30
PARAMETERS = {
    "constant": ("beta_c",),
    "isotropic": ("beta_c",),
    "COM_I": ("beta_c", "r_a"),
    "OM_I": ("r_a",),
    "OM_II": ("r_a",),
}
_LN10 = float(np.log(10.0))


def _invert():
    from . import invert

    return invert


def inversion_nodes(component: Component, settings: Settings) -> np.ndarray:
    """Return the grid q the inversion places its rows on (`inversion_df`'s:
    the pilot's feature where the family has one, then `top_limit`, the
    least depth at which the exact derivatives are finite in float64)."""
    inv = _invert()
    profile, aniso = component.unit_profile, component.anisotropy
    feature = (
        inv.pilot_feature(component, settings) if aniso.family in inv._PILOTED else None
    )
    q, _ = inversion_grid(profile, aniso, settings, feature=feature)
    return inv._below_limit(q, component)


def _with(component: Component, beta_c: float, r_a: float) -> Component:
    a = component.anisotropy
    return replace(
        component, anisotropy=Anisotropy(beta_c=beta_c, r_a=r_a, kind=a.kind)
    )


def _form_key(b: float) -> tuple[int, bool]:
    p = order_pair(b)
    return int(p.order_j), p.kernel_exponent == 0.0


def rows_jacobian(
    component: Component,
    settings: Settings,
    q: np.ndarray | None = None,
    steps: dict[tuple, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Return the inversion's rows and their derivatives in the component's
    anisotropy parameters on a fixed grid.

    Parameters
    ----------
    component : Component
        An analytic Method I component.
    settings : Settings
    q : array, optional
        The grid; `inversion_nodes` where None.

    Returns
    -------
    q, f : array
        The grid and the rows (the forward's, unchanged).
    jac : dict
        ``{"beta_c": dF/dbeta_c, "r_a": dF/dR_a}``, the parameters the family
        has.
    record : dict
        How the exponent's part was taken (``"central"`` or ``"one-sided"``),
        its steps, the kernel's accuracy and its Richardson residual max
        ``|E(d) - E(d/2)|`` over the column's scale max ``|dF/dbeta_c|`` and,
        one-sided, the two sides' spread over the same scale. The accuracy
        is one maximum over the whole column, whose entries can span some
        thirty decades (the largest near the center), so the steps suit the
        largest entries; an entry far below them is differenced at the same
        steps.

    Raises
    ------
    InputError
        Where the component is not analytic, or a COM Type II, or its
        family has no parameter.
    """
    inv = _invert()
    aniso = component.anisotropy
    profile = component.unit_profile
    family = aniso.family
    if component.profile is not profile:
        raise InputError("The inversion's rule serves an analytic profile only.")
    if family == "COM_II":
        raise InputError(
            "The inversion's rule refuses COM Type II: its interior point at "
            "P(R_a) moves with R_a and beta_c, which the rows of the mixed "
            "derivative do not carry."
        )
    params = PARAMETERS.get(family, ())
    if not params:
        raise InputError(f"A {family} model has no anisotropy parameter to vary.")
    if q is None:
        q = inversion_nodes(component, settings)
    q = np.asarray(q, dtype=np.float64)
    b = 0.0 if family.startswith("OM") else float(aniso.beta_c)
    r_a = float(aniso.r_a) if np.isfinite(aniso.r_a) else 1.0
    pair = potential_density_pair((component,))
    held: dict[str, Any] = {}

    def capture(pair_, grid, form):
        aux = inv.aux_derivatives(pair_, grid, form)
        held["aux"] = aux
        return aux

    f = np.asarray(inv._rows(component, settings, pair, q, capture)[0], np.float64)
    orders = held["aux"].orders
    jac: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {}
    for theta in params:
        src = ParameterDerivs(profile, b, family, r_a, max(2, orders[-1]), wrt=theta)
        rows = source_rows(src, q, orders)
        jac[theta] = np.asarray(
            inv._rows(component, settings, pair, q, lambda *_, _r=rows: _r)[0],
            np.float64,
        )
    if "beta_c" in params:
        e, record = _exponent_part(component, settings, q, b, r_a, orders, steps)
        jac["beta_c"] = jac["beta_c"] + e
        scale = max(float(np.max(np.abs(jac["beta_c"]))), 1e-300)
        for key in ("accuracy", "residual", "spread"):
            if key in record:
                record[key] = record[key] / scale
    return q, f, jac, record


def _region(b, s, key=None):
    r"""The length w of (0, w] over which the form ``key`` (the inversion's,
    `_form_key`, by default) at b + s x stays the one just beside b, short of
    the next half-integer :math:`\beta_{\rm c}` (the local closed form, whose
    neighbors share its key) and of :math:`\beta_{\rm c} = 1`: doubling, then
    bisection on the key."""
    key = _form_key if key is None else key
    half = math.floor(b - 0.5) + 1.5 - b if s > 0 else b - math.ceil(b - 0.5) + 0.5
    x0 = BESIDE
    beside = key(b + s * x0)

    def same(x):
        return b + s * x < 1.0 and key(b + s * x) == beside

    lo, hi = x0, 2.0 * x0
    while hi < 1.0 and same(hi):
        lo, hi = hi, 2.0 * hi
    if hi >= 1.0 and same(1.0):
        return min(1.0, half)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if same(mid) else (lo, mid)
    return min(lo, half)


def _finer(settings: Settings) -> Settings:
    """The settings with the inversion's nodes doubled: the kernel's accuracy
    is the rows' change from the settings' rule to this one."""
    groups: dict[str, Any] = {g: dict(v) for g, v in settings.overrides.items()}
    groups["inversion"] = {
        **groups.get("inversion", {}),
        "n_nodes": 2 * int(settings.inversion["n_nodes"]),
    }
    return Settings(quick=settings.tier == "quick", **groups)


def _exponent_part(component, settings, q, b, r_a, orders, steps=None):
    """E_b[g]: d/db of T_b with the rows held at b (the orders of both
    neighbors' forms, j - 1 to j + 3), by `_table_difference` with the
    kernel's accuracy from the rows at twice the nodes, and its record."""
    inv = _invert()
    profile, family = component.unit_profile, component.anisotropy.family
    ext = tuple(range(orders[0], orders[-1] + 2))
    held = source_rows(
        ExactDerivs(profile, beta_c=b, model=family, r_a=r_a, max_order=ext[-1]),
        q,
        ext,
    )

    def t(bb, s=settings):
        c = _with(component, bb, component.anisotropy.r_a)
        pair = potential_density_pair((c,))
        return np.asarray(inv._rows(c, s, pair, q, lambda *_: held)[0], np.float64)

    member = ("rows", profile, family, component.anisotropy.kind, settings)
    return _table_difference(
        t, b, _form_key, member, lambda bb: t(bb, _finer(settings)), steps
    )


def make_inversion_rows(
    component: Component, settings: Settings, q: np.ndarray | None = None
) -> tuple[Any, tuple[str, ...], np.ndarray]:
    """Return ``(rows, params, q)``: ``rows(theta)`` is the inversion's rows on
    the fixed grid q as a JAX function of the component's anisotropy
    parameters (``params``, in that order) with a custom VJP. The forward is
    `invert._rows` through a host callback (the forward's numbers), the
    backward the rule (`rows_jacobian`) at the same parameters."""
    import jax
    import jax.numpy as jnp

    inv = _invert()
    family = component.anisotropy.family
    params = PARAMETERS.get(family, ())
    if not params:
        raise InputError(f"A {family} model has no anisotropy parameter to vary.")
    if q is None:
        q = inversion_nodes(component, settings)
    q = np.asarray(q, dtype=np.float64)
    a0 = component.anisotropy

    def at(theta):
        v = dict(zip(params, np.asarray(theta, dtype=np.float64).tolist(), strict=True))
        return _with(component, v.get("beta_c", a0.beta_c), v.get("r_a", a0.r_a))

    def host_f(theta):
        c = at(theta)
        pair = potential_density_pair((c,))
        return np.asarray(inv._rows(c, settings, pair, q)[0], dtype=np.float64)

    # the exponent's steps, measured at the component's own parameters and
    # carried to every theta, so a gradient never depends on the calls before
    steps: dict[tuple, dict[str, Any]] = {}
    own = np.array([getattr(a0, p) for p in params], dtype=np.float64)

    def host_j(theta):
        if not steps and not np.array_equal(np.asarray(theta, np.float64), own):
            rows_jacobian(component, settings, q, steps)
        _, _, jac, _ = rows_jacobian(at(theta), settings, q, steps)
        return np.stack([jac[p] for p in params])

    shape_f = jax.ShapeDtypeStruct((q.size,), jnp.float64)
    shape_j = jax.ShapeDtypeStruct((len(params), q.size), jnp.float64)

    @jax.custom_vjp
    def rows(theta):
        return jax.pure_callback(host_f, shape_f, theta)

    def fwd(theta):
        return rows(theta), theta

    def bwd(theta, v):
        jac = jax.pure_callback(host_j, shape_j, theta)
        return (jac @ v,)

    rows.defvjp(fwd, bwd)
    return rows, params, q


def _checked(component: Component) -> tuple[str, tuple[str, ...]]:
    aniso = component.anisotropy
    family = aniso.family
    if component.profile is not component.unit_profile:
        raise InputError("The inversion's rule serves an analytic profile only.")
    if family == "COM_II":
        raise InputError(
            "The inversion's rule refuses COM Type II: its interior point at "
            "P(R_a) moves with R_a and beta_c, which the rows of the mixed "
            "derivative do not carry."
        )
    params = PARAMETERS.get(family, ())
    if not params:
        raise InputError(f"A {family} model has no anisotropy parameter to vary.")
    return family, params


def _floor_key(b: float) -> tuple[int, bool]:
    p = order_pair(b)
    return int(p.order_m), p.order_n == 0.0


class _Shifted:
    """The closed forms g + e dg: `ExactDerivs`' and `ParameterDerivs`'."""

    def __init__(self, base: Any, d: Any, e: float) -> None:
        self.base, self.d, self.e = base, d, e

    def r_of_p(self, p: Any) -> Any:
        return self.base.r_of_p(p)

    def _eval(self, o: int, rv: Any) -> Any:
        return self.base._eval(o, rv) + self.e * self.d._eval(o, rv)

    def derivs_at_p(self, p: Any, n: int) -> tuple[Any, ...]:
        a, d = self.base.derivs_at_p(p, n), self.d.derivs_at_p(p, n)
        return tuple(x + self.e * y for x, y in zip(a, d, strict=True))


def _along(component, settings, pair, q, aux, rows):
    """The operator's derivative along the source dg (``rows``) at g
    (``aux``): (T_b[g + e dg] - T_b[g - e dg]) / 2e on F and dF/dK. The
    operator reads a source row in ln g where it is positive throughout and in
    g otherwise, so T_b[dg] itself is not its derivative where g is read in
    ln g; e keeps every positive row positive, ``e |dg| <= 1e-6 g``."""
    inv = _invert()
    pos = np.all(aux.values[0] > 0.0, axis=1)
    ratio = (
        float(np.max(np.abs(rows.values[0][pos]) / aux.values[0][pos]))
        if pos.any()
        else 0.0
    )
    e = 1e-6 / ratio if ratio > 0.0 else 1.0
    out = []
    for sign in (1.0, -1.0):
        shifted = AuxDerivatives(
            aux.orders,
            aux.values + sign * e * rows.values,
            aux.at_floor + sign * e * rows.at_floor,
            True,
            _Shifted(aux.source, rows.source, sign * e),
        )
        r = inv._rows(component, settings, pair, q, lambda *_, _s=shifted: _s)
        out.append(np.stack([np.asarray(r[0]), np.asarray(r[1])]))
    return (out[0] - out[1]) / (2.0 * e)


def _max_abs(x: Any) -> float:
    """max |x| over x's finite entries (a table's rows hold NaN where it has
    no value), 0 where it has none."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else 0.0


def _table_difference(
    t: Any,
    b: float,
    key: Any,
    member: Any = None,
    finer: Any = None,
    store: dict[tuple, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """d/db of t(b) by its own values at b +- d, at a step set by measurement,
    and its record (absolute values).

    acc is t's accuracy: its change to ``finer`` (t at a finer rule) where
    given, else its rounding. Central (the form ``key`` the same at b and just
    beside it on both sides, n != 0), the Richardson quotient over d and d/2
    has the rounding term 3 acc / d and the truncation term d^4 |F5| / 480;
    their balance gives d^5 = 360 acc / |F5|, with F5 = d^5 t / db^5 from the
    Richardson residual at a pilot step, half the form's region: |E(h) -
    E(h/2)| = |F5| h^4 / 512. One-sided (the corners), the Richardson
    combination of the second-order quotients over d and d/2 has 12 acc / d
    and d^3 |F4| / 24, so d^4 = 96 acc / |F4|, with |E(h) - E(h/2)| =
    7 |F4| h^3 / 192 at a pilot step of a quarter of the side's region.

    With a ``member`` (hashable, t's member without its anisotropy
    parameters) and a ``store`` (a dict the caller holds), acc and the
    steps are read from the store where it has them, each step held inside
    the form's region at the new b, and measured and written to it where it
    has not: the caller measures them once, at its own parameters, and
    passes the store with every later call (a chain's next parameters).
    Without a store they are measured at each call. The record carries the steps, the
    pilots and the regions they lie in, acc, whether this call measured them,
    the Richardson residual max |E(d) - E(d/2)| at the step taken (one-sided,
    the larger side's) and, with both sides, their spread max |E_+ - E_-|."""
    k0 = key(b)
    central = not k0[1] and all(key(b + s * BESIDE) == k0 for s in (1.0, -1.0))
    cache = (
        None if member is None or store is None else (member, key.__name__, k0, central)
    )
    kept = None if cache is None or store is None else store.get(cache)
    t0 = t(b) if kept is None or not central else None
    if kept is None:
        assert t0 is not None
        rounding = float(np.finfo(np.float64).eps) * _max_abs(t0)
        acc = max(_max_abs(t0 - finer(b)), rounding) if finer is not None else rounding
    else:
        acc = kept["accuracy"]

    if central:

        def quotient(h):
            d1, d2, d4 = (
                (t(b + x) - t(b - x)) / (2.0 * x) for x in (h, 0.5 * h, 0.25 * h)
            )
            e = (4.0 * d2 - d1) / 3.0
            return e, _max_abs(e - (4.0 * d4 - d2) / 3.0)

        regions = (min(_region(b, 1.0, key), _region(b, -1.0, key)),)
        measured = kept is None
        if kept is None:
            pilot = 0.5 * regions[0]
            f5 = 512.0 * quotient(pilot)[1] / pilot**4
            d = min(pilot, (360.0 * acc / f5) ** 0.2) if f5 > 0.0 else pilot
            kept = {"accuracy": acc, "pilots": (pilot,), "steps": (d,)}
            if cache is not None and store is not None:
                store[cache] = kept
        steps = (min(kept["steps"][0], 0.5 * regions[0]),)
        e, residual = quotient(steps[0])
        return e, {
            "exponent": "central",
            "steps": steps,
            "pilots": kept["pilots"],
            "regions": regions,
            "measured": measured,
            "accuracy": acc,
            "residual": residual,
        }

    assert t0 is not None

    def one_sided(s, h):
        tt = {x: t(b + s * x) for x in (0.25 * h, 0.5 * h, h, 2.0 * h)}

        def q2(x):
            return s * (-3.0 * t0 + 4.0 * tt[x] - tt[2.0 * x]) / (2.0 * x)

        e1, e2, e4 = q2(h), q2(0.5 * h), q2(0.25 * h)
        e = (4.0 * e2 - e1) / 3.0
        return e, _max_abs(e - (4.0 * e4 - e2) / 3.0)

    sides = (1.0, -1.0)
    regions = tuple(_region(b, s, key) for s in sides)
    measured = kept is None
    if kept is None:
        pilots, chosen = [], []
        for s, w in zip(sides, regions, strict=True):
            pilot = 0.25 * w
            f4 = 192.0 * one_sided(s, pilot)[1] / (7.0 * pilot**3)
            pilots.append(pilot)
            chosen.append(min(pilot, (96.0 * acc / f4) ** 0.25) if f4 > 0.0 else pilot)
        kept = {"accuracy": acc, "pilots": tuple(pilots), "steps": tuple(chosen)}
        if cache is not None and store is not None:
            store[cache] = kept
    steps = tuple(min(d, 0.25 * w) for d, w in zip(kept["steps"], regions, strict=True))
    found, residual = [], 0.0
    for s, d in zip(sides, steps, strict=True):
        e, r = one_sided(s, d)
        found.append(e)
        residual = max(residual, r)
    return np.mean(found, axis=0), {
        "exponent": "one-sided",
        "sides": len(found),
        "steps": steps,
        "pilots": kept["pilots"],
        "regions": regions,
        "measured": measured,
        "accuracy": acc,
        "residual": residual,
        "spread": _max_abs(found[0] - found[1]),
    }


def _table_rows(
    component: Component,
    settings: Settings,
    q: np.ndarray,
    steps: dict[tuple, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    r"""The rows (F and dF/dK), their pieces, and their derivatives in each
    parameter: the operator's derivative along the mixed rows (`_along`) and,
    in :math:`\beta_{\rm c}`, its exponent's part on both F and dF/dK."""
    inv = _invert()
    family = component.anisotropy.family
    params = PARAMETERS[family]
    aniso, profile = component.anisotropy, component.unit_profile
    b = 0.0 if family.startswith("OM") else float(aniso.beta_c)
    r_a = float(aniso.r_a) if np.isfinite(aniso.r_a) else 1.0
    pair = potential_density_pair((component,))
    f, ddf, frec, aux, floor = inv._rows(component, settings, pair, q)
    orders = aux.orders
    jac: dict[str, np.ndarray] = {}
    dsrc: dict[str, Any] = {}
    record: dict[str, Any] = {}
    for theta in params:
        src = ParameterDerivs(profile, b, family, r_a, max(2, orders[-1]), wrt=theta)
        rows = source_rows(src, q, orders)
        dsrc[theta] = rows
        jac[theta] = _along(component, settings, pair, q, aux, rows)
    if "beta_c" in params:
        ext = tuple(range(orders[0], orders[-1] + 2))
        held = source_rows(
            ExactDerivs(profile, beta_c=b, model=family, r_a=r_a, max_order=ext[-1]),
            q,
            ext,
        )

        def t(bb, s=settings):
            c = _with(component, bb, aniso.r_a)
            out = inv._rows(c, s, potential_density_pair((c,)), q, lambda *_: held)
            return np.stack([np.asarray(out[0]), np.asarray(out[1])])

        member = ("table rows", profile, family, aniso.kind, settings)
        e, record = _table_difference(
            t, b, _form_key, member, lambda bb: t(bb, _finer(settings)), steps
        )
        jac["beta_c"] = jac["beta_c"] + e
    return dict(
        q=q,
        f=np.asarray(f, np.float64),
        ddf=np.asarray(ddf, np.float64),
        frec=frec,
        aux=aux,
        floor=floor,
        b=b,
        r_a=r_a,
        params=params,
        jac=jac,
        dsrc=dsrc,
        record=record,
    )


def _center_jvp(component: Component, theta: str) -> tuple[float, float]:
    r"""d kappa and d A of the center's regular branch (`central`): kappa = s0 -
    nu, A = N'(b) C cu^-s0 Gamma(nu - s0) / Gamma(-s0), s0 = (2b - gamma)/(2 -
    gamma), nu = 3/2 - b; neither depends on :math:`R_{\rm a}`."""
    if theta != "beta_c":
        return 0.0, 0.0
    profile = component.unit_profile
    b = float(component.anisotropy.beta_c)
    g = float(profile.rho_gamma)
    cu = float(profile.rho_norm) / ((2.0 - g) * (3.0 - g))
    nu = 1.5 - b
    s0 = (2.0 * b - g) / (2.0 - g)
    ds = 2.0 / (2.0 - g)
    dln_a = (
        np.log(2.0)
        + digamma(1.0 - b)
        - np.log(cu) * ds
        + digamma(nu - s0) * (-1.0 - ds)
        + digamma(-s0) * ds
    )
    return ds + 1.0, float(dln_a)


def _lstsq_jvp(m, y, c, dm, dy):
    """dc of c = argmin |M c - y| for dM and dy."""
    r = y - m @ c
    dc = np.linalg.lstsq(m, dy - dm @ c, rcond=None)[0]
    return dc + np.linalg.solve(m.T @ m, dm.T @ r)


def _patch_jvp(qa, fa, dfa, kappa, dkappa, amplitude, damp, patch):
    """d(a_anchor, b, d) of `top_patch` with its window and branch held."""
    u_all = 1.0 - qa
    u_anc = 1.0 - float(qa[-1])
    f_anchor, df_anchor = float(fa[-1]), float(dfa[-1])
    if f_anchor > 0.0:
        ok_amp = (u_all >= u_anc) & (fa > 0.0)
        idx = np.where(ok_amp)[0]
        idx = idx[np.argsort(u_all[idx])][:7]
        if idx.size >= 5:
            ln_c = float(np.mean(np.log(fa[idx]) - kappa * np.log(u_all[idx])))
            f_anchor = float(np.exp(ln_c + kappa * np.log(u_anc)))
            df_anchor = f_anchor * (
                float(np.mean(dfa[idx] / fa[idx] - dkappa * np.log(u_all[idx])))
                + dkappa * np.log(u_anc)
            )
    two_terms = bool(np.isfinite(patch.fit_window))
    if two_terms:
        ok_fit = np.isfinite(fa) & (fa > 0.0) & (u_all >= u_anc)
        m = ok_fit & (u_all <= float(patch.fit_window) * u_anc)
        x, fm, dfm = u_all[m] / u_anc, fa[m], dfa[m]
        xk = x**kappa
        dxk = xk * np.log(x) * dkappa
    if not patch.closed_form:
        if not two_terms:
            return df_anchor, 0.0, 0.0
        mm = np.vstack([xk, np.ones_like(x), x]).T / fm[:, None]
        c = np.linalg.lstsq(mm, np.ones(x.size), rcond=None)[0]
        dmm = -mm * (dfm / fm)[:, None]
        dmm[:, 0] += dxk / fm
        dc = _lstsq_jvp(mm, np.ones(x.size), c, dmm, np.zeros(x.size))
        return float(dc[0]), float(dc[1]), float(dc[2]) / u_anc
    a_term = float(amplitude) * u_anc**kappa
    da = float(damp) * u_anc**kappa + a_term * np.log(u_anc) * dkappa
    if not two_terms:
        return da, df_anchor - da, 0.0
    mb = np.vstack([np.ones_like(x), x]).T / fm[:, None]
    yb = 1.0 - a_term * xk / fm
    cb = np.linalg.lstsq(mb, yb, rcond=None)[0]
    dmb = -mb * (dfm / fm)[:, None]
    dyb = -(da * xk + a_term * dxk) / fm + a_term * xk * dfm / fm**2
    dc = _lstsq_jvp(mb, yb, cb, dmb, dyb)
    return da, float(dc[0]), float(dc[1]) / u_anc


def _below_jvp(component, settings, parts, below, theta, steps=None):
    """d F and d(dF/dK) at the nodes below the written floor."""
    floor, aux, dsrc = parts["floor"], parts["aux"], parts["dsrc"][theta]
    m, n, pref, p_min = (
        int(floor["m"]),
        float(floor["n"]),
        float(floor["pref"]),
        float(floor["p_min"]),
    )
    kb, fb = np.asarray(below["K"]), np.asarray(below["F"])
    p_d = np.asarray(below["p_source"])
    ln_pd = np.log(p_d)

    def linear(fn, order):
        if fn is None:
            return None
        ratio = np.asarray(dsrc.at(p_d, order)) / np.asarray(aux.at(p_d, order))
        cs = CubicSpline(ln_pd, ratio)
        return lambda pp: fn(pp) * cs(np.log(pp))

    g_i, r_i = below["g_i"], below["r_i"]
    ddmm = float(dsrc.at(np.array([p_min]), m)[0])
    dfb = below_integral(kb, p_min, ddmm, n, pref, linear(g_i, m + 1), linear(r_i, m))
    record: dict[str, Any] = {}
    if theta == "beta_c":
        dmm = float(floor["dmm"])

        def t(bb):
            nn = float(order_pair(bb).order_n)
            pr = inversion_prefactor(nn, bb)
            return below_integral(kb, p_min, dmm, nn, pr, g_i, r_i)[None, :]

        aniso = component.anisotropy
        member = ("below", component.unit_profile, aniso.family, aniso.kind, settings)
        e, record = _table_difference(t, parts["b"], _floor_key, member, None, steps)
        dfb = dfb + e[0]
    ln_k = np.log(kb)
    dy = dfb / fb
    ddfb = np.asarray(below["dFdK"]) * dy + fb * CubicSpline(ln_k, dy)(ln_k, 1) / kb
    return dfb, ddfb, record


def _critical(component: Component, branch: str) -> str:
    return (
        f"The center of {component.name!r} is on the {branch} branch: the "
        "critical band, where the branch changes with beta_c, is outside the "
        "table's rule."
    )


def _tabulated():
    from . import tabulated

    return tabulated


def _child_checked(component: Component) -> tuple[str, tuple[str, ...]]:
    family = component.anisotropy.family
    if family == "COM_II":
        raise InputError(
            "The inversion's rule refuses COM Type II: its interior point at "
            "P(R_a) moves with R_a and beta_c, which the rows of the mixed "
            "derivative do not carry."
        )
    params = PARAMETERS.get(family, ())
    if not params:
        raise InputError(f"A {family} model has no anisotropy parameter to vary.")
    branch = center_form(component.unit_profile, component.anisotropy)["branch"]
    if branch != "regular":
        raise InputError(_critical(component, branch))
    return family, params


def _dfactor(family: str, b: float, r_a: float, r: np.ndarray, theta: str) -> tuple:
    """d g / d theta of the factor ``g = R^(2 beta_c) u^s``,
    ``u = 1 +- R^2 / r_a^2``, and its R-derivatives to order 3, on R: closed
    forms."""
    import sympy

    from ..anisotropy.factor import inverse_factor_expr

    rs, a_s = sympy.symbols("R r_a", positive=True)
    bs = sympy.Symbol("b", real=True)
    g = inverse_factor_expr(rs, family, bs, a_s)
    dg = sympy.diff(g, bs if theta == "beta_c" else a_s)
    out = []
    for k in range(4):
        fn = sympy.lambdify(rs, sympy.diff(dg, rs, k).subs({bs: b, a_s: r_a}), "numpy")
        out.append(np.asarray(fn(r) * np.ones_like(r), dtype=np.float64))
    return tuple(out)


def _dual_series(profile, bce: float, m_typ: str, r_a: float, theta: str, x_max: float):
    """`rho_aux_series` with each coefficient's derivative in theta carried as
    a dual part: [(s, C_s, d C_s, d s)]. The series of R(t) does not depend on
    theta; theta enters through ``q = 2 beta_c - gamma``, the factor's exponent
    ``1 - beta_c`` and ``R_a^-2``."""
    from .central import TOL_EXACT_BRANCH

    a, br, g = (
        float(profile.rho_alpha),
        float(profile.rho_beta),
        float(profile.rho_gamma),
    )
    cn = float(profile.rho_norm)
    tol = TOL_EXACT_BRANCH
    e = (g - br) / a
    cu = cn / ((2.0 - g) * (3.0 - g))
    xm, eps_x = float(x_max), 1e-9
    on_b = theta == "beta_c" and m_typ in ("constant", "COM_I", "COM_II")

    def key(x):
        return round(x, 9)

    def mul(p, r):
        out: dict[float, tuple[float, float]] = {}
        for x1, (v1, d1) in p.items():
            for x2, (v2, d2) in r.items():
                x = x1 + x2
                if x <= xm + eps_x:
                    v, d = out.get(key(x), (0.0, 0.0))
                    out[key(x)] = (v + v1 * v2, d + v1 * d2 + d1 * v2)
        return out

    def pow1p(s_, p, dp=0.0):
        out = {0.0: (1.0, 0.0)}
        term = {0.0: (1.0, 0.0)}
        coef, dcoef = 1.0, 0.0
        xmin = min((x for x, (c, _) in s_.items() if c != 0.0), default=None)
        if xmin is None:
            return out
        for k in range(1, int(np.floor(xm / xmin + eps_x)) + 1):
            coef, dcoef = (
                coef * (p - (k - 1)) / k,
                (dcoef * (p - (k - 1)) + coef * dp) / k,
            )
            term = mul(term, s_)
            for x, (c, dc) in term.items():
                v, d = out.get(x, (0.0, 0.0))
                out[x] = (v + coef * c, d + coef * dc + dcoef * c)
        return out

    kk = int(np.floor(xm / a + eps_x)) + 1
    bk = [1.0]
    for k in range(1, kk + 1):
        bk.append(bk[-1] * (e - (k - 1)) / k)
    dk = {
        key(k * a): (
            bk[k] * (2.0 - g) * (3.0 - g) / ((2.0 - g + k * a) * (3.0 - g + k * a)),
            0.0,
        )
        for k in range(1, kk + 1)
        if k * a <= xm + eps_x
    }
    s_r: dict[float, tuple[float, float]] = {}
    for _ in range(kk + 1):
        dt: dict[float, tuple[float, float]] = {}
        for x, d in dk.items():
            for xx, (cc, _) in mul({x: d}, pow1p(s_r, x)).items():
                v, _ = dt.get(xx, (0.0, 0.0))
                dt[xx] = (v + cc, 0.0)
        s_r = {x: c for x, c in pow1p(dt, -1.0 / (2.0 - g)).items() if x > 0.0}
    q = (2.0 * bce - g) if m_typ in ("constant", "COM_I", "COM_II") else -g
    dq = 2.0 if on_b else 0.0
    t = pow1p(s_r, q, dq)
    t = mul(t, pow1p(mul({key(a): (1.0, 0.0)}, pow1p(s_r, a)), e))
    if m_typ in ("OM_I", "OM_II", "COM_I", "COM_II"):
        sgn = 1.0 if m_typ in ("OM_I", "COM_I") else -1.0
        mexp = 1.0 if m_typ in ("OM_I", "OM_II") else (1.0 - bce)
        dmexp = -1.0 if (on_b and m_typ.startswith("COM")) else 0.0
        c2 = sgn / float(r_a) ** 2
        dc2 = -2.0 * c2 / float(r_a) if theta == "r_a" else 0.0
        r2 = mul({2.0: (c2, dc2)}, pow1p(s_r, 2.0))
        t = mul(t, pow1p(r2, mexp, dmexp))
    ds = dq / (2.0 - g)
    out: dict[float, tuple[float, float]] = {}
    for x, (c, dc) in t.items():
        s = (q + x) / (2.0 - g)
        s_key = next((k for k in out if abs(k - s) <= tol), s)
        scale = cn * cu ** (-s)
        v0, d0 = out.get(s_key, (0.0, 0.0))
        out[s_key] = (v0 + scale * c, d0 + scale * (dc - c * np.log(cu) * ds))
    return sorted((s, v, d, ds) for s, (v, d) in out.items() if v != 0.0)


def _center_jac(component: Component, theta: str) -> tuple[np.ndarray, np.ndarray]:
    """The stated center's numbers the lookup reads above the table (kappa,
    the amplitude, and where stated the next exponent and coefficient) and
    their derivatives in theta, in closed form (the regular branch)."""
    from math import factorial

    from scipy.special import gamma as gamma_function

    profile, aniso = component.unit_profile, component.anisotropy
    res = center_form(profile, aniso)
    m_typ = aniso.family if aniso.family != "isotropic" else "constant"
    on_b = theta == "beta_c" and m_typ in ("constant", "COM_I", "COM_II")
    bce = 0.0 if m_typ.startswith("OM") else float(aniso.beta_c)
    g = float(profile.rho_gamma)
    cu = float(profile.rho_norm) / ((2.0 - g) * (3.0 - g))
    nu, dnu = 1.5 - bce, (-1.0 if on_b else 0.0)
    dln_np = (np.log(2.0) + digamma(1.0 - bce)) if on_b else 0.0
    s0 = (2.0 * bce - g) / (2.0 - g)
    ds0 = 2.0 / (2.0 - g) if on_b else 0.0
    amp = float(res["amplitude"])
    dln_a = (
        dln_np - np.log(cu) * ds0 + digamma(nu - s0) * (dnu - ds0) + digamma(-s0) * ds0
    )
    vals = [float(res["kappa_sing"]), amp]
    jac = [ds0 - dnu, amp * dln_a]
    if res["next_form"] in ("power", "log_power"):
        s2 = float(res["next_s"])
        xm = 3.0 * max(float(profile.rho_alpha), 2.0)
        series = _dual_series(profile, bce, m_typ, float(aniso.r_a), theta, xm)
        _, c2, dc2, ds2 = next(t for t in series if abs(t[0] - s2) <= 1e-10)
        n_p = 2.0**bce / ((2.0 * np.pi) ** 1.5 * gamma_function(1.0 - bce))
        coef = float(res["next_coef"])
        if res["next_form"] == "power":
            per = n_p * _gamma_ratio(nu - s2, -s2)
            d_coef = per * dc2 + coef * (
                dln_np + digamma(nu - s2) * (dnu - ds2) + digamma(-s2) * ds2
            )
            d_kappa = ds2 - dnu
        else:
            kk = int(round(float(res["next_kappa"])))
            per = n_p * (-1.0) ** kk / (factorial(kk) * gamma_function(-s2))
            d_coef = per * dc2 + coef * (dln_np + digamma(-s2) * ds2)
            d_kappa = 0.0
        vals += [float(res["next_kappa"]), coef]
        jac += [d_kappa, float(d_coef)]
    return np.array(vals), np.array(jac, dtype=np.float64)


def _child_forward(component: Component, settings: Settings, held: dict[str, Any]):
    """The child's leaves at its own parameters on the grid and kept nodes
    held (``held``: ``q``, ``keep`` and the parent's inputs)."""
    tab = _tabulated()
    made = tab.child_rows(
        held["q"],
        held["parent_df"],
        held["parent_rows"],
        held["parent_r_a"],
        component.anisotropy,
        settings,
    )
    f = np.asarray(made["f"], dtype=np.float64)[held["keep"]]
    with np.errstate(divide="ignore"):
        out = dict(log_df=np.log10(np.abs(f)))
    law = made["floor"].get("law")
    if law:
        out["edge_law"] = np.array([law["p"], law["C0"], law["C1"]])
    out["center"] = _center_jac(component, "beta_c")[0]
    return out


def _child_key(b: float) -> tuple[int, bool, int]:
    r"""The child rows' form in :math:`\beta_{\rm c}`, as `_table_difference`
    reads a key: the kernel's order, whether a side has no central quotient (a
    half-integer kernel or a floor pair with n = 0), and the floor's order."""
    j, half = _form_key(b)
    m, n_zero = _floor_key(b)
    return j, half or n_zero, m


def _child_held(component: Component, settings: Settings) -> dict[str, Any]:
    """The child's solve at its own parameters (`tabulated_df`), and what the
    rule holds of it: the grid ``q``, the kept nodes ``keep`` and the
    parent's inputs."""
    tab = _tabulated()
    _, _, _, _, _, inversion, rows = tab.tabulated_df(component, settings)
    if inversion.critical_band:
        raise InputError(_critical(component, "critical"))
    parent_df, parent_rows, parent_r_a = tab._parent_inputs(component.profile)
    f_all = np.asarray(rows.f, dtype=np.float64)[0]
    return dict(
        q=np.asarray(rows.k, dtype=np.float64)[0],
        keep=np.isfinite(f_all),
        parent_df=parent_df,
        parent_rows=parent_rows,
        parent_r_a=parent_r_a,
    )


def _child_jacobian(
    component: Component,
    settings: Settings,
    held: dict[str, Any] | None = None,
    steps: dict[tuple, dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """`table_jacobian` for a component on a truncated profile (Method II):
    the parent held, the grid and the kept nodes held (``held``, by default
    the solve's at the component's own parameters)."""
    tab = _tabulated()
    family, params = _child_checked(component)
    aniso = component.anisotropy
    if held is None:
        held = _child_held(component, settings)
    q, keep = held["q"], held["keep"]
    parent_df, parent_rows, parent_r_a = (
        held["parent_df"],
        held["parent_rows"],
        held["parent_r_a"],
    )
    b = 0.0 if family.startswith("OM") else float(aniso.beta_c)
    r_a = float(aniso.r_a)
    _, _, j, order, _, _ = tab._pair(aniso)
    p_top = float(np.asarray(parent_rows["Psi"], dtype=np.float64)[0])
    density = tab.parent_density(q, p_top, order, parent_df, parent_rows, parent_r_a)
    src = tab.factor_sources(density, family, b, r_a)
    made = tab.child_rows(
        q, parent_df, parent_rows, parent_r_a, aniso, settings, sources=src
    )
    f = np.asarray(made["f"], dtype=np.float64)[keep]
    law_d = made["floor"].get("law")
    law = (law_d["p"], law_d["C0"], law_d["C1"]) if law_d else ()

    def rows_with(sources, anisotropy=aniso, s=settings):
        made = tab.child_rows(
            q, parent_df, parent_rows, parent_r_a, anisotropy, s, sources=sources
        )
        return np.asarray(made["f"], dtype=np.float64)

    with np.errstate(divide="ignore"):
        values: dict[str, np.ndarray] = dict(k=q[keep], log_df=np.log10(np.abs(f)))
    if law:
        values["edge_law"] = np.array(law, dtype=np.float64)
    values["center"], _ = _center_jac(component, params[0])
    read = sorted({order, j})
    jac: dict[str, dict[str, np.ndarray]] = {}
    record: dict[str, Any] = {"grid": _placement(component, settings, held), "rows": {}}
    for theta in params:
        dsrc = tab.factor_sources(
            density,
            family,
            b,
            r_a,
            factor=_dfactor(family, b, r_a, density["r"], theta),
        )
        # the operator's derivative along the sources' derivative
        pos = [k for k in read if np.all(src["aux"][k] > 0.0)]
        ratio = max(
            (float(np.max(np.abs(dsrc["aux"][k]) / src["aux"][k])) for k in pos),
            default=0.0,
        )
        e = 1e-6 / ratio if ratio > 0.0 else 1.0
        shifted = []
        for sign in (1.0, -1.0):
            s = dict(src)
            s["aux"] = {
                k: src["aux"][k] + sign * e * dsrc["aux"][k] for k in src["aux"]
            }
            s["g"] = src["g"] + sign * e * dsrc["g"]
            s["g_p"] = src["g_p"] + sign * e * dsrc["g_p"]
            shifted.append(rows_with(s))
        d_f = ((shifted[0] - shifted[1]) / (2.0 * e))[keep]
        on_b = theta == "beta_c" and not family.startswith("OM")
        if on_b:

            def t(bb, s=settings):
                c = _with(component, bb, aniso.r_a)
                return rows_with(src, c.anisotropy, s)[None, :]

            parent = component.profile
            assert isinstance(parent, TruncatedProfile)
            member = (
                "child rows",
                parent.model,
                family,
                aniso.kind,
                settings,
            )
            ex, record["rows"] = _table_difference(
                t, b, _child_key, member, lambda bb: t(bb, _finer(settings)), steps
            )
            d_f = d_f + ex[0][keep]
        out = dict(log_df=d_f / (f * _LN10))
        if law:
            # C_q is linear in a_q, itself linear in (g, g_p) at the floor node;
            # in beta_c, d ln C_q = ln 2 + psi(1 - b) - psi(q - 1/2 + b) at
            # held a_q, and p = 1 + beta_c
            d_law = tab.floor_record(q, dsrc, aniso)["law"]
            d_c = np.array([d_law["C0"], d_law["C1"]])
            if on_b:
                c_q = np.array(law[1:])
                d_c = d_c + c_q * (
                    np.log(2.0)
                    + digamma(1.0 - b)
                    - digamma(np.array([2.5, 3.5]) - 0.5 + b)
                )
            out["edge_law"] = np.array([1.0 if on_b else 0.0, *d_c])
        out["center"] = _center_jac(component, theta)[1]
        jac[theta] = out
    return values, jac, record


def _placement(component: Component, settings: Settings, held: dict[str, Any]):
    r"""Where the child's grid is placed: the family's zone center
    (:math:`\Psi(R_{\rm a})` for OM I, the pilot's feature for OM II and COM I)
    and its nodes."""
    tab = _tabulated()
    aniso, unit = component.anisotropy, component.unit_profile
    out: dict[str, Any] = dict(nodes=int(held["q"].size), kept=int(held["keep"].sum()))
    if aniso.family == "OM_I":
        out["psi_r_a"] = float(unit.p(np.array([float(aniso.r_a)]))[0])
    elif aniso.family in ("OM_II", "COM_I"):
        out["q_feature"] = float(
            tab.pilot_feature(
                held["parent_df"],
                held["parent_rows"],
                held["parent_r_a"],
                aniso,
                settings,
                unit,
            )["q_feature"]
        )
    return out


def table_jacobian(
    component: Component,
    settings: Settings,
    q: np.ndarray | None = None,
    steps: dict[tuple, dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Return the DF table's values and their derivatives in the component's
    anisotropy parameters, the forward's grid and discrete choices held.

    Parameters
    ----------
    component : Component
        An analytic Method I component.
    settings : Settings
    q : array, optional
        The grid; `inversion_nodes` where None.
    steps : dict, optional
        The caller's store of the exponent's measured steps
        (`_table_difference`): read where it has them, filled where it has
        not; without one they are measured at this call.

    Returns
    -------
    values : dict
        The table as `inversion_df` makes it, as arrays: ``k``, ``log_df``,
        ``log_df_slope`` (the rows; zero derivative where a Type II DF is
        outside its support), and where the table has them ``k_below``,
        ``log_df_below``, ``log_df_slope_below``; the form above the anchor
        ``central`` (kappa, a_anchor, b, d) and the tail ``tail`` (A, a, k).
    jac : dict
        ``{theta: {key: d value / d theta}}`` for the keys of values but the
        energies.
    record : dict
        The exponent's part of the rows (``"rows"``) and of the nodes below
        the floor (``"below"``), as `rows_jacobian` states it.

    Raises
    ------
    InputError
        As `rows_jacobian`; and where the form above the anchor is used and
        the center is not on its regular branch (the critical band).
    """
    if isinstance(component.profile, TruncatedProfile):
        if q is not None:
            raise InputError("A child on a truncated profile is read on its own grid.")
        return _child_jacobian(component, settings, steps=steps)
    inv = _invert()
    _checked(component)
    branch = center_form(component.unit_profile, component.anisotropy)["branch"]
    if branch == "critical_divergent" and component.anisotropy.kind != "II":
        raise InputError(_critical(component, branch))
    if q is None:
        q = inversion_nodes(component, settings)
    parts = _table_rows(component, settings, np.asarray(q, dtype=np.float64), steps)
    q, f, ddf = parts["q"], parts["f"], parts["ddf"]
    pieces = inv._table_pieces(
        component, settings, q, f, ddf, parts["aux"], parts["floor"]
    )
    table, patch, below, tail = (
        pieces["table"],
        pieces["patch"],
        pieces["below"],
        pieces["tail"],
    )
    center = pieces["center"]
    if patch is not None and center["branch"] != "regular":
        raise InputError(_critical(component, center["branch"]))
    k_t = np.asarray(table.k[0], np.float64)
    idx = np.searchsorted(q, k_t)
    inside = k_t > pieces["k_edge"]
    log_df = np.asarray(table.log_df[0], np.float64)
    slope = np.asarray(table.log_df_slope[0], np.float64)
    values: dict[str, np.ndarray] = dict(k=k_t, log_df=log_df, log_df_slope=slope)
    if below is not None:
        values.update(
            k_below=np.asarray(table.k_below[0], np.float64),
            log_df_below=np.asarray(table.log_df_below[0], np.float64),
            log_df_slope_below=np.asarray(table.log_df_slope_below[0], np.float64),
        )
    if patch is not None:
        values["central"] = np.array([patch.kappa, patch.a_anchor, patch.b, patch.d])
    if tail is not None:
        values["tail"] = np.array([tail["a"], tail["A"], tail["k"]])

    jac: dict[str, dict[str, np.ndarray]] = {}
    record: dict[str, Any] = {"rows": parts["record"]}
    for theta in parts["params"]:
        dfr, dddf = parts["jac"][theta]
        fi, dfi, ddi = f[idx], dfr[idx], dddf[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            d_log = np.where(inside, dfi / (fi * _LN10), 0.0)
            d_slope = np.where(inside, k_t * ddi / (fi * _LN10) - slope * dfi / fi, 0.0)
        out = dict(log_df=d_log, log_df_slope=d_slope)
        dfb: Any = None
        ddfb: Any = None
        if below is not None:
            dfb, ddfb, rec = _below_jvp(component, settings, parts, below, theta, steps)
            fb, kb = np.asarray(below["F"]), np.asarray(below["K"])
            sb = values["log_df_slope_below"]
            out["log_df_below"] = dfb / (fb * _LN10)
            out["log_df_slope_below"] = kb * ddfb / (fb * _LN10) - sb * dfb / fb
            if rec:
                record["below"] = rec
        if patch is not None:
            dkap, dln_a = _center_jvp(component, theta)
            amplitude = float(center["amplitude"])
            anc = q <= patch.k_anchor
            da, db_, dd = _patch_jvp(
                q[anc],
                f[anc],
                dfr[anc],
                patch.kappa,
                dkap,
                amplitude,
                amplitude * dln_a,
                patch,
            )
            out["central"] = np.array([dkap, da, db_, dd])
        if tail is not None:
            da_t = (
                0.0
                if theta != "beta_c"
                else (-1.0 if component.anisotropy.model in ("om", "com") else 1.0)
            )
            if below is not None:
                kb0 = float(below["K"][0])
                fb0, db0 = float(below["F"][0]), float(below["dFdK"][0])
                lnb = float(np.log(1.0 / kb0))
                k_raw = (kb0 * db0 / fb0 + tail["a"]) * lnb
                dk = (
                    (kb0 * (ddfb[0] / fb0 - db0 * dfb[0] / fb0**2) + da_t) * lnb
                    if np.isfinite(k_raw)
                    else 0.0
                )
                d_amp = tail["A"] * (
                    dfb[0] / fb0 + np.log(kb0) * da_t + np.log(lnb) * dk
                )
            else:
                k_w = float(pieces["k_written"])
                i_w = int(np.flatnonzero(q == k_w)[0])
                dk = 0.0
                d_amp = dfr[i_w] * k_w ** tail["a"] + tail["A"] * np.log(k_w) * da_t
            out["tail"] = np.array([da_t, float(d_amp), float(dk)])
        jac[theta] = out
    return values, jac, record


def _at(component: Component, params: tuple[str, ...]) -> Any:
    a0 = component.anisotropy

    def at(theta):
        v = dict(zip(params, np.asarray(theta, dtype=np.float64).tolist(), strict=True))
        return _with(component, v.get("beta_c", a0.beta_c), v.get("r_a", a0.r_a))

    return at


def make_inversion_table(
    component: Component, settings: Settings, q: np.ndarray | None = None
) -> tuple[Any, tuple[str, ...], dict[str, np.ndarray]]:
    """Return ``(table, params, values)``: ``table(theta)`` is the DF table's
    values (`table_jacobian`'s, but the energies) as a JAX function of the
    component's anisotropy parameters with a custom VJP; ``values`` the
    table at the component's own parameters, energies included. The forward
    is `inversion_df`'s table through a host callback, the backward the rule.
    The grid and the forward's discrete choices are those at the component's
    parameters."""
    import jax
    import jax.numpy as jnp

    inv = _invert()
    if isinstance(component.profile, TruncatedProfile):
        return _make_child_table(component, settings)
    _, params = _checked(component)
    if q is None:
        q = inversion_nodes(component, settings)
    q = np.asarray(q, dtype=np.float64)
    # the exponent's steps, measured here at the component's own parameters
    # and carried to every theta, so a gradient never depends on the calls before
    steps: dict[tuple, dict[str, Any]] = {}
    values, _, _ = table_jacobian(component, settings, q, steps)
    keys = [k for k in values if not k.startswith("k")]
    at = _at(component, params)

    def host_f(theta):
        c = at(theta)
        pair = potential_density_pair((c,))
        f, ddf, _, aux, floor = inv._rows(c, settings, pair, q)
        f, ddf = np.asarray(f, np.float64), np.asarray(ddf, np.float64)
        pieces = inv._table_pieces(c, settings, q, f, ddf, aux, floor)
        table, patch, tail = pieces["table"], pieces["patch"], pieces["tail"]
        out = dict(
            log_df=np.asarray(table.log_df[0], np.float64),
            log_df_slope=np.asarray(table.log_df_slope[0], np.float64),
        )
        if "log_df_below" in keys:
            out.update(
                log_df_below=np.asarray(table.log_df_below[0], np.float64),
                log_df_slope_below=np.asarray(table.log_df_slope_below[0], np.float64),
            )
        if "central" in keys:
            out["central"] = np.array([patch.kappa, patch.a_anchor, patch.b, patch.d])
        if "tail" in keys:
            out["tail"] = np.array([tail["a"], tail["A"], tail["k"]])
        return {k: out[k] for k in keys}

    def host_j(theta):
        _, jac, _ = table_jacobian(at(theta), settings, q, steps)
        return {k: np.stack([jac[p][k] for p in params]) for k in keys}

    shape_f = {k: jax.ShapeDtypeStruct(values[k].shape, jnp.float64) for k in keys}
    shape_j = {
        k: jax.ShapeDtypeStruct((len(params), *values[k].shape), jnp.float64)
        for k in keys
    }

    @jax.custom_vjp
    def table(theta):
        return jax.pure_callback(host_f, shape_f, theta)

    def fwd(theta):
        return table(theta), theta

    def bwd(theta, v):
        jac = jax.pure_callback(host_j, shape_j, theta)
        return (sum(jac[k] @ v[k] for k in keys),)

    table.defvjp(fwd, bwd)
    return table, params, values


def _make_child_table(
    component: Component, settings: Settings
) -> tuple[Any, tuple[str, ...], dict[str, np.ndarray]]:
    """`make_inversion_table` for a child on a truncated profile: the grid
    and the kept nodes held at the component's own parameters."""
    import jax
    import jax.numpy as jnp

    _, params = _child_checked(component)
    held = _child_held(component, settings)
    # the steps measured at the component's own parameters (make_inversion_table)
    steps: dict[tuple, dict[str, Any]] = {}
    values, _, _ = _child_jacobian(component, settings, held, steps)
    keys = [k for k in values if k != "k"]
    at = _at(component, params)

    def host_f(theta):
        out = _child_forward(at(theta), settings, held)
        return {k: np.asarray(out[k], np.float64) for k in keys}

    def host_j(theta):
        _, jac, _ = _child_jacobian(at(theta), settings, held, steps)
        return {k: np.stack([jac[p][k] for p in params]) for k in keys}

    shape_f = {k: jax.ShapeDtypeStruct(values[k].shape, jnp.float64) for k in keys}
    shape_j = {
        k: jax.ShapeDtypeStruct((len(params), *values[k].shape), jnp.float64)
        for k in keys
    }

    @jax.custom_vjp
    def table(theta):
        return jax.pure_callback(host_f, shape_f, theta)

    def fwd(theta):
        return table(theta), theta

    def bwd(theta, v):
        jac = jax.pure_callback(host_j, shape_j, theta)
        return (sum(jac[k] @ v[k] for k in keys),)

    table.defvjp(fwd, bwd)
    return table, params, values
