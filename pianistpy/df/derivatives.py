"""The derivatives of the auxiliary density with respect to the potential.

rho_aux(R) = rho(R) g(R), with g = R^(2 beta_c) times the (C)OM factor. The
derivatives with respect to P are the operator D_P = (dP/dR)^-1 d/dR applied
N times, with dP/dR = -M(R)/R^2 and M'(R) = R^2 rho(R), so that P(R) is never
needed in closed form (Notes Section II.2 [num-pair], Notes Eq. (II.1)
[dp-operator]). The operator is applied symbolically (sympy) and the results
are evaluated in float64, and in mpmath where float64 cancels: R < 1e-2,
R > 1e6, order >= 3 beyond R = 1e2, and order >= 2 within the band about R_a
of a Type II model. M(R) is the incomplete-Beta mass, evaluated once per
point. R(P) is found by bisection in ln R with a widened bracket and, near
the ceiling, by bisection on the profile's subtraction-free depth.

The arithmetic is the working pipeline's, expression for expression, so that
its values are equal to the bit on the same backend.
"""

__all__ = ["ExactDerivs", "aux_derivatives"]

from typing import Any

import numpy as np

from ..profiles import DoublePowerLaw
from ..typing import Array
from .stages import AuxDerivatives, EnergyGrid, InversionForm, PotentialDensityPair

CONSTRUCTION_CACHE = "derivatives"


def _construction_name(key: tuple) -> str:
    """The cache file of one construction, named by the digest of its exact
    inputs, the sympy version and the construction's own source: a change to
    any of them is another file, so a stale entry cannot change a number."""
    import hashlib
    import inspect

    import sympy as sp

    text = repr((key, sp.__version__, inspect.getsource(ExactDerivs.__init__)))
    return hashlib.sha256(text.encode()).hexdigest() + ".json"


def _load_construction(key: tuple) -> list[Any] | None:
    """The derivatives' expressions of a construction made before, or None
    (the file is the runtime's: `pianistpy.runtime.read_cache`)."""
    import json

    import sympy as sp

    from ..runtime import read_cache

    text = read_cache(CONSTRUCTION_CACHE, _construction_name(key))
    try:
        stored = json.loads(text) if text is not None else None
    except ValueError:
        return None
    if not isinstance(stored, dict) or stored.get("key") != repr(key):
        return None
    return [sp.sympify(e) for e in stored["exprs"]]


def _save_construction(key: tuple, exprs: list[Any]) -> None:
    """Keep the expressions for the next construction with the same inputs
    (`pianistpy.runtime.write_cache`)."""
    import json

    import sympy as sp

    from ..runtime import write_cache

    text = json.dumps({"key": repr(key), "exprs": [sp.srepr(e) for e in exprs]})
    write_cache(CONSTRUCTION_CACHE, _construction_name(key), text)


def _lambdify_deep(r: Any, expr: Any, modules: str, chunk: int = 64) -> Any:
    """``sympy.lambdify``, split into nested callables only where the generated
    source is too deeply nested for Python's compiler; the flat form wherever
    it compiles."""
    import sympy as sp

    try:
        return sp.lambdify(r, expr, modules)
    except RecursionError:
        pass
    args = list(expr.args)
    if expr.func in (sp.Add, sp.Mul) and len(args) > chunk:
        args = [expr.func(*args[i : i + chunk]) for i in range(0, len(args), chunk)]
    if len(args) == 0:
        raise RecursionError(
            f"ExactDerivs: cannot split the expression node {expr.func}"
        )
    kids = [_lambdify_deep(r, a, modules, chunk) for a in args]
    dums = sp.symbols(f"_d0:{len(args)}")
    op = sp.lambdify(
        dums,
        expr.func(*dums, evaluate=False)
        if expr.func in (sp.Add, sp.Mul)
        else expr.func(*dums),
        modules,
    )
    return lambda *a: op(*[k(*a) for k in kids])


class ExactDerivs:
    """The exact derivatives d^N rho_aux / dP^N of an analytic double power law.

    Parameters
    ----------
    profile : DoublePowerLaw
        The density.
    beta_c : float
        The central anisotropy in the factor R^(2 beta_c) (0 for OM).
    model : str
        The anisotropy family: ``"isotropic"``, ``"constant"``, ``"OM_I"``,
        ``"OM_II"``, ``"COM_I"`` or ``"COM_II"``.
    r_a : float
        The anisotropy radius of an (C)OM model.
    max_order : int
        The highest order built.
    """

    R_SMALL = 1e-2
    TYPEII_MP_BAND = 0.75

    def __init__(
        self,
        profile: DoublePowerLaw,
        beta_c: float = 0.0,
        model: str = "constant",
        r_a: float = 1.0,
        max_order: int = 4,
    ) -> None:
        import mpmath as mp
        import sympy as sp

        self.prof = profile
        a, b, g = profile.rho_alpha, profile.rho_beta, profile.rho_gamma
        c_norm = float(profile.rho_norm)
        bc = float(beta_c)
        R = sp.symbols("R", positive=True)
        C = (
            sp.nsimplify(c_norm)
            if abs(c_norm - round(c_norm * 8) / 8) < 1e-12
            else sp.Float(c_norm, 30)
        )
        # An exponent whose exact binary value is a small fraction enters as
        # that fraction; any other enters as a symbol whose float value is
        # given at evaluation. None is rounded to a nearby fraction, so the
        # closed form is the model asked for.
        syms: list[Any] = []
        vals: list[float] = []

        def exponent(x: float, name: str, **assume: bool) -> Any:
            q = sp.Rational(x)
            if q.q <= 1000:
                return q
            s = sp.Symbol(name, **assume)
            syms.append(s)
            vals.append(float(x))
            return s

        ea = exponent(a, "alpha", positive=True)
        eb = exponent(b, "beta", positive=True)
        eg = exponent(g, "gamma", positive=True)
        bc_s = exponent(bc, "beta_c", real=True)
        rho = C * R ** (-eg) * (1 + R**ea) ** ((eg - eb) / ea)
        gfac = R ** (2 * bc_s)
        ra_s = sp.Rational(str(r_a))
        if model == "OM_I":
            gfac = gfac * (1 + R**2 / ra_s**2)
        if model == "OM_II":
            gfac = gfac * (1 - R**2 / ra_s**2)
        if model == "COM_I":
            gfac = gfac * (1 + R**2 / ra_s**2) ** (1 - bc_s)
        if model == "COM_II":
            gfac = (
                gfac
                * (1 - R**2 / ra_s**2)
                * sp.exp(-bc_s * sp.log((1 - R**2 / ra_s**2) ** 2) / 2)
            )
        rho_aux = rho * gfac
        m_sym = sp.Function("M")(R)
        dpdr = -m_sym / R**2
        sub = {sp.Derivative(m_sym, R): R**2 * rho}
        a_mp, b_mp, g_mp, c_mp = map(mp.mpf, (a, b, g, c_norm))
        a1, b1 = (3 - g_mp) / a_mp, (b_mp - 3) / a_mp

        def m_mp(rv):
            rv = mp.mpf(rv)
            ra = rv**a_mp
            eps = 1 / (1 + ra)
            if eps >= mp.mpf("1e-10"):
                return c_mp / a_mp * mp.betainc(a1, b1, 0, ra / (1 + ra))
            if b1 > 0:
                return c_mp / a_mp * (mp.beta(a1, b1) - mp.betainc(b1, a1, 0, eps))
            return (
                c_mp
                / a_mp
                * (
                    -mp.log(eps)
                    - mp.digamma(a1)
                    - mp.euler
                    + (a1 - 1) * eps
                    - (a1 - 1) * (a1 - 2) * eps**2 / 4
                    + (a1 - 1) * (a1 - 2) * (a1 - 3) * eps**3 / 18
                )
            )

        memo: dict[Any, Any] = {}

        def m_mp_once(rv):
            key = (rv, mp.mp.dps)
            if key not in memo:
                memo[key] = m_mp(rv)
            return memo[key]

        def m_np(rv):
            if np.ndim(rv):
                return np.asarray(profile.mass(rv), dtype=np.float64).reshape(
                    np.shape(rv)
                )
            return float(np.atleast_1d(profile.mass(rv))[0])

        mv = sp.Symbol("Mv")
        vals_mp = [mp.mpf(v) for v in vals]
        f0 = sp.lambdify((R, *syms), rho_aux, "numpy")
        f0_mp = sp.lambdify((R, *syms), rho_aux, "mpmath")
        self.funcs: list[Any] = [lambda rv: f0(rv, *vals)]
        self.funcs_mp: list[Any] = [lambda rv: f0_mp(rv, *vals_mp)]
        key = (a, b, g, c_norm, bc, model, float(r_a), int(max_order))
        exprs = _load_construction(key)
        if exprs is None:
            exprs = []
            expr = rho_aux
            for _ in range(1, max_order + 1):
                expr = sp.cancel((sp.diff(expr, R) / dpdr).subs(sub).doit())
                expr = expr.replace(sp.DiracDelta, lambda *args: sp.Integer(0))
                exprs.append(expr.xreplace({m_sym: mv}))
            _save_construction(key, exprs)
        # The arbitrary-precision orders are one evaluation per node and
        # precision, their common subexpressions computed once (at rounding
        # against 60 more digits where it is not to the bit); the orders one
        # at a time where sympy cannot do it.
        try:
            joint = sp.lambdify((R, mv, *syms), exprs, "mpmath", cse=True)
        except (TypeError, RecursionError):
            joint = None
        memo_joint: dict[Any, Any] = {}

        def joint_once(rv):
            key = (rv, mp.mp.dps)
            if key not in memo_joint:
                memo_joint[key] = joint(rv, m_mp_once(rv), *vals_mp)
            return memo_joint[key]

        for i, e_m in enumerate(exprs):
            fn = _lambdify_deep((R, mv, *syms), e_m, "numpy")
            self.funcs.append(lambda rv, _f=fn: _f(rv, m_np(rv), *vals))
            if joint is not None:
                self.funcs_mp.append(lambda rv, _i=i: joint_once(rv)[_i])
                continue
            fm = _lambdify_deep((R, mv, *syms), e_m, "mpmath")
            self.funcs_mp.append(lambda rv, _f=fm: _f(rv, m_mp_once(rv), *vals_mp))
        self._mp = mp
        self._typeII = model in ("OM_II", "COM_II")
        self._ra = float(r_a)

    def _pot(self, rv: Any) -> Array:
        return np.asarray(self.prof.p(rv), dtype=np.float64)

    def _dpdr(self, rv: Any) -> Array:
        return (
            -np.asarray(self.prof.mass(rv), dtype=np.float64)
            / np.asarray(rv, dtype=np.float64) ** 2
        )

    def _eval(self, n: int, rv: Any) -> Array:
        rv = np.atleast_1d(rv)
        out = np.asarray(self.funcs[n](rv), dtype=np.float64)
        small = rv < self.R_SMALL
        bad = (
            small
            | ~np.isfinite(out)
            | ((n >= 3) & (rv > 1e2))
            | ((n >= 1) & (rv > 1e6))
        )
        if self._typeII and n >= 2:
            bad = bad | (np.abs(1.0 - (rv / self._ra) ** 2) < self.TYPEII_MP_BAND)
        if bad.any():
            res = []
            try:
                for r in rv[bad]:
                    self._mp.mp.dps = int(
                        30 + 1.5 * max(0.0, np.log10(max(float(r), 1.0)))
                    )
                    if self._typeII and n >= 2:
                        u = abs(1.0 - (float(r) / self._ra) ** 2)
                        if u < self.TYPEII_MP_BAND:
                            self._mp.mp.dps += (
                                int(np.ceil(3 * n * np.log10(1.0 / max(u, 1e-30))))
                                + 2 * n
                            )
                    v = self.funcs_mp[n](self._mp.mpf(float(r)))
                    if isinstance(v, self._mp.mpc):
                        if abs(v.imag) <= 1e-20 * max(abs(v.real), 1e-300):
                            v = v.real
                        else:
                            raise FloatingPointError(
                                "ExactDerivs: complex derivative "
                                f"(N={n}) at R = {float(r)}"
                            )
                    res.append(float(v))
            finally:
                self._mp.mp.dps = 30
            out[bad] = res
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(
                f"ExactDerivs: non-finite derivative (N={n}) at R = "
                f"{rv[~np.isfinite(out)][:3]}"
            )
        return out

    def r_of_p(self, pq: Any, r_lo: float = 1e-12, r_hi: float = 1e40) -> Array:
        """Return the radius where the potential is ``pq``."""
        pq = np.asarray(pq, dtype=np.float64)
        p_max = float(np.max(pq)) if pq.size else 0.0
        while (
            float(self._pot(np.asarray([r_lo])).ravel()[0]) <= p_max and r_lo > 1e-290
        ):
            r_lo *= 1e-12
        p_min = float(np.min(pq)) if pq.size else np.inf
        while float(self._pot(np.asarray([r_hi])).ravel()[0]) >= p_min and r_hi < 1e290:
            r_hi *= 1e12
        if pq.size and not (float(self._pot(np.asarray([r_hi])).ravel()[0]) < p_min):
            raise ValueError(
                f"The potential does not fall below P = {p_min:.3e} by R = {r_hi:.1e}: "
                "the inversion cannot bracket it."
            )
        lo, hi = np.full_like(pq, np.log(r_lo)), np.full_like(pq, np.log(r_hi))
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            pm = np.asarray(self.prof.p(np.exp(mid)))
            hi = np.where(pm < pq, mid, hi)
            lo = np.where(pm >= pq, mid, lo)
        rv = np.exp(0.5 * (lo + hi))
        for _ in range(3):
            d = np.asarray(self._dpdr(rv), dtype=np.float64)
            step = (np.asarray(self.prof.p(rv)) - pq) / np.where(d != 0, d, np.nan)
            rv = np.where(np.isfinite(step) & (np.abs(step) < 0.5 * rv), rv - step, rv)
        uq = 1.0 - pq
        nt = 8
        _, bnd = self.prof.depth_inner(rv, terms=nt)
        deep = (uq > 0.0) & np.isfinite(bnd) & (bnd < np.finfo(np.float64).eps)
        if np.any(deep):
            rd = rv[deep]
            ud = uq[deep]
            lo_d, hi_d = np.log(rd) - 1.0, np.log(rd) + 1.0
            for _ in range(80):
                mid = 0.5 * (lo_d + hi_d)
                um = self.prof.depth_inner(np.exp(mid), terms=nt)[0]
                hi_d = np.where(um > ud, mid, hi_d)
                lo_d = np.where(um <= ud, mid, lo_d)
            rd = np.exp(0.5 * (lo_d + hi_d))
            for _ in range(2):
                um = self.prof.depth_inner(rd, terms=nt)[0]
                dudr = -np.asarray(self._dpdr(rd), dtype=np.float64)
                rd = rd - (um - ud) / dudr
            rv = rv.copy()
            rv[deep] = rd
        return rv

    def derivs_at_p(self, pq: Any, n: int) -> tuple[Array, Array, Array]:
        """Return (rho_aux, d rho_aux/dP, d^N rho_aux/dP^N) at the potentials ``pq``."""
        rv = self.r_of_p(pq)
        return self._eval(0, rv), self._eval(1, rv), self._eval(n, rv)


def aux_derivatives(
    pair: PotentialDensityPair, grid: EnergyGrid, form: InversionForm
) -> AuxDerivatives:
    """Return the exact derivatives of each auxiliary density on the grid.

    The orders held are j - 1 to j + 2 of each component's pair in use (the
    closure, the source of the derivative-outside rows, the integrated-by-
    parts source and its slope). `AuxDerivatives.at` evaluates any order at
    any energy from the same closed forms.

    Parameters
    ----------
    pair : PotentialDensityPair
        An analytic pair of one component.
    grid : EnergyGrid
        The energies.
    form : InversionForm
        The pair in use, j = ``form.order_m``.

    Returns
    -------
    derivatives : AuxDerivatives
    """
    from ..exceptions import InputError, ResultError

    if len(pair.components) != 1 or not pair.is_analytic:
        raise InputError(
            "This release takes the exact derivatives of one analytic component."
        )
    component = pair.components[0]
    profile = component.unit_profile
    if not isinstance(profile, DoublePowerLaw):
        raise InputError("An analytic pair's component has a double power law.")
    anisotropy = component.anisotropy
    j = int(form.order_m[0])
    orders = tuple(range(max(j - 1, 0), j + 3))
    source = ExactDerivs(
        profile,
        beta_c=0.0 if anisotropy.model == "om" else float(anisotropy.beta_c),
        model=anisotropy.family,
        r_a=float(anisotropy.r_a) if np.isfinite(anisotropy.r_a) else 1.0,
        max_order=max(2, orders[-1]),
    )
    k = np.asarray(grid.k, dtype=np.float64)[0]

    def every_order(p: Array) -> Array:
        # R(P) once for the energies, each order once on it (0 outside (0, 1),
        # as AuxDerivatives.at). The floor is its own call: the bracket of the
        # bisection is set by the energies asked for.
        out = np.zeros((len(orders), p.size))
        ok = (p > 0.0) & (p < 1.0)
        if ok.any():
            rv = source.r_of_p(p[ok])
            for i, o in enumerate(orders):
                out[i, ok] = source._eval(o, rv)
        return out

    values = every_order(k)[None, :, :]
    try:
        at_floor = every_order(k[:1])[:, 0][None, :]
    except FloatingPointError as err:
        raise ResultError(
            f"The exact derivatives at the floor are not finite ({err}): the "
            "boundary term would have taken the table's extrapolated value instead."
        ) from err
    return AuxDerivatives(orders, values, at_floor, True, source)
