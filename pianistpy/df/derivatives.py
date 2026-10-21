r"""The exact derivatives of the auxiliary density in the relative potential.

The auxiliary density is rho_aux(R) = rho(R) g(R), with rho the density and
``g = R^(2 beta_c)`` times the inverse (C)OM anisotropy factor. The N-th
derivative of rho_aux in the relative potential P is the operator
D_P = (dP/dR)^-1 d/dR applied N times, with dP/dR = -M(R)/R^2 and
M'(R) = R^2 rho(R), so that P(R) is never needed in closed form (Notes
Section T:1.2, Notes Eq. (T:1.3)). Sympy applies the operator symbolically,
and the resulting expressions are evaluated in float64, and in mpmath where
float64 loses digits to cancellation: at :math:`R < 10^{-2}`, for orders 1
and above at :math:`R > 10^{6}`, for orders 3 and above at
:math:`R > 10^{2}`, for orders 2 and above where
:math:`|1 - (R/R_{\rm a})^2| < 0.75` in a Type II model, and wherever the
float64 value is not finite. The mpmath mass M(R) is the incomplete-Beta
mass; `ExactDerivs` computes the mpmath mass once per radius and
precision. The radius R(P) is found by bisection in ln R, with a bracket
widened until the bracket holds P, and, near the center, by bisection on
the profile's subtraction-free depth u = 1 - P.
"""

__all__ = [
    "ExactDerivs",
    "ParameterDerivs",
    "aux_derivatives",
    "host_reachable_radius",
    "source_rows",
]

from functools import cache
from typing import Any

import numpy as np

from ..anisotropy.factor import inverse_factor_expr
from ..exceptions import InputError
from ..profiles import DoublePowerLaw
from ..typing import Array
from .stages import AuxDerivatives, EnergyGrid, InversionForm, PotentialDensityPair

CONSTRUCTION_CACHE = "derivatives"


@cache
def host_reachable_radius() -> float:
    """Return the smallest radius r = 2^-k whose square is positive in host
    float64 (NumPy), r*r > 0. With IEEE subnormals the radius is 2^-537."""
    r = np.ldexp(1.0, -np.arange(1100))
    return float(r[r * r > 0.0][-1])


def _construction_name(key: tuple) -> str:
    """Return the name of the cache file of one construction of derivative
    expressions.

    The name is the SHA-256 digest of the construction's exact inputs
    ``key``, the sympy version, the package version, the source of
    `ExactDerivs.__init__` and the source of `inverse_factor_expr`. A change
    to any of these inputs gives another file name, so a stale cache entry
    cannot change a number."""
    import hashlib
    import inspect

    import sympy as sp

    from .._version import __version__

    text = repr(
        (
            key,
            sp.__version__,
            __version__,
            inspect.getsource(ExactDerivs.__init__),
            inspect.getsource(inverse_factor_expr),
        )
    )
    return hashlib.sha256(text.encode()).hexdigest() + ".json"


def _load_construction(key: tuple) -> list[Any] | None:
    """Return the derivative expressions that an earlier construction with
    key ``key`` kept in the cache (`pianistpy.runtime.read_cache`), or
    None where the cache holds no readable entry for ``key``."""
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
    """Keep the derivative expressions ``exprs`` in the cache for a later
    construction with key ``key`` (`pianistpy.runtime.write_cache`)."""
    import json

    import sympy as sp

    from ..runtime import write_cache

    text = json.dumps({"key": repr(key), "exprs": [sp.srepr(e) for e in exprs]})
    write_cache(CONSTRUCTION_CACHE, _construction_name(key), text)


def _lambdify_deep(r: Any, expr: Any, modules: str, chunk: int = 64) -> Any:
    """Return ``sympy.lambdify(r, expr, modules)``, or, where the generated
    source is too deeply nested for Python's compiler, a nest of callables
    built from the arguments of ``expr``. A sum or product with more than
    ``chunk`` terms is first split into sums or products of at most
    ``chunk`` terms."""
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
    """The exact derivatives d^N rho_aux / dP^N of the auxiliary density of an
    analytic double power law, for N from 0 to ``max_order``.

    Parameters
    ----------
    profile : DoublePowerLaw
        The density, in internal units.
    beta_c : float, optional
        Central anisotropy in the factor ``R^(2 beta_c)``; 0 for an
        Osipkov-Merritt model. Default: ``0.0``.
    model : str, optional
        Anisotropy family: ``"isotropic"``, ``"constant"``, ``"OM_I"``,
        ``"OM_II"``, ``"COM_I"`` or ``"COM_II"``. Default: ``"constant"``.
    r_a : float, optional
        Anisotropy radius of an OM or COM model, in units of r_0. Default:
        ``1.0``.
    max_order : int, optional
        Highest order N built. Default: ``4``.
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
        # An exponent whose exact binary value is a fraction with denominator
        # at most 1000 enters the expressions as that fraction. Any other
        # exponent enters as a symbol whose float value is given at
        # evaluation. No exponent is rounded to a nearby fraction, so the
        # closed form is the closed form of the requested model.
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
        ra_s = sp.Rational(str(r_a))
        # The inversion of an untruncated Type II model reads rho_aux beyond
        # R_a, so the factor takes its continuation beyond R_a
        # (continuation=True).
        gfac = inverse_factor_expr(R, model, bc_s, ra_s, continuation=True)
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
        # The mpmath functions of all orders are one joint lambdify with common
        # subexpressions, evaluated once per radius and precision and memoized.
        # Where sympy cannot build the joint function, each order gets its
        # own mpmath function.
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
        # The radii in ``bad`` always go to mpmath. The float64 expression is
        # evaluated on the other radii only, so that the float64 expression
        # never overflows at a radius whose float64 value would be discarded.
        bad = (rv < self.R_SMALL) | ((n >= 3) & (rv > 1e2)) | ((n >= 1) & (rv > 1e6))
        if self._typeII and n >= 2:
            bad = bad | (np.abs(1.0 - (rv / self._ra) ** 2) < self.TYPEII_MP_BAND)
        out = np.full(rv.shape, np.nan)
        if (~bad).any():
            out[~bad] = np.asarray(self.funcs[n](rv[~bad]), dtype=np.float64)
        bad = bad | ~np.isfinite(out)
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
        """Return the radii at which the profile's relative potential equals
        ``pq``.

        The solve runs in host float64 (NumPy), so the smallest radius the
        solve reaches is `host_reachable_radius` (2^-537). Below
        `host_reachable_radius`, R*R rounds to 0 and the slope
        dP/dR = -M/R^2 has no value. JAX's CPU backend flushes subnormals to
        0, so a JAX version of `r_of_p` would reach only R = 2^-511.

        Parameters
        ----------
        pq : array-like
            Relative potentials P, in internal units.
        r_lo : float, optional
            Initial lower end of the bisection bracket in R; divided by
            :math:`10^{12}` until the potential at `r_lo` exceeds the largest
            P or `r_lo` is below :math:`10^{-290}`. Default: ``1e-12``.
        r_hi : float, optional
            Initial upper end of the bisection bracket in R; multiplied by
            :math:`10^{12}` until the potential at `r_hi` is below the
            smallest P or `r_hi` reaches :math:`10^{290}`. Default:
            ``1e40``.

        Returns
        -------
        r : ndarray
            Radii, with the shape of ``pq``, in internal units.

        Raises
        ------
        InputError
            If the smallest requested depth u = 1 - P lies below the depth at
            `host_reachable_radius`, so that the depth needs a radius that
            host float64 does not reach.
        ValueError
            If the potential is still at or above the smallest requested P
            once the bracket's upper end reaches :math:`10^{290}`.
        """
        pq = np.asarray(pq, dtype=np.float64)
        uq_all = 1.0 - pq
        if pq.size and bool(np.any(uq_all > 0.0)):
            r_floor = host_reachable_radius()
            u_floor = float(self.prof.depth_inner(np.array([r_floor]), terms=8)[0][0])
            u_top = float(np.min(uq_all[uq_all > 0.0]))
            if u_top < u_floor:
                raise InputError(
                    f"The depth u = {u_top:.6e} (the top asked for) needs a radius "
                    f"below R = {r_floor:.6e} = 2^-537, the smallest radius whose "
                    f"square host float64 holds (depth u = {u_floor:.6e} there): "
                    "the table's top must not lie deeper than that depth."
                )
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
        """Return rho_aux, d rho_aux/dP and d^n rho_aux/dP^n at the relative
        potentials ``pq``."""
        rv = self.r_of_p(pq)
        return self._eval(0, rv), self._eval(1, rv), self._eval(n, rv)


def _r_of_p(profile: DoublePowerLaw, pq: Any) -> Array:
    """Return the radii at which the profile's relative potential equals
    ``pq``, by `ExactDerivs.r_of_p`. `ExactDerivs.r_of_p` reads only the
    profile, so the derivative expressions are not built."""
    exact = ExactDerivs.__new__(ExactDerivs)
    exact.prof = profile
    return exact.r_of_p(pq)


def aux_derivatives(
    pair: PotentialDensityPair, grid: EnergyGrid, form: InversionForm
) -> AuxDerivatives:
    """Return the exact derivatives of the component's auxiliary density at
    the energies of the grid.

    The orders held are j - 1 to j + 2 of the order pair in use, with j - 1
    not below 0: the closure (j - 1), the source of the rows with the
    derivative outside the integral (j), the source of the form integrated
    by parts (j + 1) and the slope of the source of order j + 1 (j + 2).
    `AuxDerivatives.at` evaluates any order at any energy from the closed
    forms of the returned `ExactDerivs`.

    Parameters
    ----------
    pair : PotentialDensityPair
        Analytic potential-density pair of one component.
    grid : EnergyGrid
        Energies of the inversion, in internal units.
    form : InversionForm
        Form of the inversion, whose ``order_j`` gives the order j.

    Returns
    -------
    derivatives : AuxDerivatives
        Values of each order at the grid's energies and at the grid's first
        energy, 0 at energies outside (0, 1), with the `ExactDerivs` that
        made the values.

    Raises
    ------
    InputError
        If `pair` is not an analytic pair of one component, or the
        component's profile is not a `DoublePowerLaw`.
    ResultError
        If a derivative at the grid's first energy is not finite.
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
    j = int(form.order_j[0])
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
        # R(P) is solved once for the energies p, and each order is evaluated
        # once at the solved radii. An order is 0 at energies outside (0, 1),
        # as in AuxDerivatives.at. The floor gets its own call because the
        # bisection bracket of r_of_p depends on the energies requested.
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


class ParameterDerivs(ExactDerivs):
    r"""The mixed derivatives d/dtheta d^N rho_aux / dP^N at fixed P of the
    auxiliary density of an analytic double power law, for the parameter
    :math:`\theta` equal to :math:`\beta_{\rm c}` or :math:`R_{\rm a}`.

    At a fixed profile, R(P) is fixed and the operator
    D_P = (dP/dR)^-1 d/dR does not depend on :math:`\theta`, so d/dtheta
    commutes with D_P. Each order's expression, built as in `ExactDerivs`
    but with :math:`\beta_{\rm c}` and :math:`R_{\rm a}` as symbols, is
    therefore differentiated in :math:`\theta`. The expressions are
    evaluated, and R(P) is found, as in `ExactDerivs`: ``_eval(N, R)`` is
    d/dtheta of order N, and ``derivs_at_p(pq, n)`` returns d/dtheta of
    orders 0, 1 and n.

    Parameters
    ----------
    profile : DoublePowerLaw
        The density, in internal units.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`.
    model : str
        Anisotropy family, as in `ExactDerivs`.
    r_a : float
        Anisotropy radius :math:`R_{\rm a}`, in units of r_0.
    max_order : int
        Highest order N built.
    wrt : {"beta_c", "r_a"}
        Parameter theta of the derivative.

    Raises
    ------
    InputError
        If `wrt` is neither ``"beta_c"`` nor ``"r_a"``, or is a parameter
        that the family does not have: :math:`\beta_{\rm c}` of an
        Osipkov-Merritt model, or :math:`R_{\rm a}` of a model without an
        anisotropy radius.
    """

    def __init__(
        self,
        profile: DoublePowerLaw,
        beta_c: float,
        model: str,
        r_a: float,
        max_order: int,
        wrt: str,
    ) -> None:
        import inspect

        import mpmath as mp
        import sympy as sp

        if wrt not in ("beta_c", "r_a"):
            raise InputError(f"The parameter is beta_c or r_a; got {wrt!r}.")
        if wrt == "beta_c" and model.startswith("OM"):
            raise InputError("An Osipkov-Merritt model has no beta_c to vary.")
        if wrt == "r_a" and model not in ("OM_I", "OM_II", "COM_I", "COM_II"):
            raise InputError(f"A {model} model has no R_a to vary.")
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
        syms: list[Any] = []
        vals: list[float] = []

        def exponent(x: float, name: str) -> Any:
            q = sp.Rational(x)
            if q.q <= 1000:
                return q
            s = sp.Symbol(name, positive=True)
            syms.append(s)
            vals.append(float(x))
            return s

        ea = exponent(a, "alpha")
        eb = exponent(b, "beta")
        eg = exponent(g, "gamma")
        bc_s, ra_s = sp.Symbol("beta_c", real=True), sp.Symbol("r_a", positive=True)
        syms += [bc_s, ra_s]
        vals += [bc, float(r_a)]
        rho = C * R ** (-eg) * (1 + R**ea) ** ((eg - eb) / ea)
        gfac = inverse_factor_expr(R, model, bc_s, ra_s, continuation=True)
        theta = bc_s if wrt == "beta_c" else ra_s
        rho_aux = rho * gfac
        m_sym = sp.Function("M")(R)
        sub = {sp.Derivative(m_sym, R): R**2 * rho}
        mv = sp.Symbol("Mv")
        key = (
            "mixed",
            wrt,
            a,
            b,
            g,
            c_norm,
            model,
            int(max_order),
            inspect.getsource(ParameterDerivs.__init__),
        )
        exprs = _load_construction(key)
        if exprs is None:
            exprs, expr = [], rho_aux
            for _ in range(1, max_order + 1):
                expr = sp.cancel((sp.diff(expr, R) / (-m_sym / R**2)).subs(sub).doit())
                expr = expr.replace(sp.DiracDelta, lambda *args: sp.Integer(0))
                exprs.append(sp.diff(expr.xreplace({m_sym: mv}), theta))
            _save_construction(key, exprs)
        a_mp, b_mp, g_mp, c_mp = map(mp.mpf, (a, b, g, c_norm))
        a1, b1 = (3 - g_mp) / a_mp, (b_mp - 3) / a_mp

        def m_mp(rv):
            rv = mp.mpf(rv)
            ra_ = rv**a_mp
            eps = 1 / (1 + ra_)
            if eps >= mp.mpf("1e-10"):
                return c_mp / a_mp * mp.betainc(a1, b1, 0, ra_ / (1 + ra_))
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

        def m_np(rv):
            if np.ndim(rv):
                return np.asarray(profile.mass(rv), dtype=np.float64).reshape(
                    np.shape(rv)
                )
            return float(np.atleast_1d(profile.mass(rv))[0])

        vals_mp = [mp.mpf(v) for v in vals]
        d0 = sp.diff(rho_aux, theta)
        f0 = sp.lambdify((R, *syms), d0, "numpy")
        f0_mp = sp.lambdify((R, *syms), d0, "mpmath")
        self.funcs: list[Any] = [lambda rv: f0(rv, *vals)]
        self.funcs_mp: list[Any] = [lambda rv: f0_mp(rv, *vals_mp)]
        for e_m in exprs:
            fn = _lambdify_deep((R, mv, *syms), e_m, "numpy")
            fm = _lambdify_deep((R, mv, *syms), e_m, "mpmath")
            self.funcs.append(lambda rv, _f=fn: _f(rv, m_np(rv), *vals))
            self.funcs_mp.append(lambda rv, _f=fm: _f(rv, m_mp(rv), *vals_mp))
        self._mp = mp
        self._typeII = model in ("OM_II", "COM_II")
        self._ra = float(r_a)


def source_rows(
    source: ExactDerivs, k: Array, orders: tuple[int, ...]
) -> AuxDerivatives:
    """Return the values of ``source`` (an `ExactDerivs` or a
    `ParameterDerivs`) for the orders ``orders`` at the energies k, their values
    at the floor k[0], and ``source`` itself for the completion below the
    floor, in the `AuxDerivatives` layout that `aux_derivatives` returns.

    Raises
    ------
    ResultError
        If a value at the floor k[0] is not finite.
    """
    from ..exceptions import ResultError

    def every_order(p: Array) -> Array:
        out = np.zeros((len(orders), p.size))
        ok = (p > 0.0) & (p < 1.0)
        if ok.any():
            rv = source.r_of_p(p[ok])
            for i, o in enumerate(orders):
                out[i, ok] = source._eval(o, rv)
        return out

    k = np.asarray(k, dtype=np.float64)
    values = every_order(k)[None, :, :]
    try:
        at_floor = every_order(k[:1])[:, 0][None, :]
    except FloatingPointError as err:
        raise ResultError(
            f"The derivatives at the floor are not finite ({err})."
        ) from err
    return AuxDerivatives(tuple(orders), values, at_floor, True, source)
