r"""Derivatives of a component's distribution function in its anisotropy.

This module returns the distribution function (DF) :math:`F(Q)` that
`compute_df` makes for a component, at fixed relative energies :math:`Q`,
together with the derivatives of :math:`F` in the anisotropy parameters
:math:`\theta = (\beta_{\rm c}, R_{\rm a})`. `inversion_nodes` returns the
energies at which `compute_df` evaluates :math:`F`. `rows_jacobian` returns
:math:`F` and its derivatives at the energies of `inversion_nodes` as
arrays, and `table_jacobian` returns every array of the DF's table and its
derivatives. `make_inversion_rows` returns :math:`F`, and
`make_inversion_table` returns the arrays of the DF's table, as JAX
functions of :math:`\theta`, which ``jax.grad`` and ``jax.vjp``
differentiate.

The DF is :math:`F = T_b[g]`. Here :math:`b = \beta_{\rm c}` is the central
anisotropy. The source :math:`g` is the set of derivatives
:math:`{\rm d}^N \varrho_{\rm aux} / {\rm d}P^N` of the auxiliary density
:math:`\varrho_{\rm aux}` in the relative potential :math:`P`. The operator
:math:`T_b` is the inversion of the auxiliary density at
:math:`\beta_{\rm c} = b`, and :math:`T_b` is linear in :math:`g`. The
derivatives of :math:`F` are

.. math::

    \frac{\partial F}{\partial R_{\rm a}} = T_b\left[\partial_{R_{\rm a}} g\right],
    \qquad
    \frac{\partial F}{\partial \beta_{\rm c}} = T_b\left[\partial_b g\right]
    + E_b[g].

The derivative :math:`\partial_\theta g` of the source in :math:`\theta` at
a fixed :math:`P` is in closed form. The term :math:`E_b[g]` is the
derivative of the operator :math:`T_b` in :math:`b` at a fixed source
:math:`g`. :math:`E_b[g]` has no closed form and is taken by finite
differences of :math:`T_b[g]` in :math:`b`. The record returned with each
derivative in :math:`\beta_{\rm c}` states the residual of the finite
differences in :math:`b`, for comparison with the accuracy bounds
:math:`10^{-6}`, :math:`10^{-4}` and :math:`10^{-2}` of the package's
validation reports.

The derivative of the DF's table follows by the chain rule through each
array of the table. The energies and every discrete choice of the table are
held at the parameters the call is given. For a component on a
`TruncatedProfile` (Method II), the profile is held, and the derivative of
the source is the derivative of the inverse anisotropy factor in closed
form.

The derivatives are available for constant anisotropy
(:math:`\beta_{\rm c}`), Cuddeford-Osipkov-Merritt (COM) Type I
(:math:`\beta_{\rm c}` and :math:`R_{\rm a}`), and Osipkov-Merritt Type I
and Type II (:math:`R_{\rm a}`). The derivatives are not available for COM
Type II, whose interior point at the potential :math:`P(R_{\rm a})` moves
with both parameters. The derivatives are not available for a DF whose
center lies on the
critical branch, where the inner slope :math:`\gamma` of the density equals
:math:`2\beta_{\rm c}` and the form of the center changes with
:math:`\beta_{\rm c}`. The derivation is Notes Eq. (T:7.1).
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
from functools import cache
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

# BESIDE is the offset from a central anisotropy b at which `_region` and
# `_table_difference` read the inversion's form just beside b. PARAMETERS
# maps each anisotropy family to its parameters, in the order of a parameter
# vector theta.
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
    r"""Return the energies at which `compute_df` inverts a component.

    The energies are the relative energies :math:`Q` at which `compute_df`
    evaluates the DF :math:`F(Q)` for the given component and settings. The
    other public functions of `pianistpy.df.gradient` hold the energies of
    `inversion_nodes` fixed when the anisotropy parameters change.

    Parameters
    ----------
    component : Component
        Component on an analytic profile (not a `TruncatedProfile`).
    settings : Settings
        Numerical settings, the same as those given to `compute_df`.

    Returns
    -------
    q : ndarray, shape (n,)
        Relative energies :math:`Q`, strictly increasing, in internal
        units.

    Raises
    ------
    InputError
        If the energies that `settings` places include two energies that
        float64 cannot separate.
    ResultError
        If the component is Osipkov-Merritt Type II or COM, whose energies
        are placed around a feature of a first coarse inversion, and the
        coarse inversion finds a derivative of the auxiliary density that
        is not finite.

    See Also
    --------
    rows_jacobian : The DF and its derivatives at the energies of
        `inversion_nodes`.
    make_inversion_rows : The DF at the energies of `inversion_nodes` as a
        JAX function.

    Examples
    --------
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.df.gradient import inversion_nodes
    >>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
    >>> q = inversion_nodes(halo, pp.Settings())
    >>> bool(np.all(np.diff(q) > 0.0))
    True
    """
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
    r"""Return the DF at fixed energies and its derivatives in the anisotropy.

    The values are the DF :math:`F(Q)` that `compute_df` makes, at the
    relative energies `q`. The derivatives are those of :math:`F` in the
    component's anisotropy parameters at fixed `q`. For the derivatives of
    every array of the DF's table (the nodes below the table, the form at
    the center and the tail), call `table_jacobian` instead.

    Parameters
    ----------
    component : Component
        Component on an analytic profile (not a `TruncatedProfile`), with
        constant, isotropic, Osipkov-Merritt Type I or Type II, or COM Type
        I anisotropy.
    settings : Settings
        Numerical settings, the same as those given to `compute_df`.
    q : array-like, shape (n,), optional
        Relative energies :math:`Q`, increasing, in internal units.
        Default: ``None``, which means the energies that `inversion_nodes`
        returns for `component` and `settings`.
    steps : dict, optional
        Dictionary that keeps the steps of the finite differences in
        :math:`\beta_{\rm c}` between calls. A call reads the steps from
        `steps` where `steps` holds steps for the call's profile, anisotropy
        family, anisotropy kind, settings and form of the inversion. A call
        otherwise measures the steps and writes them into `steps`. Passing
        one dictionary to a sequence of calls makes every later call use the
        steps that the first call measured. Default: ``None``, which means
        each call measures its own steps and keeps them nowhere.

    Returns
    -------
    q : ndarray, shape (n,)
        Relative energies :math:`Q`, in internal units: `q` as given, or
        the energies of `inversion_nodes`.
    f : ndarray, shape (n,)
        DF :math:`F(Q)` at `q`, as `compute_df` makes it, in internal units.
    jac : dict of ndarray
        Derivatives of :math:`F` at `q`, each with the shape of `q`, in
        internal units. The key ``"beta_c"`` holds
        :math:`\partial F/\partial\beta_{\rm c}` and the key ``"r_a"`` holds
        :math:`\partial F/\partial R_{\rm a}`, for the parameters that the
        anisotropy family has.
    record : dict
        Description of the finite differences in :math:`\beta_{\rm c}`. The
        record is an empty dict for an Osipkov-Merritt component, which has
        no parameter :math:`\beta_{\rm c}`. The entry ``"exponent"`` is
        ``"central"`` where the form of the inversion on one side of
        :math:`\beta_{\rm c}` equals the form on the other side, and
        ``"one-sided"`` at a :math:`\beta_{\rm c}`
        where the form changes (such as :math:`\beta_{\rm c} = 0` or a
        half-integer :math:`\beta_{\rm c}`). The entry ``"steps"`` holds the
        steps, ``"pilots"`` the trial steps from which the steps were
        derived, ``"regions"`` the widths in :math:`\beta_{\rm c}` over which
        the form does not change, and ``"measured"`` whether the call
        measured the steps instead of reading them from `steps`. The entry
        ``"accuracy"`` is the inversion's own accuracy, the change of the
        operator's values when the number of quadrature nodes doubles. The
        entry ``"residual"`` is the Richardson residual
        :math:`\max|E(d) - E(d/2)|` of the finite differences at the step
        :math:`d`. A one-sided record also holds ``"sides"``, the number of
        sides (2), and ``"spread"``, the largest difference between the
        derivatives from the two sides. The entries ``"accuracy"``,
        ``"residual"`` and ``"spread"`` are relative to
        :math:`\max|\partial F/\partial\beta_{\rm c}|` over `q`, so the
        relative error of an entry of :math:`\partial F/\partial\beta_{\rm c}`
        far below that maximum can exceed ``"residual"``.

    Raises
    ------
    InputError
        If `component` is on a `TruncatedProfile`.
    InputError
        If the anisotropy is COM Type II, whose interior point at the
        potential :math:`P(R_{\rm a})` moves with both parameters.
    ResultError
        If a derivative of the auxiliary density is not finite at `q`.

    See Also
    --------
    inversion_nodes : The energies at which `compute_df` evaluates the DF.
    make_inversion_rows : The DF at fixed energies as a JAX function of the
        parameters.
    table_jacobian : The derivatives of every array of the DF's table.

    Notes
    -----
    Each derivative is :math:`T_b` applied to the closed-form derivative of
    the source :math:`g`. The derivative in :math:`\beta_{\rm c}` adds the
    finite differences of the operator :math:`T_b` in :math:`b`.

    A call inverts the component once for :math:`F` and once per parameter.
    In :math:`\beta_{\rm c}`, a call also inverts the component at every
    point of the finite differences, at `settings` and at twice the
    quadrature nodes of `settings`.

    References
    ----------
    Notes Eq. (T:7.1).

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.df.gradient import rows_jacobian
    >>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
    >>> q, f, jac, record = rows_jacobian(halo, pp.Settings())
    >>> sorted(jac), jac["r_a"].shape == q.shape, record
    (['r_a'], True, {})
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

    f = np.asarray(
        inv._rows(component, settings, pair, q, capture, deriv=False)[0], np.float64
    )
    orders = held["aux"].orders
    jac: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {}
    for theta in params:
        src = ParameterDerivs(profile, b, family, r_a, max(2, orders[-1]), wrt=theta)
        rows = source_rows(src, q, orders)
        jac[theta] = np.asarray(
            inv._rows(
                component, settings, pair, q, lambda *_, _r=rows: _r, deriv=False
            )[0],
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
    r"""Return the width w of the interval (0, w] of offsets x over which the
    form ``key(b + s x)`` equals the form ``key(b + s BESIDE)`` just beside b.

    ``key`` defaults to `_form_key`, the form of the inversion, and s is the
    side (+1 or -1). The width is found by doubling x from BESIDE and then
    by 40 bisection steps on the form. The width is capped at the distance
    from b to the next half-integer :math:`\beta_{\rm c}` on side s,
    because the closed form at a kernel exponent of 0 holds only at a
    half-integer :math:`\beta_{\rm c}`, the forms on the two sides of a
    half-integer have one key, and doubling and bisection would therefore
    step over the half-integer. Every offset x counted in the width keeps
    b + s x below :math:`\beta_{\rm c} = 1`."""
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
    """Return ``settings`` with the inversion's ``n_nodes`` doubled.

    The kernel's accuracy at ``settings`` is the change of the rows from
    ``settings`` to the returned settings."""
    groups: dict[str, Any] = {g: dict(v) for g, v in settings.overrides.items()}
    groups["inversion"] = {
        **groups.get("inversion", {}),
        "n_nodes": 2 * int(settings.inversion["n_nodes"]),
    }
    return Settings(quick=settings.level == "quick", **groups)


def _exponent_part(component, settings, q, b, r_a, orders, steps=None):
    """Return E_b[g], the derivative of the operator T_b in b at a fixed
    source g, and the record of its finite differences.

    The source rows are held at their values at b, for the orders
    max(j - 1, 0) to j + 3, with j the order of the inversion's form at b.
    The orders max(j - 1, 0) to j + 3 cover the forms on both sides of b.
    `_table_difference` takes the finite differences, with the kernel's
    accuracy measured against the rows at twice the nodes (`_finer`)."""
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
        return np.asarray(
            inv._rows(c, s, pair, q, lambda *_: held, deriv=False)[0], np.float64
        )

    member = ("rows", profile, family, component.anisotropy.kind, settings)
    return _table_difference(
        t, b, _form_key, member, lambda bb: t(bb, _finer(settings)), steps
    )


def make_inversion_rows(
    component: Component, settings: Settings, q: np.ndarray | None = None
) -> tuple[Any, tuple[str, ...], np.ndarray]:
    r"""Return the DF at fixed energies as a JAX function of the anisotropy.

    The returned function maps the anisotropy parameters :math:`\theta` to
    the DF :math:`F(Q)` that `compute_df` makes at the relative energies
    `q`. The reverse-mode derivative of the returned function at each
    :math:`\theta` is the derivative that `rows_jacobian` returns at that
    :math:`\theta`. For the derivatives as arrays, call `rows_jacobian`
    instead.

    Parameters
    ----------
    component : Component
        Component on an analytic profile (not a `TruncatedProfile`), with
        constant, isotropic, Osipkov-Merritt Type I or Type II, or COM Type
        I anisotropy. The anisotropy parameters of `component` are the
        point at which the steps of the finite differences in
        :math:`\beta_{\rm c}` are measured.
    settings : Settings
        Numerical settings, the same as those given to `compute_df`.
    q : array-like, shape (n,), optional
        Relative energies :math:`Q`, increasing, in internal units.
        Default: ``None``, which means the energies that `inversion_nodes`
        returns for `component` and `settings`.

    Returns
    -------
    rows : callable
        Function ``rows(theta)``. The argument ``theta`` is an array of the
        parameter values in the order of `params`. The value is a JAX array
        of shape (n,) that holds :math:`F(Q)` at `q`, in internal units.
        ``jax.grad`` and ``jax.vjp`` differentiate ``rows``.
    params : tuple of str
        Names of the anisotropy parameters in ``theta``: ``("beta_c",)``,
        ``("r_a",)`` or ``("beta_c", "r_a")``.
    q : ndarray, shape (n,)
        Relative energies :math:`Q`, in internal units: `q` as given, or the
        energies of `inversion_nodes`.

    Raises
    ------
    InputError
        If the anisotropy is COM Type II.
    InputError
        If `q` is ``None`` and the energies that `settings` places include
        two energies that float64 cannot separate.
    ResultError
        If `q` is ``None``, the component is Osipkov-Merritt Type II or
        COM, and the coarse inversion that places the energies finds a
        derivative of the auxiliary density that is not finite.

    See Also
    --------
    rows_jacobian : The DF at fixed energies and its derivatives as arrays.
    make_inversion_table : Every array of the DF's table as a JAX function.
    inversion_nodes : The energies at which `compute_df` evaluates the DF.

    Notes
    -----
    Each call of ``rows`` inverts the component again at the new
    parameters, at the energies `q`. Each derivative of ``rows`` calls
    `rows_jacobian`, which raises InputError for a component on a
    `TruncatedProfile`.

    The steps of the finite differences in :math:`\beta_{\rm c}` are
    measured once, at the parameters of `component`, and are used at every
    ``theta``. The derivative at one ``theta`` therefore does not depend on
    the values of ``theta`` in earlier calls.

    References
    ----------
    Notes Eq. (T:7.1).

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.df.gradient import make_inversion_rows
    >>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
    >>> rows, params, q = make_inversion_rows(halo, pp.Settings())
    >>> params
    ('r_a',)
    >>> theta = np.array([10.0])
    >>> rows(theta).shape == q.shape
    True
    >>> jax.grad(lambda t: rows(t)[-1])(theta).shape
    (1,)
    """
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
        return np.asarray(
            inv._rows(c, settings, pair, q, deriv=False)[0], dtype=np.float64
        )

    # The steps of the finite differences in beta_c are measured at the
    # component's own parameters and reused at every theta, so that the
    # derivative at one theta does not depend on the theta of earlier calls.
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
    """The closed-form source g + e dg, with g an `ExactDerivs` and dg the
    `ParameterDerivs` of the same profile, for a fixed shift e."""

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
    r"""Return the derivative of the operator T_b at the source g (``aux``)
    along the source derivative dg (``rows``), stacked for F and dF/dQ.

    The derivative is the central difference
    (T_b[g + e dg] - T_b[g - e dg]) / (2 e). T_b[dg] differs from the
    central difference because the operator interpolates a source row in
    ln g where the row is positive at every energy, and in g otherwise. The
    shift e is the largest shift for which
    :math:`e\, |{\rm d}g| \le 10^{-6}\, g` holds on every row of g that is
    positive at every energy, so that every such row of g stays positive
    under the shift; e is 1 where dg is 0 on all such rows."""
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
    """Return the largest |x| over the finite entries of x, or 0 where x has
    no finite entry. Entries that are not finite are skipped because a
    table's rows hold NaN where the table has no value."""
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
    """Return the derivative dt/db of the function t at b by finite
    differences of t, and the record of the finite differences in absolute
    values.

    The rounding of t(b) is the double-precision epsilon times max |t(b)|.
    The accuracy acc of t is the larger of the rounding of t(b) and
    max |t(b) - finer(b)|, with ``finer(b)`` the value of t at a finer
    quadrature, where ``finer`` is given; acc is the rounding of t(b) where
    ``finer`` is None.

    The central case holds where the form ``key`` at b equals the form just
    beside b on both sides and the second entry of ``key(b)``, which marks
    an exponent n = 0, is false. In the central case, the Richardson
    quotient E(d) over the steps d and d/2 has the rounding term 3 acc / d
    and the truncation term d^4 |F5| / 480, with F5 = d^5 t / db^5. The sum
    of the two terms is least at d^5 = 360 acc / |F5|. |F5| follows from
    the Richardson residual at a pilot step h equal to half the smaller of
    the two sides' regions (`_region`): |E(h) - E(h/2)| = |F5| h^4 / 512.

    The one-sided case holds at every other b: where the form changes at b
    on one side, or where the exponent n at b is 0. In the one-sided case,
    the Richardson combination of the second-order one-sided quotients over
    d and d/2 has the rounding term 12 acc / d and the truncation term
    d^3 |F4| / 24, with F4 = d^4 t / db^4. The sum of the two terms is least
    at d^4 = 96 acc / |F4|. |F4| follows from
    |E(h) - E(h/2)| = 7 |F4| h^3 / 192 at a pilot step h equal to a quarter
    of the side's region. The one-sided derivative is the mean of the
    derivatives from the two sides. In both cases, no measured step exceeds
    its pilot step.

    ``member`` is a hashable description of t without its anisotropy
    parameters, and ``store`` is a dictionary that the caller holds. With
    both given, acc and the steps are read from ``store`` where ``store``
    holds them for ``member``, the name of ``key``, the form at b and the
    case, and are measured and written into ``store`` otherwise. A step
    read from ``store`` is capped at half the region at the new b in the
    central case, and at a quarter of the side's region at the new b in the
    one-sided case. Without ``store``, the steps are measured at each call.

    The record holds the case (``"exponent"``), the steps, the pilot steps,
    the regions, acc, whether the call measured the steps, and the
    Richardson residual
    max |E(d) - E(d/2)| at the step d taken (in the one-sided case, the
    larger residual of the two sides). The one-sided record also holds the
    number of sides and the spread max |E_+ - E_-| between the derivatives
    E_+ and E_- of the two sides."""
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
    r"""Return the rows F and dF/dQ, the parts of the inversion from which
    the DF's table is built, and the derivatives of the rows in each
    parameter.

    The derivative in each parameter is the operator's derivative along the
    derivative of the source in that parameter (`ParameterDerivs`, `_along`).
    The derivative in :math:`\beta_{\rm c}` adds E_b[g], the operator's own
    derivative in b at a fixed source, on both F and dF/dQ
    (`_table_difference`)."""
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
    r"""Return d kappa / d theta and d ln A / d theta for the center's
    regular branch (`central`).

    The exponent is kappa = s0 - nu and the amplitude is
    A = N'(b) C cu^-s0 Gamma(nu - s0) / Gamma(-s0), with
    N'(b) = 2^b / ((2 pi)^(3/2) Gamma(1 - b)), s0 = (2 b - gamma) / (2 - gamma),
    nu = 3/2 - b, cu = C / ((2 - gamma) (3 - gamma)), b the central
    anisotropy, gamma the inner slope of the density and C the profile's
    density normalization ``rho_norm``. Neither kappa nor A depends on
    :math:`R_{\rm a}`, so both derivatives are 0 for theta = ``"r_a"``."""
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
    """Return the change dc of the least-squares solution c = argmin |M c - y|
    for the changes dM of M and dy of y."""
    r = y - m @ c
    dc = np.linalg.lstsq(m, dy - dm @ c, rcond=None)[0]
    return dc + np.linalg.solve(m.T @ m, dm.T @ r)


def _patch_jvp(qa, fa, dfa, kappa, dkappa, amplitude, damp, patch):
    """Return the derivatives of the coefficients (a_anchor, b, d) of the
    form above the anchor (`top_patch`), with the fit window
    ``patch.fit_window`` and the closed-form choice ``patch.closed_form``
    held at their values in ``patch``. The depth below the central
    potential is u = 1 - Q, the central potential being 1 in internal
    units."""
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
    """Return the derivatives of F and of dF/dQ in theta at the nodes below
    the written floor, and the record of the finite differences in b."""
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
    """Return d g / d theta of the inverse anisotropy factor
    g (`inverse_factor_expr`) and the derivatives of d g / d theta in R of
    orders 1 to 3, in closed form at the radii r. The factor is
    ``g = R^(2 beta_c)`` for the constant and isotropic families and
    ``g = R^(2 beta_c) u^s`` otherwise, with ``u = 1 + R^2 / r_a^2`` for
    Type I, ``u = 1 - R^2 / r_a^2`` for Type II, ``s = 1`` for OM and
    ``s = 1 - beta_c`` for COM."""
    import sympy

    rs, bs, a_s, derivs = _dfactor_exprs(family, theta)
    out = []
    for d in derivs:
        fn = sympy.lambdify(rs, d.subs({bs: b, a_s: r_a}), "numpy")
        out.append(np.asarray(fn(r) * np.ones_like(r), dtype=np.float64))
    return tuple(out)


@cache
def _dfactor_exprs(family: str, theta: str) -> tuple:
    """Return the sympy symbols ``(R, b, r_a)`` and the derivatives of
    d g / d theta in R of orders 0 to 3 as sympy expressions, with g the
    inverse anisotropy factor (`_dfactor`). The expressions are formed once
    per anisotropy family and parameter."""
    import sympy

    from ..anisotropy.factor import inverse_factor_expr

    rs, a_s = sympy.symbols("R r_a", positive=True)
    bs = sympy.Symbol("b", real=True)
    g = inverse_factor_expr(rs, family, bs, a_s)
    dg = sympy.diff(g, bs if theta == "beta_c" else a_s)
    return rs, bs, a_s, tuple(sympy.diff(dg, rs, k) for k in range(4))


def _dual_series(profile, bce: float, m_typ: str, r_a: float, theta: str, x_max: float):
    """Return the inner series of the auxiliary density (`rho_aux_series`)
    with the derivative of each term in theta, as a sorted list of tuples
    (s, C_s, d C_s / d theta, d s / d theta).

    The series of the radius R in the depth does not depend on theta. Theta
    enters only through ``q = 2 beta_c - gamma``, the factor's exponent
    ``1 - beta_c`` and the coefficient ``R_a^-2``."""
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
    """Return the numbers of the center's closed form that the lookup reads
    above the table, and their derivatives in theta in closed form, on the
    regular branch.

    The numbers are the exponent kappa, the amplitude, and, where the
    center's next term is a power or a logarithm times a power
    (``next_form``), the next term's exponent and coefficient."""
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
    """Return the arrays of a Method II child's table (log_df, edge_law where
    the floor has a law, and center) at the child's own parameters.

    ``held`` holds what stays fixed: the grid ``q``, the kept nodes
    ``keep`` and the parent's inputs."""
    tab = _tabulated()
    aniso = component.anisotropy
    sources = tab.factor_sources(
        _held_density(held, tab._pair(aniso)[3]),
        aniso.family,
        float(aniso.beta_c),
        float(aniso.r_a),
    )
    made = tab.child_rows(
        held["q"],
        held["parent_df"],
        held["parent_rows"],
        held["parent_r_a"],
        aniso,
        settings,
        sources=sources,
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
    r"""Return the form of a child's rows at :math:`\beta_{\rm c} = b`, as
    `_table_difference` reads a key: the kernel's order, whether b has no
    central quotient (a kernel exponent of 0 or a floor pair with n = 0),
    and the floor's order."""
    j, half = _form_key(b)
    m, n_zero = _floor_key(b)
    return j, half or n_zero, m


def _child_held(component: Component, settings: Settings) -> dict[str, Any]:
    """Solve a Method II child at its own parameters (`tabulated_df`) and
    return what the derivatives hold fixed: the grid ``q``, the kept nodes
    ``keep`` (the finite rows) and the parent's inputs."""
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


def _held_density(held: dict[str, Any], order: int) -> dict[str, Any]:
    """Return the parent's part of the child's sources on the held grid
    (`parent_density`) at the given order. The parent's part does not
    depend on the child's parameters, so the parent's part is formed once
    per order and kept in ``held``."""
    formed = held.setdefault("density", {})
    if order not in formed:
        rows = held["parent_rows"]
        formed[order] = _tabulated().parent_density(
            held["q"],
            float(np.asarray(rows["Psi"], dtype=np.float64)[0]),
            order,
            held["parent_df"],
            rows,
            held["parent_r_a"],
        )
    return formed[order]


def _child_jacobian(
    component: Component,
    settings: Settings,
    held: dict[str, Any] | None = None,
    steps: dict[tuple, dict[str, Any]] | None = None,
    placement: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Return `table_jacobian`'s values, derivatives and record for a
    component on a truncated profile (Method II).

    The parent is held fixed. The grid and the kept nodes are read from
    ``held``, whose default is the grid and the kept nodes of the solve at
    the component's own parameters (`_child_held`). The record's entry
    ``"grid"`` (`_placement`) costs a coarse inversion for OM Type II and
    COM Type I; with ``placement`` false, the record has no entry
    ``"grid"``."""
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
    density = _held_density(held, order)
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
    record: dict[str, Any] = {"rows": {}}
    if placement:
        record["grid"] = _placement(component, settings, held)
    for theta in params:
        dsrc = tab.factor_sources(
            density,
            family,
            b,
            r_a,
            factor=_dfactor(family, b, r_a, density["r"], theta),
        )
        # The operator's derivative along the sources' derivative is a central
        # difference with the shift e, which keeps every positive source row
        # positive (e |dsrc| <= 1e-6 src), as in `_along`.
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
            # The edge law's coefficient C_q is linear in a_q, and a_q is linear
            # in the floor node's (g, g_p), so the floor record of the source
            # derivative gives dC_q. In beta_c at a fixed a_q, C_q also gains
            # d ln C_q = ln 2 + psi(1 - b) - psi(q - 1/2 + b), with psi the
            # digamma function, q = 5/2 for C_0 and q = 7/2 for C_1, and the
            # law's power p = 1 + beta_c gains dp = 1.
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
    r"""Return where a child's grid is placed: the number of nodes, the
    number of kept nodes, and the energy around which the grid's zones are
    placed. The energy around which the zones are placed is the potential
    :math:`P(R_{\rm a})`
    (``psi_r_a``) for OM Type I and the feature energy of a coarse
    inversion (``q_feature``) for OM Type II and COM Type I; the other
    families have no such entry."""
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
    r"""Return the arrays of the DF's table and their derivatives in the anisotropy.

    The values are the arrays of the table that `compute_df` makes for the
    component. The derivatives are those of each array in the component's
    anisotropy parameters, with the energies and every discrete choice of
    the table held at the component's own parameters. The discrete choices
    are the placement of the energies, the form of the inversion, the
    anchor of the form at the center and the nodes kept. For the DF at the
    energies of the inversion alone, call `rows_jacobian` instead.

    Parameters
    ----------
    component : Component
        Component with constant, isotropic, Osipkov-Merritt Type I or Type
        II, or COM Type I anisotropy, on an analytic profile or on a
        `TruncatedProfile` (Method II).
    settings : Settings
        Numerical settings, the same as those given to `compute_df`.
    q : array-like, shape (n,), optional
        Relative energies :math:`Q`, increasing, in internal units, for a
        component on an analytic profile. Default: ``None``, which means
        the energies that `inversion_nodes` returns for `component` and
        `settings`. A component on a `TruncatedProfile` takes the energies
        of its own solve and accepts only ``None``.
    steps : dict, optional
        Dictionary that keeps the steps of the finite differences in
        :math:`\beta_{\rm c}` between calls, as in `rows_jacobian`.
        Default: ``None``, which means each call measures its own steps and
        keeps them nowhere.

    Returns
    -------
    values : dict of ndarray
        Arrays of the table, in internal units. For a component on an
        analytic profile, ``k`` holds the energies of the table,
        ``log_df`` the base-10 logarithm of the DF at ``k``, and
        ``log_df_slope`` the slope :math:`{\rm d}\log_{10} F/{\rm d}\ln Q`
        at ``k``. Where the table has nodes below its first energy,
        ``k_below``, ``log_df_below`` and ``log_df_slope_below`` hold the
        energy, the logarithm of the DF and the slope at those nodes. Where
        the table has a closed form above its anchor,
        :math:`F = a_{\rm anchor} (u/u_{\rm anchor})^\kappa + b + d\,u` in
        the depth :math:`u = 1 - Q`, ``central`` holds the array
        :math:`(\kappa, a_{\rm anchor}, b, d)`. Where the table has a tail
        below its lowest node, ``tail`` holds the tail's exponent,
        amplitude and logarithmic exponent. For a component on a
        `TruncatedProfile`, the keys are ``k``, ``log_df``, ``edge_law``
        (the power and the two coefficients of the edge law below the
        floor, where the component has one) and ``center`` (the exponent
        and the amplitude of the center's closed form, and the next term's
        exponent and coefficient where the center's form states them).
    jac : dict of dict of ndarray
        Derivatives ``jac[name][key]`` of ``values[key]`` in the parameter
        ``name`` (``"beta_c"`` or ``"r_a"``), for every key of `values`
        except the energies ``k`` and ``k_below``, each with the shape of
        ``values[key]``. Where a Type II DF is outside its support, the
        derivatives of ``log_df`` and ``log_df_slope`` are 0.
    record : dict
        Description of the finite differences in :math:`\beta_{\rm c}`,
        with the entries of the record of `rows_jacobian` in absolute
        values, not relative ones. For a component on an analytic profile,
        ``record["rows"]`` describes the finite differences of the DF and
        its slope at the energies of the inversion, and ``record["below"]``,
        where present, describes the finite differences at the nodes below
        the table. For a component
        on a `TruncatedProfile`, ``record["rows"]`` describes the finite
        differences of the DF, and ``record["grid"]`` states the number of
        energies, the number of energies kept, and the energy around which
        the energies are placed.

    Raises
    ------
    InputError
        If the anisotropy is COM Type II, whose interior point at the
        potential :math:`P(R_{\rm a})` moves with both parameters.
    InputError
        If `component` is on an analytic profile and the DF's center lies
        on the critical branch, where the inner slope :math:`\gamma` of the
        density equals :math:`2\beta_{\rm c}`, with a divergent center of a
        Type I model or with a closed form above the anchor.
    InputError
        If `component` is on a `TruncatedProfile` and the DF's center is
        not on the regular branch :math:`\gamma > 2\beta_{\rm c}`, or the
        solve of `component` finds the center in the band of
        :math:`\beta_{\rm c}` around the critical branch.
    InputError
        If `q` is given for a component on a `TruncatedProfile`.
    InputError
        If `component` is on a `TruncatedProfile` that Method II does not
        invert. Method II does not invert a profile that carries no DF of
        its own model, a profile whose model's anisotropy is neither
        isotropic nor Osipkov-Merritt Type I, a component whose
        :math:`\beta_{\rm c}` needs sources of an order above 3, or a Type
        II component whose :math:`R_{\rm a}` does not lie outside the
        profile's edge.
    AccuracyError
        If a value of the DF inside the table is not finite or is 0, or
        exceeds :math:`10^{300}` where no closed form covers the center.
    AccuracyError
        If the tail's logarithmic exponent below the floor is not finite.
    UnphysicalModelError
        If `component` is on a `TruncatedProfile` whose model is truncated
        but not spatially finite.
    ResultError
        If a derivative of the auxiliary density is not finite at the
        energies, or if the DF of a component on a `TruncatedProfile` has a
        defective band at the top of its energies.

    See Also
    --------
    make_inversion_table : The arrays of the DF's table as a JAX function
        of the parameters.
    rows_jacobian : The DF and its derivatives at the energies of the
        inversion.
    inversion_nodes : The energies at which `compute_df` evaluates the DF.

    Notes
    -----
    The derivative of each array of the table follows by the chain rule
    from the derivatives of the DF at the energies of the inversion and
    from the closed-form derivatives of the center's exponent and
    amplitude.

    A call costs several inversions of the component, as `rows_jacobian`
    does, and for a component on a `TruncatedProfile` one solve of the
    component in addition.

    References
    ----------
    Notes Eq. (T:7.1).

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.df.gradient import table_jacobian
    >>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
    >>> values, jac, record = table_jacobian(halo, pp.Settings())
    >>> sorted(jac)
    ['r_a']
    >>> jac["r_a"]["log_df"].shape == values["log_df"].shape
    True
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
    r"""Return the arrays of the DF's table as a JAX function of the anisotropy.

    The returned function maps the anisotropy parameters :math:`\theta` to
    the arrays of the table that `compute_df` makes. The energies of the
    table and the set of its arrays are held at the component's own
    parameters. The other discrete choices of the table, the anchor of the
    form at the center and the nodes kept, are made again at each
    :math:`\theta`, except that a component on a `TruncatedProfile` also
    holds the nodes kept at the component's own parameters. The
    reverse-mode derivative of the returned function at
    each :math:`\theta` is the derivative that `table_jacobian` returns at
    that :math:`\theta`. For the derivatives as arrays, call
    `table_jacobian` instead.

    Parameters
    ----------
    component : Component
        Component with constant, isotropic, Osipkov-Merritt Type I or Type
        II, or COM Type I anisotropy, on an analytic profile or on a
        `TruncatedProfile` (Method II).
    settings : Settings
        Numerical settings, the same as those given to `compute_df`.
    q : array-like, shape (n,), optional
        Relative energies :math:`Q`, increasing, in internal units, for a
        component on an analytic profile. Default: ``None``, which means
        the energies that `inversion_nodes` returns for `component` and
        `settings`. A component on a `TruncatedProfile` takes the energies
        of its own solve, and `q` is not used.

    Returns
    -------
    table : callable
        Function ``table(theta)``. The argument ``theta`` is an array of the
        parameter values in the order of `params`. The value is a dict of
        JAX arrays with every key of `table_jacobian`'s ``values`` except
        the energies ``k`` and ``k_below``. ``jax.grad`` and ``jax.vjp``
        differentiate ``table``.
    params : tuple of str
        Names of the anisotropy parameters in ``theta``: ``("beta_c",)``,
        ``("r_a",)`` or ``("beta_c", "r_a")``.
    values : dict of ndarray
        Arrays of the table at the component's own parameters, energies
        included, as `table_jacobian` returns them.

    Raises
    ------
    InputError
        If the anisotropy is COM Type II, or the DF's center lies on a
        branch that `table_jacobian` rejects.
    InputError
        If `component` is on a `TruncatedProfile` that Method II does not
        invert, as in `table_jacobian`.
    AccuracyError
        If a value of the DF inside the table is not finite, is 0 or
        exceeds :math:`10^{300}`, or the tail's logarithmic exponent below
        the floor is not finite.
    UnphysicalModelError
        If `component` is on a `TruncatedProfile` whose model is truncated
        but not spatially finite.
    ResultError
        If a derivative of the auxiliary density is not finite at the
        energies, or if the DF of a component on a `TruncatedProfile` has a
        defective band at the top of its energies.

    See Also
    --------
    table_jacobian : The arrays of the DF's table and their derivatives as
        arrays.
    make_inversion_rows : The DF at the energies of the inversion as a JAX
        function.

    Notes
    -----
    The call computes the table and its derivatives once, at the
    component's own parameters. Each call of ``table`` inverts the
    component again at the new parameters, with the held energies. Each
    derivative of ``table`` calls `table_jacobian`. The steps of the finite
    differences in :math:`\beta_{\rm c}` are measured once, at the
    component's own parameters, and are used at every ``theta``.

    References
    ----------
    Notes Eq. (T:7.1).

    Examples
    --------
    >>> import numpy as np
    >>> import pianistpy as pp
    >>> from pianistpy.df.gradient import make_inversion_table
    >>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
    >>> table, params, values = make_inversion_table(halo, pp.Settings())
    >>> params
    ('r_a',)
    >>> "k" in values, "k" in table(np.array([10.0]))
    (True, False)
    """
    import jax
    import jax.numpy as jnp

    inv = _invert()
    if isinstance(component.profile, TruncatedProfile):
        return _make_child_table(component, settings)
    _, params = _checked(component)
    if q is None:
        q = inversion_nodes(component, settings)
    q = np.asarray(q, dtype=np.float64)
    # The steps of the finite differences in beta_c are measured here at the
    # component's own parameters and reused at every theta, so that the
    # derivative at one theta does not depend on the theta of earlier calls.
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
    """Return `make_inversion_table`'s function, parameters and values for a
    component on a truncated profile (Method II), with the grid and the
    kept nodes held at the component's own parameters."""
    import jax
    import jax.numpy as jnp

    _, params = _child_checked(component)
    held = _child_held(component, settings)
    # The steps of the finite differences in beta_c are measured at the
    # component's own parameters and reused at every theta, as in
    # `make_inversion_table`.
    steps: dict[tuple, dict[str, Any]] = {}
    values, _, _ = _child_jacobian(component, settings, held, steps, placement=False)
    keys = [k for k in values if k != "k"]
    at = _at(component, params)

    def host_f(theta):
        out = _child_forward(at(theta), settings, held)
        return {k: np.asarray(out[k], np.float64) for k in keys}

    def host_j(theta):
        _, jac, _ = _child_jacobian(at(theta), settings, held, steps, placement=False)
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
