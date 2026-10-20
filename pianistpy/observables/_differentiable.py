r"""The observables as JAX functions of a component's anisotropy.

`Differentiable` holds a model of one component on a truncated profile whose
anisotropy :math:`(\beta_{\rm c}, R_{\rm a})`, ``theta``, is the variable. Each
of its methods is a JAX function of theta with a custom VJP: its value is the
plain call on the model rebuilt at theta on the host (at the model's own theta,
the plain call's own value), and its derivative is the tangent rows of the
observable in theta on the tables' own rows, their placement frozen, as the
likelihood's free components carry it. The machinery of a free component
(`_Free`, its tables at any theta and their tangent rows) lives here and the
likelihood reads it.
"""

__all__ = ["Differentiable"]

import math
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..anisotropy import Anisotropy
from ..df import Model, compute_df, truncate
from ..df.gradient import _child_checked, table_jacobian
from ..exceptions import (
    AccuracyError,
    InputError,
    ResultError,
    UnphysicalModelError,
)
from ..profiles import TruncatedProfile
from ..runtime import log
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..validation import refuse_unless_asked
from ._aperture import aperture_mean_square
from ._dispersion import beta_sigma_tangent
from ._panels import panels, read_panels_vjp
from ._rows_theta import losvd_rows_explicit
from ._table_vjp import (
    DF_PARAMETERS,
    aperture_losvd_vjp,
    losvd_read_vjp,
    losvd_rows_vjp,
    speed_distribution_vjp,
)
from ._table_vjp import _add as add_cotangents
from .aperture import Aperture
from .stages import (
    INNER_DISK_TOLERANCE,
    TANGENT_PARAMETERS,
    aperture_dispersion,
    dispersion_profile,
    dispersion_table_reader,
    projected_dispersion,
    speed_distribution,
    surface_density_reader,
)
from .stages import aperture_losvd as stage_aperture_losvd

# A free component's anisotropy (beta_c, R_a): its parameters' names, the
# theta whose tables are kept, and the rows its table may hold as a multiple
# of its rows at its own theta (the fixed shape of the traced reading).
THETA_NAMES = ("beta_c", "r_a")
THETA_KEPT = 8
BUCKET_FACTOR = 4
# The families whose observables are not differentiable, and why (the
# distribution function's own rule refuses them: docs/guide/gradient.rst).
REFUSED_FAMILIES = {
    "COM_II": "COM Type II: its interior point at P(R_a) moves with both parameters",
}


class _Free:
    r"""A component whose anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` is a
    parameter, on the host: its projected table at any theta (the DF inverted
    on its truncated profile, the gate, then the two tables), with the tangent
    rows of :math:`\sigma_{\rm los}^2` in theta on the table's own rows, their
    placement frozen (`projected_dispersion`'s `tangents`); kept for the last
    `THETA_KEPT` theta.

    The tangent rows are taken in the family's own parameters: at
    :math:`\beta_{\rm c} = 0` the family is OM, and until its rows are taken in
    :math:`\beta_{\rm c}` too (the COM Type I tangent at :math:`\beta_{\rm c} = 0`,
    asked of the observables) the derivative in :math:`\beta_{\rm c}` there is
    not carried (NaN). An isotropic or constant member has no :math:`R_{\rm a}`:
    its :math:`R_{\rm a}` column is held at its own (infinite) value, with a
    zero gradient."""

    def __init__(self, member: Model, settings: Settings) -> None:
        anisotropy = member.components[0].anisotropy
        self.member = member
        self.settings = settings
        self.theta = np.array([anisotropy.beta_c, anisotropy.r_a], dtype=np.float64)
        self.held = math.isinf(anisotropy.r_a)
        profile, self.projected = _tangent_tables(member, settings)
        self._own = self.theta.tobytes()
        self._kept: dict[bytes, Any] = {self._own: (profile, self.projected)}
        self._tangents: dict[bytes, Array] = {}
        self.bucket = BUCKET_FACTOR * int(np.size(self.projected.r_2d))

    def placed(self, theta: np.ndarray) -> np.ndarray:
        r"""Theta as the member reads it: a held :math:`R_{\rm a}` column at its
        own value."""
        theta = np.array(theta, dtype=np.float64)
        if self.held:
            theta[1] = self.theta[1]
        return theta

    def tables(self, theta: np.ndarray) -> Any:
        """The dispersion profile and projected table at theta, or None where
        the model at theta is refused."""
        theta = self.placed(theta)
        key = theta.tobytes()
        if key not in self._kept:
            component = self.member.components[0]
            try:
                anisotropy = Anisotropy(
                    beta_c=float(theta[0]),
                    r_a=float(theta[1]),
                    kind=component.anisotropy.kind,
                )
                model = truncate(
                    compute_df(
                        replace(component, anisotropy=anisotropy), self.member.settings
                    )
                )
                entry = _tangent_tables(model, self.settings)
            except (
                InputError,
                UnphysicalModelError,
                AccuracyError,
                ResultError,
            ) as error:
                log.info(
                    "The anisotropy (beta_c, R_a) = (%r, %r) of %r is refused: %s",
                    float(theta[0]),
                    float(theta[1]),
                    component.name,
                    error,
                )
                entry = None
            if len(self._kept) >= THETA_KEPT:
                oldest = next(k for k in self._kept if k != self._own)
                del self._kept[oldest]
                self._tangents.pop(oldest, None)
            self._kept[key] = entry
        return self._kept[key]

    def forward(self, theta: np.ndarray) -> tuple[Array, Array, Array]:
        r"""The breaks and coefficients of the :math:`\sigma_{\rm los}^2`
        reading at theta, padded to the bucket, and whether theta is valid; a
        refused theta reads its own theta's table."""
        theta = self.placed(theta)
        entry = self.tables(theta)
        valid = entry is not None
        projected = (entry if valid else self._kept[self._own])[1]
        spline = dispersion_table_reader(
            projected.r_2d, np.asarray(projected.sigma_los[0]) ** 2
        )._spline
        n = spline.c.shape[1]
        if n > self.bucket:
            log.warning(
                "The projected table at (beta_c, R_a) = (%r, %r) has %d rows, more "
                "than the %d its likelihood reads: refused.",
                float(theta[0]),
                float(theta[1]),
                n + 1,
                self.bucket,
            )
            return self.forward(self.theta)[:2] + (np.float64(0.0),)
        breaks = np.concatenate(
            [spline.x, spline.x[-1] + np.arange(1.0, self.bucket - n + 1.0)]
        )
        coefficients = np.zeros((4, self.bucket))
        coefficients[:, :n] = spline.c
        return breaks, coefficients, np.float64(valid)

    def tangents(self, theta: np.ndarray) -> Array:
        r"""The coefficients of the tangent rows' readings in
        :math:`\beta_{\rm c}` and :math:`R_{\rm a}`, padded to the bucket, kept
        with the tables; zero at a refused theta and in a held
        :math:`R_{\rm a}`, NaN in a parameter the family's rows are not taken
        in."""
        theta = self.placed(theta)
        out = np.zeros((2, 4, self.bucket))
        entry = self.tables(theta)
        if entry is None:
            return out
        key = theta.tobytes()
        if key in self._tangents:
            return self._tangents[key]
        projected = entry[1]
        for i, name in enumerate(THETA_NAMES):
            if name == "r_a" and self.held:
                continue
            if name not in projected.tangents:
                out[i] = np.nan
                continue
            rows = np.asarray(projected.tangents[name])[0]
            c = dispersion_table_reader(projected.r_2d, rows)._spline.c
            out[i, :, : c.shape[1]] = c
        self._tangents[key] = out
        return out


def _tangent_tables(model: Model, settings: Settings) -> Any:
    """The dispersion profile and projected table of a free component, with
    the tangent rows in its family's own anisotropy parameters."""
    family = model.components[0].anisotropy.family
    names = tuple(n for n in THETA_NAMES if n in TANGENT_PARAMETERS[family])
    profile = dispersion_profile(model, settings, tangents=names)
    return profile, projected_dispersion(model, profile, settings, tangents=names)


def _free_reading(free: _Free) -> Any:
    r"""The :math:`\sigma_{\rm los}^2` reading of a free component as a JAX
    function of its theta: the breaks, coefficients and validity at theta
    through a host callback, with a custom VJP whose backward contracts the
    coefficients' cotangent with the tangent rows' coefficients (the
    placement frozen: the breaks carry no gradient)."""
    shapes = (
        jax.ShapeDtypeStruct((free.bucket + 1,), jnp.float64),
        jax.ShapeDtypeStruct((4, free.bucket), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    tangent_shape = jax.ShapeDtypeStruct((2, 4, free.bucket), jnp.float64)

    @jax.custom_vjp
    def reading(theta: Array) -> Any:
        return jax.pure_callback(free.forward, shapes, theta)

    def forward(theta: Array) -> Any:
        out = reading(theta)
        return out, (theta, out[2])

    def backward(residuals: Any, cotangent: Any) -> Any:
        theta, valid = residuals
        tangents = jax.pure_callback(free.tangents, tangent_shape, theta)
        gradient = jnp.tensordot(tangents, cotangent[1], axes=2)
        return (jnp.where(valid > 0.0, gradient, 0.0),)

    reading.defvjp(forward, backward)
    return reading


def refuse_unless_free(model: Model, name: str) -> None:
    r"""Refuse, by name, a component whose anisotropy cannot be a parameter:
    not spatially finite, not a DF on a truncated profile that carries its
    parent's DF, or a family with neither :math:`\beta_{\rm c}` nor
    :math:`R_{\rm a}` to free."""
    component = model.components[0]
    if not model.spatially_finite:
        raise InputError(
            f"{name} is not spatially finite: its anisotropy cannot be "
            "free (its edge terms read its DF)."
        )
    if not isinstance(component.profile, TruncatedProfile) or (
        np.size(component.profile.df_k) == 0
    ):
        raise InputError(
            f"{name} is not a DF on a truncated profile that carries its "
            "parent's DF (a model made by `truncate` on such a profile): its "
            "anisotropy cannot be free. A Method I truncated model's profile is "
            "its own DF's and moves with beta_c and R_a, so neither the Jeans "
            "path (a fixed profile) nor the DF path at a fixed potential carries "
            "the full derivative; the anisotropy is a parameter of a Method II "
            "model, a DF on another model's truncated profile, held fixed."
        )
    family = component.anisotropy.family
    if not any(n in TANGENT_PARAMETERS[family] for n in THETA_NAMES):
        raise InputError(
            f"{name}'s anisotropy ({family}) has neither beta_c nor R_a to free."
        )


class Differentiable:
    r"""A model's observables as JAX functions of its anisotropy.

    Built once from a model of one component on a truncated profile (a DF
    inverted on its parent's profile, as `truncate` makes it): its tables are
    computed here, at the model's own anisotropy. Each method takes
    ``theta = (beta_c, R_a)``, the component's anisotropy in internal units,
    first, and is a JAX function of it, traceable and differentiable.

    Parameters
    ----------
    model : Model
        A model of one component on a truncated profile, spatially finite,
        whose family has :math:`\beta_{\rm c}` or :math:`R_{\rm a}`: constant
        anisotropy (:math:`\beta_{\rm c}`; an isotropic or constant
        component's :math:`R_{\rm a}` is held at its own, infinite, value with
        a zero gradient), Cuddeford-Osipkov-Merritt (COM) Type I (both),
        Osipkov-Merritt Type I and Type II (:math:`R_{\rm a}`).
    settings : Settings, optional
        Numerical settings of the observables; the model's by default.

    Attributes
    ----------
    model : Model
        The model.
    theta : array, shape (2,)
        Its own :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.

    Raises
    ------
    InputError
        For a superposition or a model of several components, a component not
        spatially finite or not on a truncated profile that carries its
        parent's DF, and a family whose observables are not differentiable
        (COM Type II), each named.

    Notes
    -----
    Radii and velocities are in internal units, whatever the model's unit
    system: theta, as the likelihood's, is internal, and so is everything a
    method reads and returns.

    A method's value at a theta is computed on the host the first time the
    method meets it: the DF inverted at theta on the component's truncated
    profile, then the observable's tables, at the cost of the likelihood's
    own rebuild of a free component (about that of the plain call on the
    model rebuilt at theta); the last eight theta are kept. At a theta where
    the model is refused the value is NaN and its gradient zero. The
    derivative is the observable's tangent rows in theta on the tables' own
    rows, their placement frozen: between two placements the value moves by
    the tables' tolerance, which the derivative does not carry. A method is
    not differentiable in its radii (their gradient is NaN).
    """

    def __init__(self, model: Model, settings: Settings | None = None) -> None:
        if not isinstance(model, Model) or len(model.components) != 1:
            raise InputError(
                "Differentiable takes a Model of one component; a superposition's "
                "components are differentiated through the likelihood."
            )
        refuse_unless_free(model, "The model's component")
        refuse_unless_asked(
            f"The model {model.components[0].name!r}", model.validation, model.settings
        )
        family = model.components[0].anisotropy.family
        if family in REFUSED_FAMILIES:
            raise InputError(
                f"The observables of a {family} component are not differentiable: "
                f"{REFUSED_FAMILIES[family]}."
            )
        refusal = _df_refusal(model.components[0])
        self.model = model
        self.settings = model.settings if settings is None else settings
        self._free = _Free(model, self.settings)
        self.theta = jnp.asarray(self._free.theta)
        self._sigma_r = _radial_function(self._free, _sigma_r_host(self._free, 0))
        self._sigma_t = _radial_function(self._free, _sigma_r_host(self._free, 1))
        self._sigma_los = _radial_function(self._free, _sigma_los_host(self._free))
        self._df = _DfPath(self._free, refusal)
        self._losvd = _losvd_function(self._df)
        self._pdf_v = _pdf_v_function(self._df)

    def sigma_r(self, theta: ArrayLike, r: ArrayLike) -> Array:
        r"""Return the radial velocity dispersion at radii, as
        `pianistpy.sigma_r` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        r : array-like
            Radii, in internal units.

        Returns
        -------
        sigma_r : array
            The dispersion in internal units, with the shape of `r`; NaN where
            the model at theta is refused.
        """
        return self._sigma_r(_theta(theta), jnp.asarray(r, dtype=jnp.float64))

    def sigma_t(self, theta: ArrayLike, r: ArrayLike) -> Array:
        r"""Return the tangential velocity dispersion at radii (both tangential
        components together), as `pianistpy.sigma_t` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        r : array-like
            Radii, in internal units.

        Returns
        -------
        sigma_t : array
            The dispersion in internal units, with the shape of `r`; NaN where
            the model at theta is refused.
        """
        return self._sigma_t(_theta(theta), jnp.asarray(r, dtype=jnp.float64))

    def sigma_los(self, theta: ArrayLike, r_2d: ArrayLike) -> Array:
        r"""Return the line-of-sight velocity dispersion at projected radii, as
        `pianistpy.sigma_los` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        r_2d : array-like
            Projected radii, in internal units.

        Returns
        -------
        sigma_los : array
            The dispersion in internal units, with the shape of `r_2d`; zero
            beyond the last row, NaN where the model at theta is refused.
        """
        return self._sigma_los(_theta(theta), jnp.asarray(r_2d, dtype=jnp.float64))

    def sigma_los_aperture(self, theta: ArrayLike, aperture: Aperture) -> Array:
        r"""Return the line-of-sight velocity dispersion averaged over an
        aperture, weighted by the surface density, as
        `pianistpy.sigma_los_aperture` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        aperture : Aperture
            The aperture, its lengths in internal units.

        Returns
        -------
        sigma_los : scalar array
            The dispersion in internal units; NaN where the model at theta is
            refused.

        Notes
        -----
        The derivative is the aperture's integral of the tangent rows of
        :math:`\sigma_{\rm los}^2`, on the nodes and weights of the plain call's own
        quadrature (the integral is linear in the rows), over twice the value.
        """
        value, slope = _aperture_host(self._free, aperture)
        return _radial_function(self._free, (value, slope))(
            _theta(theta), jnp.zeros((), dtype=jnp.float64)
        )

    def losvd(self, theta: ArrayLike, r_2d: ArrayLike, v: ArrayLike) -> Array:
        r"""Return the line-of-sight velocity distribution at the pairs
        ``(r_2d, v)``, as `pianistpy.losvd` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        r_2d, v : array-like
            Projected radii and line-of-sight velocities, in internal units;
            they broadcast against each other.

        Returns
        -------
        losvd : array
            The distribution per unit internal velocity, with the broadcast
            shape; NaN where the model at theta is refused.

        Notes
        -----
        The derivative is the DF path's, its placement frozen: the reading's
        transpose onto the table's rows (`losvd_read_vjp`), the rows' onto the
        DF's table and forms (`losvd_rows_vjp`) with the rows' explicit part
        in the anisotropy, and the DF table's derivative in theta
        (`df.gradient.table_jacobian`). Not differentiable in ``r_2d`` or v.
        """
        self._df.served()
        x, u = jnp.broadcast_arrays(
            jnp.asarray(r_2d, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )
        return self._losvd(jnp.asarray(theta, dtype=jnp.float64), x, u)

    def pdf_v(self, theta: ArrayLike, r: ArrayLike, v: ArrayLike) -> Array:
        r"""Return the local speed distribution at the pairs (r, v), as
        `pianistpy.pdf_v` of the model at theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        r, v : array-like
            Radii and speeds, in internal units; they broadcast against each
            other.

        Returns
        -------
        pdf_v : array
            The distribution per unit internal velocity, with the broadcast
            shape; NaN where the model at theta is refused.

        Notes
        -----
        The derivative is the DF path's: at each radius the speed
        distribution's transpose (`speed_distribution_vjp`) at the speeds
        asked, the distribution there computed by the rule its record
        checked, as `SpeedDistribution.read` computes it, with its explicit
        part in the anisotropy, and the DF table's derivative in theta. Not
        differentiable in r or v.
        """
        self._df.served()
        x, u = jnp.broadcast_arrays(
            jnp.asarray(r, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )
        return self._pdf_v(jnp.asarray(theta, dtype=jnp.float64), x, u)

    def aperture_losvd(
        self, theta: ArrayLike, aperture: Aperture, v: ArrayLike
    ) -> Array:
        r"""Return the line-of-sight velocity distribution inside an aperture
        at the velocities v, as `pianistpy.aperture_losvd` of the model at
        theta.

        Parameters
        ----------
        theta : array-like, shape (2,)
            The component's :math:`(\beta_{\rm c}, R_{\rm a})`, in internal units.
        aperture : Aperture
            The aperture, its lengths in internal units.
        v : array-like
            Line-of-sight velocities, in internal units.

        Returns
        -------
        losvd : array
            The distribution per unit internal velocity, with the shape of
            `v`; NaN where the model at theta is refused.

        Notes
        -----
        The derivative is the DF path's: the transpose of the reading of the
        aperture's panels (each panel's polynomial through its nodes) onto
        its values at the nodes, theirs onto the table's rows
        (`aperture_losvd_vjp`), then as `losvd`. Not differentiable in v.
        """
        self._df.served()
        if not isinstance(aperture, Aperture):
            raise InputError("aperture_losvd reads an Aperture.")
        function = _aperture_function(self._df, aperture)
        return function(
            jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )


def _theta(theta: ArrayLike) -> Array:
    return jnp.asarray(theta, dtype=jnp.float64)


def _radial_function(free: _Free, host: tuple[Any, Any]) -> Any:
    """An observable of a free component as a JAX function of (theta, x): its
    value and its derivative in theta (shape (2,) + x's) computed on the host
    by the pair `host`, with a custom VJP; the derivative in x is not carried
    (NaN)."""
    value, slope = host

    @jax.custom_vjp
    def function(theta: Array, x: Array) -> Array:
        return jax.pure_callback(
            value, jax.ShapeDtypeStruct(x.shape, jnp.float64), theta, x
        )

    def forward(theta: Array, x: Array) -> Any:
        return function(theta, x), (theta, x)

    def backward(residuals: Any, cotangent: Array) -> Any:
        theta, x = residuals
        rows = jax.pure_callback(
            slope, jax.ShapeDtypeStruct((2,) + x.shape, jnp.float64), theta, x
        )
        gradient = jnp.sum(rows * cotangent, axis=tuple(range(1, rows.ndim)))
        return gradient, jnp.full_like(x, jnp.nan)

    function.defvjp(forward, backward)
    return function


def _each_parameter(free: _Free, entry: Any, tangents: Any, out: np.ndarray) -> Any:
    r"""The parameters whose derivative is computed: yields (i, name) for each
    the family carries; leaves a held :math:`R_{\rm a}` at zero and sets one
    the family's rows are not taken in to NaN."""
    for i, name in enumerate(THETA_NAMES):
        if name == "r_a" and free.held:
            continue
        if name not in tangents:
            out[i] = np.nan
            continue
        yield i, name


def _sigma_r_host(free: _Free, which: int) -> tuple[Any, Any]:
    r""":math:`\sigma_r` (which 0) or :math:`\sigma_t` (which 1) on the host:
    the value read from the dispersion profile at theta as the plain call
    reads it; the derivative from the profile's tangent reading of
    :math:`\sigma_r^2` and, for
    :math:`\sigma_t^2 = 2 (1 - \beta_\sigma) \sigma_r^2`, from
    :math:`\beta_\sigma`'s tangent."""

    def value(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        entry = free.tables(np.asarray(theta))
        if entry is None:
            return np.full(x.shape, np.nan)
        out = entry[0].read(x.ravel())[which][0]
        return np.asarray(out, dtype=np.float64).reshape(x.shape)

    def slope(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros((2,) + x.shape)
        entry = free.tables(free.placed(np.asarray(theta)))
        if entry is None:
            return out
        profile = entry[0]
        flat = x.ravel()
        sigma_r, sigma_t, beta = (np.asarray(a)[0] for a in profile.read(flat))
        anisotropy = profile.model.components[0].anisotropy
        for i, name in _each_parameter(free, entry, profile.tangents, out):
            d_square = np.asarray(profile._read_tangent(name)(flat), dtype=np.float64)
            if which == 1:
                d_beta = beta_sigma_tangent(flat, anisotropy, name)
                d_square = 2.0 * (1.0 - beta) * d_square - 2.0 * sigma_r**2 * d_beta
            own = sigma_r if which == 0 else sigma_t
            positive = own > 0.0
            d = np.where(positive, d_square / (2.0 * np.where(positive, own, 1.0)), 0.0)
            out[i] = d.reshape(x.shape)
        return out

    return value, slope


def _sigma_los_host(free: _Free) -> tuple[Any, Any]:
    r""":math:`\sigma_{\rm los}` on the host: the value read from the projected
    table at theta as the plain call reads it; the derivative the tangent rows
    of :math:`\sigma_{\rm los}^2` read at the same radii (placement frozen)
    over twice the value, zero beyond the last row."""

    def value(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        entry = free.tables(np.asarray(theta))
        if entry is None:
            return np.full(x.shape, np.nan)
        out = entry[1].read(x.ravel())[1][0]
        return np.asarray(out, dtype=np.float64).reshape(x.shape)

    def slope(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros((2,) + x.shape)
        entry = free.tables(free.placed(np.asarray(theta)))
        if entry is None:
            return out
        projected = entry[1]
        own = np.asarray(projected.read(x.ravel())[1][0]).reshape(x.shape)
        inside = (x > 0.0) & (x <= projected.r_2d[-1]) & (own > 0.0)
        log_x = np.log(np.where(inside, x, projected.r_2d[-1]))
        for i, name in _each_parameter(free, entry, projected.tangents, out):
            rows = np.asarray(projected.tangents[name])[0]
            spline = dispersion_table_reader(projected.r_2d, rows)._spline
            out[i] = np.where(
                inside, spline(log_x) / (2.0 * np.where(inside, own, 1.0)), 0.0
            )
        return out

    return value, slope


def _aperture_host(free: _Free, aperture: Aperture) -> tuple[Any, Any]:
    r""":math:`\sigma_{\rm los}` over an aperture on the host: the value as the
    plain call computes it (`aperture_dispersion` on the projected table at
    theta); the derivative the same quadrature of the tangent rows of
    :math:`\sigma_{\rm los}^2` (its nodes and weights, the integral linear in
    the rows) over twice the value."""
    observables = free.settings.observables

    def mean_square(projected: Any, rows: Array) -> float:
        model = projected.model
        surface = surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        )
        return aperture_mean_square(
            surface,
            # a tangent row is signed: read it before the reader's clip at zero
            dispersion_table_reader(projected.r_2d, np.asarray(rows)).raw,
            aperture,
            int(observables["n_aperture_nodes"]),
            int(observables["aperture_nodes_per_decade"]),
            INNER_DISK_TOLERANCE,
        )

    def value(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        entry = free.tables(np.asarray(theta))
        if entry is None:
            return np.full(np.shape(x), np.nan)
        return np.asarray(aperture_dispersion(entry[1], aperture, free.settings)[0])

    def slope(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros((2,) + np.shape(x))
        entry = free.tables(free.placed(np.asarray(theta)))
        if entry is None:
            return out
        projected = entry[1]
        own = float(aperture_dispersion(projected, aperture, free.settings)[0])
        for i, name in _each_parameter(free, entry, projected.tangents, out):
            rows = np.asarray(projected.tangents[name])[0]
            out[i] = mean_square(projected, rows) / (2.0 * own) if own > 0.0 else 0.0
        return out

    return value, slope


# The numbers of the DF's center that the table's derivative carries, in the
# order of `df.gradient.table_jacobian`'s ``center``.
_CENTER_LEAVES = (
    "central_exponent",
    "central_amplitude",
    "central_next_exponent",
    "central_next_coefficient",
)

# The DF path's refusal of an isotropic member (`DF_PARAMETERS`).
ISOTROPIC_DF = (
    "an isotropic member's beta_c is not taken: the derivative of its child's "
    "DF table from beta_c = 0 needs orders of the sources that the table's "
    "held sources do not carry, and its rows read G with no L factor."
)


def _df_refusal(component: Any) -> str | None:
    """Why the DF table's rule refuses a component (`df.gradient`: its center
    on a critical branch, or no parameter to vary), or None where it serves
    it."""
    try:
        _child_checked(component)
    except InputError as error:
        return str(error)
    return None


class _DfPath:
    r"""The DF path of a free component on the host: the model at any theta
    (the member itself at its own), its LosTable and its dispersion profile
    when first asked, and the gradient in theta of a cotangent on its DF table
    and forms (`df.gradient.table_jacobian`, the grid and the kept nodes held
    at that theta) with the explicit part; kept for the last `THETA_KEPT`
    theta.

    The parameters are the DF path's (`DF_PARAMETERS`): an Osipkov-Merritt
    member's :math:`\beta_{\rm c}` is not taken (NaN), and a held
    :math:`R_{\rm a}` has a zero gradient. A member the DF table's rule refuses
    (its center on a critical branch) is named at construction, and its DF
    methods refuse it by that name; so is an isotropic member, whose
    :math:`\beta_{\rm c}` the DF path does not take (`DF_PARAMETERS`)."""

    def __init__(self, free: _Free, refusal: str | None) -> None:
        self.free = free
        family = free.member.components[0].anisotropy.family
        self.names = tuple(
            n for n in DF_PARAMETERS.get(family, ()) if not (n == "r_a" and free.held)
        )
        if refusal is None and family == "isotropic":
            refusal = ISOTROPIC_DF
        self.refusal = refusal
        self._kept: dict[bytes, dict[str, Any] | None] = {}
        # the DF table derivative's exponent steps, measured at the member's
        # own theta and carried to every theta (`df.gradient.table_jacobian`)
        self._steps: dict[tuple, dict[str, Any]] = {}

    def served(self) -> None:
        """Refuse, by name, a member whose DF table's rule refuses it."""
        if self.refusal is not None:
            raise InputError(
                f"The DF path of the model's component is not differentiable: "
                f"{self.refusal}"
            )

    def entry(self, theta: np.ndarray) -> dict[str, Any] | None:
        """The model at theta and what has been made from it, or None where
        the model at theta is refused."""
        free = self.free
        theta = free.placed(np.asarray(theta))
        key = theta.tobytes()
        if key not in self._kept:
            if key == free._own:
                model = free.member
            else:
                component = free.member.components[0]
                try:
                    anisotropy = Anisotropy(
                        beta_c=float(theta[0]),
                        r_a=float(theta[1]),
                        kind=component.anisotropy.kind,
                    )
                    model = truncate(
                        compute_df(
                            replace(component, anisotropy=anisotropy),
                            free.member.settings,
                        )
                    )
                except (
                    InputError,
                    UnphysicalModelError,
                    AccuracyError,
                    ResultError,
                ) as error:
                    log.info(
                        "The anisotropy (beta_c, R_a) = (%r, %r) of %r is refused: %s",
                        float(theta[0]),
                        float(theta[1]),
                        component.name,
                        error,
                    )
                    model = None
            if model is not None:
                model = model.to("internal")
                if free.settings is not model.settings:
                    model = replace(model, settings=free.settings)
            if len(self._kept) >= THETA_KEPT:
                oldest = next(k for k in self._kept if k != free._own)
                del self._kept[oldest]
            self._kept[key] = None if model is None else {"model": model}
        return self._kept[key]

    def los_table(self, entry: dict[str, Any]) -> Any:
        """The LosTable of an entry's model, made when first asked."""
        if "los_table" not in entry:
            from .core import LosTable

            entry["los_table"] = LosTable(entry["model"])
        return entry["los_table"]

    def _jacobian(self, entry: dict[str, Any]) -> tuple[Any, Any]:
        """The DF table's derivative at an entry's theta, made when first
        asked, with the steps of the store (`_steps`)."""
        if "jac" not in entry:
            model = entry["model"]
            values, jac, _ = table_jacobian(
                model.components[0], model.settings, steps=self._steps
            )
            if values["k"].tobytes() != np.asarray(model.table.k[0]).tobytes():
                raise ResultError(
                    "The DF table's derivative is not on the model's own grid: the "
                    "table was not made by the rule its derivative holds."
                )
            entry["jac"] = (values, jac)
        return entry["jac"]

    def gradient(
        self, entry: dict[str, Any], cotangent: Any, explicit: dict[str, float]
    ) -> np.ndarray:
        """The gradient in theta of a cotangent on the entry's DF table and
        forms, with the explicit part in each parameter."""
        if "jac" not in entry and not self._steps:
            # the steps are measured at the member's own theta first
            own = self.entry(self.free.theta)
            if own is not None and own is not entry:
                self._jacobian(own)
        values, jac = self._jacobian(entry)
        carried = _carried(cotangent, values)
        out = np.zeros(2)
        for i, name in enumerate(THETA_NAMES):
            if name == "r_a" and self.free.held:
                continue
            if name not in self.names:
                out[i] = np.nan
                continue
            d = jac[name]
            c = np.asarray(cotangent.log_df, dtype=np.float64)[0]
            dl = np.asarray(d["log_df"], dtype=np.float64)
            g = float(np.sum(np.where(c != 0.0, c * np.where(c != 0.0, dl, 0.0), 0.0)))
            if "edge_law" in d:
                g += float(np.dot(carried["below_floor_coefficients"], d["edge_law"]))
            center = np.asarray(d["center"], dtype=np.float64)
            g += float(np.dot(carried["center"][: center.size], center))
            out[i] = g + float(explicit.get(name, 0.0))
        return out


def _carried(cotangent: Any, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The cotangent's leaves the DF table's derivative carries (the floor's
    edge law and the center's numbers), refusing by name a cotangent on a
    leaf it does not carry: the observable would read a part of the DF whose
    motion in theta is not held."""
    form = dict(cotangent.form)
    law = np.asarray(form.pop("below_floor_coefficients", ()), dtype=np.float64)
    n_law = np.size(values.get("edge_law", ()))
    n_center = np.size(values["center"])
    center = np.array([float(form.pop(k, 0.0)) for k in _CENTER_LEAVES])
    loose = [k for k, x in form.items() if np.any(np.asarray(x, dtype=np.float64))]
    if np.any(law[n_law:]):
        loose.append("below_floor_coefficients")
    if np.any(center[n_center:]):
        loose.extend(_CENTER_LEAVES[n_center:])
    for name in ("log_df_slope", "log_df_below", "log_df_slope_below"):
        if np.any(np.asarray(getattr(cotangent, name))):
            loose.append(name)
    if loose:
        raise ResultError(
            "The observable reads parts of the DF whose derivative in theta the "
            f"table's rule does not carry: {', '.join(sorted(set(loose)))}."
        )
    return {
        "below_floor_coefficients": law[:n_law],
        "center": center[:n_center],
    }


def _df_function(value: Any, backward: Any) -> Any:
    """A JAX function of (theta, *arrays) computed on the host, its output the
    shape of the last array, with a custom VJP whose backward is the host's
    gradient in theta of the output's cotangent; the arrays carry no
    gradient (NaN)."""

    @jax.custom_vjp
    def function(theta: Array, *arrays: Array) -> Array:
        return jax.pure_callback(
            value, jax.ShapeDtypeStruct(arrays[-1].shape, jnp.float64), theta, *arrays
        )

    def forward(theta: Array, *arrays: Array) -> Any:
        return function(theta, *arrays), (theta, arrays)

    def back(residuals: Any, cotangent: Array) -> Any:
        theta, arrays = residuals
        gradient = jax.pure_callback(
            backward, jax.ShapeDtypeStruct((2,), jnp.float64), theta, *arrays, cotangent
        )
        return (gradient, *(jnp.full_like(a, jnp.nan) for a in arrays))

    function.defvjp(forward, back)
    return function


def _rows_gradient(
    path: _DfPath, entry: dict[str, Any], part: Any, rows: list[Array]
) -> np.ndarray:
    """The gradient in theta of cotangents on a LosTable part's rows: the
    rows' transpose onto the DF's table and forms, and their explicit part."""
    settings = part.model.settings
    cotangent = losvd_rows_vjp(part.model, part.table, settings, rows)
    explicit = (
        losvd_rows_explicit(part.model, part.table, settings, rows, path.names)
        if path.names
        else {}
    )
    return path.gradient(entry, cotangent, explicit)


def _losvd_function(path: _DfPath) -> Any:
    """`losvd` of a free component as a JAX function of ``(theta, r_2d, v)``."""
    from .core import losvd

    def value(theta: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.full(np.shape(u), np.nan)
        out = losvd(path.los_table(entry), np.asarray(x), np.asarray(u))
        return np.asarray(out, dtype=np.float64)

    def backward(theta, x, u, cotangent) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.zeros(2)
        part = path.los_table(entry).parts[0]
        rows = losvd_read_vjp(
            part,
            np.asarray(x, dtype=np.float64).ravel(),
            np.asarray(u, dtype=np.float64).ravel(),
            np.asarray(cotangent, dtype=np.float64).ravel(),
        )
        return _rows_gradient(path, entry, part, rows)

    return _df_function(value, backward)


def _aperture_function(path: _DfPath, aperture: Aperture) -> Any:
    """`aperture_losvd` of a free component in one aperture as a JAX
    function of (theta, v)."""
    from .core import _internal_aperture
    from .core import aperture_losvd as plain

    def value(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.full(np.shape(v), np.nan)
        out = plain(path.los_table(entry), aperture, np.asarray(v))
        return np.asarray(out, dtype=np.float64)

    def backward(theta, v, cotangent) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.zeros(2)
        table = path.los_table(entry)
        part = table.parts[0]
        settings = part.model.settings
        inside = _internal_aperture(table.model, aperture)
        result = stage_aperture_losvd(
            part.model, part.table, part.projected, inside, settings
        )
        rule = panels(result.v_nodes, result.v_weights, result.losvd[0])
        at_nodes = read_panels_vjp(
            rule,
            np.abs(np.asarray(v, dtype=np.float64)).ravel(),
            np.asarray(cotangent, dtype=np.float64).ravel(),
        )
        rows = aperture_losvd_vjp(
            part.model, part.table, part.projected, inside, settings, result, at_nodes
        )
        return _rows_gradient(path, entry, part, rows)

    return _df_function(value, backward)


def _pdf_v_function(path: _DfPath) -> Any:
    """`pdf_v` of a free component as a JAX function of (theta, r, v)."""
    from .core import _profiles, pdf_v

    def value(theta: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.full(np.shape(u), np.nan)
        out = pdf_v(entry["model"], np.asarray(x), np.asarray(u))
        return np.asarray(out, dtype=np.float64)

    def backward(theta, x, u, cotangent) -> np.ndarray:
        entry = path.entry(theta)
        if entry is None:
            return np.zeros(2)
        if "profiles" not in entry:
            _, views, profiles = _profiles(entry["model"], "pdf_v")
            entry["profiles"] = (views[0], profiles[0])
        view, profile = entry["profiles"]
        xs = np.asarray(x, dtype=np.float64).ravel()
        us = np.asarray(u, dtype=np.float64).ravel()
        cs = np.asarray(cotangent, dtype=np.float64).ravel()
        total, explicit = None, dict.fromkeys(path.names, 0.0)
        for radius in np.unique(xs):
            at = xs == radius
            result = speed_distribution(view, float(radius), view.settings, profile)
            speeds, c = us[at], cs[at]
            inside = (speeds >= 0.0) & (speeds <= result.v_max) & (c != 0.0)
            if not np.any(inside):
                continue
            # the distribution at the speeds asked, by the rule its record
            # checked (`SpeedDistribution.read`), and its transpose there
            asked = replace(
                result, v_nodes=speeds[inside], pdf=result.read(speeds[inside])
            )
            cotangent_df, part = speed_distribution_vjp(asked, c[inside], path.names)
            total = (
                cotangent_df if total is None else add_cotangents(total, cotangent_df)
            )
            for name in path.names:
                explicit[name] += float(part[name])
        if total is None:
            return np.zeros(2)
        return path.gradient(entry, total, explicit)

    return _df_function(value, backward)
