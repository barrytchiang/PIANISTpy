r"""The observables of one component as JAX functions of the component's anisotropy.

Users call `Differentiable`, defined in the module `_differentiable`. The
likelihood (`pianistpy.likelihood`) reads the class `_Free` of `_differentiable`
for each component whose anisotropy is a free parameter.

The anisotropy is the pair :math:`\theta = (\beta_{\rm c}, R_{\rm a})` of the
central anisotropy :math:`\beta_{\rm c}` and the anisotropy radius
:math:`R_{\rm a}`. Each method of `Differentiable` is a JAX function of
:math:`\theta` with a custom vector-Jacobian product. The value of a method at
:math:`\theta` is the plain call of the same name as the method (`pianistpy.sigma_los`
for `Differentiable.sigma_los`), evaluated on the host on the model rebuilt at
:math:`\theta`. At the model's own :math:`\theta`, the value of a method
is the plain call on the model itself.

The derivative of the line-of-sight dispersion :math:`\sigma_{\rm los}` in a
parameter :math:`\theta_i` (:math:`\beta_{\rm c}` or :math:`R_{\rm a}`) is

.. math::

    \frac{\partial \sigma_{\rm los}}{\partial \theta_i}
    = \frac{1}{2\sigma_{\rm los}}
    \frac{\partial \sigma_{\rm los}^2}{\partial \theta_i},

with :math:`\partial \sigma_{\rm los}^2/\partial \theta_i` the tangent rows of
the projected table at :math:`\theta`. The tangent rows sit at the projected
radii of the projected table at :math:`\theta`, and the derivative holds the
tabulated projected radii fixed. The projected table places the projected radii so that
the table, read between two tabulated projected radii, differs from the
line-of-sight integral by a relative error of at most :math:`10^{-8}`
(`CERTIFICATE_TOLERANCE`). The derivatives of the radial and tangential
dispersions divide the tangent rows of the squared dispersions in the
dispersion profile by twice the dispersion, at fixed tabulated radii.
The line-of-sight velocity distribution, the aperture's line-of-sight velocity
distribution and the local velocity distribution take their derivatives through
the derivative of the DF table in :math:`\theta` (`df.gradient.table_jacobian`).

References: Notes Section T:7.
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
    RowsBasis,
    aperture_losvd_vjp,
    losvd_read_vjp,
    losvd_rows_vjp,
    velocity_distribution_vjp,
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
    surface_density_reader,
    velocity_distribution,
)
from .stages import aperture_losvd as stage_aperture_losvd

# The names of the two anisotropy parameters of a free component, in the order
# of theta: the central anisotropy beta_c and the anisotropy radius R_a.
THETA_NAMES = ("beta_c", "r_a")
# The largest number of theta whose tables a free component keeps.
THETA_KEPT = 8
# The largest number of rows of the projected table at any theta, as a multiple
# of the number of rows at the component's own theta. The reading of the
# projected table under JAX tracing needs one fixed shape for every theta.
BUCKET_FACTOR = 4
# The anisotropy families whose observables are not differentiable, each with
# the reason. The DF table's derivative (df.gradient) raises InputError for
# the families of REFUSED_FAMILIES. The user guide page
# docs/guide/differentiability.rst lists the families.
REFUSED_FAMILIES = {
    "COM_II": (
        "the COM Type II DF has an interior point at P(R_a), which moves with both "
        "beta_c and R_a"
    ),
}


class _Free:
    r"""A component whose anisotropy :math:`\theta = (\beta_{\rm c}, R_{\rm a})`
    is a free parameter, evaluated on the host.

    For each theta, a `_Free` inverts the DF of the component at theta on the
    component's truncated profile and builds two tables: the dispersion profile
    and the projected table. Both tables carry tangent rows, the derivatives of
    the squared dispersions in theta at the table's own radii
    (`projected_dispersion` with ``tangents``). A `_Free` keeps the tables of
    the component's own theta and of the `THETA_KEPT` - 1 other theta first
    met most recently.

    The tangent rows are taken in the parameters of `THETA_NAMES` that the
    anisotropy family at theta varies in (`TANGENT_PARAMETERS`). The
    derivative in a parameter without tangent rows is NaN. An isotropic or
    constant component has an infinite :math:`R_{\rm a}`. For an isotropic or
    constant component, the :math:`R_{\rm a}` entry of every theta is held at
    infinity, and the derivative in :math:`R_{\rm a}` is zero."""

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
        r"""Return theta as a float64 copy, with :math:`R_{\rm a}` replaced by
        the component's own :math:`R_{\rm a}` when :math:`R_{\rm a}` is held."""
        theta = np.array(theta, dtype=np.float64)
        if self.held:
            theta[1] = self.theta[1]
        return theta

    def tables(self, theta: np.ndarray) -> Any:
        """Return the dispersion profile and the projected table at theta.

        The return value is None when rebuilding the model at theta raises
        InputError, UnphysicalModelError, AccuracyError or ResultError."""
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
                    "The anisotropy (beta_c, R_a) = (%r, %r) of %r gives an invalid "
                    "model, because rebuilding the model with the anisotropy raised "
                    "the error %s",
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
        r"""Return the spline of :math:`\sigma_{\rm los}^2` at theta as three
        arrays: the breaks, the coefficients and a validity flag.

        The breaks and the coefficients are padded to `bucket` intervals. The
        model at theta is invalid when rebuilding the model at theta fails, or
        when the spline at theta has more intervals than `bucket`. The validity
        flag is 1.0 for a valid model at theta and 0.0 for an invalid one. For
        an invalid model at theta, the method returns the breaks and the
        coefficients of the component's own theta."""
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
                "The projected table at the anisotropy (beta_c, R_a) = (%r, %r) has %d "
                "tabulated radii, more than the %d tabulated radii that the "
                "likelihood reads, so the likelihood takes the model at the "
                "anisotropy as invalid.",
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
        r"""Return the spline coefficients of the tangent rows of
        :math:`\sigma_{\rm los}^2` in :math:`\beta_{\rm c}` and
        :math:`R_{\rm a}`, shape (2, 4, `bucket`).

        The coefficients are zero at a theta where the model does not exist
        and in a held :math:`R_{\rm a}`. The coefficients are NaN in a
        parameter that the projected table at theta has no tangent rows in.
        The coefficients at a theta are dropped when the tables at the theta
        are dropped."""
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
    """Return the dispersion profile and the projected table of a free
    component, each with tangent rows in the parameters of `THETA_NAMES` that
    the component's anisotropy family varies in (`TANGENT_PARAMETERS`)."""
    family = model.components[0].anisotropy.family
    names = tuple(n for n in THETA_NAMES if n in TANGENT_PARAMETERS[family])
    profile = dispersion_profile(model, settings, tangents=names)
    return profile, projected_dispersion(model, profile, settings, tangents=names)


def _free_reading(free: _Free) -> Any:
    r"""Return the spline of :math:`\sigma_{\rm los}^2` of a free component as
    a JAX function of theta.

    The function returns the breaks, the coefficients and the validity flag of
    `_Free.forward` through a host callback. The function carries a custom
    vector-Jacobian product. The backward pass contracts the cotangent on the
    coefficients with the coefficients of the tangent rows (`_Free.tangents`).
    The breaks carry no gradient, so the derivative holds the projected radii
    of the table fixed. The gradient is zero at a theta with an invalid
    model."""
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
    r"""Raise InputError, whose message refers to the component as `name`,
    when the anisotropy of the model's component cannot be a free parameter.

    The anisotropy cannot be a free parameter in three cases: the model is not
    spatially finite; the component's profile is not a `TruncatedProfile`
    that carries the DF of a parent model; or the component's anisotropy
    family varies in neither :math:`\beta_{\rm c}` nor :math:`R_{\rm a}`
    (`TANGENT_PARAMETERS`)."""
    component = model.components[0]
    if not model.spatially_finite:
        raise InputError(
            f"{name} is not spatially finite, so the anisotropy of {name} cannot "
            f"be a free parameter; the edge terms of the derivative depend on the "
            f"DF of {name}."
        )
    if not isinstance(component.profile, TruncatedProfile) or (
        np.size(component.profile.df_k) == 0
    ):
        raise InputError(
            f"{name} is not a DF on a truncated profile that carries the parent "
            "model's DF (a model made by truncate on such a profile), so the "
            f"anisotropy of {name} cannot be a free parameter. The profile of a "
            "Method I truncated model follows the model's own DF and moves with "
            "beta_c and R_a, so neither the Jeans path (a fixed profile) nor the DF "
            "path at a fixed potential gives the full derivative. A free "
            "anisotropy is a parameter of a Method II model, a DF on another "
            "model's truncated profile, with the truncated profile held fixed."
        )
    family = component.anisotropy.family
    if not any(n in TANGENT_PARAMETERS[family] for n in THETA_NAMES):
        raise InputError(
            f"The anisotropy family {family} of {name} varies in neither beta_c nor "
            f"R_a, so the anisotropy of {name} has no free parameter."
        )


class Differentiable:
    r"""The observables of a one-component model as JAX functions of the
    component's anisotropy.

    The anisotropy is the pair :math:`\theta = (\beta_{\rm c}, R_{\rm a})` of
    the central anisotropy :math:`\beta_{\rm c}` and the anisotropy radius
    :math:`R_{\rm a}`. Each method of `Differentiable` takes :math:`\theta` as
    the first argument. Each method returns the value of the public function
    of the same name as the method (`pianistpy.sigma_los` for
    `Differentiable.sigma_los`) on the model rebuilt at :math:`\theta`. Each method is a
    JAX function of :math:`\theta` and works under ``jax.jit`` and
    ``jax.grad``. To differentiate the observables of a model with several
    components, call `likelihood`.

    Parameters
    ----------
    model : Model
        Model of one component. The model must meet three conditions: the
        model is spatially finite (``model.spatially_finite`` is True); the
        component's profile is the truncated profile of a parent model and
        carries the parent's distribution function (DF); and the component's
        anisotropy family is isotropic, constant, Osipkov-Merritt (OM) Type
        I, OM Type II or Cuddeford-Osipkov-Merritt (COM) Type I.
    settings : Settings, optional
        Numerical settings of the observables. Default: ``None``, which
        means the settings of `model`.

    Attributes
    ----------
    model : Model
        The model passed to the constructor.
    theta : array, shape (2,)
        The anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the model's
        component. :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}`
        is in internal units in every unit system of the model (internal,
        model and physical). An isotropic or constant component has an
        infinite :math:`R_{\rm a}`.

    Raises
    ------
    InputError
        If `model` is not a `Model`, or if `model` has more than one
        component. The components of a model with several components are
        differentiated through `likelihood`.
    InputError
        If the model is not spatially finite, or if the component's profile
        is not a truncated profile that carries the DF of a parent model.
    InputError
        If the component's anisotropy family is COM Type II. The observables
        of a COM Type II component are not differentiable in the anisotropy.
    InputError
        If a setting of the ``observables`` group is outside the setting's
        range: ``n_projection_nodes`` below 20.
    AccuracyError
        If the validation report of the model has tier ``"refused"`` (an
        error above :math:`10^{-2}`) and the settings of the model do not set
        ``write_anyway=True``.
    AccuracyError
        If the squared line-of-sight dispersion of the model is negative at
        a tabulated projected radius.
    ResultError
        If an intermediate result fails a check that the next step of the
        computation needs, for example non-finite values from a degenerate
        model.

    See Also
    --------
    likelihood : The likelihood of data given a model, differentiable in
        the anisotropy of each component.
    truncate : The truncated model whose profile a differentiable component
        uses.
    sigma_los : The line-of-sight velocity dispersion of a model.
    losvd : The line-of-sight velocity distribution of a model.
    pdf_v : The local velocity distribution of a model.

    Notes
    -----
    Every radius, velocity, aperture length and :math:`R_{\rm a}` that a
    method reads is in internal units in every unit system of the model.
    Every value that a method returns is in internal units.

    The first call of a method at a new :math:`\theta` inverts the DF of the
    component at :math:`\theta` on the component's truncated profile. The
    first call then builds the tables of the observable at :math:`\theta`. A
    `Differentiable` keeps the tables of at most eight values of
    :math:`\theta`: the model's own :math:`\theta` and the seven other values
    of :math:`\theta` first met most recently.

    At a :math:`\theta` where rebuilding the model raises `InputError`,
    `UnphysicalModelError`, `AccuracyError` or `ResultError`, a method
    returns NaN and a zero gradient. Any other package error raised during
    the computation of a method reaches the caller as
    ``jax.errors.JaxRuntimeError``. The message of the
    ``jax.errors.JaxRuntimeError`` carries the message of the package error.
    Three checks run before the computation and raise the package error
    directly: the check of a concrete radius against the tabulated radii in
    `sigma_r`, `sigma_t` and `sigma_los`; the check that the component's DF
    is differentiable in `losvd`, `pdf_v` and `aperture_losvd`; and the
    check that the aperture passed to `aperture_losvd` is an `Aperture`.

    The derivative of a method holds fixed the radii and velocities at which
    the tables at :math:`\theta` are tabulated. The value of a method follows
    the change of the tabulated radii and velocities from one :math:`\theta`
    to another. The derivative of a method omits the change of the tabulated
    radii and velocities. A method is not differentiable in the radii or
    velocities that the method reads. The gradient in a radius or a velocity
    is NaN.

    The parameters that carry a derivative depend on the method:

    - `sigma_r`, `sigma_t`, `sigma_los` and `sigma_los_aperture` carry the
      derivative in :math:`\beta_{\rm c}` for every accepted family, and in
      :math:`R_{\rm a}` for the OM and COM families.
    - `losvd`, `pdf_v` and `aperture_losvd` carry the derivative in
      :math:`\beta_{\rm c}` for the constant and COM Type I families, and in
      :math:`R_{\rm a}` for the OM and COM Type I families. The gradient of
      `losvd`, `pdf_v` or `aperture_losvd` in :math:`\beta_{\rm c}` is NaN
      for an OM component.
    - For an isotropic or constant component, every method holds
      :math:`R_{\rm a}` at infinity, and the gradient in :math:`R_{\rm a}` is
      zero.

    References
    ----------
    Notes Section T:7.2. Notes Section T:7.3.

    Examples
    --------
    >>> import jax
    >>> import pianistpy as pp
    >>> parent = pp.truncate(
    ...     pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy())),
    ...     pp.Truncation(k_t=0.05),
    ... )
    >>> member = pp.truncate(
    ...     pp.compute_df(
    ...         pp.Component(
    ...             profile=parent.profile, anisotropy=pp.Anisotropy(beta_c=0.25)
    ...         )
    ...     )
    ... )
    >>> d = pp.Differentiable(member)
    >>> grad = jax.grad(lambda theta: d.sigma_los(theta, [0.1, 0.3]).sum())(d.theta)
    >>> grad.shape
    (2,)
    """

    def __init__(self, model: Model, settings: Settings | None = None) -> None:
        if not isinstance(model, Model) or len(model.components) != 1:
            raise InputError(
                "Differentiable takes a Model of one component; the components of "
                "a superposition are differentiated through the likelihood."
            )
        refuse_unless_free(model, "The model's component")
        refuse_unless_asked(
            f"The model {model.components[0].name!r}", model.validation, model.settings
        )
        family = model.components[0].anisotropy.family
        if family in REFUSED_FAMILIES:
            raise InputError(
                f"The observables of a {family} component are not differentiable, "
                f"because {REFUSED_FAMILIES[family]}."
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
        r"""Return the radial velocity dispersion at user-specified 3D radii
        for the model at the anisotropy theta.

        The value is `pianistpy.sigma_r` of the model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        r : array-like
            User-specified 3D radii :math:`r`. The radii are in internal units
            in every unit system of the model.

        Returns
        -------
        sigma_r : array
            Radial velocity dispersion :math:`\sigma_{\rm r}` in internal
            units, with the shape of `r`. Every value is NaN when rebuilding
            the model at `theta` raises InputError, UnphysicalModelError,
            AccuracyError or ResultError.

        Raises
        ------
        InputError
            If a radius lies inside the innermost or beyond the outermost
            radius that the model tabulates. A radius inside a function
            compiled by ``jax.jit`` is not checked before the computation.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error.

        See Also
        --------
        `pianistpy.sigma_r` : The radial velocity dispersion of a model.
        `Differentiable.sigma_t` : The tangential velocity dispersion as a JAX
            function of the anisotropy.
        """
        self._refuse_outside(r, projected=False)
        return self._sigma_r(_theta(theta), jnp.asarray(r, dtype=jnp.float64))

    def sigma_t(self, theta: ArrayLike, r: ArrayLike) -> Array:
        r"""Return the tangential velocity dispersion at user-specified 3D
        radii for the model at the anisotropy theta.

        The tangential velocity dispersion :math:`\sigma_{\rm t}` combines
        both tangential velocity components. The value is `pianistpy.sigma_t`
        of the model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        r : array-like
            User-specified 3D radii :math:`r`. The radii are in internal units
            in every unit system of the model.

        Returns
        -------
        sigma_t : array
            Tangential velocity dispersion :math:`\sigma_{\rm t}` in internal
            units, with the shape of `r`. Every value is NaN when rebuilding
            the model at `theta` raises InputError, UnphysicalModelError,
            AccuracyError or ResultError.

        Raises
        ------
        InputError
            If a radius lies inside the innermost or beyond the outermost
            radius that the model tabulates. A radius inside a function
            compiled by ``jax.jit`` is not checked before the computation.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error.

        See Also
        --------
        `pianistpy.sigma_t` : The tangential velocity dispersion of a model.
        `Differentiable.sigma_r` : The radial velocity dispersion as a JAX
            function of the anisotropy.
        """
        self._refuse_outside(r, projected=False)
        return self._sigma_t(_theta(theta), jnp.asarray(r, dtype=jnp.float64))

    def sigma_los(self, theta: ArrayLike, r_2d: ArrayLike) -> Array:
        r"""Return the line-of-sight velocity dispersion at user-specified
        projected radii for the model at the anisotropy theta.

        The value is `pianistpy.sigma_los` of the model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        r_2d : array-like
            User-specified projected radii :math:`R_{\rm 2D}`. The projected
            radii are in internal units in every unit system of the model.

        Returns
        -------
        sigma_los : array
            Line-of-sight velocity dispersion :math:`\sigma_{\rm los}` in
            internal units, with the shape of `r_2d`. Every value is NaN when
            rebuilding the model at `theta` raises InputError,
            UnphysicalModelError, AccuracyError or ResultError.

        Raises
        ------
        InputError
            If a projected radius lies inside the innermost or beyond the
            outermost projected radius that the model tabulates. A projected
            radius inside a function compiled by ``jax.jit`` is not checked
            before the computation.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error.

        See Also
        --------
        `pianistpy.sigma_los` : The line-of-sight velocity dispersion of a
            model.
        Differentiable.sigma_los_aperture : The line-of-sight velocity
            dispersion over an aperture as a JAX function of the anisotropy.
        """
        self._refuse_outside(r_2d, projected=True)
        return self._sigma_los(_theta(theta), jnp.asarray(r_2d, dtype=jnp.float64))

    def _refuse_outside(self, radii: ArrayLike, projected: bool) -> None:
        """Raise InputError for a radius outside the radii tabulated at the
        model's own theta.

        The check reads the dispersion profile (3D radii) or the projected
        table (projected radii) at the model's own theta. The first and last
        tabulated radii come from the component's profile and do not change
        with theta. A radius traced by JAX is not checked."""
        if isinstance(radii, jax.core.Tracer):
            return
        entry = self._free.tables(self._free.theta)
        if entry is None:
            return
        r = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        if projected:
            entry[1].read(r)
        else:
            entry[0].read(r)

    def sigma_los_aperture(self, theta: ArrayLike, aperture: Aperture) -> Array:
        r"""Return the line-of-sight velocity dispersion over an aperture for
        the model at the anisotropy theta.

        The aperture dispersion :math:`\sigma_{\rm ape}` is the root mean
        square of :math:`\sigma_{\rm los}` over the aperture, weighted by the
        surface density. The value is `pianistpy.sigma_los_aperture` of the
        model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        aperture : Aperture
            User-specified aperture. The lengths of the aperture are in
            internal units in every unit system of the model.

        Returns
        -------
        sigma_los : scalar array
            Aperture dispersion :math:`\sigma_{\rm ape}` in internal units.
            The value is NaN when rebuilding the model at `theta` raises
            InputError, UnphysicalModelError, AccuracyError or ResultError.

        Raises
        ------
        jax.errors.JaxRuntimeError
            If `pianistpy.sigma_los_aperture` raises `InputError` for the
            aperture on the model at `theta`, because the tables of the model
            do not cover the aperture. The message of the
            ``jax.errors.JaxRuntimeError`` carries the message of the
            `InputError`.

        See Also
        --------
        pianistpy.sigma_los_aperture : The line-of-sight velocity dispersion
            of a model over an aperture.
        Aperture : The apertures of the observables.
        Differentiable.aperture_losvd : The line-of-sight velocity
            distribution in an aperture as a JAX function of the anisotropy.

        Notes
        -----
        The derivative of :math:`\sigma_{\rm ape}` in a parameter
        :math:`\theta_i` (:math:`\beta_{\rm c}` or :math:`R_{\rm a}`) is

        .. math::

            \frac{\partial \sigma_{\rm ape}}{\partial \theta_i}
            = \frac{1}{2\sigma_{\rm ape}}
            \frac{\int_A \Sigma\, \partial \sigma_{\rm los}^2/\partial
            \theta_i \, {\rm d}A}{\int_A \Sigma \, {\rm d}A},

        with :math:`A` the aperture and :math:`\Sigma` the surface density of
        the component. The surface density does not depend on
        :math:`\theta`, since the component's profile is fixed. Both
        integrals use the quadrature nodes and weights of
        `pianistpy.sigma_los_aperture`.
        """
        value, slope = _aperture_host(self._free, aperture)
        return _radial_function(self._free, (value, slope))(
            _theta(theta), jnp.zeros((), dtype=jnp.float64)
        )

    def losvd(self, theta: ArrayLike, r_2d: ArrayLike, v: ArrayLike) -> Array:
        r"""Return the line-of-sight velocity distribution at the pairs
        ``(r_2d, v)`` for the model at the anisotropy theta.

        The value is `pianistpy.losvd` of the model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        r_2d : array-like
            User-specified projected radii :math:`R_{\rm 2D}`. The projected
            radii are in internal units in every unit system of the model.
        v : array-like
            Line-of-sight velocities :math:`V`, in internal units in every
            unit system of the model. The arrays `r_2d` and `v` broadcast
            against each other.

        Returns
        -------
        losvd : array
            Line-of-sight velocity distribution per unit internal velocity,
            with the broadcast shape of `r_2d` and `v`. Every value is NaN
            when rebuilding the model at `theta` raises InputError,
            UnphysicalModelError, AccuracyError or ResultError.

        Raises
        ------
        InputError
            If the component's DF is not differentiable in the anisotropy.
            The DF of an isotropic component is not differentiable in the
            anisotropy. The DF of a component whose DF center lies on a
            critical branch is not differentiable in the anisotropy. The
            message states the reason.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error. A projected radius outside the
            table, for which `pianistpy.losvd` raises a package error, gives
            a ``jax.errors.JaxRuntimeError``.

        See Also
        --------
        pianistpy.losvd : The line-of-sight velocity distribution of a model.
        Differentiable.aperture_losvd : The line-of-sight velocity
            distribution in an aperture as a JAX function of the anisotropy.

        Notes
        -----
        The derivative in a parameter :math:`\theta_i` (:math:`\beta_{\rm c}`
        or :math:`R_{\rm a}`) is the sum of two parts. The first part is the
        derivative in :math:`\theta_i` at a fixed DF table. The second part
        is the derivative through the values of the DF table, with the
        energies of the DF table held fixed. The method is not
        differentiable in `r_2d` or `v`.

        References
        ----------
        Notes Section T:7.1.
        """
        self._df.served()
        x, u = jnp.broadcast_arrays(
            jnp.asarray(r_2d, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )
        return self._losvd(jnp.asarray(theta, dtype=jnp.float64), x, u)

    def pdf_v(self, theta: ArrayLike, r: ArrayLike, v: ArrayLike) -> Array:
        r"""Return the local velocity distribution at the pairs ``(r, v)`` for the
        model at the anisotropy theta.

        The value is `pianistpy.pdf_v` of the model rebuilt at `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        r : array-like
            User-specified 3D radii :math:`r`. The radii are in internal units
            in every unit system of the model.
        v : array-like
            Velocities :math:`v`, in internal units in every unit system of the
            model. The arrays `r` and `v` broadcast against each other.

        Returns
        -------
        pdf_v : array
            Local velocity distribution per unit internal velocity, with the
            broadcast shape of `r` and `v`. Every value is NaN when rebuilding
            the model at `theta` raises InputError, UnphysicalModelError,
            AccuracyError or ResultError.

        Raises
        ------
        InputError
            If the component's DF is not differentiable in the anisotropy.
            The DF of an isotropic component is not differentiable in the
            anisotropy. The DF of a component whose DF center lies on a
            critical branch is not differentiable in the anisotropy. The
            message states the reason.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error. A radius outside the radii of the
            model, for which `pianistpy.pdf_v` raises a package error, gives a
            ``jax.errors.JaxRuntimeError``.

        See Also
        --------
        pianistpy.pdf_v : The local velocity distribution of a model.
        Differentiable.losvd : The line-of-sight velocity distribution as a
            JAX function of the anisotropy.

        Notes
        -----
        The derivative in a parameter :math:`\theta_i` (:math:`\beta_{\rm c}`
        or :math:`R_{\rm a}`) is the sum of two parts. The first part is the
        derivative in :math:`\theta_i` at a fixed DF table and at fixed
        velocities. The second part is the derivative through the values of the
        DF table, with the energies of the DF table held fixed. The method is
        not differentiable in `r` or `v`.

        References
        ----------
        Notes Section T:7.1.
        """
        self._df.served()
        x, u = jnp.broadcast_arrays(
            jnp.asarray(r, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )
        return self._pdf_v(jnp.asarray(theta, dtype=jnp.float64), x, u)

    def aperture_losvd(
        self, theta: ArrayLike, aperture: Aperture, v: ArrayLike
    ) -> Array:
        r"""Return the line-of-sight velocity distribution in an aperture at
        user-specified line-of-sight velocities for the model at the
        anisotropy theta.

        The value is `pianistpy.aperture_losvd` of the model rebuilt at
        `theta`.

        Parameters
        ----------
        theta : array-like, shape (2,)
            Anisotropy :math:`(\beta_{\rm c}, R_{\rm a})` of the component.
            :math:`\beta_{\rm c}` is dimensionless. :math:`R_{\rm a}` is in
            internal units in every unit system of the model.
        aperture : Aperture
            User-specified aperture. The lengths of the aperture are in
            internal units in every unit system of the model.
        v : array-like
            Line-of-sight velocities :math:`V`, in internal units in every
            unit system of the model.

        Returns
        -------
        losvd : array
            Line-of-sight velocity distribution in the aperture per unit
            internal velocity, with the shape of `v`. Every value is NaN when
            rebuilding the model at `theta` raises InputError,
            UnphysicalModelError, AccuracyError or ResultError.

        Raises
        ------
        InputError
            If the component's DF is not differentiable in the anisotropy.
            The DF of an isotropic component is not differentiable in the
            anisotropy. The DF of a component whose DF center lies on a
            critical branch is not differentiable in the anisotropy.
        InputError
            If `aperture` is not an `Aperture`.
        jax.errors.JaxRuntimeError
            If the computation raises a package error other than the four
            package errors that give NaN, for example
            `pianistpy.exceptions.WorkerError` when a parallel process ends.
            The message of the ``jax.errors.JaxRuntimeError`` carries the
            message of the package error. An aperture that the table does not
            cover, for which `pianistpy.aperture_losvd` raises a package
            error, gives a ``jax.errors.JaxRuntimeError``.

        See Also
        --------
        pianistpy.aperture_losvd : The line-of-sight velocity distribution of
            a model in an aperture.
        Aperture : The apertures of the observables.
        Differentiable.losvd : The line-of-sight velocity distribution as a
            JAX function of the anisotropy.

        Notes
        -----
        The derivative in a parameter :math:`\theta_i` (:math:`\beta_{\rm c}`
        or :math:`R_{\rm a}`) is the sum of two parts. The first part is the
        derivative in :math:`\theta_i` at a fixed DF table. The second part
        is the derivative through the values of the DF table, with the
        energies of the DF table held fixed. The method is not
        differentiable in `v`.

        References
        ----------
        Notes Section T:7.1.
        """
        self._df.served()
        if not isinstance(aperture, Aperture):
            raise InputError(
                "The aperture of aperture_losvd must be an Aperture; the value "
                "given is not an Aperture."
            )
        function = _aperture_function(self._df, aperture)
        return function(
            jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64)
        )


def _theta(theta: ArrayLike) -> Array:
    return jnp.asarray(theta, dtype=jnp.float64)


def _radial_function(free: _Free, host: tuple[Any, Any]) -> Any:
    """Return an observable of a free component as a JAX function of
    ``(theta, x)`` with a custom vector-Jacobian product.

    The pair `host` holds two host functions of ``(theta, x)``: the value of
    the observable, and the derivative of the observable in theta with shape
    ``(2,) + x.shape``. The gradient in x is NaN. The argument `free` is not
    read."""
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
    r"""Yield ``(i, name)`` for each parameter of `THETA_NAMES` whose
    derivative is computed.

    A parameter is skipped in two cases. A held :math:`R_{\rm a}` is skipped,
    and ``out[i]`` keeps the value already in `out`. A parameter not in
    `tangents` is skipped after ``out[i]`` is set to NaN. The argument `entry`
    is not read."""
    for i, name in enumerate(THETA_NAMES):
        if name == "r_a" and free.held:
            continue
        if name not in tangents:
            out[i] = np.nan
            continue
        yield i, name


def _sigma_r_host(free: _Free, which: int) -> tuple[Any, Any]:
    r"""Return the host pair (value, derivative) of :math:`\sigma_{\rm r}`
    (`which` 0) or :math:`\sigma_{\rm t}` (`which` 1).

    The value function reads the dispersion profile at theta, as the plain
    call reads the dispersion profile. The derivative function divides the
    derivative of the squared dispersion by twice the dispersion. The
    derivative of :math:`\sigma_{\rm r}^2` is the tangent reading of the
    dispersion profile. The derivative of
    :math:`\sigma_{\rm t}^2 = 2 (1 - \beta_\sigma) \sigma_{\rm r}^2` adds the
    derivative of the anisotropy profile :math:`\beta_\sigma`
    (`beta_sigma_tangent`). The derivative is zero where the dispersion is
    not positive."""

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
    r"""Return the host pair (value, derivative) of :math:`\sigma_{\rm los}`.

    The value function reads the projected table at theta, as the plain call
    reads the projected table. The derivative function reads the spline of
    the tangent rows of :math:`\sigma_{\rm los}^2` at the requested projected
    radii and divides the result by twice the value. The tangent rows sit at
    the tabulated projected radii of the projected table at theta. The
    derivative is zero at a projected radius that is not positive, at a
    projected radius beyond the last tabulated projected radius, and where
    the value is not positive. `Differentiable.sigma_los` raises InputError
    for a concrete projected radius beyond the last tabulated projected radius
    before the computation."""

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
    r"""Return the host pair (value, derivative) of the aperture dispersion
    :math:`\sigma_{\rm ape}` in `aperture`.

    The value function computes :math:`\sigma_{\rm ape}` as the plain call
    computes :math:`\sigma_{\rm ape}` (`aperture_dispersion` on the projected
    table at theta). The derivative function integrates the tangent rows of
    :math:`\sigma_{\rm los}^2` over the aperture with the quadrature nodes and
    weights of the plain call, and divides the integral by
    :math:`2\sigma_{\rm ape}`. The quadrature of the aperture is linear in the
    integrand, so the quadrature of the tangent rows is the derivative of the
    quadrature of :math:`\sigma_{\rm los}^2`. The derivative is zero where
    :math:`\sigma_{\rm ape}` is not positive."""
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
            # A tangent row can be negative. The reader's raw method reads a
            # negative row, and the reader's call raises AccuracyError on one.
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


# The numbers of the DF's central form that the derivative of the DF table
# carries, in the order of the entry "center" of df.gradient.table_jacobian.
_CENTER_LEAVES = (
    "central_exponent",
    "central_amplitude",
    "central_next_exponent",
    "central_next_coefficient",
)

# The reason, carried by an InputError of `_DfPath.served`, why the DF path
# takes no derivative of an isotropic component (DF_PARAMETERS gives the
# isotropic family no parameter).
ISOTROPIC_DF = (
    "The DF path takes no derivative in beta_c of an isotropic member. At "
    "beta_c = 0, the form of the inversion changes, and the derivatives of the "
    "child model's DF table from the two sides of beta_c = 0 do not agree to "
    "0.000001 (the line-of-sight velocity distributions of an isotropic member "
    "are computed from G alone, with no L factor). sigma_los and "
    "sigma_los_aperture take the derivative in beta_c of an isotropic member by "
    "the Jeans path."
)


def _df_refusal(component: Any) -> str | None:
    """Return the message of the InputError that the derivative of the DF
    table (`df.gradient._child_checked`) raises for a component, or None
    when `df.gradient._child_checked` accepts the component.

    `df.gradient._child_checked` raises InputError for a COM Type II
    component, for a family with no anisotropy parameter to vary, and for a
    component whose DF center lies on a critical branch."""
    try:
        _child_checked(component)
    except InputError as error:
        return str(error)
    return None


class _DfPath:
    r"""The DF path of a free component, evaluated on the host.

    For each theta, a `_DfPath` holds the model rebuilt at theta, converted to
    internal units and given the settings of the `_Free`. At the component's
    own theta, the model is the component's own model, also converted to
    internal units and given the settings of the `_Free`. The `LosTable`, the
    `RowsBasis`, the derivative of the DF table and the velocity-distribution
    profiles of the model at theta are built at the first request. A
    `_DfPath` computes the gradient in theta of a cotangent on the DF table
    and the DF's forms (`df.gradient.table_jacobian`, with the energies of the
    DF table at theta held fixed), plus the explicit part. A `_DfPath` keeps
    the entries of the component's own theta and of the `THETA_KEPT` - 1
    other theta first met most recently.

    The parameters of the DF path are the parameters that `DF_PARAMETERS`
    gives the component's family. The gradient
    in :math:`\beta_{\rm c}` of an OM component is NaN. The gradient in a
    held :math:`R_{\rm a}` is zero. The constructor stores, in `refusal`,
    the reason why the DF path takes no derivative of the component: the
    message of `_df_refusal`, or `ISOTROPIC_DF` for an isotropic component.
    `served` raises InputError with the stored reason."""

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
        # The exponent steps of the DF table's derivative
        # (df.gradient.table_jacobian). `gradient` measures the steps at the
        # component's own theta before the first derivative at another theta,
        # and the derivative at every theta reuses the steps.
        self._steps: dict[tuple, dict[str, Any]] = {}

    def served(self) -> None:
        """Raise InputError with the stored reason when the DF path takes no
        derivative of the component."""
        if self.refusal is not None:
            raise InputError(
                f"The DF path of the model's component is not differentiable. "
                f"{self.refusal}"
            )

    def entry(self, theta: np.ndarray) -> dict[str, Any] | None:
        """Return the entry at theta: a dict that holds the model at theta,
        in internal units, and the objects built from the model at theta.

        The return value is None when rebuilding the model at theta raises
        InputError, UnphysicalModelError, AccuracyError or ResultError."""
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
                        "The anisotropy (beta_c, R_a) = (%r, %r) of %r gives an "
                        "invalid model, because rebuilding the model with the "
                        "anisotropy raised the error %s",
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
        """Return the LosTable of the entry's model, built at the first
        request."""
        if "los_table" not in entry:
            from .core import LosTable

            entry["los_table"] = LosTable(entry["model"])
        return entry["los_table"]

    def rows_basis(self, entry: dict[str, Any], part: Any) -> RowsBasis:
        """Return the `RowsBasis` of the model of a LosTable part, built at
        the first request and kept in the entry.

        The `RowsBasis` holds the route with the DF and the prepared lookup
        that the transpose of the rows and the explicit part read. Keeping the
        `RowsBasis` in the entry builds the route with the DF once per
        theta."""
        if "rows_basis" not in entry:
            entry["rows_basis"] = RowsBasis(part.model, part.model.settings)
        return entry["rows_basis"]

    def _jacobian(self, entry: dict[str, Any]) -> tuple[Any, Any]:
        """Return the derivative of the DF table at the entry's theta, built at
        the first request with the stored exponent steps (`_steps`).

        Raises ResultError when the derivative's energies differ from the
        energies of the model's DF table."""
        if "jac" not in entry:
            model = entry["model"]
            values, jac, _ = table_jacobian(
                model.components[0], model.settings, steps=self._steps
            )
            if values["k"].tobytes() != np.asarray(model.table.k[0]).tobytes():
                raise ResultError(
                    "The energies of the DF table's derivative differ from the "
                    "energies of the model's own DF table, so the model's DF table "
                    "was not made by the rule that the derivative holds."
                )
            entry["jac"] = (values, jac)
        return entry["jac"]

    def gradient(
        self, entry: dict[str, Any], cotangent: Any, explicit: dict[str, float]
    ) -> np.ndarray:
        """Return the gradient in theta, shape (2,), of a cotangent on the
        entry's DF table and forms, plus the explicit part `explicit` of each
        parameter.

        A `cotangent` of None stands for a zero cotangent. The gradient is
        NaN in a parameter that the DF path does not take. The gradient is
        zero in a held anisotropy radius."""
        if cotangent is None or _is_zero(cotangent):
            # A zero cotangent adds nothing through the DF table, so the DF
            # table's derivative is not built. Each parameter takes 0.0 plus
            # the explicit part. The general branch below also gives +0.0 for
            # the DF table's part of a zero cotangent, because the mask on
            # log_df drops every zero entry of the cotangent.
            out = np.zeros(2)
            for i, name in _each_parameter(self.free, entry, self.names, out):
                out[i] = 0.0 + float(explicit.get(name, 0.0))
            return out
        if "jac" not in entry and not self._steps:
            # The exponent steps are measured at the component's own theta
            # before the derivative at any other theta.
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


def _is_zero(cotangent: Any) -> bool:
    """Return True when every number of a cotangent on the DF table and
    forms is zero. A NaN counts as nonzero."""
    return not any(
        np.any(np.asarray(x, dtype=np.float64))
        for x in (
            cotangent.log_df,
            cotangent.log_df_slope,
            cotangent.log_df_below,
            cotangent.log_df_slope_below,
            *cotangent.form.values(),
        )
    )


def _carried(cotangent: Any, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the parts of a cotangent that the derivative of the DF table
    carries: the coefficients of the floor's edge law and the numbers of the
    DF's central form (`_CENTER_LEAVES`).

    Raises ResultError when the cotangent is nonzero on a part of the DF
    table or forms that the derivative of the DF table does not carry. The
    message of the ResultError lists each such part. The derivative in theta
    of such a part is not computed, so the gradient would be incomplete."""
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
            "The observable depends on the parts "
            f"{', '.join(sorted(set(loose)))} of the DF, whose derivative in theta "
            "the DF table's derivative does not carry, so the gradient would be "
            "incomplete."
        )
    return {
        "below_floor_coefficients": law[:n_law],
        "center": center[:n_center],
    }


def _df_function(value: Any, backward: Any) -> Any:
    """Return a JAX function of ``(theta, *arrays)`` evaluated on the host by
    `value`, with a custom vector-Jacobian product.

    The output has the shape of the last array. The backward pass calls the
    host function `backward` with theta, the arrays and the cotangent on the
    output, and `backward` returns the gradient in theta, shape (2,). The
    gradient in each array is NaN."""

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
    """Return the gradient in theta of cotangents on the rows of a LosTable
    part.

    The transpose of the rows (`losvd_rows_vjp`) carries the cotangents onto
    the DF table and forms. The explicit part (`losvd_rows_explicit`) is the
    derivative of the rows in each parameter at a fixed DF table.
    `_DfPath.gradient` adds the two parts."""
    settings = part.model.settings
    basis = path.rows_basis(entry, part)
    cotangent = losvd_rows_vjp(part.model, part.table, settings, rows, basis)
    explicit = (
        losvd_rows_explicit(part.model, part.table, settings, rows, path.names, basis)
        if path.names
        else {}
    )
    return path.gradient(entry, cotangent, explicit)


def _losvd_function(path: _DfPath) -> Any:
    """Return `losvd` of a free component as a JAX function of
    ``(theta, r_2d, v)``."""
    from .core import _reader_of, losvd

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
        c = np.asarray(cotangent, dtype=np.float64).ravel()
        if not np.any(c):
            # A zero cotangent adds nothing to the gradient, so no reading
            # and no row is transposed.
            return path.gradient(entry, None, {})
        part = path.los_table(entry).parts[0]
        rows = losvd_read_vjp(
            _reader_of(part),
            np.asarray(x, dtype=np.float64).ravel(),
            np.asarray(u, dtype=np.float64).ravel(),
            c,
        )
        return _rows_gradient(path, entry, part, rows)

    return _df_function(value, backward)


def _aperture_function(path: _DfPath, aperture: Aperture) -> Any:
    """Return `aperture_losvd` of a free component in one aperture as a JAX
    function of ``(theta, v)``."""
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
        c = np.asarray(cotangent, dtype=np.float64).ravel()
        if not np.any(c):
            # A zero cotangent adds nothing to the gradient, so the aperture's
            # distribution is not computed again, and no reading and no row
            # is transposed.
            return path.gradient(entry, None, {})
        table = path.los_table(entry)
        part = table.parts[0]
        settings = part.model.settings
        inside = _internal_aperture(table.model, aperture)
        # The aperture's distribution is computed once, and `record` collects
        # the table readings behind each value for the transpose.
        record: list[Any] = []
        result = stage_aperture_losvd(
            part.model, part.table, part.projected, inside, settings, record=record
        )
        rule = panels(result.v_nodes, result.v_weights, result.losvd[0])
        at_nodes = read_panels_vjp(
            rule,
            np.abs(np.asarray(v, dtype=np.float64)).ravel(),
            c,
        )
        rows = aperture_losvd_vjp(
            part.model,
            part.table,
            part.projected,
            inside,
            settings,
            result,
            at_nodes,
            record=record,
        )
        return _rows_gradient(path, entry, part, rows)

    return _df_function(value, backward)


def _pdf_v_function(path: _DfPath) -> Any:
    """Return `pdf_v` of a free component as a JAX function of
    ``(theta, r, v)``."""
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
            velocities, c = us[at], cs[at]
            # A radius whose cotangents are all zero adds nothing to the
            # gradient, so the velocity distribution at the radius is skipped.
            if not np.any(c):
                continue
            result = velocity_distribution(view, float(radius), view.settings, profile)
            inside = (velocities >= 0.0) & (velocities <= result.v_max) & (c != 0.0)
            if not np.any(inside):
                continue
            # The distribution is read at the requested velocities by the rule
            # that the distribution's record holds (`VelocityDistribution.read`),
            # and the transpose is taken at the requested velocities.
            asked = replace(
                result, v_nodes=velocities[inside], pdf=result.read(velocities[inside])
            )
            cotangent_df, part = velocity_distribution_vjp(asked, c[inside], path.names)
            total = (
                cotangent_df if total is None else add_cotangents(total, cotangent_df)
            )
            for name in path.names:
                explicit[name] += float(part[name])
        if total is None:
            # No velocity carries a nonzero cotangent. The gradient is zero in
            # each parameter that the DF path takes and NaN in each other
            # parameter, as in `_DfPath.gradient`.
            out = np.zeros(2)
            for i, _ in _each_parameter(path.free, entry, path.names, out):
                out[i] = 0.0
            return out
        return path.gradient(entry, total, explicit)

    return _df_function(value, backward)
