r"""The log-likelihood of a model against measured data.

A user stores each set of measurements in a `Data` and calls `likelihood` on
a model and the data. `likelihood` computes the projected tables of the
model once and returns a `Likelihood`, a JAX function of the component
weights, of the mass and length scales and, for chosen components, of the
anisotropy parameters. `log_likelihood` returns the value of the
log-likelihood at the model's own weights and scales.

For measured values :math:`d`, model values :math:`m` and the covariance
matrix :math:`C` of the errors (diagonal, with :math:`C_{ii} = e_i^2`, for
independent errors :math:`e_i`), the log-likelihood of one `Data` is

.. math::

    \ln \mathcal{L} = -\frac{1}{2} (d - m)^{\rm T} C^{-1} (d - m)
    - \frac{1}{2} \ln \det (2 \pi C),

and the log-likelihoods of several `Data` are summed. A projected radius in
the units of the data is :math:`R = L\,l\,x`, with :math:`x` the projected
radius in internal units, :math:`L` the length factor of the unit system of
the data and :math:`l` the length scale divided by the model's own length
scale. A velocity in the units of the data is
:math:`v = V (\mu / l)^{1/2} u`, with :math:`u` the velocity in internal units,
:math:`V` the velocity factor of the unit system of the data and :math:`\mu`
the mass scale divided by the model's own mass scale. In the physical
system the mass scale is :math:`M_0` in solar masses and the length scale
is :math:`r_0` in parsecs, as `Model.to` takes them. Notes Section T:7.4
derives the likelihood.
"""

__all__ = ["OBSERVABLES", "Data", "Likelihood", "likelihood", "log_likelihood"]

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .._version import __version__
from ..df import Model
from ..exceptions import (
    InputError,
    ResultError,
)
from ..identity import identity_string
from ..io import StoredResult, load_result, register_reader, write_result
from ..numerics.pytrees import register_result
from ..observables._differentiable import (
    BUCKET_FACTOR,
    _Free,
    _free_reading,
    refuse_unless_free,
)
from ..observables._projection import dispersion_table_reader
from ..observables.aperture import Aperture
from ..observables.stages import (
    END_TOLERANCE,
    INNER_DISK_TOLERANCE,
    TINY,
    DispersionProfile,
    GaussHermiteTable,
    ProjectedDispersion,
    aperture_losvd,
    aperture_nodes,
    dispersion_profile,
    gauss_hermite_table,
    kept_ranges,
    losvd_table,
    projected_dispersion,
    spline_value,
    surface_density_reader,
)
from ..particles.core import component_view, is_superposition
from ..settings import Settings
from ..superposition import mixture
from ..typing import Array, ArrayLike
from ..units.system import SYSTEMS
from ..validation import refuse_unless_asked
from . import _binned
from ._binned import BINNED

RESULT_KIND = "data"
# The observables h3 and h4 are the Gauss-Hermite moments h_3 and h_4 of the
# line-of-sight velocity distribution; their values are dimensionless.
MOMENTS = ("h3", "h4")
OBSERVABLES = ("sigma_los", "sigma_los_aperture") + BINNED + MOMENTS


@dataclass(frozen=True)
class Data:
    r"""Measurements of one observable, with their errors.

    A `Data` holds one set of measured values, the projected radii,
    apertures or 3D radius at which the values were measured, and the
    errors of the values. `likelihood` compares a model with one `Data` or
    with several.

    Parameters
    ----------
    observable : str
        The measured quantity, one of seven observables. ``"sigma_los"``
        is the line-of-sight velocity dispersion :math:`\sigma_{\rm los}`
        at user-specified projected radii. ``"sigma_los_aperture"`` is
        :math:`\sigma_{\rm los}` averaged over each aperture with the
        surface density as the weight.
        ``"losvd"`` is the binned line-of-sight velocity distribution at one
        projected radius, and ``"losvd_aperture"`` the binned line-of-sight
        velocity distribution in one aperture. ``"pdf_v"`` is the binned
        local velocity distribution at one 3D radius. ``"h3"`` and ``"h4"`` are
        the Gauss-Hermite moments :math:`h_3` and :math:`h_4` of the
        line-of-sight velocity distribution at user-specified projected
        radii or in apertures.
    values : array-like, shape (n,)
        The measured values. A value of ``"sigma_los"`` or
        ``"sigma_los_aperture"`` is a dispersion in the velocity unit of
        `unit_system`. A value of ``"losvd"``, ``"losvd_aperture"`` or
        ``"pdf_v"`` is the fraction of the distribution inside the
        user-specified range ``[v_edges[0], v_edges[-1]]`` that falls in one
        bin. A value of ``"h3"`` or ``"h4"`` is dimensionless.
    r_2d : array-like, shape (n,), optional
        User-specified projected radii :math:`R_{\rm 2D}`, positive, in the
        length unit of `unit_system`: one projected radius per value for
        ``"sigma_los"``, ``"h3"`` and ``"h4"``, and one projected radius for
        ``"losvd"``. Default: ``None``, which means no projected radii, as
        ``"sigma_los_aperture"``, ``"losvd_aperture"`` and ``"pdf_v"`` take.
    apertures : sequence of Aperture, optional
        The apertures, with lengths in the length unit of `unit_system`: one
        aperture per value for ``"sigma_los_aperture"``, ``"h3"`` and
        ``"h4"``, and one aperture for ``"losvd_aperture"``. Default:
        ``()``, which means no apertures.
    errors : array-like, shape (n,), optional
        The standard deviation of the error of each value, positive, for
        independent errors. Exactly one of `errors` and `covariance` is
        given. Default: ``None``, which means `covariance` gives the errors.
    covariance : array-like, shape (n, n), optional
        The covariance matrix of the errors, symmetric. `likelihood`
        requires the covariance matrix to be positive definite as well.
        Default: ``None``, which means `errors` gives the errors.
    unit_system : {"internal", "model", "physical"}, optional
        The unit system of the data. In the physical system a length is in
        parsecs and a velocity in kilometers per second. `likelihood`
        compares a model only in the unit system of the data. Default:
        ``"internal"``, which means internal units.
    v_edges : array-like, shape (n + 1,), optional
        The bin edges of ``"losvd"``, ``"losvd_aperture"`` or ``"pdf_v"``,
        increasing, in the velocity unit of `unit_system`. The first edge of
        ``"pdf_v"`` is not negative. Default: ``None``, which means no bins;
        ``"losvd"``, ``"losvd_aperture"`` and ``"pdf_v"`` require `v_edges`.
    r : array-like, shape (1,), optional
        The 3D radius of ``"pdf_v"``, positive, in the length unit of
        `unit_system`. Default: ``None``, which means no 3D radius;
        ``"pdf_v"`` requires `r`.

    Raises
    ------
    InputError
        If `observable` is not one of the seven observables, or
        `unit_system` is not one of the three unit systems.
    InputError
        If `values` is not a non-empty one-dimensional array.
    InputError
        If a field that `observable` requires (`r_2d`, `apertures`,
        `v_edges` or `r`) is missing, has the wrong length or holds a
        radius that is not positive, or if a field that `observable` does
        not take is given.
    InputError
        If `v_edges` is not increasing, or the first edge of ``"pdf_v"`` is
        negative.
    InputError
        If both `errors` and `covariance` are given or neither is given, if
        an error is not positive, or if `covariance` is not a symmetric
        matrix of n rows and n columns.

    See Also
    --------
    likelihood : The log-likelihood of a model against one or several `Data`.
    log_likelihood : The log-likelihood at the model's own weights and scales.
    Aperture : An aperture of ``"sigma_los_aperture"``, ``"losvd_aperture"``,
        ``"h3"`` or ``"h4"``.

    Examples
    --------
    >>> import pianistpy as pp
    >>> data = pp.Data(
    ...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
    ... )
    >>> data.values.shape
    (2,)
    >>> data.unit_system
    'internal'
    """

    observable: str
    values: Array
    r_2d: Array | None = None
    apertures: tuple[Aperture, ...] = ()
    errors: Array | None = None
    covariance: Array | None = None
    unit_system: str = "internal"
    v_edges: Array | None = None
    r: Array | None = None

    def __post_init__(self) -> None:
        if self.observable not in OBSERVABLES:
            raise InputError(
                f"The observable of Data must be one of {OBSERVABLES}; the value "
                f"given is {self.observable!r}."
            )
        if self.unit_system not in SYSTEMS:
            raise InputError(
                f"The unit system of Data must be one of {SYSTEMS}; the value "
                f"given is {self.unit_system!r}."
            )
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise InputError(
                "The values of Data must be a non-empty one-dimensional array."
            )
        object.__setattr__(self, "values", values)
        n = values.size
        binned = self.observable in BINNED
        if binned:
            if self.v_edges is None:
                raise InputError(
                    f"Data of the observable {self.observable} need v_edges, the "
                    "edges of the velocity bins."
                )
            edges = np.asarray(self.v_edges, dtype=np.float64)
            if edges.shape != (n + 1,) or not np.all(np.diff(edges) > 0.0):
                raise InputError(
                    f"The velocity bin edges v_edges must be {n + 1} increasing "
                    "velocities, one more than the number of values."
                )
            if self.observable == "pdf_v" and edges[0] < 0.0:
                raise InputError(
                    "The velocity bin edges of pdf_v data are velocity magnitudes "
                    "and must not be negative; the first edge given is negative."
                )
            object.__setattr__(self, "v_edges", edges)
        elif self.v_edges is not None or self.r is not None:
            raise InputError(
                f"Data of the observable {self.observable} take neither v_edges nor r: "
                "v_edges belong to the binned observables, and r belongs to pdf_v."
            )
        if self.observable == "pdf_v":
            if self.r is None or self.r_2d is not None or self.apertures:
                raise InputError(
                    "Data of the observable pdf_v need r, the 3D radius, and take "
                    "neither r_2d nor apertures."
                )
            r = np.asarray(self.r, dtype=np.float64).reshape(-1)
            if r.shape != (1,) or not r[0] > 0.0:
                raise InputError(
                    "The 3D radius r of pdf_v data must be one positive radius."
                )
            object.__setattr__(self, "r", r)
        elif self.r is not None:
            raise InputError(
                f"Data of the observable {self.observable} take no 3D radius r; only "
                "pdf_v data take r."
            )
        elif self.observable in ("sigma_los", "losvd") or (
            self.observable in MOMENTS and self.r_2d is not None
        ):
            m = 1 if binned else n
            if self.r_2d is None or self.apertures:
                raise InputError(
                    f"Data of the observable {self.observable} need the projected "
                    "radii r_2d and take no apertures."
                )
            r_2d = np.asarray(self.r_2d, dtype=np.float64).reshape(-1)
            if r_2d.shape != (m,) or not np.all(r_2d > 0.0):
                raise InputError(
                    f"The projected radii r_2d must be {m} positive radii."
                )
            object.__setattr__(self, "r_2d", r_2d)
        else:
            m = 1 if binned else n
            if self.r_2d is not None or len(self.apertures) != m:
                raise InputError(
                    f"Data of the observable {self.observable} need {m} apertures"
                    + (
                        f" or {m} projected radii r_2d."
                        if self.observable in MOMENTS
                        else " and take no projected radii r_2d."
                    )
                )
            if not all(isinstance(a, Aperture) for a in self.apertures):
                raise InputError("Every entry of apertures must be an Aperture object.")
            object.__setattr__(self, "apertures", tuple(self.apertures))
        if (self.errors is None) == (self.covariance is None):
            raise InputError(
                "Data need exactly one of errors and covariance; the values given set "
                "both or neither."
            )
        if self.errors is not None:
            errors = np.asarray(self.errors, dtype=np.float64)
            if errors.shape != (n,) or not np.all(errors > 0.0):
                raise InputError(
                    f"The errors of Data must be {n} positive values, one for each "
                    "value."
                )
            object.__setattr__(self, "errors", errors)
        else:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if covariance.shape != (n, n) or not np.allclose(
                covariance, covariance.T, rtol=0.0, atol=0.0
            ):
                raise InputError(
                    f"The covariance of Data must be a symmetric {n} by {n} matrix, "
                    "one row and one column for each value."
                )
            object.__setattr__(self, "covariance", covariance)

    @property
    def identity(self) -> str:
        """The identity string of the data, a digest of every field of the data."""
        return identity_string(_data_record(self), {}, __version__)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the data to an HDF5 file, which `pianistpy.read` reads back.

        Parameters
        ----------
        path : str or path-like
            The path of the HDF5 file to write.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means an existing file at `path` raises
            `InputError`.

        Raises
        ------
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as the data.

        See Also
        --------
        pianistpy.read : Read a result file, a file of `Data` included.
        """
        arrays = {"values": self.values}
        for name in ("r_2d", "errors", "covariance", "v_edges", "r"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        record = {
            "observable": self.observable,
            "unit_system": self.unit_system,
            "apertures": [a.to_dict() for a in self.apertures],
        }
        write_result(
            path, RESULT_KIND, self.identity, record, arrays, overwrite=overwrite
        )


register_result(Data, ("values", "r_2d", "errors", "covariance", "v_edges", "r"))


def _data_record(data: Data) -> dict[str, Any]:
    """The dictionary form of `data`, with every array as a list, from which
    the identity string of `data` is computed."""

    def plain(value: Any) -> Any:
        return None if value is None else np.asarray(value).tolist()

    record = {
        "observable": data.observable,
        "unit_system": data.unit_system,
        "apertures": [a.to_dict() for a in data.apertures],
        "values": plain(data.values),
        "r_2d": plain(data.r_2d),
        "errors": plain(data.errors),
        "covariance": plain(data.covariance),
    }
    # The fields v_edges and r enter the record only where the data hold
    # them, so that the identity string of data without bins depends only on
    # observable, unit_system, apertures, values, r_2d, errors and covariance.
    for name in ("v_edges", "r"):
        if getattr(data, name) is not None:
            record[name] = plain(getattr(data, name))
    return record


def _read_data(source: StoredResult | str | os.PathLike) -> Data:
    """Return the `Data` stored in a result file or in a stored result."""
    stored = load_result(source)
    if stored.kind != RESULT_KIND:
        raise ResultError(
            f"The stored result read is of kind {stored.kind!r}; Data are read only "
            f"from a result of kind {RESULT_KIND!r}."
        )
    arrays = stored.arrays
    return Data(
        observable=str(stored.record["observable"]),
        values=arrays["values"],
        r_2d=arrays.get("r_2d"),
        apertures=tuple(Aperture.from_dict(a) for a in stored.record["apertures"]),
        errors=arrays.get("errors"),
        covariance=arrays.get("covariance"),
        unit_system=str(stored.record["unit_system"]),
        v_edges=arrays.get("v_edges"),
        r=arrays.get("r"),
    )


register_reader(RESULT_KIND, _read_data)


@dataclass(frozen=True)
class _Table:
    """The projected table of one component, with the reader of the
    component's surface density and the reader of the component's squared
    line-of-sight dispersion. The likelihood evaluates both readers in JAX,
    as `ProjectedDispersion.read_traced` evaluates them, and places the
    aperture nodes on the surface density reader."""

    projected: ProjectedDispersion
    surface: Any
    square: Any
    # The intrinsic dispersion profile from which `projected` was computed is
    # kept where `_table` computed the profile, so that the binned
    # distributions read the profile instead of computing the profile again.
    profile: DispersionProfile | None = None

    def read(self, x: Array) -> tuple[Array, Array]:
        """Return the surface density and the squared line-of-sight
        dispersion at the internal projected radii `x`, as JAX arrays. Both
        readings are zero beyond the outermost tabulated radius, and inside
        the innermost tabulated radius both readings continue the cubic of
        the innermost interval. The preparation in `likelihood` rejects or
        moves every radius outside the tabulated radii before `_Table.read`
        receives the radius."""
        return self.surface.traced(x, jnp), self.square.traced(x, jnp)


@jax.custom_vjp
def _not_carried(length: Array) -> Array:
    """Return zero, with a gradient of NaN in the length scale. The binned
    distributions are computed at the model's own length scale and carry
    no gradient in the length scale, so the likelihood adds `_not_carried`
    of the length scale: a caller then reads the gradient in the length
    scale as NaN instead of as a wrong number."""
    return jnp.zeros_like(length)


def _not_carried_forward(length: Array) -> Any:
    return jnp.zeros_like(length), None


def _not_carried_backward(_: Any, cotangent: Array) -> Any:
    return (jnp.full_like(cotangent, jnp.nan),)


_not_carried.defvjp(_not_carried_forward, _not_carried_backward)


def _square(breaks: Array, coefficients: Array, x: Array, x_end: float) -> Array:
    """Return the squared dispersion of a padded spline table at the internal
    radii `x`, read as `DispersionReading.traced` reads a dispersion table.
    The value is NaN
    where the spline is negative and zero where the logarithm of `x`
    exceeds `x_end`, the logarithm of the outermost tabulated radius."""
    log_r = jnp.log(jnp.maximum(x, TINY))
    value = spline_value(breaks, coefficients, log_r, jnp)
    value = jnp.where(log_r > x_end, 0.0, value)
    return jnp.where(value < 0.0, jnp.nan, value)


@dataclass(frozen=True)
class _Block:
    """One `Data` of dispersions prepared for the likelihood. The block holds
    the internal radii `x` of the nodes at the model's own length scale; for
    apertures, the weight `q` of each node in each measurement (a projected
    radius is read at the projected radius itself, without weights); and the
    Cholesky factor or the errors and the normalization of the Gaussian,
    which no parameter of the likelihood changes."""

    x: Array
    q: Array | None
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float
    points: bool


@dataclass(frozen=True)
class _BinnedBlock:
    """One binned distribution prepared for the likelihood. The block holds
    the observable, the bin edges in the velocity unit of the data, each
    component's share of the mixture at the position of the data, each
    component's distribution as padded velocity intervals `a` to `b` with
    the Legendre series of the antiderivative on each interval, and the
    Cholesky factor or the errors and the normalization of the Gaussian."""

    kind: str
    v_edges: Array
    share: Array
    a: Array
    b: Array
    series: Array
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float


@dataclass(frozen=True)
class _Moments:
    """One `Data` of Gauss-Hermite moments in apertures prepared for the
    likelihood. For each component and each aperture, the block holds the
    component's velocity distribution in the aperture on the component's
    own velocity nodes `v`, mirrored to negative velocities, padded with
    zero weights to one length and normalized to unit integral, and the
    component's projected mass in the aperture. The block also holds the
    Cholesky factor or the errors and the normalization of the Gaussian."""

    order: int
    v: Array
    w: Array
    d: Array
    mass: Array
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float

    def predict(self, weights: Array) -> Array:
        """Return the Gauss-Hermite moment of the mixture in each aperture.
        The mixture's velocity distribution is the sum of the components'
        distributions, each weighted by the component's entry of `weights`
        times the component's projected mass in the aperture. The moment is
        taken about the mixture's own mean velocity, in units of the
        mixture's own dispersion."""
        share = weights[:, None] * self.mass
        share = share / jnp.sum(share, axis=0)
        wd = share[:, :, None] * self.w * self.d
        mean = jnp.sum(wd * self.v, axis=(0, 2))
        dv = self.v - mean[None, :, None]
        sigma = jnp.sqrt(jnp.sum(wd * dv * dv, axis=(0, 2)))
        y = dv / sigma[None, :, None]
        gauss = 2.0 * jnp.sqrt(jnp.pi) * jnp.exp(-0.5 * y * y) / jnp.sqrt(2.0 * jnp.pi)
        if self.order == 3:
            h = (2.0 * y**3 - 3.0 * y) / jnp.sqrt(3.0)
        else:
            h = (4.0 * y**4 - 12.0 * y * y + 3.0) / (2.0 * jnp.sqrt(6.0))
        return jnp.sum(wd * h * gauss, axis=(0, 2))


@dataclass(frozen=True)
class _PointMoments:
    """One `Data` of Gauss-Hermite moments at projected radii prepared for the
    likelihood, on a model of one component. The block holds the table of
    the model's moments over projected radius (`gauss_hermite_table`), the
    internal radii `x` of the data at the model's own length scale (a length
    scale l reads the table at x = R / (L l)), the innermost radius `r_min`
    of the table, and the Cholesky factor or the errors and the
    normalization of the Gaussian."""

    key: str
    table: GaussHermiteTable
    x: Array
    r_min: float
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float


@dataclass(frozen=True)
class Likelihood:
    r"""The log-likelihood of a model against data, as a JAX function.

    `likelihood` builds a `Likelihood`. A call of a `Likelihood` with the
    component weights, the mass and length scales and, for free components,
    the anisotropy parameters returns the Gaussian log-likelihood. JAX can
    compile the call and differentiate the call in each argument.

    Attributes
    ----------
    tables : tuple
        The projected table of each component, read in JAX.
    blocks : tuple
        Each `Data` of dispersions or Gauss-Hermite moments, prepared.
    weights : array, shape (n_components,)
        The model's own component weights, which are the masses of the
        components.
    scale : array, shape (2,)
        The model's own mass scale and length scale, in the units of the
        data.
    length : float
        The length factor :math:`L` of the unit system of the data at the
        model's own scales.
    velocity : float
        The velocity factor :math:`V` of the unit system of the data at the
        model's own scales.
    r_min : float
        The innermost projected radius of the tables, in internal units.
    r_max : float
        The outermost projected radius of the tables, in internal units.
    free : tuple of int
        The indices of the components whose anisotropy parameters
        (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) are parameters of the
        likelihood.
    theta : array, shape (n_free, 2)
        The model's own (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) of each
        free component, in internal units.
    members : tuple
        The free components, each with the tables computed at every
        anisotropy at which the `Likelihood` has been called.
    readings : tuple
        The squared line-of-sight dispersion :math:`\sigma_{\rm los}^2` of
        each free component, as a JAX function of the component's
        anisotropy parameters.
    binned : tuple
        Each `Data` of ``"losvd"``, ``"losvd_aperture"`` or ``"pdf_v"``,
        prepared.

    See Also
    --------
    likelihood : Build the `Likelihood` of a model against data.
    log_likelihood : The log-likelihood at the model's own weights and scales.
    Data : The measurements a `Likelihood` compares with.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> data = pp.Data(
    ...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
    ... )
    >>> like = pp.likelihood(model, data)
    >>> bool(float(like()) > float("-inf"))
    True
    >>> float(like(scale=[1.0, -1.0]))
    -inf
    """

    tables: tuple[_Table, ...]
    blocks: tuple[_Block | _Moments | _PointMoments, ...]
    weights: Array
    scale: Array
    length: float
    velocity: float
    r_min: float
    r_max: float
    free: tuple[int, ...] = ()
    theta: Array | None = None
    members: tuple[_Free, ...] = ()
    readings: tuple[Any, ...] = ()
    binned: tuple[_BinnedBlock, ...] = ()

    def __call__(
        self,
        weights: ArrayLike | None = None,
        scale: ArrayLike | None = None,
        theta: ArrayLike | None = None,
    ) -> Array:
        r"""Return the log-likelihood at the given parameters.

        Parameters
        ----------
        weights : array-like, shape (n_components,), optional
            The component weights, non-negative. The call divides the
            weights by the sum of the weights. Default: ``None``, which
            means the model's own component weights.
        scale : array-like, shape (2,), optional
            The mass scale and the length scale, in the units of the data.
            Default: ``None``, which means the model's own mass scale and
            length scale.
        theta : array-like, shape (n_free, 2), optional
            The anisotropy parameters (:math:`\beta_{\rm c}`,
            :math:`R_{\rm a}`) of each free component, in the order of
            `free`, in internal units. For an isotropic or constant-anisotropy
            component, the :math:`R_{\rm a}` column stays at the component's
            own value, infinity, and the gradient in the :math:`R_{\rm a}`
            column is zero. Default: ``None``, which means the free
            components' own anisotropy parameters.

        Returns
        -------
        log_likelihood : scalar array
            The Gaussian log-likelihood summed over the data. The value is
            minus infinity where any of the following conditions holds:

            - a weight is negative, or the sum of the weights is zero;
            - a scale is not positive;
            - a projected radius of the data, at the given length scale,
              lies inside the innermost tabulated projected radius;
            - the nodes of a measurement carry no surface density;
            - a free component's :math:`\beta_{\rm c}` is not below 1, the
              component's :math:`R_{\rm a}` is not positive, or the
              component's model at the given anisotropy cannot be built (the
              gradient is then zero);
            - the data include binned distributions or Gauss-Hermite moments
              in apertures, and the length scale differs from the model's
              own length scale, because the distributions of binned data and
              of moments in apertures are computed at the model's own radii.

        Raises
        ------
        InputError
            If `theta` is given and the `Likelihood` has no free component.
        """
        w = self.weights if weights is None else jnp.asarray(weights, dtype=jnp.float64)
        s = self.scale if scale is None else jnp.asarray(scale, dtype=jnp.float64)
        valid = jnp.all(w >= 0.0) & (jnp.sum(w) > 0.0) & jnp.all(s > 0.0)
        # Invalid weights and scales are replaced by the model's own before
        # the weights and scales are used, so that neither the value nor the
        # gradient of the valid branch of jnp.where meets a NaN.
        w = jnp.where(valid, w, self.weights)
        s = jnp.where(valid, s, self.scale)
        squares: dict[int, tuple[Array, Array]] = {}
        if theta is not None and not self.free:
            raise InputError(
                "The likelihood is called with theta, but no component's anisotropy is "
                "free in the likelihood."
            )
        if self.free:
            assert self.theta is not None
            t = (
                self.theta
                if theta is None
                else jnp.asarray(theta, dtype=jnp.float64).reshape(len(self.free), 2)
            )
            valid = valid & jnp.all(t[:, 0] < 1.0) & jnp.all(t[:, 1] > 0.0)
            t = jnp.where(valid, t, self.theta)
            for j, k in enumerate(self.free):
                breaks, coefficients, ok = self.readings[j](t[j])
                valid = valid & (ok > 0.0)
                squares[k] = (breaks, coefficients)
        ratio = s / self.scale
        stretch = 1.0 / ratio[1]
        velocity = self.velocity * jnp.sqrt(ratio[0] / ratio[1])
        total: Array = 0.0
        for block in self.blocks:
            if isinstance(block, _PointMoments):
                # Gauss-Hermite moments are dimensionless, so the mass scale
                # does not change the moments; the length scale moves the
                # radii at which the moments' table is read.
                x = block.x * stretch
                inside = jnp.all(x >= block.r_min * (1.0 - END_TOLERANCE))
                valid = valid & inside
                x = jnp.where(inside, x, block.x)
                moment = block.table.read_traced(x, jnp)[block.key]
                total = total - _gaussian(block, block.values - moment)
                continue
            if isinstance(block, _Moments):
                # Gauss-Hermite moments are dimensionless, so the mass scale
                # does not change the moments; the distributions in the
                # apertures are computed at the model's own length scale, so
                # any other length scale makes the log-likelihood minus
                # infinity.
                valid = valid & (s[1] == self.scale[1])
                total = total - _gaussian(block, block.values - block.predict(w))
                continue
            x = block.x * stretch
            if block.points:
                inside = jnp.all(x >= self.r_min * (1.0 - END_TOLERANCE))
                valid = valid & inside
                x = jnp.where(inside, x, block.x)
            # A node beyond the outermost tabulated radius is read at the
            # outermost tabulated radius and set to zero afterwards. Reading
            # the node at the node's own radius would evaluate the logarithm
            # of the edge term beyond the table, whose NaN gradient would pass
            # through jnp.where.
            beyond = x > self.r_max * (1.0 + END_TOLERANCE)
            x = jnp.where(beyond, self.r_max, x)
            read = [
                (
                    table.surface.traced(x, jnp),
                    _square(*squares[k], x, float(np.log(table.surface.r_max))),
                )
                if k in squares
                else table.read(x)
                for k, table in enumerate(self.tables)
            ]
            sigma = jnp.where(beyond, 0.0, jnp.stack([r[0] for r in read]))
            square = jnp.where(beyond, 0.0, jnp.stack([r[1] for r in read]))
            numerator = mixture(w, _by_measurement(sigma * square, block), jnp)
            denominator = mixture(w, _by_measurement(sigma, block), jnp)
            has_mass = jnp.all(denominator > 0.0)
            valid = valid & has_mass
            mean_square = jnp.where(
                has_mass, numerator / jnp.where(has_mass, denominator, 1.0), 1.0
            )
            residual = block.values - velocity * jnp.sqrt(mean_square)
            if block.chol is None:
                z = residual / block.errors
            else:
                z = solve_triangular(block.chol, residual, lower=True)
            total = total - 0.5 * jnp.dot(z, z) - block.log_norm
        if self.binned:
            # The binned distributions are computed at the model's own length
            # scale, so any other length scale makes the log-likelihood minus
            # infinity, and _not_carried makes the gradient in the length
            # scale NaN.
            valid = valid & (s[1] == self.scale[1])
            total = total + _not_carried(s[1])
            velocity = self.velocity * jnp.sqrt(ratio[0])
            for bin_block in self.binned:
                a, b, series = bin_block.a, bin_block.b, bin_block.series
                u = bin_block.v_edges / velocity
                cumulative = jnp.stack(
                    [
                        _binned.cumulative(a[k], b[k], series[k], jnp.abs(u))
                        for k in range(a.shape[0])
                    ]
                )
                if bin_block.kind != "pdf_v":
                    # The line-of-sight velocity distribution is even in the
                    # velocity, so the cumulative distribution from 0 to a
                    # negative edge u is minus the cumulative distribution
                    # from 0 to -u.
                    cumulative = jnp.sign(u) * cumulative
                parts = cumulative[:, 1:] - cumulative[:, :-1]
                inside = (w * bin_block.share) @ parts
                window = jnp.sum(inside)
                has_mass = window > 0.0
                valid = valid & has_mass
                residual = bin_block.values - inside / jnp.where(has_mass, window, 1.0)
                if bin_block.chol is None:
                    z = residual / bin_block.errors
                else:
                    z = solve_triangular(bin_block.chol, residual, lower=True)
                total = total - 0.5 * jnp.dot(z, z) - bin_block.log_norm
        return jnp.where(valid, total, -jnp.inf)


def _gaussian(block: _Moments | _PointMoments, residual: Array) -> Array:
    """Return minus the Gaussian log-likelihood of the residuals `residual`:
    r^T C^-1 r / 2 for residuals r and covariance matrix C, plus the
    normalization `block.log_norm`."""
    if block.chol is None:
        z = residual / block.errors
    else:
        z = solve_triangular(block.chol, residual, lower=True)
    return 0.5 * jnp.dot(z, z) + block.log_norm


def _internal(aperture: Aperture, length: float) -> Aperture:
    """Return `aperture`, given in the length unit of the data, with every
    length divided by `length`, which gives the aperture in internal
    units."""
    return replace(
        aperture,
        r_off=aperture.r_off / length,
        r_ape=None if aperture.r_ape is None else aperture.r_ape / length,
        side_x=None if aperture.side_x is None else aperture.side_x / length,
        side_y=None if aperture.side_y is None else aperture.side_y / length,
    )


def _factors(model: Model, unit_system: str) -> tuple[float, float, np.ndarray]:
    """Return the length factor and the velocity factor of `unit_system` at
    the model's own scales, and the model's own mass scale and length scale
    in `unit_system`."""
    units = model.units
    if units.system != unit_system:
        raise InputError(
            f"The data are in the {unit_system} unit system and the model is in the "
            f"{units.system} unit system; convert the model to the data's unit "
            "system with Model.to."
        )
    if unit_system == "internal":
        return 1.0, 1.0, np.ones(2)
    factors = units.factors(model.components[0].unit_profile)
    if unit_system == "model":
        return 1.0, float(factors.model_v), np.ones(2)
    length, velocity = factors.physical_r_pc, factors.physical_v_kms
    assert length is not None and velocity is not None
    assert units.mass_0_msun is not None and units.r_0_pc is not None
    return length, velocity, np.array([units.mass_0_msun, units.r_0_pc])


def _by_measurement(values: Array, block: _Block) -> Array:
    """Return each component's values at the nodes of `block` carried to the
    measurements of `block`: the values themselves for projected radii, and
    the values weighted by the aperture weights `block.q` for apertures."""
    if block.points:
        return values
    assert block.q is not None
    return values @ block.q.T


def _block(data: Data, surface: Any, length: float, settings: Settings) -> _Block:
    """Return one `Data` of dispersions prepared: the nodes of the data, the
    weights of the nodes, and the Cholesky factor or the errors and the
    normalization of the Gaussian."""
    n = data.values.size
    if data.observable == "sigma_los":
        x = np.asarray(data.r_2d) / length
        if np.any(x < surface.r_min * (1.0 - END_TOLERANCE)):
            raise InputError(
                f"A projected radius of the data lies inside the innermost projected "
                f"radius of the model's table, {surface.r_min * length:.4e} in the "
                "data's units."
            )
        if np.any(x > surface.r_max * (1.0 + END_TOLERANCE)):
            raise InputError(
                f"A projected radius of the data lies beyond the outermost projected "
                f"radius of the model's table, {surface.r_max * length:.4e} in the "
                "data's units, where the model has no surface density."
            )
        q = None
        points = True
    else:
        observables = settings.observables
        nodes, weights = [], []
        for aperture in data.apertures:
            scaled = replace(
                aperture,
                r_off=aperture.r_off / length,
                r_ape=None if aperture.r_ape is None else aperture.r_ape / length,
                side_x=None if aperture.side_x is None else aperture.side_x / length,
                side_y=None if aperture.side_y is None else aperture.side_y / length,
            )
            r, w, _ = aperture_nodes(
                surface,
                scaled,
                int(observables["n_aperture_nodes"]),
                int(observables["aperture_nodes_per_decade"]),
                INNER_DISK_TOLERANCE,
            )
            nodes.append(r)
            weights.append(w)
        x = np.concatenate(nodes)
        q = np.zeros((n, x.size))
        start = 0
        for i, w in enumerate(weights):
            q[i, start : start + w.size] = w
            start += w.size
        points = False
    if data.errors is not None:
        errors = np.asarray(data.errors)
        chol = None
        log_norm = float(np.sum(np.log(errors))) + 0.5 * n * math.log(2.0 * math.pi)
    else:
        errors = None
        chol = np.linalg.cholesky(np.asarray(data.covariance))
        log_norm = float(np.sum(np.log(np.diag(chol)))) + 0.5 * n * math.log(
            2.0 * math.pi
        )
    return _Block(
        x=jnp.asarray(x),
        q=None if q is None else jnp.asarray(q),
        values=jnp.asarray(data.values),
        chol=None if chol is None else jnp.asarray(chol),
        errors=None if errors is None else jnp.asarray(errors),
        log_norm=log_norm,
        points=points,
    )


class _Shared:
    """Each component's kept ranges and table of velocity distributions,
    computed once for all the `Data` of one likelihood, when the first
    `Data` that reads a component's table needs the table."""

    def __init__(
        self, views: tuple[Model, ...], tables: tuple[_Table, ...], settings: Settings
    ) -> None:
        self.views, self.tables, self.settings = views, tables, settings
        self._kept: dict[int, Any] = {}
        self._losvd: dict[int, Any] = {}

    def kept(self, k: int) -> Any:
        if k not in self._kept:
            self._kept[k] = kept_ranges(
                self.views[k], self.tables[k].projected, self.settings
            )
        return self._kept[k]

    def losvd(self, k: int) -> Any:
        if k not in self._losvd:
            self._losvd[k] = losvd_table(
                self.views[k], self.kept(k), self.tables[k].projected, self.settings
            )
        return self._losvd[k]


def _point_moments(
    data: Data,
    views: tuple[Model, ...],
    tables: tuple[_Table, ...],
    length: float,
    settings: Settings,
    shared: _Shared,
) -> _PointMoments:
    """Return one `Data` of Gauss-Hermite moments at projected radii
    prepared: the model's table of velocity distributions, computed once,
    and the table of the model's moments over projected radius."""
    if len(views) != 1:
        raise InputError(
            "Gauss-Hermite moments at projected radii are computed for a model of one "
            "component only. The moments of a superposition are the moments of the "
            "superposition's summed distribution, not the members' moments, so the "
            "moments of a superposition are given over apertures."
        )
    table = gauss_hermite_table(shared.losvd(0))
    x = np.asarray(data.r_2d) / length
    r_min, r_max = float(table.r_2d[0]), float(table.r_2d[-1])
    if np.any(x < r_min * (1.0 - END_TOLERANCE)) or np.any(
        x > r_max * (1.0 + END_TOLERANCE)
    ):
        raise InputError(
            f"A projected radius of the data lies outside the projected radii of the "
            f"model's moments, "
            f"[{r_min * length:.4e}, {r_max * length:.4e}] in the data's units."
        )
    chol, errors, log_norm = _noise(data)
    return _PointMoments(
        key=data.observable,
        table=table,
        x=jnp.asarray(x),
        r_min=r_min,
        values=jnp.asarray(data.values),
        chol=chol,
        errors=errors,
        log_norm=log_norm,
    )


def _moments(
    data: Data,
    views: tuple[Model, ...],
    tables: tuple[_Table, ...],
    length: float,
    settings: Settings,
    shared: _Shared,
) -> _Moments:
    """Return one `Data` of Gauss-Hermite moments in apertures prepared: each
    component's table of velocity distributions (`kept_ranges` and
    `losvd_table`, computed once for all the `Data` through `_Shared`) and
    each component's distribution in each aperture (`aperture_losvd`),
    mirrored to negative velocities."""
    rows = []
    for k, (view, table) in enumerate(zip(views, tables, strict=True)):
        projected = table.projected
        losvd = shared.losvd(k)
        rows.append(
            [
                aperture_losvd(
                    view, losvd, projected, _internal(aperture, length), settings
                )
                for aperture in data.apertures
            ]
        )
    n_max = 2 * max(a.v_nodes.size for row in rows for a in row)
    shape = (len(views), len(data.apertures), n_max)
    v, w, d = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    mass = np.zeros(shape[:2])
    for k, row in enumerate(rows):
        for i, a in enumerate(row):
            n = a.v_nodes.size
            v[k, i, : 2 * n] = np.concatenate([-a.v_nodes[::-1], a.v_nodes])
            w[k, i, : 2 * n] = np.concatenate([a.v_weights[::-1], a.v_weights])
            mirrored = np.concatenate([a.losvd[0][::-1], a.losvd[0]])
            d[k, i, : 2 * n] = mirrored / float(np.dot(w[k, i, : 2 * n], mirrored))
            mass[k, i] = a.mass[0]
    chol, errors, log_norm = _noise(data)
    return _Moments(
        order=int(data.observable[1]),
        v=jnp.asarray(v),
        w=jnp.asarray(w),
        d=jnp.asarray(d),
        mass=jnp.asarray(mass),
        values=jnp.asarray(data.values),
        chol=chol,
        errors=errors,
        log_norm=log_norm,
    )


def _table(
    model: Model, settings: Settings, projected: ProjectedDispersion | None = None
) -> _Table:
    """Return the projected table of a model of one component, with the
    readers that evaluate the table as `ProjectedDispersion.read` evaluates
    the table. The dispersion profile and the projection are computed here
    unless `projected` is given."""
    profile = None
    if projected is None:
        profile = dispersion_profile(model, settings)
        projected = projected_dispersion(model, profile, settings)
    return _Table(
        projected,
        surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        ),
        dispersion_table_reader(
            projected.r_2d, np.asarray(projected.sigma_los[0]) ** 2
        ),
        profile,
    )


def likelihood(
    model: Model,
    data: Data | Sequence[Data],
    settings: Settings | None = None,
    free: Sequence[int] = (),
) -> Likelihood:
    r"""Return the log-likelihood of a model against data, as a JAX function.

    `likelihood` computes the projected table of each component once and
    returns a `Likelihood`, which reads the tables in JAX at every call. A
    model and data that cannot be compared raise an exception in
    `likelihood`, never in a call of the returned `Likelihood`. For one
    value at the model's own weights and scales, call `log_likelihood`.

    Parameters
    ----------
    model : Model
        A model of one component, or a superposition (`superpose`), in the
        unit system of the data.
    data : Data or sequence of Data
        The measurements. The log-likelihoods of several `Data` are summed.
    settings : Settings, optional
        The numerical settings of the observables. Default: ``None``, which
        means the model's own settings.
    free : sequence of int, optional
        The indices of the components whose anisotropy parameters
        (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) become the parameter
        ``theta`` of the returned `Likelihood`. Each free component is a
        spatially finite Method II model: a DF computed on the truncated
        profile of another model, where the truncated profile carries the
        DF of the other model.
        For an isotropic or constant-anisotropy component,
        :math:`\beta_{\rm c}` is free and the :math:`R_{\rm a}` column stays
        at the component's own value, infinity, with a zero gradient,
        because the isotropic and constant-anisotropy models have no
        anisotropy radius. Default: ``()``, which means no anisotropy
        parameter is a parameter of the likelihood.

    Returns
    -------
    likelihood : Likelihood
        The log-likelihood as a JAX function of the component weights, the
        mass and length scales and, for the components in `free`, the
        anisotropy parameters. A call of the `Likelihood` without arguments
        returns the value at the model's own weights and scales, the value
        `log_likelihood` returns.

    Raises
    ------
    InputError
        If `model` is not a `Model`, or `model` has several components and
        is not a superposition.
    InputError
        If `data` is empty or holds an object that is not a `Data`.
    InputError
        If the data are in several unit systems, or in a unit system other
        than the unit system of `model` (`Model.to` presents a model in
        another unit system).
    InputError
        If `free` repeats a component index or lists a component index that
        `model` does not have.
    InputError
        If a free component is not spatially finite, is not a DF on a
        truncated profile that carries the DF of another model, or has an
        anisotropy model with neither a :math:`\beta_{\rm c}` nor an
        :math:`R_{\rm a}`.
    InputError
        If `free` is given together with binned distributions or
        Gauss-Hermite moments.
    InputError
        If the projected tables of the components differ in the innermost
        tabulated radius, the outermost tabulated radius or the edge radius.
    InputError
        If a projected radius of ``"sigma_los"`` data lies inside the
        innermost or beyond the outermost projected radius of the model's
        table. The likelihood does not extrapolate the table.
    InputError
        If Gauss-Hermite moments at projected radii are given on a
        superposition, or a projected radius of the moments lies outside the
        projected radii of the model's table of moments. A superposition
        takes Gauss-Hermite moments in apertures only.
    InputError
        If an aperture encloses no tabulated surface density, or if the disk
        inside the innermost tabulated projected radius carries more than
        :math:`10^{-6}` of the aperture's weight.
    InputError
        If the 3D radius of ``"pdf_v"`` data lies outside the profile or,
        for a Type II model, at or beyond the anisotropy radius.
    AccuracyError
        If the error of a component's accuracy check exceeds
        :math:`10^{-2}` and the component's settings do not set
        ``write_anyway=True``.
    AccuracyError
        If the projection gives a negative squared line-of-sight dispersion.
    ResultError
        If a table of velocity distributions fails the table's own
        consistency check (for binned distributions and Gauss-Hermite
        moments).
    WorkerError
        If a parallel process ends while a table of velocity distributions
        is computed (for binned distributions and Gauss-Hermite moments).

    See Also
    --------
    log_likelihood : The log-likelihood at the model's own weights and scales.
    Likelihood : The JAX function `likelihood` returns.
    Data : The measurements a model is compared with.
    superpose : A superposition of models that share one density.

    Notes
    -----
    Each dispersion measurement :math:`i` is the mean of
    :math:`\sigma_{\rm los}^2` over the measurement's nodes :math:`x_j`,
    weighted by the node weights :math:`q_{ij}` and the surface density
    :math:`\Sigma`,

    .. math::

        \sigma_i^2 = \frac{\sum_j q_{ij}\, \Sigma(x_j)\,
        \sigma_{\rm los}^2(x_j)}{\sum_j q_{ij}\, \Sigma(x_j)}.

    A projected radius has one node, the projected radius itself. An
    aperture has the nodes and weights of `sigma_los_aperture`. On a
    superposition, each component's table is the component's own table on
    the shared density, and the component weights mix the numerator and
    the denominator of :math:`\sigma_i^2` (`mixture`), so each measurement
    is a ratio of two sums linear in the weights. A length scale :math:`l`
    other than the model's own length scale moves each node to
    :math:`x_j = R_j / (L\,l)` and keeps the node weights :math:`q_{ij}`:
    the move is a change of variable to the projected radius of the data,
    in which the aperture is fixed.

    A free component's tables at an anisotropy :math:`\theta` are computed
    at the first call of the `Likelihood` at :math:`\theta`: first the
    component's DF on the component's truncated profile, then the
    component's two projected tables, with the tabulated radii placed for
    :math:`\theta`. Where the DF cannot be built at :math:`\theta`, the
    `Likelihood` returns minus infinity. The gradient in :math:`\theta` is
    the gradient of the reading at the placed radii, with the placed radii
    held fixed. Between two placements of the radii, the value changes by
    no more than the tolerance of the tables. An untruncated component
    cannot be free, because the edge terms of an untruncated component read
    the component's DF.

    A Gauss-Hermite moment (``"h3"`` or ``"h4"``) in an aperture is the
    moment of the mixture's velocity distribution in the aperture. The
    mixture's distribution is the sum of the components' distributions
    (`aperture_losvd`), each weighted by the component weight and the
    component's projected mass, and the moment is taken about the
    mixture's own mean velocity, in units of the mixture's own dispersion.
    A moment in an aperture is differentiable in the weights and does not
    depend on the mass scale. The distributions in apertures are computed
    at the model's own radii, so the length scale stays at the model's own
    length scale. A Gauss-Hermite moment at a projected radius is read on a
    model of one component from a table of the model's moments over
    projected radius, at the projected radius that the length scale moves,
    so a moment at a projected radius is also differentiable in the length
    scale. Neither kind of Gauss-Hermite moment is taken together with a
    free component.

    Each call of `likelihood` computes the projected table of every
    component. For many evaluations on one model, call `likelihood` once
    and call the returned `Likelihood` many times.

    References
    ----------
    Notes Section T:7.4.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> data = pp.Data(
    ...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
    ... )
    >>> like = pp.likelihood(model, data)
    >>> like.weights.shape
    (1,)
    >>> bool(float(like()) > float("-inf"))
    True
    """
    if not isinstance(model, Model):
        raise InputError(
            "The likelihood takes a Model of one component or a superposition made by "
            "superpose; the model given is not a Model."
        )
    settings = model.settings if settings is None else settings
    if len(model.components) == 1:
        views: tuple[Model, ...] = (model,)
    elif is_superposition(model):
        views = tuple(component_view(model, k) for k in range(len(model.components)))
    else:
        raise InputError(
            "The likelihood takes a Model of one component or a superposition made by "
            "superpose; the model given has several components and is not a "
            "superposition."
        )
    datasets = (data,) if isinstance(data, Data) else tuple(data)
    if not datasets or not all(isinstance(d, Data) for d in datasets):
        raise InputError(
            "The data must be one Data object or a non-empty sequence of Data objects."
        )
    systems = {d.unit_system for d in datasets}
    if len(systems) != 1:
        raise InputError(
            f"All the data must be in one unit system; the data given are in the unit "
            f"systems {sorted(systems)}."
        )
    length, velocity, scale = _factors(model, systems.pop())
    free = tuple(int(k) for k in free)
    if len(set(free)) != len(free) or not all(0 <= k < len(views) for k in free):
        raise InputError(
            f"free must list distinct components among the model's {len(views)} "
            f"components, by index from 0; the value given is {free}."
        )
    for k in free:
        refuse_unless_free(views[k], f"Component {k}")
    for k, view in enumerate(views):
        # A component whose accuracy check exceeds 1e-2 raises AccuracyError
        # unless the component's settings set write_anyway=True.
        refuse_unless_asked(f"Component {k}", view.validation, view.settings)
    binned_kinds = sorted(
        {d.observable for d in datasets if d.observable in BINNED + MOMENTS}
    )
    if free and binned_kinds:
        raise InputError(
            "A free component's anisotropy cannot be fitted to binned data or "
            f"Gauss-Hermite moments ({', '.join(binned_kinds)}): the likelihood has "
            "no validated derivative of a binned distribution or of a "
            "Gauss-Hermite moment with respect to the anisotropy."
        )
    members = {k: _Free(views[k], settings) for k in free}
    tables = tuple(
        _table(view, settings, members[k].projected if k in members else None)
        for k, view in enumerate(views)
    )
    # The members of a superposition share one truncated profile, and the
    # truncated profile fixes the innermost tabulated radius, the outermost
    # tabulated radius and the edge radius of every member's projected table;
    # the members' tables differ only in the radii added between the
    # innermost and the outermost tabulated radius.
    extents = {(t.surface.r_min, t.surface.r_max, t.surface.r_t) for t in tables}
    if len(extents) != 1:
        raise InputError(
            f"The projected tables of the model's components must share the "
            "innermost radius, the outermost radius and the edge radius; the "
            f"tables have {sorted(extents)}."
        )
    # Each component's kept ranges and table of velocity distributions are
    # computed once and shared by every Data that reads the component's
    # table, so that no component's table is computed twice.
    shared = _Shared(views, tables, settings)
    blocks = tuple(
        (
            _point_moments(d, views, tables, length, settings, shared)
            if d.r_2d is not None
            else _moments(d, views, tables, length, settings, shared)
        )
        if d.observable in MOMENTS
        else _block(d, tables[0].surface, length, settings)
        for d in datasets
        if d.observable not in BINNED
    )
    binned = _binned_blocks(
        [d for d in datasets if d.observable in BINNED], views, length, settings, shared
    )
    return Likelihood(
        tables=tables,
        blocks=blocks,
        weights=jnp.asarray([float(c.mass) for c in model.components]),
        scale=jnp.asarray(scale),
        length=length,
        velocity=velocity,
        r_min=float(tables[0].surface.r_min),
        r_max=float(tables[0].surface.r_max),
        free=free,
        theta=jnp.asarray(np.array([members[k].theta for k in free]).reshape(-1, 2)),
        members=tuple(members[k] for k in free),
        readings=tuple(_free_reading(members[k]) for k in free),
        binned=binned,
    )


def _noise(data: Data) -> tuple[Array | None, Array | None, float]:
    """Return the Cholesky factor of the covariance matrix of `data` (or None),
    the errors of `data` (or None), and the normalization of the Gaussian,
    half the logarithm of the determinant of 2 pi times the covariance
    matrix."""
    n = data.values.size
    if data.errors is not None:
        errors = np.asarray(data.errors)
        return (
            None,
            jnp.asarray(errors),
            float(np.sum(np.log(errors))) + 0.5 * n * (math.log(2.0 * math.pi)),
        )
    chol = np.linalg.cholesky(np.asarray(data.covariance))
    log_norm = float(np.sum(np.log(np.diag(chol)))) + 0.5 * n * math.log(2.0 * math.pi)
    return jnp.asarray(chol), None, log_norm


def _binned_blocks(
    datasets: list[Data],
    views: tuple[Model, ...],
    length: float,
    settings: Settings,
    shared: _Shared,
) -> tuple[_BinnedBlock, ...]:
    """Return each `Data` of a binned distribution prepared, with each
    component's distribution at the position of the data, computed in NumPy
    before any JAX call."""
    if not datasets:
        return ()
    specs = []
    for d in datasets:
        if d.observable == "losvd_aperture":
            aperture = d.apertures[0]
            where: Any = replace(
                aperture,
                r_off=aperture.r_off / length,
                r_ape=None if aperture.r_ape is None else aperture.r_ape / length,
                side_x=None if aperture.side_x is None else aperture.side_x / length,
                side_y=None if aperture.side_y is None else aperture.side_y / length,
            )
        elif d.observable == "losvd":
            where = float(np.asarray(d.r_2d)[0]) / length
        else:
            where = float(np.asarray(d.r)[0]) / length
        specs.append((d.observable, where))
    with_table = any(kind == "losvd_aperture" for kind, _ in specs)
    reads = []
    for k, view in enumerate(views):
        got = _binned.pieces(
            view,
            settings,
            with_table,
            kept=shared.kept(k),
            table=shared.losvd(k) if with_table else None,
            profile=shared.tables[k].profile,
            projected=shared.tables[k].projected,
        )
        reads.append([_binned.read(kind, w, got, settings) for kind, w in specs])
    blocks = []
    for i, d in enumerate(datasets):
        n = BUCKET_FACTOR * max(r[i][0].a.size for r in reads)
        padded = [_binned.padded(r[i][0], n) for r in reads]
        # The members' Legendre series are padded to one number of
        # coefficients so that the series stack into one array; a Legendre
        # series padded with zero coefficients has unchanged values.
        m = max(p[2].shape[1] for p in padded)
        padded = [
            (p[0], p[1], np.pad(p[2], ((0, 0), (0, m - p[2].shape[1])))) for p in padded
        ]
        chol, errors, log_norm = _noise(d)
        blocks.append(
            _BinnedBlock(
                kind=d.observable,
                v_edges=jnp.asarray(d.v_edges),
                share=jnp.asarray([r[i][1] for r in reads]),
                a=jnp.asarray(np.stack([p[0] for p in padded])),
                b=jnp.asarray(np.stack([p[1] for p in padded])),
                series=jnp.asarray(np.stack([p[2] for p in padded])),
                values=jnp.asarray(d.values),
                chol=chol,
                errors=errors,
                log_norm=log_norm,
            )
        )
    return tuple(blocks)


def log_likelihood(
    model: Model, data: Data | Sequence[Data], settings: Settings | None = None
) -> float:
    r"""Return the Gaussian log-likelihood of a model against data.

    `log_likelihood` returns one number, the value of `likelihood` at the
    model's own component weights and scales. For many evaluations, or for
    gradients, call `likelihood` once and call the returned `Likelihood`.

    Parameters
    ----------
    model : Model
        A model of one component, or a superposition (`superpose`), in the
        unit system of the data.
    data : Data or sequence of Data
        The measurements. The log-likelihoods of several `Data` are summed.
    settings : Settings, optional
        The numerical settings of the observables. Default: ``None``, which
        means the model's own settings.

    Returns
    -------
    log_likelihood : float
        The Gaussian log-likelihood at the model's own component weights
        and scales, summed over the data. The value is minus infinity where
        the nodes of a measurement carry no surface density or where a
        projected radius of the data lies inside the innermost tabulated
        projected radius.

    Raises
    ------
    InputError
        If `model` and `data` cannot be compared, under each condition the
        Raises section of `likelihood` lists for `InputError`.
    AccuracyError
        If the error of a component's accuracy check exceeds
        :math:`10^{-2}` and the component's settings do not set
        ``write_anyway=True``, or if the projection gives a negative squared
        line-of-sight dispersion.
    ResultError
        If a table of velocity distributions fails the table's own
        consistency check (for binned distributions and Gauss-Hermite
        moments).
    WorkerError
        If a parallel process ends while a table of velocity distributions
        is computed (for binned distributions and Gauss-Hermite moments).

    See Also
    --------
    likelihood : The log-likelihood as a JAX function of the parameters.
    Data : The measurements a model is compared with.

    Notes
    -----
    For residuals :math:`r` of the measured values minus the model's values
    and the covariance matrix :math:`C` of the errors (diagonal, with
    :math:`C_{ii} = e_i^2`, for independent errors :math:`e_i`),

    .. math::

        \ln \mathcal{L} = -\frac{1}{2} r^{\rm T} C^{-1} r
        - \frac{1}{2} \ln \det (2 \pi C).

    References
    ----------
    Notes Section T:7.4.

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.inference import log_likelihood
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> data = pp.Data(
    ...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
    ... )
    >>> value = log_likelihood(model, data)
    >>> isinstance(value, float)
    True
    """
    return float(likelihood(model, data, settings)())
