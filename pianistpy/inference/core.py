"""The log-likelihood of a model against data.

The observables a model predicts are read on the host once, as tables over
projected radius; the likelihood is a JAX function of the parameters that
enter after them: the weights of the components and the scales of the
units. `likelihood` returns that function, prepared; `log_likelihood` is its
value at the model as it stands.

A projected radius in the data's units is ``R = L l x``, with ``x`` in
internal units, and a speed is ``v = V (m / l)^(1/2) u``, with ``u`` in
internal units: ``L`` and ``V`` are the factors of the model's units, and
``m`` and ``l`` the mass and the length scale relative to the model's own
(one there). In the physical system the scales are the mass M_0 in solar
masses and the radius r_0 in parsecs, as `Model.to` takes them.
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
from ..observables.aperture import Aperture
from ..observables.stages import (
    END_TOLERANCE,
    INNER_DISK_TOLERANCE,
    TINY,
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
# Gauss-Hermite moments of the line-of-sight velocity distribution.
MOMENTS = ("h3", "h4")
OBSERVABLES = ("sigma_los", "sigma_los_aperture") + BINNED + MOMENTS


@dataclass(frozen=True)
class Data:
    """Measurements of one observable, with their errors.

    Parameters
    ----------
    observable : str
        One of ``"sigma_los"``, ``"sigma_los_aperture"``, ``"losvd"``,
        ``"losvd_aperture"``, ``"pdf_v"``, ``"h3"`` and ``"h4"``. What was
        measured: the line-of-sight dispersion at projected radii, or
        averaged over apertures (weighted by the surface density); or a
        binned distribution of velocity: the line-of-sight velocity
        distribution at one projected radius or in one aperture, or the
        local speed distribution at one 3D radius; or a Gauss-Hermite
        moment of the line-of-sight velocity distribution at projected radii
        or inside apertures (`gauss_hermite_moments`; dimensionless).
    values : array-like, shape (n,)
        The measured values. For a binned distribution, the share of the
        distribution inside the window ``[v_edges[0], v_edges[-1]]`` that
        falls in each bin.
    r_2d : array-like, shape (n,), optional
        The projected radii, for ``"sigma_los"`` and for moments at radii;
        one radius for ``"losvd"``.
    apertures : sequence of Aperture, optional
        The apertures, for ``"sigma_los_aperture"`` and for moments over
        apertures, their lengths in the units of the data; one aperture for
        ``"losvd_aperture"``.
    errors : array-like, shape (n,), optional
        The standard deviations of independent errors.
    covariance : array-like, shape (n, n), optional
        The covariance of the errors, in place of `errors`.
    unit_system : {"internal", "model", "physical"}
        The units of the data: in the physical system radii in parsecs and
        speeds in kilometers per second. A model is compared in the same
        system.
    v_edges : array-like, shape (n + 1,), optional
        The edges of the bins of a binned distribution, increasing; speeds
        of ``"pdf_v"`` not negative.
    r : array-like, shape (1,), optional
        The 3D radius of ``"pdf_v"``.
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
                f"observable must be one of {OBSERVABLES}; got {self.observable!r}."
            )
        if self.unit_system not in SYSTEMS:
            raise InputError(
                f"unit_system must be one of {SYSTEMS}; got {self.unit_system!r}."
            )
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise InputError("values must be a non-empty one-dimensional array.")
        object.__setattr__(self, "values", values)
        n = values.size
        binned = self.observable in BINNED
        if binned:
            if self.v_edges is None:
                raise InputError(f"{self.observable} data take v_edges.")
            edges = np.asarray(self.v_edges, dtype=np.float64)
            if edges.shape != (n + 1,) or not np.all(np.diff(edges) > 0.0):
                raise InputError(f"v_edges must hold {n + 1} increasing velocities.")
            if self.observable == "pdf_v" and edges[0] < 0.0:
                raise InputError("The speeds of pdf_v data are not negative.")
            object.__setattr__(self, "v_edges", edges)
        elif self.v_edges is not None or self.r is not None:
            raise InputError(f"{self.observable} data take no v_edges and no r.")
        if self.observable == "pdf_v":
            if self.r is None or self.r_2d is not None or self.apertures:
                raise InputError("pdf_v data take r and no r_2d or apertures.")
            r = np.asarray(self.r, dtype=np.float64).reshape(-1)
            if r.shape != (1,) or not r[0] > 0.0:
                raise InputError("r must hold one positive radius.")
            object.__setattr__(self, "r", r)
        elif self.r is not None:
            raise InputError(f"{self.observable} data take no r.")
        elif self.observable in ("sigma_los", "losvd") or (
            self.observable in MOMENTS and self.r_2d is not None
        ):
            m = 1 if binned else n
            if self.r_2d is None or self.apertures:
                raise InputError(f"{self.observable} data take r_2d and no apertures.")
            r_2d = np.asarray(self.r_2d, dtype=np.float64).reshape(-1)
            if r_2d.shape != (m,) or not np.all(r_2d > 0.0):
                raise InputError(f"r_2d must hold {m} positive radii.")
            object.__setattr__(self, "r_2d", r_2d)
        else:
            m = 1 if binned else n
            if self.r_2d is not None or len(self.apertures) != m:
                raise InputError(
                    f"{self.observable} data take {m} apertures"
                    + (
                        f" or {m} projected radii."
                        if self.observable in MOMENTS
                        else " and no r_2d."
                    )
                )
            if not all(isinstance(a, Aperture) for a in self.apertures):
                raise InputError("apertures must be Aperture objects.")
            object.__setattr__(self, "apertures", tuple(self.apertures))
        if (self.errors is None) == (self.covariance is None):
            raise InputError("Give errors or covariance, not both.")
        if self.errors is not None:
            errors = np.asarray(self.errors, dtype=np.float64)
            if errors.shape != (n,) or not np.all(errors > 0.0):
                raise InputError(f"errors must hold {n} positive values.")
            object.__setattr__(self, "errors", errors)
        else:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if covariance.shape != (n, n) or not np.allclose(
                covariance, covariance.T, rtol=0.0, atol=0.0
            ):
                raise InputError(f"covariance must be a symmetric {n} by {n} matrix.")
            object.__setattr__(self, "covariance", covariance)

    @property
    def identity(self) -> str:
        """The identity string of the data: a digest of everything it holds."""
        return identity_string(_data_record(self), {}, __version__)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the data to a file; `pianistpy.read` reads it back."""
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
    """The dictionary form of data, with its arrays as lists."""

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
    # a binned distribution's own fields, only where it has them: the
    # identity of every other Data is as before
    for name in ("v_edges", "r"):
        if getattr(data, name) is not None:
            record[name] = plain(getattr(data, name))
    return record


def _read_data(source: StoredResult | str | os.PathLike) -> Data:
    """Return data from a result file or from a stored result."""
    stored = load_result(source)
    if stored.kind != RESULT_KIND:
        raise ResultError(
            f"The result is of kind {stored.kind!r}, not {RESULT_KIND!r}."
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
    """One component's projected table: its result, read in JAX by
    `ProjectedDispersion.read_traced`, and the host reading of its surface
    density, on which the aperture nodes are laid."""

    projected: ProjectedDispersion
    surface: Any

    def read(self, x: Array) -> tuple[Array, Array]:
        """The surface density and the squared dispersion at internal radii,
        zero beyond the last row, the first rows' cubics continued inside the
        first row."""
        surface, square = self.projected.read_traced(x, jnp)
        return surface[0], square[0]


@jax.custom_vjp
def _not_carried(length: Array) -> Array:
    """Zero, whose gradient in the length scale is NaN: the binned
    distributions hold the length scale at the model's own and carry no
    gradient in it, which a caller then reads as not a number rather than as
    a number."""
    return jnp.zeros_like(length)


def _not_carried_forward(length: Array) -> Any:
    return jnp.zeros_like(length), None


def _not_carried_backward(_: Any, cotangent: Array) -> Any:
    return (jnp.full_like(cotangent, jnp.nan),)


_not_carried.defvjp(_not_carried_forward, _not_carried_backward)


def _square(breaks: Array, coefficients: Array, x: Array, x_end: float) -> Array:
    """`DispersionReading.traced` on a padded table: clipped at zero, zero
    beyond the last row."""
    log_r = jnp.log(jnp.maximum(x, TINY))
    value = spline_value(breaks, coefficients, log_r, jnp)
    value = jnp.where(log_r > x_end, 0.0, value)
    return jnp.maximum(value, 0.0)


@dataclass(frozen=True)
class _Block:
    """One `Data` prepared: the nodes in internal radius at the model's own
    scale, for apertures the weight of each node in each measurement (radii
    are read at their nodes directly), and the Gaussian's terms that do not
    move."""

    x: Array
    q: Array | None
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float
    points: bool


@dataclass(frozen=True)
class _BinnedBlock:
    """One binned distribution prepared: what it is and where, the bins'
    edges in the data's velocity, each component's share of the mixture
    there and its distribution's padded sub-panels at its own theta, and the
    Gaussian's terms that do not move."""

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
    """One `Data` of Gauss-Hermite moments over apertures prepared: for each
    component and aperture, its distribution on its own velocity nodes
    mirrored to both signs (padded with zero weights to one length),
    normalized to unit integral, and its projected mass; the Gaussian's
    terms that do not move."""

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
        """The moment of the mixture in each aperture: its distribution the
        components' weighted by weight and projected mass, its mean and
        dispersion its own, as `gauss_hermite_moments` takes them."""
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
    """One `Data` of Gauss-Hermite moments at projected radii prepared, on a
    model of one component: its moments' table in radius
    (`gauss_hermite_table`), read in JAX at the radii the length scale moves
    (``x = R / (L l)``), and the Gaussian's terms that do not move."""

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

    Built by `likelihood`. Called with the weights of the components and the
    scales of the units, it returns the Gaussian log-likelihood, traceable
    and differentiable in both.

    Attributes
    ----------
    tables : tuple
        The projected table of each component, read in JAX.
    blocks : tuple
        Each `Data`, prepared.
    weights : array, shape (n_components,)
        The model's own weights (its components' masses).
    scale : array, shape (2,)
        The model's own mass and length scales in the data's units.
    length, velocity : float
        The factors of the data's units at the model's own scales.
    r_min, r_max : float
        The innermost and outermost projected radius of the tables, in
        internal units.
    free : tuple of int
        The components whose anisotropy
        (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) is a parameter.
    theta : array, shape (n_free, 2)
        Their own (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`), in internal units.
    members : tuple
        Their tables at any theta, on the host.
    readings : tuple
        Their :math:`\sigma_{\rm los}^2` readings as JAX functions of theta.
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
        r"""Return the log-likelihood.

        Parameters
        ----------
        weights : array-like, shape (n_components,), optional
            Non-negative weights of the components, normalized here; the
            model's own by default.
        scale : array-like, shape (2,), optional
            The mass and the length scale in the data's units; the model's
            own by default.
        theta : array-like, shape (n_free, 2), optional
            The (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) of each free
            component (`likelihood`'s ``free``), in internal units; their own
            by default. An isotropic or constant member's :math:`R_{\rm a}`
            column is held at its own value (infinite), its gradient zero.

        Returns
        -------
        log_likelihood : scalar array
            Minus infinity where the weights leave the simplex, a scale is
            not positive, a projected radius falls inside the tables'
            first row or where the model has no surface density, or a free
            component's model at its theta is refused (its gradient there
            is zero); with Gauss-Hermite data over apertures, where the
            length scale is not the model's own (the members' distributions
            are read on the host at the model's own radii until a traced
            reader of them in radius lands).
        """
        w = self.weights if weights is None else jnp.asarray(weights, dtype=jnp.float64)
        s = self.scale if scale is None else jnp.asarray(scale, dtype=jnp.float64)
        valid = jnp.all(w >= 0.0) & (jnp.sum(w) > 0.0) & jnp.all(s > 0.0)
        # Refused values are replaced before they are used, so that neither the
        # value nor the gradient of the kept branch meets a NaN.
        w = jnp.where(valid, w, self.weights)
        s = jnp.where(valid, s, self.scale)
        squares: dict[int, tuple[Array, Array]] = {}
        if theta is not None and not self.free:
            raise InputError("theta is given, but no component's anisotropy is free.")
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
        speed = self.velocity * jnp.sqrt(ratio[0] / ratio[1])
        total: Array = 0.0
        for block in self.blocks:
            if isinstance(block, _PointMoments):
                # Dimensionless: the mass scale leaves them; the length scale
                # moves the radii they are read at.
                x = block.x * stretch
                inside = jnp.all(x >= block.r_min * (1.0 - END_TOLERANCE))
                valid = valid & inside
                x = jnp.where(inside, x, block.x)
                moment = block.table.read_traced(x, jnp)[block.key]
                total = total - _gaussian(block, block.values - moment)
                continue
            if isinstance(block, _Moments):
                # The moments are dimensionless: the mass scale leaves them,
                # and the length scale is held at the model's own.
                valid = valid & (s[1] == self.scale[1])
                total = total - _gaussian(block, block.values - block.predict(w))
                continue
            x = block.x * stretch
            if block.points:
                inside = jnp.all(x >= self.r_min * (1.0 - END_TOLERANCE))
                valid = valid & inside
                x = jnp.where(inside, x, block.x)
            # A node beyond the last row reads zero: it is read at the last row
            # and zeroed, so that no reader's gradient meets the edge's
            # logarithm beyond the table (a NaN through the where).
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
            residual = block.values - speed * jnp.sqrt(mean_square)
            if block.chol is None:
                z = residual / block.errors
            else:
                z = solve_triangular(block.chol, residual, lower=True)
            total = total - 0.5 * jnp.dot(z, z) - block.log_norm
        if self.binned:
            # The binned distributions hold the length scale at the model's
            # own: another is refused, and its gradient is not carried.
            valid = valid & (s[1] == self.scale[1])
            total = total + _not_carried(s[1])
            speed = self.velocity * jnp.sqrt(ratio[0])
            for bin_block in self.binned:
                a, b, series = bin_block.a, bin_block.b, bin_block.series
                u = bin_block.v_edges / speed
                cumulative = jnp.stack(
                    [
                        _binned.cumulative(a[k], b[k], series[k], jnp.abs(u))
                        for k in range(a.shape[0])
                    ]
                )
                if bin_block.kind != "pdf_v":
                    # an even distribution: the integral from 0 to u, signed
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
    """Minus the Gaussian log-likelihood of residuals: ``r^T C^-1 r / 2`` and
    the normalization."""
    if block.chol is None:
        z = residual / block.errors
    else:
        z = solve_triangular(block.chol, residual, lower=True)
    return 0.5 * jnp.dot(z, z) + block.log_norm


def _internal(aperture: Aperture, length: float) -> Aperture:
    """An aperture in the data's units, in internal units."""
    return replace(
        aperture,
        r_off=aperture.r_off / length,
        r_ape=None if aperture.r_ape is None else aperture.r_ape / length,
        side_x=None if aperture.side_x is None else aperture.side_x / length,
        side_y=None if aperture.side_y is None else aperture.side_y / length,
    )


def _factors(model: Model, unit_system: str) -> tuple[float, float, np.ndarray]:
    """The length and velocity factors of a system at the model's own scales,
    and those scales."""
    units = model.units
    if units.system != unit_system:
        raise InputError(
            f"The data are in the {unit_system} system and the model in the "
            f"{units.system} system; present the model with Model.to."
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
    """Each component's values at the block's nodes carried to its
    measurements: read directly for radii, weighted by the aperture weights
    for apertures."""
    if block.points:
        return values
    assert block.q is not None
    return values @ block.q.T


def _block(data: Data, surface: Any, length: float, settings: Settings) -> _Block:
    """Prepare one `Data`: its nodes, their weights and the Gaussian's terms."""
    n = data.values.size
    if data.observable == "sigma_los":
        x = np.asarray(data.r_2d) / length
        if np.any(x < surface.r_min * (1.0 - END_TOLERANCE)):
            raise InputError(
                f"A projected radius lies inside the innermost row of the model's "
                f"table, {surface.r_min * length:.4e} in the data's units."
            )
        if np.any(x > surface.r_max * (1.0 + END_TOLERANCE)):
            raise InputError(
                f"A projected radius lies beyond the last row of the model's "
                f"table, {surface.r_max * length:.4e} in the data's units, where "
                f"it has no surface density."
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
    """Each component's kept ranges and table of distributions, built once
    for every Data of one likelihood, where the first of them needs it."""

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
    """Prepare one `Data` of Gauss-Hermite moments at projected radii: the
    model's table of distributions once and its moments' table in radius."""
    if len(views) != 1:
        raise InputError(
            "Gauss-Hermite moments at projected radii are read on a model of one "
            "component; a superposition's moments are those of its distribution, "
            "not of its members' moments: give them over "
            "apertures."
        )
    table = gauss_hermite_table(shared.losvd(0))
    x = np.asarray(data.r_2d) / length
    r_min, r_max = float(table.r_2d[0]), float(table.r_2d[-1])
    if np.any(x < r_min * (1.0 - END_TOLERANCE)) or np.any(
        x > r_max * (1.0 + END_TOLERANCE)
    ):
        raise InputError(
            f"A projected radius lies outside the rows of the model's moments, "
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
    """Prepare one `Data` of Gauss-Hermite moments over apertures: each
    component's table of distributions (`kept_ranges`, `losvd_table`, built
    once for every Data, `_Shared`), its distribution in each aperture
    (`aperture_losvd`), mirrored to both signs of velocity."""
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
    """The projected table of a model of one component, read as
    `ProjectedDispersion.read` reads it."""
    if projected is None:
        projected = projected_dispersion(
            model, dispersion_profile(model, settings), settings
        )
    return _Table(
        projected,
        surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        ),
    )


def likelihood(
    model: Model,
    data: Data | Sequence[Data],
    settings: Settings | None = None,
    free: Sequence[int] = (),
) -> Likelihood:
    r"""Return the log-likelihood of a model against data, prepared.

    The projected table of each component is computed here, once; the
    returned function reads them in JAX. A model or data that cannot be
    compared are refused here, with an exception.

    Parameters
    ----------
    model : Model
        A model of one component, or a superposition (`superpose`), in the
        unit system of the data.
    data : Data or sequence of Data
        The measurements; their log-likelihoods are summed.
    settings : Settings, optional
        Numerical settings of the observables; the model's by default.
    free : sequence of int, optional
        The components whose anisotropy
        (:math:`\beta_{\rm c}`, :math:`R_{\rm a}`) is a parameter of the
        likelihood (its ``theta``): each a spatially finite model on a
        truncated profile (a DF inverted on its parent's profile). An
        isotropic or constant member's :math:`\beta_{\rm c}` is free and its
        :math:`R_{\rm a}` column is held at its own value (infinite), with a
        zero gradient: the family has no :math:`R_{\rm a}`.

    Returns
    -------
    likelihood : Likelihood

    Notes
    -----
    Each measurement is the surface-density weighted mean of
    :math:`\sigma_{\rm los}^2` over its nodes, ``sum_j q_ij Sigma(x_j)
    sigma^2(x_j) / sum_j q_ij Sigma(x_j)``: for a projected radius one node,
    for an aperture the nodes and weights of `aperture_dispersion`. On a
    superposition each component's table is its member's own, on the shared
    density, and the two sums are mixed by the weights (`mixture`): the
    measurement is a ratio of two sums linear in the weights. A scale
    of length other than the model's own moves the nodes (``x = R / (L l)``)
    and keeps their weights, a change of variable to the data's radius in
    which the aperture is fixed.

    A free component's tables at a theta are built on the host when the
    function first meets it: its DF on its truncated profile (refused there,
    the function is minus infinity), then the two tables, their rows placed
    at that theta. The gradient in theta is that of the reading on those rows
    (their placement frozen); between placements the value moves within the
    tables' tolerance. An untruncated component is not free: its edge terms
    read its DF.

    A Gauss-Hermite moment (``"h3"``, ``"h4"``) over an aperture is that of
    the mixture's distribution in the aperture, the components' own
    (`aperture_losvd`, each on its velocity nodes, mirrored to both signs)
    weighted by weight and projected mass, with its own mean and dispersion
    as `gauss_hermite_moments` takes them: traced in the weights; the mass
    scale leaves it. Its distributions are read at the model's own radii, so
    the length scale is held there (until a traced reader of each member's
    distribution in radius lands). Moments at projected radii
    are read on a model of one component from its moments' table
    (`gauss_hermite_table`, `read_traced`) at the radii the length scale
    moves, so they are traced in it too. Neither is served with a free
    component: its distributions are read at its own theta.
    """
    if not isinstance(model, Model):
        raise InputError(
            "The likelihood takes a Model of one component or a superposition "
            "(`superpose`)."
        )
    settings = model.settings if settings is None else settings
    if len(model.components) == 1:
        views: tuple[Model, ...] = (model,)
    elif is_superposition(model):
        views = tuple(component_view(model, k) for k in range(len(model.components)))
    else:
        raise InputError(
            "The likelihood takes a model of one component or a superposition "
            "(`superpose`)."
        )
    datasets = (data,) if isinstance(data, Data) else tuple(data)
    if not datasets or not all(isinstance(d, Data) for d in datasets):
        raise InputError("data must be a Data object or a sequence of them.")
    systems = {d.unit_system for d in datasets}
    if len(systems) != 1:
        raise InputError(f"The data are in several unit systems: {sorted(systems)}.")
    length, velocity, scale = _factors(model, systems.pop())
    free = tuple(int(k) for k in free)
    if len(set(free)) != len(free) or not all(0 <= k < len(views) for k in free):
        raise InputError(
            f"free must name distinct components of the {len(views)}; got {free}."
        )
    for k in free:
        refuse_unless_free(views[k], f"Component {k}")
    for k, view in enumerate(views):
        # a component beyond the hard limit is refused unless its settings ask
        refuse_unless_asked(f"Component {k}", view.validation, view.settings)
    binned_kinds = sorted(
        {d.observable for d in datasets if d.observable in BINNED + MOMENTS}
    )
    if free and binned_kinds:
        raise InputError(
            "A free component's anisotropy is not served with binned data or "
            f"Gauss-Hermite moments ({', '.join(binned_kinds)}) "
            "until the observables' VJP of the distributions lands: the "
            "likelihood carries no derivative its record cannot validate."
        )
    members = {k: _Free(views[k], settings) for k in free}
    tables = tuple(
        _table(view, settings, members[k].projected if k in members else None)
        for k, view in enumerate(views)
    )
    # The members share one truncated profile, which fixes a projected table's
    # first row, last row and edge; only the rows added between differ.
    extents = {(t.surface.r_min, t.surface.r_max, t.surface.r_t) for t in tables}
    if len(extents) != 1:
        raise InputError(
            f"The components' projected tables differ in extent: {sorted(extents)}."
        )
    # each component's kept ranges and table of distributions, once for all
    # the Data that read them
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
    """The Gaussian's Cholesky factor or errors, and its normalization."""
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
    """Prepare the binned distributions: each component's distribution for
    each, on the host."""
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
        )
        reads.append([_binned.read(kind, w, got, settings) for kind, w in specs])
    blocks = []
    for i, d in enumerate(datasets):
        n = BUCKET_FACTOR * max(r[i][0].a.size for r in reads)
        padded = [_binned.padded(r[i][0], n) for r in reads]
        # the members' series to one length: a Legendre series padded with
        # zero coefficients is the same series
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
    """Return the Gaussian log-likelihood of a model against data.

    Parameters
    ----------
    model : Model
        The model, in the unit system of the data.
    data : Data or sequence of Data
        The measurements; their log-likelihoods are summed.
    settings : Settings, optional
        Numerical settings of the observables; the model's by default.

    Returns
    -------
    log_likelihood : float

    Notes
    -----
    With residuals ``r`` of the measured values against the model's, and
    errors ``e`` or a covariance ``C``, ``-r^T C^-1 r / 2 - ln det(2 pi C) / 2``.
    The value of `likelihood` at the model's own weights and scales.
    """
    return float(likelihood(model, data, settings)())
