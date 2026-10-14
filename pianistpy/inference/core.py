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

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .._version import __version__
from ..df import Model
from ..exceptions import InputError, ResultError
from ..identity import identity_string
from ..io import StoredResult, load_result, register_reader, write_result
from ..numerics.pytrees import register_result
from ..observables._aperture import aperture_nodes
from ..observables._projection import surface_density_reader
from ..observables.aperture import Aperture
from ..observables.stages import (
    END_TOLERANCE,
    INNER_DISC_TOLERANCE,
    ProjectedDispersion,
    dispersion_profile,
    projected_dispersion,
)
from ..particles.core import component_view, is_superposition
from ..settings import Settings
from ..superposition import mixture
from ..typing import Array, ArrayLike
from ..units.system import SYSTEMS

RESULT_KIND = "data"
OBSERVABLES = ("sigma_los", "sigma_los_aperture")


@dataclass(frozen=True)
class Data:
    """Measurements of one observable, with their errors.

    Parameters
    ----------
    observable : {"sigma_los", "sigma_los_aperture"}
        What was measured: the line-of-sight dispersion at projected radii,
        or averaged over apertures (weighted by the surface density).
    values : array-like, shape (n,)
        The measured values.
    r_2d : array-like, shape (n,), optional
        The projected radii, for ``"sigma_los"``.
    apertures : sequence of Aperture, optional
        The apertures, for ``"sigma_los_aperture"``, their lengths in the
        units of the data.
    errors : array-like, shape (n,), optional
        The standard deviations of independent errors.
    covariance : array-like, shape (n, n), optional
        The covariance of the errors, in place of `errors`.
    unit_system : {"internal", "model", "physical"}
        The units of the data: in the physical system radii in parsecs and
        speeds in kilometres per second. A model is compared in the same
        system.
    """

    observable: str
    values: Array
    r_2d: Array | None = None
    apertures: tuple[Aperture, ...] = ()
    errors: Array | None = None
    covariance: Array | None = None
    unit_system: str = "internal"

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
        if self.observable == "sigma_los":
            if self.r_2d is None or self.apertures:
                raise InputError("sigma_los data take r_2d and no apertures.")
            r_2d = np.asarray(self.r_2d, dtype=np.float64)
            if r_2d.shape != (n,) or not np.all(r_2d > 0.0):
                raise InputError(f"r_2d must hold {n} positive radii.")
            object.__setattr__(self, "r_2d", r_2d)
        else:
            if self.r_2d is not None or len(self.apertures) != n:
                raise InputError(
                    f"sigma_los_aperture data take {n} apertures and no r_2d."
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
        for name in ("r_2d", "errors", "covariance"):
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


register_result(Data, ("values", "r_2d", "errors", "covariance"))


def _data_record(data: Data) -> dict[str, Any]:
    """The dictionary form of data, with its arrays as lists."""

    def plain(value: Any) -> Any:
        return None if value is None else np.asarray(value).tolist()

    return {
        "observable": data.observable,
        "unit_system": data.unit_system,
        "apertures": [a.to_dict() for a in data.apertures],
        "values": plain(data.values),
        "r_2d": plain(data.r_2d),
        "errors": plain(data.errors),
        "covariance": plain(data.covariance),
    }


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


@dataclass(frozen=True)
class _Block:
    """One `Data` prepared: the nodes in internal radius at the model's own
    scale, the weight of each node in each measurement (one node a
    measurement for radii), and the Gaussian's terms that do not move."""

    x: Array
    q: Array
    values: Array
    chol: Array | None
    errors: Array | None
    log_norm: float
    points: bool


@dataclass(frozen=True)
class Likelihood:
    """The log-likelihood of a model against data, as a JAX function.

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
    r_min : float
        The innermost projected radius of the tables, in internal units.
    """

    tables: tuple[_Table, ...]
    blocks: tuple[_Block, ...]
    weights: Array
    scale: Array
    length: float
    velocity: float
    r_min: float

    def __call__(
        self, weights: ArrayLike | None = None, scale: ArrayLike | None = None
    ) -> Array:
        """Return the log-likelihood.

        Parameters
        ----------
        weights : array-like, shape (n_components,), optional
            Non-negative weights of the components, normalized here; the
            model's own by default.
        scale : array-like, shape (2,), optional
            The mass and the length scale in the data's units; the model's
            own by default.

        Returns
        -------
        log_likelihood : scalar array
            Minus infinity where the weights leave the simplex, a scale is
            not positive, or a projected radius falls inside the tables'
            first row or where the model has no surface density.
        """
        w = self.weights if weights is None else jnp.asarray(weights, dtype=jnp.float64)
        s = self.scale if scale is None else jnp.asarray(scale, dtype=jnp.float64)
        valid = jnp.all(w >= 0.0) & (jnp.sum(w) > 0.0) & jnp.all(s > 0.0)
        # Refused values are replaced before they are used, so that neither the
        # value nor the gradient of the kept branch meets a NaN.
        w = jnp.where(valid, w, self.weights)
        s = jnp.where(valid, s, self.scale)
        ratio = s / self.scale
        stretch = 1.0 / ratio[1]
        speed = self.velocity * jnp.sqrt(ratio[0] / ratio[1])
        total: Array = 0.0
        for block in self.blocks:
            x = block.x * stretch
            if block.points:
                inside = jnp.all(x >= self.r_min * (1.0 - END_TOLERANCE))
                valid = valid & inside
                x = jnp.where(inside, x, block.x)
            read = [table.read(x) for table in self.tables]
            sigma = jnp.stack([r[0] for r in read])
            square = jnp.stack([r[1] for r in read])
            numerator = mixture(w, (sigma * square) @ block.q.T, jnp)
            denominator = mixture(w, sigma @ block.q.T, jnp)
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
        return jnp.where(valid, total, -jnp.inf)


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
        q = np.eye(n)
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
                INNER_DISC_TOLERANCE,
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
        q=jnp.asarray(q),
        values=jnp.asarray(data.values),
        chol=None if chol is None else jnp.asarray(chol),
        errors=None if errors is None else jnp.asarray(errors),
        log_norm=log_norm,
        points=points,
    )


def _table(model: Model, settings: Settings) -> _Table:
    """The projected table of a model of one component, read as
    `ProjectedDispersion.read` reads it."""
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
    model: Model, data: Data | Sequence[Data], settings: Settings | None = None
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
    tables = tuple(_table(view, settings) for view in views)
    # The members share one truncated profile, which fixes a projected table's
    # first row, last row and edge; only the rows added between differ.
    extents = {(t.surface.r_min, t.surface.r_max, t.surface.r_t) for t in tables}
    if len(extents) != 1:
        raise InputError(
            f"The components' projected tables differ in extent: {sorted(extents)}."
        )
    blocks = tuple(_block(d, tables[0].surface, length, settings) for d in datasets)
    return Likelihood(
        tables=tables,
        blocks=blocks,
        weights=jnp.asarray([float(c.mass) for c in model.components]),
        scale=jnp.asarray(scale),
        length=length,
        velocity=velocity,
        r_min=float(tables[0].surface.r_min),
    )


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
