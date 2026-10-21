"""Tests of the log-likelihood of a model of one component against an
independent computation in NumPy, of the gradient in the scales, and of the
exceptions and the values of minus infinity."""

import math

import jax
import numpy as np
import pytest

from pianistpy import InputError, Settings, read
from pianistpy.df.analytic import analytic_model
from pianistpy.inference import Data, likelihood, log_likelihood
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    aperture_dispersion,
    dispersion_profile,
    projected_dispersion,
)
from pianistpy.particles.tests.test_stages import _lowered_hernquist

# The observables' grids are smaller than the grids of the default settings,
# with the sizes the tests of the stages use, to keep each test short.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
# The physical scales at which the models are presented: the mass scale M_0
# in solar masses, the scale radius r_0 in parsecs, and the concentration 10,
# which makes M_0 the mass inside ten scale radii.
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


def _physical(model):
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


@pytest.fixture(scope="module", params=["plummer", "lowered_hernquist"])
def case(request):
    """Return a model in internal units, the model presented in physical units,
    and two `Data` on the model: dispersions at projected radii and in
    apertures inside the model's table, each value the model's own value
    moved by a few percent, with errors of five percent."""
    if request.param == "plummer":
        base = analytic_model("plummer")
    else:
        base = _lowered_hernquist()
    model = _physical(base)
    projected = projected_dispersion(
        model, dispersion_profile(model, SETTINGS), SETTINGS
    )
    r_max = float(projected.r_2d[-1])
    r_2d = R_0 * np.geomspace(0.05, 0.5 * r_max, 7)
    apertures = (
        Aperture.circle(0.3 * R_0),
        Aperture.circle(0.2 * R_0, r_off=0.8 * R_0),
        Aperture.rectangle(0.4 * R_0, 0.1 * R_0, r_off=1.5 * R_0, angle=0.4),
    )
    v = model.units.factors(model.components[0].unit_profile).physical_v_kms
    shift = 1.0 + 0.03 * np.cos(np.arange(7))
    points = Data(
        "sigma_los",
        v * projected.read(r_2d / R_0)[1][0] * shift,
        r_2d=r_2d,
        errors=0.05 * v * np.ones(7),
        unit_system="physical",
    )
    ape = np.array(
        [aperture_dispersion(projected, _internal(a), SETTINGS)[0] for a in apertures]
    )
    in_apertures = Data(
        "sigma_los_aperture",
        v * ape * shift[:3],
        apertures=apertures,
        covariance=(0.05 * v) ** 2 * (np.eye(3) + 0.3 * (1.0 - np.eye(3))),
        unit_system="physical",
    )
    return base, model, points, in_apertures


def _internal(aperture, length=R_0):
    """Return `aperture`, given in parsecs, with every length divided by
    `length`, which gives the aperture in internal units."""
    record = aperture.to_dict()
    for name in ("r_off", "r_ape", "side_x", "side_y"):
        if record[name] is not None:
            record[name] = record[name] / length
    return Aperture.from_dict(record)


def _numpy_log_likelihood(base, data, mass_0=MASS_0, r_0=R_0):
    """Return the log-likelihood computed in NumPy from the stages' own
    readings: the model presented at the given scales, the model's table
    read at the projected radii of the data in internal units, or the
    model's dispersion in each aperture."""
    model = base.to("physical", concentration=CONCENTRATION, mass_0=mass_0, r_0=r_0)
    v = model.units.factors(model.components[0].unit_profile).physical_v_kms
    projected = projected_dispersion(
        model, dispersion_profile(model, SETTINGS), SETTINGS
    )
    total = 0.0
    for d in data:
        if d.observable == "sigma_los":
            predicted = v * projected.read(d.r_2d / r_0)[1][0]
        else:
            predicted = v * np.array(
                [
                    aperture_dispersion(projected, _internal(a, r_0), SETTINGS)[0]
                    for a in d.apertures
                ]
            )
        residual = d.values - predicted
        if d.errors is not None:
            total += -0.5 * np.sum((residual / d.errors) ** 2) - np.sum(
                np.log(d.errors)
            )
            total -= 0.5 * residual.size * math.log(2.0 * math.pi)
        else:
            _, log_det = np.linalg.slogdet(2.0 * math.pi * d.covariance)
            total += -0.5 * residual @ np.linalg.solve(d.covariance, residual)
            total -= 0.5 * log_det
    return total


def test_the_value_matches_numpy(case):
    base, model, points, in_apertures = case
    for data in ([points], [in_apertures], [points, in_apertures]):
        expected = _numpy_log_likelihood(base, data)
        value = log_likelihood(model, data, SETTINGS)
        assert abs(value / expected - 1.0) < 1e-12


@pytest.mark.parametrize("length", [1.0, 0.8])
def test_the_gradient_in_the_scales_matches_central_differences(case, length):
    # The NumPy log-likelihood lays the aperture nodes anew at each scale, so
    # central differences of the NumPy log-likelihood are an independent
    # reference for the JAX gradient. At a length scale below the model's own
    # length scale, the outermost aperture nodes move beyond the edge of the
    # table, where the readings are zero.
    base, model, points, in_apertures = case
    data = [points, in_apertures]
    function = likelihood(model, data, SETTINGS)
    scale = np.array([MASS_0, length * R_0])
    gradient = np.asarray(jax.grad(lambda s: function(scale=s))(scale))
    for i in range(2):
        h = 1e-5 * scale[i]
        up, down = scale.copy(), scale.copy()
        up[i] += h
        down[i] -= h
        difference = (
            _numpy_log_likelihood(base, data, *up)
            - _numpy_log_likelihood(base, data, *down)
        ) / (2.0 * h)
        assert abs(gradient[i] * scale[i] / (difference * scale[i]) - 1.0) < 1e-6


def test_refusals_are_minus_infinity_inside_the_function(case):
    _, model, points, in_apertures = case
    function = likelihood(model, [points, in_apertures], SETTINGS)
    assert np.isfinite(function())
    assert function(weights=[-1.0]) == -np.inf
    assert function(weights=[0.0]) == -np.inf
    assert function(scale=[MASS_0, -R_0]) == -np.inf
    # A length scale of 1e6 times r_0 moves the innermost projected radius of
    # the data inside the innermost tabulated projected radius.
    assert function(scale=[MASS_0, 1e6 * R_0]) == -np.inf
    gradient = jax.grad(lambda s: function(scale=s))(np.array([MASS_0, R_0]))
    assert np.all(np.isfinite(gradient))
    jitted = jax.jit(lambda s: function(scale=s))
    # The compiled function orders the floating-point operations differently
    # from the function run without compilation, which can change the last
    # bit of the value, so the two values agree to a relative 1e-14.
    assert np.isclose(
        float(jitted(np.array([MASS_0, R_0]))), float(function()), rtol=1e-14, atol=0.0
    )


def test_refusals_at_the_boundary(case):
    base, model, points, _ = case
    with pytest.raises(InputError, match="system"):
        likelihood(base, points, SETTINGS)
    far = Data(
        "sigma_los", [1.0], r_2d=[1e6 * R_0], errors=[1.0], unit_system="physical"
    )
    with pytest.raises(InputError, match="beyond the outermost projected radius"):
        likelihood(model, far, SETTINGS)
    with pytest.raises(InputError, match="exactly one of errors and covariance"):
        Data("sigma_los", [1.0], r_2d=[1.0])
    with pytest.raises(InputError, match="apertures"):
        Data("sigma_los_aperture", [1.0], r_2d=[1.0], errors=[1.0])


def test_data_are_a_pytree_and_a_file(case, tmp_path):
    _, _, points, in_apertures = case
    for data in (points, in_apertures):
        leaves, tree = jax.tree_util.tree_flatten(data)
        back = jax.tree_util.tree_unflatten(tree, leaves)
        assert back.identity == data.identity
        path = tmp_path / f"{data.observable}.h5"
        data.write(path)
        read_back = read(path)
        assert isinstance(read_back, Data)
        assert read_back.identity == data.identity
        assert read_back.apertures == data.apertures
