"""Tests of Gauss-Hermite moments at projected radii in the log-likelihood,
differentiable in the length scale, and of the exception for moments at
projected radii on a superposition."""

import math

import jax
import numpy as np
import pytest

from pianistpy import InputError, Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.inference import Data, likelihood
from pianistpy.observables.stages import (
    dispersion_profile,
    gauss_hermite_table,
    kept_ranges,
    losvd_table,
    projected_dispersion,
)

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
RADII = R_0 * np.array([0.1, 0.3, 1.0, 3.0])


@pytest.fixture(scope="module")
def case():
    """Return the isotropic Plummer model in physical units, the table of the
    model's moments computed by the stages, and h4 data at four projected
    radii, each value the model's own h4 moved by at most 0.002."""
    model = analytic_model("plummer").to(
        "physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0
    )
    p = projected_dispersion(model, dispersion_profile(model, SETTINGS), SETTINGS)
    table = gauss_hermite_table(
        losvd_table(model, kept_ranges(model, p, SETTINGS), p, SETTINGS)
    )
    h4 = table.read(RADII / R_0)["h4"]
    data = Data(
        "h4",
        h4 + 0.002 * np.array([1.0, -1.0, 0.5, -0.5]),
        r_2d=RADII,
        errors=np.full(4, 0.003),
        unit_system="physical",
    )
    return model, table, data


def _numpy(table, data, r_0):
    """Return the log-likelihood computed in NumPy from the table's reading
    at R / r_0, for the scale radius `r_0`."""
    r = data.values - table.read(data.r_2d / r_0)["h4"]
    return (
        -0.5 * np.sum((r / data.errors) ** 2)
        - np.sum(np.log(data.errors))
        - 0.5 * r.size * math.log(2.0 * math.pi)
    )


def test_the_value_and_the_length_scale_gradient(case):
    model, table, data = case
    function = likelihood(model, data, SETTINGS)
    assert abs(float(function()) / _numpy(table, data, R_0) - 1.0) < 1e-12
    scale = np.array([MASS_0, R_0])
    gradient = np.asarray(jax.grad(lambda s: function(scale=s))(scale))
    # The moments are dimensionless, so the gradient in the mass scale is
    # zero.
    assert gradient[0] == 0.0

    # The gradient in r_0 is small compared with the log-likelihood, so the
    # central difference is judged against the central difference's own
    # error estimate: the roundoff eps |f| / h plus the truncation error,
    # estimated as the change of the central difference from step h to step
    # h / 3. The step h = 3e-4 r_0 balances the roundoff and the truncation
    # error.
    def central(h):
        return (_numpy(table, data, R_0 + h) - _numpy(table, data, R_0 - h)) / (2 * h)

    h = 3e-4 * R_0
    c, c3 = central(h), central(h / 3)
    value = abs(_numpy(table, data, R_0))
    estimate = (np.finfo(float).eps * value / h + abs(c - c3)) / abs(c)
    assert estimate < 1e-2
    assert abs(gradient[1] / c - 1.0) < estimate
    # A length scale of 1e6 times r_0 moves a projected radius of the data
    # inside the innermost radius of the table of moments, which gives minus
    # infinity.
    assert function(scale=[MASS_0, 1e6 * R_0]) == -np.inf


def test_a_superposition_takes_moments_over_apertures():
    from pianistpy import superpose
    from pianistpy.particles.tests.test_stages import _lowered_hernquist
    from pianistpy.superposition.tests.test_core import _member

    parent = _lowered_hernquist()
    model = superpose({_member(parent, 0.0): 0.5, _member(parent, -0.2): 0.5})
    data = Data("h4", [0.0], r_2d=[0.5], errors=[0.01])
    with pytest.raises(InputError, match="given over apertures"):
        likelihood(model, data, SETTINGS)
