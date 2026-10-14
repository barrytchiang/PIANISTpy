"""A fit to Gauss-Hermite moments carries the length scale in its gradient.

The moments are read across a table's rows by `GaussHermiteTable.read_traced`,
so a log-likelihood of point moments traced in the scales (sigma in the
velocity unit sqrt(G M_0 / r_0), h3 and h4 dimensionless, each at R / r_0) has
a gradient in r_0 as well as in M_0, which matches central differences of the
same log-likelihood read on the host.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy.df.analytic import analytic_model
from pianistpy.observables.stages import (
    dispersion_profile,
    gauss_hermite_table,
    kept_ranges,
    losvd_table,
    projected_dispersion,
)
from pianistpy.observables.tests.test_losvd_stages import SETTINGS
from pianistpy.validation import GOAL

# The scales the moments are read at, in the model's own units.
SCALE = np.array([1.0, 1.0])


@pytest.fixture(scope="module")
def moments():
    """The Gauss-Hermite table of the isotropic Plummer model, and data on it:
    seven radii inside the rows, the model's own moments moved by a few
    percent, with their errors."""
    model = analytic_model("plummer")
    projected = projected_dispersion(
        model, dispersion_profile(model, SETTINGS), SETTINGS
    )
    table = losvd_table(
        model, kept_ranges(model, projected, SETTINGS), projected, SETTINGS
    )
    gh = gauss_hermite_table(table)
    r_2d = np.geomspace(1.5 * gh.r_2d[0], 0.5 * gh.r_2d[-1], 7)
    own = gh.read(r_2d)
    k = np.arange(7)
    values = {
        "sigma": own["sigma"] * (1.0 + 0.03 * np.cos(k)),
        "h3": own["h3"] + 0.01 * np.sin(k),
        "h4": own["h4"] + 0.01 * np.cos(k),
    }
    errors = {
        "sigma": 0.05 * np.ones(7),
        "h3": 0.02 * np.ones(7),
        "h4": 0.02 * np.ones(7),
    }
    return gh, r_2d, values, errors


def _log_likelihood(scale, read, xp, r_2d, values, errors):
    speed = xp.sqrt(scale[0] / scale[1])
    m = read(xp.asarray(r_2d) / scale[1])
    predicted = {"sigma": speed * m["sigma"], "h3": m["h3"], "h4": m["h4"]}
    total = 0.0
    for key, value in values.items():
        z = (value - predicted[key]) / errors[key]
        total = total - 0.5 * xp.sum(z * z) - float(np.sum(np.log(errors[key])))
    return total - 0.5 * 21 * math.log(2.0 * math.pi)


def test_the_gradient_in_the_length_scale_matches_central_differences(moments):
    gh, r_2d, values, errors = moments

    def traced(s):
        return _log_likelihood(
            s, lambda x: gh.read_traced(x, jnp), jnp, r_2d, values, errors
        )

    def host(s):
        return _log_likelihood(s, gh.read, np, r_2d, values, errors)

    # the traced value is the host's
    assert abs(float(jax.jit(traced)(SCALE)) / host(SCALE) - 1.0) < 1e-13
    gradient = np.asarray(jax.jit(jax.grad(traced))(jnp.asarray(SCALE)))
    for i in range(2):
        h = 1e-5 * SCALE[i]
        up, down = SCALE.copy(), SCALE.copy()
        up[i] += h
        down[i] -= h
        difference = (host(up) - host(down)) / (2.0 * h)
        assert difference != 0.0
        assert abs(gradient[i] / difference - 1.0) < GOAL
