"""The Gauss-Hermite moments as observables of the log-likelihood, on the
isotropic Method II member of the lowered Hernquist pair."""

import math

import jax
import numpy as np
import pytest

from pianistpy import Settings
from pianistpy.inference import Data, likelihood
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    aperture_losvd,
    dispersion_profile,
    gauss_hermite_moments,
    kept_ranges,
    losvd_table,
    projected_dispersion,
)
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition.tests.test_core import _member

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
WEIGHTS = np.array([1.0])
APERTURES = (
    Aperture.circle(0.3 * R_0),
    Aperture.circle(0.2 * R_0, r_off=0.8 * R_0),
)


def _internal(aperture):
    record = aperture.to_dict()
    for name in ("r_off", "r_ape", "side_x", "side_y"):
        if record[name] is not None:
            record[name] = record[name] / R_0
    return Aperture.from_dict(record)


@pytest.fixture(scope="module")
def members():
    """The isotropic Method II member on the lowered Hernquist pair, in
    physical units, and its distribution in each aperture by the stages,
    mirrored to both signs of velocity."""
    parent = _lowered_hernquist()
    out = []
    for beta_c in (0.0,):
        m = _member(parent, beta_c).to(
            "physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0
        )
        p = projected_dispersion(m, dispersion_profile(m, SETTINGS), SETTINGS)
        table = losvd_table(m, kept_ranges(m, p, SETTINGS), p, SETTINGS)
        rows = []
        for a in APERTURES:
            r = aperture_losvd(m, table, p, _internal(a), SETTINGS)
            v = np.concatenate([-r.v_nodes[::-1], r.v_nodes])
            w = np.concatenate([r.v_weights[::-1], r.v_weights])
            d = np.concatenate([r.losvd[0][::-1], r.losvd[0]])
            rows.append((v, w, d / float(w @ d), r.mass[0]))
        out.append((m, rows))
    return out


def _numpy_moments(weights, members, key):
    """The mixture's moment in each aperture by `gauss_hermite_moments` on the
    members' nodes put together, each member's weights scaled by its share."""
    n = np.asarray(weights) / np.sum(weights)
    values = []
    for i in range(len(APERTURES)):
        share = np.array([n[k] * rows[i][3] for k, (_, rows) in enumerate(members)])
        share = share / share.sum()
        v = np.concatenate([rows[i][0] for _, rows in members])
        w = np.concatenate(
            [s * rows[i][1] for s, (_, rows) in zip(share, members, strict=True)]
        )
        d = np.concatenate([rows[i][2] for _, rows in members])
        values.append(gauss_hermite_moments(v, w, d)[key])
    return np.array(values)


def _numpy_log_likelihood(weights, members, data):
    total = 0.0
    for d in data:
        r = d.values - _numpy_moments(weights, members, d.observable)
        total += -0.5 * np.sum((r / d.errors) ** 2) - np.sum(np.log(d.errors))
        total -= 0.5 * r.size * math.log(2.0 * math.pi)
    return total


@pytest.fixture(scope="module")
def data(members):
    h4 = _numpy_moments(WEIGHTS, members, "h4")
    return [
        Data(
            "h4",
            h4 + 0.01 * np.array([1.0, -1.0]),
            apertures=APERTURES,
            errors=[0.01, 0.01],
            unit_system="physical",
        ),
        Data(
            "h3",
            [0.003, -0.002],
            apertures=APERTURES,
            errors=[0.01, 0.01],
            unit_system="physical",
        ),
    ]


def test_the_moments_against_numpy(members, data):
    function = likelihood(members[0][0], data, SETTINGS)
    value = _numpy_log_likelihood(WEIGHTS, members, data)
    assert abs(float(function()) / value - 1.0) < 1e-12


def test_the_scales_of_moment_data(members, data):
    function = likelihood(members[0][0], data, SETTINGS)
    own = np.array([MASS_0, R_0])
    # the mass scale leaves the moments; another length scale is refused
    assert float(function(scale=[3.0 * MASS_0, R_0])) == float(function())
    assert np.asarray(jax.grad(lambda s: function(scale=s))(own))[0] == 0.0
    assert function(scale=[MASS_0, 1.01 * R_0]) == -np.inf
