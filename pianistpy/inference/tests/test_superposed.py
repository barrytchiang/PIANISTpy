"""The log-likelihood of a superposed model, in its weights."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy import Settings, superpose
from pianistpy.inference import Data, likelihood
from pianistpy.observables._aperture import aperture_nodes
from pianistpy.observables._projection import (
    dispersion_table_reader,
    surface_density_reader,
)
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    INNER_DISC_TOLERANCE,
    dispersion_profile,
    projected_dispersion,
)
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition.tests.test_core import _member

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
WEIGHTS = np.array([0.4, 0.6])
RADII = R_0 * np.geomspace(0.05, 0.5, 6)
APERTURES = (
    Aperture.circle(0.3 * R_0),
    Aperture.circle(0.2 * R_0, r_off=0.8 * R_0),
    Aperture.rectangle(0.4 * R_0, 0.1 * R_0, r_off=0.4 * R_0, angle=0.4),
)


def _physical(model):
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


@pytest.fixture(scope="module")
def members():
    parent = _lowered_hernquist()
    return [_physical(_member(parent, beta_c)) for beta_c in (0.0, -0.2)]


@pytest.fixture(scope="module")
def sums(members):
    """For each component and measurement, the two sums of the mixture, from
    the members' own tables by the stages' readers in NumPy: A = the weighted
    Sigma sigma^2 and B = the weighted Sigma, at the radii, and over each
    aperture on the first member's nodes."""
    readers = []
    for m in members:
        p = projected_dispersion(m, dispersion_profile(m, SETTINGS), SETTINGS)
        readers.append(
            (
                surface_density_reader(
                    p.r_2d,
                    p.surface_density[0],
                    float(m.profile.r_table_max),
                    m.spatially_finite,
                ),
                dispersion_table_reader(p.r_2d, np.asarray(p.sigma_los[0]) ** 2),
            )
        )
    obs = SETTINGS.observables
    nodes = []
    for a in APERTURES:
        record = a.to_dict()
        for name in ("r_off", "r_ape", "side_x", "side_y"):
            if record[name] is not None:
                record[name] = record[name] / R_0
        r, w, _ = aperture_nodes(
            readers[0][0],
            Aperture.from_dict(record),
            int(obs["n_aperture_nodes"]),
            int(obs["aperture_nodes_per_decade"]),
            INNER_DISC_TOLERANCE,
        )
        nodes.append((r, w))
    a_points = np.array([s(RADII / R_0) * d(RADII / R_0) for s, d in readers])
    b_points = np.array([s(RADII / R_0) for s, _ in readers])
    a_ape = np.array([[w @ (s(r) * d(r)) for r, w in nodes] for s, d in readers])
    b_ape = np.array([[w @ s(r) for r, w in nodes] for s, _ in readers])
    v = members[0].units.factors(members[0].components[0].unit_profile)
    return (a_points, b_points), (a_ape, b_ape), float(v.physical_v_kms)


def _predict(w, a, b, v):
    return v * np.sqrt((w @ a) / (w @ b))


@pytest.fixture(scope="module")
def data(sums):
    """Data on the mixture at WEIGHTS, moved by 3 percent cosines: radii with
    errors of 2 percent, apertures with a correlated covariance."""
    (a_p, b_p), (a_a, b_a), v = sums
    sigma_p = _predict(WEIGHTS, a_p, b_p, v)
    sigma_a = _predict(WEIGHTS, a_a, b_a, v)
    points = Data(
        "sigma_los",
        sigma_p * (1.0 + 0.03 * np.cos(np.arange(6))),
        r_2d=RADII,
        errors=0.02 * sigma_p,
        unit_system="physical",
    )
    covariance = np.outer(0.02 * sigma_a, 0.02 * sigma_a) * (
        np.eye(3) + 0.3 * (1.0 - np.eye(3))
    )
    in_apertures = Data(
        "sigma_los_aperture",
        sigma_a * (1.0 + 0.03 * np.sin(np.arange(3))),
        apertures=APERTURES,
        covariance=covariance,
        unit_system="physical",
    )
    return [points, in_apertures]


def _numpy(w, sums, data):
    """The log-likelihood and its gradient in the weights, written out: each
    measurement f = (w.A)/(w.B) with w normalized, df/dw_k = (A_k (w.B) -
    B_k (w.A)) / (w.B)^2 / sum(w), and sigma = v f^(1/2)."""
    w = np.asarray(w, dtype=np.float64)
    total, gradient = 0.0, np.zeros(w.size)
    for (a, b), d in zip(sums[:2], data, strict=True):
        n = w / w.sum()
        f = (n @ a) / (n @ b)
        df = (a * (n @ b) - b * (n @ a)) / (n @ b) ** 2
        df = (df - n @ df) / w.sum()  # through the normalization
        sigma = sums[2] * np.sqrt(f)
        dsigma = sums[2] * 0.5 / np.sqrt(f) * df
        r = d.values - sigma
        if d.errors is not None:
            c_inv = np.diag(1.0 / d.errors**2)
            log_det = 2.0 * np.sum(np.log(d.errors))
        else:
            c_inv = np.linalg.inv(d.covariance)
            log_det = np.linalg.slogdet(d.covariance)[1]
        total += -0.5 * r @ c_inv @ r - 0.5 * log_det
        total -= 0.5 * r.size * math.log(2.0 * math.pi)
        gradient += dsigma @ c_inv @ r
    return total, gradient


def test_one_weight_is_the_member(members, data):
    model = superpose(dict(zip(members, WEIGHTS, strict=True)))
    function = likelihood(model, data, SETTINGS)
    for k, member in enumerate(members):
        alone = float(likelihood(member, data, SETTINGS)())
        w = np.zeros(2)
        w[k] = 1.0
        assert abs(float(function(weights=w)) / alone - 1.0) < 1e-12


def test_the_value_and_gradient_in_the_weights(members, sums, data):
    model = superpose(dict(zip(members, WEIGHTS, strict=True)))
    function = likelihood(model, data, SETTINGS)
    gradient = jax.grad(lambda w: function(weights=w))
    for w in (WEIGHTS, np.array([0.1, 0.9]), np.array([2.0, 1.0])):
        value, analytic = _numpy(w, sums, data)
        assert abs(float(function(weights=w)) / value - 1.0) < 1e-12
        traced = np.asarray(gradient(w))
        scale = np.max(np.abs(analytic))
        assert np.max(np.abs(traced - analytic)) < 1e-10 * scale
        for k in range(2):
            h = 1e-5
            up, down = w.copy(), w.copy()
            up[k] += h
            down[k] -= h
            central = (_numpy(up, sums, data)[0] - _numpy(down, sums, data)[0]) / (
                2.0 * h
            )
            assert abs(traced[k] - central) < 1e-6 * scale


def test_the_weights_are_recovered_from_their_own_data(members, sums):
    # data drawn on the mixture at WEIGHTS with noise of 1 percent (seed 7);
    # a Newton iteration on w = (t, 1 - t) finds t within three of its errors
    # (the Fisher error), and without noise to 1e-8
    (a_p, b_p), (a_a, b_a), v = sums
    model = superpose(dict(zip(members, WEIGHTS, strict=True)))
    exact = _predict(WEIGHTS, a_p, b_p, v)
    rng = np.random.default_rng(7)
    for noise in (0.0, 1.0):
        values = exact * (1.0 + noise * 0.01 * rng.standard_normal(exact.size))
        d = Data(
            "sigma_los",
            values,
            r_2d=RADII,
            errors=0.01 * exact,
            unit_system="physical",
        )
        function = likelihood(model, d, SETTINGS)

        def of_t(t, function=function):
            return function(weights=jnp.stack([t, 1.0 - t]))

        first, second = jax.grad(of_t), jax.grad(jax.grad(of_t))
        t = 0.5
        for _ in range(30):
            step = float(first(t)) / float(second(t))
            t -= step
            if abs(step) < 1e-12:
                break
        error = 1.0 / math.sqrt(-float(second(t)))
        if noise == 0.0:
            assert abs(t - WEIGHTS[0]) < 1e-8
        else:
            assert abs(t - WEIGHTS[0]) < 3.0 * error
