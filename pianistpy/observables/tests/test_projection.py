"""Tests of the projection and its table against a uniform sphere."""

import numpy as np

from pianistpy import Anisotropy
from pianistpy.observables._projection import (
    certify_projection,
    dispersion_table_reader,
    edge_variable,
    edge_variable_inverse,
    project,
    surface_density_reader,
)
from pianistpy.validation import GOAL

R_T = 2.0
SIGMA_R2 = 0.3


def _density(r):
    return np.ones_like(np.asarray(r, dtype=np.float64))


def _sigma_r2(r):
    return np.full_like(np.asarray(r, dtype=np.float64), SIGMA_R2)


def test_projection_of_a_uniform_isotropic_sphere():
    # Sigma = 2 (R_t^2 - R_2D^2)^(1/2); with beta = 0 and sigma_r^2 constant,
    # sigma_los^2 equals it (paper Eqs. B45, B46)
    r_2d = np.array([1e-3, 0.1, 0.5, 1.0, 1.9])
    surface_density, sigma_los2 = project(
        r_2d, _density, _sigma_r2, R_T, Anisotropy(), 200
    )
    np.testing.assert_allclose(
        surface_density, 2.0 * np.sqrt(R_T**2 - r_2d**2), rtol=1e-12
    )
    np.testing.assert_allclose(sigma_los2, SIGMA_R2, rtol=1e-12)


def test_edge_variable_round_trip():
    r = np.array([1e-3, 0.5, 1.9, 1.999999])
    np.testing.assert_allclose(
        edge_variable_inverse(edge_variable(r, R_T), R_T), r, rtol=1e-13
    )


def test_readers_and_certificate():
    r_2d = np.logspace(-3, np.log10(0.9999 * R_T), 60)

    def rule(radii):
        return project(radii, _density, _sigma_r2, R_T, Anisotropy(), 200)

    surface_density, sigma_los2 = rule(r_2d)
    rows = certify_projection(
        r_2d, surface_density, sigma_los2, R_T, True, rule, 0.01 * GOAL
    )
    r_rows, sigma_rows, los_rows, record = rows
    assert record["end"] == "tolerance"
    read_sigma = surface_density_reader(r_rows, sigma_rows, R_T, True)
    read_los = dispersion_table_reader(r_rows, los_rows)
    r_mid = np.sqrt(r_rows[:-1] * r_rows[1:])
    np.testing.assert_allclose(
        read_sigma(r_mid), 2.0 * np.sqrt(R_T**2 - r_mid**2), rtol=1e-8
    )
    np.testing.assert_allclose(read_los(r_mid), SIGMA_R2, rtol=1e-8)
