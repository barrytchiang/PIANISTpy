"""Tests of the integrals over an aperture, on closed forms."""

import numpy as np
import pytest

from pianistpy.exceptions import InputError
from pianistpy.observables._aperture import aperture_mean_square, aperture_nodes
from pianistpy.observables.aperture import Aperture
from pianistpy.validation import GOAL

# The rule of the working code: 100 nodes a panel, 8 a decade where more.
RULE = dict(n_nodes=100, nodes_per_decade=8, inner_tolerance=1e-6)


class _Reading:
    """An exact surface density on [r_min, r_max], with the attributes of a
    table's reading."""

    def __init__(self, function, r_min, r_max, r_t=None):
        self._function = function
        self.r_min, self.r_max, self.r_t = r_min, r_max, r_t
        # The disc inside the first row, for a profile flat there.
        self.inner_disc = float(function(np.array([r_min]))[0]) * r_min**2 / 2.0

    def __call__(self, r):
        r = np.asarray(r, dtype=np.float64)
        return np.where(r > self.r_max, 0.0, self._function(r))


def _flat(r):
    return np.ones_like(r)


@pytest.mark.parametrize(
    "aperture, area",
    [
        (Aperture.circle(0.7), np.pi * 0.7**2),
        (Aperture.circle(0.3, r_off=1.0), np.pi * 0.3**2),
        (Aperture.circle(1.5, r_off=1.0), np.pi * 1.5**2),
        (Aperture.circle(1e-3, r_off=2.0), np.pi * 1e-6),
        (Aperture.rectangle(2.0, 0.5), 1.0),
        (Aperture.rectangle(2.0, 0.5, r_off=0.1, angle=0.3), 1.0),
        (Aperture.rectangle(0.4, 1.2, r_off=2.0, angle=1.1), 0.48),
    ],
)
def test_nodes_integrate_the_area(aperture, area):
    # The radial integral of arc(R) R dR is the aperture's area, wherever the
    # aperture lies against the center (a flat surface density to R = 10); a
    # centerd circle's weights leave out the common 2 pi.
    reading = _Reading(_flat, 1e-9, 10.0)
    _, weights, sigma = aperture_nodes(reading, aperture, **RULE)
    if aperture.shape == "circle" and aperture.r_off == 0.0:
        area /= 2.0 * np.pi
    assert abs(float(np.dot(weights, sigma)) / area - 1.0) < 1e-13


def test_plummer_centerd_aperture():
    # Sigma = 4 (1 + R^2)^-2 and sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2) give
    # over a centerd circle of radius A the mean square
    # (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^-1].
    reading = _Reading(lambda r: 4.0 / (1.0 + r**2) ** 2, 1e-6, 1e3)

    def sigma_los2(r):
        return 3.0 * np.pi / 64.0 / np.sqrt(1.0 + r**2)

    for a in (1e-2, 0.5, 3.0, 100.0):
        exact = (
            np.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        mean = aperture_mean_square(reading, sigma_los2, Aperture.circle(a), **RULE)
        assert abs(mean / exact - 1.0) < GOAL


def test_edge_panels_in_v():
    # On a reading with an edge R_t the panels beyond R_t / 2^(1/2) run in v; a
    # surface density (R_t^2 - R^2)^(1/2) integrates over the whole table to
    # 2 pi R_t^3 / 3 (the centerd weights leave out the 2 pi).
    r_t = 2.0
    reading = _Reading(
        lambda r: np.sqrt(np.maximum(r_t**2 - r**2, 0.0)), 1e-9, r_t, r_t
    )
    _, weights, sigma = aperture_nodes(reading, Aperture.circle(r_t), **RULE)
    assert abs(float(np.dot(weights, sigma)) / (r_t**3 / 3.0) - 1.0) < 1e-13


def test_refusals():
    reading = _Reading(lambda r: r**-1.5, 1e-3, 10.0)
    # A cusp whose disc inside the first row is not negligible in the aperture.
    with pytest.raises(InputError, match="innermost"):
        aperture_mean_square(reading, _flat, Aperture.circle(2e-3), **RULE)
    # An aperture beyond the table.
    with pytest.raises(InputError, match="lies outside the tabulated projected radii"):
        aperture_mean_square(reading, _flat, Aperture.circle(1.0, r_off=20.0), **RULE)
