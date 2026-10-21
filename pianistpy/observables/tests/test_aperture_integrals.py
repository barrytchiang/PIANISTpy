"""Tests of `aperture_nodes` and `aperture_mean_square` against closed-form
integrals over an aperture."""

import numpy as np
import pytest

from pianistpy.exceptions import AccuracyError, InputError
from pianistpy.observables._aperture import aperture_mean_square, aperture_nodes
from pianistpy.observables.aperture import Aperture
from pianistpy.validation import TIER_BOUNDS

# RULE is the default aperture quadrature of the package. Each panel takes at
# least 100 Gauss-Legendre nodes (n_aperture_nodes). A panel takes 8 nodes per
# decade of its span in ln R where 8 per decade exceeds 100 nodes
# (aperture_nodes_per_decade). The estimated share of the aperture's weight
# inside the first row may be at most 1e-6 (INNER_DISK_TOLERANCE).
RULE = dict(n_nodes=100, nodes_per_decade=8, inner_tolerance=1e-6)


class _Reading:
    """An exact surface density on [r_min, r_max], with the attributes of the
    surface density reader of a projected table.

    The reading returns zero beyond r_max.
    """

    def __init__(self, function, r_min, r_max, r_t=None):
        self._function = function
        self.r_min, self.r_max, self.r_t = r_min, r_max, r_t
        # The integral of Sigma R dR from zero to r_min, for a surface density
        # constant inside r_min.
        self.inner_disk = float(function(np.array([r_min]))[0]) * r_min**2 / 2.0
        self.inner_note = ""

    def __call__(self, r):
        r = np.asarray(r, dtype=np.float64)
        return np.where(r > self.r_max, 0.0, self._function(r))


def _flat(r):
    """Return a unit surface density, with the shape of `r`."""
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
    """Assert that the nodes integrate a unit surface density to the aperture's
    area, to a relative difference below ``1e-13``.

    The integral of arc(R) R dR over a unit surface density out to R = 10 equals
    the aperture's area, wherever the aperture lies relative to the model's
    center. The weights of a centered circle leave out the factor 2 pi common to
    the nodes, and the test divides the centered circle's area by 2 pi.
    """
    reading = _Reading(_flat, 1e-9, 10.0)
    _, weights, sigma = aperture_nodes(reading, aperture, **RULE)
    if aperture.shape == "circle" and aperture.r_off == 0.0:
        area /= 2.0 * np.pi
    assert abs(float(np.dot(weights, sigma)) / area - 1.0) < 1e-13


def test_plummer_centered_aperture():
    r"""Assert that `aperture_mean_square` gives the closed form of the Plummer
    sphere over centered circles, to a relative difference below :math:`10^{-6}`
    (TIER_BOUNDS[0]).

    The Plummer sphere in internal units has :math:`\Sigma = 4 (1 + R^2)^{-2}` and
    :math:`\sigma_{\rm los}^2 = (3 \pi / 64) (1 + R^2)^{-1/2}`. Over a centered
    circle of radius :math:`A`, the mean of :math:`\sigma_{\rm los}^2` weighted by
    :math:`\Sigma` is :math:`(\pi / 32) [1 - (1 + A^2)^{-3/2}] / [1 - (1 + A^2)^{-1}]`.
    The test takes :math:`A` in (0.01, 0.5, 3, 100).
    """
    reading = _Reading(lambda r: 4.0 / (1.0 + r**2) ** 2, 1e-6, 1e3)

    def sigma_los2(r):
        return 3.0 * np.pi / 64.0 / np.sqrt(1.0 + r**2)

    for a in (1e-2, 0.5, 3.0, 100.0):
        exact = (
            np.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        mean = aperture_mean_square(reading, sigma_los2, Aperture.circle(a), **RULE)
        assert abs(mean / exact - 1.0) < TIER_BOUNDS[0]


def test_edge_panels_in_v():
    r"""Assert that the nodes of a centered circle integrate a surface density
    with an edge to the closed form, to a relative difference below ``1e-13``.

    The reading has the edge :math:`R_{\rm t} = 2`. The panels beyond
    :math:`R_{\rm t} / 2^{1/2}` are taken in :math:`v = (R_{\rm t}^2 - R^2)^{1/2}`.
    The integral of :math:`\Sigma R \, dR` for
    :math:`\Sigma = (R_{\rm t}^2 - R^2)^{1/2}` over a centered circle of radius
    :math:`R_{\rm t}` is :math:`R_{\rm t}^3 / 3`, without the factor 2 pi that the
    weights of a centered circle leave out.
    """
    r_t = 2.0
    reading = _Reading(
        lambda r: np.sqrt(np.maximum(r_t**2 - r**2, 0.0)), 1e-9, r_t, r_t
    )
    _, weights, sigma = aperture_nodes(reading, Aperture.circle(r_t), **RULE)
    assert abs(float(np.dot(weights, sigma)) / (r_t**3 / 3.0) - 1.0) < 1e-13


def test_refusals():
    """Assert that `aperture_mean_square` raises InputError for a large inner
    disk and for an aperture beyond the table."""
    reading = _Reading(lambda r: r**-1.5, 1e-3, 10.0)
    # A circle of radius 2e-3 on the surface density R^-1.5, whose first row lies
    # at 1e-3, has more than 1e-6 of its estimated weight inside the first row.
    with pytest.raises(InputError, match="innermost"):
        aperture_mean_square(reading, _flat, Aperture.circle(2e-3), **RULE)
    # A circle of radius 1 offset by 20 lies beyond the last row at 10 and
    # encloses no tabulated surface density.
    with pytest.raises(InputError, match="lies outside the tabulated projected radii"):
        aperture_mean_square(reading, _flat, Aperture.circle(1.0, r_off=20.0), **RULE)


class _SteepReading(_Reading):
    """A flat surface density whose disk inside the first row cannot be
    estimated, as on a table whose inner slope is 2 or more."""

    @property
    def inner_disk(self):
        raise AccuracyError("The inner slope is not below 2.")

    @inner_disk.setter
    def inner_disk(self, value):
        pass


def test_a_rectangle_clear_of_the_center_is_served_on_a_steep_table():
    # The rectangle's arc is zero at the first row, so the integral of a flat
    # surface density over the rectangle is the rectangle's area, and the
    # estimate of the disk inside the first row is not read.
    reading = _SteepReading(_flat, 1e-3, 10.0)
    clear = Aperture.rectangle(0.4, 0.2, r_off=1.0, angle=0.3)
    _, weights, sigma = aperture_nodes(reading, clear, **RULE)
    assert abs(float(np.dot(weights, sigma)) / (0.4 * 0.2) - 1.0) < 1e-12
    # A rectangle that contains the center reads the disk's estimate, which
    # raises AccuracyError on such a table.
    with pytest.raises(AccuracyError, match="not below 2"):
        aperture_nodes(reading, Aperture.rectangle(0.4, 0.2), **RULE)


def test_a_rectangle_clear_of_the_center_reads_a_slope_of_2_5():
    # On the package's own reading of a surface density R^-2.5, whose disk
    # inside the first row is not finite, a rectangle clear of the center
    # returns the integral of the reading over the rectangle.
    from scipy import integrate

    from pianistpy.observables._projection import surface_density_reader

    r_rows = np.logspace(-3, 0.5, 60)
    steep = surface_density_reader(r_rows, r_rows**-2.5, 10.0, False)
    clear = Aperture.rectangle(0.4, 0.2, r_off=1.0)
    _, weights, sigma = aperture_nodes(steep, clear, **RULE)
    value, _ = integrate.dblquad(
        lambda y, x: float(steep(np.array([np.hypot(x, y)]))[0]),
        0.8,
        1.2,
        -0.1,
        0.1,
        epsabs=0.0,
        epsrel=1e-11,
    )
    assert abs(float(np.dot(weights, sigma)) / value - 1.0) < 1e-8
    with pytest.raises(AccuracyError, match="is not below 2"):
        aperture_nodes(steep, Aperture.rectangle(0.4, 0.2), **RULE)
