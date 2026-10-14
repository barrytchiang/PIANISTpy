"""Tests of an aperture's arc and break radii against a direct angular count."""

import numpy as np
import pytest

from pianistpy.observables import Aperture

N_ANGLES = 400_000


def _counted_arc(aperture, r):
    """The angular measure by membership of N_ANGLES equally spaced points of
    the circle, the aperture in its own frame with the model's center at
    R_off (cos angle, sin angle) for a rectangle and at the origin for a
    circle displaced by R_off along x."""
    phi = (np.arange(N_ANGLES) + 0.5) * (2.0 * np.pi / N_ANGLES)
    out = np.empty(np.shape(r))
    for i, rv in np.ndenumerate(np.asarray(r, dtype=np.float64)):
        if aperture.shape == "circle":
            x, y = rv * np.cos(phi), rv * np.sin(phi)
            inside = np.hypot(x - aperture.r_off, y) <= aperture.r_ape
        else:
            cx = aperture.r_off * np.cos(aperture.angle)
            cy = aperture.r_off * np.sin(aperture.angle)
            x, y = cx + rv * np.cos(phi), cy + rv * np.sin(phi)
            a, b = 0.5 * aperture.side_x, 0.5 * aperture.side_y
            inside = (np.abs(x) <= a) & (np.abs(y) <= b)
        out[i] = 2.0 * np.pi * np.mean(inside)
    return out


CASES = [
    Aperture.circle(1.0),
    Aperture.circle(1.0, r_off=10.0),
    Aperture.circle(1.0, r_off=0.5),
    Aperture.rectangle(1.0, 0.8, angle=0.3),
    Aperture.rectangle(1.0, 0.8, r_off=0.5, angle=0.0),
    Aperture.rectangle(4.0, 0.5, r_off=10.0, angle=0.6),
]


@pytest.mark.parametrize("aperture", CASES, ids=[str(i) for i in range(len(CASES))])
def test_the_arc_is_the_counted_angular_measure(aperture):
    r = np.geomspace(1e-3, 20.0, 160)
    exact = aperture.arc(r)
    counted = _counted_arc(aperture, r)
    # the count resolves 2 pi / N_ANGLES per crossing, at most eight crossings
    assert np.all(np.abs(exact - counted) <= 10.0 * 2.0 * np.pi / N_ANGLES)
    assert exact.shape == r.shape
    assert np.all((exact >= 0.0) & (exact <= 2.0 * np.pi + 1e-15))


def test_a_centerd_circle_is_the_full_circle_inside_and_nothing_outside():
    aperture = Aperture.circle(2.0)
    assert np.array_equal(
        aperture.arc([0.5, 2.0, 2.5]), [2.0 * np.pi, 2.0 * np.pi, 0.0]
    )
    assert aperture.breaks() == [2.0]


def test_a_rectangle_holding_the_center_is_full_inside_its_inscribed_circle():
    aperture = Aperture.rectangle(2.0, 1.0, angle=0.7)
    assert np.allclose(
        aperture.arc([0.0, 0.25, 0.5]), 2.0 * np.pi, rtol=0.0, atol=1e-15
    )
    assert aperture.arc([0.5000001])[0] < 2.0 * np.pi
    assert aperture.arc([np.hypot(1.0, 0.5) + 1e-9])[0] == 0.0


@pytest.mark.parametrize("aperture", CASES, ids=[str(i) for i in range(len(CASES))])
def test_the_arc_is_smooth_between_its_breaks(aperture):
    # the second difference of the arc on a fine grid is small inside every
    # interval between breaks and jumps across a break
    breaks = aperture.breaks()
    assert breaks == sorted(breaks) and all(v > 0.0 for v in breaks)
    edges = [0.0, *breaks, 1.5 * breaks[-1]]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        r = np.linspace(lo, hi, 2001)[1:-1]
        if r.size < 3:
            continue
        arc = aperture.arc(r)
        if np.all(arc == arc[0]):
            continue
        second = np.abs(np.diff(arc, 2)) / np.diff(r)[0] ** 2
        # a smooth function of r: the second difference is bounded by its
        # second derivative, which stays finite away from the breaks
        assert np.max(second[50:-50]) < 1e3 * np.max(np.abs(arc)) / (hi - lo) ** 2


def test_a_circle_offset_breaks_are_the_band_edges_and_the_core_end():
    assert Aperture.circle(1.0, r_off=10.0).breaks() == [9.0, 11.0]
    assert Aperture.circle(1.0, r_off=0.25).breaks() == [0.75, 1.25]


def test_a_rectangle_breaks_are_the_feet_and_the_corners():
    aperture = Aperture.rectangle(2.0, 1.0)
    # the center at the origin: feet at 1 and 0.5, corners at hypot(1, 0.5)
    assert aperture.breaks() == sorted({0.5, 1.0, float(np.hypot(1.0, 0.5))})
    aperture = Aperture.rectangle(2.0, 1.0, r_off=10.0, angle=0.0)
    # the center far along x: only the x edges' feet, and the four corners
    expected = sorted(
        {9.0, 11.0}
        | {float(np.hypot(xe - 10.0, ye)) for xe in (-1.0, 1.0) for ye in (-0.5, 0.5)}
    )
    assert aperture.breaks() == expected
