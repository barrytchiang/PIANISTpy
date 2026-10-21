"""Tests of `Aperture.arc` and `Aperture.breaks` against a direct angular count.

`Aperture.arc` returns, for each radius r, the angle in radians of the circle of
radius r about the model's center that lies inside the aperture.
`Aperture.breaks` returns the radii at which `Aperture.arc` is not smooth.
"""

import numpy as np
import pytest

from pianistpy.observables import Aperture

N_ANGLES = 400_000


def _counted_arc(aperture, r):
    """Return the angle inside the aperture, counted on N_ANGLES points per circle.

    The N_ANGLES points are equally spaced in angle on the circle of radius r
    about the model's center. The returned angle is 2 pi times the fraction of
    the points inside the aperture. A circular aperture has its center at
    (R_off, 0), with the model's center at the origin. A rectangular aperture
    has its center at the origin and its sides along the axes, with the model's
    center at R_off (cos angle, sin angle).
    """
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
    """Assert that `Aperture.arc` equals the angle of `_counted_arc` within 10
    times 2 pi / N_ANGLES, at 160 radii from :math:`10^{-3}` to 20.

    The test also asserts that `arc` returns the shape of its input and values
    between 0 and 2 pi.
    """
    r = np.geomspace(1e-3, 20.0, 160)
    exact = aperture.arc(r)
    counted = _counted_arc(aperture, r)
    # The counted angle errs by at most 2 pi / N_ANGLES at each crossing of the
    # circle with the aperture's edge, and a circle crosses a rectangle's edge
    # at most eight times.
    assert np.all(np.abs(exact - counted) <= 10.0 * 2.0 * np.pi / N_ANGLES)
    assert exact.shape == r.shape
    assert np.all((exact >= 0.0) & (exact <= 2.0 * np.pi + 1e-15))


def test_a_centered_circle_is_the_full_circle_inside_and_nothing_outside():
    """Assert that a centered circle of radius 2 has an arc of 2 pi at radii 0.5
    and 2, an arc of zero at radius 2.5, and the single break 2."""
    aperture = Aperture.circle(2.0)
    assert np.array_equal(
        aperture.arc([0.5, 2.0, 2.5]), [2.0 * np.pi, 2.0 * np.pi, 0.0]
    )
    assert aperture.breaks() == [2.0]


def test_a_rectangle_holding_the_center_is_full_inside_its_inscribed_circle():
    """Assert the arc of a 2 by 1 rectangle centered on the model.

    The arc is 2 pi to ``atol=1e-15`` at radii 0, 0.25 and 0.5, the radius of the
    rectangle's inscribed circle. The arc is below 2 pi at radius 0.5000001. The
    arc is zero at the corner distance hypot(1, 0.5) plus :math:`10^{-9}`.
    """
    aperture = Aperture.rectangle(2.0, 1.0, angle=0.7)
    assert np.allclose(
        aperture.arc([0.0, 0.25, 0.5]), 2.0 * np.pi, rtol=0.0, atol=1e-15
    )
    assert aperture.arc([0.5000001])[0] < 2.0 * np.pi
    assert aperture.arc([np.hypot(1.0, 0.5) + 1e-9])[0] == 0.0


@pytest.mark.parametrize("aperture", CASES, ids=[str(i) for i in range(len(CASES))])
def test_the_arc_is_smooth_between_its_breaks(aperture):
    """Assert that the breaks are positive and increasing, and that `arc` is
    smooth between consecutive breaks.

    The intervals run between consecutive values of zero, the breaks, and 1.5
    times the last break. On each interval where the arc is not constant, the
    test evaluates the arc at 1999 interior points. Away from the 50 points
    nearest each end of the interval, the scaled second difference
    ``abs(diff(arc, 2)) / dr**2`` stays below
    ``1e3 * max(abs(arc)) / (hi - lo)**2``.
    """
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
        # The scaled second difference approximates the second derivative of
        # the arc, which is finite away from the breaks.
        assert np.max(second[50:-50]) < 1e3 * np.max(np.abs(arc)) / (hi - lo) ** 2


def test_a_circle_offset_breaks_are_the_band_edges_and_the_core_end():
    """Assert the breaks of circles of radius 1 offset by R_off.

    For R_off = 10, the breaks are R_off - 1 = 9 and R_off + 1 = 11, where the
    band of circles that cross the aperture begins and ends. For R_off = 0.25,
    the breaks are 1 - R_off = 0.75, where the core of circles inside the
    aperture ends, and 1 + R_off = 1.25.
    """
    assert Aperture.circle(1.0, r_off=10.0).breaks() == [9.0, 11.0]
    assert Aperture.circle(1.0, r_off=0.25).breaks() == [0.75, 1.25]


def test_a_rectangle_breaks_are_the_feet_and_the_corners():
    """Assert the breaks of a 2 by 1 rectangle.

    A break lies at the distance from the model's center to an edge line where
    the nearest point of the line lies on the edge, and at the distance to each
    corner.
    """
    aperture = Aperture.rectangle(2.0, 1.0)
    # With the model's center at the rectangle's center, the edge lines lie at
    # the distances 1 and 0.5, and the corners lie at the distance hypot(1, 0.5).
    assert aperture.breaks() == sorted({0.5, 1.0, float(np.hypot(1.0, 0.5))})
    aperture = Aperture.rectangle(2.0, 1.0, r_off=10.0, angle=0.0)
    # With the model's center at x = 10, the edges at x = -1 and x = 1 give the
    # breaks 9 and 11. The nearest points of the lines y = -0.5 and y = 0.5 lie
    # at x = 10, outside the edges, so the edges at y = -0.5 and y = 0.5 give no
    # break. Each of the four corners gives one break.
    expected = sorted(
        {9.0, 11.0}
        | {float(np.hypot(xe - 10.0, ye)) for xe in (-1.0, 1.0) for ye in (-0.5, 0.5)}
    )
    assert aperture.breaks() == expected
