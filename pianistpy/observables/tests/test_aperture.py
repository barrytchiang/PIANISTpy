"""Tests of `Aperture`: the attributes that `Aperture.circle` and
`Aperture.rectangle` set, the default offset, the round trip through a
dictionary, and the InputError that an invalid aperture raises."""

import pytest

from pianistpy import Aperture, InputError


def test_circle_holds_its_radius_and_offset():
    """Assert that `Aperture.circle(1.0, r_off=10.0)` has the shape "circle",
    the radius r_ape = 1, the offset r_off = 10, and no side_x or side_y."""
    aperture = Aperture.circle(1.0, r_off=10.0)
    assert (aperture.shape, aperture.r_ape, aperture.r_off) == ("circle", 1.0, 10.0)
    assert aperture.side_x is None and aperture.side_y is None


def test_rectangle_holds_its_sides_offset_and_angle():
    """Assert that `Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3)` has
    the shape "rectangle", the sides side_x = 2 and side_y = 0.5, the offset
    r_off = 10, the angle 0.3 and no r_ape."""
    aperture = Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3)
    assert aperture.shape == "rectangle"
    assert (aperture.side_x, aperture.side_y) == (2.0, 0.5)
    assert (aperture.r_off, aperture.angle) == (10.0, 0.3)
    assert aperture.r_ape is None


def test_default_offset_is_the_center():
    """Assert that a circle and a rectangle built without r_off have the
    offset r_off = 0, at the center of the model."""
    assert Aperture.circle(1.0).r_off == 0.0
    assert Aperture.rectangle(1.0, 1.0).r_off == 0.0


@pytest.mark.parametrize(
    "aperture",
    [Aperture.circle(1.0, r_off=3.0), Aperture.rectangle(2.0, 0.5, 1.0, 0.3)],
)
def test_dictionary_round_trip(aperture):
    """Assert that `Aperture.from_dict` of `Aperture.to_dict` returns an
    aperture equal to the original aperture, for an offset circle and for an
    offset, rotated rectangle."""
    assert Aperture.from_dict(aperture.to_dict()) == aperture


@pytest.mark.parametrize(
    "build",
    [
        lambda: Aperture.circle(0.0),
        lambda: Aperture.circle(-1.0),
        lambda: Aperture.circle(1.0, r_off=-1.0),
        lambda: Aperture.rectangle(0.0, 1.0),
        lambda: Aperture.rectangle(1.0, -1.0),
        lambda: Aperture(shape="ellipse", r_ape=1.0),
        lambda: Aperture(shape="circle", r_ape=1.0, side_x=1.0),
        lambda: Aperture(shape="circle", r_ape=1.0, angle=0.1),
        lambda: Aperture(shape="rectangle", side_x=1.0),
        lambda: Aperture(shape="rectangle", side_x=1.0, side_y=1.0, r_ape=1.0),
    ],
)
def test_illegal_apertures_are_refused(build):
    """Assert that building each of ten invalid apertures raises InputError.

    The ten invalid apertures are: a circle of radius 0; a circle of radius
    -1; a circle of offset -1; a rectangle of side_x = 0; a rectangle of
    side_y = -1; the unknown shape "ellipse"; a circle given side_x; a circle
    given a nonzero angle; a rectangle given no side_y; and a rectangle given
    r_ape.
    """
    with pytest.raises(InputError):
        build()
