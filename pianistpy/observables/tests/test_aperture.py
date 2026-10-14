"""Tests of the aperture."""

import pytest

from pianistpy import Aperture, InputError


def test_circle_holds_its_radius_and_offset():
    aperture = Aperture.circle(1.0, r_off=10.0)
    assert (aperture.shape, aperture.r_ape, aperture.r_off) == ("circle", 1.0, 10.0)
    assert aperture.side_x is None and aperture.side_y is None


def test_rectangle_holds_its_sides_offset_and_angle():
    aperture = Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3)
    assert aperture.shape == "rectangle"
    assert (aperture.side_x, aperture.side_y) == (2.0, 0.5)
    assert (aperture.r_off, aperture.angle) == (10.0, 0.3)
    assert aperture.r_ape is None


def test_default_offset_is_the_center():
    assert Aperture.circle(1.0).r_off == 0.0
    assert Aperture.rectangle(1.0, 1.0).r_off == 0.0


@pytest.mark.parametrize(
    "aperture",
    [Aperture.circle(1.0, r_off=3.0), Aperture.rectangle(2.0, 0.5, 1.0, 0.3)],
)
def test_dictionary_round_trip(aperture):
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
    with pytest.raises(InputError):
        build()
