"""Apertures on the plane of the sky.

An aperture enters the observables through two closed forms of its geometry:
the angular measure ``arc(r)`` of the circle of projected radius `r` that
lies inside it, which weights the model's circular symmetry over the
aperture, and the radii ``breaks()`` at which that measure is not smooth, so
that a quadrature in `r` is laid between them and no tolerance enters.
"""

__all__ = ["Aperture"]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import InputError
from ..typing import Array


@dataclass(frozen=True)
class Aperture:
    r"""An aperture on the plane of the sky, circular or rectangular.

    Build one with `Aperture.circle` or `Aperture.rectangle`. Lengths are
    in the unit system of the model (or table) it is used with, as the
    observables' radii are.

    Parameters
    ----------
    shape : {"circle", "rectangle"}
        Shape of the aperture.
    r_off : float
        Projected distance :math:`R_{\rm off}` of the center of the aperture
        from the center of the model.
    r_ape : float, optional
        Radius :math:`R_{\rm ape}` of a circle.
    side_x, side_y : float, optional
        Full sides of a rectangle.
    angle : float, optional
        For a rectangle, the angle between its x side and the direction from
        its center to the center of the model, in radians.

    Examples
    --------
    >>> Aperture.circle(1.0, r_off=10.0).shape
    'circle'
    >>> Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3).side_y
    0.5
    """

    shape: str
    r_off: float = 0.0
    r_ape: float | None = None
    side_x: float | None = None
    side_y: float | None = None
    angle: float = 0.0

    def __post_init__(self) -> None:
        if self.shape not in ("circle", "rectangle"):
            raise InputError(
                f"shape must be 'circle' or 'rectangle'; got {self.shape!r}."
            )
        if not self.r_off >= 0.0:
            raise InputError(f"r_off must not be negative; got {self.r_off}.")
        circle = self.shape == "circle"
        used = ("r_ape",) if circle else ("side_x", "side_y")
        unused = ("side_x", "side_y") if circle else ("r_ape",)
        for name in used:
            value = getattr(self, name)
            if value is None or not value > 0.0:
                raise InputError(
                    f"{name} must be positive for a {self.shape}; got {value}."
                )
        for name in unused:
            if getattr(self, name) is not None:
                raise InputError(f"{name} is not used by a {self.shape}.")
        if circle and self.angle != 0.0:
            raise InputError("angle is not used by a circle.")

    @classmethod
    def circle(cls, r_ape: float, r_off: float = 0.0) -> "Aperture":
        """Return a circular aperture of radius `r_ape` at offset `r_off`."""
        return cls(shape="circle", r_off=r_off, r_ape=r_ape)

    @classmethod
    def rectangle(
        cls, side_x: float, side_y: float, r_off: float = 0.0, angle: float = 0.0
    ) -> "Aperture":
        """Return a rectangular aperture of full sides `side_x` by `side_y`.

        Its center is at offset `r_off`, and `angle` is the angle between its
        x side and the direction from its center to the center of the model.
        """
        return cls(
            shape="rectangle", r_off=r_off, side_x=side_x, side_y=side_y, angle=angle
        )

    def arc(self, r: Any) -> Array:
        r"""Return the angular measure of the circle of radius `r` inside the
        aperture, in radians.

        For a circle it is the closed form of the overlap of two circles:
        :math:`2\pi` where the whole circle is inside, zero where it is
        outside, and the band
        :math:`4 \arcsin\sqrt{x/2}` with
        :math:`x = [R_{\rm ape}^2 - (r - R_{\rm off})^2] / (2 r R_{\rm off})`
        between. For a rectangle, each of the four edges is a half-plane
        constraint on the angle that becomes active at angles in closed form;
        those angles cut the circle into intervals on which membership is
        constant, and the measure is the length of the intervals whose
        midpoint lies inside. One code path serves a rectangle that contains
        the center of the model, touches it, or is clear of it.

        Parameters
        ----------
        r : array-like
            Projected radii, not negative.

        Returns
        -------
        arc : array, the shape of `r`
        """
        r = np.asarray(r, dtype=np.float64)
        if self.shape == "circle":
            return _arc_circle(r, self._set("r_ape"), float(self.r_off))
        return _arc_rectangle(
            r,
            self._set("side_x"),
            self._set("side_y"),
            float(self.r_off),
            float(self.angle),
        )

    def _set(self, name: str) -> float:
        """A field the aperture's shape requires, as a float."""
        value = getattr(self, name)
        if value is None:
            raise InputError(f"A {self.shape} aperture needs {name}.")
        return float(value)

    def breaks(self) -> list[float]:
        """Return the radii at which `arc` is not smooth, increasing.

        For a circle: the edges of the band and, where the aperture contains
        the center of the model, the radius at which the full circle ends.
        For a rectangle: the perpendicular distances from the center of the
        model to the four edge lines, kept where the foot of the perpendicular
        lies on the edge, and the distances to the four corners; at most
        eight radii, between two of which `arc` is smooth.
        """
        if self.shape == "circle":
            values = _breaks_circle(self._set("r_ape"), float(self.r_off))
        else:
            values = _breaks_rectangle(
                self._set("side_x"),
                self._set("side_y"),
                float(self.r_off),
                float(self.angle),
            )
        return sorted(set(values))

    def to_dict(self) -> dict[str, Any]:
        """Return the aperture as a plain dictionary."""
        return {
            "shape": self.shape,
            "r_off": self.r_off,
            "r_ape": self.r_ape,
            "side_x": self.side_x,
            "side_y": self.side_y,
            "angle": self.angle,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Aperture":
        """Build the aperture from its dictionary form."""
        return cls(
            shape=str(record["shape"]),
            r_off=float(record.get("r_off", 0.0)),
            r_ape=record.get("r_ape"),
            side_x=record.get("side_x"),
            side_y=record.get("side_y"),
            angle=float(record.get("angle", 0.0)),
        )


# The closed forms, as the working observables script has them, operation
# for operation.


def _arc_circle(r: np.ndarray, r_ape: float, r_off: float) -> np.ndarray:
    if r_off <= 0.0:
        return np.where(r <= r_ape, 2.0 * np.pi, 0.0)
    lo, hi = abs(r_off - r_ape), r_off + r_ape
    x = (
        (r_ape - (r - r_off))
        * (r_ape + (r - r_off))
        / (2.0 * np.maximum(r, 1e-300) * r_off)
    )
    band = 4.0 * np.arcsin(np.sqrt(np.clip(0.5 * x, 0.0, 1.0)))
    return np.where(
        r <= r_ape - r_off, 2.0 * np.pi, np.where((r >= lo) & (r <= hi), band, 0.0)
    )


def _breaks_circle(r_ape: float, r_off: float) -> list[float]:
    if r_off <= 0.0:
        return [float(r_ape)]
    return [v for v in (abs(r_off - r_ape), r_ape - r_off, r_off + r_ape) if v > 0.0]


def _arc_rectangle(
    r: np.ndarray, side_x: float, side_y: float, r_off: float, angle: float
) -> np.ndarray:
    a, b = 0.5 * side_x, 0.5 * side_y
    cx, cy = r_off * np.cos(angle), r_off * np.sin(angle)
    out = np.empty(np.shape(r), dtype=np.float64)
    for i, rv in np.ndenumerate(r):
        if rv <= 0.0:
            out[i] = 2.0 * np.pi if (-a <= cx <= a and -b <= cy <= b) else 0.0
            continue
        ang = [0.0, 2.0 * np.pi]
        for val, fn in (
            (-a - cx, "cos"),
            (a - cx, "cos"),
            (-b - cy, "sin"),
            (b - cy, "sin"),
        ):
            u = val / rv
            if -1.0 <= u <= 1.0:
                if fn == "cos":
                    t0 = float(np.arccos(u))
                    ang += [t0, 2.0 * np.pi - t0]
                else:
                    t0 = float(np.arcsin(u))
                    ang += [t0 % (2.0 * np.pi), (np.pi - t0) % (2.0 * np.pi)]
        angles = np.unique(np.clip(np.asarray(ang, dtype=np.float64), 0.0, 2.0 * np.pi))
        mid = 0.5 * (angles[:-1] + angles[1:])
        x, y = cx + rv * np.cos(mid), cy + rv * np.sin(mid)
        inside = (x >= -a) & (x <= a) & (y >= -b) & (y <= b)
        out[i] = float(np.sum(np.diff(angles)[inside]))
    return out


def _breaks_rectangle(
    side_x: float, side_y: float, r_off: float, angle: float
) -> list[float]:
    a, b = 0.5 * side_x, 0.5 * side_y
    cx, cy = r_off * np.cos(angle), r_off * np.sin(angle)
    values = []
    for xe in (-a, a):
        if -b <= cy <= b:
            values.append(abs(xe - cx))
    for ye in (-b, b):
        if -a <= cx <= a:
            values.append(abs(ye - cy))
    for xe in (-a, a):
        for ye in (-b, b):
            values.append(float(np.hypot(xe - cx, ye - cy)))
    return [v for v in values if v > 0.0]
