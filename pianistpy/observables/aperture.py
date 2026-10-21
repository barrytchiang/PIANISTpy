r"""Circular and rectangular apertures on the plane of the sky.

A user builds an `Aperture` with `Aperture.circle` or `Aperture.rectangle`
and passes the aperture to `sigma_los_aperture` or `aperture_losvd`. The
model is spherical, so the integral of a quantity :math:`f(R)` over the
aperture is

.. math::

    \int_A f\,{\rm d}A = \int f(R)\,{\rm arc}(R)\,R\,{\rm d}R,

with :math:`R` the projected radius and :math:`{\rm arc}(R)` the angle, in
radians, of the part of the circle of radius :math:`R` about the center of
the model that lies inside the aperture. `Aperture.arc` evaluates
:math:`{\rm arc}(R)` in closed form. `Aperture.breaks` returns the projected
radii at which :math:`{\rm arc}(R)` is not smooth. A quadrature in
:math:`R` with interval ends at the projected radii of `Aperture.breaks`
integrates a smooth function on each interval. The closed forms of `Aperture.arc` and
`Aperture.breaks` are exact and carry no tolerance. The apertures are the
subject of Notes Section T:6.3.
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

    The constructors `Aperture.circle` and `Aperture.rectangle` build an
    aperture. The lengths of an aperture are in the unit system of the
    model or `LosTable` that the aperture is used with. In the internal and
    the model systems, the unit of length is the scale radius :math:`r_0`.
    In the physical system, the unit of length is the parsec.

    Parameters
    ----------
    shape : {"circle", "rectangle"}
        Shape of the aperture. ``"circle"`` is a circle of radius `r_ape`.
        ``"rectangle"`` is a rectangle of full sides `side_x` and `side_y`.
    r_off : float, optional
        Projected distance :math:`R_{\rm off}` of the center of the aperture
        from the center of the model. `r_off` must be zero or positive.
        Default: ``0.0``, which means the aperture is centered on the model.
    r_ape : float, optional
        Radius :math:`R_{\rm ape}` of a circle. `r_ape` must be positive for
        a circle and must be ``None`` for a rectangle. Default: ``None``.
    side_x : float, optional
        Full length of the rectangle's side along the rectangle's x axis.
        `side_x` must be positive for a rectangle and must be ``None`` for a
        circle. Default: ``None``.
    side_y : float, optional
        Full length of the rectangle's side along the rectangle's y axis.
        `side_y` must be positive for a rectangle and must be ``None`` for a
        circle. Default: ``None``.
    angle : float, optional
        Angle, in radians, between the rectangle's x axis and the line
        through the center of the model and the center of the rectangle.
        Default: ``0.0``, which means the rectangle's x axis lies on the
        line through the center of the model and the center of the
        rectangle. A circle accepts only ``0.0``.

    Raises
    ------
    InputError
        If `shape` is neither ``"circle"`` nor ``"rectangle"``.
    InputError
        If `r_off` is negative.
    InputError
        If a length the shape requires is missing or not positive.
    InputError
        If a length the shape does not use is given, or a circle is given a
        nonzero `angle`.

    See Also
    --------
    sigma_los_aperture : Line-of-sight velocity dispersion inside an
        aperture.
    aperture_losvd : Line-of-sight velocity distribution inside an aperture.

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
                "The aperture's shape must be 'circle' or 'rectangle'; the shape "
                f"given is {self.shape!r}."
            )
        if not self.r_off >= 0.0:
            raise InputError(
                "The aperture's offset r_off must not be negative; the value given "
                f"is {self.r_off}."
            )
        circle = self.shape == "circle"
        used = ("r_ape",) if circle else ("side_x", "side_y")
        unused = ("side_x", "side_y") if circle else ("r_ape",)
        for name in used:
            value = getattr(self, name)
            if value is None or not value > 0.0:
                raise InputError(
                    f"The aperture's {name} must be positive for a {self.shape}; "
                    f"the value given is {value}."
                )
        for name in unused:
            if getattr(self, name) is not None:
                raise InputError(
                    f"A {self.shape} aperture does not take {name}; {name} must be "
                    "None."
                )
        if circle and self.angle != 0.0:
            raise InputError(
                "A circle aperture does not take an angle; the angle must be 0."
            )

    @classmethod
    def circle(cls, r_ape: float, r_off: float = 0.0) -> "Aperture":
        r"""Return a circular aperture of radius `r_ape` at offset `r_off`.

        Parameters
        ----------
        r_ape : float
            Radius :math:`R_{\rm ape}` of the circle. `r_ape` must be
            positive.
        r_off : float, optional
            Projected distance :math:`R_{\rm off}` of the circle's center
            from the center of the model. `r_off` must be zero or positive.
            Default: ``0.0``, which means the circle is centered on the
            model.

        Returns
        -------
        aperture : Aperture
            The circular aperture. The lengths `r_ape` and `r_off` are in
            the unit system of the model that the aperture is used with.

        Raises
        ------
        InputError
            If `r_ape` is not positive or `r_off` is negative.

        See Also
        --------
        Aperture.rectangle : A rectangular aperture.
        """
        return cls(shape="circle", r_off=r_off, r_ape=r_ape)

    @classmethod
    def rectangle(
        cls, side_x: float, side_y: float, r_off: float = 0.0, angle: float = 0.0
    ) -> "Aperture":
        r"""Return a rectangular aperture of full sides `side_x` by `side_y`.

        Parameters
        ----------
        side_x : float
            Full length of the rectangle's side along the rectangle's x
            axis. `side_x` must be positive.
        side_y : float
            Full length of the rectangle's side along the rectangle's y
            axis. `side_y` must be positive.
        r_off : float, optional
            Projected distance :math:`R_{\rm off}` of the rectangle's center
            from the center of the model. `r_off` must be zero or positive.
            Default: ``0.0``, which means the rectangle is centered on the
            model.
        angle : float, optional
            Angle, in radians, between the rectangle's x axis and the line
            through the center of the model and the center of the rectangle.
            Default: ``0.0``, which means the rectangle's x axis lies on the
            line through the center of the model and the center of the
            rectangle.

        Returns
        -------
        aperture : Aperture
            The rectangular aperture. The lengths `side_x`, `side_y` and
            `r_off` are in the unit system of the model that the aperture is
            used with.

        Raises
        ------
        InputError
            If `side_x` or `side_y` is not positive, or `r_off` is negative.

        See Also
        --------
        Aperture.circle : A circular aperture.
        """
        return cls(
            shape="rectangle", r_off=r_off, side_x=side_x, side_y=side_y, angle=angle
        )

    def arc(self, r: Any) -> Array:
        r"""Return the angle, in radians, of the circle of radius `r` inside
        the aperture.

        The circle of radius `r` is centered on the model. The angle
        :math:`{\rm arc}(r)` is :math:`2\pi` where the whole circle lies
        inside the aperture. The angle is zero where no part of the circle
        lies inside the aperture. The integral over the aperture of a
        quantity :math:`f(R)` of projected radius is
        :math:`\int f(R)\,{\rm arc}(R)\,R\,{\rm d}R`.

        Parameters
        ----------
        r : array-like
            User-specified projected radii, zero or positive, in the unit of
            length of the aperture.

        Returns
        -------
        arc : array
            The angle :math:`{\rm arc}(r)` in radians, from zero to
            :math:`2\pi`, with the shape of `r`.

        See Also
        --------
        Aperture.breaks : The projected radii at which :math:`{\rm arc}(r)`
            is not smooth.

        Notes
        -----
        For a circle of radius :math:`R_{\rm ape}` at offset
        :math:`R_{\rm off} > 0`, the angle is :math:`2\pi` for
        :math:`r \le R_{\rm ape} - R_{\rm off}`, zero for
        :math:`r < |R_{\rm off} - R_{\rm ape}|` or
        :math:`r > R_{\rm off} + R_{\rm ape}`, and otherwise

        .. math::

            {\rm arc}(r) = 4 \arcsin\sqrt{x/2}, \qquad
            x = \frac{R_{\rm ape}^2 - (r - R_{\rm off})^2}{2 r R_{\rm off}}.

        For a centered circle (:math:`R_{\rm off} = 0`), the angle is
        :math:`2\pi` for :math:`r \le R_{\rm ape}` and zero otherwise.

        For a rectangle, each of the rectangle's four edge lines crosses
        the circle of radius `r` at angles given in closed form. These
        crossing angles cut the circle into arcs. Each arc lies wholly
        inside or wholly outside the rectangle. The angle
        :math:`{\rm arc}(r)` is the sum of the arcs inside the rectangle.
        The same closed form holds for a rectangle that contains the center
        of the model, a rectangle whose edge passes through the center of
        the model, and a rectangle that excludes the center of the model.
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
        """Return the field `name`, which the aperture's shape requires, as a
        float."""
        value = getattr(self, name)
        if value is None:
            raise InputError(f"A {self.shape} aperture needs a value of {name}.")
        return float(value)

    def breaks(self) -> list[float]:
        r"""Return the projected radii at which `Aperture.arc` is not smooth.

        A quadrature in projected radius whose intervals end at the returned
        projected radii integrates a smooth function on each interval.

        Returns
        -------
        breaks : list of float
            The positive projected radii, in increasing order and in the
            unit of length of the aperture. For a centered circle, the
            list holds :math:`R_{\rm ape}`. For an offset circle, the list
            holds the positive values among
            :math:`|R_{\rm off} - R_{\rm ape}|`,
            :math:`R_{\rm ape} - R_{\rm off}` and
            :math:`R_{\rm off} + R_{\rm ape}`. For a rectangle, the list
            holds at most eight projected radii. Four of the eight are the
            distances from the center of the model to the rectangle's four
            edge lines. Each such distance is included only where the point
            of the edge line nearest to the center of the model lies on the
            edge. The other four are the distances from the center of the
            model to the rectangle's four corners.

        See Also
        --------
        Aperture.arc : The angle of the circle of radius `r` inside the
            aperture.
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
        """Return the aperture as a plain dictionary.

        Returns
        -------
        record : dict
            The six fields of the aperture by name: ``shape``, ``r_off``,
            ``r_ape``, ``side_x``, ``side_y`` and ``angle``.
            `Aperture.from_dict` builds the same aperture from the
            returned dictionary.

        See Also
        --------
        Aperture.from_dict : Build an aperture from a dictionary.
        """
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
        """Return the aperture described by a dictionary of its fields.

        Parameters
        ----------
        record : mapping
            The fields of the aperture by name, as `Aperture.to_dict`
            writes them. The key ``shape`` is required. A missing ``r_off``
            or ``angle`` takes the value ``0.0``. A missing ``r_ape``,
            ``side_x`` or ``side_y`` takes the value ``None``.

        Returns
        -------
        aperture : Aperture
            The aperture with the fields of `record`.

        Raises
        ------
        InputError
            If the fields of `record` do not make a valid aperture, under
            the rules of the `Aperture` constructor.

        See Also
        --------
        Aperture.to_dict : Return the aperture as a plain dictionary.
        """
        return cls(
            shape=str(record["shape"]),
            r_off=float(record.get("r_off", 0.0)),
            r_ape=record.get("r_ape"),
            side_x=record.get("side_x"),
            side_y=record.get("side_y"),
            angle=float(record.get("angle", 0.0)),
        )


# The closed forms of arc(r) and of its break radii for a circle and for a
# rectangle. Each function takes lengths in the unit of the aperture.


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
