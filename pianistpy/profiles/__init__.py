r"""Density profiles: the analytic input profile and the profile of a model.

A user builds the input profile of a component with `DoublePowerLaw`, for
example ``pp.DoublePowerLaw.nfw()``. The self-consistent profile of a model
is a `TruncatedProfile`, read as ``model.profile``.
`TruncatedProfile.interpolation_tables` returns the `ProfileTables` of a
`TruncatedProfile`.

The input profile is the double power law

.. math::

    \varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha},

where :math:`R` is the radius in units of the scale radius :math:`r_0`,
:math:`\varrho` is the density in internal units, :math:`\alpha` is the
sharpness of the transition between the inner and the outer slope,
:math:`\beta` is the outer logarithmic slope, :math:`\gamma` is the inner
logarithmic slope, and :math:`C` is the normalization that sets the central
relative potential :math:`P(0)` to 1. Near the center, a model's profile
takes the depth :math:`P(0) - P(R)` from the depth's series in
:math:`R^{\alpha}`, out to the largest radius at which the series' next
term is at most :math:`10^{-7}` of the depth. The formulas are in Notes
Section T:1.
"""

__all__ = ["DoublePowerLaw", "ProfileTables", "TruncatedProfile"]

from .core import DoublePowerLaw
from .truncated import ProfileTables, TruncatedProfile
