r"""Velocity anisotropy models of a component's distribution function (DF).

A user builds the anisotropy of a component with `Anisotropy` and passes
the `Anisotropy` object to `Component`. One class covers four models: the
isotropic model, the constant-anisotropy model, the Osipkov-Merritt (OM)
model and the Cuddeford-Osipkov-Merritt (COM) model. The COM model of central
anisotropy :math:`\beta_{\rm c}` and anisotropy radius :math:`R_{\rm a}` has
the anisotropy profile

.. math::

    \beta(R) = \frac{\beta_{\rm c} \pm R^2/R_{\rm a}^2}{1 \pm R^2/R_{\rm a}^2},

with the upper sign for Type I and the lower sign for Type II, and
:math:`R` the radius. The OM model is the COM model with
:math:`\beta_{\rm c} = 0`. The constant-anisotropy model is the COM model
with an infinite :math:`R_{\rm a}`, and the isotropic model is the
constant-anisotropy model with :math:`\beta_{\rm c} = 0`. The factor
:math:`g(R)` that relates the augmented density to the auxiliary density is
given in Chiang+(2026d), Eq. (B44).
"""

__all__ = ["Anisotropy"]

from .core import Anisotropy
