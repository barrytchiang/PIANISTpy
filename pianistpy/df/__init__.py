r"""Distribution functions and equilibrium models of spherical systems.

The `pianistpy.df` module computes the distribution function (DF) of a
spherical density profile with a given velocity anisotropy, and the
equilibrium model that the DF supports. A `Component` holds one density
profile with its anisotropy. `solve` returns the model of a component.
`compute_df` returns the untruncated DF of a component, and `truncate`
builds the model from that DF, with or without a `Truncation`.

The DF of every anisotropy model of the package has the separable form

.. math::

    F(\mathcal{K}, \mathcal{L}) = F_{\mathcal{K}}(\mathcal{K})\,
    \mathcal{L}^{-2\beta_{\rm c}}

(Chiang+(2026d), Eq. (8)). :math:`\mathcal{K}` is the energy-like
variable in internal units, with :math:`0 \le \mathcal{K} \le 1`.
:math:`\mathcal{K}` is the relative energy :math:`Z` for a constant
anisotropy, and :math:`\mathcal{K}` is :math:`\mathcal{Q}_\pm` for an
Osipkov-Merritt (OM) or Cuddeford-Osipkov-Merritt (COM) model.
:math:`\mathcal{L}` is the angular momentum in internal units, and
:math:`\beta_{\rm c}` is the central anisotropy. A truncation at the
energy :math:`\mathcal{K}_{\rm t}` (Method I of Chiang+(2026d)) gives the truncated DF

.. math::

    \hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
    F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
    - F(\mathcal{K}_{\rm t}, \mathcal{L})

for :math:`\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0`,
and :math:`\hat{F} = 0` for :math:`\hat{\mathcal{K}} < 0`.

Each DF and each model carries a validation report. `truncate` and `solve`
raise `AccuracyError` if a check of the model's validation report lies
above :math:`10^{-2}` (the band "refused"), the round trip of the model's
DF against the model's density included, unless the settings set
``write_anyway=True``. `compute_df`, `truncate` and `solve` issue an
`AccuracyWarning` if a check of the returned result's validation report
lies above :math:`10^{-4}`.

The derivation is in Chiang+(2026d), Section 3.1, and the numerical method
in Notes Section T:2.
"""

__all__ = [
    "Component",
    "DistributionFunction",
    "Model",
    "Truncation",
    "compute_df",
    "solve",
    "truncate",
]

from .core import compute_df, solve, truncate
from .inputs import Component, Truncation
from .results import DistributionFunction, Model
