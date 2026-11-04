Release notes
=============

Ver. 1.0.1
----------

New features
^^^^^^^^^^^^

- ``DistributionFunction.evaluate`` returns the distribution function at given values of :math:`\mathcal{K}`.
- ``round_trip`` integrates a distribution function to its density, solves the Poisson equation for the potential,
  and reports the relative difference from the input density profile at each radius.
- A model's profile continues inside its innermost tabulated radius by the inner series of its input density,
  corrected to first order for the truncation; the particle sampler reads the same series.

Changes
^^^^^^^

- The energy-like variable is written :math:`\mathcal{K}` throughout: in docstrings, messages and generated tables,
  and in the names ``EnergyGrid`` (now ``KGrid``), ``n_energy_nodes`` and ``energy_layer_ratio`` (now ``n_k_nodes``
  and ``k_layer_ratio``), the timing stage ``"energy_grid"`` (now ``"k_grid"``) and the accuracy files' column
  "lowest energy of the support" (now "lowest K of the support"). Likewise ``"edge_law"`` is ``"boundary_form"``,
  and "edge law", "Kepler form", "truncation energy" and "Poisson solve" read "boundary form", "point-mass
  (Keplerian) potential", "truncation value" and "Poisson solver". Every former name is accepted through ver. 1.0.x
  and removed in ver. 1.1.0; the known limitations of ver. 1.0.0 stand.

Ver. 1.0.0
----------

The initial public release of PIANISTpy, described in |paper|.

Known limitations
^^^^^^^^^^^^^^^^^

- Spherical, non-rotating systems with a double-power-law input density profile.
- One density profile per potential (:doc:`upcoming-features`).
- Gradients in the anisotropy parameters are available only for Method II models, whose input density profile
  is the truncated density profile of another model. A Method I truncated model raises
  :class:`~pianistpy.InputError`. Separately, for Cuddeford-Osipkov-Merritt (COM) Type II models, the
  distribution function and every observable have no gradient in the anisotropy parameters
  (:doc:`../guide/differentiability`).
- With ``Settings(check_rows=True)``, the gradient of ``losvd`` and ``aperture_losvd`` recomputes each row of
  the line-of-sight table that the gradient reads and requires the recomputed row to equal the stored row bit for
  bit, and a JAX copy of the row's computation to reproduce the row to :math:`10^{-12}` of the row's largest
  value. On a GPU, whose order of summation does not promise the same bits, the comparison can fail a row and the
  gradient raises ``ResultError`` instead of returning an inexact value (:doc:`../guide/differentiable`). The
  default, ``check_rows=False``, recomputes and compares no row.
- The derivative of ``losvd`` and ``aperture_losvd`` through the line-of-sight table takes minutes to hours
  (:doc:`../guide/differentiable`).

Dependencies: the standard and the minimum versions are listed on :doc:`../start/install`.
