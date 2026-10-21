Radial velocity dispersion
==========================

:func:`pianistpy.sigma_r`, :class:`pianistpy.LosTable`

``sigma_r(source, r)`` returns the intrinsic radial velocity dispersion :math:`\sigma_{\rm r}` at user-specified 3D radii.
:math:`\sigma_{\rm r}^2` is the second moment of the radial velocity over the model's distribution function at a
fixed radius. As in |paper|, :math:`\sigma_{\rm r}^2` is the velocity moment of the augmented density
:math:`\varrho_{\rm aug}`, integrated over the energy :math:`\mathcal{K}` up to the relative potential
:math:`P = P(R)` at the radius :math:`R`,

.. math::

   \sigma_{\rm r}^2(R) = \frac{1}{\varrho_{\rm aug}(P, R)} \int_0^{P} \varrho_{\rm aug}(\mathcal{K}, R)\,
   {\rm d}\mathcal{K}, \qquad P = P(R).

For the anisotropy models of the package, the augmented density separates as
:math:`\varrho_{\rm aug}(P, R) = \varrho_{\rm aux}(P)\,g(R)`, with :math:`\varrho_{\rm aux}` the auxiliary density
and :math:`g` the anisotropy factor (``Anisotropy.factor``). :math:`\sigma_{\rm r}^2` is then the Jeans integral of
:math:`\varrho_{\rm aux}` over the potential. The radius :math:`R` of |paper| is ``r`` in the package.

Arguments and units
-------------------

- ``source`` is a :class:`~pianistpy.Model` (one component, or a superposition) or the model's
  line-of-sight table (:class:`~pianistpy.LosTable`), which holds the same intrinsic profile.
- ``r`` is an array of 3D radii in the model's unit system. In the internal and the model systems, the unit of ``r``
  is the scale radius. After ``model.to("physical", ...)``, ``r`` is in parsecs, or an Astropy quantity of any
  length unit.
- The result has the shape of ``r``, in the model's velocity unit (km/s in the physical system).

The members of a superposition, Method II models on one truncated density, share that density, so the members' second moments
add with the weights:
:math:`\sigma_{\rm r}^2 = \sum_k w_k \sigma_{{\rm r},k}^2`.

Accuracy
--------

The profile is tabulated in radius once per model. Radii are added to the profile until :math:`\sigma_{\rm r}` at
the midpoint of every interval between the profile's radii is within a relative ``profile_tolerance``:
:math:`10^{-7}` with the default settings and :math:`10^{-6}` with ``Settings(quick=True)``. The default value of
``profile_tolerance``, :math:`10^{-7}`, is a tenth of :math:`10^{-6}`, so the profile adds little to the error of the
projected quantities computed from the profile. As for every observable, a model whose relative error is above
:math:`10^{-2}` raises :class:`~pianistpy.AccuracyError`, unless the model's settings set ``write_anyway=True``
(:doc:`accuracy`).

Example
-------

.. code-block:: python

   import numpy as np
   import pianistpy as pp

   # a Hernquist density with a constant anisotropy of 0.25, truncated at K_t = 0.05
   model = pp.solve(
       pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
       pp.Truncation(k_t=0.05),
       settings=pp.Settings(quick=True),
   )
   r = np.array([0.1, 1.0, 10.0])
   pp.sigma_r(model, r)                          # internal units

   physical = model.to("physical", concentration=10.0, mass_0=1e9, r_0=500.0)
   pp.sigma_r(physical, [50.0, 500.0, 5000.0])   # parsecs in, km/s out
