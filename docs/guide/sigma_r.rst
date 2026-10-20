Radial velocity dispersion
==========================

:func:`pianistpy.sigma_r`, :class:`pianistpy.LosTable`

``sigma_r(source, r)`` returns the intrinsic radial velocity dispersion :math:`\sigma_{\rm r}` at 3D radii: the second
moment of the radial velocity over the model's distribution function at a fixed radius. As in |paper|, it is
the velocity moment of the augmented density :math:`\varrho_{\rm aug}(P, R)` taken at the potential of the radius,

.. math::

   \sigma_{\rm r}^2(R) = \frac{1}{\varrho_{\rm aug}(P, R)} \int_0^{P} \varrho_{\rm aug}(\mathcal{K}, R)\,
   {\rm d}\mathcal{K}, \qquad P = P(R),

which for the anisotropy models of the package, whose augmented density separates as
:math:`\varrho_{\rm aug}(P, R) = \varrho_{\rm aux}(P)\,g(R)`, is the Jeans integral of :math:`\varrho_{\rm aux}`
over the potential. The radius :math:`R` of |paper| is ``r`` in the package.

Arguments and units
-------------------

``source`` is a :class:`~pianistpy.Model` (one component, or a superposition) or its prepared
:class:`~pianistpy.LosTable`, which holds the same intrinsic profile. ``r`` is an array of radii in the model's
unit system: internal units by default (the scale radius is 1), parsecs (a float or an Astropy quantity) after
``model.to("physical", ...)``. The result has the shape of ``r``, in the model's velocity unit (km/s in the
physical system).

Of a superposition, whose components share one density, the second moments add with the weights:
:math:`\sigma_{\rm r}^2 = \sum_k w_k \sigma_{{\rm r},k}^2`.

Tiers
-----

The profile is tabulated once per model in radius, with rows added until every interval between rows reads
within ``profile_tolerance`` at its midpoint (:math:`10^{-7}` at the default tier, :math:`10^{-6}` at the quick tier; a tenth of the
goal, the profile's share of the error of what is projected from it). A model beyond the hard limit is refused
unless its settings ask for it (``write_anyway``), as for every observable (:doc:`accuracy`).

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
