Line-of-sight velocity distribution in an aperture
==================================================

:func:`pianistpy.aperture_losvd`, :class:`pianistpy.Aperture`, :class:`pianistpy.LosTable`

``aperture_losvd(source, aperture, v)`` returns the line-of-sight velocity distribution of the model's mass inside
an aperture. :math:`\overline{D}` is the line-of-sight velocity distribution :math:`D` at each projected radius
(:doc:`losvd`), averaged over the aperture with the surface density as the weight:

.. math::

   \overline{D}(V_{\rm los}) = \frac{\int_{\rm aperture} \Sigma(R_{\rm 2D})\, D(R_{\rm 2D}, V_{\rm los})\,{\rm d}A}
   {\int_{\rm aperture} \Sigma(R_{\rm 2D})\,{\rm d}A}.

The distribution is even in :math:`V_{\rm los}` and integrates to one over all :math:`V_{\rm los}`.
:math:`\overline{D}` is zero at every :math:`|V_{\rm los}|` above the largest velocity that the distribution function
reaches on the lines of sight through the aperture. The second moment of :math:`\overline{D}` is the squared
dispersion inside the aperture, :math:`\overline{\sigma}_{\rm los}^2`. Given the same ``LosTable``,
:func:`pianistpy.sigma_los_aperture` returns :math:`\overline{\sigma}_{\rm los}`.

Arguments and units
-------------------

- ``source`` is a :class:`~pianistpy.Model` or the model's line-of-sight table (:class:`~pianistpy.LosTable`).
  Given a model, ``aperture_losvd`` builds the model's ``LosTable`` at every call. For more than one call, build
  the ``LosTable`` once (:doc:`lostable`).
- ``aperture`` is an :class:`~pianistpy.Aperture`: circular or rectangular, centered or offset (:doc:`observables`).
  The aperture's lengths are in the model's unit of length.
- ``v`` is an array of line-of-sight velocities in the model's unit of velocity (km/s in the physical system).

The result has the shape of ``v`` and is a probability density per unit velocity of the model's unit system.

For a superposition, the call sums the components' distributions in the aperture. The distribution of component
:math:`k` is weighted by the component's weight :math:`w_k` times the component's projected mass :math:`M_k` inside
the aperture:
:math:`\overline{D} = \sum_k w_k M_k \overline{D}_k / \sum_k w_k M_k`.

Accuracy report
---------------

The report of ``aperture_losvd`` checks the second moment of :math:`\overline{D}` against
:math:`\overline{\sigma}_{\rm los}^2` computed from the projected dispersion, without the velocity distributions. The
tier of the result is the worse of the aperture's check and the tier of the ``LosTable`` (:doc:`lostable`).
``aperture_losvd`` issues an :class:`~pianistpy.AccuracyWarning` if the relative error of :math:`\overline{D}` or
of the ``LosTable`` is above :math:`10^{-4}`.

The quick settings lose the most accuracy in a small aperture. Take an NFW halo of concentration 30 with a
Cuddeford-Osipkov-Merritt (COM) Type I anisotropy (:math:`\beta_{\rm c} = -0.25`, :math:`R_{\rm a} = 10`):

- In a centered circle of radius 0.05 scale radii, the second moment with the quick settings differs from the
  second moment with the default settings by a relative :math:`6.7 \times 10^{-6}`. The check of the
  quick-settings result reports :math:`6.2 \times 10^{-6}` (tier ``"recorded"``). With the default settings the
  check reports :math:`1.0 \times 10^{-7}`.
- In a circle of radius 1 scale radius, the second moments with the quick and with the default settings differ by
  a relative :math:`1.5 \times 10^{-6}`. With the default settings the check reports :math:`6.4 \times 10^{-8}`.

For this NFW model, the quick settings build the ``LosTable`` in about a quarter of the time that the default
settings take.

Example
-------

.. code-block:: python

   import numpy as np
   import pianistpy as pp

   model = pp.solve(
       pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
       pp.Truncation(k_t=0.05),
       settings=pp.Settings(quick=True),
   )
   table = pp.LosTable(model)
   circle = pp.Aperture.circle(r_ape=1.0)

   v = np.linspace(-1.5, 1.5, 3001)
   d = pp.aperture_losvd(table, circle, v)       # shape (3001,)
   dv = v[1] - v[0]
   d.sum() * dv                                   # one, to the accuracy of the sum on this grid
   np.sqrt((v**2 * d).sum() * dv), pp.sigma_los_aperture(table, circle)

   pp.aperture_losvd(table, pp.Aperture.circle(r_ape=1.0, r_off=10.0), v)   # an offset circle
