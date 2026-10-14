Observables and apertures
=========================

:func:`pianistpy.sigma_los`, :func:`pianistpy.sigma_los_aperture`, :class:`pianistpy.Aperture`,
:class:`pianistpy.LosTable`

Observables are functions of a model. A single request computes only what it needs; for repeated requests on one
model, the prepared object ``LosTable(model)`` tabulates the line-of-sight velocity distributions once and every
observable reads from it.

.. code-block:: python

   sigma = pp.sigma_los(model, [0.1, 1.0, 10.0])        # at projected radii

   circle = pp.Aperture.circle(r_ape=1.0, r_off=0.0)
   slit = pp.Aperture.rectangle(side_x=2.0, side_y=0.2, r_off=0.0, angle=0.0)
   pp.sigma_los_aperture(model, circle)                 # averaged over an aperture
   pp.sigma_los_aperture(model, slit)

   table = pp.LosTable(model)                           # once, for many requests
   pp.sigma_los(table, [0.1, 1.0, 10.0])
   pp.sigma_los_aperture(table, circle)

An ``Aperture`` is circular or rectangular, centered or offset, the rectangle at an angle; every aperture call
takes the same object. The intrinsic radial and tangential dispersions, the line-of-sight velocity distribution
itself and its Gauss-Hermite moments are available in the same way; their public names are listed in the
:doc:`API reference <../reference/api/observables>`.

An aperture of arbitrary shape is not an exact route: the notebook of the same name shows how a mask on
projected draws gives it, with the statistical error printed beside every number (:doc:`../notebooks/index`).
