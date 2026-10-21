Density profiles
================

:class:`pianistpy.DoublePowerLaw`

``DoublePowerLaw(rho_alpha, rho_beta, rho_gamma)`` is the analytical double-power-law density profile
:math:`\varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha}`, with :math:`R` the radius in units of
the scale radius and :math:`C` a normalization constant. The class takes three arguments:

- ``rho_alpha``, the sharpness :math:`\alpha` of the transition between the inner and the outer slope, at least
  1/2;
- ``rho_beta``, the outer slope :math:`\beta`, greater than 2 for the profile and at least 3 for a component;
- ``rho_gamma``, the inner slope :math:`\gamma`, with :math:`0 \leq \gamma < 2`, so that the central potential is
  finite.

In internal units (:doc:`Units <units>`), the scale radius and the central potential of a ``DoublePowerLaw``
profile are 1. The NFW, Hernquist and Plummer profiles are parameter sets of ``DoublePowerLaw``, not classes of
their own:

.. code-block:: python

   pp.DoublePowerLaw.nfw()        # Navarro-Frenk-White
   pp.DoublePowerLaw.hernquist()
   pp.DoublePowerLaw.plummer()
   pp.DoublePowerLaw(1.0, 3.0, 1.0)   # the NFW profile, given by (rho_alpha, rho_beta, rho_gamma)

The methods ``rho(r)``, ``mass(r)`` and ``p(r)`` of a ``DoublePowerLaw`` profile return the density, the enclosed mass
and the relative potential in closed form.

A component (:class:`pianistpy.Component`) of a model is a density profile together with a velocity anisotropy
(:doc:`Velocity anisotropy <anisotropy>`). The density profile of a component needs an outer slope
:math:`\beta \geq 3`. In ver. 1.0.0, a component whose profile has :math:`2 < \beta < 3` raises
:class:`~pianistpy.InputError`, with a message that begins ``OUTER SLOPE``.
