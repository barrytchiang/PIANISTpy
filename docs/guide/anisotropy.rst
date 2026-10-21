Velocity anisotropy
===================

:class:`pianistpy.Anisotropy`, :class:`pianistpy.Component`

One class, ``Anisotropy(beta_c, r_a, kind)``, covers every analytical anisotropy model of |paper|:

- the default, ``Anisotropy()``, is isotropic;
- a nonzero central anisotropy ``beta_c`` alone, with :math:`\beta_{\rm c} < 1`, gives constant anisotropy;
- a finite anisotropy radius ``r_a`` gives the Osipkov-Merritt model (with ``beta_c = 0``) or the
  Cuddeford-Osipkov-Merritt (COM) model (with a nonzero ``beta_c``). ``kind="I"`` (the default) is radially biased,
  ``kind="II"`` tangentially biased.

The anisotropy model is chosen by exact comparison of the parameters. ``beta_c=0.0`` gives the isotropic model, and
``beta_c=1e-12`` gives a constant-anisotropy model. The attribute ``Anisotropy.model`` gives the anisotropy model:
``"isotropic"``, ``"constant"``, ``"om"`` or ``"com"``. Results are continuous across :math:`\beta_{\rm c} = 0` and
across an infinite :math:`R_{\rm a}`. A model with :math:`\beta_{\rm c} = 10^{-12}` agrees with the isotropic model
within the accuracy of the settings (:doc:`accuracy`).

.. code-block:: python

   pp.Anisotropy()                           # isotropic
   pp.Anisotropy(beta_c=0.25)                # constant anisotropy
   pp.Anisotropy(r_a=10.0, kind="I")         # Osipkov-Merritt, radially biased
   pp.Anisotropy(beta_c=-0.25, r_a=10.0)     # COM, Type I by default

``Component(profile, anisotropy)`` joins a density profile and an anisotropy. A ``Component`` is the input of
:func:`~pianistpy.compute_df` and :func:`~pianistpy.solve`. In ver. 1.0.0, ``compute_df`` and ``solve`` take one
component. Several components in one potential are among the :doc:`../project/upcoming-features`.
