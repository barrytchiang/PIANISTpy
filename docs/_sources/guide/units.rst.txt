Units
=====

Everything is computed in internal units, the dimensionless normalization of `Widrow (2000) <https://doi.org/10.1086/317367>`__. In internal units,
the scale radius and the central potential of the untruncated profile are 1.

The ``to`` method of a distribution function, a model or a :class:`~pianistpy.LosTable` presents the result in model
units :math:`(G = M_0 = r_0 = 1)` or, through `Astropy <https://www.astropy.org/>`_, in physical units. Both systems
need the concentration :math:`c`, which defines :math:`M_0` as the mass of the untruncated profile inside
:math:`c\,r_0`. The numbers are not recomputed. Observables and particles computed from the presented result are
returned in the unit system of the presented result. Some inputs and outputs stay in internal units whatever the
model's unit system: ``Sampler``'s ``r_sample`` and ``r_range``, the radii, the ``v_range`` and the returned velocities of
``Sampler.velocities``, every argument and value of :class:`~pianistpy.Differentiable`, and the anisotropy ``theta`` of
``likelihood``.

Exactly one of the three inputs of a :class:`~pianistpy.Truncation` is given, in these units:

- the truncation energy ``k_t`` in internal units;
- a target radius ``r_t`` in units of the scale radius;
- a target mass ``mass_t`` in units of :math:`M_0`, together with the concentration that defines :math:`M_0`.
