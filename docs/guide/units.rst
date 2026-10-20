Units
=====

Everything is computed in internal units, the dimensionless normalization of Widrow (2000) in which the scale
radius and the central potential of the untruncated profile are 1. The ``to`` method of a model (and of a
line-of-sight table) presents it in model units :math:`(G = M_0 = r_0 = 1)` or, through
`Astropy <https://www.astropy.org/>`_, in physical units, without recomputing it, and what is computed from it
(the observables, the particles) is returned in that system.
The ranges of a draw are the exception: ``Sampler``'s ``r_range`` and the ``v_range`` of ``Sampler.speeds`` are
given in internal units.

Three inputs of a truncation are given in the units that are natural for them, as their documentation states:
the truncation energy ``k_t`` in internal units, a target radius ``r_t`` in units of the scale radius, and a
target mass ``mass_t`` in units of :math:`M_0`, together with the concentration that defines :math:`M_0`.
