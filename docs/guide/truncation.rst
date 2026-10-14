Truncation in phase space
=========================

:class:`pianistpy.Truncation`, :func:`pianistpy.truncate`, :func:`pianistpy.solve`, :class:`pianistpy.Model`

Method I
--------

``truncate(df, truncation)`` returns a :class:`~pianistpy.Model`: the distribution function truncated at an energy,
with the self-consistent potential-density pair that follows from it. A ``Model`` is by construction a physical
and stable equilibrium; a request for which none exists is refused with :class:`~pianistpy.UnphysicalModelError`,
which carries the distribution function for inspection.

``Truncation`` takes exactly one of three targets:

.. code-block:: python

   pp.Truncation(k_t=0.2)                          # the truncation energy, in internal units
   pp.Truncation(mass_t=1.0, concentration=10.0)   # a target mass, in units of M_0 (model units)
   pp.Truncation(r_t=5.0)                          # a target radius, in units of the scale radius

A target mass is given in units of M_0, so it needs the concentration that defines M_0; a target radius and a
truncation energy do not, and a concentration given with them is refused.

``solve(component, truncation)`` runs ``compute_df`` and ``truncate`` at once.

What a model holds: the truncated distribution function, the truncated density, potential and enclosed mass on
one radial grid (``model.profile.r``, ``.rho_trunc``, ``.p_trunc``, ``.mass_trunc``), the truncation radius and
the total mass, the validation report and the identity. A model that is not spatially finite is still a model:
it says so (``spatially_finite`` is false), states its radius and mass as not finite, and what needs finiteness
asks for it by name (:doc:`results`).

Method II
---------

Method II gives a new anisotropy to a density that is already truncated: a component whose profile is the
truncated profile of another model, through the same functions.

.. code-block:: python

   parent = pp.solve(pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy()), pp.Truncation(k_t=0.2))
   child = pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=-0.25)))

A Method II model exists only when its anisotropy is compatible with the edge of the parent's density; the
condition is checked at the call and a violation is refused by name, with the admissible range stated. The
technical notes derive it (:doc:`../reference/notes`).
