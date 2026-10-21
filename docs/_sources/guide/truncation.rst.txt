Truncation in phase space
=========================

:class:`pianistpy.Truncation`, :func:`pianistpy.truncate`, :func:`pianistpy.solve`, :class:`pianistpy.Model`

Method I
--------

``truncate(df, truncation)`` returns a :class:`~pianistpy.Model`, which holds the distribution function truncated at an
energy and the self-consistent potential-density pair of the truncated distribution function. If the untruncated
distribution function is everywhere non-negative and monotonic, the truncated distribution function inherits both
properties, and the truncation boundary is in dynamical equilibrium. A request with no physical model raises
:class:`~pianistpy.UnphysicalModelError`, which carries the distribution function, or the model without an edge, for
inspection.

At :math:`\beta_{\rm c} = -1/2` and a steep inner density slope :math:`\gamma`, the iteration for the center of the
truncated model does not converge, and ``truncate`` raises :class:`~pianistpy.AccuracyError`. For example, with
:math:`(\alpha, \beta) = (2, 4)`, :math:`\beta_{\rm c} = -1/2` and :math:`\mathcal{K}_{\rm t} = 0.3`, ``truncate``
raises :class:`~pianistpy.AccuracyError` from :math:`\gamma = 1.735` upward.

``Truncation`` takes exactly one of three inputs:

.. code-block:: python

   pp.Truncation(k_t=0.2)                          # the truncation energy, in internal units
   pp.Truncation(mass_t=1.0, concentration=10.0)   # a target mass, in units of M_0 (model units)
   pp.Truncation(r_t=5.0)                          # a target radius, in units of the scale radius

A target mass is given in units of :math:`M_0`, so it needs the concentration that defines :math:`M_0`. A target
radius or a truncation energy needs no concentration, and a concentration given with ``r_t`` or ``k_t`` raises
:class:`~pianistpy.InputError`.

``solve(component, truncation)`` runs ``compute_df`` and ``truncate`` in one call. Without a ``Truncation``,
``truncate`` and ``solve`` return the untruncated model, or, for a Method II component, the model that ends at the
edge of the truncated density that the component uses.

A model holds:

- the truncated distribution function;
- the truncated density, potential and enclosed mass on one radial grid (``model.profile.r``, ``.rho_trunc``,
  ``.p_trunc``, ``.mass_trunc``);
- the truncation radius and the total mass;
- the validation report and the identity string, a digest of the inputs, the settings and the package version.

An untruncated model is not spatially finite. The untruncated model's ``spatially_finite`` is false, and the
untruncated model's truncation radius ``r_t`` is infinite. A function that needs a spatially finite model, such as
``Sampler`` without ``r_sample``, raises :class:`~pianistpy.InputError` for a model that is not spatially finite
(:doc:`Flags and refusals <flags-refusals>`).

Type II models
^^^^^^^^^^^^^^

A Type II model (Osipkov-Merritt or Cuddeford-Osipkov-Merritt) must end inside its anisotropy radius
:math:`R_{\rm a}`, where the model's density :math:`\varrho_{\rm aux} / (1 - R^2/R_{\rm a}^2)`, with
:math:`\varrho_{\rm aux}` the auxiliary density, has a pole.

- The truncation energy must satisfy :math:`\mathcal{K}_{\rm t} > P(R_{\rm a})`. Where this bound sets ``k_t_min``,
  ``k_t_min`` is the next floating-point number above :math:`P(R_{\rm a})`.
- The bound is necessary, not sufficient. The bound keeps inside :math:`R_{\rm a}` the radius at which the
  untruncated potential equals :math:`\mathcal{K}_{\rm t}`, but the truncated model's own edge lies farther out than
  that radius. Where the truncated model's edge would reach :math:`R_{\rm a}`, ``truncate`` raises
  :class:`~pianistpy.UnphysicalModelError` with a message that begins ``TYPE II EDGE`` and states the potential at
  :math:`R_{\rm a}` or the radius of the edge. A larger :math:`\mathcal{K}_{\rm t}` moves the truncated model's edge
  inside :math:`R_{\rm a}`.

Osipkov-Merritt Type I models with a small anisotropy radius
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An Osipkov-Merritt Type I model with a small :math:`R_{\rm a}` and a low :math:`\mathcal{K}_{\rm t}` can have no edge
inside the largest radius at which ``truncate`` solves for the potential. ``truncate`` then raises
:class:`~pianistpy.UnphysicalModelError` with a message that begins ``NO EDGE``. On the Hernquist density with the
quick settings, ``truncate`` raises :class:`~pianistpy.UnphysicalModelError` (``NO EDGE``) for the following
Osipkov-Merritt Type I models:

- :math:`R_{\rm a} = 1` at every :math:`\mathcal{K}_{\rm t}` from 0.005 to 0.3;
- :math:`R_{\rm a} = 2` below :math:`\mathcal{K}_{\rm t} = 0.2`;
- :math:`R_{\rm a} = 5` below :math:`\mathcal{K}_{\rm t} = 0.03`;
- :math:`R_{\rm a} = 10` at :math:`\mathcal{K}_{\rm t} = 0.005`.

The Osipkov-Merritt Type I models on the Hernquist density with :math:`R_{\rm a}` from 1 to 10 and
:math:`\mathcal{K}_{\rm t}` from 0.005 to 0.3 that ``truncate`` builds have the tier ``"recorded"``, above
:math:`10^{-6}` and at most :math:`10^{-4}` (:doc:`accuracy`). The check ``high_k_slope_mismatch`` of the distribution
function of each built model is :math:`3.4 \times 10^{-6}`, at :math:`\mathcal{K} = 1`.

Method II
---------

Method II gives a new anisotropy to a density that is already truncated. The new component takes the truncated
profile of a Method I model as its profile. ``solve``, or ``compute_df`` and ``truncate``, without a
``Truncation``, build the Method II model from the new component.

.. code-block:: python

   truncated = pp.solve(pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy()), pp.Truncation(k_t=0.2))
   model = pp.solve(pp.Component(truncated.profile, pp.Anisotropy(beta_c=-0.25)))

A Method II model exists only when its anisotropy is compatible with the edge of the Method I model's truncated
density. Where that density goes as :math:`\hat{P}^q` at the edge, with :math:`\hat{P}` the Method I model's truncated
relative potential, which is zero at the edge, the central anisotropy of the new component must satisfy
:math:`\beta_{\rm c} > 1/2 - q`. A ``Component`` whose :math:`\beta_{\rm c}` is not above :math:`1/2 - q` raises
:class:`~pianistpy.UnphysicalModelError` when the ``Component`` is created, and the error message states the bound.
The :doc:`Technical notes <../reference/notes>` derive the condition :math:`\beta_{\rm c} > 1/2 - q`.
