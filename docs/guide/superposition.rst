Superposition and inference
===========================

:func:`pianistpy.superpose`, :mod:`pianistpy.inference`

.. _guide-superposition:

Superposition
-------------

Method II models built on the truncated density of one Method I model
(:doc:`Truncation in phase space <truncation>`) share that density. The distribution functions of those Method II models, combined with
positive weights that sum to one, form one equilibrium model in that truncated density. ``superpose`` returns the combined model as a
:class:`~pianistpy.Model`, which ``Sampler``, the observables, ``likelihood`` and ``write`` take as any other model.
``Differentiable`` raises :class:`~pianistpy.InputError` for a combined model, whose components are differentiated
through ``likelihood``.

.. code-block:: python

   members = {model_a: 0.6, model_b: 0.3, model_c: 0.1}   # Method II models on one truncated density
   composite = pp.superpose(members)
   pp.Sampler(composite).sample(1_000_000, seed=1)

``superpose`` takes a mapping of two or more Method II models on one truncated density, the members, to the members'
weights.
``superpose`` raises :class:`~pianistpy.InputError` when a weight is not positive and finite, or when the members
are built on different truncated profiles or with different settings or units. The weights are normalized to sum to
one, and a warning goes to the package's log when the sum of the weights differs from one by more than
:math:`10^{-10}`.

.. _guide-inference:

Inference
---------

:mod:`pianistpy.inference` gives the log-likelihood of a model against data as a JAX function. The function's arguments
are the weights of the components, the mass scale and the length scale in the units of the data, and, for the
components listed in ``free=``, the anisotropy :math:`(\beta_{\rm c}, R_{\rm a})`. Each modeled dispersion is a
ratio of two sums that are linear in the weights, so the gradient with respect to the weights is exact and needs no
new table.

.. code-block:: python

   from pianistpy.inference import Data, likelihood

   data = Data("sigma_los", values=sigma_obs, r_2d=radii, errors=sigma_err, unit_system="physical")
   loglike = likelihood(composite.to("physical", concentration=..., mass_0=..., r_0=...), data)
   loglike(weights)                       # a JAX scalar; jax.grad(loglike)(weights) is its gradient

``likelihood`` computes the tables of the observables of each component once. Each later call of the returned
function takes a fraction of a millisecond, except a call at a new anisotropy of a free component, which first
rebuilds the free component's tables. An invalid argument of the returned function, such as a negative weight, a
scale that is not positive, or a length scale that moves a data radius inside the innermost tabulated radius,
gives minus infinity, not an exception, so a Monte Carlo sampler over the weights always receives a value.
A model and data that cannot be compared raise :class:`~pianistpy.InputError` in ``likelihood`` itself, never in a
call of the returned function.

:doc:`differentiability` states what the likelihood is differentiable in, through which observables, and at what accuracy.
