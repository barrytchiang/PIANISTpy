Differentiable observables
==========================

:class:`pianistpy.Differentiable`

The object
----------

``pp.Differentiable(model)`` holds a model of one component on a truncated profile (a distribution function inverted
on its parent's profile, as :func:`pianistpy.truncate` makes it) and serves its observables as JAX functions of the
component's anisotropy, :math:`\theta = (\beta_{\rm c}, R_{\rm a})`, passed as ``theta``. Each method takes :math:`\theta` first and mirrors the public call of the
same name, traceable and differentiable in :math:`\theta`:

.. code-block:: python

   d = pp.Differentiable(model)
   d.sigma_r(theta, r)                    # as pp.sigma_r(model at theta, r)
   d.sigma_t(theta, r)                    # as pp.sigma_t
   d.sigma_los(theta, r_2d)               # as pp.sigma_los
   d.sigma_los_aperture(theta, aperture)  # as pp.sigma_los_aperture
   d.losvd(theta, r_2d, v)                # as pp.losvd
   d.pdf_v(theta, r, v)                   # as pp.pdf_v
   d.aperture_losvd(theta, aperture, v)   # as pp.aperture_losvd

Everything is in internal units, whatever the model's unit system: :math:`\theta`, as the likelihood's, the radii and apertures
a method reads, and the velocities it returns. ``d.theta`` is the model's own anisotropy.

The families are those of the distribution function's own rule (:doc:`gradient`): constant anisotropy (:math:`\beta_{\rm c}`; an
isotropic or constant component's :math:`R_{\rm a}` is held at its own, infinite, value with a zero gradient), Cuddeford-Osipkov-Merritt (COM) Type I
(:math:`\beta_{\rm c}` and :math:`R_{\rm a}`), Osipkov-Merritt Type I and Type II :math:`(R_{\rm a})`. Refused at construction, each with an error that names it:
COM Type II (its interior point at :math:`P(R_{\rm a})` moves with both parameters), a superposition or a model of several
components (its components are differentiated through the likelihood), a component that is not spatially finite, and
one not on a truncated profile that carries its parent's distribution function.

What a method computes
----------------------

A method's value at :math:`\theta` is the public call on the model rebuilt at :math:`\theta`: at the model's own :math:`\theta`, the plain call's
own value, to the byte. Its derivative is the observable's tangent rows in :math:`\theta` on the tables' own rows, their
placement frozen, as the likelihood's free components carry it: between two placements the value moves by the tables'
tolerance, which the derivative does not carry. Over an aperture the derivative is the aperture's integral of the
tangent rows of the squared dispersion, on the plain call's own nodes and weights (the integral is linear in the
rows). The line-of-sight velocity distribution, the speed distribution and the aperture's line-of-sight velocity
distribution take theirs through the distribution function's own derivative (:doc:`gradient`), their placement frozen;
a member whose distribution function that rule refuses is named at construction, and these three methods raise with
it. At a :math:`\theta` where the model is refused the value is not a number and its gradient zero. A method is not
differentiable in its radii.

The cost
--------

The first time a method meets a :math:`\theta`, the host rebuilds the model there: the distribution function inverted at :math:`\theta`
on the component's truncated profile, then the dispersion profile and the projected table with their tangent rows.
That is the cost of the likelihood's own rebuild of a free component, about that of the plain call on the model at
:math:`\theta`. The last eight :math:`\theta` are kept, so the value and the gradient at one :math:`\theta` share one rebuild.

The derivative of ``losvd`` and ``aperture_losvd`` costs far more: it goes back through the rows of the line-of-sight
table, whose transpose recomputes each row and differentiates it in :math:`\theta`, one row after another in the
calling process. On a CPU node of 32 cores (AMD Ryzen Threadripper PRO 5975WX), the derivative of ``losvd`` in
:math:`\beta_{\rm c}` at one projected radius and 21 velocities takes 41 minutes with the quick settings (2474 s) and
22 minutes with a smaller table, and the derivative of ``aperture_losvd`` on one circle about 6.5 hours with the
smaller table; the derivative of the example below takes 65 s there. The rows of the table the
read does not touch are not visited; the cost of the rows it touches is reduced in ver. 1.0.0. The values of these
methods are not affected.

Example
-------

The derivative of the line-of-sight dispersion at one projected radius in :math:`\beta_{\rm c}`, by the rule and by a central
difference of the same function:

.. code-block:: python

   import jax
   import numpy as np

   import pianistpy as pp

   parent = pp.solve(
       pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy()), pp.Truncation(k_t=0.05)
   )
   model = pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=0.25)))
   d = pp.Differentiable(model)

   rule = float(jax.grad(lambda t: d.sigma_los(t, 0.3))(d.theta)[0])
   h = np.array([1e-3, 0.0])
   difference = float(d.sigma_los(d.theta + h, 0.3) - d.sigma_los(d.theta - h, 0.3)) / 2e-3
   print(rule, difference)

The two agree to the difference's own error; each of the two shifted :math:`\theta` is one rebuild.
