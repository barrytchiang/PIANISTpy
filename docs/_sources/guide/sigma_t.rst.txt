Tangential velocity dispersion
==============================

:func:`pianistpy.sigma_t`, :class:`pianistpy.LosTable`

``sigma_t(source, r)`` returns the intrinsic tangential velocity dispersion :math:`\sigma_{\rm t}` at user-specified 3D radii.
:math:`\sigma_{\rm t}^2` is the sum of the squared dispersions of the two tangential velocity components,
:math:`\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2`, with :math:`\theta` and :math:`\phi` the polar and the
azimuthal directions. As in |paper|, :math:`\sigma_{\rm t}^2` is the tangential velocity moment of the augmented
density,

.. math::

   \sigma_{\rm t}^2(R) = \frac{2}{\varrho_{\rm aug}(P, R)} \int_0^{P}
   \frac{\partial}{\partial R^2}\Big[R^2 \varrho_{\rm aug}(\mathcal{K}, R)\Big]\,{\rm d}\mathcal{K},
   \qquad P = P(R).

The velocity anisotropy is :math:`\beta_\sigma = 1 - \sigma_{\rm t}^2 / (2\sigma_{\rm r}^2)`. :math:`\beta_\sigma`
is zero for an isotropic model, :math:`\beta_{\rm c}` for a constant-anisotropy model, and
:math:`R^2 / (R^2 + R_{\rm a}^2)` for an Osipkov-Merritt Type I model. The package evaluates :math:`\sigma_{\rm t}^2 = 2(1 - \beta_\sigma)\sigma_{\rm r}^2`,
with :math:`\sigma_{\rm r}` from :doc:`sigma_r` and :math:`\beta_\sigma` in closed form from the model's anisotropy.

Arguments and units
-------------------

As for :doc:`sigma_r`, ``source`` is a :class:`~pianistpy.Model` or the model's line-of-sight table
(:class:`~pianistpy.LosTable`), ``r`` is in the model's unit system, and the result has the shape of ``r``, in the
model's velocity unit. For a superposition,
:math:`\sigma_{\rm t}^2 = \sum_k w_k \sigma_{{\rm t},k}^2`.

Accuracy
--------

``sigma_t`` has the accuracy of ``sigma_r`` (:doc:`sigma_r`), because ``sigma_t`` reads the radial dispersion
profile of ``sigma_r``. The factor :math:`2(1 - \beta_\sigma)` is computed in closed form from the model's
anisotropy.

Example
-------

The anisotropy computed from the two dispersions is the model's own, here the constant
:math:`\beta_{\rm c} = 0.25`.

.. code-block:: python

   import numpy as np
   import pianistpy as pp

   model = pp.solve(
       pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
       pp.Truncation(k_t=0.05),
       settings=pp.Settings(quick=True),
   )
   r = np.array([0.1, 1.0, 10.0])
   beta = 1.0 - pp.sigma_t(model, r) ** 2 / (2.0 * pp.sigma_r(model, r) ** 2)
   np.max(np.abs(beta - 0.25))   # of the order of the floating-point rounding error
