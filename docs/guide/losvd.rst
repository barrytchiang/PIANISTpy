Line-of-sight velocity distribution
===================================

:func:`pianistpy.losvd`, :class:`pianistpy.LosTable`

``losvd(source, r_2d, v)`` returns the line-of-sight velocity distribution :math:`D(R_{\rm 2D}, V_{\rm los})` at
projected radii: the probability density of the velocity along the line of sight of the tracers seen at a
projected distance :math:`R_{\rm 2D}` from the center. As in |paper|, it is the distribution function
integrated over the line of sight and over the two velocity components on the plane of the sky,

.. math::

   D(R_{\rm 2D}, V_{\rm los}) = \frac{2}{\Sigma(R_{\rm 2D})} \int_{R_{\rm 2D}}^{R_{\rm t}}
   \frac{R\,{\rm d}R}{(R^2 - R_{\rm 2D}^2)^{1/2}} \iint F(\mathcal{Z}, L)\,{\rm d}V_{\perp 1}\,{\rm d}V_{\perp 2},
   \qquad
   \Sigma(R_{\rm 2D}) = 2 \int_{R_{\rm 2D}}^{R_{\rm t}} \frac{\varrho(R)\,R\,{\rm d}R}{(R^2 - R_{\rm 2D}^2)^{1/2}},

with the inner integral over the plane-of-sky velocities at which the truncated distribution function is
positive. It is even in :math:`V_{\rm los}`, integrates to one over all :math:`V_{\rm los}`, and is zero beyond the
escape velocity along the line of sight. Its second moment is :math:`\sigma_{\rm los}^2` (:func:`pianistpy.sigma_los`).

Arguments and units
-------------------

``source`` is a :class:`~pianistpy.Model` or its prepared :class:`~pianistpy.LosTable`; given a model, its table
is built first, so for more than one call build the table once (:doc:`lostable`, which also says how to call it
from a script). ``r_2d`` and ``v`` are arrays
that broadcast against each other: projected radii in the model's unit of length (internal units by default,
parsecs or an Astropy quantity in the physical system) and line-of-sight velocities in its unit of velocity (km/s
in the physical system). The result has their broadcast shape, per unit velocity of the model's unit system.

Of a superposition, each component's distribution is read from its own table and the components are summed with
their weights times their surface densities, :math:`D = \sum_k w_k \Sigma_k D_k / \sum_k w_k \Sigma_k`.

Tiers
-----

The distribution is read from the rows of the table between rows by the same reading the table checked its rows with,
so the tier is that of the table (:doc:`lostable`): each row's norm against the surface density and its second
moment against the projected dispersion, and the table's estimate of its reading between rows. A projected radius
in an interval where the table could not meet its tolerance is refused by name.

Example
-------

.. code-block:: python

   import numpy as np
   import pianistpy as pp

   model = pp.solve(
       pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25)),
       pp.Truncation(k_t=0.05),
       settings=pp.Settings(quick=True),
   )
   table = pp.LosTable(model)

   v = np.linspace(-1.5, 1.5, 3001)
   d = pp.losvd(table, 1.0, v)                    # at R_2D = 1, shape (3001,)
   dv = v[1] - v[0]
   d.sum() * dv                                   # one, to the grid's rule
   np.sqrt((v**2 * d).sum() * dv), pp.sigma_los(table, 1.0)

   pp.losvd(table, np.array([[0.1], [1.0], [10.0]]), v).shape   # (3, 3001): radii by velocities
