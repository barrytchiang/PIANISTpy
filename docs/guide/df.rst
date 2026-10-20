Distribution functions
======================

:func:`pianistpy.compute_df`, :class:`pianistpy.DistributionFunction`

``compute_df(component)`` returns the untruncated distribution function of a component, from the analytic
inversion of its density, as a :class:`~pianistpy.DistributionFunction`. This is a mathematical object: it is
returned whether or not it is everywhere non-negative and monotonic, and it records where it is not, together
with the smallest truncation value above which it is. It is also the base of Method I truncation, so a scan over
truncations of one model reuses one inversion.

.. code-block:: python

   halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))
   df = pp.compute_df(halo)

A distribution function from elsewhere, a fitting function or another code's output, is validated the same way:
``DistributionFunction.from_function(component, f)`` takes ``f(k, depth)``, which returns the DF (not its
logarithm) at an array of energies in internal units and their depths below the center,
:math:`P_0 - \mathcal{K}` (``depth``), as the grid carries it (a fit in the depth near the center reads it there; a function of the
energy alone may ignore it), tabulates it on the energy grid ``compute_df`` places for that component
and settings, and continues it beyond the table, signs it and validates it against the density exactly as the
inversion's own table. The function gives F only, so the table has no derivative column: its slopes are the
spline's through F, and the report states that its slope column is not judged. The file stores the table, not
the function, and records the function's name; a value that is not finite is refused, with the first such
energy. Truncation needs the inversion's rows, which a distribution function from a function does not carry:
``truncate`` refuses it, by name.

.. code-block:: python

   import numpy as np
   from scipy.special import beta

   # the isotropic Plummer model's DF in closed form, F = A K^(7/2)
   plummer = pp.Component(pp.DoublePowerLaw.plummer(), pp.Anisotropy())
   amplitude = 3.0 / (4.0 * np.pi * np.sqrt(2.0) * beta(4.5, 1.5))
   df = pp.DistributionFunction.from_function(plummer, lambda k, depth: amplitude * k**3.5)

What the result holds: the tabulated distribution function on its energy grid with its logarithmic slope, the
asymptotic form at the center and at the outer end (how the function continues beyond the table), the sign
report, the inputs and the settings it was computed from, its validation report and its identity. A result can
be written to a file and read back equal (:doc:`files`).

A distribution function is not yet a model: only a :doc:`truncated model <truncation>` can be sampled or
projected.
