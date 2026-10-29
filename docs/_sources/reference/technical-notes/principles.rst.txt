.. _tn-sec-principles:

T:0 Principles and conventions
==============================

.. _tn-sec-notation:

T:0.1 Notation
--------------

The notation follows |paper| throughout. Unless stated otherwise, all quantities are expressed in the internal units of PIANISTpy, the natural dimensionless units of `Widrow (2000) <https://doi.org/10.1086/317367>`__, in which the scale radius and the central potential of the untruncated profile are both unity. :math:`R` denotes the radius, :math:`P` the relative potential, and :math:`P_0 \equiv P(R = 0)` its central value. Truncated quantities carry a caret (e.g., :math:`\hat{P}` is the truncated potential), except :math:`P_0`, which denotes the central potential in truncated and untruncated models alike; a model truncated at the value :math:`\mathcal{K}_\mathrm{t}` then has :math:`P_0 = \hat{P}(0) = 1 - \mathcal{K}_\mathrm{t}`. We consider DFs of the parametric form :math:`F(\mathcal{K}, \mathcal{L}) = F_\mathcal{K}(\mathcal{K})\,\mathcal{L}^{-2\beta_\mathrm{c}}`, where :math:`\mathcal{K}` is the energy-like variable, :math:`\mathcal{L}` the angular momentum and :math:`\beta_\mathrm{c}` the central anisotropy. The input double-power-law analytic density profiles follow `Zhao (1996) <https://doi.org/10.1093/mnras/278.2.488>`__, as written in |paper|, Eq. (5), whose exponents :math:`(\alpha, \beta, \gamma)` set the sharpness of the transition, the outer slope and the inner slope. :math:`\varrho_\mathrm{aux}` denotes the auxiliary density that enters the Cuddeford inversion, as defined in |paper|, Eq. (B23). The order of the inversion is

.. _tn-eq-order:

.. math::
   :nowrap:

   \begin{equation}
     \nu \equiv \tfrac{3}{2} - \beta_\mathrm{c} = m + n, \qquad m \equiv \lfloor \nu \rfloor, \qquad 0 \leq n < 1 ,
     \tag{T:0.1}
   \end{equation}

where :math:`\nu` is a shorthand of these notes, and :math:`m` and :math:`n` are the integer and the fractional part of :math:`\nu`, respectively; :math:`m` sets the number of derivatives of :math:`\varrho_\mathrm{aux}` that the Cuddeford inversion takes and :math:`n` the exponent of its Abel kernel (|paper|, Eqs. (B9) and (B22); :ref:`Section T:2.1 <tn-sec-forms>`). The exponent :math:`\delta` of the asymptotic expansion of the potential–density pair about the center (termed the "inner series") and the depth :math:`u`, the distance by which :math:`\mathcal{K}` lies below the central potential, are

.. _tn-eq-delta-u:

.. math::
   :nowrap:

   \begin{equation}
     \delta \equiv \frac{\alpha}{2-\gamma}, \qquad u \equiv P_0 - \mathcal{K} .
     \tag{T:0.2}
   \end{equation}

As :math:`\mathcal{K}` approaches :math:`P_0` from below, the floating-point subtraction :math:`P_0 - \mathcal{K}` loses significant digits to cancellation. To circumvent this precision loss, the depth is computed directly, from the inner series (:ref:`Section T:1.2.2 <tn-sec-series>`) and from node differences (:ref:`Section T:2.5.4 <tn-sec-lookup>`).

.. _tn-sec-rules:

T:0.2 Accuracy tiers and settings
---------------------------------

Every result of PIANISTpy carries an accuracy band (termed "tier"), set by the maximum error across all of its checks (:ref:`Section T:8.1 <tn-sec-reports>`) against three thresholds: :math:`10^{-6}`, :math:`10^{-4}`, and :math:`10^{-2}`. The four tiers are "silent" (:math:`\leq 10^{-6}`), "recorded" (:math:`10^{-6}` to :math:`10^{-4}`, noted in the log), "warned" (:math:`10^{-4}` to :math:`10^{-2}`, with a persistent warning in the result and all downstream calculations), and "refused" (:math:`> 10^{-2}`, raised as an error unless explicitly overridden in the settings). Every PIANISTpy output records both its own tier and separately the worst tier among all of its upstream calculations. All other user-facing settings and numerical setups (e.g., the :math:`\mathcal{K}` grid's floor at :math:`10^{-10}`, :ref:`Section T:2.3.2 <tn-sec-floor>`) are decoupled from and do not alter these three thresholds in tier assignment. Any refused result (e.g., a DF negative somewhere, a truncation that yields no finite spatial extent, or a check above :math:`10^{-2}`) is raised as an error that states the cause.

PIANISTpy supports two accuracy settings: "default" (used throughout the :doc:`Representative computing time </start/computing-time>` and :doc:`Accuracy </guide/accuracy>` pages) and "quick," for rapid evaluations at lesser accuracy by reduced node counts and tolerances (see :doc:`Numerical settings </reference/settings>`). The tier-based reports and accuracy checks remain unchanged under both settings. The runtime options for verbosity, timing, cores and the compilation cache leave PIANISTpy outputs identical bit for bit; the choice of CPU-only or GPU-enabled can change the outputs below :math:`10^{-10}` in relative terms, due to a different order of floating-point summation (see :doc:`Runtime options </guide/runtime>`).

.. _tn-sec-exact:

T:0.3 Symbolic derivatives and arbitrary-precision arithmetic
-------------------------------------------------------------

To guard against precision loss in the demanding evaluations (e.g., high-order derivatives at large radii, near-divergent terms of opposite sign at small radii, and the band around :math:`R_\mathrm{a}` of a Type II model), PIANISTpy carries every closed-form expression and its derivatives symbolically, through `SymPy <https://www.sympy.org/>`__, up to the point of numerical evaluation, instead of using finite differences. In the evaluations where double precision loses digits to cancellation in internal benchmarks, and wherever a double-precision value is not finite, PIANISTpy switches to arbitrary-precision floating-point arithmetic through `mpmath <https://mpmath.org/doc/current/>`__. See, for instance, the derivatives of the auxiliary density (:ref:`Section T:1.2.1 <tn-sec-derivatives>`), its central series (:ref:`Section T:2.2.1 <tn-sec-center>`), and the arbitrary-precision references of the test suite (:ref:`Section T:8 <tn-sec-verification>`).
