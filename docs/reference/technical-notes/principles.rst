.. _tn-sec-principles:

T:0 Principles and conventions
==============================

.. _tn-sec-notation:

T:0.1 Notation
--------------

We adopt the notation of |paper|. The relative potential is :math:`P`, with the central value :math:`P_0`; the energy-like variable is :math:`\mathcal{K}`; the distribution function (DF) is :math:`F(\mathcal{K}, L) = F_\mathcal{K}(\mathcal{K})\,L^{-2\beta_\mathrm{c}}`; the density is the double power law of |paper|, Eq. (5) with exponents :math:`(\alpha, \beta, \gamma)`; and :math:`\varrho_\mathrm{aux}` is the auxiliary density, which for the (C)OM models is given by |paper|, Eq. (B23). Three derived quantities are used throughout these notes. The first is the order of the inversion,

.. _tn-eq-order:

.. math::
   :nowrap:

   \begin{equation}
     \nu \equiv \tfrac{3}{2} - \beta_\mathrm{c} = m + n, \qquad m \equiv \lfloor \nu \rfloor, \qquad 0 \leq n < 1 ,
     \tag{T:0.1}
   \end{equation}

and the other two are the exponent of the inner expansion and the depth below the center,

.. _tn-eq-delta-u:

.. math::
   :nowrap:

   \begin{equation}
     \delta \equiv \frac{\alpha}{2-\gamma}, \qquad u \equiv P_0 - \mathcal{K} .
     \tag{T:0.2}
   \end{equation}

Internally, every model is expressed in the units of |paper|, in which the scale radius and the central potential of the untruncated profile are both unity; a truncated model therefore has :math:`P_0 = 1 - \mathcal{K}_\mathrm{t}`. The depth :math:`u` is carried as a variable of its own wherever the difference :math:`P_0 - \mathcal{K}` would lose significant digits.

.. _tn-sec-rules:

T:0.2 Thresholds derived from the inputs
----------------------------------------

Every threshold that judges a result is one of the three levels of :ref:`Section T:0.5 <tn-sec-tiers>` or a stated fraction of :math:`10^{-6}`, and it is derived from the problem at hand wherever the problem gives one; the method's remaining fixed numbers, such as the floor and the gap of the energy grid (:ref:`Section T:0.4 <tn-sec-host>`), are its own and change with no settings level. For example, the depth at which a table ends is the depth at which the exact derivatives leave the range of a double (:ref:`Section T:2.5.1 <tn-sec-top>`); the radius out to which the inner series is trusted is the radius at which its next term reaches the tolerance (:ref:`Section T:1.2.2 <tn-sec-series>`). A settings level changes only node counts and tolerances; of the runtime options, verbosity, timing and cores change no number, and the device can, since the order of summation on a GPU is not the CPU's (:ref:`Section T:7.3 <tn-sec-tangent-rows>`). Whatever cannot be computed is refused by an error that states the reason; a refused truncation carries the DF it started from, and a model refused after it is built carries the model, as the error's result. Nothing is silently approximated.

.. _tn-sec-exact:

T:0.3 Closed forms, symbolic derivatives and arbitrary precision
----------------------------------------------------------------

Wherever a quantity has a closed form, we use that closed form, and we form its derivatives symbolically rather than by finite differences. The derivatives of the auxiliary density with respect to the potential are built by computer algebra from the closed-form density, enclosed mass, and anisotropy factor (:ref:`Section T:1.2.1 <tn-sec-derivatives>`). They are evaluated in double precision where that evaluation is stable, and in arbitrary precision where it suffers from cancellation: at small and at large radii, at high orders far out, and within a band around :math:`R_\mathrm{a}` for a Type II model. The coefficients of the central series of :math:`\varrho_\mathrm{aux}` are exact (:ref:`Section T:2.2.1 <tn-sec-center>`); the amplitude of the central term of the DF is a closed form; and the enclosed mass is the incomplete Beta function, evaluated in arbitrary precision where its closed form fails. Every exact form is checked against an arbitrary-precision reference in the test suite (:ref:`Section T:8 <tn-sec-verification>`).

.. _tn-sec-host:

T:0.4 Limits of double precision and of the energy grid
-------------------------------------------------------

Two limits of double precision are read from the host rather than assumed, and the energy grid has fixed ends of its own. The energy grid has a floor at :math:`q_0 = 10^{-10}` in units of :math:`P_0`, and it ends at a gap of :math:`10^{-9}` below :math:`P_0`, which is the depth at which energies are still separable from the center; no node lies above this gap, and the closed form of the center carries the DF beyond the last node (:ref:`Section T:2.5.1 <tn-sec-top>`). The top of a table may end deeper still, at the deepest of three depths: the depth at which the exact derivatives needed by the inversion exceed the largest double, the depth at the smallest radius whose square the host can represent (:math:`2^{-537}` with subnormals), and the depth at which the closed form of the center exceeds :math:`10^{300}`. All three are computed from the model and the host alone. Near the center, the depth is formed without subtraction, from the inner series (:ref:`Section T:1.2.2 <tn-sec-series>`) and, in the lookup, from node differences that are exact by Sterbenz's lemma (:ref:`Section T:2.5.4 <tn-sec-lookup>`).

.. _tn-sec-tiers:

T:0.5 Accuracy tiers, settings levels, and devices
--------------------------------------------------

Every product is judged against three levels of relative error: :math:`10^{-6}`, :math:`10^{-4}`, and :math:`10^{-2}`. A check at most :math:`10^{-6}` is silent. A check above :math:`10^{-6}` and at most :math:`10^{-4}` is recorded: its tier stays attached to the result and one line is written to the log. A check above :math:`10^{-4}` is returned with a warning that stays attached to the result and to everything computed from it. A check beyond :math:`10^{-2}` refuses the result, unless the settings explicitly ask for it. The four tiers are silent, recorded, warned, and refused. Every result carries a report with a fixed set of checks (:ref:`Section T:8.1 <tn-sec-reports>`); a derived result holds its own checks together with the worst tier of the results it was computed from, never a copy of their reports. With the default settings, a component of a product that feeds a projected quantity is held to a fraction of :math:`10^{-6}`, :math:`10^{-7}` or :math:`10^{-8}` as stated where it is used, which is its share of the error budget of what is computed from it.

Two settings levels are tested, the default settings and the quick settings for exploration. Both report the same checks, the test suite runs both, and the quick settings change only node counts and tolerances. The CPU-only mode gives the same numbers on every machine, and every result file records the versions and the backend of the session that wrote it. A stage whose wall time scales with a sequential count (the steps of a solve, or rows computed one at a time) is latency-bound on a GPU, and it therefore stays on the CPU on every device; a stage whose wall time scales with a batch (the velocity table, the particle draw) runs on a GPU when one is present. The rows of the line-of-sight tables are computed in a pool of CPU processes, each process taking one row at a time. This classification rests on measured scaling, with no margin constant.
