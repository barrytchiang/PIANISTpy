.. _tn-sec-sup:

T:4 Linear superposition of Method II DFs
=========================================

Distribution functions inverted from one and the same truncated density, that is, Method II models of one Method I model, combined with positive weights that sum to one, form a single equilibrium model in that density. Members on different truncated profiles are refused, and every component keeps its own energy grid, checked on its own nodes. The anisotropy profile of the superposition is given in |paper|, Eq. (B74). In terms of the radial dispersions :math:`\sigma_{r,i}^2` and the anisotropy profiles :math:`\beta_i` of the members, with weights :math:`c_i > 0` and :math:`\sum_i c_i = 1`, it reads

.. _tn-eq-betasup:

.. math::
   :nowrap:

   \begin{equation}
     \beta_{\rm sup}(R) = \frac{\sum_i c_i\,\beta_i(R)\,\sigma_{r,i}^2(R)}{\sum_i c_i\,\sigma_{r,i}^2(R)} .
     \tag{T:4.1}
   \end{equation}

The range of anisotropy profiles that a superposition can reach is set by three limits.

- *Outer value.* At the truncation radius,

  .. _tn-eq-sup-outer:

  .. math::
     :nowrap:

     \begin{equation}
       \beta_{\rm sup}(R_\mathrm{t}) = \sum_i c_i\,\beta_i(R_\mathrm{t}),
       \tag{T:4.2}
     \end{equation}

  which is a single linear constraint on the weights. In practice this is the limit that binds.

- *Central value.* For a density with central slope :math:`\gamma`, and with :math:`\beta_{\mathrm{c},i}` the central anisotropy of member :math:`i` (zero for the isotropic and OM members), :math:`\beta_{\rm sup}(0)` is the largest :math:`\beta_{\mathrm{c},i}` among the members in use if one of them is at least :math:`\gamma - 1`; if all are below :math:`\gamma - 1`, :math:`\beta_{\rm sup}(0) = (\gamma - 1) - \left[\sum_i c_i/(\gamma - 1 - \beta_{\mathrm{c},i})\right]^{-1}`, which for :math:`\gamma = 1` is the weighted harmonic mean :math:`\left(\sum_i c_i/\beta_{\mathrm{c},i}\right)^{-1}` of the central values.

- *Transition.* The Type I (C)OM profiles change from :math:`10` to :math:`90` percent of their range between :math:`R_\mathrm{a}/3` and :math:`3R_\mathrm{a}`, about one decade in radius; the Type II profiles fall without bound toward :math:`R_\mathrm{a}`. A target profile that changes over a wider range is approached by mixing members with different anisotropy radii.

The composite dispersions are computed from :ref:`Eq. (T:4.1) <tn-eq-betasup>` and the Jeans integral of :ref:`Section T:6.1.1 <tn-sec-jeans-section>` without any sampling. Downstream of this section, a superposition is treated as one model: each of its particles takes a component at random with probabilities equal to the weights and follows the enclosed mass of the mixture (:ref:`Section T:5.3 <tn-sec-draw>`), its projected observables are formed one component at a time and mixed by the weights (:ref:`Section T:6.9 <tn-sec-mixing>`), and the second moments its likelihood compares with the data are linear in the normalized weights, so a new set of weights is evaluated from the members' tabulated moments without recomputing any member (:ref:`Section T:7.4 <tn-sec-likelihood>`).
