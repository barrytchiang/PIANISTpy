.. _tn-sec-sup:

T:4 Linear superposition of Method II DFs
=========================================

.. rst-class:: tn-body

Under the same input potential–density pair, DFs are additive in phase space (|paper|, Appendix B). Method II DFs automatically satisfy this condition, as they derive from the identical already truncated base model; a set of them can then be linearly superposed to form a single equilibrium model in that density (|paper|, Section 3.2). Input DFs corresponding to distinct potential–density pairs, whether physically different or computed with different settings, are thus refused by PIANISTpy.

The composite dispersions are the weighted sums of the members' dispersions from the dispersion integral (:ref:`Section T:6.1.1 <tn-sec-jeans-section>`), and the composite anisotropy follows as in |paper|, Eq. (B74). Each superposition product is treated as "one model" in the downstream numerical evaluations of PIANISTpy. :math:`N`-body initial conditions assign each particle to a member DF probabilistically, by the weights (:ref:`Section T:5.3 <tn-sec-draw>`). Projected velocity observables are first evaluated for each member DF and then summed with the weights, exactly, given the shared surface density (:ref:`Section T:6.9 <tn-sec-mixing>`). In Monte Carlo fitting, once all members' tabulated moments are computed, every subsequent evaluation of the composite observables under new weights reuses these "bases" without recomputation (:ref:`Section T:7.4 <tn-sec-likelihood>`).
