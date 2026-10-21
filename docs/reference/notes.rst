Technical notes
===============



The Technical notes detail the analytical and numerical treatments implemented in PIANISTpy, building upon the high-level overviews provided in |paper|. The Technical notes follow and refer to the notations, equations, and results of |paper|. The Technical notes are organized by the objects of the pipeline, from upstream to downstream, and each object-specific section first treats the analytical theory (whenever applicable) and then the accompanying numerical implementations. :ref:`Section T:0 <tn-sec-principles>` states the conventions and notations used; we then discuss the input potential-density models (:ref:`Section T:1 <tn-sec-model>`), the distribution function (:ref:`Section T:2 <tn-sec-df>`), the truncation (:ref:`Section T:3 <tn-sec-truncation>`), the linear superposition of components (:ref:`Section T:4 <tn-sec-sup>`), the particle sampling (:ref:`Section T:5 <tn-sec-sampling>`), the velocity profiles and projections (:ref:`Section T:6 <tn-sec-observables>`), and the JAX-based gradients (:ref:`Section T:7 <tn-sec-gradients>`). :ref:`Section T:8 <tn-sec-verification>` documents the accuracy and validation tests and related information.



This document was last updated on October 21, 2026. Section, equation and table numbers of the Technical notes carry the prefix T: (Section T:2.3, Eq. (T:2.4), Table T:2.1). The :doc:`Functions by subpackage <api/index>` page gives the form in which the docstrings cite the Technical notes.

.. toctree::
   :maxdepth: 2

   technical-notes/principles
   technical-notes/model
   technical-notes/df
   technical-notes/truncation
   technical-notes/sup
   technical-notes/sampling
   technical-notes/observables
   technical-notes/gradients
   technical-notes/verification
   technical-notes/references
