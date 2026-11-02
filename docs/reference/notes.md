# Technical notes

Source: https://pianistpy.com/docs/reference/notes

The Technical notes detail the analytical and numerical treatments implemented in PIANISTpy, building upon the high-level overviews provided in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). The Technical notes follow and refer to the notations, equations, and results of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). The Technical notes are organized by the objects of the pipeline, from upstream to downstream, and each object-specific section first treats the analytical theory (whenever applicable) and then the accompanying numerical implementations. [Section T:0](https://pianistpy.com/docs/reference/technical-notes/principles#tn-sec-principles) states the conventions and notations used; we then discuss the input potential–density models ([Section T:1](https://pianistpy.com/docs/reference/technical-notes/model#tn-sec-model)), the distribution function (DF; [Section T:2](https://pianistpy.com/docs/reference/technical-notes/df#tn-sec-df)), the phase-space truncation ([Section T:3](https://pianistpy.com/docs/reference/technical-notes/truncation#tn-sec-truncation)), the linear superposition of DFs ([Section T:4](https://pianistpy.com/docs/reference/technical-notes/sup#tn-sec-sup)), the particle sampling ([Section T:5](https://pianistpy.com/docs/reference/technical-notes/sampling#tn-sec-sampling)), the velocity profiles and projections ([Section T:6](https://pianistpy.com/docs/reference/technical-notes/observables#tn-sec-observables)), and the JAX-based gradients ([Section T:7](https://pianistpy.com/docs/reference/technical-notes/gradients#tn-sec-gradients)). [Section T:8](https://pianistpy.com/docs/reference/technical-notes/verification#tn-sec-verification) documents the accuracy and validation tests and related information.

This document was last updated on November 1, 2026. Section, equation and table numbers of the Technical notes carry the prefix T: (e.g., Section T:2.3, Eq. (T:2.4), Table T:2.1) and are cited as such throughout the website and the docstrings.

- [T:0 Principles and conventions](https://pianistpy.com/docs/reference/technical-notes/principles)

- [T:0.1 Notation](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-1-notation)

- [T:0.2 Accuracy tiers and settings](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-2-accuracy-tiers-and-settings)

- [T:0.3 Symbolic derivatives and arbitrary-precision arithmetic](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-3-symbolic-derivatives-and-arbitrary-precision-arithmetic)

- [T:1 The potential–density pair and the auxiliary density](https://pianistpy.com/docs/reference/technical-notes/model)

- [T:1.1 Asymptotic forms of the potential–radius relation](https://pianistpy.com/docs/reference/technical-notes/model#t-1-1-asymptotic-forms-of-the-potentialradius-relation)

- [T:1.2 Derivatives of the auxiliary density](https://pianistpy.com/docs/reference/technical-notes/model#t-1-2-derivatives-of-the-auxiliary-density)

- [T:1.3 Interpolation of the tabulated truncated profile](https://pianistpy.com/docs/reference/technical-notes/model#t-1-3-interpolation-of-the-tabulated-truncated-profile)

- [T:2 The DF](https://pianistpy.com/docs/reference/technical-notes/df)

- [T:2.1 The general and simplified forms of the Cuddeford inversion](https://pianistpy.com/docs/reference/technical-notes/df#t-2-1-the-general-and-simplified-forms-of-the-cuddeford-inversion)

- [T:2.2 Asymptotic behavior of the DF](https://pianistpy.com/docs/reference/technical-notes/df#t-2-2-asymptotic-behavior-of-the-df)

- [T:2.3 Evaluating the inversion on the grid of the energy-like variable](https://pianistpy.com/docs/reference/technical-notes/df#t-2-3-evaluating-the-inversion-on-the-grid-of-the-energy-like-variable)

- [T:2.4 The grid of the energy-like variable](https://pianistpy.com/docs/reference/technical-notes/df#t-2-4-the-grid-of-the-energy-like-variable)

- [T:2.5 The DF beyond the ends of its grid](https://pianistpy.com/docs/reference/technical-notes/df#t-2-5-the-df-beyond-the-ends-of-its-grid)

- [T:2.6 Method II: inversion of a truncated potential–density pair](https://pianistpy.com/docs/reference/technical-notes/df#t-2-6-method-ii-inversion-of-a-truncated-potentialdensity-pair)

- [T:2.7 A user-supplied DF](https://pianistpy.com/docs/reference/technical-notes/df#t-2-7-a-user-supplied-df)

- [T:3 Phase-space truncation](https://pianistpy.com/docs/reference/technical-notes/truncation)

- [T:3.1 (C)OM Type II models beyond the anisotropy radius](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-1-c-om-type-ii-models-beyond-the-anisotropy-radius)

- [T:3.2 The truncated DF and its density](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-2-the-truncated-df-and-its-density)

- [T:3.3 Solving the truncated Poisson equation](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-3-solving-the-truncated-poisson-equation)

- [T:3.4 Completing the profile at the center and at the truncation boundary](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-4-completing-the-profile-at-the-center-and-at-the-truncation-boundary)

- [T:3.5 The lowest admissible truncation value](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-5-the-lowest-admissible-truncation-value)

- [T:3.6 Finding the truncation value for a target mass or truncation radius](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-6-finding-the-truncation-value-for-a-target-mass-or-truncation-radius)

- [T:3.7 The closed-form profile of an untruncated model](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-7-the-closed-form-profile-of-an-untruncated-model)

- [T:3.8 Self-consistency checks](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-8-self-consistency-checks)

- [T:4 Linear superposition of Method II DFs](https://pianistpy.com/docs/reference/technical-notes/sup)

- [T:5 Particle sampling](https://pianistpy.com/docs/reference/technical-notes/sampling)

- [T:5.1 Factorization of the velocity distribution at a given radius](https://pianistpy.com/docs/reference/technical-notes/sampling#t-5-1-factorization-of-the-velocity-distribution-at-a-given-radius)

- [T:5.2 Tabulated inverse CDFs](https://pianistpy.com/docs/reference/technical-notes/sampling#t-5-2-tabulated-inverse-cdfs)

- [T:5.3 Drawing particles by inverse-CDF sampling](https://pianistpy.com/docs/reference/technical-notes/sampling#t-5-3-drawing-particles-by-inverse-cdf-sampling)

- [T:5.4 Checks of the drawn particles and of the written files](https://pianistpy.com/docs/reference/technical-notes/sampling#t-5-4-checks-of-the-drawn-particles-and-of-the-written-files)

- [T:6 Velocity distributions and dispersion profiles](https://pianistpy.com/docs/reference/technical-notes/observables)

- [T:6.1 3D velocity dispersion profiles](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-1-3d-velocity-dispersion-profiles)

- [T:6.2 Surface density and line-of-sight velocity dispersion](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-2-surface-density-and-line-of-sight-velocity-dispersion)

- [T:6.3 Aperture-averaged velocity dispersions](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-3-aperture-averaged-velocity-dispersions)

- [T:6.4 The line-of-sight velocity distribution](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-4-the-line-of-sight-velocity-distribution)

- [T:6.5 Tabulating the line-of-sight velocity distribution over projected radius](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-5-tabulating-the-line-of-sight-velocity-distribution-over-projected-radius)

- [T:6.6 The aperture-averaged line-of-sight velocity distribution](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-6-the-aperture-averaged-line-of-sight-velocity-distribution)

- [T:6.7 Gauss–Hermite moments](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-7-gausshermite-moments)

- [T:6.8 The local velocity distribution](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-8-the-local-velocity-distribution)

- [T:6.9 Velocity distributions and dispersions of a linear superposition](https://pianistpy.com/docs/reference/technical-notes/observables#t-6-9-velocity-distributions-and-dispersions-of-a-linear-superposition)

- [T:7 Gradients](https://pianistpy.com/docs/reference/technical-notes/gradients)

- [T:7.1 Derivatives of the DF with respect to the anisotropy parameters](https://pianistpy.com/docs/reference/technical-notes/gradients#t-7-1-derivatives-of-the-df-with-respect-to-the-anisotropy-parameters)

- [T:7.2 Anisotropy models and parameters with derivatives, and the refused cases](https://pianistpy.com/docs/reference/technical-notes/gradients#t-7-2-anisotropy-models-and-parameters-with-derivatives-and-the-refused-cases)

- [T:7.3 Derivatives of the velocity dispersions and distributions](https://pianistpy.com/docs/reference/technical-notes/gradients#t-7-3-derivatives-of-the-velocity-dispersions-and-distributions)

- [T:7.4 The likelihood](https://pianistpy.com/docs/reference/technical-notes/gradients#t-7-4-the-likelihood)

- [T:8 Verification](https://pianistpy.com/docs/reference/technical-notes/verification)

- [T:8.1 The checks of each validation report](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-1-the-checks-of-each-validation-report)

- [T:8.2 Tests against closed forms and arbitrary-precision references](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-2-tests-against-closed-forms-and-arbitrary-precision-references)

- [T:8.3 The roundtrip test](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-3-the-roundtrip-test)

- [T:8.4 The accuracy tables at each release](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-4-the-accuracy-tables-at-each-release)

- [References](https://pianistpy.com/docs/reference/technical-notes/references)
