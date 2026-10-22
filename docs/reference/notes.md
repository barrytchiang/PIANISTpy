# Technical notes

Source: https://pianistpy.com/docs/reference/notes

The Technical notes detail the analytical and numerical treatments implemented in PIANISTpy, building upon the high-level overviews provided in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). The Technical notes follow and refer to the notations, equations, and results of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). The Technical notes are organized by the objects of the pipeline, from upstream to downstream, and each object-specific section first treats the analytical theory (whenever applicable) and then the accompanying numerical implementations. [Section T:0](https://pianistpy.com/docs/reference/technical-notes/principles#tn-sec-principles) states the conventions and notations used; we then discuss the input potential-density models ([Section T:1](https://pianistpy.com/docs/reference/technical-notes/model#tn-sec-model)), the distribution function ([Section T:2](https://pianistpy.com/docs/reference/technical-notes/df#tn-sec-df)), the truncation ([Section T:3](https://pianistpy.com/docs/reference/technical-notes/truncation#tn-sec-truncation)), the linear superposition of components ([Section T:4](https://pianistpy.com/docs/reference/technical-notes/sup#tn-sec-sup)), the particle sampling ([Section T:5](https://pianistpy.com/docs/reference/technical-notes/sampling#tn-sec-sampling)), the velocity profiles and projections ([Section T:6](https://pianistpy.com/docs/reference/technical-notes/observables#tn-sec-observables)), and the JAX-based gradients ([Section T:7](https://pianistpy.com/docs/reference/technical-notes/gradients#tn-sec-gradients)). [Section T:8](https://pianistpy.com/docs/reference/technical-notes/verification#tn-sec-verification) documents the accuracy and validation tests and related information.

This document was last updated on October 21, 2026. Section, equation and table numbers of the Technical notes carry the prefix T: (Section T:2.3, Eq. (T:2.4), Table T:2.1). The [Functions by subpackage](https://pianistpy.com/docs/reference/api/index) page gives the form in which the docstrings cite the Technical notes.

- [T:0 Principles and conventions](https://pianistpy.com/docs/reference/technical-notes/principles)

- [T:0.1 Notation](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-1-notation)

- [T:0.2 Thresholds derived from the inputs](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-2-thresholds-derived-from-the-inputs)

- [T:0.3 Closed forms, symbolic derivatives and arbitrary precision](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-3-closed-forms-symbolic-derivatives-and-arbitrary-precision)

- [T:0.4 Limits of double precision and of the energy grid](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-4-limits-of-double-precision-and-of-the-energy-grid)

- [T:0.5 Accuracy tiers, settings levels, and devices](https://pianistpy.com/docs/reference/technical-notes/principles#t-0-5-accuracy-tiers-settings-levels-and-devices)

- [T:1 The potential–density pair and the auxiliary density](https://pianistpy.com/docs/reference/technical-notes/model)

- [T:1.1 The potential–radius relation as P → P₀ and P → 0](https://pianistpy.com/docs/reference/technical-notes/model#t-1-1-the-potentialradius-relation-as-p-p0-and-p-0)

- [T:1.2 Derivatives of the auxiliary density](https://pianistpy.com/docs/reference/technical-notes/model#t-1-2-derivatives-of-the-auxiliary-density)

- [T:1.3 Interpolation of the tabulated truncated profile](https://pianistpy.com/docs/reference/technical-notes/model#t-1-3-interpolation-of-the-tabulated-truncated-profile)

- [T:2 The distribution function](https://pianistpy.com/docs/reference/technical-notes/df)

- [T:2.1 The two forms of the inversion and the forward relation](https://pianistpy.com/docs/reference/technical-notes/df#t-2-1-the-two-forms-of-the-inversion-and-the-forward-relation)

- [T:2.2 Asymptotic behavior of the DF](https://pianistpy.com/docs/reference/technical-notes/df#t-2-2-asymptotic-behavior-of-the-df)

- [T:2.3 Evaluating the inversion at each tabulated energy](https://pianistpy.com/docs/reference/technical-notes/df#t-2-3-evaluating-the-inversion-at-each-tabulated-energy)

- [T:2.4 The energy grid](https://pianistpy.com/docs/reference/technical-notes/df#t-2-4-the-energy-grid)

- [T:2.5 The distribution function outside the tabulated energies](https://pianistpy.com/docs/reference/technical-notes/df#t-2-5-the-distribution-function-outside-the-tabulated-energies)

- [T:2.6 Method II: inversion of a truncated potential–density pair](https://pianistpy.com/docs/reference/technical-notes/df#t-2-6-method-ii-inversion-of-a-truncated-potentialdensity-pair)

- [T:2.7 A distribution function supplied as input](https://pianistpy.com/docs/reference/technical-notes/df#t-2-7-a-distribution-function-supplied-as-input)

- [T:3 Truncation](https://pianistpy.com/docs/reference/technical-notes/truncation)

- [T:3.1 Type II models beyond the anisotropy radius](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-1-type-ii-models-beyond-the-anisotropy-radius)

- [T:3.2 The truncated DF and its density](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-2-the-truncated-df-and-its-density)

- [T:3.3 Solving the truncated Poisson equation](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-3-solving-the-truncated-poisson-equation)

- [T:3.4 Completing the truncated profile toward the center and the truncation radius](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-4-completing-the-truncated-profile-toward-the-center-and-the-truncation-radius)

- [T:3.5 The lowest admissible truncation value](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-5-the-lowest-admissible-truncation-value)

- [T:3.6 Finding the truncation value for a target mass or truncation radius](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-6-finding-the-truncation-value-for-a-target-mass-or-truncation-radius)

- [T:3.7 The closed-form profile of an untruncated model](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-7-the-closed-form-profile-of-an-untruncated-model)

- [T:3.8 Checks of the self-consistency of a model](https://pianistpy.com/docs/reference/technical-notes/truncation#t-3-8-checks-of-the-self-consistency-of-a-model)

- [T:4 Linear superposition of Method II DFs](https://pianistpy.com/docs/reference/technical-notes/sup)

- [T:5 Particle sampling](https://pianistpy.com/docs/reference/technical-notes/sampling)

- [T:5.1 The velocity distribution at a given radius factorizes](https://pianistpy.com/docs/reference/technical-notes/sampling#t-5-1-the-velocity-distribution-at-a-given-radius-factorizes)

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

- [T:8.2 Tests against independent references](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-2-tests-against-independent-references)

- [T:8.3 The roundtrip test](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-3-the-roundtrip-test)

- [T:8.4 The accuracy tables run at every release](https://pianistpy.com/docs/reference/technical-notes/verification#t-8-4-the-accuracy-tables-run-at-every-release)

- [References](https://pianistpy.com/docs/reference/technical-notes/references)
