# Technical notes

Source: https://pianistpy.com/docs/reference/notes

# Technical notes

This technical document details the analytical and numerical treatments implemented in PIANISTpy, building upon the high-level overviews provided in the initial code release paper of Chiang, van den Bosch and Schive (2026d), cited throughout as Chiang+(2026d). Here, we follow and routinely refer to the notations, equations, and results of that paper. This document is organized by the objects of the pipeline, from upstream to downstream, and each object-specific section first treats the analytical theory (whenever applicable) and then the accompanying numerical implementations. Section T:0 states the conventions and notations used; we then discuss the input potential-density models (Section T:1), the distribution function (Section T:2), the truncation (Section T:3), the linear superposition of components (Section T:4), the particle sampling (Section T:5), the velocity profiles and projections (Section T:6), and the JAX-based gradients (Section T:7). Section T:8 documents the accuracy and validation tests and related information.

This is version 7 of the notes. Their numbers carry the prefix T: (sections T:2.3, equations (T:2.4), tables T:2.1); the package’s docstrings cite them in the form the API reference states. The notes are also published as a PDF with each release; the link is added with the 1.0.0 release.

T:0 Principles and conventions
T:0.1 Notation

T:0.2 Rules derived from the inputs, not fitted constants

T:0.3 Exact arithmetic

T:0.4 Host floors

T:0.5 Accuracy tiers, reports, and refusals

T:1 The model: density and potential
T:1.1 The two maps

T:1.2 Derivatives of the auxiliary density

T:1.3 The profile of a truncated model

T:2 The distribution function
T:2.1 The inversion

T:2.2 Asymptotic behavior of the DF

T:2.3 The form of every row

T:2.4 The energy grid

T:2.5 Beyond the table

T:2.6 A component on a tabulated pair (Method II)

T:2.7 A distribution function from a function

T:3 Truncation
T:3.1 Type II models beyond the anisotropy radius

T:3.2 The truncated DF and the Poisson source

T:3.3 The Poisson solve

T:3.4 The fills and the rows of the profile

T:3.5 The gates on the truncation energy

T:3.6 A target mass or a target radius

T:3.7 The radius of trust of an untruncated product

T:3.8 What a model checks

T:4 Superposition

T:5 Particle sampling
T:5.1 The velocity distribution at a given radius factorizes

T:5.2 The table of speeds

T:5.3 The draw

T:5.4 What a draw checks

T:6 Velocity profiles and projections
T:6.1 Intrinsic moments

T:6.2 Projection

T:6.3 Apertures

T:6.4 The line-of-sight velocity distribution

T:6.5 The table of rows and its certificate

T:6.6 The distribution in an aperture

T:6.7 Gauss-Hermite moments

T:6.8 The local speed distribution

T:6.9 Superposed models

T:7 Gradients
T:7.1 The inversion’s rule

T:7.2 What is served

T:7.3 The observables’ tangent rows

T:7.4 The likelihood

T:8 Verification
T:8.1 Reports and tiers

T:8.2 Reference values

T:8.3 The roundtrip test

T:8.4 The accuracy grid

References
