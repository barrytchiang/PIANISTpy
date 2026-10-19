# Release notes

Source: https://pianistpy.com/docs/project/changelog

# Release notes

What each version adds, changes and fixes, newest first. A change that moves a number beyond rounding is named
with the size of the move; the standard and the minimum requirements are listed with each release.

## 1.0.0

Draft: this entry is final at the release.

The first release, with the paper that describes the method (Chiang, van den Bosch and Schive 2026, MNRAS,
submitted). Everything 0.9.0 announced for this version is in.

### Added

The certified accuracy tiers (goal, ceiling, warning, refused) reported with every result, on the CPU and on
the GPU; the accuracy grid of the release is on Accuracy tables, and every release’s on
Accuracy by release.

The observables as calls of their own: sigma_r, sigma_t, losvd, aperture_losvd and pdf_v
(the local speed distribution at a 3D radius), beside sigma_los and sigma_los_aperture; every
observable carries its tier.

Ranged draws: a range of radii for Sampler (r_range), a range of speeds at given radii
(Sampler.speeds, v_range), and the projected sampler of (R, v_los) pairs (ProjectedSampler).

The likelihood of the inference is differentiable in the superposition weights, the unit scales and the
anisotropy parameters (beta_c, R_a) of the constant, Cuddeford Type I and Osipkov-Merritt families; and,
through pp.Differentiable, the seven observables as JAX functions of those parameters
(End-to-end differentiability).

The inference fits a superposition to Gauss-Hermite moments (h3, h4), at points or averaged over
apertures, beside the binned data; a free component with moment data is refused, as with binned data.

### Changed

The lowered center’s map evaluates its two large, nearly equal terms with expm1 and log1p: the center
now converges in 20 iterations with no floor; no value of the accuracy ledger moves beyond 1.8e-15.

The Gauss-Hermite moments’ edge rule is set by the moment’s own scale (the superposed moments’ edge points move
by at most 3.7e-10).

Every inner cut of the distribution function’s solve is taken from the inputs (no fixed constant).

The anisotropy factor and its radial derivatives have one form, read by every part of the solve; a Type II
radius at or beyond R_a is refused by name where small clamps stood before, and a Type II truncation whose edge
lies at or beyond R_a is refused after the solve (TYPE II EDGE).

The smallest truncation energy of a Type II distribution function bounded by its domain edge (k_t_min) is
the next float above P(R_a), the smallest the strict gate admits.

The slope check of an untruncated Type II distribution function leaves out the first node of its support, where
the function diverges as (K - P(R_a))^(-1/2) and a one-sided difference misreads it by 2.2e-3; the record
states the node left out. Seven models of the accuracy ledger move from warning to goal; no tolerance changes.

The density round trip of a truncated model is read on every row of the edge approach, where a sampling of
radii could miss a peak a few rows wide.

The observables’ worker processes run on the CPU whatever the platform of the calling process.

The minimum versions of the dependencies are found by test, one package at a time and then as one environment,
and listed on Install (NumPy 2.3, SciPy 1.15, h5py 3.11, Astropy 6.1, JAX 0.6, Python 3.10);
Astropy below 7.2 needs NumPy below 2.4. filelock is no longer a dependency.

The examples of the observables’ guide pages use a Hernquist model with a constant anisotropy of 0.25, and
The prepared table states the cost of its example.

The rows of the line-of-sight table are computed in blocks sized to the per-core cache (n_block_elements
25000, was 250000): the rows are the same to the bit, and the table builds in about half the wall time.

### Fixed

A truncation whose solve reaches its outer radius with no edge is refused by name (NO EDGE, with the radius,
the potential and its slope there); before, it was returned as a truncated model. A component on a parent
profile that is not spatially finite is refused by name.

At an inner slope near 2 with a positive beta_c, the inversion’s table reaches the depth the host’s smallest
radius allows, and these models build where they were refused; the validity certificate is judged at that
depth. A central series that degenerates (gamma = 2 beta_c at the center) is refused by name
(CENTRAL SERIES).

At an inner slope near 2 the inner extension’s radius no longer underflows to zero; where it lies below the
double’s range the solve is refused by name.

Cuddeford Type II models at an inner slope near 2 no longer fail in the inversion with non-finite slopes.

A cusp at the marginal anisotropy 2 beta_c = gamma solves its central deficit: on NFW with beta_c = 0.5 the
density from the distribution function moves from 7.7e-4 to 3.2e-9 of the profile (warning to goal).

The line-of-sight velocity distribution near the center reads the distribution function by the solve’s depth:
one unit in the last place of the potential moved it by 1.1e-4 at R_2D = 1e-9, now by 1.6e-15 or less.

The lowest row of the truncated density continues the edge law (it read 2.5e-4 above it).

DistributionFunction.from_function reads the function below the written floor as compute_df does: the
Plummer and Hernquist closed forms certify at the goal (round trip 2.0e-14 and 4.6e-11, where 1.0 before).

The particle sampler’s speed table reads its rows at their exact depths and its first slope from the rule below
it; on the speed tables of the Fig. 7 models with beta_c = 1/2 the worst row falls from 4.1e-4 to 1.7e-6. Where
the table stops at its row cap above the goal, its log line and the sampler say why.

The pilot grids keep one node at their seam (two nodes 7e-18 apart at q = 0.05 before).

A product of the earlier pipeline whose potential does not strictly decrease is refused when it is loaded, with
the first two rows where it rises and the number of such steps.

The critical band carries the parent’s central deficit to its cubic term, each term from the deficit’s own
integral, and its range reads the deficit’s truncation: where the band meets the rows on the
Fig. 7 children with beta_c = 1/2 the step in log10 F falls from 2.5e-6 to 7.9e-9 or less; one member of the
accuracy ledger moves (its roundtrip median by 2.3 percent), and no tier changes.

### Known limitations

One component per model at the public calls; several components in one potential are planned
(Upcoming features).

The gradient in the anisotropy parameters refuses Cuddeford Type II, whose interior point moves with R_a.

The gradient in the anisotropy parameters refuses a Method I truncated model, whose profile moves with them; a
Method II model on a fixed parent takes it (End-to-end differentiability).

On a GPU, the derivative of losvd and aperture_losvd may refuse a row of the line-of-sight table by name
(its transpose holds the row recomputed to the forward’s bit for bit); it never returns a wrong number
(End-to-end differentiability).

Spherical, non-rotating systems with an analytical double-power-law parent profile.

Requirements: the standard and the minimum lists on Install.

## 0.9.0

A pre-release with the whole public interface: the distribution functions and their truncation
(compute_df, truncate, solve), the particles, the observables, the superposition and the inference.
Version 1.0.0, released with the paper that describes the method, adds the certified accuracy tiers on both
backends, the remaining capabilities and the citation.

Standard requirements (the versions this release was developed and tested with): Python 3.11.13, NumPy
2.4.3, SciPy 1.17.1, h5py 3.15.1, Astropy 7.2.0, mpmath 1.3.0, SymPy 1.14.0, JAX 0.9.2, Diffrax 0.7.2,
Optimistix 0.1.0, filelock 4.0.12.

Minimum requirements (the oldest versions that pass the full test suite, measured one package at a time):
Python 3.10; NumPy 2.0.0; SciPy 1.14.0; h5py 3.11.0; Astropy 7.2.0; mpmath 0.17; SymPy 1.2; JAX 0.6.1 with
Diffrax 0.7.0 and Optimistix 0.0.11 (these three are bound together); filelock. On Python 3.10 the newest JAX
is 0.6.2 (JAX 0.7 and later need Python 3.11). GPU: jax[cuda12] from 0.4.38 passes the suite on a GPU node;
the declared bound is the joint one above. Python 3.12 and 3.13 pass.

     On this page
