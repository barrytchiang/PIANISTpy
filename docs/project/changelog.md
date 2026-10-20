# Release notes

Source: https://pianistpy.com/docs/project/changelog

# Release notes

What each version adds, changes and fixes, newest first. A change that moves a number beyond rounding is named
with the size of the move; the standard and the minimum dependencies are listed with each release.

## Ver. 1.0.0

Draft: this entry is final at the release.

The first stable release: the code of ver. 0.9.9 with the citation entry of Chiang+(2026d) and the DOI of the software
release. Changes to the code since ver. 0.9.9 are listed here at the release.

## Ver. 0.9.9

Draft: this entry is final at the release.

The code of the paper that describes the method, Chiang+(2026d) (MNRAS, submitted), released with its arXiv
submission. Ver. 1.0.0 adds the citation entry.

### Added

A report of the numerical accuracy with every result, on the CPU and on the GPU: the relative error of each
check against the goal of \(10^{-6}\), the ceiling of \(10^{-4}\) and the hard limit of \(10^{-2}\)
(Accuracy); the accuracy grid of the release is on Accuracy tables, and every release’s on
Accuracy by release.

The observables as calls of their own: sigma_r, sigma_t, losvd, aperture_losvd and pdf_v
(the local speed distribution at a 3D radius), beside sigma_los and sigma_los_aperture; every
observable carries its tier.

Ranged draws: a range of radii for Sampler (r_range), a range of speeds at given radii
(Sampler.speeds, v_range), and the projected sampler of \((R, V_{\rm los})\) pairs (ProjectedSampler).

The likelihood of the inference is differentiable in the superposition weights, the unit scales and the
anisotropy parameters \((\beta_{\rm c}, R_{\rm a})\) of the constant, Cuddeford-Osipkov-Merritt (COM) Type I and Osipkov-Merritt families; and,
through pp.Differentiable, the seven observables as JAX functions of those parameters
(End-to-end differentiability).

The inference fits a superposition to Gauss-Hermite moments (h3, h4), at points or averaged over
apertures, beside the binned data; a free component with moment data is refused, as with binned data.

likelihood, Data, ProjectedSampler and Particles are exported by the package; the gradient
functions (pianistpy.df.gradient) and pianistpy.observables.stages.speed_distribution have entries in
the API reference.

The package includes its type information (py.typed) and its test suite, run on an installation with
pytest --pyargs pianistpy (the test extra installs pytest); the repository has a citation file.

### Changed

The lowered center’s map evaluates its two large, nearly equal terms with expm1 and log1p: the center
now converges in 20 iterations with no floor; no value of the accuracy ledger moves beyond \(1.8 \times 10^{-15}\).

The Gauss-Hermite moments’ edge rule is set by the moment’s own scale (the superposed moments’ edge points move
by at most \(3.7 \times 10^{-10}\)).

Every inner cut of the distribution function’s solve is taken from the inputs (no fixed constant).

The anisotropy factor and its radial derivatives have one form, read by every part of the solve; a Type II
radius at or beyond \(R_{\rm a}\) is refused by name where small clamps stood before, and a Type II truncation whose edge
lies at or beyond \(R_{\rm a}\) is refused after the solve (TYPE II EDGE).

The smallest truncation energy of a Type II distribution function bounded by its domain edge (k_t_min) is
the next float above \(P(R_{\rm a})\), the smallest the strict gate admits.

The slope check of an untruncated Type II distribution function leaves out the first node of its support, where
the function diverges as \((\mathcal{K} - P(R_{\rm a}))^{-1/2}\) and a one-sided difference misreads it by \(2.2 \times 10^{-3}\); the record
states the node left out. Seven models of the accuracy ledger move from warning to goal; no tolerance changes.

The density round trip of a truncated model is read on every row of the edge approach, where a sampling of
radii could miss a peak a few rows wide.

The observables’ worker processes run on the CPU whatever the platform of the calling process.

The minimum versions of the dependencies are found by test, one package at a time and then as one environment,
and listed on Install (NumPy 2.3, SciPy 1.15, h5py 3.11, Astropy 6.1, JAX 0.6, Python 3.11);
Astropy below 7.2 needs NumPy below 2.4. filelock is no longer a dependency.

The examples of the observables’ guide pages use a Hernquist model with a constant anisotropy of 0.25, and
The prepared table states the cost of its example.

The rows of the line-of-sight table are computed in blocks sized to the per-core cache (n_block_elements
25000, was 250000): the rows are the same to the bit, and the table builds in about half the wall time.

The rows of the line-of-sight table take 12 sightline nodes a panel (n_sightline_nodes, was 16): a row
moves by at most \(3.7 \times 10^{-9}\) of its peak (\(4.2 \times 10^{-7}\) at the innermost rows of the
\(\beta_{\rm c} = -0.5\) model of Fig. 15, where 16 and 24 nodes differ by as much), the central and aperture
distributions by at most \(3.7 \times 10^{-10}\), pdf_v and
sigma_los not at all; the anisotropic tables of Fig. 15 build in 14 to 18 percent less wall time.

Python 3.11 is the oldest supported (NumPy 2.3, the oldest NumPy that passes the suite, needs it); JAX is
capped below 0.10.2, on which one test of the suite fails.

One exception class per kind of refusal: AccuracyError carries the refused model (result), as
UnphysicalModelError carries its DF; the sampler’s refusals beyond its limits are AccuracyError (were
ResultError), a model of infinite mass at its sampling UnphysicalModelError, and an existing file at
write InputError for every writer.

A result file is read back after it is written and refused unless it holds what was written, to the bit, as
the particle files are.

A report read from a file is checked against the reports of its kind: a check no such report has is refused
by name, and a check the file lacks (an earlier version made fewer) reads as not judged, so a ver. 0.9.0 file loads
and says what it lacks. An untruncated model’s outer join and eps, which its closed-form profile does not
measure, are not judged (they read 0 before).

Anisotropy stores kind="I" where r_a is infinite, since the type does not enter the model there,
so the same model has one identity.

pianistpy.numerics no longer exports bell_partitions and derivative_from_log, which nothing in the
package used.

The rows of the line-of-sight table end at the first row beyond which less than losvd.outer_mass_share of
the total projected mass lies (the accuracy goal by default; 0 keeps them to the model’s edge), and start at the
profile’s innermost row unless losvd.innermost_read_radius names the smallest projected radius the table will
be read at (The prepared table). Every row kept is the row the full table holds at its radius.

### Fixed

A read of the line-of-sight table beyond its last row or inside its first, at a point, was extrapolated from
the nearest rows; it is refused by name, with the row’s radius and the setting that moves it, as the annulus
read was.

A truncation whose solve reaches its outer radius with no edge is refused by name (NO EDGE, with the radius,
the potential and its slope there); before, it was returned as a truncated model. A component on a parent
profile that is not spatially finite is refused by name.

At an inner slope near 2 with a positive \(\beta_{\rm c}\), the inversion’s table reaches the depth the host’s smallest
radius allows, and these models build where they were refused; the distribution function’s validity is checked
at that depth. A central series that degenerates (\(\gamma = 2\beta_{\rm c}\) at the center) is refused by name
(CENTRAL SERIES).

At an inner slope near 2 the inner extension’s radius no longer underflows to zero; where it lies below the
double’s range the solve is refused by name.

COM Type II models at an inner slope near 2 no longer fail in the inversion with non-finite slopes.

A cusp at the marginal anisotropy \(2\beta_{\rm c} = \gamma\) solves its central deficit: on NFW with \(\beta_{\rm c} = 0.5\) the
density from the distribution function moves from \(7.7 \times 10^{-4}\) to \(3.2 \times 10^{-9}\) of the profile (warning to goal).

The line-of-sight velocity distribution near the center reads the distribution function by the solve’s depth:
one unit in the last place of the potential moved it by \(1.1 \times 10^{-4}\) at \(R_{\rm 2D} = 10^{-9}\), now by \(1.6 \times 10^{-15}\) or less.

The lowest row of the truncated density continues the edge law (it read \(2.5 \times 10^{-4}\) above it).

DistributionFunction.from_function reads the function below the written floor as compute_df does: the
Plummer and Hernquist closed forms are read back with a density round trip of \(2.0 \times 10^{-14}\) and \(4.6 \times 10^{-11}\) (1.0 before).

The particle sampler’s speed table reads its rows at their exact depths and its first slope from the rule below
it; on the speed tables of the Fig. 7 models of Chiang+(2026d) with \(\beta_{\rm c} = 1/2\) the worst row falls from \(4.1 \times 10^{-4}\) to \(1.7 \times 10^{-6}\). Where
the table stops at its row cap above the goal, its log line and the sampler say why.

The pilot grids keep one node at their seam (two nodes \(7 \times 10^{-18}\) apart at \(q = 0.05\) before).

A product of the earlier pipeline whose potential does not strictly decrease is refused when it is loaded, with
the first two rows where it rises and the number of such steps.

The critical band carries the parent’s central deficit to its cubic term, each term from the deficit’s own
integral, and its range reads the deficit’s truncation: where the band meets the rows on the
Fig. 7 children with \(\beta_{\rm c} = 1/2\) the step in \(\log_{10} F\) falls from \(2.5 \times 10^{-6}\) to \(7.9 \times 10^{-9}\) or less; one member of the
accuracy ledger moves (its roundtrip median by 2.3 percent), and no tier changes.

The environment’s switches read one set of words in any case: PIANISTPY_TIMING=false is off (it turned the
timing on), PIANISTPY_VERBOSITY=INFO is info (it raised at import), and an unknown value is refused by
name.

### Known limitations

One component per model at the public calls; several components in one potential are planned
(Upcoming features).

The gradient in the anisotropy parameters refuses COM Type II, whose interior point moves with \(R_{\rm a}\).

The gradient in the anisotropy parameters refuses a Method I truncated model, whose profile moves with them; a
Method II model on a fixed parent takes it (End-to-end differentiability).

On a GPU, the derivative of losvd and aperture_losvd may refuse a row of the line-of-sight table by name
(its transpose holds the row recomputed to the forward’s bit for bit); it never returns a wrong number
(End-to-end differentiability).

Spherical, non-rotating systems with an analytical double-power-law parent profile.

Dependencies: the standard and the minimum lists on Install.

## Ver. 0.9.0

A pre-release with the whole public interface: the distribution functions and their truncation
(compute_df, truncate, solve), the particles, the observables, the superposition and the inference.

Standard dependencies (the versions this release was developed and tested with): Python 3.11.13, NumPy
2.4.3, SciPy 1.17.1, h5py 3.15.1, Astropy 7.2.0, mpmath 1.3.0, SymPy 1.14.0, JAX 0.9.2, Diffrax 0.7.2,
Optimistix 0.1.0, filelock 4.0.12.

Minimum dependencies (the oldest versions that pass the full test suite, measured one package at a time):
Python 3.10; NumPy 2.0.0; SciPy 1.14.0; h5py 3.11.0; Astropy 7.2.0; mpmath 0.17; SymPy 1.2; JAX 0.6.1 with
Diffrax 0.7.0 and Optimistix 0.0.11 (these three are bound together); filelock. On Python 3.10 the newest JAX
is 0.6.2 (JAX 0.7 and later need Python 3.11). GPU: jax[cuda12] from 0.4.38 passes the suite on a GPU node;
the declared bound is the joint one above. Python 3.12 and 3.13 pass.

     On this page
