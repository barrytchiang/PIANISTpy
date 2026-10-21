.. _tn-sec-verification:

T:8 Verification
================

.. _tn-sec-reports:

T:8.1 The checks of each validation report
------------------------------------------

Every result carries a validation report with a fixed set of keys for its kind, one check per key, holding an error (relative unless the key's line below states an absolute error), or a flag that is zero when the check passes and infinite when the check fails, and, where applicable, the radius or energy at which the check is worst. The report is not a log: nothing is appended to it, and a derived result holds its own checks together with the worst tier of the results it was computed from. The checks are the following, one line per kind of result.

- *DF, untruncated* (:ref:`Section T:2.5.5 <tn-sec-df-checks>`): ``top_form``, the relative residual of the table's top decade against the center's stated form; ``slope_column``, the largest relative error in :math:`F` between neighboring nodes that the stated slope column makes beyond the uncertainty of its finite differences; ``below_floor_join``, the relative mismatch of :math:`F` where the nodes below the floor meet the table's first row (zero where the table has no nodes below its floor); ``validity``, a flag, zero where float64 holds the table's columns at the table's top and infinite where float64 does not; ``high_k_slope_mismatch``, the absolute difference between the table's exponent :math:`\mathrm{d}\ln F/\mathrm{d}\ln u` at the table's top and the exponent of the center's form; ``round_trip``, the largest relative difference between the density computed from the DF and the profile's density; and, for a DF with a tail form, ``tail_join``, the relative mismatch of the tail form against :math:`F` at the table's floor.
- *DF of a Method II model*: ``top_form``, ``slope_column`` and ``below_floor_join``, as for an untruncated DF, with the Method I model's tier carried as the DF's.
- *DF negative somewhere*: ``top_form`` and ``slope_column``, both written as not a number, which places the DF in the tier refused.
- *Model, truncated by Method I* (:ref:`Section T:3.8 <tn-sec-model-checks>`): ``top_form`` and ``slope_column`` of the truncated DF's table, as for a DF; ``edge_join``, the larger of the two relative mismatches at the edge of the truncated DF, the edge integral of :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}` against the difference table at the splice and the Taylor form against the edge integral at the switch; ``df_clip``, the largest negative value of the truncated DF's difference set to zero, relative to :math:`F(\mathcal{K}_\mathrm{t})`; and the three checks of the profile: ``poisson_residual``, the largest relative residual of the potential and of the enclosed mass between rows against the integrals of the Poisson equation; ``round_trip``, the largest relative difference between the density computed from the model's DF and the profile's density at half-decade radii and at every row toward the edge; ``row_density``, a flag, zero where every bin of the potential and of the depth holds the rows that the row-count rule asks for, and infinite at any shortfall.
- *Model, untruncated*: ``outer_join`` and ``eps``, the join of the outer rows and the offset of the potential, recorded as not judged because the model's profile is the closed form on every row; and ``poisson_residual``, ``round_trip`` and ``row_density``, as for a truncated model, with the radii of trust.
- *Model of a Method II model*: ``top_form``, as for a DF; ``edge_join``, the relative mismatch of the edge law against the table's bottom row; and ``poisson_residual``, ``round_trip`` and ``row_density``, as for a truncated model.
- *Velocity table of a sampler*: ``row_error``, the table's own error across rows, the larger of the error in the cumulative probability :math:`\mathcal{U}` and the error relative to the scaled velocity :math:`x`; ``polish_residual``, the largest absolute residual in :math:`\lambda_\mathcal{U}` that the polish leaves at a node; the sampler carries the model's tier.
- *Sampler of projected velocities*: ``v_table``, the largest absolute difference between the interpolated velocity table's cumulative probability and each check row's own cumulative probability; ``window``, the largest difference between the ends of the user-specified range read from the table and from each check row; ``radius_quadrature``, the relative difference of the radial mass between the radial Gauss rule and a finer rule; ``row_norm``, the largest departure of a row's velocity distribution from unit norm; ``annulus_cut``, the fraction of the annulus's mass that lies beyond the table's rows.
- *Line-of-sight table* (:ref:`Section T:6.5 <tn-sec-losvd-table>`): ``norm``, the worst relative difference of a row's norm against the surface density; ``second_moment``, the worst relative difference of a row's second moment against the projected dispersion; ``unjudged``, a flag, infinite where a row with positive surface density could not be judged; ``t_converged``, a flag, infinite where a row's rule along the line of sight ran out of doublings before the rule's tests passed; ``certificate``, the table's last estimate of the table's reading between rows, in units of the table's tolerances, times the larger tolerance; ``layer_floor``, the relative error at which the layers toward the :math:`L = 0` point stop converging (zero where the layers converge).
- *Aperture distribution*: ``norm``, the integral of the distribution by the distribution's own weights less one; ``second_moment``, the second moment divided by the distribution's integral, against the aperture's dispersion :math:`\sigma_\mathrm{los}^2`; the distribution carries the line-of-sight table's tier.
- *Local velocity distribution*: ``norm``, the integral of the distribution less one; ``second_moment``, the second moment against :math:`\sigma_r^2 (3 - 2\beta)`; ``halving``, the largest change of the distribution under the last doubling of the angle rule, relative to the distribution's peak.
- *Draw*: ``phase_space_bound``, every particle's scaled velocity :math:`x \le 1` and distance :math:`d \ge 0` from the edge of the support; ``velocity_construction``, every particle's velocity within the support's bound up to the rounding floor; both written as zero, because a draw in which any particle fails either check raises ``AccuracyError``.
- *Superposition*: the members' keys, each check the worst over the members, and the worst tier of the members as the tier it carries.

Each tier is read as in :ref:`Section T:0.5 <tn-sec-tiers>`.

.. _tn-sec-references:

T:8.2 Tests against independent references
------------------------------------------

The exact forms are checked against references that owe nothing to the numerical pipeline. The symbolic derivatives of :math:`\varrho_\mathrm{aux}` are compared with quadrature and differentiation in arbitrary precision at thirty digits; the derivatives of the anisotropy factor with the symbolic third derivative of its expression evaluated in arbitrary precision; the central series at the anchor with the symbolic derivatives of the closed-form density; and the closed-form DFs of the isotropic Hernquist and Plummer models with the table of the inversion; the closed form of the NFW profile at :math:`\beta_\mathrm{c} = 1/2` (:ref:`Section T:2.2.5 <tn-sec-nfw-half>`) is checked where a Method II model uses that closed form, against the Method II model's computed rows at the anchor. The two routes to every projected quantity (:ref:`Section T:6 <tn-sec-observables>`) are held to :math:`10^{-6}` against each other, and a run of the full pipeline is compared bit for bit with the stored products that it reproduces on the same backend; every file is read back and compared with what was written.

.. _tn-sec-roundtrip:

T:8.3 The roundtrip test
------------------------

The test of |paper|, Section 4, shown in |paper|, Fig. 8, starts from the untruncated isotropic NFW profile, computes the DF, integrates it to a density, and solves the Poisson equation with :math:`P(0) = 1`. Every curve is numerical, and the only analytical quantity is the target density.

- *PIANISTpy.* The standard untruncated isotropic product: the inverted DF, the velocity quadrature, and a numerical Poisson solve.

- *AGAMA.* All defaults: the multipole potential built from the density on its default grid with the radial extent detected automatically, the default isotropic quasi-spherical DF with the exact action finder, and the density from its moments routine.

- *galpy.* Its interpolated spherical potential on a grid denser and wider than its default, and its Eddington DF with every default. Its DF is zero beyond the default outer radius of that routine. galpy has no Poisson solver and therefore no density curve.

- *Fitting functions.* Passed through the same velocity quadrature and Poisson solve as the PIANISTpy DF.

The DF comparison of the figure's upper panel is reproduced by the notebook that accompanies these notes; the density comparison of the lower panel is not.

.. _tn-sec-accuracy-grid:

T:8.4 The accuracy tables run at every release
----------------------------------------------

A fixed set of models, four tables of rows by columns over the exponents of the profile and the anisotropy models, is run unchanged at every release with both settings levels. The DF of each model is inverted, its density is computed back from it, and the roundtrip error is read on a dense grid of radii. Each cell records the maximum and the median over the stated range and the tier that the result itself reports, or the name of the refusal where the package refuses the model, or "excluded" where the model breaks the cusp bound; the grid states for every cell the outcome it expects, so that a release is compared with the last one cell by cell. A Type II model without a truncation has its DF only above :math:`P(R_\mathrm{a})`, and its cell gives the tier that the package reports over that support. The tables are published with the documentation; the measured values live there and in the notebooks, not in these notes.
