# T:8 Verification

Source: https://pianistpy.com/docs/reference/technical-notes/verification

# T:8 Verification

## T:8.1 Reports and tiers

Every result carries a validation report with a fixed set of keys for its kind, one check per key, with its relative error and, where applicable, the radius or energy at which it is worst. The report is not a log: nothing is appended to it, and a derived result holds its own checks together with the worst tier of its parents. The checks are the following. For a DF: the slope column, the center against the top decade, the joins below the floor, the sign report, and the validity certificate (Section T:2.5.5). For a model: the Poisson residual, the round trip, and the row density (Section T:3.8), with the radii of trust of an untruncated result. For a line-of-sight table: the norm, the second moment, the unjudged rows, the convergence of the depth rule, the certificate, and the layer floor (Section T:6.5). For an aperture distribution: its norm by its own weights and its second moment against the dispersion of the aperture. For the local speed distribution: its norm, its second moment, and its change under doubling. For a draw: the bound on the support. Each tier is read as in Section T:0.5.

## T:8.2 Reference values

The exact forms are certified against references that owe nothing to the numerical pipeline. The symbolic derivatives of \(\varrho_\mathrm{aux}\) are compared with quadrature and differentiation in arbitrary precision at thirty digits; the derivatives of the anisotropy factor with the symbolic third derivative of its expression evaluated in arbitrary precision; the central series at the anchor with the symbolic derivatives of the closed-form density; and the closed-form DFs of the isotropic Hernquist and Plummer models, and of the NFW profile at \(\beta_\mathrm{c} = 1/2\) (Section T:2.2.5), with the table of the inversion. The two routes to every projected quantity (Section T:6) are held to the goal against each other, and a run of the full pipeline is compared bit for bit with the stored products that it reproduces on the same backend; every file is read back and compared with what was written.

## T:8.3 The roundtrip test

The test of Chiang et al. (2026d), Section 4, shown in Chiang et al. (2026d), Fig. 8, starts from the untruncated isotropic NFW profile, computes the DF, integrates it to a density, and solves the Poisson equation with \(P(0) = 1\). Every curve is numerical, and the only analytical quantity is the target density.

PIANISTpy. The standard untruncated isotropic product: the inverted DF, the velocity quadrature, and a numerical Poisson solve.

AGAMA. All defaults: the multipole potential built from the density on its default grid with the radial extent detected automatically, the default isotropic quasi-spherical DF with the exact action finder, and the density from its moments routine.

galpy. Its interpolated spherical potential on a grid denser and wider than its default, and its Eddington DF with every default. Its DF is zero beyond the default outer radius of that routine. galpy has no Poisson solver and therefore no density curve.

Fitting functions. Passed through the same velocity quadrature and Poisson solve as the PIANISTpy DF.

The numbers behind the figure are reproduced by the notebooks that accompany these notes.

## T:8.4 The accuracy grid

A fixed set of models, four tables of rows by columns over the exponents of the profile and the anisotropy models, is run unchanged at every release at both settings tiers. The DF of each model is inverted, its density is computed back from it, and the roundtrip error is read on a dense grid of radii. Each cell records the maximum and the median over the stated range, the tier that the result itself reports, or the name of the refusal where the package refuses the model; the grid states for every cell the outcome it expects, so that a release is compared with the last one cell by cell. A Type II model without a truncation has its DF only above \(P(R_\mathrm{a})\), and its cell gives the tier that the package reports over that support. The tables are published with the documentation; the measured values live there and in the notebooks, not in these notes.

     On this page
