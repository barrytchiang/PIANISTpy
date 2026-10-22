# Accuracy tables

Source: https://pianistpy.com/docs/reference/accuracy

The accuracy tables below record the relative error of extensive “roundtrip” tests over a grid of systematically
varying input density profiles and anisotropy models. In each test, the package first
numerically computes the distribution function inverted from the input density profile \(\varrho\); the density
profile \(\varrho_{\rm inv}\) numerically evaluated from that distribution function is then compared against
the analytical input (the truth). Each cell reports the relative error
\(|(\varrho_{\rm inv} - \varrho)/\varrho|\) over \(10^{-8} < R < 10^{8}\), with \(R\) in units of the
profile’s scale radius. The column heads give the anisotropy models: OM is
Osipkov-Merritt and COM is Cuddeford-Osipkov-Merritt, of Type I or Type II ([Velocity anisotropy](https://pianistpy.com/docs/guide/anisotropy)). A cell
shows one of three outcomes:

- the error’s **maximum** (bold) and **median** (gray), and the result’s **tier** ([Accuracy](https://pianistpy.com/docs/guide/accuracy)); a
cell without a tier is silent. The tier comes from every check of the result, so a tier can be higher than the
cell’s error alone gives.

- **error** and the exception’s class, where the package raises an exception instead of returning a result:
`UnphysicalModelError` where no physical equilibrium exists for the model’s parameters,
`InputError` where the parameters are outside what the package supports, and
`AccuracyError` where the package cannot compute the model to the required accuracy. The
exception’s full message is in the cell’s tooltip and in the CSV.

- **excluded**, where the model violates the Cusp Slope–Central Anisotropy theorem
([An & Evans 2006b](https://doi.org/10.1086/501040)), which requires the inner slope of the density
profile, \(\gamma\), to be at least twice the central velocity anisotropy, \(\beta_{\rm c}\) (i.e.,
\(\gamma \ge 2\beta_{\rm c}\)). The model is still
computed and recorded in the CSV.

The distribution function of an untruncated Type II model is defined only at energies above
\(P(R_{\rm a})\), the relative potential at the anisotropy radius \(R_{\rm a}\). The cell of such a model
gives the tier that the package reports at energies above \(P(R_{\rm a})\), with the note “support above
\(P(R_{\rm a})\)”. Below \(P(R_{\rm a})\), the density \(\varrho_{\rm inv}\) does not measure the
distribution function’s accuracy, so the cell gives no error values.

A † marks a cell whose outcome differs from the outcome that the CSV’s `expected outcome` column states for the
model. The tables use the default settings level. The
downloadable files below hold every radial range the grid reads and the radius of each maximum, for the default
and the quick settings level. The counts of each release are on [Accuracy by release](https://pianistpy.com/docs/project/accuracy).

The grid of ver. 1.0.0 with the default settings level; the files that hold it, with the quick level’s grid, are listed under [The downloadable files](https://pianistpy.com/docs/reference/accuracy#the-downloadable-files) below.

## The main grid

Untruncated models: each density profile (rows), the double power law at inner slope \({\gamma}\) and outer slope \({\beta}\) with \({\alpha = 1}\) and the Plummer profile, under each anisotropy model (columns), with \({R_{\rm a} = 10}\) for the Osipkov-Merritt (OM) and Cuddeford-Osipkov-Merritt (COM) models.

isotropic
\({\beta_{\rm c} = -0.50}\)
\({\beta_{\rm c} = -0.25}\)
\({\beta_{\rm c} = 0.25}\)
\({\beta_{\rm c} = 0.50}\)
OM I \({R_{\rm a} = 10}\)
OM II \({R_{\rm a} = 10}\)
COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\)
COM I \({(\beta_{\rm c}, R_{\rm a}) = (0.25, 10)}\)
COM II \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\)

\({(\gamma, \beta) = (0, 3)}\)
\({3.1 \times 10^{-11}}\)\({3.2 \times 10^{-12}}\)
\({2.6 \times 10^{-8}}\)\({9.4 \times 10^{-14}}\)
\({1.6 \times 10^{-8}}\)\({2.2 \times 10^{-12}}\)
excluded
excluded
\({1.2 \times 10^{-10}}\)\({1.9 \times 10^{-11}}\)
\({6.4 \times 10^{-1}}\)\({1.2 \times 10^{-3}}\)
\({1.4 \times 10^{-8}}\)\({1.5 \times 10^{-9}}\)
excluded
\({4.9 \times 10^{-1}}\)\({9.4 \times 10^{-6}}\)

\({(\gamma, \beta) = (0.5, 3)}\)
\({7.6 \times 10^{-10}}\)\({2.1 \times 10^{-12}}\)
\({1.1 \times 10^{-10}}\)\({1.3 \times 10^{-13}}\)
\({7.7 \times 10^{-9}}\)\({2.1 \times 10^{-12}}\)
\({1.7 \times 10^{-11}}\)\({5.1 \times 10^{-12}}\)
excluded
\({7.6 \times 10^{-10}}\)\({4.9 \times 10^{-11}}\)
\({6.3 \times 10^{-1}}\)\({1.9 \times 10^{-5}}\)
\({4.4 \times 10^{-9}}\)\({1.9 \times 10^{-9}}\)
\({1.8 \times 10^{-10}}\)\({1.8 \times 10^{-11}}\)
\({4.7 \times 10^{-1}}\)\({1.3 \times 10^{-7}}\)

\({(\gamma, \beta) = (1, 3)}\)
\({2.9 \times 10^{-10}}\)\({1.7 \times 10^{-12}}\)
\({7.7 \times 10^{-11}}\)\({1.7 \times 10^{-13}}\)
\({1.2 \times 10^{-8}}\)\({1.7 \times 10^{-12}}\)
\({8.8 \times 10^{-9}}\)\({4.7 \times 10^{-12}}\)
\({1.1 \times 10^{-13}}\)\({3.9 \times 10^{-15}}\)
\({2.9 \times 10^{-10}}\)\({5.9 \times 10^{-11}}\)recorded
\({6.2 \times 10^{-1}}\)\({4.7 \times 10^{-7}}\)
\({1.2 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)
\({8.8 \times 10^{-9}}\)\({4.7 \times 10^{-11}}\)
\({4.6 \times 10^{-1}}\)\({1.4 \times 10^{-8}}\)

\({(\gamma, \beta) = (1.5, 3)}\)
\({1.6 \times 10^{-10}}\)\({7.7 \times 10^{-13}}\)
\({1.3 \times 10^{-10}}\)\({1.2 \times 10^{-13}}\)
\({6.6 \times 10^{-9}}\)\({2.0 \times 10^{-12}}\)
\({6.9 \times 10^{-9}}\)\({5.8 \times 10^{-12}}\)
\({3.5 \times 10^{-11}}\)\({3.4 \times 10^{-14}}\)
\({1.9 \times 10^{-10}}\)\({5.4 \times 10^{-11}}\)
\({6.5 \times 10^{-1}}\)\({1.4 \times 10^{-8}}\)
\({2.7 \times 10^{-8}}\)\({3.4 \times 10^{-9}}\)
\({6.9 \times 10^{-9}}\)\({5.1 \times 10^{-11}}\)
\({4.6 \times 10^{-1}}\)\({1.7 \times 10^{-8}}\)

\({(\gamma, \beta) = (1.9, 3)}\)
\({3.3 \times 10^{-10}}\)\({2.5 \times 10^{-13}}\)
\({1.7 \times 10^{-12}}\)\({7.6 \times 10^{-14}}\)
\({3.7 \times 10^{-10}}\)\({2.2 \times 10^{-12}}\)
\({9.0 \times 10^{-10}}\)\({7.5 \times 10^{-12}}\)
\({4.7 \times 10^{-13}}\)\({2.6 \times 10^{-14}}\)
\({2.5 \times 10^{-10}}\)\({2.6 \times 10^{-11}}\)recorded
\({6.2 \times 10^{-1}}\)\({2.0 \times 10^{-9}}\)
\({3.3 \times 10^{-8}}\)\({4.5 \times 10^{-9}}\)recorded
\({9.0 \times 10^{-10}}\)\({3.7 \times 10^{-11}}\)
refused

\({(\gamma, \beta) = (1.99, 3)}\)
\({4.3 \times 10^{-9}}\)\({2.2 \times 10^{-12}}\)
\({3.3 \times 10^{-11}}\)\({8.0 \times 10^{-13}}\)
\({5.9 \times 10^{-10}}\)\({9.0 \times 10^{-12}}\)
\({1.1 \times 10^{-9}}\)\({1.5 \times 10^{-11}}\)
\({1.4 \times 10^{-11}}\)\({2.6 \times 10^{-13}}\)
\({1.6 \times 10^{-7}}\)\({2.4 \times 10^{-10}}\)warned
errorAccuracyError
\({8.3 \times 10^{-5}}\)\({7.5 \times 10^{-9}}\)warned
\({2.0 \times 10^{-8}}\)\({4.2 \times 10^{-11}}\)warned
refused

\({(\gamma, \beta) = (0, 4)}\)
\({9.6 \times 10^{-10}}\)\({1.2 \times 10^{-11}}\)
\({3.1 \times 10^{-9}}\)\({5.7 \times 10^{-14}}\)
\({3.9 \times 10^{-9}}\)\({1.1 \times 10^{-11}}\)
excluded
excluded
\({1.7 \times 10^{-11}}\)\({2.5 \times 10^{-12}}\)
\({5.7 \times 10^{-1}}\)\({1.1 \times 10^{-4}}\)
\({6.7 \times 10^{-9}}\)\({9.8 \times 10^{-11}}\)
excluded
\({3.7 \times 10^{-1}}\)\({9.7 \times 10^{-7}}\)

\({(\gamma, \beta) = (0.5, 4)}\)
\({9.6 \times 10^{-10}}\)\({1.4 \times 10^{-11}}\)
\({7.8 \times 10^{-11}}\)\({9.4 \times 10^{-14}}\)
\({7.5 \times 10^{-9}}\)\({1.0 \times 10^{-11}}\)
\({5.4 \times 10^{-10}}\)\({5.0 \times 10^{-12}}\)
excluded
\({8.4 \times 10^{-10}}\)\({1.9 \times 10^{-12}}\)
\({5.6 \times 10^{-1}}\)\({2.1 \times 10^{-6}}\)
\({8.7 \times 10^{-9}}\)\({1.6 \times 10^{-10}}\)
\({1.8 \times 10^{-10}}\)\({5.1 \times 10^{-12}}\)
\({4.3 \times 10^{-1}}\)\({1.9 \times 10^{-8}}\)

\({(\gamma, \beta) = (1, 4)}\)
\({1.4 \times 10^{-9}}\)\({1.2 \times 10^{-11}}\)
\({7.4 \times 10^{-11}}\)\({8.4 \times 10^{-14}}\)
\({1.1 \times 10^{-8}}\)\({8.2 \times 10^{-12}}\)
\({8.0 \times 10^{-9}}\)\({1.5 \times 10^{-11}}\)
\({1.7 \times 10^{-14}}\)\({2.8 \times 10^{-15}}\)
\({1.4 \times 10^{-10}}\)\({1.2 \times 10^{-12}}\)recorded
\({5.7 \times 10^{-1}}\)\({5.1 \times 10^{-8}}\)
\({1.2 \times 10^{-8}}\)\({1.7 \times 10^{-10}}\)
\({8.0 \times 10^{-9}}\)\({3.3 \times 10^{-11}}\)
\({3.7 \times 10^{-1}}\)\({1.1 \times 10^{-8}}\)

\({(\gamma, \beta) = (1.5, 4)}\)
\({2.9 \times 10^{-9}}\)\({4.1 \times 10^{-12}}\)
\({1.3 \times 10^{-10}}\)\({5.7 \times 10^{-14}}\)
\({5.8 \times 10^{-9}}\)\({6.0 \times 10^{-12}}\)
\({6.2 \times 10^{-9}}\)\({1.7 \times 10^{-11}}\)
\({3.4 \times 10^{-11}}\)\({1.0 \times 10^{-14}}\)
\({1.8 \times 10^{-10}}\)\({1.3 \times 10^{-12}}\)
\({5.8 \times 10^{-1}}\)\({4.3 \times 10^{-9}}\)
\({2.7 \times 10^{-8}}\)\({2.8 \times 10^{-11}}\)
\({6.2 \times 10^{-9}}\)\({2.3 \times 10^{-11}}\)
\({4.3 \times 10^{-1}}\)\({1.5 \times 10^{-8}}\)

\({(\gamma, \beta) = (1.9, 4)}\)
\({1.1 \times 10^{-8}}\)\({3.0 \times 10^{-13}}\)
\({2.1 \times 10^{-12}}\)\({6.7 \times 10^{-14}}\)
\({2.5 \times 10^{-9}}\)\({4.5 \times 10^{-12}}\)
\({5.4 \times 10^{-9}}\)\({1.9 \times 10^{-11}}\)
\({3.2 \times 10^{-13}}\)\({2.5 \times 10^{-14}}\)
\({3.8 \times 10^{-9}}\)\({2.8 \times 10^{-12}}\)recorded
\({5.9 \times 10^{-1}}\)\({1.2 \times 10^{-9}}\)
\({2.8 \times 10^{-8}}\)\({5.9 \times 10^{-11}}\)recorded
\({7.6 \times 10^{-10}}\)\({5.0 \times 10^{-12}}\)
refused

\({(\gamma, \beta) = (1.99, 4)}\)
\({6.5 \times 10^{-8}}\)\({3.3 \times 10^{-12}}\)
\({3.2 \times 10^{-11}}\)\({7.7 \times 10^{-13}}\)
\({1.5 \times 10^{-8}}\)\({9.2 \times 10^{-12}}\)
\({2.7 \times 10^{-8}}\)\({2.5 \times 10^{-11}}\)
\({8.8 \times 10^{-12}}\)\({2.6 \times 10^{-13}}\)
\({1.4 \times 10^{-7}}\)\({2.8 \times 10^{-10}}\)warned
errorAccuracyError
\({6.5 \times 10^{-5}}\)\({5.2 \times 10^{-11}}\)warned
\({1.9 \times 10^{-8}}\)\({7.5 \times 10^{-12}}\)warned
refused

\({(\gamma, \beta) = (0, 5)}\)
\({2.3 \times 10^{-9}}\)\({1.2 \times 10^{-11}}\)
\({3.8 \times 10^{-9}}\)\({5.0 \times 10^{-14}}\)
\({4.3 \times 10^{-9}}\)\({1.8 \times 10^{-11}}\)
excluded
excluded
\({7.8 \times 10^{-10}}\)\({1.1 \times 10^{-11}}\)
\({5.0 \times 10^{-1}}\)\({1.2 \times 10^{-5}}\)
\({4.3 \times 10^{-9}}\)\({2.9 \times 10^{-11}}\)
excluded
\({3.8 \times 10^{-1}}\)\({1.1 \times 10^{-7}}\)

\({(\gamma, \beta) = (0.5, 5)}\)
\({3.1 \times 10^{-9}}\)\({2.1 \times 10^{-11}}\)
\({9.7 \times 10^{-11}}\)\({6.7 \times 10^{-14}}\)
\({7.3 \times 10^{-9}}\)\({2.1 \times 10^{-11}}\)
\({1.3 \times 10^{-9}}\)\({6.0 \times 10^{-12}}\)
excluded
\({7.9 \times 10^{-10}}\)\({1.5 \times 10^{-11}}\)
\({5.0 \times 10^{-1}}\)\({2.3 \times 10^{-7}}\)
\({8.2 \times 10^{-9}}\)\({1.2 \times 10^{-11}}\)
\({4.5 \times 10^{-10}}\)\({8.1 \times 10^{-12}}\)
refused

\({(\gamma, \beta) = (1, 5)}\)
\({4.3 \times 10^{-9}}\)\({2.1 \times 10^{-11}}\)
\({7.7 \times 10^{-11}}\)\({6.9 \times 10^{-14}}\)
\({1.0 \times 10^{-8}}\)\({1.9 \times 10^{-11}}\)
\({7.6 \times 10^{-9}}\)\({2.8 \times 10^{-11}}\)
\({1.7 \times 10^{-13}}\)\({2.7 \times 10^{-15}}\)
\({8.6 \times 10^{-10}}\)\({1.5 \times 10^{-11}}\)
\({5.0 \times 10^{-1}}\)\({5.5 \times 10^{-9}}\)
\({1.0 \times 10^{-8}}\)\({6.1 \times 10^{-12}}\)
\({7.6 \times 10^{-9}}\)\({3.9 \times 10^{-11}}\)
\({2.6 \times 10^{-1}}\)\({8.7 \times 10^{-9}}\)

\({(\gamma, \beta) = (1.5, 5)}\)
\({6.3 \times 10^{-9}}\)\({4.8 \times 10^{-12}}\)
\({1.4 \times 10^{-10}}\)\({7.4 \times 10^{-14}}\)
\({5.3 \times 10^{-9}}\)\({1.1 \times 10^{-11}}\)
\({5.8 \times 10^{-9}}\)\({2.7 \times 10^{-11}}\)
\({3.7 \times 10^{-11}}\)\({2.8 \times 10^{-14}}\)
\({1.7 \times 10^{-9}}\)\({1.1 \times 10^{-11}}\)
\({5.0 \times 10^{-1}}\)\({4.1 \times 10^{-9}}\)
\({5.3 \times 10^{-9}}\)\({5.0 \times 10^{-12}}\)
\({5.8 \times 10^{-9}}\)\({3.9 \times 10^{-11}}\)
refused

\({(\gamma, \beta) = (1.9, 5)}\)
\({2.8 \times 10^{-8}}\)\({3.5 \times 10^{-13}}\)
\({2.7 \times 10^{-12}}\)\({6.8 \times 10^{-14}}\)
\({9.1 \times 10^{-9}}\)\({7.4 \times 10^{-12}}\)
\({1.5 \times 10^{-8}}\)\({2.6 \times 10^{-11}}\)
\({8.8 \times 10^{-13}}\)\({3.1 \times 10^{-14}}\)
\({7.7 \times 10^{-9}}\)\({5.1 \times 10^{-12}}\)
\({5.3 \times 10^{-1}}\)\({9.2 \times 10^{-10}}\)recorded
\({2.9 \times 10^{-10}}\)\({4.2 \times 10^{-12}}\)recorded
\({6.3 \times 10^{-9}}\)\({2.7 \times 10^{-11}}\)
refused

\({(\gamma, \beta) = (1.99, 5)}\)
\({6.2 \times 10^{-8}}\)\({3.9 \times 10^{-12}}\)
\({3.2 \times 10^{-11}}\)\({8.1 \times 10^{-13}}\)
\({1.9 \times 10^{-8}}\)\({9.5 \times 10^{-12}}\)
\({4.2 \times 10^{-8}}\)\({3.2 \times 10^{-11}}\)
\({7.2 \times 10^{-12}}\)\({2.9 \times 10^{-13}}\)
\({1.5 \times 10^{-8}}\)\({1.3 \times 10^{-10}}\)
errorAccuracyError
errorAccuracyError†</sup>
\({1.5 \times 10^{-8}}\)\({5.4 \times 10^{-11}}\)recorded
refused

Plummer \({(\alpha = 2, \beta = 5, \gamma = 0)}\)
\({2.3 \times 10^{-9}}\)\({9.2 \times 10^{-15}}\)
\({4.2 \times 10^{-11}}\)\({1.9 \times 10^{-14}}\)
\({3.6 \times 10^{-9}}\)\({2.8 \times 10^{-11}}\)
excluded
excluded
\({5.9 \times 10^{-10}}\)\({8.1 \times 10^{-15}}\)
\({4.9 \times 10^{-1}}\)\({1.5 \times 10^{-5}}\)
\({3.6 \times 10^{-9}}\)\({1.6 \times 10^{-11}}\)
excluded
\({3.0 \times 10^{-1}}\)\({1.3 \times 10^{-7}}\)

## Variations of the main grid

### Anisotropy radius

The OM and COM models of the main grid (COM at \({\beta_{\rm c} = -0.25}\)), Type I and Type II, on three profiles (rows), with the anisotropy radius from \({R_{\rm a} = 0.3}\) to 1000 (columns).

\({R_{\rm a} = 0.3}\)
\({R_{\rm a} = 1}\)
\({R_{\rm a} = 3}\)
\({R_{\rm a} = 10}\)
\({R_{\rm a} = 30}\)
\({R_{\rm a} = 100}\)
\({R_{\rm a} = 1000}\)

NFW · OM I
refused
\({2.9 \times 10^{-10}}\)\({6.2 \times 10^{-11}}\)recorded
\({2.9 \times 10^{-10}}\)\({6.0 \times 10^{-11}}\)recorded
\({2.9 \times 10^{-10}}\)\({5.9 \times 10^{-11}}\)recorded
\({2.9 \times 10^{-10}}\)\({6.0 \times 10^{-11}}\)recorded
\({2.9 \times 10^{-10}}\)\({5.7 \times 10^{-11}}\)recorded
\({9.9 \times 10^{-9}}\)\({7.0 \times 10^{-11}}\)recorded

NFW · OM II
\({8.9 \times 10^{-1}}\)\({1.3 \times 10^{-4}}\)
\({6.9 \times 10^{-1}}\)\({2.5 \times 10^{-5}}\)
\({8.3 \times 10^{-1}}\)\({4.9 \times 10^{-6}}\)
\({6.2 \times 10^{-1}}\)\({4.7 \times 10^{-7}}\)
\({8.4 \times 10^{-1}}\)\({4.7 \times 10^{-8}}\)
\({6.7 \times 10^{-1}}\)\({5.0 \times 10^{-9}}\)
\({6.3 \times 10^{-1}}\)\({9.2 \times 10^{-10}}\)

NFW · COM I
refused
\({1.3 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)
\({1.3 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)
\({1.2 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)
\({1.3 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)
\({1.3 \times 10^{-8}}\)\({2.6 \times 10^{-9}}\)
\({1.3 \times 10^{-8}}\)\({2.4 \times 10^{-9}}\)

NFW · COM II
\({5.5 \times 10^{-1}}\)\({1.8 \times 10^{-6}}\)
\({4.9 \times 10^{-1}}\)\({2.3 \times 10^{-7}}\)
\({4.9 \times 10^{-1}}\)\({5.0 \times 10^{-8}}\)
\({4.6 \times 10^{-1}}\)\({1.4 \times 10^{-8}}\)
\({3.9 \times 10^{-1}}\)\({1.2 \times 10^{-8}}\)
\({4.4 \times 10^{-1}}\)\({8.9 \times 10^{-9}}\)
\({3.9 \times 10^{-1}}\)\({7.0 \times 10^{-9}}\)

cored \({(\gamma = 0, \beta = 4)}\) · OM I
refused
\({3.8 \times 10^{-11}}\)\({1.5 \times 10^{-11}}\)
\({1.7 \times 10^{-11}}\)\({2.0 \times 10^{-12}}\)
\({1.7 \times 10^{-11}}\)\({2.5 \times 10^{-12}}\)
\({1.8 \times 10^{-11}}\)\({1.9 \times 10^{-12}}\)
\({1.4 \times 10^{-10}}\)\({7.2 \times 10^{-12}}\)
\({1.7 \times 10^{-10}}\)\({1.5 \times 10^{-11}}\)

cored \({(\gamma = 0, \beta = 4)}\) · OM II
\({8.9 \times 10^{-1}}\)\({2.8 \times 10^{-1}}\)
\({6.6 \times 10^{-1}}\)\({6.0 \times 10^{-2}}\)
\({8.0 \times 10^{-1}}\)\({4.2 \times 10^{-3}}\)
\({5.7 \times 10^{-1}}\)\({1.1 \times 10^{-4}}\)
\({8.5 \times 10^{-1}}\)\({3.1 \times 10^{-6}}\)
\({5.7 \times 10^{-1}}\)\({4.0 \times 10^{-8}}\)
\({5.3 \times 10^{-1}}\)\({8.0 \times 10^{-11}}\)

cored \({(\gamma = 0, \beta = 4)}\) · COM I
refused
\({2.4 \times 10^{-9}}\)\({3.0 \times 10^{-12}}\)
\({2.4 \times 10^{-9}}\)\({5.5 \times 10^{-12}}\)
\({6.7 \times 10^{-9}}\)\({9.8 \times 10^{-11}}\)
\({1.5 \times 10^{-8}}\)\({1.2 \times 10^{-10}}\)
\({2.4 \times 10^{-9}}\)\({3.9 \times 10^{-11}}\)
\({2.3 \times 10^{-9}}\)\({4.9 \times 10^{-11}}\)

cored \({(\gamma = 0, \beta = 4)}\) · COM II
\({6.1 \times 10^{-1}}\)\({3.1 \times 10^{-3}}\)
\({4.4 \times 10^{-1}}\)\({6.3 \times 10^{-4}}\)
\({4.3 \times 10^{-1}}\)\({3.8 \times 10^{-5}}\)
\({3.7 \times 10^{-1}}\)\({9.7 \times 10^{-7}}\)
\({6.0 \times 10^{-1}}\)\({2.9 \times 10^{-8}}\)
\({3.1 \times 10^{-1}}\)\({2.7 \times 10^{-9}}\)
\({3.8 \times 10^{-1}}\)\({1.6 \times 10^{-9}}\)

steep \({(\gamma = 1.5, \beta = 4)}\) · OM I
\({1.9 \times 10^{-10}}\)\({1.1 \times 10^{-12}}\)
\({1.9 \times 10^{-10}}\)\({5.3 \times 10^{-13}}\)
\({1.9 \times 10^{-10}}\)\({5.6 \times 10^{-13}}\)
\({1.8 \times 10^{-10}}\)\({1.3 \times 10^{-12}}\)
\({3.5 \times 10^{-10}}\)\({2.1 \times 10^{-12}}\)
\({1.9 \times 10^{-10}}\)\({2.6 \times 10^{-12}}\)
\({1.9 \times 10^{-10}}\)\({8.0 \times 10^{-12}}\)

steep \({(\gamma = 1.5, \beta = 4)}\) · OM II
\({8.2 \times 10^{-1}}\)\({1.4 \times 10^{-6}}\)
\({6.4 \times 10^{-1}}\)\({3.1 \times 10^{-7}}\)
\({8.2 \times 10^{-1}}\)\({3.4 \times 10^{-8}}\)
\({5.8 \times 10^{-1}}\)\({4.3 \times 10^{-9}}\)
\({8.5 \times 10^{-1}}\)\({4.1 \times 10^{-9}}\)
\({5.8 \times 10^{-1}}\)\({4.1 \times 10^{-9}}\)
\({5.5 \times 10^{-1}}\)\({4.0 \times 10^{-9}}\)

steep \({(\gamma = 1.5, \beta = 4)}\) · COM I
\({2.7 \times 10^{-8}}\)\({1.1 \times 10^{-12}}\)
\({2.7 \times 10^{-8}}\)\({9.0 \times 10^{-13}}\)
\({2.7 \times 10^{-8}}\)\({1.6 \times 10^{-10}}\)
\({2.7 \times 10^{-8}}\)\({2.8 \times 10^{-11}}\)
\({2.8 \times 10^{-8}}\)\({3.1 \times 10^{-11}}\)
\({2.7 \times 10^{-8}}\)\({1.7 \times 10^{-11}}\)
\({2.7 \times 10^{-8}}\)\({1.6 \times 10^{-11}}\)

steep \({(\gamma = 1.5, \beta = 4)}\) · COM II
\({5.1 \times 10^{-1}}\)\({1.8 \times 10^{-8}}\)
\({5.0 \times 10^{-1}}\)\({1.8 \times 10^{-8}}\)
\({4.6 \times 10^{-1}}\)\({1.7 \times 10^{-8}}\)
\({4.3 \times 10^{-1}}\)\({1.5 \times 10^{-8}}\)
\({3.5 \times 10^{-1}}\)\({1.5 \times 10^{-8}}\)
\({3.6 \times 10^{-1}}\)\({1.4 \times 10^{-8}}\)
\({3.8 \times 10^{-1}}\)\({1.3 \times 10^{-8}}\)

### Truncation energy

Four profiles as Method I models under six anisotropy models of the main grid, each truncated by Method I and by Method II (rows), with the truncation energy from \({\mathcal{K}_{\rm t} = 0.5}\) to 0.002 (columns).

\({\mathcal{K}_{\rm t} = 0.5}\)
\({\mathcal{K}_{\rm t} = 0.2}\)
\({\mathcal{K}_{\rm t} = 0.05}\)
\({\mathcal{K}_{\rm t} = 0.01}\)
\({\mathcal{K}_{\rm t} = 0.002}\)

NFW · isotropic · Method I
\({1.0 \times 10^{-7}}\)\({2.3 \times 10^{-12}}\)
\({1.0 \times 10^{-7}}\)\({4.5 \times 10^{-12}}\)
\({1.0 \times 10^{-7}}\)\({8.8 \times 10^{-12}}\)
\({1.0 \times 10^{-7}}\)\({2.0 \times 10^{-11}}\)
\({8.3 \times 10^{-7}}\)\({6.8 \times 10^{-11}}\)

NFW · isotropic · Method II
\({1.0 \times 10^{-7}}\)\({9.1 \times 10^{-11}}\)
\({1.0 \times 10^{-7}}\)\({1.6 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({1.3 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({1.0 \times 10^{-10}}\)
\({8.3 \times 10^{-7}}\)\({1.4 \times 10^{-10}}\)

NFW · \({\beta_{\rm c} = -0.50}\) · Method I
\({2.0 \times 10^{-7}}\)\({1.8 \times 10^{-11}}\)
\({2.0 \times 10^{-7}}\)\({1.2 \times 10^{-11}}\)recorded
\({2.0 \times 10^{-7}}\)\({1.5 \times 10^{-11}}\)
\({2.0 \times 10^{-7}}\)\({2.5 \times 10^{-11}}\)
\({2.2 \times 10^{-6}}\)\({4.6 \times 10^{-11}}\)recorded

NFW · \({\beta_{\rm c} = -0.50}\) · Method II
\({2.2 \times 10^{-7}}\)\({1.9 \times 10^{-9}}\)
\({2.2 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)
\({2.2 \times 10^{-7}}\)\({5.7 \times 10^{-10}}\)
\({2.2 \times 10^{-7}}\)\({3.0 \times 10^{-10}}\)
\({8.3 \times 10^{-7}}\)\({1.8 \times 10^{-10}}\)

NFW · \({\beta_{\rm c} = 0.25}\) · Method I
\({3.6 \times 10^{-8}}\)\({1.5 \times 10^{-12}}\)
\({5.5 \times 10^{-8}}\)\({3.6 \times 10^{-12}}\)
\({5.4 \times 10^{-8}}\)\({6.8 \times 10^{-12}}\)
\({5.3 \times 10^{-8}}\)\({1.8 \times 10^{-11}}\)
\({4.2 \times 10^{-7}}\)\({7.9 \times 10^{-11}}\)

NFW · \({\beta_{\rm c} = 0.25}\) · Method II
\({4.1 \times 10^{-8}}\)\({2.2 \times 10^{-9}}\)
\({4.1 \times 10^{-8}}\)\({9.2 \times 10^{-10}}\)
\({4.1 \times 10^{-8}}\)\({7.8 \times 10^{-10}}\)
\({4.0 \times 10^{-8}}\)\({8.0 \times 10^{-10}}\)
\({8.3 \times 10^{-7}}\)\({5.7 \times 10^{-10}}\)

NFW · OM I \({R_{\rm a} = 10}\) · Method I
\({1.0 \times 10^{-7}}\)\({2.3 \times 10^{-12}}\)recorded
\({1.0 \times 10^{-7}}\)\({8.9 \times 10^{-12}}\)recorded
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

NFW · OM I \({R_{\rm a} = 10}\) · Method II
\({1.0 \times 10^{-7}}\)\({3.3 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({2.8 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({5.8 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({6.8 \times 10^{-10}}\)
\({8.3 \times 10^{-7}}\)\({5.2 \times 10^{-10}}\)

NFW · OM II \({R_{\rm a} = 10}\) · Method I
\({1.9 \times 10^{-7}}\)\({7.7 \times 10^{-11}}\)
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

NFW · OM II \({R_{\rm a} = 10}\) · Method II
\({1.0 \times 10^{-7}}\)\({2.7 \times 10^{-10}}\)
errorInputError
errorInputError
errorInputError
errorInputError

NFW · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({1.5 \times 10^{-7}}\)\({6.0 \times 10^{-12}}\)
\({1.5 \times 10^{-7}}\)\({1.0 \times 10^{-11}}\)
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

NFW · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({1.7 \times 10^{-7}}\)\({1.7 \times 10^{-9}}\)
\({1.6 \times 10^{-7}}\)\({9.8 \times 10^{-10}}\)
\({1.7 \times 10^{-7}}\)\({1.0 \times 10^{-9}}\)
\({1.7 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)
\({8.3 \times 10^{-7}}\)\({5.0 \times 10^{-10}}\)

Hernquist · isotropic · Method I
\({1.1 \times 10^{-8}}\)\({2.7 \times 10^{-12}}\)
\({1.0 \times 10^{-7}}\)\({5.4 \times 10^{-12}}\)
\({1.0 \times 10^{-7}}\)\({1.2 \times 10^{-11}}\)
\({1.0 \times 10^{-7}}\)\({3.4 \times 10^{-11}}\)
\({1.3 \times 10^{-6}}\)\({8.4 \times 10^{-11}}\)recorded

Hernquist · isotropic · Method II
\({2.0 \times 10^{-8}}\)\({9.4 \times 10^{-11}}\)
\({1.0 \times 10^{-7}}\)\({1.4 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({1.5 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({1.8 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({1.6 \times 10^{-10}}\)recorded

Hernquist · \({\beta_{\rm c} = -0.50}\) · Method I
\({2.0 \times 10^{-7}}\)\({1.9 \times 10^{-11}}\)
\({2.0 \times 10^{-7}}\)\({1.5 \times 10^{-11}}\)recorded
\({2.0 \times 10^{-7}}\)\({2.0 \times 10^{-11}}\)
\({2.0 \times 10^{-7}}\)\({2.8 \times 10^{-11}}\)
\({2.7 \times 10^{-6}}\)\({5.0 \times 10^{-11}}\)recorded

Hernquist · \({\beta_{\rm c} = -0.50}\) · Method II
\({4.3 \times 10^{-8}}\)\({1.2 \times 10^{-9}}\)
\({2.2 \times 10^{-7}}\)\({1.5 \times 10^{-9}}\)
\({2.2 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)
\({2.2 \times 10^{-7}}\)\({5.1 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({2.4 \times 10^{-10}}\)recorded

Hernquist · \({\beta_{\rm c} = 0.25}\) · Method I
\({8.3 \times 10^{-9}}\)\({1.6 \times 10^{-12}}\)
\({1.1 \times 10^{-8}}\)\({4.4 \times 10^{-12}}\)
\({5.4 \times 10^{-8}}\)\({9.7 \times 10^{-12}}\)
\({5.4 \times 10^{-8}}\)\({3.4 \times 10^{-11}}\)
\({8.1 \times 10^{-7}}\)\({9.6 \times 10^{-11}}\)

Hernquist · \({\beta_{\rm c} = 0.25}\) · Method II
\({3.2 \times 10^{-8}}\)\({2.9 \times 10^{-9}}\)
\({4.1 \times 10^{-8}}\)\({1.8 \times 10^{-9}}\)
\({4.0 \times 10^{-8}}\)\({9.8 \times 10^{-10}}\)
\({5.2 \times 10^{-8}}\)\({1.2 \times 10^{-9}}\)
\({1.3 \times 10^{-6}}\)\({7.9 \times 10^{-10}}\)recorded

Hernquist · OM I \({R_{\rm a} = 10}\) · Method I
\({1.1 \times 10^{-8}}\)\({2.7 \times 10^{-12}}\)recorded
\({1.0 \times 10^{-7}}\)\({5.5 \times 10^{-12}}\)recorded
\({1.0 \times 10^{-7}}\)\({1.2 \times 10^{-11}}\)recorded
\({1.0 \times 10^{-7}}\)\({3.0 \times 10^{-11}}\)recorded
errorUnphysicalModelError

Hernquist · OM I \({R_{\rm a} = 10}\) · Method II
\({2.0 \times 10^{-8}}\)\({5.8 \times 10^{-11}}\)
\({1.0 \times 10^{-7}}\)\({1.2 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({1.7 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({4.3 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({3.5 \times 10^{-10}}\)recorded

Hernquist · OM II \({R_{\rm a} = 10}\) · Method I
\({1.9 \times 10^{-7}}\)\({7.9 \times 10^{-11}}\)
\({1.8 \times 10^{-7}}\)\({1.0 \times 10^{-10}}\)
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

Hernquist · OM II \({R_{\rm a} = 10}\) · Method II
\({2.0 \times 10^{-8}}\)\({3.4 \times 10^{-10}}\)
\({1.0 \times 10^{-7}}\)\({4.3 \times 10^{-10}}\)
errorInputError
errorInputError
errorInputError

Hernquist · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({1.6 \times 10^{-7}}\)\({5.7 \times 10^{-12}}\)
\({1.6 \times 10^{-7}}\)\({7.3 \times 10^{-12}}\)
\({1.5 \times 10^{-7}}\)\({2.3 \times 10^{-11}}\)
\({1.5 \times 10^{-7}}\)\({3.1 \times 10^{-11}}\)
errorUnphysicalModelError

Hernquist · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({3.4 \times 10^{-8}}\)\({1.9 \times 10^{-9}}\)
\({1.7 \times 10^{-7}}\)\({1.5 \times 10^{-9}}\)
\({1.7 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)
\({1.7 \times 10^{-7}}\)\({1.3 \times 10^{-9}}\)
\({1.3 \times 10^{-6}}\)\({1.1 \times 10^{-9}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · isotropic · Method I
\({2.3 \times 10^{-3}}\)\({4.6 \times 10^{-12}}\)warned
\({1.8 \times 10^{-4}}\)\({7.9 \times 10^{-12}}\)warned
\({5.2 \times 10^{-6}}\)\({1.6 \times 10^{-11}}\)recorded
\({9.3 \times 10^{-8}}\)\({4.2 \times 10^{-11}}\)
\({1.3 \times 10^{-6}}\)\({8.5 \times 10^{-11}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · isotropic · Method II
\({2.3 \times 10^{-3}}\)\({1.1 \times 10^{-10}}\)warned
\({1.8 \times 10^{-4}}\)\({1.7 \times 10^{-10}}\)warned
\({5.2 \times 10^{-6}}\)\({1.8 \times 10^{-10}}\)recorded
\({9.6 \times 10^{-8}}\)\({2.2 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({1.8 \times 10^{-10}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · \({\beta_{\rm c} = -0.50}\) · Method I
\({1.4 \times 10^{-3}}\)\({2.5 \times 10^{-11}}\)warned
\({5.4 \times 10^{-5}}\)\({2.2 \times 10^{-11}}\)recorded
\({6.9 \times 10^{-7}}\)\({2.2 \times 10^{-11}}\)
\({1.1 \times 10^{-7}}\)\({4.1 \times 10^{-11}}\)
\({2.7 \times 10^{-6}}\)\({7.4 \times 10^{-11}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · \({\beta_{\rm c} = -0.50}\) · Method II
errorAccuracyError
errorAccuracyError
\({1.4 \times 10^{-4}}\)\({7.5 \times 10^{-10}}\)warned
\({2.5 \times 10^{-6}}\)\({3.8 \times 10^{-10}}\)recorded
\({1.3 \times 10^{-6}}\)\({2.2 \times 10^{-10}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · \({\beta_{\rm c} = 0.25}\) · Method I
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

cored \({(\gamma = 0, \beta = 4)}\) · \({\beta_{\rm c} = 0.25}\) · Method II
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

cored \({(\gamma = 0, \beta = 4)}\) · OM I \({R_{\rm a} = 10}\) · Method I
\({2.3 \times 10^{-3}}\)\({4.6 \times 10^{-12}}\)warned
\({2.2 \times 10^{-4}}\)\({7.9 \times 10^{-12}}\)warned
\({2.9 \times 10^{-5}}\)\({1.2 \times 10^{-11}}\)recorded
errorUnphysicalModelError
errorUnphysicalModelError

cored \({(\gamma = 0, \beta = 4)}\) · OM I \({R_{\rm a} = 10}\) · Method II
\({2.3 \times 10^{-3}}\)\({4.3 \times 10^{-10}}\)warned
\({1.8 \times 10^{-4}}\)\({6.1 \times 10^{-10}}\)warned
\({5.2 \times 10^{-6}}\)\({4.1 \times 10^{-10}}\)recorded
\({9.6 \times 10^{-8}}\)\({1.0 \times 10^{-9}}\)
\({1.3 \times 10^{-6}}\)\({4.6 \times 10^{-10}}\)recorded

cored \({(\gamma = 0, \beta = 4)}\) · OM II \({R_{\rm a} = 10}\) · Method I
\({2.2 \times 10^{-3}}\)\({1.0 \times 10^{-10}}\)warned
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

cored \({(\gamma = 0, \beta = 4)}\) · OM II \({R_{\rm a} = 10}\) · Method II
\({2.3 \times 10^{-3}}\)\({7.1 \times 10^{-10}}\)warned
errorInputError
errorInputError
errorInputError
errorInputError

cored \({(\gamma = 0, \beta = 4)}\) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({1.0 \times 10^{-2}}\)\({8.0 \times 10^{-12}}\)warned
\({7.2 \times 10^{-4}}\)\({1.2 \times 10^{-11}}\)warned
\({8.1 \times 10^{-5}}\)\({4.7 \times 10^{-11}}\)recorded
errorUnphysicalModelError
errorUnphysicalModelError

cored \({(\gamma = 0, \beta = 4)}\) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
errorAccuracyError
\({2.3 \times 10^{-3}}\)\({1.1 \times 10^{-9}}\)warned
\({6.8 \times 10^{-5}}\)\({9.0 \times 10^{-10}}\)warned
\({1.2 \times 10^{-6}}\)\({8.8 \times 10^{-10}}\)recorded
\({1.3 \times 10^{-6}}\)\({8.2 \times 10^{-10}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · isotropic · Method I
\({9.3 \times 10^{-9}}\)\({3.8 \times 10^{-12}}\)
\({3.0 \times 10^{-7}}\)\({3.5 \times 10^{-12}}\)
\({3.0 \times 10^{-7}}\)\({1.2 \times 10^{-11}}\)
\({3.0 \times 10^{-7}}\)\({3.8 \times 10^{-11}}\)
\({1.3 \times 10^{-6}}\)\({9.3 \times 10^{-11}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · isotropic · Method II
\({4.9 \times 10^{-8}}\)\({1.4 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({1.3 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({1.6 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({1.7 \times 10^{-10}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · \({\beta_{\rm c} = -0.50}\) · Method I
\({5.1 \times 10^{-7}}\)\({3.2 \times 10^{-11}}\)
\({4.9 \times 10^{-7}}\)\({1.3 \times 10^{-11}}\)recorded
\({4.9 \times 10^{-7}}\)\({2.1 \times 10^{-11}}\)
\({4.9 \times 10^{-7}}\)\({4.4 \times 10^{-11}}\)
\({2.7 \times 10^{-6}}\)\({8.0 \times 10^{-11}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · \({\beta_{\rm c} = -0.50}\) · Method II
\({3.8 \times 10^{-8}}\)\({8.1 \times 10^{-10}}\)
\({5.1 \times 10^{-7}}\)\({8.0 \times 10^{-10}}\)
\({5.1 \times 10^{-7}}\)\({1.7 \times 10^{-9}}\)
\({5.1 \times 10^{-7}}\)\({9.4 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({3.5 \times 10^{-10}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · \({\beta_{\rm c} = 0.25}\) · Method I
\({8.3 \times 10^{-9}}\)\({3.2 \times 10^{-12}}\)
\({2.1 \times 10^{-7}}\)\({3.0 \times 10^{-12}}\)
\({2.0 \times 10^{-7}}\)\({1.1 \times 10^{-11}}\)
\({2.0 \times 10^{-7}}\)\({3.9 \times 10^{-11}}\)
\({8.1 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)

steep \({(\gamma = 1.5, \beta = 4)}\) · \({\beta_{\rm c} = 0.25}\) · Method II
\({7.0 \times 10^{-8}}\)\({2.4 \times 10^{-9}}\)
\({1.9 \times 10^{-7}}\)\({2.2 \times 10^{-9}}\)
\({1.9 \times 10^{-7}}\)\({9.1 \times 10^{-10}}\)
\({1.9 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)
\({1.3 \times 10^{-6}}\)\({7.5 \times 10^{-10}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · OM I \({R_{\rm a} = 10}\) · Method I
\({9.3 \times 10^{-9}}\)\({4.0 \times 10^{-12}}\)
\({3.0 \times 10^{-7}}\)\({3.6 \times 10^{-12}}\)
\({3.0 \times 10^{-7}}\)\({1.2 \times 10^{-11}}\)
\({3.0 \times 10^{-7}}\)\({3.3 \times 10^{-11}}\)
errorUnphysicalModelError

steep \({(\gamma = 1.5, \beta = 4)}\) · OM I \({R_{\rm a} = 10}\) · Method II
\({4.9 \times 10^{-8}}\)\({1.4 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({9.2 \times 10^{-11}}\)
\({3.0 \times 10^{-7}}\)\({1.5 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({2.7 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({2.8 \times 10^{-10}}\)recorded

steep \({(\gamma = 1.5, \beta = 4)}\) · OM II \({R_{\rm a} = 10}\) · Method I
\({1.3 \times 10^{-6}}\)\({8.5 \times 10^{-11}}\)
\({1.3 \times 10^{-6}}\)\({7.0 \times 10^{-11}}\)
errorUnphysicalModelError
errorUnphysicalModelError
errorUnphysicalModelError

steep \({(\gamma = 1.5, \beta = 4)}\) · OM II \({R_{\rm a} = 10}\) · Method II
\({4.9 \times 10^{-8}}\)\({3.2 \times 10^{-10}}\)
\({3.0 \times 10^{-7}}\)\({3.4 \times 10^{-10}}\)
errorInputError
errorInputError
errorInputError

steep \({(\gamma = 1.5, \beta = 4)}\) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({4.2 \times 10^{-7}}\)\({5.6 \times 10^{-12}}\)
\({4.0 \times 10^{-7}}\)\({4.0 \times 10^{-12}}\)
\({4.0 \times 10^{-7}}\)\({1.4 \times 10^{-11}}\)
\({4.0 \times 10^{-7}}\)\({3.6 \times 10^{-11}}\)
errorUnphysicalModelError

steep \({(\gamma = 1.5, \beta = 4)}\) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({2.4 \times 10^{-8}}\)\({1.8 \times 10^{-9}}\)
\({4.1 \times 10^{-7}}\)\({1.9 \times 10^{-9}}\)
\({4.1 \times 10^{-7}}\)\({1.5 \times 10^{-9}}\)
\({4.1 \times 10^{-7}}\)\({9.8 \times 10^{-10}}\)
\({1.3 \times 10^{-6}}\)\({4.6 \times 10^{-10}}\)recorded

## Limiting cases

Anisotropies and truncations at the edges of the supported range (columns): the limit of the Cusp Slope-Central Anisotropy theorem ([An & Evans 2006b](https://doi.org/10.1086/501040)), the degenerate and steep constant anisotropies, and a small anisotropy radius inside a deep truncation. The rows include an outer slope of 2.5, which the package refuses.

\({\beta_{\rm c} = \gamma/2}\)
\({\beta_{\rm c} = -1/2 + 10^{-7}}\)
\({\beta_{\rm c} = -1}\)
\({\beta_{\rm c} = -1.5}\)
\({\beta_{\rm c} = +0.75}\)
OM I \({R_{\rm a} = 0.3}\) \({\mathcal{K}_{\rm t} = 0.01}\)
\({\beta_{\rm c} = -1/2}\) \({\mathcal{K}_{\rm t} = 0.3}\)

\({(\gamma, \beta) = (0, 4)}\)
\({9.6 \times 10^{-10}}\)\({1.2 \times 10^{-11}}\)
\({3.3 \times 10^{-9}}\)\({5.3 \times 10^{-12}}\)
\({6.1 \times 10^{-9}}\)\({1.8 \times 10^{-12}}\)
\({9.2 \times 10^{-9}}\)\({8.2 \times 10^{-14}}\)
excluded
errorUnphysicalModelError
\({2.1 \times 10^{-4}}\)\({2.2 \times 10^{-11}}\)warned

\({(\gamma, \beta) = (0.5, 4)}\)
\({5.4 \times 10^{-10}}\)\({5.0 \times 10^{-12}}\)
\({5.7 \times 10^{-9}}\)\({4.9 \times 10^{-12}}\)
\({3.9 \times 10^{-9}}\)\({1.0 \times 10^{-11}}\)
\({1.6 \times 10^{-10}}\)\({7.0 \times 10^{-14}}\)
excluded
errorUnphysicalModelError
\({1.0 \times 10^{-7}}\)\({1.7 \times 10^{-11}}\)

\({(\gamma, \beta) = (1, 4)}\)
\({1.7 \times 10^{-14}}\)\({2.8 \times 10^{-15}}\)
\({8.7 \times 10^{-10}}\)\({3.4 \times 10^{-12}}\)
\({3.0 \times 10^{-9}}\)\({2.6 \times 10^{-11}}\)recorded
\({1.3 \times 10^{-10}}\)\({1.6 \times 10^{-13}}\)
excluded
errorUnphysicalModelError
\({2.0 \times 10^{-7}}\)\({1.5 \times 10^{-11}}\)recorded

\({(\gamma, \beta) = (1.5, 4)}\)
\({8.2 \times 10^{-11}}\)\({1.4 \times 10^{-12}}\)
\({1.6 \times 10^{-9}}\)\({3.7 \times 10^{-12}}\)
\({6.2 \times 10^{-9}}\)\({9.5 \times 10^{-12}}\)
\({2.4 \times 10^{-10}}\)\({1.2 \times 10^{-13}}\)
\({8.2 \times 10^{-11}}\)\({1.4 \times 10^{-12}}\)
errorUnphysicalModelError
\({4.9 \times 10^{-7}}\)\({1.7 \times 10^{-11}}\)recorded

\({(\gamma, \beta) = (1.9, 4)}\)
\({4.0 \times 10^{-10}}\)\({2.2 \times 10^{-11}}\)
\({5.4 \times 10^{-9}}\)\({3.1 \times 10^{-13}}\)
\({1.6 \times 10^{-8}}\)\({6.2 \times 10^{-13}}\)
\({3.9 \times 10^{-12}}\)\({1.1 \times 10^{-13}}\)
\({4.0 \times 10^{-10}}\)\({6.0 \times 10^{-13}}\)recorded
errorUnphysicalModelError
errorAccuracyError

\({(\gamma, \beta) = (1.99, 4)}\)
errorUnphysicalModelError
\({1.7 \times 10^{-7}}\)\({5.6 \times 10^{-11}}\)
\({1.1 \times 10^{-7}}\)\({6.1 \times 10^{-12}}\)
\({5.7 \times 10^{-11}}\)\({1.3 \times 10^{-12}}\)recorded
\({2.1 \times 10^{-9}}\)\({2.9 \times 10^{-12}}\)recorded
errorInputError
errorInputError

Plummer \({(\alpha = 2, \beta = 5, \gamma = 0)}\)
\({2.3 \times 10^{-9}}\)\({9.2 \times 10^{-15}}\)
\({3.0 \times 10^{-9}}\)\({5.9 \times 10^{-12}}\)
\({4.4 \times 10^{-9}}\)\({2.2 \times 10^{-12}}\)
\({7.9 \times 10^{-11}}\)\({1.7 \times 10^{-14}}\)
excluded
errorUnphysicalModelError
\({9.7 \times 10^{-3}}\)\({3.0 \times 10^{-11}}\)warned

\({(\gamma, \beta) = (2, 4)}\)
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError

\({(\gamma, \beta) = (1, 2.5)}\)
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError
errorInputError

## The downloadable files

The grid of ver. 1.0.0 is published as three files, each of which opens with a comment block that repeats the definitions below (`#` lines, which `pandas.read_csv(path, comment='#')` skips):

- [`grid_default.csv`](https://pianistpy.com/docs/_downloads/6f15053f3a7f4c7d100c986afa508c4e/grid_default.csv): the grid with the default settings level. One row is one model of the grid.

- [`grid_quick.csv`](https://pianistpy.com/docs/_downloads/2e7555df8c0babbc2a009c85f6173c54/grid_quick.csv): the same grid with the quick settings level, in the same columns.

- [`members.csv`](https://pianistpy.com/docs/_downloads/962e66e37a969157210bdebc01173211/members.csv): the list of the grid’s models with the place of each in the tables above and the outcome the grid expects of it. One row is one model.

The columns of the grid files, in their order:

`model`
the model’s key in the grid: its table (T1 the main grid, T2 anisotropy radius, T3 truncation energy, T4 limiting cases, the sections of the Accuracy tables page), its profile, its anisotropy model and its truncation

`settings level`
the settings level of the run: “default” or “quick” (the two tested levels of pianistpy.Settings)

`truncation method`
1 where the distribution function is inverted from the model’s own density (Method I, with or without a truncation); 2 where the model is a Method II model, the distribution function inverted from the truncated density of a Method I model

`anisotropy model`
the velocity anisotropy model: “iso” (isotropic), “constant” (constant anisotropy \(\beta_{\rm c}\)), “OM_I” and “OM_II” (Osipkov-Merritt of Type I and Type II), “COM_I” and “COM_II” (Cuddeford-Osipkov-Merritt of Type I and Type II)

`alpha`
the transition sharpness of the double power law

`beta`
the outer slope of the double power law

`gamma`
the inner slope of the double power law

`beta_c`
the central velocity anisotropy; 0 for the isotropic and the Osipkov-Merritt models

`r_a`
the anisotropy radius in scale radii; blank for the isotropic and the constant-anisotropy models

`k_t`
the truncation energy as a fraction of the central potential; blank for an untruncated model

`source model`
for a Method II model, the Method I model whose truncated density it is built on (its profile, anisotropy model and truncation energy); blank otherwise

`expected outcome`
what the grid expects of the model: “value” (a result is served), “value: not physical” (a result is served for a model that is not a physical equilibrium), “value: the Method I model’s DF within \(10^{-6}\), read by the package’s rule” (a Method II model whose distribution function equals its Method I model’s), or a refusal with the error the package raises (its kind, then the opening words of its message)

`physical`
whether the model is a physical equilibrium by the grid’s own criteria: True, False, or blank where the model is refused before the criteria apply; a result can be served for a model that is not physical

`outcome`
what the package did: “value” (the model was computed and the error of the density computed back from its distribution function measured); “refused: the package’s tier (checks)” (the model was computed and its validation report is in the tier refused, with the checks beyond \(10^{-2}\) listed); “refusal: <error>: <message>” (the package raised that error; no model exists)

`outcome as expected`
True where the outcome is the expected outcome, False where it is not (the Accuracy tables page marks such a cell)

`tier`
the tier of the model’s validation report: silent, every check at most \(10^{-6}\); recorded, at most \(10^{-4}\); warned, at most \(10^{-2}\); refused, beyond \(10^{-2}\); blank where the model is refused before a report exists

`tier of the distribution function`
the tier of the distribution function’s own report, read as the model’s tier (the model’s tier is at least this one, since a model’s report carries its distribution function’s checks)

`lowest energy of the support`
for a Type II model without a truncation, the energy \(P(R_{\rm a})\), as a fraction of the central potential, above which its distribution function is defined; the errors are read over that support only; blank otherwise

`median relative error, 1e-4 < R < 1e4`
the median over the radii \(10^{-4} < R < 10^{4}\) (\(R\) in scale radii) of the relative error of the density computed back from the distribution function against the model’s density; blank where no result exists

`maximum relative error, 1e-4 < R < 1e4`
the maximum of the same relative error over the radii \(10^{-4} < R < 10^{4}\)

`radius of the maximum, 1e-4 < R < 1e4`
the radius, in scale radii, at which the maximum over \(10^{-4} < R < 10^{4}\) is taken

`number of radii, 1e-4 < R < 1e4`
the number of radii over which the error is read in \(10^{-4} < R < 10^{4}\)

`radii with a non-finite error, 1e-4 < R < 1e4`
the number of those radii at which the computed density is not finite; 0 where every radius is finite

`median relative error, 1e-4 < R < 1e5`
the median over the radii \(10^{-4} < R < 10^{5}\) (\(R\) in scale radii) of the relative error of the density computed back from the distribution function against the model’s density; blank where no result exists

`maximum relative error, 1e-4 < R < 1e5`
the maximum of the same relative error over the radii \(10^{-4} < R < 10^{5}\)

`radius of the maximum, 1e-4 < R < 1e5`
the radius, in scale radii, at which the maximum over \(10^{-4} < R < 10^{5}\) is taken

`number of radii, 1e-4 < R < 1e5`
the number of radii over which the error is read in \(10^{-4} < R < 10^{5}\)

`radii with a non-finite error, 1e-4 < R < 1e5`
the number of those radii at which the computed density is not finite; 0 where every radius is finite

`median relative error, 1e-8 < R < 1e8`
the median over the radii \(10^{-8} < R < 10^{8}\) (\(R\) in scale radii) of the relative error of the density computed back from the distribution function against the model’s density; blank where no result exists

`maximum relative error, 1e-8 < R < 1e8`
the maximum of the same relative error over the radii \(10^{-8} < R < 10^{8}\)

`radius of the maximum, 1e-8 < R < 1e8`
the radius, in scale radii, at which the maximum over \(10^{-8} < R < 10^{8}\) is taken

`number of radii, 1e-8 < R < 1e8`
the number of radii over which the error is read in \(10^{-8} < R < 10^{8}\)

`radii with a non-finite error, 1e-8 < R < 1e8`
the number of those radii at which the computed density is not finite; 0 where every radius is finite

`refusal of the distribution function`
the package’s message where it refused to compute the model’s distribution function; blank where it did not

`refusal of the truncation`
the package’s message where it refused a truncation of the model: for a truncated model, its own truncation; for an untruncated model, the grid’s probe that asks the package to truncate it (a profile whose total mass diverges, outer slope at most 3, and a Type II model whose support ends at the anisotropy’s domain edge are refused by design); blank where nothing was refused

`refusal of the density check`
the package’s message where the density could not be computed back from the distribution function (a distribution function that is negative somewhere has no logarithm); blank otherwise

The columns of the model list, in their order:

`model`
the model’s key, as in the grid files

`table`
the table of the Accuracy tables page the model is in: 1 the main grid, 2 anisotropy radius, 3 truncation energy, 4 limiting cases (the page’s sections)

`row`
the key of the model’s row in that table, as the page labels it

`column`
the key of the model’s column in that table, as the page labels it

`alpha`
the transition sharpness of the double power law

`beta`
the outer slope of the double power law

`gamma`
the inner slope of the double power law

`anisotropy model`
the velocity anisotropy model, as in the grid files

`beta_c`
the central velocity anisotropy

`r_a`
the anisotropy radius in scale radii; blank where the model has none

`truncation method`
1 or 2, as in the grid files

`k_t`
the truncation energy as a fraction of the central potential; blank for an untruncated model

`source model`
for a Method II model, the Method I model it is built on; blank otherwise

`expected outcome`
what the grid expects of the model, as in the grid files
