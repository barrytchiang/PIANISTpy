# Accuracy tables

Source: https://pianistpy.com/docs/reference/accuracy

# Accuracy tables

The accuracy grid is a fixed set of models, four tables of rows by columns, run unchanged at every release. Each
model’s distribution function is inverted and its density \(\varrho_{\rm inv}\) computed back from it; the
roundtrip error \(|(\varrho_{\rm inv} - \varrho)/\varrho|\), against the input profile \(\varrho\), is read at
513 radii, and a cell gives that error over \(10^{-4} < R < 10^{5}\) (\(R\) in scale radii):

its maximum, then its median, then the tier the result itself reports (goal: \(10^{-6}\) or less;
ceiling: \(10^{-4}\) or less; warning: \(10^{-2}\) or less; refused: beyond, see Accuracy);

refused and the refusal’s name, where the package refuses the model (the full message is in the cell’s
tooltip and in the CSV);

excluded where the anisotropy breaks the cusp bound \(\gamma \ge 2\beta_{\rm c}\) (computed, and recorded in the
CSV). A Type II model without a truncation has its distribution function only above the energy \(P(R_{\rm a})\): its
cell gives the tier the package reports over that support and the note “support above \(P(R_{\rm a})\)”, not the
density’s roundtrip, which is no measure there.

A † marks a cell whose outcome is not the one the grid expects. The tables are at the default tier; the CSV holds
every range the grid reads, the radius of each maximum and the compute times. The tables of earlier releases are
on Accuracy by release.

The grid of ver. 0.9.9, run at d59731fb, whose numerical code is 52a0ef0b’s (cpu), the default tier: the full grid as CSV and its member list.

### Table 1. Profiles by anisotropy

The double power law at inner slope \({\gamma}\) and outer slope \({\beta}\) (\({\alpha = 1}\)), and the Plummer profile, under ten anisotropy models; untruncated.

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
\({2.8 \times 10^{-11}}\)\({2.1 \times 10^{-13}}\)goal
\({2.6 \times 10^{-8}}\)\({1.5 \times 10^{-13}}\)goal
\({1.6 \times 10^{-8}}\)\({8.8 \times 10^{-13}}\)goal
excluded
excluded
\({1.2 \times 10^{-10}}\)\({2.0 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({1.4 \times 10^{-8}}\)\({1.2 \times 10^{-9}}\)goal
excluded
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (0.5, 3)}\)
\({8.0 \times 10^{-11}}\)\({1.7 \times 10^{-13}}\)goal
\({1.1 \times 10^{-10}}\)\({1.3 \times 10^{-13}}\)goal
\({2.6 \times 10^{-9}}\)\({6.8 \times 10^{-13}}\)goal
\({1.7 \times 10^{-11}}\)\({3.3 \times 10^{-12}}\)goal
excluded
\({9.4 \times 10^{-11}}\)\({3.7 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({3.0 \times 10^{-9}}\)\({1.7 \times 10^{-9}}\)goal
\({1.8 \times 10^{-10}}\)\({1.7 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1, 3)}\)
\({8.8 \times 10^{-11}}\)\({1.5 \times 10^{-13}}\)goal
\({7.3 \times 10^{-11}}\)\({1.1 \times 10^{-13}}\)goal
\({1.8 \times 10^{-9}}\)\({7.0 \times 10^{-13}}\)goal
\({1.6 \times 10^{-9}}\)\({2.9 \times 10^{-12}}\)goal
\({1.1 \times 10^{-13}}\)\({4.2 \times 10^{-15}}\)goal
\({9.0 \times 10^{-11}}\)\({3.7 \times 10^{-11}}\)ceiling
goalsupport above \({P(R_{\rm a})}\)
\({5.8 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({1.6 \times 10^{-9}}\)\({3.5 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1.5, 3)}\)
\({1.5 \times 10^{-10}}\)\({1.2 \times 10^{-13}}\)goal
\({1.3 \times 10^{-10}}\)\({8.7 \times 10^{-14}}\)goal
\({6.5 \times 10^{-10}}\)\({4.3 \times 10^{-13}}\)goal
\({9.5 \times 10^{-10}}\)\({2.5 \times 10^{-12}}\)goal
\({2.9 \times 10^{-11}}\)\({2.5 \times 10^{-14}}\)goal
\({1.8 \times 10^{-10}}\)\({2.8 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({2.6 \times 10^{-8}}\)\({3.0 \times 10^{-9}}\)goal
\({9.5 \times 10^{-10}}\)\({3.0 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1.9, 3)}\)
\({1.2 \times 10^{-12}}\)\({1.1 \times 10^{-13}}\)goal
\({1.3 \times 10^{-12}}\)\({7.1 \times 10^{-14}}\)goal
\({3.2 \times 10^{-11}}\)\({5.9 \times 10^{-13}}\)goal
\({1.3 \times 10^{-10}}\)\({1.8 \times 10^{-12}}\)goal
\({7.4 \times 10^{-14}}\)\({2.4 \times 10^{-14}}\)goal
\({7.3 \times 10^{-11}}\)\({2.5 \times 10^{-11}}\)ceiling
goalsupport above \({P(R_{\rm a})}\)
\({5.3 \times 10^{-9}}\)\({2.7 \times 10^{-9}}\)ceiling
\({1.3 \times 10^{-10}}\)\({2.4 \times 10^{-11}}\)goal
refusednot physical

\({(\gamma, \beta) = (1.99, 3)}\)
\({3.5 \times 10^{-11}}\)\({2.7 \times 10^{-12}}\)refused
\({3.3 \times 10^{-11}}\)\({1.9 \times 10^{-12}}\)goal
\({2.8 \times 10^{-10}}\)\({2.4 \times 10^{-12}}\)goal
\({6.4 \times 10^{-11}}\)\({1.3 \times 10^{-12}}\)goal
\({1.4 \times 10^{-11}}\)\({1.2 \times 10^{-13}}\)goal
\({1.6 \times 10^{-7}}\)\({3.9 \times 10^{-10}}\)refused
refusedAccuracyError†
\({4.0 \times 10^{-6}}\)\({7.4 \times 10^{-9}}\)warning
\({6.0 \times 10^{-10}}\)\({2.6 \times 10^{-11}}\)warning
refusednot physical

\({(\gamma, \beta) = (0, 4)}\)
\({1.7 \times 10^{-11}}\)\({1.3 \times 10^{-13}}\)goal
\({1.9 \times 10^{-9}}\)\({6.3 \times 10^{-14}}\)goal
\({2.9 \times 10^{-9}}\)\({2.0 \times 10^{-13}}\)goal
excluded
excluded
\({1.7 \times 10^{-11}}\)\({1.4 \times 10^{-12}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({6.7 \times 10^{-9}}\)\({1.4 \times 10^{-10}}\)goal
excluded
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (0.5, 4)}\)
\({7.1 \times 10^{-11}}\)\({1.2 \times 10^{-13}}\)goal
\({7.8 \times 10^{-11}}\)\({5.0 \times 10^{-14}}\)goal
\({2.2 \times 10^{-9}}\)\({1.8 \times 10^{-13}}\)goal
\({1.2 \times 10^{-11}}\)\({1.9 \times 10^{-12}}\)goal
excluded
\({7.2 \times 10^{-11}}\)\({9.0 \times 10^{-13}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({8.7 \times 10^{-9}}\)\({1.2 \times 10^{-10}}\)goal
\({1.8 \times 10^{-10}}\)\({4.1 \times 10^{-12}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1, 4)}\)
\({8.6 \times 10^{-11}}\)\({1.2 \times 10^{-13}}\)goal
\({7.2 \times 10^{-11}}\)\({3.9 \times 10^{-14}}\)goal
\({1.4 \times 10^{-9}}\)\({1.8 \times 10^{-13}}\)goal
\({1.3 \times 10^{-9}}\)\({1.8 \times 10^{-12}}\)goal
\({1.0 \times 10^{-14}}\)\({2.8 \times 10^{-15}}\)goal
\({8.6 \times 10^{-11}}\)\({5.6 \times 10^{-13}}\)ceiling
goalsupport above \({P(R_{\rm a})}\)
\({1.2 \times 10^{-8}}\)\({8.9 \times 10^{-11}}\)goal
\({1.3 \times 10^{-9}}\)\({1.8 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1.5, 4)}\)
\({1.4 \times 10^{-10}}\)\({1.0 \times 10^{-13}}\)goal
\({1.3 \times 10^{-10}}\)\({4.3 \times 10^{-14}}\)goal
\({4.4 \times 10^{-10}}\)\({2.0 \times 10^{-13}}\)goal
\({6.8 \times 10^{-10}}\)\({1.8 \times 10^{-12}}\)goal
\({2.3 \times 10^{-11}}\)\({7.0 \times 10^{-15}}\)goal
\({1.7 \times 10^{-10}}\)\({9.9 \times 10^{-13}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({2.6 \times 10^{-8}}\)\({1.3 \times 10^{-11}}\)goal
\({6.8 \times 10^{-10}}\)\({1.2 \times 10^{-11}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1.9, 4)}\)
\({1.1 \times 10^{-11}}\)\({1.1 \times 10^{-13}}\)goal
\({2.1 \times 10^{-12}}\)\({6.2 \times 10^{-14}}\)goal
\({2.3 \times 10^{-11}}\)\({5.1 \times 10^{-13}}\)goal
\({1.0 \times 10^{-10}}\)\({1.8 \times 10^{-12}}\)goal
\({2.5 \times 10^{-13}}\)\({2.3 \times 10^{-14}}\)goal
\({3.8 \times 10^{-9}}\)\({2.2 \times 10^{-12}}\)ceiling
goalsupport above \({P(R_{\rm a})}\)
\({3.9 \times 10^{-9}}\)\({4.9 \times 10^{-11}}\)ceiling
\({9.8 \times 10^{-11}}\)\({3.1 \times 10^{-12}}\)goal
refusednot physical

\({(\gamma, \beta) = (1.99, 4)}\)
\({1.4 \times 10^{-10}}\)\({5.0 \times 10^{-12}}\)refused
\({3.2 \times 10^{-11}}\)\({9.9 \times 10^{-13}}\)goal
\({2.7 \times 10^{-10}}\)\({2.6 \times 10^{-12}}\)goal
\({5.7 \times 10^{-11}}\)\({3.3 \times 10^{-12}}\)goal
\({8.8 \times 10^{-12}}\)\({8.7 \times 10^{-14}}\)goal
\({1.4 \times 10^{-7}}\)\({2.1 \times 10^{-10}}\)refused
refusedAccuracyError†
\({2.4 \times 10^{-6}}\)\({3.7 \times 10^{-11}}\)warning
\({5.6 \times 10^{-10}}\)\({4.1 \times 10^{-12}}\)warning
refusednot physical

\({(\gamma, \beta) = (0, 5)}\)
\({1.5 \times 10^{-11}}\)\({1.7 \times 10^{-13}}\)goal
\({1.5 \times 10^{-9}}\)\({4.3 \times 10^{-14}}\)goal
\({2.7 \times 10^{-9}}\)\({3.7 \times 10^{-13}}\)goal
excluded
excluded
\({1.6 \times 10^{-11}}\)\({7.5 \times 10^{-13}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({2.7 \times 10^{-9}}\)\({9.4 \times 10^{-12}}\)goal
excluded
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (0.5, 5)}\)
\({6.9 \times 10^{-11}}\)\({1.7 \times 10^{-13}}\)goal
\({7.2 \times 10^{-11}}\)\({4.8 \times 10^{-14}}\)goal
\({1.9 \times 10^{-9}}\)\({3.7 \times 10^{-13}}\)goal
\({1.0 \times 10^{-11}}\)\({3.8 \times 10^{-12}}\)goal
excluded
\({6.9 \times 10^{-11}}\)\({6.0 \times 10^{-13}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({1.8 \times 10^{-9}}\)\({4.7 \times 10^{-12}}\)goal
\({2.2 \times 10^{-10}}\)\({3.0 \times 10^{-12}}\)goal
refusednot physical

\({(\gamma, \beta) = (1, 5)}\)
\({8.7 \times 10^{-11}}\)\({1.5 \times 10^{-13}}\)goal
\({7.7 \times 10^{-11}}\)\({5.7 \times 10^{-14}}\)goal
\({1.2 \times 10^{-9}}\)\({3.9 \times 10^{-13}}\)goal
\({1.1 \times 10^{-9}}\)\({3.8 \times 10^{-12}}\)goal
\({1.7 \times 10^{-13}}\)\({2.8 \times 10^{-15}}\)goal
\({8.7 \times 10^{-11}}\)\({9.4 \times 10^{-13}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({1.2 \times 10^{-9}}\)\({3.0 \times 10^{-12}}\)goal
\({1.1 \times 10^{-9}}\)\({3.9 \times 10^{-12}}\)goal
goalsupport above \({P(R_{\rm a})}\)

\({(\gamma, \beta) = (1.5, 5)}\)
\({1.4 \times 10^{-10}}\)\({1.8 \times 10^{-13}}\)goal
\({1.3 \times 10^{-10}}\)\({5.0 \times 10^{-14}}\)goal
\({3.3 \times 10^{-10}}\)\({4.6 \times 10^{-13}}\)goal
\({5.6 \times 10^{-10}}\)\({3.8 \times 10^{-12}}\)goal
\({2.3 \times 10^{-11}}\)\({1.8 \times 10^{-14}}\)goal
\({1.4 \times 10^{-10}}\)\({1.3 \times 10^{-12}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({3.3 \times 10^{-10}}\)\({3.3 \times 10^{-12}}\)goal
\({5.6 \times 10^{-10}}\)\({2.9 \times 10^{-12}}\)goal
refusednot physical

\({(\gamma, \beta) = (1.9, 5)}\)
\({3.6 \times 10^{-11}}\)\({2.1 \times 10^{-13}}\)goal
\({2.7 \times 10^{-12}}\)\({6.3 \times 10^{-14}}\)goal
\({1.9 \times 10^{-11}}\)\({1.1 \times 10^{-12}}\)goal
\({8.7 \times 10^{-11}}\)\({3.8 \times 10^{-12}}\)goal
\({8.8 \times 10^{-13}}\)\({2.6 \times 10^{-14}}\)goal
\({5.0 \times 10^{-10}}\)\({2.3 \times 10^{-12}}\)goal
ceilingsupport above \({P(R_{\rm a})}\)
\({4.2 \times 10^{-11}}\)\({3.3 \times 10^{-12}}\)ceiling
\({8.5 \times 10^{-11}}\)\({3.2 \times 10^{-12}}\)goal
refusednot physical

\({(\gamma, \beta) = (1.99, 5)}\)
\({5.5 \times 10^{-10}}\)\({6.3 \times 10^{-12}}\)refused
\({3.2 \times 10^{-11}}\)\({9.1 \times 10^{-13}}\)goal
\({2.6 \times 10^{-10}}\)\({4.1 \times 10^{-12}}\)goal
\({2.5 \times 10^{-10}}\)\({6.0 \times 10^{-12}}\)goal
\({7.2 \times 10^{-12}}\)\({3.0 \times 10^{-13}}\)goal
\({1.5 \times 10^{-8}}\)\({2.1 \times 10^{-10}}\)refused
refusedAccuracyError†
\({2.0 \times 10^{-9}}\)\({5.5 \times 10^{-12}}\)refused†
\({1.1 \times 10^{-10}}\)\({7.4 \times 10^{-12}}\)ceiling
refusednot physical

Plummer (\({\alpha = 2, \beta = 5, \gamma = 0}\))
\({1.1 \times 10^{-12}}\)\({9.4 \times 10^{-15}}\)goal
\({4.2 \times 10^{-11}}\)\({1.4 \times 10^{-14}}\)goal
\({2.7 \times 10^{-9}}\)\({1.7 \times 10^{-13}}\)goal
excluded
excluded
\({7.2 \times 10^{-12}}\)\({8.3 \times 10^{-15}}\)goal
goalsupport above \({P(R_{\rm a})}\)
\({2.7 \times 10^{-9}}\)\({7.6 \times 10^{-12}}\)goal
excluded
goalsupport above \({P(R_{\rm a})}\)

### Table 2. The anisotropy radius

Osipkov-Merritt (OM) and Cuddeford-Osipkov-Merritt (COM, \({\beta_{\rm c} = -0.25}\)) models of Type I and Type II on three profiles, from \({R_{\rm a} = 0.3}\) to 1000.

\({R_{\rm a} = 0.3}\)
\({R_{\rm a} = 1}\)
\({R_{\rm a} = 3}\)
\({R_{\rm a} = 10}\)
\({R_{\rm a} = 30}\)
\({R_{\rm a} = 100}\)
\({R_{\rm a} = 1000}\)

NFW · OM I
refused
\({1.5 \times 10^{-10}}\)\({4.1 \times 10^{-11}}\)ceiling
\({1.2 \times 10^{-10}}\)\({3.8 \times 10^{-11}}\)ceiling
\({9.0 \times 10^{-11}}\)\({3.7 \times 10^{-11}}\)ceiling
\({9.0 \times 10^{-11}}\)\({3.4 \times 10^{-11}}\)ceiling
\({9.0 \times 10^{-11}}\)\({3.3 \times 10^{-11}}\)ceiling
\({9.9 \times 10^{-9}}\)\({4.5 \times 10^{-11}}\)ceiling

NFW · OM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

NFW · COM I
refused
\({5.9 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({5.8 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({5.8 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({5.7 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({5.7 \times 10^{-9}}\)\({2.9 \times 10^{-9}}\)goal
\({5.9 \times 10^{-9}}\)\({1.5 \times 10^{-9}}\)goal

NFW · COM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

cored (\({\gamma = 0, \beta = 4}\)) · OM I
refused
\({3.8 \times 10^{-11}}\)\({1.2 \times 10^{-11}}\)goal
\({1.7 \times 10^{-11}}\)\({6.8 \times 10^{-13}}\)goal
\({1.7 \times 10^{-11}}\)\({1.4 \times 10^{-12}}\)goal
\({1.8 \times 10^{-11}}\)\({9.8 \times 10^{-13}}\)goal
\({1.4 \times 10^{-10}}\)\({3.6 \times 10^{-12}}\)goal
\({1.7 \times 10^{-10}}\)\({1.2 \times 10^{-11}}\)goal

cored (\({\gamma = 0, \beta = 4}\)) · OM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

cored (\({\gamma = 0, \beta = 4}\)) · COM I
refused
\({2.0 \times 10^{-9}}\)\({3.7 \times 10^{-13}}\)goal
\({2.0 \times 10^{-9}}\)\({2.1 \times 10^{-12}}\)goal
\({6.7 \times 10^{-9}}\)\({1.4 \times 10^{-10}}\)goal
\({1.5 \times 10^{-8}}\)\({2.0 \times 10^{-10}}\)goal
\({2.0 \times 10^{-9}}\)\({1.8 \times 10^{-11}}\)goal
\({2.1 \times 10^{-9}}\)\({2.8 \times 10^{-11}}\)goal

cored (\({\gamma = 0, \beta = 4}\)) · COM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

steep (\({\gamma = 1.5, \beta = 4}\)) · OM I
\({1.7 \times 10^{-10}}\)\({6.3 \times 10^{-13}}\)goal
\({1.7 \times 10^{-10}}\)\({2.2 \times 10^{-13}}\)goal
\({1.7 \times 10^{-10}}\)\({3.6 \times 10^{-13}}\)goal
\({1.7 \times 10^{-10}}\)\({9.9 \times 10^{-13}}\)goal
\({3.5 \times 10^{-10}}\)\({1.5 \times 10^{-12}}\)goal
\({1.7 \times 10^{-10}}\)\({1.5 \times 10^{-12}}\)goal
\({1.7 \times 10^{-10}}\)\({3.6 \times 10^{-12}}\)goal

steep (\({\gamma = 1.5, \beta = 4}\)) · OM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

steep (\({\gamma = 1.5, \beta = 4}\)) · COM I
\({2.6 \times 10^{-8}}\)\({6.9 \times 10^{-13}}\)goal
\({2.6 \times 10^{-8}}\)\({3.9 \times 10^{-13}}\)goal
\({2.6 \times 10^{-8}}\)\({9.4 \times 10^{-11}}\)goal
\({2.6 \times 10^{-8}}\)\({1.3 \times 10^{-11}}\)goal
\({2.6 \times 10^{-8}}\)\({1.8 \times 10^{-11}}\)goal
\({2.6 \times 10^{-8}}\)\({9.6 \times 10^{-12}}\)goal
\({2.6 \times 10^{-8}}\)\({9.5 \times 10^{-12}}\)goal

steep (\({\gamma = 1.5, \beta = 4}\)) · COM II
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)
goalsupport above \({P(R_{\rm a})}\)

### Table 3. Truncations

Method I and Method II truncations at energy \({\mathcal{K}_{\rm t}}\) on four parents under six anisotropy models.

\({\mathcal{K}_{\rm t} = 0.5}\)
\({\mathcal{K}_{\rm t} = 0.2}\)
\({\mathcal{K}_{\rm t} = 0.05}\)
\({\mathcal{K}_{\rm t} = 0.01}\)
\({\mathcal{K}_{\rm t} = 0.002}\)

NFW · isotropic · Method I
\({1.0 \times 10^{-7}}\)\({1.7 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({2.6 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({5.6 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({1.4 \times 10^{-11}}\)goal
\({8.3 \times 10^{-7}}\)\({5.7 \times 10^{-11}}\)goal

NFW · isotropic · Method II
\({1.0 \times 10^{-7}}\)\({1.2 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({1.4 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({8.9 \times 10^{-11}}\)goal
\({1.0 \times 10^{-7}}\)\({7.3 \times 10^{-11}}\)goal
\({8.3 \times 10^{-7}}\)\({1.2 \times 10^{-10}}\)goal

NFW · \({\beta_{\rm c} = -0.50}\) · Method I
\({2.0 \times 10^{-7}}\)\({1.4 \times 10^{-11}}\)goal
\({2.0 \times 10^{-7}}\)\({8.4 \times 10^{-12}}\)ceiling
\({2.0 \times 10^{-7}}\)\({1.1 \times 10^{-11}}\)goal
\({2.0 \times 10^{-7}}\)\({2.0 \times 10^{-11}}\)goal
\({2.2 \times 10^{-6}}\)\({5.4 \times 10^{-11}}\)ceiling

NFW · \({\beta_{\rm c} = -0.50}\) · Method II
\({2.2 \times 10^{-7}}\)\({2.3 \times 10^{-9}}\)goal
\({2.2 \times 10^{-7}}\)\({1.2 \times 10^{-9}}\)goal
\({2.2 \times 10^{-7}}\)\({6.1 \times 10^{-10}}\)goal
\({2.2 \times 10^{-7}}\)\({3.3 \times 10^{-10}}\)goal
\({8.3 \times 10^{-7}}\)\({1.9 \times 10^{-10}}\)goal

NFW · \({\beta_{\rm c} = 0.25}\) · Method I
\({3.6 \times 10^{-8}}\)\({1.3 \times 10^{-12}}\)goal
\({5.5 \times 10^{-8}}\)\({2.0 \times 10^{-12}}\)goal
\({5.4 \times 10^{-8}}\)\({4.6 \times 10^{-12}}\)goal
\({5.3 \times 10^{-8}}\)\({1.3 \times 10^{-11}}\)goal
\({4.2 \times 10^{-7}}\)\({6.6 \times 10^{-11}}\)goal

NFW · \({\beta_{\rm c} = 0.25}\) · Method II
\({4.1 \times 10^{-8}}\)\({1.9 \times 10^{-9}}\)goal
\({4.1 \times 10^{-8}}\)\({5.9 \times 10^{-10}}\)goal
\({4.1 \times 10^{-8}}\)\({7.0 \times 10^{-10}}\)goal
\({4.0 \times 10^{-8}}\)\({6.9 \times 10^{-10}}\)goal
\({8.3 \times 10^{-7}}\)\({5.1 \times 10^{-10}}\)goal

NFW · OM I \({R_{\rm a} = 10}\) · Method I
\({1.0 \times 10^{-7}}\)\({1.8 \times 10^{-12}}\)ceiling
\({1.0 \times 10^{-7}}\)\({6.2 \times 10^{-12}}\)ceiling
refusedNO EDGE
refusedUnphysicalModelError
refusedUnphysicalModelError

NFW · OM I \({R_{\rm a} = 10}\) · Method II
\({1.0 \times 10^{-7}}\)\({1.7 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({9.7 \times 10^{-11}}\)goal
\({1.0 \times 10^{-7}}\)\({5.9 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({8.0 \times 10^{-10}}\)goal
\({8.3 \times 10^{-7}}\)\({5.1 \times 10^{-10}}\)goal

NFW · OM II \({R_{\rm a} = 10}\) · Method I
\({4.6 \times 10^{-7}}\)\({5.9 \times 10^{-11}}\)goal
refusedDOMAIN
refusedDOMAIN
refusedDOMAIN
refusedDOMAIN

NFW · OM II \({R_{\rm a} = 10}\) · Method II
\({1.0 \times 10^{-7}}\)\({2.2 \times 10^{-10}}\)goal
refusedInputError
refusedInputError
refusedInputError
refusedInputError

NFW · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({1.5 \times 10^{-7}}\)\({4.3 \times 10^{-12}}\)goal
\({1.5 \times 10^{-7}}\)\({6.7 \times 10^{-12}}\)goal
refusedNO EDGE
refusedUnphysicalModelError
refusedUnphysicalModelError

NFW · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({1.7 \times 10^{-7}}\)\({1.7 \times 10^{-9}}\)goal
\({1.7 \times 10^{-7}}\)\({9.7 \times 10^{-10}}\)goal
\({1.7 \times 10^{-7}}\)\({9.1 \times 10^{-10}}\)goal
\({1.7 \times 10^{-7}}\)\({8.8 \times 10^{-10}}\)goal
\({8.3 \times 10^{-7}}\)\({4.5 \times 10^{-10}}\)goal

Hernquist · isotropic · Method I
\({1.1 \times 10^{-8}}\)\({1.6 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({2.5 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({6.2 \times 10^{-12}}\)goal
\({1.0 \times 10^{-7}}\)\({2.0 \times 10^{-11}}\)goal
\({1.3 \times 10^{-6}}\)\({6.5 \times 10^{-11}}\)ceiling

Hernquist · isotropic · Method II
\({2.1 \times 10^{-8}}\)\({1.1 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({1.0 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({9.2 \times 10^{-11}}\)goal
\({1.0 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({1.3 \times 10^{-10}}\)ceiling

Hernquist · \({\beta_{\rm c} = -0.50}\) · Method I
\({2.0 \times 10^{-7}}\)\({1.3 \times 10^{-11}}\)goal
\({2.0 \times 10^{-7}}\)\({8.7 \times 10^{-12}}\)ceiling
\({2.0 \times 10^{-7}}\)\({1.1 \times 10^{-11}}\)goal
\({2.0 \times 10^{-7}}\)\({2.7 \times 10^{-11}}\)goal
\({2.7 \times 10^{-6}}\)\({6.4 \times 10^{-11}}\)ceiling

Hernquist · \({\beta_{\rm c} = -0.50}\) · Method II
\({4.3 \times 10^{-8}}\)\({2.2 \times 10^{-9}}\)goal
\({2.2 \times 10^{-7}}\)\({1.9 \times 10^{-9}}\)goal
\({2.2 \times 10^{-7}}\)\({1.2 \times 10^{-9}}\)goal
\({2.2 \times 10^{-7}}\)\({6.4 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({2.8 \times 10^{-10}}\)ceiling

Hernquist · \({\beta_{\rm c} = 0.25}\) · Method I
\({7.3 \times 10^{-9}}\)\({9.9 \times 10^{-13}}\)goal
\({7.5 \times 10^{-9}}\)\({2.1 \times 10^{-12}}\)goal
\({5.4 \times 10^{-8}}\)\({5.2 \times 10^{-12}}\)goal
\({5.4 \times 10^{-8}}\)\({1.9 \times 10^{-11}}\)goal
\({8.1 \times 10^{-7}}\)\({6.9 \times 10^{-11}}\)goal

Hernquist · \({\beta_{\rm c} = 0.25}\) · Method II
\({3.2 \times 10^{-8}}\)\({2.7 \times 10^{-9}}\)goal
\({4.1 \times 10^{-8}}\)\({1.5 \times 10^{-9}}\)goal
\({4.0 \times 10^{-8}}\)\({8.3 \times 10^{-10}}\)goal
\({5.2 \times 10^{-8}}\)\({1.0 \times 10^{-9}}\)goal
\({1.3 \times 10^{-6}}\)\({6.8 \times 10^{-10}}\)ceiling

Hernquist · OM I \({R_{\rm a} = 10}\) · Method I
\({1.1 \times 10^{-8}}\)\({1.6 \times 10^{-12}}\)ceiling
\({1.0 \times 10^{-7}}\)\({2.5 \times 10^{-12}}\)ceiling
\({1.0 \times 10^{-7}}\)\({6.1 \times 10^{-12}}\)ceiling
\({1.0 \times 10^{-7}}\)\({2.4 \times 10^{-11}}\)ceiling
refusedNO EDGE

Hernquist · OM I \({R_{\rm a} = 10}\) · Method II
\({2.1 \times 10^{-8}}\)\({8.0 \times 10^{-11}}\)goal
\({1.0 \times 10^{-7}}\)\({2.4 \times 10^{-11}}\)goal
\({1.0 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({3.1 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({2.9 \times 10^{-10}}\)ceiling

Hernquist · OM II \({R_{\rm a} = 10}\) · Method I
\({4.6 \times 10^{-7}}\)\({5.9 \times 10^{-11}}\)goal
\({1.8 \times 10^{-7}}\)\({6.4 \times 10^{-11}}\)goal
refusedDOMAIN
refusedDOMAIN
refusedDOMAIN

Hernquist · OM II \({R_{\rm a} = 10}\) · Method II
\({2.1 \times 10^{-8}}\)\({2.7 \times 10^{-10}}\)goal
\({1.0 \times 10^{-7}}\)\({2.8 \times 10^{-10}}\)goal
refusedInputError
refusedInputError
refusedInputError

Hernquist · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({1.6 \times 10^{-7}}\)\({3.7 \times 10^{-12}}\)goal
\({1.6 \times 10^{-7}}\)\({3.8 \times 10^{-12}}\)goal
\({1.5 \times 10^{-7}}\)\({1.3 \times 10^{-11}}\)goal
\({1.5 \times 10^{-7}}\)\({2.9 \times 10^{-11}}\)goal
refusedNO EDGE

Hernquist · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({3.3 \times 10^{-8}}\)\({1.8 \times 10^{-9}}\)goal
\({1.7 \times 10^{-7}}\)\({1.4 \times 10^{-9}}\)goal
\({1.7 \times 10^{-7}}\)\({1.2 \times 10^{-9}}\)goal
\({1.7 \times 10^{-7}}\)\({1.1 \times 10^{-9}}\)goal
\({1.3 \times 10^{-6}}\)\({9.9 \times 10^{-10}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · isotropic · Method I
\({2.2 \times 10^{-3}}\)\({4.3 \times 10^{-12}}\)warning
\({1.8 \times 10^{-4}}\)\({5.7 \times 10^{-12}}\)warning
\({5.2 \times 10^{-6}}\)\({1.2 \times 10^{-11}}\)ceiling
\({9.2 \times 10^{-8}}\)\({3.2 \times 10^{-11}}\)goal
\({1.3 \times 10^{-6}}\)\({7.5 \times 10^{-11}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · isotropic · Method II
\({2.2 \times 10^{-3}}\)\({1.7 \times 10^{-10}}\)warning
\({1.8 \times 10^{-4}}\)\({1.9 \times 10^{-10}}\)warning
\({5.2 \times 10^{-6}}\)\({1.6 \times 10^{-10}}\)ceiling
\({9.5 \times 10^{-8}}\)\({1.8 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({1.5 \times 10^{-10}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · \({\beta_{\rm c} = -0.50}\) · Method I
\({1.4 \times 10^{-3}}\)\({2.1 \times 10^{-11}}\)warning
\({5.3 \times 10^{-5}}\)\({1.6 \times 10^{-11}}\)ceiling
\({6.9 \times 10^{-7}}\)\({1.7 \times 10^{-11}}\)goal
\({1.1 \times 10^{-7}}\)\({3.4 \times 10^{-11}}\)goal
\({2.7 \times 10^{-6}}\)\({6.7 \times 10^{-11}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · \({\beta_{\rm c} = -0.50}\) · Method II
refusedAccuracyError
refusedAccuracyError
\({4.7 \times 10^{-6}}\)\({7.0 \times 10^{-10}}\)warning
\({1.3 \times 10^{-7}}\)\({3.6 \times 10^{-10}}\)ceiling
\({1.3 \times 10^{-6}}\)\({2.1 \times 10^{-10}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · \({\beta_{\rm c} = 0.25}\) · Method I
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY

cored (\({\gamma = 0, \beta = 4}\)) · \({\beta_{\rm c} = 0.25}\) · Method II
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY

cored (\({\gamma = 0, \beta = 4}\)) · OM I \({R_{\rm a} = 10}\) · Method I
\({2.3 \times 10^{-3}}\)\({4.2 \times 10^{-12}}\)warning
\({2.2 \times 10^{-4}}\)\({5.9 \times 10^{-12}}\)warning
\({2.9 \times 10^{-5}}\)\({9.3 \times 10^{-12}}\)ceiling
refusedNO EDGE
refusedNO EDGE

cored (\({\gamma = 0, \beta = 4}\)) · OM I \({R_{\rm a} = 10}\) · Method II
\({2.2 \times 10^{-3}}\)\({3.8 \times 10^{-10}}\)warning
\({1.8 \times 10^{-4}}\)\({4.9 \times 10^{-10}}\)warning
\({5.2 \times 10^{-6}}\)\({3.4 \times 10^{-10}}\)ceiling
\({9.5 \times 10^{-8}}\)\({8.6 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({4.1 \times 10^{-10}}\)ceiling

cored (\({\gamma = 0, \beta = 4}\)) · OM II \({R_{\rm a} = 10}\) · Method I
\({2.2 \times 10^{-3}}\)\({1.0 \times 10^{-10}}\)warning
refusedTYPE II EDGE
refusedDOMAIN
refusedDOMAIN
refusedDOMAIN

cored (\({\gamma = 0, \beta = 4}\)) · OM II \({R_{\rm a} = 10}\) · Method II
\({2.2 \times 10^{-3}}\)\({6.8 \times 10^{-10}}\)warning
refusedInputError
refusedInputError
refusedInputError
refusedInputError

cored (\({\gamma = 0, \beta = 4}\)) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({9.5 \times 10^{-3}}\)\({7.2 \times 10^{-12}}\)warning
\({6.8 \times 10^{-4}}\)\({9.9 \times 10^{-12}}\)warning
\({7.7 \times 10^{-5}}\)\({4.1 \times 10^{-11}}\)ceiling
refusedNO EDGE
refusedNO EDGE

cored (\({\gamma = 0, \beta = 4}\)) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
refusedAccuracyError
\({1.9 \times 10^{-4}}\)\({1.3 \times 10^{-9}}\)warning
\({5.5 \times 10^{-6}}\)\({1.1 \times 10^{-9}}\)warning
\({1.8 \times 10^{-7}}\)\({8.5 \times 10^{-10}}\)ceiling
\({1.3 \times 10^{-6}}\)\({7.6 \times 10^{-10}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · isotropic · Method I
\({9.3 \times 10^{-9}}\)\({4.0 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({2.5 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({8.2 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({2.7 \times 10^{-11}}\)goal
\({1.3 \times 10^{-6}}\)\({8.1 \times 10^{-11}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · isotropic · Method II
\({9.3 \times 10^{-9}}\)\({1.8 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({1.1 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({1.2 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({1.5 \times 10^{-10}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · \({\beta_{\rm c} = -0.50}\) · Method I
\({5.1 \times 10^{-7}}\)\({2.9 \times 10^{-11}}\)goal
\({4.9 \times 10^{-7}}\)\({9.3 \times 10^{-12}}\)ceiling
\({4.9 \times 10^{-7}}\)\({1.5 \times 10^{-11}}\)goal
\({4.9 \times 10^{-7}}\)\({3.6 \times 10^{-11}}\)goal
\({2.7 \times 10^{-6}}\)\({8.5 \times 10^{-11}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · \({\beta_{\rm c} = -0.50}\) · Method II
\({2.7 \times 10^{-8}}\)\({1.9 \times 10^{-9}}\)goal
\({5.1 \times 10^{-7}}\)\({2.4 \times 10^{-9}}\)goal
\({5.1 \times 10^{-7}}\)\({1.8 \times 10^{-9}}\)goal
\({5.1 \times 10^{-7}}\)\({9.8 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({3.6 \times 10^{-10}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · \({\beta_{\rm c} = 0.25}\) · Method I
\({8.3 \times 10^{-9}}\)\({3.3 \times 10^{-12}}\)goal
\({2.1 \times 10^{-7}}\)\({2.2 \times 10^{-12}}\)goal
\({2.0 \times 10^{-7}}\)\({7.6 \times 10^{-12}}\)goal
\({2.0 \times 10^{-7}}\)\({2.8 \times 10^{-11}}\)goal
\({8.1 \times 10^{-7}}\)\({9.2 \times 10^{-11}}\)goal

steep (\({\gamma = 1.5, \beta = 4}\)) · \({\beta_{\rm c} = 0.25}\) · Method II
\({1.3 \times 10^{-8}}\)\({1.7 \times 10^{-9}}\)goal
\({1.9 \times 10^{-7}}\)\({2.1 \times 10^{-9}}\)goal
\({1.9 \times 10^{-7}}\)\({8.1 \times 10^{-10}}\)goal
\({1.9 \times 10^{-7}}\)\({1.0 \times 10^{-9}}\)goal
\({1.3 \times 10^{-6}}\)\({6.8 \times 10^{-10}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · OM I \({R_{\rm a} = 10}\) · Method I
\({9.3 \times 10^{-9}}\)\({4.1 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({2.6 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({8.4 \times 10^{-12}}\)goal
\({3.0 \times 10^{-7}}\)\({2.5 \times 10^{-11}}\)goal
refusedNO EDGE

steep (\({\gamma = 1.5, \beta = 4}\)) · OM I \({R_{\rm a} = 10}\) · Method II
\({9.3 \times 10^{-9}}\)\({1.9 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({1.0 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({1.2 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({2.0 \times 10^{-10}}\)goal
\({1.3 \times 10^{-6}}\)\({2.5 \times 10^{-10}}\)ceiling

steep (\({\gamma = 1.5, \beta = 4}\)) · OM II \({R_{\rm a} = 10}\) · Method I
\({4.6 \times 10^{-7}}\)\({6.3 \times 10^{-11}}\)goal
\({1.2 \times 10^{-7}}\)\({5.4 \times 10^{-11}}\)goal
refusedTYPE II EDGE
refusedDOMAIN
refusedDOMAIN

steep (\({\gamma = 1.5, \beta = 4}\)) · OM II \({R_{\rm a} = 10}\) · Method II
\({9.3 \times 10^{-9}}\)\({3.4 \times 10^{-10}}\)goal
\({3.0 \times 10^{-7}}\)\({2.7 \times 10^{-10}}\)goal
refusedInputError
refusedInputError
refusedInputError

steep (\({\gamma = 1.5, \beta = 4}\)) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method I
\({4.2 \times 10^{-7}}\)\({5.4 \times 10^{-12}}\)goal
\({4.0 \times 10^{-7}}\)\({2.9 \times 10^{-12}}\)goal
\({4.0 \times 10^{-7}}\)\({9.4 \times 10^{-12}}\)goal
\({4.0 \times 10^{-7}}\)\({2.9 \times 10^{-11}}\)goal
refusedNO EDGE

steep (\({\gamma = 1.5, \beta = 4}\)) · COM I \({(\beta_{\rm c}, R_{\rm a}) = (-0.25, 10)}\) · Method II
\({2.0 \times 10^{-8}}\)\({1.9 \times 10^{-9}}\)goal
\({4.1 \times 10^{-7}}\)\({1.9 \times 10^{-9}}\)goal
\({4.1 \times 10^{-7}}\)\({1.5 \times 10^{-9}}\)goal
\({4.1 \times 10^{-7}}\)\({1.0 \times 10^{-9}}\)goal
\({1.3 \times 10^{-6}}\)\({5.1 \times 10^{-10}}\)ceiling

### Table 4. Edges

Profiles at the edges of the conditions: the cusp bound, the degenerate and steep constant anisotropies, a small anisotropy radius inside a deep truncation, and the outer slope 2.5 profile, which is refused.

\({\beta_{\rm c} = \gamma/2}\)
\({\beta_{\rm c} = -1/2 + 10^{-7}}\)
\({\beta_{\rm c} = -1}\)
\({\beta_{\rm c} = -1.5}\)
\({\beta_{\rm c} = +0.75}\)
OM I \({R_{\rm a} = 0.3}\) \({\mathcal{K}_{\rm t} = 0.01}\)
\({\beta_{\rm c} = -1/2}\) \({\mathcal{K}_{\rm t} = 0.3}\)

\({(\gamma, \beta) = (0, 4)}\)
\({1.7 \times 10^{-11}}\)\({1.3 \times 10^{-13}}\)goal
\({1.9 \times 10^{-9}}\)\({2.3 \times 10^{-13}}\)goal
\({3.7 \times 10^{-9}}\)\({3.0 \times 10^{-13}}\)goal
\({5.6 \times 10^{-9}}\)\({1.2 \times 10^{-13}}\)goal
excluded
refusedUnphysicalModelError
\({2.1 \times 10^{-4}}\)\({1.7 \times 10^{-11}}\)warning

\({(\gamma, \beta) = (0.5, 4)}\)
\({1.2 \times 10^{-11}}\)\({1.9 \times 10^{-12}}\)goal
\({1.3 \times 10^{-10}}\)\({1.9 \times 10^{-13}}\)goal
\({1.9 \times 10^{-10}}\)\({3.0 \times 10^{-13}}\)goal
\({1.6 \times 10^{-10}}\)\({9.7 \times 10^{-14}}\)goal
excluded
refusedUnphysicalModelError
\({1.0 \times 10^{-7}}\)\({1.1 \times 10^{-11}}\)goal

\({(\gamma, \beta) = (1, 4)}\)
\({1.0 \times 10^{-14}}\)\({2.8 \times 10^{-15}}\)goal
\({3.9 \times 10^{-11}}\)\({1.8 \times 10^{-13}}\)goal
\({1.9 \times 10^{-10}}\)\({2.7 \times 10^{-13}}\)ceiling
\({1.3 \times 10^{-10}}\)\({7.7 \times 10^{-14}}\)goal
excluded
refusedUnphysicalModelError
\({2.0 \times 10^{-7}}\)\({7.0 \times 10^{-12}}\)ceiling

\({(\gamma, \beta) = (1.5, 4)}\)
\({1.5 \times 10^{-11}}\)\({1.9 \times 10^{-13}}\)goal
\({1.7 \times 10^{-10}}\)\({1.5 \times 10^{-13}}\)goal
\({3.3 \times 10^{-10}}\)\({2.4 \times 10^{-13}}\)goal
\({2.4 \times 10^{-10}}\)\({7.5 \times 10^{-14}}\)goal
\({1.5 \times 10^{-11}}\)\({1.9 \times 10^{-13}}\)goal
refusedUnphysicalModelError
\({4.9 \times 10^{-7}}\)\({1.3 \times 10^{-11}}\)ceiling

\({(\gamma, \beta) = (1.9, 4)}\)
\({3.8 \times 10^{-10}}\)\({2.1 \times 10^{-11}}\)goal
\({7.0 \times 10^{-12}}\)\({2.0 \times 10^{-13}}\)goal
\({2.3 \times 10^{-11}}\)\({2.2 \times 10^{-13}}\)goal
\({3.9 \times 10^{-12}}\)\({1.1 \times 10^{-13}}\)goal
\({3.0 \times 10^{-10}}\)\({1.6 \times 10^{-13}}\)ceiling
refusedUnphysicalModelError
refusedAccuracyError

\({(\gamma, \beta) = (1.99, 4)}\)
refusedCENTRAL SERIES
\({6.5 \times 10^{-10}}\)\({8.3 \times 10^{-12}}\)goal
\({2.8 \times 10^{-10}}\)\({8.8 \times 10^{-12}}\)goal
\({5.7 \times 10^{-11}}\)\({1.7 \times 10^{-12}}\)ceiling
\({8.9 \times 10^{-10}}\)\({1.9 \times 10^{-12}}\)ceiling
refusedInputError
refusedInputError

Plummer (\({\alpha = 2, \beta = 5, \gamma = 0}\))
\({1.1 \times 10^{-12}}\)\({9.4 \times 10^{-15}}\)goal
\({1.5 \times 10^{-10}}\)\({1.2 \times 10^{-13}}\)goal
\({3.1 \times 10^{-10}}\)\({2.4 \times 10^{-13}}\)goal
\({7.9 \times 10^{-11}}\)\({3.1 \times 10^{-14}}\)goal
excluded
refusedCENTRAL SIGN
\({9.7 \times 10^{-3}}\)\({2.2 \times 10^{-11}}\)warning

\({(\gamma, \beta) = (2, 4)}\)
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError

\({(\gamma, \beta) = (1, 2.5)}\)
refusedOUTER SLOPE
refusedOUTER SLOPE
refusedOUTER SLOPE
refusedOUTER SLOPE
refusedOUTER SLOPE
refusedOUTER SLOPE
refusedOUTER SLOPE
