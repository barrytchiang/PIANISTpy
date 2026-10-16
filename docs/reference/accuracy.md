# Accuracy tables

Source: https://pianistpy.com/docs/reference/accuracy.html

# Accuracy tables

The accuracy grid is a fixed set of models, four tables of rows by columns, run unchanged at every release. Each
model’s distribution function is inverted, its density computed back from it, and the roundtrip error
|rho_DF / rho - 1| read at 513 radii; a cell gives that error over 1e-4 < r < 1e5 (scale radii):

its maximum, then its median, then the tier the result itself reports (goal: 1e-6 or less;
ceiling: 1e-4 or less; warning: 1e-2 or less; refused: beyond, see Accuracy);

refused and the refusal’s name, where the package refuses the model (the full message is in the cell’s
tooltip and in the CSV);

excluded where the anisotropy breaks the cusp bound gamma >= 2 beta_c (computed, and recorded in the
CSV). A Type II model without a truncation has its distribution function only above the energy P(R_a): its
cell gives the tier the package reports over that support and the note “support above P(R_a)”, not the
density’s roundtrip, which is no measure there.

A † marks a cell whose outcome is not the one the grid expects. The tables are at the default tier; the CSV holds
every range the grid reads, the radius of each maximum and the compute times. The tables of earlier releases are
on Accuracy by release.

The grid of version 0.9.0 (commit f331d969, cpu), the default tier: the full grid as CSV and its member list.

### Table 1. Profiles by anisotropy

The double power law at inner slope gamma and outer slope beta (alpha 1), and the Plummer profile, under ten anisotropy models; untruncated.

isotropic
βc −0.50
βc −0.25
βc +0.25
βc +0.50
OM I Ra 10
OM II Ra 10
COM I (βc −0.25, Ra 10)
COM I (βc +0.25, Ra 10)
COM II (βc −0.25, Ra 10)

γ 0, β 3
2.8e-112.1e-13goal
2.6e-81.5e-13goal
1.6e-88.8e-13goal
excluded
excluded
1.2e-102.0e-11goal
goalsupport above P(Ra)
1.4e-81.2e-9goal
excluded
warningsupport above P(Ra)

γ 0.5, β 3
8.0e-111.7e-13goal
1.1e-101.3e-13goal
2.6e-96.8e-13goal
1.7e-113.3e-12goal
excluded
9.4e-113.7e-11goal
goalsupport above P(Ra)
3.0e-91.7e-9goal
1.8e-101.7e-11goal
warningsupport above P(Ra)

γ 1, β 3
8.8e-111.5e-13goal
7.3e-111.1e-13goal
1.8e-97.0e-13goal
1.6e-92.9e-12goal
1.1e-134.2e-15goal
9.0e-113.7e-11ceiling
goalsupport above P(Ra)
5.8e-92.9e-9goal
1.6e-93.5e-11goal
warningsupport above P(Ra)

γ 1.5, β 3
1.5e-101.2e-13goal
1.3e-108.3e-14goal
6.5e-104.3e-13goal
9.5e-102.5e-12goal
2.9e-112.5e-14goal
1.8e-102.8e-11goal
goalsupport above P(Ra)
2.6e-83.0e-9goal
9.5e-103.0e-11goal
warningsupport above P(Ra)

γ 1.9, β 3
1.2e-121.1e-13goal
1.3e-127.0e-14goal
3.2e-115.9e-13goal
1.3e-101.8e-12goal
7.4e-142.4e-14goal
7.3e-112.5e-11ceiling
goalsupport above P(Ra)
9.3e-85.1e-9ceiling
1.3e-102.4e-11goal
refusedsupport above P(Ra)

γ 1.99, β 3
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedInputError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedValueError†

γ 0, β 4
1.7e-111.3e-13goal
1.9e-96.3e-14goal
2.9e-92.0e-13goal
excluded
excluded
1.7e-111.4e-12goal
goalsupport above P(Ra)
6.7e-91.4e-10goal
excluded
warningsupport above P(Ra)

γ 0.5, β 4
7.1e-111.2e-13goal
7.8e-115.0e-14goal
2.2e-91.8e-13goal
1.2e-111.9e-12goal
excluded
7.2e-119.0e-13goal
goalsupport above P(Ra)
8.7e-91.2e-10goal
1.7e-104.1e-12goal
warningsupport above P(Ra)

γ 1, β 4
8.6e-111.2e-13goal
7.2e-113.9e-14goal
1.4e-91.8e-13goal
1.3e-91.8e-12goal
1.0e-142.8e-15goal
8.6e-115.6e-13ceiling
goalsupport above P(Ra)
1.2e-88.9e-11goal
1.3e-91.8e-11goal
warningsupport above P(Ra)

γ 1.5, β 4
1.4e-101.0e-13goal
1.3e-104.3e-14goal
4.4e-102.0e-13goal
6.8e-101.8e-12goal
2.3e-117.0e-15goal
1.7e-109.9e-13goal
goalsupport above P(Ra)
2.6e-81.3e-11goal
6.8e-101.2e-11goal
warningsupport above P(Ra)

γ 1.9, β 4
1.1e-111.1e-13goal
2.1e-126.2e-14goal
2.3e-115.1e-13goal
1.0e-101.8e-12goal
2.5e-132.3e-14goal
3.8e-92.2e-12ceiling
goalsupport above P(Ra)
3.9e-94.9e-11ceiling
9.8e-113.1e-12goal
refusedsupport above P(Ra)

γ 1.99, β 4
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedInputError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedValueError†

γ 0, β 5
1.5e-111.7e-13goal
1.5e-94.3e-14goal
2.7e-93.7e-13goal
excluded
excluded
1.6e-117.5e-13goal
goalsupport above P(Ra)
2.7e-99.4e-12goal
excluded
warningsupport above P(Ra)

γ 0.5, β 5
6.9e-111.7e-13goal
7.2e-114.9e-14goal
1.9e-93.7e-13goal
1.0e-113.8e-12goal
excluded
6.9e-116.0e-13goal
goalsupport above P(Ra)
1.8e-94.7e-12goal
2.2e-103.0e-12goal
refusedsupport above P(Ra)

γ 1, β 5
8.7e-111.5e-13goal
7.7e-115.7e-14goal
1.2e-93.9e-13goal
1.1e-93.8e-12goal
1.7e-132.8e-15goal
8.7e-119.4e-13goal
goalsupport above P(Ra)
1.2e-93.0e-12goal
1.1e-93.9e-12goal
warningsupport above P(Ra)

γ 1.5, β 5
1.4e-101.8e-13goal
1.3e-105.0e-14goal
3.3e-104.6e-13goal
5.6e-103.8e-12goal
2.3e-111.8e-14goal
1.4e-101.3e-12goal
goalsupport above P(Ra)
3.3e-103.3e-12goal
5.6e-102.9e-12goal
refusedsupport above P(Ra)

γ 1.9, β 5
3.6e-112.1e-13goal
2.7e-126.3e-14goal
1.9e-111.1e-12goal
8.7e-113.8e-12goal
8.8e-132.6e-14goal
5.0e-102.3e-12goal
ceilingsupport above P(Ra)
4.2e-113.3e-12ceiling
8.5e-113.2e-12goal
refusedsupport above P(Ra)

γ 1.99, β 5
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedInputError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedValueError†

Plummer (α 2, β 5, γ 0)
1.1e-129.3e-15goal
4.2e-111.4e-14goal
2.7e-91.7e-13goal
excluded
excluded
7.2e-128.3e-15goal
goalsupport above P(Ra)
2.7e-97.6e-12goal
excluded
warningsupport above P(Ra)

### Table 2. The anisotropy radius

Osipkov-Merritt (OM) and generalized Osipkov-Merritt (COM, beta_c -0.25) models of Type I and Type II on three profiles, from R_a 0.3 to 1000.

Ra 0.3
Ra 1
Ra 3
Ra 10
Ra 30
Ra 100
Ra 1000

NFW · OM I
refused
1.5e-104.1e-11ceiling
1.2e-103.8e-11ceiling
9.0e-113.7e-11ceiling
9.0e-113.4e-11ceiling
9.0e-113.3e-11ceiling
9.9e-94.5e-11ceiling

NFW · OM II
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)

NFW · COM I
refused
5.9e-92.9e-9goal
5.8e-92.9e-9goal
5.8e-92.9e-9goal
5.7e-92.9e-9goal
5.7e-92.9e-9goal
5.8e-91.6e-9goal

NFW · COM II
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)

cored (γ 0, β 4) · OM I
refused
3.8e-111.2e-11goal
1.7e-116.8e-13goal
1.7e-111.4e-12goal
1.8e-119.8e-13goal
1.4e-103.6e-12goal
1.7e-101.2e-11goal

cored (γ 0, β 4) · OM II
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)

cored (γ 0, β 4) · COM I
refused
2.0e-93.7e-13goal
2.0e-92.1e-12goal
6.7e-91.4e-10goal
1.5e-82.0e-10goal
2.0e-91.8e-11goal
2.1e-92.8e-11goal

cored (γ 0, β 4) · COM II
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)

steep (γ 1.5, β 4) · OM I
1.7e-106.3e-13goal
1.7e-102.2e-13goal
1.7e-103.6e-13goal
1.7e-109.9e-13goal
3.5e-101.5e-12goal
1.7e-101.5e-12goal
1.7e-103.6e-12goal

steep (γ 1.5, β 4) · OM II
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)
goalsupport above P(Ra)

steep (γ 1.5, β 4) · COM I
2.6e-86.9e-13goal
2.6e-83.9e-13goal
2.6e-89.2e-11goal
2.6e-81.3e-11goal
2.6e-81.8e-11goal
2.6e-89.6e-12goal
2.6e-89.5e-12goal

steep (γ 1.5, β 4) · COM II
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)
warningsupport above P(Ra)

### Table 3. Truncations

Method I and Method II truncations at energy k_t on four parents under six anisotropy models.

kt 0.5
kt 0.2
kt 0.05
kt 0.01
kt 0.002

NFW · isotropic · Method I
1.0e-71.7e-12goal
1.0e-72.1e-12goal
1.0e-75.5e-12goal
1.0e-71.4e-11goal
8.3e-75.6e-11goal

NFW · isotropic · Method II
1.1e-71.2e-10goal†
1.0e-79.4e-11goal†
1.0e-78.8e-11goal†
1.0e-77.2e-11goal†
8.3e-71.2e-10goal†

NFW · βc −0.50 · Method I
2.0e-71.4e-11ceiling
2.0e-77.1e-12ceiling
2.0e-71.1e-11goal
2.0e-72.0e-11goal
2.2e-65.3e-11goal

NFW · βc −0.50 · Method II
2.2e-72.3e-9goal
2.2e-71.2e-9goal
2.2e-76.1e-10goal
2.2e-73.2e-10goal
8.3e-72.0e-10goal

NFW · βc +0.25 · Method I
3.6e-81.2e-12goal
5.5e-81.7e-12goal
5.4e-84.6e-12goal
5.4e-81.2e-11goal
4.2e-76.5e-11goal

NFW · βc +0.25 · Method II
4.1e-81.8e-9goal
4.1e-86.7e-10goal
4.1e-86.9e-10goal
4.1e-86.8e-10goal
8.3e-75.1e-10goal

NFW · OM I Ra 10 · Method I
1.0e-71.7e-12ceiling
1.0e-74.8e-12ceiling
1.0e-73.6e-11ceiling
refusedUnphysicalModelError†
refusedUnphysicalModelError†

NFW · OM I Ra 10 · Method II
1.1e-71.5e-10goal
1.0e-76.7e-11goal
1.0e-75.8e-10goal
1.0e-78.1e-10goal
8.3e-75.1e-10goal

NFW · OM II Ra 10 · Method I
4.6e-75.6e-11goal
refusedDOMAIN†
refusedDOMAIN†
refusedDOMAIN†
refusedDOMAIN†

NFW · OM II Ra 10 · Method II
1.1e-72.1e-10goal
refusedInputError
refusedInputError
refusedInputError
refusedInputError

NFW · COM I (βc −0.25, Ra 10) · Method I
1.5e-74.3e-12goal
1.5e-76.3e-12goal
1.5e-71.2e-10goal
refusedUnphysicalModelError†
refusedUnphysicalModelError†

NFW · COM I (βc −0.25, Ra 10) · Method II
1.7e-71.7e-9goal
1.6e-79.9e-10goal
1.7e-79.3e-10goal
1.7e-79.0e-10goal
8.3e-74.4e-10goal

Hernquist · isotropic · Method I
1.1e-81.5e-12goal
1.0e-71.8e-12goal
1.0e-75.9e-12goal
1.0e-72.0e-11goal
1.3e-66.3e-11goal

Hernquist · isotropic · Method II
2.1e-81.1e-10goal†
1.0e-77.9e-11goal†
1.0e-79.0e-11goal†
1.0e-71.0e-10goal†
1.3e-61.3e-10goal†

Hernquist · βc −0.50 · Method I
2.0e-71.3e-11ceiling
2.0e-76.2e-12ceiling
2.0e-71.1e-11goal
2.0e-72.7e-11goal
2.7e-66.3e-11goal

Hernquist · βc −0.50 · Method II
4.3e-82.3e-9goal
2.2e-72.0e-9goal
2.2e-71.2e-9goal
2.2e-76.4e-10goal
1.3e-62.8e-10goal

Hernquist · βc +0.25 · Method I
7.3e-99.6e-13goal
7.5e-91.5e-12goal
5.4e-85.1e-12goal
5.4e-81.9e-11goal
8.1e-76.7e-11goal

Hernquist · βc +0.25 · Method II
3.2e-82.6e-9goal
4.0e-81.5e-9goal
4.0e-88.3e-10goal
5.2e-81.0e-9goal
1.3e-66.9e-10goal

Hernquist · OM I Ra 10 · Method I
1.1e-81.6e-12ceiling
1.0e-71.8e-12ceiling
1.0e-76.0e-12ceiling
1.0e-72.3e-11ceiling
1.0e-79.5e-12ceiling

Hernquist · OM I Ra 10 · Method II
2.1e-87.2e-11goal
1.0e-71.6e-11goal
1.0e-71.1e-10goal
1.0e-73.1e-10goal
1.3e-62.8e-10goal

Hernquist · OM II Ra 10 · Method I
4.6e-75.8e-11goal
1.8e-75.3e-11goal
refusedDOMAIN†
refusedDOMAIN†
refusedDOMAIN†

Hernquist · OM II Ra 10 · Method II
2.1e-82.7e-10goal
1.0e-72.0e-10goal
refusedInputError
refusedInputError
refusedInputError

Hernquist · COM I (βc −0.25, Ra 10) · Method I
1.6e-73.7e-12goal
1.5e-73.6e-12goal
1.5e-71.3e-11goal
1.5e-72.9e-11goal
1.5e-72.4e-11goal

Hernquist · COM I (βc −0.25, Ra 10) · Method II
3.3e-81.8e-9goal
1.6e-71.5e-9goal
1.7e-71.1e-9goal
1.7e-71.2e-9goal
1.3e-69.8e-10goal

cored (γ 0, β 4) · isotropic · Method I
2.2e-34.3e-12warning
1.8e-44.9e-12warning
5.2e-61.2e-11ceiling
9.2e-83.2e-11goal
1.3e-67.5e-11goal

cored (γ 0, β 4) · isotropic · Method II
2.2e-31.7e-10warning†
1.8e-41.6e-10warning†
5.2e-61.6e-10ceiling†
9.5e-81.8e-10goal†
1.3e-61.5e-10goal†

cored (γ 0, β 4) · βc −0.50 · Method I
1.4e-32.1e-11warning
5.3e-51.3e-11ceiling
6.9e-71.7e-11goal
1.1e-73.4e-11goal
2.7e-66.7e-11goal

cored (γ 0, β 4) · βc −0.50 · Method II
2.1e-37.8e-9refused
1.6e-42.2e-9refused
4.7e-67.0e-10warning
1.3e-73.6e-10ceiling
1.3e-62.1e-10goal

cored (γ 0, β 4) · βc +0.25 · Method I
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY
refusedCENTRAL VALIDITY

cored (γ 0, β 4) · βc +0.25 · Method II
refusedUnphysicalModelError†
refusedUnphysicalModelError†
refusedUnphysicalModelError†
refusedUnphysicalModelError†
refusedUnphysicalModelError†

cored (γ 0, β 4) · OM I Ra 10 · Method I
2.3e-34.2e-12warning
2.2e-45.0e-12warning
2.9e-59.3e-12ceiling
1.1e-58.3e-12ceiling
5.1e-69.7e-12ceiling

cored (γ 0, β 4) · OM I Ra 10 · Method II
2.2e-33.8e-10warning
1.8e-44.2e-10warning
5.2e-63.4e-10ceiling
9.5e-88.6e-10goal
1.3e-64.1e-10goal

cored (γ 0, β 4) · OM II Ra 10 · Method I
2.2e-31.0e-10warning
refusedTYPE II EDGE†
refusedDOMAIN†
refusedDOMAIN†
refusedDOMAIN†

cored (γ 0, β 4) · OM II Ra 10 · Method II
2.2e-36.8e-10warning
refusedInputError
refusedInputError
refusedInputError
refusedInputError

cored (γ 0, β 4) · COM I (βc −0.25, Ra 10) · Method I
9.5e-37.2e-12warning
6.8e-48.8e-12warning
7.7e-54.0e-11ceiling
4.0e-51.6e-10ceiling
2.7e-53.7e-10ceiling

cored (γ 0, β 4) · COM I (βc −0.25, Ra 10) · Method II
2.4e-34.2e-9refused
1.9e-41.5e-9warning
5.5e-61.1e-9warning
1.8e-78.5e-10ceiling
1.3e-67.6e-10goal

steep (γ 1.5, β 4) · isotropic · Method I
refusedAccuracyError†
3.0e-72.3e-12goal
3.0e-77.5e-12goal
3.0e-72.6e-11goal
1.3e-67.8e-11goal

steep (γ 1.5, β 4) · isotropic · Method II
refusedAccuracyError†
3.0e-71.1e-10goal†
3.0e-71.0e-10goal†
3.0e-71.1e-10goal†
1.3e-61.5e-10goal†

steep (γ 1.5, β 4) · βc −0.50 · Method I
5.0e-72.7e-11ceiling
4.9e-78.4e-12ceiling
4.9e-71.4e-11goal
4.9e-73.5e-11goal
2.7e-68.3e-11goal

steep (γ 1.5, β 4) · βc −0.50 · Method II
refusedAccuracyError†
5.2e-72.9e-9goal
5.1e-71.8e-9goal
5.1e-71.0e-9goal
1.3e-63.8e-10goal

steep (γ 1.5, β 4) · βc +0.25 · Method I
refusedAccuracyError†
2.2e-72.0e-12goal
2.0e-76.9e-12goal
2.0e-72.7e-11goal
8.1e-78.9e-11goal

steep (γ 1.5, β 4) · βc +0.25 · Method II
refusedAccuracyError†
1.9e-72.1e-9goal
1.9e-78.1e-10goal
1.9e-71.2e-9goal
1.3e-66.9e-10goal

steep (γ 1.5, β 4) · OM I Ra 10 · Method I
refusedAccuracyError†
3.0e-72.3e-12goal
3.0e-77.7e-12goal
3.0e-72.4e-11goal
3.0e-76.9e-12goal

steep (γ 1.5, β 4) · OM I Ra 10 · Method II
refusedAccuracyError†
3.0e-79.7e-11goal
3.0e-71.2e-10goal
3.0e-72.0e-10goal
1.3e-62.4e-10goal

steep (γ 1.5, β 4) · OM II Ra 10 · Method I
refusedAccuracyError†
1.2e-75.1e-11goal
refusedTYPE II EDGE†
refusedDOMAIN†
refusedDOMAIN†

steep (γ 1.5, β 4) · OM II Ra 10 · Method II
refusedAccuracyError†
3.0e-72.8e-10goal
refusedInputError
refusedInputError
refusedInputError

steep (γ 1.5, β 4) · COM I (βc −0.25, Ra 10) · Method I
refusedAccuracyError†
4.0e-72.8e-12goal
4.0e-78.7e-12goal
4.0e-72.7e-11goal
4.0e-79.8e-12goal

steep (γ 1.5, β 4) · COM I (βc −0.25, Ra 10) · Method II
refusedAccuracyError†
4.1e-72.1e-9goal
4.1e-71.5e-9goal
4.1e-71.2e-9goal
1.3e-65.2e-10goal

### Table 4. Edges

Profiles at the edges of the conditions: the cusp bound, the degenerate and steep constant anisotropies, a small anisotropy radius inside a deep truncation, and the outer slope 2.5 profile, which is refused.

βc γ/2
βc −1/2 + 10⁻⁷
βc −1
βc −1.5
βc +0.75
OM I Ra 0.3 kt 0.01
βc −1/2 kt 0.3

γ 0, β 4
1.7e-111.3e-13goal
1.9e-92.3e-13goal
3.7e-93.0e-13goal
5.6e-91.2e-13goal
excluded
refusedD13†
1.9e-96.3e-14goal

γ 0.5, β 4
1.2e-111.9e-12goal
1.3e-101.9e-13goal
1.9e-103.0e-13goal
1.6e-109.7e-14goal
excluded
refusedD13†
7.8e-115.0e-14goal

γ 1, β 4
1.0e-142.8e-15goal
3.9e-111.8e-13goal
1.9e-102.7e-13ceiling
1.3e-107.7e-14goal
excluded
refusedUnphysicalModelError†
7.2e-113.9e-14goal

γ 1.5, β 4
1.5e-111.9e-13goal
1.7e-101.5e-13goal
3.3e-102.4e-13goal
2.4e-107.5e-14goal
1.5e-111.9e-13goal
refusedUnphysicalModelError†
1.3e-104.3e-14goal

γ 1.9, β 4
3.8e-102.1e-11goal
7.0e-122.0e-13goal
2.3e-112.2e-13goal
3.9e-121.1e-13goal
3.0e-101.6e-13ceiling
refusedUnphysicalModelError†
refusedValueError†

γ 1.99, β 4
refusedInputError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedInputError†
refusedResultError†
refusedResultError†

Plummer (α 2, β 5, γ 0)
1.1e-129.3e-15goal
1.5e-101.2e-13goal
3.1e-102.4e-13goal
7.9e-113.1e-14goal
excluded
refusedCENTRAL SIGN†
4.2e-111.4e-14goal

γ 2, β 4
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError
refusedInputError

γ 1, β 2.5
refusedResultError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedResultError†
refusedResultError†
