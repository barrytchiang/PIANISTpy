# Names and paper symbols

Source: https://pianistpy.com/docs/reference/symbols

# Names and paper symbols

The package’s names follow the notation of Chiang+(2026d): a quantity that has a symbol there or in the technical notes
is named <family>_<symbol>, the family saying what kind of quantity it is and the rest being that symbol
or subscript in lower case. A truncated quantity (hatted in Chiang+(2026d)) takes the suffix _trunc, the
untruncated one the bare name. Every docstring gives that symbol beside the name the first time it
describes it.

Family

Prefix

Names

Paper

density and its parameters

rho_

rho, rho_aux, rho_alpha, rho_beta, rho_gamma, rho_norm

\(\varrho\), \(\varrho_{\rm aux}\), \((\alpha, \beta, \gamma)\), \(C\)

anisotropy

beta_

beta_c, beta_sigma, beta_sup

\(\beta_{\rm c}\), \(\beta_\sigma\), \(\beta_{\rm sup}\)

radii

r_

r, r_a, r_t, r_2d, r_off, r_ape

\(R\), \(R_{\rm a}\), \(R_{\rm t}\), \(R_{\rm 2D}\), \(R_{\rm off}\), \(R_{\rm ape}\)

relative potential

p_

p, p_0

\(P\), \(P_0\)

energy-like variables

k_, q_

k, k_t, q, q_min, q_0

\(\mathcal{K}\), \(\mathcal{K}_{\rm t}\), \(\mathcal{Q}\), \(\mathcal{Q}_{\rm min}\), \(\mathcal{Q}_0\)

distribution function

df_

df, df_k

\(F\), \(F_\mathcal{K}\)

mass

mass_

mass, mass_t, mass_vir

\(\mathcal{M}\), \(M_{\rm t}\), \(M_{\rm vir}\)

order of the inversion

order_

order_m, order_n

\((m, n)\)

velocities and dispersions

v_, sigma_

v_r, v_t, v_los, v_esc, sigma_r, sigma_t, sigma_los

\(V_{\rm r}\), \(V_{\rm t}\), \(V_{\rm los}\), \(V_{\rm esc}\), \(\sigma_{\rm r}\), \(\sigma_{\rm t}\), \(\sigma_{\rm los}\)

probability densities

pdf_

pdf_x, pdf_phi

\(p_x\), \(p_\varphi\)

Where the letter of Chiang+(2026d) cannot be used, a word is fixed: the normalization \(C\) of the density is
rho_norm (rho_c would read as a central density), and the concentration \(c\) is concentration.
The letter c alone is never a name; as a subscript it means central (beta_c). The quantities of the
truncated model carry the suffix: p_trunc, df_trunc, rho_trunc, mass_trunc.

## Derived names

What

Form

Example

logarithm of a quantity

ln_, log10_ prefix

ln_r, log10_df

derivative

d<y>_d<x>, the order as a digit

drho_dp, d2rho_dp2

count

n_ prefix

n_nodes, n_rows

index

i_ prefix

i_split

quadrature nodes and weights

_nodes, _weights suffix

k_nodes, k_weights

tabulated values on a grid

_grid suffix

r_grid, df_grid

boolean

is_, has_ prefix

is_truncated, has_tail

The three unit systems are named internal (the dimensionless normalization of Widrow 2000, in which the
scale radius and the central potential of the untruncated profile are 1), model (\(G = M_0 = r_0 = 1\))
and physical (through Astropy); see Units.

     On this page
