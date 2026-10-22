# Code and paper notations

Source: https://pianistpy.com/docs/reference/symbols

The package’s identifiers follow the notation of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). A quantity with a symbol in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758) or in the
Technical notes is called `<family>_<symbol>`. The family prefix gives the kind of quantity, and the rest of the
name is the symbol or the symbol’s subscript in lower case. A truncated quantity (hatted in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758)) takes the
suffix `_trunc`, as in `p_trunc`, `df_trunc`, `rho_trunc` and `mass_trunc`. The untruncated quantity
keeps the name without the suffix. Every docstring gives the
symbol beside the name the first time the docstring describes the quantity.

Family

Prefix

Names

Paper

density and its parameters

`rho_`

`rho`, `rho_aux`, `rho_alpha`, `rho_beta`, `rho_gamma`, `rho_norm`

\(\varrho\), \(\varrho_{\rm aux}\), \((\alpha, \beta, \gamma)\), \(C\)

anisotropy

`beta_`

`beta_c`, `beta_sigma`

\(\beta_{\rm c}\), \(\beta_\sigma\)

radii

`r_`

`r`, `r_a`, `r_t`, `r_2d`, `r_off`, `r_ape`

\(R\), \(R_{\rm a}\), \(R_{\rm t}\), \(R_{\rm 2D}\), \(R_{\rm off}\), \(R_{\rm ape}\)

relative potential

`p_`

`p`, `p_0`

\(P\), \(P_0\)

energy-like variables

`k_`, `q_`

`k`, `k_t`, `q`, `q_min`, `q_0`

\(\mathcal{K}\), \(\mathcal{K}_{\rm t}\), \(\mathcal{Q}\), \(\mathcal{Q}_{\rm min}\), \(\mathcal{Q}_0\)

distribution function

`df_`

`df`, `df_k`

\(F\), \(F_\mathcal{K}\)

mass

`mass_`

`mass`, `mass_t`, `mass_0`

\(\mathcal{M}\), \(M_{\rm t}\), \(M_0\)

order of the inversion

`order_`

`order_m`, `order_n`

\((m, n)\)

velocities and dispersions

`v_`, `sigma_`

`v_r`, `v_t`, `v_los`, `v_esc`, `sigma_r`, `sigma_t`, `sigma_los`

\(V_{\rm r}\), \(V_{\rm t}\), \(V_{\rm los}\), \(V_{\rm esc}\), \(\sigma_{\rm r}\), \(\sigma_{\rm t}\), \(\sigma_{\rm los}\)

probability densities

`pdf_`

`pdf_v`

\(p(V \,|\, R)\)

Where the letter of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758) cannot serve as a name, a fixed word replaces the letter. The normalization \(C\) of the
density is `rho_norm`, since `rho_c` would read as a central density. The concentration \(c\) is
`concentration`. The letter `c` alone is never a name. As a subscript, `c` means central (`beta_c`).

## Derived names

What

Form

Example

logarithm of a quantity

`ln_`, `log10_` prefix

`ln_r`, `log10_df`

derivative

`d<y>_d<x>`, the order as a digit

`drho_dp`, `d2rho_dp2`

count

`n_` prefix

`n_nodes`, `n_rows`

index

`i_` prefix

`i_first`, `i_last`

quadrature nodes and weights

`_nodes`, `_weights` suffix

`k_nodes`, `k_weights`

tabulated values on a grid

`_grid` suffix

`r_grid`, `df_grid`

boolean

`is_`, `has_` prefix

`is_physical`, `has_edge`

The three unit systems are `internal`, `model` and `physical` ([Units](https://pianistpy.com/docs/guide/units)).
