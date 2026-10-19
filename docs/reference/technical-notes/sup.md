# T:4 Superposition

Source: https://pianistpy.com/docs/reference/technical-notes/sup

# T:4 Superposition

Distribution functions inverted from one and the same truncated density, that is, Method II children of one parent, combined with positive weights that sum to one, form a single equilibrium model in that density. Members on different truncated profiles are refused, and every component keeps its own energy grid, certified on its own nodes. The anisotropy profile of the superposition is given in Chiang+(2026d), Eq. (B74). In terms of the radial dispersions \(\sigma_{r,i}^2\) and the anisotropy profiles \(\beta_i\) of the members, with weights \(c_i > 0\) and \(\sum_i c_i = 1\), it reads

\begin{equation}
  \beta_{\rm sup}(R) = \frac{\sum_i c_i\,\beta_i(R)\,\sigma_{r,i}^2(R)}{\sum_i c_i\,\sigma_{r,i}^2(R)} .
  \tag{T:4.1}
\end{equation}
The range of anisotropy profiles that a superposition can reach is set by three limits.

Outer value. At the truncation radius,

\begin{equation}
  \beta_{\rm sup}(R_\mathrm{t}) = \sum_i c_i\,\beta_i(R_\mathrm{t}),
  \tag{T:4.2}
\end{equation}
which is a single linear constraint on the weights. In practice this is the limit that binds.

Central value. \(\beta_{\rm sup}(0)\) equals the largest positive central anisotropy among the members in use; if all central anisotropies are negative, it is the weighted harmonic mean of their central values.

Transition. The (C)OM profiles change over about one decade in radius. A target profile that changes over a wider range is approached by mixing members with different anisotropy radii.

The composite dispersions are computed from Eq. (T:4.1) and the Jeans integral of Section T:6.1.1 without any sampling. Downstream of this section, a superposition is treated as one model: its particles are drawn through the enclosed mass of the mixture with each component’s share (Section T:5.3), its projected observables are formed one component at a time and mixed by the weights (Section T:6.9), and its likelihood is linear in the weights (Section T:7.4).
