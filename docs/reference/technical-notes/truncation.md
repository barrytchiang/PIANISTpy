# T:3 Truncation

Source: https://pianistpy.com/docs/reference/technical-notes/truncation

# T:3 Truncation

The two truncation methods are defined in Chiang+(2026d), Section 3.1.

## T:3.1 Type II models beyond the anisotropy radius

The model is not defined for \(R > R_\mathrm{a}\). In the Type II variable \(Q_+\), the tangential velocity enters with the factor \(1 - R^2/R_\mathrm{a}^2\). For \(R > R_\mathrm{a}\) this factor is negative, the velocity integral diverges, and no DF of this form yields a density. An untruncated profile that extends beyond \(R_\mathrm{a}\) therefore has no Type II DF, whereas a truncated model with \(R_\mathrm{t} \leq R_\mathrm{a}\) is well defined.

Method I nevertheless needs \(\varrho_\mathrm{aux}\) there. The inversion solves the Volterra equation Eq. (T:2.3), whose lower limit lies at \(R = \infty\). The values of \(\varrho_\mathrm{aux}\) on \(P > P(R_\mathrm{a})\) determine \(F_\mathcal{K}\) there only in combination with \(F_\mathcal{K}\) on \(P < P(R_\mathrm{a})\). Method I constructs the truncated DF from the untruncated one by the shift of Chiang+(2026d), Eq. (B27), and the contribution of the stretch \(0 < P < P(R_\mathrm{a})\) to \(F_\mathcal{K}\) depends on \(\mathcal{K}\), so it does not cancel in that difference. A value of \(\varrho_\mathrm{aux}\) must therefore be supplied on that stretch.

The continuation. For OM Type II, the factor \(1 - R^2/R_\mathrm{a}^2\) is a polynomial and is used as it stands. For COM Type II with non-integer \(1 - \beta_\mathrm{c}\), the factor \(w^{1-\beta_\mathrm{c}}\) is not real for \(w < 0\). PIANISTpy adopts the continuation

\begin{equation}
  w^{1-\beta_\mathrm{c}} \;\to\; \mathrm{sign}(w)\,|w|^{1-\beta_\mathrm{c}},
  \tag{T:3.1}
\end{equation}
which is odd in \(w\) and reduces to the OM form at \(\beta_\mathrm{c} = 0\). The continuation is requested through an explicit argument in the symbolic form of the Method I inversion alone; every other reading of the factor at or beyond \(R_\mathrm{a}\) is refused with a named reason. The adopted convention is recorded in the product file.

The support. An untruncated Type II model is defined only inside \(R_\mathrm{a}\), so its DF is returned over \(\mathcal{K} > P(R_\mathrm{a})\) only. The table keeps its fixed grid, the rows at and below \(P(R_\mathrm{a})\) hold no value, and the record states the first row of the support. The values that the inversion forms below \(P(R_\mathrm{a})\) belong to the continuation rather than to the model, whatever their sign; they are used by the inversion above \(P(R_\mathrm{a})\), which depends on them non-locally, and they are not reported. The gates on the sign and on the last minimum of \(F_\mathcal{K}\) are evaluated on the support, and the lowest admissible truncation value is \(\max(\mathcal{K}_0, \mathcal{K}_{\min}, P(R_\mathrm{a}))\), so that every truncated Type II model has \(R_\mathrm{t} < R_\mathrm{a}\). The anisotropy object declares the edge of its domain (\(R_\mathrm{a}\) for Type II, none otherwise), and no rule reads the Type directly. The same bound holds for a Method II child whose anisotropy is of Type II: its \(R_\mathrm{a}\) lies beyond the parent’s edge by the existence condition.

What the convention can and cannot change. Every DF of the form \(\hat F(\hat\mathcal{K})\,L^{-2\beta_\mathrm{c}}\) in the Type II variable has the COM Type II anisotropy profile, and the truncated density is computed from \(\hat F\). The convention therefore affects neither the anisotropy profile nor the self-consistency of a Method I model. What it selects is which truncated density the model has, and under Method I that density is an output. For that density the DF is unique. The effect of the convention on the truncated density exceeds the accuracy goal of Section T:0.5.

Method II inverts the truncated pair, which lies within \(R_\mathrm{a}\), and never evaluates the continuation. A user who needs a Type II model with a prescribed density should therefore use Method II.

## T:3.2 The truncated DF and the Poisson source

From the table that the inversion leaves, the truncation energy, and the settings, the truncated DF \(F(\mathcal{K} + \mathcal{K}_\mathrm{t}) - F(\mathcal{K}_\mathrm{t})\) is formed on the source grid, together with its Taylor form at the edge and the edge integral that joins the form to the table. The auxiliary density

\begin{equation}
  \varrho_\mathrm{aux}(P) = \int_0^{P} \hat F(\mathcal{K})\,\left[2(P - \mathcal{K})\right]^{n}\,\mathrm{d}\mathcal{K}
  \tag{T:3.2}
\end{equation}
is what the Poisson solve takes as its source; the density of a component is \(\varrho_\mathrm{aux}\) multiplied by the factor of its anisotropy in \(r\), which the solve applies. The source grid runs from the floor to the gap below the truncated central potential, with logarithmic tails toward both ends over a stated fraction of the top. The integral at each potential is taken on two intervals: nodes uniform in \(\ln\mathcal{K}\) below a split at a stated fraction of \(P\), and above the split the map \(\mathcal{K} = P - v^{k}\), which removes the end-point behavior of the kernel where \(n\) is not negative. The lookup at a grid node returns the node’s value; no clip by an absolute epsilon enters a quantity that spans ten decades, and the rows below the floor are zero by the DF’s own rule.

### T:3.2.1 The tail of the DF in the density integral

For an untruncated model whose DF has the tail of Eq. (T:2.13), the lower sub-interval is evaluated as

\begin{equation}
  F_\mathcal{K} = A\,\mathcal{K}^{-a} + F_\mathcal{K}^{\rm smooth},
  \tag{T:3.3}
\end{equation}
with the power law integrated in closed form by Eq. (T:2.14) and the remainder integrated numerically. In Method I, the part of the integral below the floor uses the exact stretch of Section T:2.3.2 rather than the power law. A truncated DF has no such tail, and the term is absent.

## T:3.3 The Poisson solve

The truncated Poisson equation, Chiang+(2026d), Eq. (B28), is integrated outward in \(s = \ln r\) as a first-order system in \([\Phi,\; W = r\,\mathrm{d}\Phi/\mathrm{d} r,\; u = P_0 - \Phi]\), under relative and absolute tolerances of \(10^{-11}\), starting from the series of the cusp (Section T:1.2.2). The right-hand side is the density of Chiang+(2026d), Eq. (B29), read from the source table of Section T:3.2. The solve ends at the event where \(\Phi\) reaches the floor, which is the edge of a spatially finite model, or else at the outer radius; the last rows of a truncated model are re-solved on a grid geometric in \(R_\mathrm{t} - r\), and the row at the floor is the crossing of the event. A request whose solve reaches no edge within its radius is refused with a named reason. The solve is a serial integration, and it runs on the CPU on every device (Section T:0.5).

## T:3.4 The fills and the rows of the profile

The rows of the solve are sparse where the potential approaches the floor and where it approaches the center, so both ends are filled to the row density of the profile: at least a stated number of rows per decade of the potential and of the depth, in bins of a stated width, over the profile’s own span. The inner end is the inner series of Section T:1.2.2, lowered by the central deficit where the model carries one, out to the join where the series and the rows of the solve agree; between the join and the threshold of the fill, the solve’s own interpolant is used. The outer end is a log-log interpolation of the solve’s rows in the potential, continued below its lowest rows by their own slope and, below the Taylor switch, by the closed form of the edge. The rows that the profile keeps are those which the solve reached reliably, from the innermost radius requested to the last row at or above the floor; a profile that reaches no edge is read to its last row.

## T:3.5 The gates on the truncation energy

The truncation of Chiang+(2026d), Eq. (B27) requires a DF that is non-negative and non-decreasing above the truncation energy. From the computed untruncated DF, the code takes the highest zero crossing and the position of the last minimum, and it accepts a truncation energy only above the larger of the two and, for a Type II model, strictly above \(P(R_\mathrm{a})\) (Section T:3.1). The bound at \(P(R_\mathrm{a})\) is necessary but not sufficient: only the truncated solve places the edge, and a solve whose edge lies at or beyond \(R_\mathrm{a}\) is refused by the same name afterward. A truncation that no model can carry is refused before any stage runs: a cusp shallower than \(2\beta_\mathrm{c}(0)\) at the center, where the DF is negative there for every truncation (An & Evans 2006); a degenerate central series (Section T:2.5.1); and a central DF that tends to a negative value.

## T:3.6 A target mass or a target radius

Under Method I the untruncated DF does not depend on the truncation, so it is inverted once, and every evaluation of a target costs one truncated solve. The search runs on \(x = \ln\mathcal{K}_\mathrm{t}\) for a target \(v = M_\mathrm{t}\) or \(v = R_\mathrm{t}\), in four stages: deeper truncation from a probe until the model is admissible; a bracket by secant steps in \((\ln\mathcal{K}_\mathrm{t}, \ln v)\) toward the target, by at most a factor of 16 per step, until one admissible model lies on each side; where a step lands outside the admissible set, its boundary, namely the \(\mathcal{K}_\mathrm{t}\) at which \(R_\mathrm{t}\) reaches its limit (the cap, or \(R_\mathrm{a}\)), found by regula falsi on \(1/R_\mathrm{t}\); and finally the Illinois form of regula falsi on \((\ln\mathcal{K}_\mathrm{t}, \ln(v/\text{target}))\) until \(|v/\text{target} - 1|\) is within the goal. A value of \(\mathcal{K}_\mathrm{t}\) is admissible when its solve succeeds, the model is spatially finite, \(R_\mathrm{t}\) is within the cap and, for Type II, inside \(R_\mathrm{a}\), and \(\mathcal{K}_\mathrm{t}\) lies at or above the gates and at or below \(0.9\) of the central potential; on the admissible set both targets decrease with \(\mathcal{K}_\mathrm{t}\). When no admissible model reaches the target, the request is refused with a named reason and the value reached at the limit that binds; the solve takes no fallback.

## T:3.7 The radius of trust of an untruncated product

The solution without truncation must recover \(P \to 0\) as \(R \to \infty\). In practice it differs from the exact potential at large radii by a small constant \(\epsilon\), its integration constant, which we measure against the potential of the source profile beyond \(R = 10^3\). A constant offset in \(P\) changes the potential and the density by

\begin{equation}
  \left|\frac{\delta P}{P}\right| = \frac{|\epsilon|}{P}, \qquad \left|\frac{\delta\varrho}{\varrho}\right| = \left|\frac{\mathrm{d}\ln\varrho}{\mathrm{d}\ln P}\right|\frac{|\epsilon|}{P},
  \tag{T:3.4}
\end{equation}
both of which grow outward. The radius of trust \(R_{\rm trust}(g)\) is the largest radius at which both stay below a relative accuracy \(g\). It is recorded in the product for several values of \(g\) between the goal and the ceiling of Section T:0.5, and untruncated tables are cut at \(R_{\rm trust}(10^{-4})\) by default. Beyond the radius where the potential has fallen to half its central value, the rows of an untruncated model are the closed forms of the source profile on a fixed logarithmic grid, out to the radius where \(P\) reaches the floor, so that the end of the table does not move with the offset. Truncated products are not cut.

## T:3.8 What a model checks

Each check is read on the rows of the profile through the profile object of Section T:1.3, so a model from a run and a model from a file are checked alike. The Poisson residual is Poisson’s equation in integral form on every interval between consecutive rows, including the edge row, \(\Phi(r_a) - \Phi(r_b) = \int M/r^2\,\mathrm{d} r\) and \(M(r_b) - M(r_a) = \int \varrho\,r^2\,\mathrm{d} r\), each relative to the larger of its two rows. The round trip compares the density that the model’s own DF gives at a row, \(\varrho = C(\beta_\mathrm{c})\int_0^{\Phi} F(Q)\,[2(\Phi - Q)]^{1/2-\beta_\mathrm{c}}\,\mathrm{d} Q\) divided by the anisotropy’s factor in \(r\), with \(F\) read through the lookup, against the density of the row. It is evaluated at the rows nearest to each half-decade of radius and at every row from the last of those to the edge, so that the check reads what a dense read would read. The row density is the count of rows per bin that the fills keep; it is silent where every bin is full and refused at any shortfall.

     On this page
