# T:1 The model: density and potential

Source: https://pianistpy.com/docs/reference/technical-notes/model

# T:1 The model: density and potential

## T:1.1 The two maps

The potential of the double power law and its expansions are given in Chiang+(2026d), Appendix A3, Eq. (A11), and Eq. (A13). Two consequences of those expansions are used below.

Near the center, \(P_0 - P \propto R^{2-\gamma}\,[1 + \mathcal{O}(R^{\alpha})]\). Since \(R^{\alpha} \propto (P_0-P)^{\delta}\), any function of \(R\) that is a series in \(R^{\alpha}\) is also a series in \((P_0 - P)^{\delta}\):

\begin{equation}
  R^{\alpha} \propto (P_0 - P)^{\delta}\left[1 + \mathcal{O}\!\left((P_0-P)^{\delta}\right)\right] .
  \tag{T:1.1}
\end{equation}
The coefficients of such a series are exact, and no fit is needed. The depth \(P_0 - P\) is itself a series in \(R^{\alpha}\) with known coefficients, Chiang+(2026d), Eq. (A11). Inverting that series order by order gives \(R = t\,[1 + S(t)]\), with \(t \propto (P_0-P)^{1/(2-\gamma)}\) and \(S\) a series in \(t^{\alpha}\). Substituting this inverse into the density and into the anisotropy factor of \(\varrho_\mathrm{aux}\) yields the coefficient of every power of \(P_0 - P\) as a finite sum of products of binomial coefficients. The package forms these coefficients in closed form by a power-series algebra, keeping every term up to three times the larger of \(\alpha\) and 2 beyond the leading power.

At large radii, the potential of an untruncated profile behaves as

\begin{equation}
  P \propto \begin{cases} R^{-1} & (\beta > 3),\\
   R^{-1}\ln R & (\beta = 3),\\
   R^{-(\beta-2)} & (2 < \beta < 3). \end{cases}
  \tag{T:1.2}
\end{equation}

## T:1.2 Derivatives of the auxiliary density

### T:1.2.1 The operator

The inversion requires \(\mathrm{d}^j\varrho_\mathrm{aux}/\mathrm{d} P^j\) up to \(j = m + 1\), and one order higher wherever the slope of the DF is tabulated as well. In Method I the profile is analytical and these derivatives are exact. The operator

\begin{equation}
  \frac{\mathrm{d}}{\mathrm{d} P} = -\frac{R^2}{M(R)}\,\frac{\mathrm{d}}{\mathrm{d} R},
  \tag{T:1.3}
\end{equation}
with the closed-form enclosed mass \(M(R)\) of Chiang+(2026d), Eq. (A7), is applied symbolically to the closed-form auxiliary density \(\varrho_\mathrm{aux}(R) = \varrho(R)\,h(R)\), where \(h = R^{2\beta_\mathrm{c}}(1 \pm R^2/R_\mathrm{a}^2)^{s}\) is the inverse of the anisotropy factor (\(s = 1\) for OM, \(s = 1 - \beta_\mathrm{c}\) for COM, and \(s = 0\) otherwise). The result is evaluated at \(R(P)\), which we obtain by bisection in \(\ln R\) on the monotonic \(P(R)\) and, near the center, on the subtraction-free depth. The potential \(P(R)\) itself never enters in closed form. A single expression of \(h\) serves every occurrence of the factor in the package: the Method I sources, the Method II sources, the edge law, the density of the Poisson solve, and the parameter derivatives of Section T:7. In Method II the input is a tabulated pair, and the derivatives are integrals of the parent’s DF (Section T:2.6).

Our first design for the tabulated case was a spline of \(\ln{\varrho_\mathrm{aux}}\) in \(P\) with knots placed by the equidistribution principle (de Boor 1978). Its derivatives of second and higher order carried the conditioning of the spline, which grows as the square of the ratio of its largest to its smallest knot spacing, into the inversion. The exact and integral forms replaced it.

### T:1.2.2 The inner series and where it is trusted

Near the center, the density, the enclosed mass, the depth, and the potential are the series of Chiang+(2026d), Eq. (A9), Eq. (A12), and Eq. (A11), summed term by term with no subtraction from the central value. The series is trusted out to the largest radius at which its next term, relative to the depth, stays within one tenth of the goal. This bound follows from the parameters of the profile alone, and no fixed depth is assumed. The Poisson solve of a truncated model starts from the same series, and its inner rows are filled from the series out to that join (Section T:3.4).

## T:1.3 The profile of a truncated model

The self-consistent profile of a truncated model is a table of rows (radius, potential, density, and enclosed mass), and a single object owns its reading: every stage and every reader evaluates the profile through the four functions of that object and holds no interpolant of its own. The potential and the mass are cubic Hermite interpolants in the logarithms, with the exact slopes that the rows state, \(\mathrm{d}\ln P/\mathrm{d}\ln r = -M/(rP)\) and \(\mathrm{d}\ln M/\mathrm{d}\ln r = r^3\varrho/M\) (the Poisson equation and the mass integral). The density is a cubic spline of \(\ln\varrho\) in \(\ln r\), since the rows state no density slope. The radius at a given potential is the guarded inverse of the potential interpolant. A spatially finite profile carries its edge as its last row, where the density is zero and the radius is \(R_\mathrm{t}\). On the last interval the potential takes the Kepler form \(M_\mathrm{t}(1/r - 1/R_\mathrm{t})\) and the density follows the edge law \(\varrho_L\,(P/P_L)^{q}\), with the exponent \(q\) stated by the profile; beyond \(R_\mathrm{t}\) the potential is \(M_\mathrm{t}/r\) and the density is zero. Inside the first row, the interpolants continue as the power laws of the end rows. Nothing outside the domain is clipped: a radius that lies beyond the edge by a rounding error is treated as the edge, and anything further out is refused with a named reason.

     On this page
