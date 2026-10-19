# T:6 Velocity profiles and projections

Source: https://pianistpy.com/docs/reference/technical-notes/observables

# T:6 Velocity profiles and projections

An observable is a function of a model, and a request computes only what it needs. Two routes exist, and they share nothing but the model: the second moments follow from the density and the potential, whereas the velocity distributions follow from the DF, so each route checks the other. The prepared table of a model holds both routes and reads from them.

## T:6.1 Intrinsic moments

### T:6.1.1 The Jeans integral

For each component, the radial dispersion follows from the Jeans equation in the form

\begin{equation}
  \varrho_\mathrm{aux}\,\sigma_r^2\,(R) = \int_0^{P(R)} \varrho_\mathrm{aux}\,\mathrm{d} P ,
  \tag{T:6.1}
\end{equation}
with the member’s own \(\varrho_\mathrm{aux}\); the part of the integral below the lowest potential of the table is taken over the DF, and it is zero for a spatially finite model. The profile is tabulated in radius on rows from the first row of the profile to its last. Rows are added until every interval reads within the profile tolerance at its midpoint, and the nodes of the panels are doubled until \(\sigma_r\) moves by less than that tolerance. The superposition’s own route evaluates the integral on Gauss-Legendre nodes in \(\ln(1 - \mathcal{K}/P_0)\), which resolves the center, so that the two routes agree to rounding. The tangential dispersion of both tangential components together is \(\sigma_t^2 = 2(1 - \beta_\sigma)\,\sigma_r^2\), with \(\beta_\sigma\) the anisotropy of the model in closed form.

## T:6.2 Projection

The surface density and the line-of-sight dispersion, Chiang et al. (2026d), Eq. (B46), come from a single line-of-sight rule with a single density, on two panels in the depth along the line of sight, at projected radii spaced logarithmically from the innermost row to a fixed fraction of the last row. Rows are added until the table reads itself between rows to within the tolerance of the certificate: the value read at a radius between two rows is compared with a row computed at that radius. Near a spatially finite edge, the table is read in \(x = \ln[R/(1 - R/R_\mathrm{t})]\), which is \(\ln R\) far inside and \(-\ln(1 - R/R_\mathrm{t})\) at the edge, so that the edge factor is resolved. A projected radius inside the innermost row is refused.

## T:6.3 Apertures

The dispersion inside an aperture is the surface-density weighted average of \(\sigma_{\rm los}^2\), Chiang et al. (2026d), Eq. (B50). For an axisymmetric integrand every aperture reduces to a one-dimensional integral, \(\int f(R)\,\mathrm{arc}(R)\,R\,\mathrm{d} R\), where \(\mathrm{arc}(R)\) is the angular measure of the circle of radius \(R\) that lies inside the aperture. A centered circle takes nodes uniform in \(\ln R\) from the first row of the table. An offset circle has a full-circle core and a band of its own, the band under an end-point map in \(R - R_{\rm off}\) where it is narrow. A rectangle takes one panel between consecutive breaks of its arc, each with at least a stated number of Gauss-Legendre nodes and more per decade of its span. Above \(R_\mathrm{t}/\sqrt{2}\), where the factors \(R\) and \(v\) of the measure \(R\,\mathrm{d} R = -v\,\mathrm{d} v\) are equal, a panel is taken in \(v = (R_\mathrm{t}^2 - R^2)^{1/2}\), in which the integrand is smooth up to the edge. An aperture that encloses no tabulated surface density, or whose disk inside the first row carries more than a stated share of its weight, is refused with a named reason.

## T:6.4 The line-of-sight velocity distribution

The distribution \(D(R_{\rm 2D}, V)\) of Chiang et al. (2026d), Eq. (B53) is computed row by row at projected radii. On the plane \(V_{\rm los} = V\) at depth \(u\) along the line of sight, the level sets of \(Q\) are ellipses centered on the maximum of \(Q\), so the integral over the velocity plane is

\begin{equation}
  \iint F\,\mathrm{d} V_{\perp 1}\,\mathrm{d} V_{\perp 2} = \frac{R/R_{\rm 2D}}{\sqrt{Aq}}\int_0^{Q_c} F(Q_c - s)\,\Lambda(s)\,\mathrm{d} s ,
  \tag{T:6.2}
\end{equation}
with \(Q_c\) the maximum, \(Aq\) the product of the coefficients of the ellipse’s axes, and \(\Lambda(s)\) the integral of \((L^2)^{-\beta_\mathrm{c}}\) around the level set. Where the DF carries no \(L\) factor, \(\Lambda = 2\pi\), and the \(s\) integral is the cumulative DF \(G(Q_c)\), read from a table of three branches. With the \(L\) factor, \(\Lambda = \int_0^{\pi}[(c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi]^{-\beta_\mathrm{c}}\,\mathrm{d}\phi\) is taken by a split rule that is dense toward the minimum of its base on each half, \(\phi = \lambda\sinh(\mu x)\) on \([0, \pi/2]\) with \(\mu = \mathrm{arsinh}(\pi/2\lambda)\), at a stated number of Gauss-Legendre nodes per half, as a JAX kernel; at \(\beta_\mathrm{c} = \pm 1/2\) it is a complete elliptic integral in closed form. The \(s\) integral is taken on Gauss-Legendre panels graded toward the energy of the DF’s floor, where the rule of the DF below its table meets the table. The point of zero angular momentum, where the non-analytic part of \(\Lambda\) behaves as \(|s - s_0|^{1-2\beta_\mathrm{c}}\), and the point \(V = 0\), where \(\Lambda \propto s^{-\beta_\mathrm{c}}\), take graded layers whose count follows from \(\beta_\mathrm{c}\) alone, namely the least count that integrates the end-point power to the tolerance, with a stated floor where a negative power is self-similar under the layers. The depth integral is taken on panels of each interval of the support along the line of sight, doubled until the tests of the row pass.

A row is taken on panels in \(V\), with breaks at the velocities where its layout changes: the velocity of the radial orbit, \(V_{\rm rad}\), and the velocity \(V_k\) at which the level set first meets the floor of the DF. Each is kept as a break only where the two panels adjacent to it carry more than a stated share of the row (decided once on the base rows, with each change bisected in \(\ln R_{\rm 2D}\)), with sub-panels graded toward zero velocity and toward each break at a stated ratio, and with the innermost sub-panel kept above the spacing of doubles at the break. Every row is normalized to one and checked against two quantities computed without it: its integral against the surface density, and its second moment against the projected dispersion (Section T:6.2).

A general-purpose integrator applied to the same integral, as in AGAMA (Vasiliev 2019), stops by default at a relative tolerance of \(10^{-3}\) for the moments and of \(10^{-2}\) for the distribution. The rows here are fixed rules whose every end-point behavior is removed by construction, and they are certified at the goal.

## T:6.5 The table of rows and its certificate

The base rows are placed at a stated number per decade of projected radius. By default they run from the first row of the profile out to the last base row beyond which the projected mass falls below the goal times the total projected mass; the rows that the profile would carry beyond that radius, where the DF is zero below the floor or the surface density is about zero, are not computed, and a read beyond the last row, including a read at a single point, is refused with the radius of the last row and the setting that moves it named. The inner end is kept at full depth by default. A user who will read the table only outside a stated projected radius may give that radius, and the rows then start where the projected mass inside the first row falls below the goal times the projected mass inside the stated radius; a read inside the first row is refused in the same words. Rows are added toward each radius where the layout changes, toward the end, and wherever a stretch is short. Rows are then added until the distribution read between two neighboring rows, by a stencil of a stated number of rows, agrees with a row computed there to the tolerance of the settings. An interval that cannot be closed is recorded with its reason, and a request inside it is refused with that reason named. The certificate of the table is its last estimate of its reading between rows, in units of its tolerances. The rows are computed in a pool of processes, one row per process. The report of the table (Section T:8.1) carries the worst norm and second moment over the rows, the rows that could not be judged, the rows whose depth rule did not converge, the certificate, and the floor of the layers at the point of zero angular momentum where \(\beta_\mathrm{c}\) lies between \(1/2\) and \(1\).

## T:6.6 The distribution in an aperture

The distribution inside an aperture is the surface-density weighted average of the rows over the aperture, that is, the form of Chiang et al. (2026d), Eq. (B77) for one component. No row is computed for it. The rows are read at the radii of the aperture’s rule (Section T:6.3), with every break of a row followed as a function of radius, and each radius is scanned once. The result is formed on its own panels in \(V\), with breaks at zero, at the smallest and the largest value of each row break over the aperture, and at the top; a velocity that meets a break curve inside the aperture takes the radii where the curve equals it as breaks of its radial rule. The output rule refines itself by bisecting its widest sub-panels until its norm and its second moment are within one hundredth of the goal of the rows’ own rule. At any velocity, the distribution is the polynomial of its panel through the panel’s own nodes, namely the Legendre series of the antiderivative of the quadrature, so that it integrates exactly as the quadrature does. Its second moment is checked against the dispersion of the aperture from the projected tables, which reads no distribution.

## T:6.7 Gauss-Hermite moments

With \(y = (V - V_*)/\sigma_*\), where \(V_*\) and \(\sigma_*\) are the mean and the dispersion of the distribution itself, the moments of Chiang et al. (2026d), Eq. (B63) are

\begin{equation}
  h_n = 2\sqrt{\pi}\int D(V)\,H_n(y)\,\frac{e^{-y^2/2}}{\sqrt{2\pi}}\,\mathrm{d} V , \qquad H_3 = \frac{2y^3 - 3y}{\sqrt{3}}, \quad H_4 = \frac{4y^4 - 12y^2 + 3}{2\sqrt{6}} ,
  \tag{T:6.3}
\end{equation}
with the Hermite functions of Chiang et al. (2026d), Eq. (B61). They are taken on the velocity nodes and weights of a row or of an aperture distribution, mirrored to both signs and normalized to unit integral. The standardized moments by direct sums are reported beside them; to first order, \(s_3 = 4\sqrt{3}\,h_3\) and \(s_4 = 3 + 8\sqrt{6}\,h_4\). Moments at projected radii are tabulated once over radius and read between rows; moments over apertures are taken on the distribution of each aperture.

## T:6.8 The local speed distribution

With \(t = \cos\eta\), where \(\eta\) is the angle between the velocity and the radius, \(\xi = (r/R_\mathrm{a})^2\), and \(s\) the sign of the anisotropy (\(+1\) for Type I, \(-1\) for Type II, and \(0\) without \(R_\mathrm{a}\)),

\begin{equation}
  p(V\,|\,r) = \frac{4\pi}{\varrho(r)}\,r^{-2b}\,V^{2-2b}\int_0^{1}(1 - t^2)^{-b}\,F(Q)\,\mathrm{d} t , \qquad Q = P(r) - \frac{V^2}{2}(1 + s\xi) + \frac{s\xi V^2}{2}\,t^2 ,
  \tag{T:6.4}
\end{equation}
where \(b\) is \(\beta_\mathrm{c}\) for the constant and COM classes and 0 otherwise. Since \(Q\) is quadratic in \(t\), the edge of the support of the DF is an end of the interval in \(t\) and never an interior point. The interval is split where \(Q\) crosses the first energy of the table, and for \(b > 0\) the piece that ends at \(t = 1\) is integrated in \(w\), with \(1 - t = w^{p}\) and \(p = 1/(1 - b)\), so that \((1 - t)^{-b}\,\mathrm{d} t = p\,\mathrm{d} w\). A Type II model holds orbits of negative energy, and its speeds reach \(V_{\rm esc}/\sqrt{1 - r^2/R_\mathrm{a}^2}\); the share beyond the escape speed is reported. Each radius is computed with three checks: the norm against the density, the second moment against \(\sigma_r^2(3 - 2\beta_\sigma)\), and the change of the distribution when the nodes of its angle rule are doubled, at most three times, until that change falls below the tolerance.

## T:6.9 Superposed models

A superposed model (Section T:4) is projected one component at a time, each on its own tables, and the components are then combined with their weights. The members share one surface density, so the projected second moment is linear in the weights,

\begin{equation}
  \sigma_{\rm los}^2(R) = \sum_i c_i\,\sigma_{{\rm los},i}^2(R) ,
  \tag{T:6.5}
\end{equation}
and the same holds for the intrinsic second moments. With the projected moments of each member tabulated once, the composite profile for a new set of weights costs one matrix-vector product, which is what the likelihood of Section T:7.4 evaluates. A distribution at a projected radius mixes by the surface densities, \(D = \sum_k w_k\Sigma_k D_k/\sum_k w_k\Sigma_k\); a distribution in an aperture mixes by the projected masses inside it; and the local speed distribution mixes by the densities at that radius.

     On this page
