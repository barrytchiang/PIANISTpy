# T:5 Particle sampling

Source: https://pianistpy.com/docs/reference/technical-notes/sampling

# T:5 Particle sampling

## T:5.1 The velocity distribution at a given radius factorizes

At radius \(R\), let \(V\) be the speed and \(\varphi\) the angle between the velocity and the radial direction, so that \(L = R\,V\sin\varphi\). For the (C)OM models, the argument of the DF is

\begin{equation}
  Q_\mp = P - \tfrac{1}{2}\,f\,V^2, \qquad f(R, \varphi) \equiv 1 \pm \frac{R^2}{R_\mathrm{a}^2}\sin^2\varphi ,
  \tag{T:5.1}
\end{equation}
with the upper sign for Type I and the lower sign for Type II; \(f = 1\) for constant anisotropy. The probability density of \((V, \varphi)\) at \(R\), Chiang+(2026d), Eq. (B40), is \(\propto V^{2-2\beta_\mathrm{c}}\sin^{1-2\beta_\mathrm{c}}\!\varphi\;F_\mathcal{K}(P - \tfrac12 f V^2)\). In the variable

\begin{equation}
  x \equiv V\sqrt{\frac{f}{2P}} \in [0, 1]
  \tag{T:5.2}
\end{equation}
it becomes a product,

\begin{equation}
  p(x, \varphi\,|\,R) \propto \underbrace{\sin^{1-2\beta_\mathrm{c}}\!\varphi\; f^{-\nu}}_{p(\varphi|R)} \;\times\; \underbrace{x^{2-2\beta_\mathrm{c}}\,F_\mathcal{K}\!\left[P\,(1-x^2)\right]}_{p(x|P)} .
  \tag{T:5.3}
\end{equation}
Three consequences follow.

The angle and the scaled speed are independent, and each can be drawn from a one-dimensional distribution.

The distribution of \(x\) depends on the radius only through \(P(R)\) and is the same for every \(\varphi\); the speed is \(V = x\sqrt{2P/f}\).

The distribution of the angle is known in closed form. For Type I at \(R > R_\mathrm{a}\), it turns over at \(\sin\varphi_{\rm c} = R_\mathrm{a}/R\), the angle at which \(f = 2\); the orbits become radial over an angular scale that shrinks as \(R_\mathrm{a}/R\).

## T:5.2 The table of speeds

### T:5.2.1 The rows

By Eq. (T:5.3), the scaled speed \(x\) has a distribution that depends on \(P\) alone. One table therefore serves the whole model. Its rows are indexed by the potential, uniform in \(\ln[P/(P_0 - P)]\) over the range of the profile with one guard row beyond each end, and no grid in radius or in angle is needed. Each row stores the inverse of the cumulative distribution \(\mathcal{U}(x)\) in logit variables,

\begin{equation}
  s = \ln\frac{x}{x_{\rm e} - x} \quad\text{against}\quad \lambda_{\mathcal{U}} = \ln\frac{\mathcal{U}}{1-\mathcal{U}} ,
  \tag{T:5.4}
\end{equation}
where \(x_{\rm e} \leq 1\) is the edge of the support of \(x\) on that row. Both ends of the distribution are power laws, so \(s\) is close to linear in \(\lambda_{\mathcal{U}}\) toward either end, and one table per row covers the whole range of \(\mathcal{U}\). The table carries its exact slope,

\begin{equation}
  \frac{\mathrm{d} s}{\mathrm{d}\lambda_{\mathcal{U}}} = \frac{x_{\rm e}\,\mathcal{U}\,(1-\mathcal{U})}{p(x|P)\,x\,(x_{\rm e}-x)} ,
  \tag{T:5.5}
\end{equation}
with \(p(x|P)\) normalized on the row, and it is read by cubic Hermite interpolation. Between two rows, \(s\) is interpolated by a cubic Hermite in the row variable, with the exact slope across rows at fixed \(\mathcal{U}\) stored as a third channel. A particle at radius \(R\) and angle \(\varphi\) takes \(x\) at \(P(R)\) and the speed \(V = x\sqrt{2P/f}\) of Eq. (T:5.2); nothing is inverted when a particle is drawn.

A row is built from the mass of the density on the panels of a velocity grid, refined at both ends by Gauss-Legendre panels in the logarithm of \(x\) and of \(x_{\rm e} - x\), with the closed forms of the power laws beyond. The cumulative masses from each end at the edges of the panels are the construction nodes; their cubic Hermite is inverted at the nodes of the table, and each node is polished by a safeguarded Newton step on the true residual. The rows are refined, with every row kept, until the table’s own error across rows falls below the goal or the row count reaches its cap; the report carries that error, and a table beyond the hard limit is refused. Our first design was a two-dimensional grid of the cumulative distribution in \((\ln R, V/V_\mathrm{esc})\), read by bilinear interpolation. The factorization of Eq. (T:5.3) removes the radius from the table, and the logit variables resolve both ends of each row.

### T:5.2.2 The radius and the angle

The radius is drawn from the inverse of the enclosed-mass fraction \(\mathcal{U} = M(R)/M(R_{\rm e})\), tabulated in the same logit variables with \(R\) in place of \(x\) and the outer radius of the sampling, \(R_{\rm e}\), in place of \(x_{\rm e}\). Its slope follows from \(\mathrm{d}\mathcal{U}/\mathrm{d} R = 4\pi R^2\varrho/M(R_{\rm e})\), which is exact everywhere, and the mass fraction is exact at the rows of the profile. The variables are power laws at both ends whether the edge is smooth (the truncation radius) or sharp (a radius given by the user); at a sharp edge the last interval is refined at halved distances from the edge. The angle of an anisotropic model needs no table in radius. By Eq. (T:5.3), \(p(\varphi|R) \propto \sin^{1-2\beta_\mathrm{c}}\!\varphi\,f^{-\nu}\), and the substitution

\begin{equation}
  \sin^2\varphi = \frac{y}{1 \pm \xi\,(1-y)}, \qquad \xi \equiv \frac{R^2}{R_\mathrm{a}^2},
  \tag{T:5.6}
\end{equation}
with the upper sign for Type I and the lower sign for Type II, turns it into a Beta distribution of \(y\) with parameters \((1-\beta_\mathrm{c},\,1/2)\) at every radius. One inverse table of that variate serves the whole model, each half tabulated against the logarithm of the probability at its own end, with the slope \(u/[y\,p(y)]\) in closed form. The angle is drawn in \([0, \pi/2]\), and the sign of the radial velocity is drawn independently; the direction of the tangential component is a Gram-Schmidt projection of an independent isotropic Gaussian draw onto the plane perpendicular to the radius.

## T:5.3 The draw

Every variable is drawn by inversion, and nothing is rejected. The random numbers of particle \(i\) depend only on the seed, the component, and \(i\): its key is folded from the seed and the component and then from \(i\), and each variable takes its own channel of that key. The first \(n\) particles of a larger draw are therefore the same particles, on every machine and backend and for any block size. The potential at a particle is the profile’s own (Section T:1.3), never a linear interpolation on a table. A draw is made in blocks of a stated size, so that its working memory is that of one block, whatever the total count. The particles of a superposed model (Section T:4) are drawn through the enclosed mass of the mixture, with each component’s share of the particles.

A draw bounded to a range of radii takes the uniform variate between the mass fractions of the two radii, so that every particle lands in the range and the particles carry the mass of the range. A speed drawn in a window at a radius takes the variate between the table’s own values at the ends of the window, for a constant anisotropy, where the window does not couple to the angle. A projected pair sampler draws \((R_{\rm 2D}, V_{\rm los})\) on an annulus and in a window of velocity from the cumulative surface density and the line-of-sight tables (Section T:6.5), exactly conditioned.

## T:5.4 What a draw checks

Every particle must lie inside the phase-space support of the DF of its component. The bound is checked on the draw’s own variables, the scaled speed and its distance from the edge of the support, which hold it exactly; the velocity is then checked against the bound from which it was built, \(V^2 + \sigma\,\xi\,V_t^2 \leq 2\Psi\) with \(\xi = (r/R_\mathrm{a})^2\) (zero for a constant anisotropy), to within a stated multiple of its own rounding floor. A particle that fails either check is refused. Every writer reads its file back and compares it with what it wrote, bit for bit after the cast to the type of the file.

     On this page
