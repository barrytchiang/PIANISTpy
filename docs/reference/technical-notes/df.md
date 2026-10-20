# T:2 The distribution function

Source: https://pianistpy.com/docs/reference/technical-notes/df

# T:2 The distribution function

## T:2.1 The inversion

Chiang+(2026d) gives the inversion in two forms: with the derivative outside the integral, Eq. (B22), and with the derivative inside the integral together with a surface term at \(P = 0\), Eq. (B9). The second form follows from the first by one integration by parts and is simpler to evaluate. The integration by parts requires that

\begin{equation}
  \frac{\mathrm{d}^m\varrho_\mathrm{aux}}{\mathrm{d} P^m}\ \text{continuous on}\ [0, \mathcal{K}] \qquad\text{and}\qquad \int_0^{\mathcal{K}}\left|\frac{\mathrm{d}^{m+1}\varrho_\mathrm{aux}}{\mathrm{d} P^{m+1}}\right|(\mathcal{K}-P)^{-n}\,\mathrm{d} P < \infty .
  \tag{T:2.1}
\end{equation}
Neither condition is guaranteed. The first can fail at the outer end, \(P = 0\) (Section T:2.2.6), and at the anisotropy radius of a Type II model (Section T:2.2.8).

For half-integer \(\beta_\mathrm{c}\) we have \(n = 0\), and the general form reduces to a derivative with no integral,

\begin{equation}
  F_\mathcal{K}(\mathcal{K}) = \frac{2^{\beta_\mathrm{c}}}{(2\pi)^{3/2}\,\Gamma(1-\beta_\mathrm{c})}\left.\frac{\mathrm{d}^m\varrho_\mathrm{aux}}{\mathrm{d} P^m}\right|_{P=\mathcal{K}} .
  \tag{T:2.2}
\end{equation}
The forward relation is a Volterra equation whose lower limit lies at \(P = 0\), that is, at \(R = \infty\):

\begin{equation}
  \varrho_\mathrm{aux}(P) = \frac{2\pi^{3/2}\,\Gamma(1-\beta_\mathrm{c})}{\Gamma(3/2-\beta_\mathrm{c})} \int_0^{P} F_\mathcal{K}(\mathcal{K})\,\left[2(P - \mathcal{K})\right]^{1/2-\beta_\mathrm{c}}\,\mathrm{d}\mathcal{K} .
  \tag{T:2.3}
\end{equation}

## T:2.2 Asymptotic behavior of the DF

### T:2.2.1 The center: the singular part of one power-law term

Consider a single term \(\varrho_\mathrm{aux} \supset A_s\,(P_0 - P)^{s}\) of the expansion of \(\varrho_\mathrm{aux}\) about the center. Its \((m+1)\)-th derivative is \((-1)^{m+1} A_s\,[\Gamma(s+1)/\Gamma(s-m)]\,(P_0-P)^{s-m-1}\). With \(t = \mathcal{K} - P\), the integral of Chiang+(2026d), Eq. (B9) becomes \(\int_0^{\mathcal{K}} t^{-n} (u+t)^{s-m-1}\,\mathrm{d} t\), whose behavior as \(u \to 0\) follows from the Mellin-type integral

\begin{equation}
  \int_0^{\infty} t^{a-1}(u+t)^{-b}\,\mathrm{d} t = u^{a-b}\,\frac{\Gamma(a)\,\Gamma(b-a)}{\Gamma(b)},
  \tag{T:2.4}
\end{equation}
with \(a = 1-n\) and \(b = m+1-s\). The finite upper limit adds only a part that is analytic at \(u = 0\). Using \(\Gamma(s-m)\,\Gamma(m+1-s) = \pi/\sin[\pi(s-m)]\) and \(\Gamma(-s)\,\Gamma(1+s) = -\pi/\sin(\pi s)\), the term contributes

\begin{equation}
  F_\mathcal{K}^{\rm sing}(\mathcal{K}) = \frac{2^{\beta_\mathrm{c}}}{(2\pi)^{3/2}\,\Gamma(1-\beta_\mathrm{c})}\; A_s\,\frac{\Gamma(\nu - s)}{\Gamma(-s)}\; u^{\,s-\nu},
  \tag{T:2.5}
\end{equation}
plus a part that is analytic in \(u\). Three properties of Eq. (T:2.5) decide what follows.

For \(s\) a non-negative integer, \(1/\Gamma(-s) = 0\), and the term has no singular part.

For \(s = \nu + k\) with \(k\) a non-negative integer and \(\nu\) not an integer, \(\Gamma(\nu - s)\) has a pole, and the power is replaced by a logarithm,

\begin{equation}
  F_\mathcal{K}^{\rm sing}(\mathcal{K}) = \frac{2^{\beta_\mathrm{c}}}{(2\pi)^{3/2}\,\Gamma(1-\beta_\mathrm{c})}\,\frac{(-1)^{k}\,A_s}{k!\;\Gamma(-s)}\;u^{k}\,(-\ln u) ,
  \tag{T:2.6}
\end{equation}
which for \(k = 0\) reduces to \(-\,2^{\beta_\mathrm{c}}\,A_\nu\,\ln u\,/\,[(2\pi)^{3/2}\,\Gamma(1-\beta_\mathrm{c})\,\Gamma(-\nu)]\).

Otherwise the term is a power law \(u^{\,s-\nu}\), whose sign is that of \(A_s\,\Gamma(\nu-s)/\Gamma(-s)\).

Since the inversion is linear, every power in the expansion of \(\varrho_\mathrm{aux}\) contributes its own term of the form Eq. (T:2.5) or Eq. (T:2.6), and the singular part of \(F_\mathcal{K}\) at the center is the sum of these terms in order of increasing \(s\). Only the non-negative integer powers drop out; a negative integer \(s\) is singular like any other. The coefficients \(A_s\) follow from the expansion of Section T:1.1. A ratio of two Gamma functions is formed as the exponential of a difference of log-Gamma values together with its sign, never as a direct quotient, so that a steep center whose Gamma values overflow still has a finite amplitude.

### T:2.2.2 The regular branch, \(\gamma > 2\beta_\mathrm{c}\)

The leading term of \(\varrho_\mathrm{aux}\) is \(A\,(P_0-P)^{s_0}\) with \(s_0 = (2\beta_\mathrm{c} - \gamma)/(2-\gamma) < 0\). Eq. (T:2.5) reproduces the exponent \(\kappa\) of Chiang+(2026d), Eq. (B36) and supplies its amplitude,

\begin{equation}
  F_\mathcal{K} \simeq \frac{2^{\beta_\mathrm{c}}}{(2\pi)^{3/2}\,\Gamma(1-\beta_\mathrm{c})}\,A\,\frac{\Gamma(\nu - s_0)}{\Gamma(-s_0)}\,u^{\kappa},
  \tag{T:2.7}
\end{equation}
which is positive. For the isotropic Hernquist profile in units \(G = M = a = 1\), \(\varrho_\mathrm{aux} \simeq (2\pi u)^{-1}\) and \(F \simeq 3\,u^{-5/2}/(16\sqrt{2}\,\pi^2)\), which is the leading term of its known closed form.

As \(\gamma \to 2\beta_\mathrm{c}\), the amplitude vanishes linearly, \(\Gamma(\nu - s_0)/\Gamma(-s_0) = -s_0\,\Gamma(\nu) + \mathcal{O}(s_0^2)\), while the exponent tends to \(-\nu\). The regular and the critical branch therefore join in amplitude but not in exponent.

### T:2.2.3 The critical branch, \(\gamma = 2\beta_\mathrm{c}\)

By Eq. (T:1.1), \(\varrho_\mathrm{aux} = C_1 + C_2\,(P_0-P)^{\delta} + \dots\) is a series in \((P_0-P)^{\delta}\). The constant term has \(s = 0\) and no singular part. The next term has \(s = \delta\) and the exponent \(\kappa' = \delta - \nu\) of Chiang+(2026d), Eq. (B37). Table T:2.1 lists the cases; its last four rows assume that \(\delta\) is not an integer.

Table T:2.1. The central DF in the critical case \(\gamma = 2\beta_\mathrm{c}\), for constant anisotropy and OM models.

Condition

Central behavior

Leading form

\(\delta\) a positive integer

finite

analytic in \(u\)

\(\kappa' < 0\)

power-law divergence

Eq. (T:2.5) with \(s = \delta\), \(A_s = C_2\)

\(\kappa' = 0\)

logarithmic divergence

Eq. (T:2.6) with \(A_\nu = C_2\)

\(\kappa' > 0\), \(n \neq 0\)

finite, non-zero

\(F_\mathcal{K}(P_0) + \mathcal{O}(u^{\kappa'})\)

\(\kappa' > 0\), \(n = 0\)

vanishing

\(\propto u^{\kappa'}\)

Sign. With \(C_2 < 0\), the divergence is toward negative values for \(1 < \delta < \nu\), and also for \(\kappa' = 0\) with \(\beta_\mathrm{c} < 1/2\). Such models are unphysical even though they satisfy Chiang+(2026d), Eq. (12).

The finite case at \(\kappa' = 0\). This case requires \(\delta = \nu\) to be an integer. Within \(0 \leq \gamma < 2\) it occurs only for \(\alpha = 1\), \(\gamma = 1\), \(\beta_\mathrm{c} = 1/2\), which includes the NFW and Hernquist profiles.

The local case. For \(n = 0\) the inversion is the derivative of Eq. (T:2.2), and the non-local constant \(F_\mathcal{K}(P_0)\) is absent.

### T:2.2.4 COM models

For COM models, the factor \((1 \pm R^2/R_\mathrm{a}^2)^{1-\beta_\mathrm{c}}\) of \(\varrho_\mathrm{aux}\) contributes a series in \(R^2 \propto (P_0-P)^{2/(2-\gamma)}\). The expansion of \(\varrho_\mathrm{aux}\) then contains the powers

\begin{equation}
  s_{kj} = k\,\delta + \frac{2j}{2-\gamma}, \qquad k, j \geq 0,
  \tag{T:2.8}
\end{equation}
and the leading singular term is the smallest \(s_{kj}\) that is not an integer. The result of Section T:2.2.3 is modified when \(2/(2-\gamma)\) is not an integer and, in addition, either \(2/(2-\gamma) < \delta\) or \(\delta\) is an integer. Since \(2/(2-\gamma)\) is an integer at \(\gamma = 0\) and at \(\gamma = 1\), OM models (which are critical only at \(\gamma = 0\)) and all \(\gamma = 1\) profiles are unaffected.

Where the modification applies, the exponent is \(2/(2-\gamma) - \nu\), which is negative for \(0 < \gamma < (5-\sqrt{17})/2 \simeq 0.438\). A center that is finite under constant anisotropy is then divergent under COM Type I. For example, \((\alpha, \gamma, \beta_\mathrm{c}) = (1.8, 0.2, 0.1)\) has \(\delta = 1\) and a finite center under constant anisotropy, but \(F_\mathcal{K} \propto u^{-13/45}\) under COM Type I.

### T:2.2.5 The NFW profile at \(\beta_\mathrm{c} = 1/2\)

With \(\varrho = 1/[R(1+R)^2]\) and \(P = \ln(1+R)/R\), Eq. (T:2.2) gives

\begin{equation}
  F_\mathcal{K} = \frac{1}{2\pi^2}\frac{\mathrm{d}(R\varrho)}{\mathrm{d} P} = \frac{1}{\pi^2}\,\frac{R^2}{(1+R)^3\left[\ln(1+R) - R/(1+R)\right]},
  \tag{T:2.9}
\end{equation}
evaluated at \(R(\mathcal{K})\). Its expansion about the center contains integer powers only,

\begin{equation}
  \pi^2 F_\mathcal{K} = 2 - \frac{20}{3}u + \frac{28}{3}u^2 - \frac{1024}{135}u^3 + \mathcal{O}(u^4),
  \tag{T:2.10}
\end{equation}
so the central DF is finite, \(F_\mathcal{K}(P_0) = 2/\pi^2\).

### T:2.2.6 The outer end, \(P = 0\)

Let \(\varrho_\mathrm{aux} \propto P^{\,y}\) as \(P \to 0\). Then \(\mathrm{d}^m\varrho_\mathrm{aux}/\mathrm{d} P^m \propto P^{\,y-m}\), and the surface term of Chiang+(2026d), Eq. (B9)

\begin{equation}
  \text{vanishes for}\ y > m, \qquad \text{is finite for}\ y = m, \qquad \text{diverges for}\ y < m .
  \tag{T:2.11}
\end{equation}
For an untruncated profile with \(\beta > 3\), Eq. (T:1.2) gives

\begin{equation}
  y = \beta - 2\beta_\mathrm{c} \ \ \text{(constant anisotropy)}, \qquad y = \beta - 2 \ \ \text{[(C)OM Type~I]}.
  \tag{T:2.12}
\end{equation}
Table T:2.2 lists the consequences. The bound \(\beta > m+2\) is sharp.

Table T:2.2. The surface term at \(P = 0\) for untruncated profiles with \(\beta > 3\).

Model

\(m\)

Surface term vanishes

constant anisotropy

\(\lfloor 3/2 - \beta_\mathrm{c}\rfloor\)

always

(C)OM Type I, \(\beta_\mathrm{c} > 1/2\)

0

always

(C)OM Type I, \(-1/2 < \beta_\mathrm{c} \leq 1/2\)

1

always

(C)OM Type I, \(\beta_\mathrm{c} \leq -1/2\)

\(\geq 2\)

for \(\beta > m + 2\)

At \(\beta = 3\) the constant-anisotropy surface term still vanishes. For (C)OM Type I, \(\varrho_\mathrm{aux} \propto P/\ln(1/P)\), so the term vanishes only as \(1/\ln(1/P)\) for \(m = 1\) and diverges for \(m \geq 2\). For \(2 < \beta < 3\) and \(m = 1\) the term is finite and non-zero.

### T:2.2.7 The tail of the DF and its density

For an untruncated (C)OM Type I model with \(\beta > 3\), Eq. (T:2.12) and Chiang+(2026d), Eq. (B25) give a power-law tail,

\begin{equation}
  F_\mathcal{K}(\mathcal{K}) \simeq A\,\mathcal{K}^{-a} \quad (\mathcal{K} \to 0), \qquad a = \nu - (\beta - 2) = \tfrac{7}{2} - \beta_\mathrm{c} - \beta ,
  \tag{T:2.13}
\end{equation}
which diverges for \(a > 0\); a profile with an outer slope of exactly 3 carries a logarithmic factor beside the power. By Eq. (T:2.3), the part of the tail below an energy \(q\) contributes to the auxiliary density at \(P > q\) the amount

\begin{equation}
  \varrho_\mathrm{aux}^{\rm tail}(P) = \frac{2\pi^{3/2}\,\Gamma(1-\beta_\mathrm{c})}{\Gamma(3/2-\beta_\mathrm{c})}\; 2^{\,\nu-1} A\, P^{\,\nu-a}\; B\!\left(\frac{q}{P};\, 1-a,\, \nu\right),
  \tag{T:2.14}
\end{equation}
where \(B(x; p, r) = \int_0^x t^{p-1}(1-t)^{r-1}\,\mathrm{d} t\) is the incomplete Beta function. The integral converges for \(a < 1\). For \(a \geq 1\) the density integral diverges at \(\mathcal{K} = 0\), and the untruncated model does not exist; a truncated model does exist, since the truncated DF of Chiang+(2026d), Eq. (B27) vanishes at its lower end.

### T:2.2.8 The anisotropy radius of a Type II model

For Type II models, \(\varrho_\mathrm{aux} \propto w^{1-\beta_\mathrm{c}}\) as \(w \equiv 1 - R^2/R_\mathrm{a}^2 \to 0\), and \(w\) is linear in \(P - P(R_\mathrm{a})\) there. Hence

\begin{equation}
  \frac{\mathrm{d}^m\varrho_\mathrm{aux}}{\mathrm{d} P^m} \propto |P - P(R_\mathrm{a})|^{\,1-\beta_\mathrm{c}-m} = |P - P(R_\mathrm{a})|^{\,n - 1/2},
  \tag{T:2.15}
\end{equation}
which diverges at \(P(R_\mathrm{a})\) for \(n < 1/2\). The divergence is approached from inside \(R_\mathrm{a}\) and does not depend on how \(\varrho_\mathrm{aux}\) is continued beyond it (Section T:3.1). For OM Type II, \(n = 1/2\) and the factor is a polynomial, so the conditions of Eq. (T:2.1) hold. A truncated model with \(R_\mathrm{t} \leq {R_\mathrm{a}}\) does not contain the point at all.

## T:2.3 The form of every row

### T:2.3.1 The order pair and the choice of form

The form of the inversion is fixed from the model parameters alone, before any integral is evaluated.

Order pair. From \(\beta_\mathrm{c}\) the code forms the pair \((m, n)\) of Eq. (T:0.1). For \(n > 1/2\) it replaces the pair by

\begin{equation}
  (j, n') = (m+1,\; n-1),
  \tag{T:2.16}
\end{equation}
so that the kernel exponent always satisfies \(|n'| \leq 1/2\); otherwise \((j, n') = (m, n)\). With the higher order, the surface term can diverge as \(P \to 0\) (Section T:2.2.6) and cancel against the integral, so a replaced pair is evaluated in the general form, Chiang+(2026d), Eq. (B22), on every row.

Half-integer \(\beta_\mathrm{c}\). For \(n = 0\) the closed form of Eq. (T:2.2) is used, and no quadrature is performed.

Interior point. For an untruncated COM Type II model, the point \(P(R_\mathrm{a})\) and the exponent of Eq. (T:2.15) are known in closed form. Rows with \(\mathcal{K} < P(R_\mathrm{a})\) use the form with the surface term, Chiang+(2026d), Eq. (B9); rows above it use the general form, with the integral split at \(P(R_\mathrm{a})\) and a Gauss-Jacobi rule on either side that carries the exponent as its weight. For odd integer \(1-\beta_\mathrm{c}\) the factor is a polynomial, and there is no such point.

An unbounded outer end. Where \(\varrho_\mathrm{aux}^{(j)}\) is unbounded at \(P = 0\), the rows above a stated energy use the general form.

All other rows use the form with the surface term.

Applying the same operator one derivative higher gives \(\mathrm{d} F_\mathcal{K}/\mathrm{d}\mathcal{K}\) on the same rows, so the table carries an exact slope column.

### T:2.3.2 The floor of the table

The table of \(\varrho_\mathrm{aux}\) and its derivatives starts at a floor \(q_0 > 0\), not at \(P = 0\). Splitting the integral of the general form at \(q_0\) and integrating its upper part by parts gives, with no assumption about the behavior at \(P = 0\),

\begin{equation}
  \frac{\mathrm{d}}{\mathrm{d}\mathcal{K}}\int_0^{\mathcal{K}}\frac{\varrho_\mathrm{aux}^{(j)}(P)}{(\mathcal{K}-P)^{n}}\,\mathrm{d} P = \int_{q_0}^{\mathcal{K}}\frac{\varrho_\mathrm{aux}^{(j+1)}(P)}{(\mathcal{K}-P)^{n}}\,\mathrm{d} P + \frac{\varrho_\mathrm{aux}^{(j)}(q_0)}{(\mathcal{K}-q_0)^{n}} - n\int_0^{q_0}\frac{\varrho_\mathrm{aux}^{(j)}(P)}{(\mathcal{K}-P)^{n+1}}\,\mathrm{d} P ,
  \tag{T:2.17}
\end{equation}
where \(\varrho_\mathrm{aux}^{(j)} \equiv {\mathrm{d}}^j\varrho_\mathrm{aux}/\mathrm{d} P^j\). The first two terms need only the table. The third term is the contribution of the stretch below the floor. In Method I it is evaluated from the exact derivatives of Section T:1.2.1 on Gauss-Legendre nodes in \(\ln P\), down to a lower limit far below \(q_0\), so that the floor of the table is not the floor of the model; in Method II it is the closed form of Section T:2.6. The default floor is \(q_0 = 10^{-10}\) in units of the central potential, and untruncated DF tables are written from \(2\,q_0\) upward.

### T:2.3.3 The general form on a split

Rows that use the general form (Section T:2.3.1) split the integral at an interior point \(c < \mathcal{K}\). Below \(c\) the derivative is taken under the integral sign; above \(c\) the integral is integrated by parts:

\begin{equation}
  \frac{\mathrm{d}}{\mathrm{d}\mathcal{K}}\int_{q_0}^{\mathcal{K}}\frac{\varrho_\mathrm{aux}^{(j)}(P)}{(\mathcal{K}-P)^{n}}\,\mathrm{d} P = -n\int_{q_0}^{c}\frac{\varrho_\mathrm{aux}^{(j)}(P)}{(\mathcal{K}-P)^{n+1}}\,\mathrm{d} P + \frac{\varrho_\mathrm{aux}^{(j)}(c)}{(\mathcal{K}-c)^{n}} + \int_{c}^{\mathcal{K}}\frac{\varrho_\mathrm{aux}^{(j+1)}(P)}{(\mathcal{K}-P)^{n}}\,\mathrm{d} P .
  \tag{T:2.18}
\end{equation}
The right-hand side requires \(\varrho_\mathrm{aux}^{(j)}\) only where it is bounded, and \(\varrho_\mathrm{aux}^{(j+1)}\) only above \(c\). For a replaced pair, Eq. (T:2.16), this avoids \(\varrho_\mathrm{aux}^{(m+1)}\) at the floor, which the form with the surface term would require and which can diverge as \(P \to 0\). For rows above the interior point of a COM Type II model, \(c\) lies above \(P(R_\mathrm{a})\), and the lower integral is evaluated on the two sides of that point with a rule that carries the exponent of Eq. (T:2.15) as its weight.

### T:2.3.4 Quadrature

Each row is integrated on three sub-intervals: from the floor to one tenth of the split point, on nodes uniform in \(\ln P\); from there to the split point \(c = 0.9\,\mathcal{K}\), on Gauss-Legendre nodes; and from the split point to \(\mathcal{K}\), where the kernel is singular. On the last sub-interval the integration variable is \(v\), with

\begin{equation}
  P = P_0 - (P_0 - \mathcal{K})\,e^{y}, \qquad y = v^{k}, \qquad k = \frac{1}{1-n} .
  \tag{T:2.19}
\end{equation}
Near the upper limit, \(\mathcal{K} - P \simeq (P_0 - \mathcal{K})\,y\), so that

\begin{equation}
  (\mathcal{K} - P)^{-n}\,\mathrm{d} P \;\propto\; y^{-n}\,\mathrm{d} y = k\,v^{\,k(1-n)-1}\,\mathrm{d} v = k\,\mathrm{d} v ,
  \tag{T:2.20}
\end{equation}
and the kernel singularity is removed for every exponent that the order pair allows. Away from the upper limit, the nodes are uniform in \(\ln(P_0 - P)\), which resolves an integrand that steepens as a power of \(P_0 - P\) toward the center.

## T:2.4 The energy grid

The DF is tabulated on a grid of energies in five zones, from the floor to the gap below the central potential: zone A, logarithmic in \(\mathcal{K}\) from the floor; zone AB, a buffer; zone B, linear; zone BC, a buffer; and zone C, logarithmic in \(P_0 - \mathcal{K}\) up to the gap. A buffer carries the step geometrically from the spacing on one side to the spacing on the other. Its number of steps is the one whose total comes closest to the width of the buffer, and the steps are rescaled to sum to that width exactly, so that no jump in spacing enters the rows. The linear zone and the two logarithmic zones take their row counts from the settings, and the logarithmic zones extend over a stated fraction of the central potential. An OM Type I model places zones A and AB around the potential of its anisotropy radius; an OM Type II or COM model places them around the feature that a pilot inversion on a coarse grid finds (the minimum or the zero crossing of its DF), refined by a parabola through the neighboring nodes. The upper end is never moved. Below the floor, the grid is extended at a stated number of nodes per decade down to the table’s own floor, where the rule below the table takes over (Section T:2.5.4). A grid whose energies are not strictly increasing in double precision is refused.

## T:2.5 Beyond the table

### T:2.5.1 Where the rows end

The rows end at the last energy of the grid, a gap of \(10^{-9}\) below \(P_0\), unless the exact derivatives required by the inversion cannot be formed that deep. Each term \(C_s u^{s}\) of the central series of \(\varrho_\mathrm{aux}\) gives \(\mathrm{d}^N\varrho_\mathrm{aux}/\mathrm{d} P^N = (-1)^N C_s\, s(s-1)\cdots(s-N+1)\,u^{s-N}\), which exceeds the largest double below \(u = (|C_s\,s\cdots(s-N+1)|/\mathrm{DBL\_MAX})^{1/(N-s)}\) for \(s < N\). The largest such depth over the terms and over the orders that the form needs is the top limit of the table, and it is zero where no term grows. The rows end at the shallower of that depth and the depth at the smallest radius whose square the host represents (Section T:0.4); the closed form of the center owns the energies above. Where the leading exponent of the series vanishes (\(\gamma = 2\beta_\mathrm{c}\) at the center), there is no closed form to join to, and a model whose host floor alone would cut its table is refused with a named reason.

### T:2.5.2 Above the top of the table

The table of \(F_\mathcal{K}\) ends at a top node, a small depth \(u_{\rm top}\) below the center. Between that node and the center, the DF is continued by its central form, Section T:2.2.1 to Section T:2.2.4, which is computed from the model alone and stored with the table. The stored form consists of the leading singular term \(S_1(u)\) and the next one, \(S_2(u)\), each a power or a logarithm with the coefficient of Eq. (T:2.5) or Eq. (T:2.6). The coefficients of \(\varrho_\mathrm{aux}\) that they need come from the exact expansion of Section T:1.1, carried to the second power that is not a non-negative integer. No exponent is fitted to the table, and the amplitude is the closed form wherever the center states one.

What remains of \(F_\mathcal{K}\) once the two singular terms are removed is analytic in \(u\), and it changes across the gap by a relative amount of order \(u_{\rm top}\). It is therefore held at its value at the top node,

\begin{equation}
  F_\mathcal{K}(u) = S_1(u) + S_2(u) + \left[F_\mathcal{K}(u_{\rm top}) - S_1(u_{\rm top}) - S_2(u_{\rm top})\right], \qquad u < u_{\rm top} .
  \tag{T:2.21}
\end{equation}
Two cases depart from Eq. (T:2.21). Where the leading power is positive and the inversion is local (the last row of Table T:2.1), \(F_\mathcal{K}\) vanishes at the center, and the bracket is multiplied by \(u/u_{\rm top}\) so that the continuation vanishes with it. Where the center is analytic, there is no singular term, and \(F_\mathcal{K}\) is held at its top value. For a divergent power, the expression is evaluated in logarithms relative to \(S_1\), so that a DF whose central values exceed the range of double precision is continued without overflow.

Where the rows end below the gap (Section T:2.5.1), the node at which the form takes over is called the anchor. The regular part is then a fit \(B + D u\) to the rows below the anchor, over a window that widens until the three-term and the four-term fits agree in amplitude to one tenth of the goal; the singular exponent is never fitted. A band of rows at the top that is not positive is excluded only where the form holds there, as judged by the local slope \(\mathrm{d}\ln F/\mathrm{d}\ln u\) against \(\kappa\).

A truncated Method I model has \(\hat{F}_\mathcal{K}(\hat{\mathcal{K}}) = F_\mathcal{K}(\hat{\mathcal{K}} + \mathcal{K}_\mathrm{t}) - F_\mathcal{K}(\mathcal{K}_\mathrm{t})\), which differs from the untruncated DF by a constant. Its singular terms and their coefficients are those of the untruncated model, with \(u\) measured from the truncated central potential.

The stored form is tested against the table that it continues. Across the top decade of the table, \(u_{\rm top} \leq u \leq 10\,u_{\rm top}\), the table less its singular terms may change only by the variation of the analytic part. A change above the accuracy goal is reported with the product; a change above the hard limit means that the stored form belongs to a different model, and the product is refused.

### T:2.5.3 Below the floor

Below its first energy, the DF follows a stated rule that is recorded with the table: the tail \(A\,\mathcal{K}^{-a}[\ln(E_0/\mathcal{K})]^{-b}\) of Eq. (T:2.13) for an untruncated model, with its amplitude taken from the boundary term of Eq. (T:2.17) at the floor; the Taylor form \(F_1\mathcal{K} + F_2\mathcal{K}^2/2\) of a truncated Method I model at its edge, together with the energy up to which it holds; the edge law \(\mathcal{K}^{p}(C_0 + C_1\mathcal{K})\) of a Method II child (Section T:2.6.1); or zero. The integral of the rule from zero up to an energy below the table is the same set of numbers in closed form, so a density integral never truncates the DF at its table.

### T:2.5.4 The lookup

Inside the table, the base-10 logarithm of the DF is the cubic Hermite interpolant in \(\ln{\mathcal{K}}\) through the nodes with the exact slope column. Where a table carries no exact slopes (a DF from a function, Section T:2.7), the slopes are those of the not-a-knot spline through the nodes. The lookup has two entry points, by the energy and by the depth \(u = P_0 - \mathcal{K}\), and a caller asks by whichever argument it holds exactly. For example, a speed draw near the top holds the depth \(u = (P_0 - \Psi) + \Psi x^2\) exactly, whereas the energy \(\Psi(1 - x^2)\) formed in doubles loses it. In the entry by \(u\), the cell position is formed from \(u_i - u\) with \(u_i = P_0 - \mathcal{K}_i\), which is exact for \(\mathcal{K}_i \geq P_0/2\), and the form above the table is read at \(u\) itself. The nodes and every number of the forms are arrays, so that under JAX the table is a traced argument, and two tables with the same node bucket and the same kinds of forms share one compiled function.

### T:2.5.5 What a table states about itself

A table is written together with its checks: the stated slope column against the table’s own \(F\), the stated center against the top decade of the table, and the joins of the forms below the floor with the bottom row. A reader refuses a table that fails any of these beyond the hard limit. The sign report states where the DF is negative and the smallest truncation above which it is not (Section T:3.5). The validity certificate is a closed form: from the model and the central exponent, it states whether the columns of the table are representable in double precision over the energies that it would need, judged at the deepest of the gap, the top limit, and the host floor.

## T:2.6 A component on a tabulated pair (Method II)

Method II is not a second pipeline. A component whose profile is the truncated profile of another model runs through the same stages, and only the source of the derivatives differs. The density of the parent is the integral of its DF,

\begin{equation}
  \hat\varrho(P) = 4\pi\sqrt{2}\int_0^{P} F(\mathcal{K})\,(P - \mathcal{K})^{1/2}\,\mathrm{d}\mathcal{K} ,
  \tag{T:2.22}
\end{equation}
so its derivatives with respect to \(P\) are integrals of the DF and of its derivatives against \((P - \mathcal{K})^{-1/2}\), evaluated from the parent’s table, its exact slope column, and its Taylor edge; no spline of the rows is ever differentiated. The auxiliary density of the child is \(\varrho_\mathrm{aux} = h(R)\,\hat\varrho\) with the child’s own factor, and its derivatives follow by the Leibniz rule with \(\mathrm{d} R/\mathrm{d} P\) taken from the rows. Orders up to three are served, the third only where the parent states its Taylor edge. An isotropic or OM Type I parent is served, and a Type II child must have its \(R_\mathrm{a}\) beyond the parent’s edge.

The grid is the five-zone grid of Section T:2.4 on the first potential of the parent, compact, with no extension below the floor, since the support of the child ends at the parent’s edge. The stretch below the floor has a closed form. For each power \(P^{q}\) of the two-term edge expansion of \(\varrho_\mathrm{aux}\) (Section T:2.6.1), with \(a = q - m_f\), \(b = 1 - n_f\), and \(x = q_0/\mathcal{K}\),

\begin{equation}
  C(n_f)\,a_q\,\frac{\Gamma(q+1)}{\Gamma(a+1)}\;\mathcal{K}^{\,a-n_f}\,x^{a}\left[{}_2F_1(a, n_f;\, a+1;\, x) - 1\right]
  \tag{T:2.23}
\end{equation}
is the exact contribution of \([0, q_0]\) to the row at \(\mathcal{K}\), relative to the boundary form that the kernel carries, and the exact value of \(\varrho_\mathrm{aux}^{(m_f)}\) at the floor is the boundary term.

At \(\beta_\mathrm{c} = 1/2\) on a parent with \(\gamma = \alpha = 1\), the inversion is a derivative whose two terms are both of order \(1/u\) toward the center, so the rows there carry the parent’s accuracy multiplied by \(1/(4u)\). Above an anchor, the rows are the closed form \(F = A\sum_{k \leq 4} a_k u^{k}\) with \(A = 1/(2\pi^2)\) (Section T:2.2.5), which is exact in the truncation through the parent’s central deficit. The anchor is the smaller of the series’ own range at the goal and the top zone of the grid, and the computed rows over the outermost decade inside it must agree with the form to within the goal; otherwise the anchor steps inward one decade at a time until they do.

### T:2.6.1 The edge of a Method II child

A density that ends as \(P^{q}\) at its edge has a DF of central anisotropy \(\beta_\mathrm{c}\) if and only if \(\beta_\mathrm{c} > 1/2 - q\), and the DF then behaves as \(\mathcal{K}^{\,q - 3/2 + \beta_\mathrm{c}}\) there. This statement holds for every profile and every anisotropy model of the package, and a child that violates it is refused before any stage runs, with the admissible range stated. Below the parent’s switch energy, the parent’s DF is its Taylor form \(F_1\mathcal{K} + F_2\mathcal{K}^2/2\), so the parent’s density near the edge is \(4\pi\sqrt{2}\,[\tfrac{4}{15}F_1 P^{5/2} + \tfrac{8}{105}F_2 P^{7/2}]\) to the order kept. The child’s augmented density, with \(h = h_\mathrm{t} + H_1 P\), \(H_1 = h'(R_\mathrm{t})\,R'(P)\), and \(R' = -R_\mathrm{t}^2/M_\mathrm{t}\), is then

\begin{equation}
  \varrho_{\rm aug}(P) = a_{5/2}\,P^{5/2} + a_{7/2}\,P^{7/2} + \mathcal{O}(P^{9/2}) .
  \tag{T:2.24}
\end{equation}
Each power inverts to a power, so the child’s DF below its switch is the edge law \(\mathcal{K}^{p}(C_0 + C_1\mathcal{K})\) with \(p = 1 + \beta_\mathrm{c}\), and the same coefficients give the floor terms of Eq. (T:2.23).

## T:2.7 A distribution function from a function

A DF obtained elsewhere, from a fitting function or from the output of another code, is certified in the same way. Its values on the energy grid that the inversion would place for the component enter at the point where the rows of the inversion would enter, and everything downstream is unchanged: the forms beyond the table, the sign report, the checks, and the record. Such a table has no exact slope column, which its report states, and a value that is not finite is refused, with the first such energy named. Truncation needs the rows of the inversion, which such a table does not carry, and it is refused with a named reason.

     On this page
