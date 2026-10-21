.. _tn-sec-observables:

T:6 Velocity profiles and projections
=====================================

An observable is a function of a model, and a request builds only the stages it needs: a second moment builds the dispersion profile and the projected table, over all their rows, and no table of velocity distributions. Two routes exist: the second moments follow from the density and the potential, and, on a model that is not spatially finite, from the DF below the table's lowest potential, whereas the velocity distributions follow from the DF. The second route reads the first's surface density and projected dispersion only to place its first and last rows, to decide which breaks of its velocity panels it keeps, and to check its rows, so the two routes check each other. The prepared table of a model holds both routes and reads from them.

.. _tn-sec-num-sup:

T:6.1 Intrinsic moments
-----------------------

.. _tn-sec-jeans-section:

T:6.1.1 The Jeans integral
~~~~~~~~~~~~~~~~~~~~~~~~~~

For each component, the radial dispersion follows from the Jeans equation in the form

.. _tn-eq-jeans:

.. math::
   :nowrap:

   \begin{equation}
     \varrho_\mathrm{aux}\,\sigma_r^2\,(R) = \int_0^{P(R)} \varrho_\mathrm{aux}\,\mathrm{d} P ,
     \tag{T:6.1}
   \end{equation}

with the member's own :math:`\varrho_\mathrm{aux}`; the part of the integral below the lowest potential of the table is taken over the DF, and it is zero for a spatially finite model. The profile is tabulated in radius on rows from the first row of the profile to its last. Rows are added at the midpoints, in :math:`\ln R`, of the intervals whose reading misses the profile tolerance in :math:`\sigma_r` there, round by round, until every interval reads within it, or until three rounds running make no progress or :math:`40` rounds have run, when the profile's certificate records the intervals left above the tolerance. Before the rows are placed, the integral is taken over Gauss-Legendre panels between the profile's own table potentials, and the nodes of each panel, from eight, are doubled until :math:`\sigma_r` at the midpoints of the starting rows moves by less than the profile tolerance, at most six times; a rule that has not converged by then is recorded in the certificate, and the rows are placed against it. The superposition's own moments take the same integral at each radius on one panel of a stated number of Gauss-Legendre nodes in :math:`\ln(1 - \mathcal{K}/P_0)`, which resolves the center; the two routes agree to the accuracy of that fixed rule, which no certificate tests (:math:`6.5\times10^{-8}` at the innermost row at the default of :math:`500` nodes, measured on a :math:`64`-member subset), not to rounding. The tangential dispersion of both tangential components together is :math:`\sigma_t^2 = 2(1 - \beta_\sigma)\,\sigma_r^2`, with :math:`\beta_\sigma` the anisotropy of the model in closed form.

.. _tn-sec-projection:

T:6.2 Projection
----------------

The surface density and the line-of-sight dispersion, |paper|, Eqs. (B47) and (B46), come from a single line-of-sight rule with a single density, on two panels in the depth along the line of sight, at projected radii spaced logarithmically from the innermost row to a fixed fraction of the last row. Rows are added until the table reads itself between rows to within the certificate's tolerance, :math:`10^{-8}`: at the midpoint of every interval the value read is compared with a row computed there, and a midpoint that misses becomes a row; the rows also stop when two rounds running do not halve the worst miss, and the certificate then records the end as stalled. On a spatially finite model, the surface density is read as :math:`\Sigma = (R_\mathrm{t}^2 - R^2)^{1/2} h`, the factor exact and :math:`\ln h` a cubic in :math:`x = \ln[R/(1 - R/R_\mathrm{t})]`, which goes as :math:`\ln R` far inside and as :math:`\ln R_\mathrm{t} - \ln(1 - R/R_\mathrm{t})` toward the edge, so that a power law of :math:`h` in :math:`R_\mathrm{t} - R` is a straight line in :math:`x`; the squared dispersion is read in :math:`\ln R`. A projected radius inside the innermost row or beyond the last is refused, with the row and what places it stated; a negative squared dispersion, from the rule at a row or read between rows, is refused as well.

.. _tn-sec-aperture:

T:6.3 Apertures
---------------

The dispersion inside an aperture is the surface-density weighted average of :math:`\sigma_{\rm los}^2`, |paper|, Eq. (B50). For an axisymmetric integrand every aperture reduces to a one-dimensional integral, :math:`\int f(R)\,\mathrm{arc}(R)\,R\,\mathrm{d} R`, where :math:`\mathrm{arc}(R)` is the angular measure of the circle of radius :math:`R` that lies inside the aperture. A centered circle takes a Gauss-Legendre rule in :math:`\ln R` from the first row of the table, with at least a stated number of nodes and more per decade of its span. An offset circle has a full-circle core and a band of its own, the band under an end-point map in :math:`R - R_{\rm off}` where it is narrow. A rectangle takes one panel between consecutive breaks of its arc, each with at least a stated number of Gauss-Legendre nodes and more per decade of its span. Above :math:`R_\mathrm{t}/\sqrt{2}`, where the factors :math:`R` and :math:`v` of the measure :math:`R\,\mathrm{d} R = -v\,\mathrm{d} v` are equal, a panel is taken in :math:`v = (R_\mathrm{t}^2 - R^2)^{1/2}`, in which the integrand carries the edge factor :math:`v` as a smooth factor; the narrow band of an offset circle is the exception and stays one panel in :math:`R - R_{\rm off}`. An aperture that encloses no tabulated surface density, or whose disk inside the first row carries more than a stated share of its weight, is refused with the reason stated. That disk is estimated from the slope :math:`s = -\mathrm{d}\ln\Sigma/\mathrm{d}\ln R` between the first two rows, as the power law :math:`\Sigma_0 (R/R_0)^{-s}` continued inward, which holds :math:`\Sigma_0 R_0^2/(2 - s)`; the estimate is finite only for :math:`s < 2`, and a slope at or above 2 is refused.

.. _tn-sec-losvd-rows:

T:6.4 The line-of-sight velocity distribution
---------------------------------------------

The distribution :math:`D(R_{\rm 2D}, V)` of |paper|, Eq. (B53) is computed row by row at projected radii. On the plane :math:`V_{\rm los} = V` at depth :math:`u` along the line of sight, the level sets of :math:`Q` are ellipses centered on the maximum of :math:`Q`, so the integral over the velocity plane is

.. _tn-eq-ellipse:

.. math::
   :nowrap:

   \begin{equation}
     \iint F\,\mathrm{d} V_{\perp 1}\,\mathrm{d} V_{\perp 2} = \frac{R/R_{\rm 2D}}{\sqrt{Aq}}\int_0^{Q_c} F_\mathcal{K}(Q_c - s)\,\Lambda(s)\,\mathrm{d} s ,
     \tag{T:6.2}
   \end{equation}

with :math:`Q_c` the maximum, :math:`q = 1 + (R/R_\mathrm{a})^2` for Type I, :math:`1 - (R/R_\mathrm{a})^2` for Type II and :math:`1` without :math:`R_\mathrm{a}`, and :math:`A = 1 + q u^2/R_{\rm 2D}^2`, so that :math:`Aq\,(R_{\rm 2D}/R)^2` is the product of the coefficients of the ellipse's axes, and :math:`\Lambda(s)` the integral of :math:`(L^2)^{-\beta_\mathrm{c}}` around the level set. Where the DF carries no :math:`L` factor, :math:`\Lambda = 2\pi`, and the :math:`s` integral is the cumulative DF :math:`G(Q_c)`, read from a table of three branches. With the :math:`L` factor, :math:`\Lambda = 2R^{-2\beta_\mathrm{c}}\int_0^{\pi}[(c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi]^{-\beta_\mathrm{c}}\,\mathrm{d}\phi`, with :math:`c_1 = VR/(R_{\rm 2D}A)`, :math:`c_2 = (u/R_{\rm 2D})(2s/A)^{1/2}` and :math:`c_3 = 2s/q`; the integral over :math:`[0, \pi]` is taken by a split rule dense toward :math:`\phi = 0` on :math:`[0, \pi/2]` and toward :math:`\phi = \pi` on :math:`[\pi/2, \pi]`, with the width :math:`\lambda` of the base's quadratic minimum at that end (:math:`\pi` where that end is no local minimum), :math:`\phi = \lambda\sinh(\mu x)` on :math:`[0, \pi/2]` with :math:`\mu = \mathrm{arsinh}(\pi/2\lambda)`, at a count of Gauss-Legendre nodes per half chosen for each node from a stated list as the least whose convergence rate, read from a table of the distances of the zeros of the base, reaches :math:`10^{-8}`, a node beyond the list or off the table taking the largest, as a JAX kernel; at :math:`\beta_\mathrm{c} = \pm 1/2` it is a complete elliptic integral in closed form. The :math:`s` integral is taken on Gauss-Legendre panels graded toward :math:`s = Q_c`, where :math:`K = Q_c - s = 0`; the last panel is mapped by the power of the DF's rule below its table, so that the integral of that rule's power law is exact. The point of zero angular momentum :math:`s_0`, where the non-analytic part of :math:`\Lambda` behaves as :math:`|s - s_0|^{1-2\beta_\mathrm{c}}` (times :math:`\ln|s - s_0|` at a non-negative integer power), is the split between the two pieces of the :math:`s` rule, and each side takes graded layers toward it whose count, beyond the layers that bring the panel next to it within the pair's own scale, follows from :math:`\beta_\mathrm{c}` alone: the least count that integrates the end-point power to :math:`10^{-8}`; where that power is negative (:math:`1/2 < \beta_\mathrm{c} < 1`) the layers are self-similar, the count is the one at which the error stops falling, and that error, the floor, is recorded and refused above :math:`10^{-6}`. For :math:`0 < \beta_\mathrm{c} < 1` the first panel above the split is mapped as :math:`s - s_0 \propto w^{1/(1-\beta_\mathrm{c})}`, which carries the point :math:`V = 0`, where :math:`\Lambda \propto s^{-\beta_\mathrm{c}}`; for :math:`\beta_\mathrm{c} < 0` not an integer that point takes the least count that integrates :math:`s^{-\beta_\mathrm{c}}` to :math:`10^{-8}`. The depth integral is taken in :math:`t = \mathrm{arsinh}(u/R_{\rm 2D})` on Gauss-Legendre panels of each interval of the support along the line of sight, split where the point of zero angular momentum enters the :math:`s` range, with layers toward each interval's lower end; the panels are doubled, at most a stated number of times, until the row changes by less than :math:`10^{-6}` both in its weighted :math:`L^1` norm and pointwise, and a row that does not converge is kept and reported.

A row is taken on panels in :math:`V` between 0 and :math:`V_{\rm end}`, the largest velocity with a support along the line of sight, with breaks at the velocities where its layout changes: the velocity of the radial orbit, :math:`V_{\rm rad}` (on the families with an :math:`L` factor), the velocity :math:`V_k = [2P(R_{\rm 2D})/q(R_{\rm 2D})]^{1/2}` above which :math:`Q_c < 0` at :math:`u = 0`, the velocity :math:`V_{\rm edge}` at the end of a line of sight that ends with :math:`P > 0`, and the velocity :math:`V_c = [2(P_0 - P(R_{\rm 2D}))]^{1/2}` where the DF's top turns the distribution over too sharply for a sub-panel. :math:`V_{\rm rad}` is kept as a break only where :math:`2D(V_{\rm rad})(V_{\rm end} - V_{\rm rad})`, a share of the row's norm, or of its second moment with the weight :math:`V_{\rm end}^2`, exceeds :math:`10^{-8}`, and :math:`V_k` only where the same estimate for each of :math:`[V_k, V_{\rm end}]` and :math:`[0, V_k]` does, or where :math:`V_k` falls among the nodes of the panel it would be merged into (decided once on the base rows and, for :math:`V_k`, on samples beyond the last toward :math:`R_{\rm t}`, with each change bisected in :math:`\ln R_{\rm 2D}`); any break whose panel the row's nodes cannot resolve in doubles is merged. Every panel takes sub-panels graded toward zero velocity and toward each break at a stated ratio, and with the innermost sub-panel kept above the spacing of doubles at the break. Every row is normalized to one and checked against two quantities computed without it: its integral against the surface density, and its second moment against the projected dispersion (:ref:`Section T:6.2 <tn-sec-projection>`).

A general-purpose integrator applied to the same integral, as in AGAMA (`Vasiliev 2019 <https://doi.org/10.1093/mnras/sty2672>`__), stops by default at a relative tolerance of :math:`10^{-3}` for the moments and of :math:`10^{-2}` for the distribution. The rows here are rules built on each end-point behavior of the integrand, the depth rule and the angle count adaptive within stated caps, the end point at :math:`L = 0` for :math:`1/2 < \beta_\mathrm{c} < 1` integrated to a recorded floor no larger than :math:`10^{-6}`, and each row is held to :math:`10^{-6}` in its own convergence test and against the surface density and the projected dispersion, with a row that does not converge reported.

.. _tn-sec-losvd-table:

T:6.5 The table of rows and its certificate
-------------------------------------------

The base rows are placed at a stated number per decade of projected radius. The base rows run from the first row of the profile to :math:`0.9999` of the radius where the rows end (the last row of the profile, or the radius where the potential reaches the DF's floor where the DF is zero below it). By default the table keeps the rows of that grid, with the rows added toward each R-break and toward the end, out to the first row at or beyond the radius outside which :math:`10^{-6}` of the total projected mass lies; the rows beyond it are not computed, and a read beyond the last kept row, including a read at a single point, is refused, the refusal stating the radius of that row and the setting that moves it. A table whose rows reach the end is read up to the end. The inner end is kept at full depth by default. A user who will read the table only outside a stated projected radius may give that radius, and the rows then start where the projected mass inside the first row falls below :math:`10^{-6}` times the projected mass inside the stated radius; a read inside the first row is refused in the same words. Rows are added toward each R-break (each radius where the layout of the panels changes and each radius where a break merged by the lattice crosses one of its nodes), toward the end, and into any stretch between R-breaks that holds fewer than six rows; a row that its nodes cannot resolve is dropped. Rows are then added at the middle of each interval between two neighboring rows of one stretch (in the stencil's variable) where the stencil of a stated number of rows and the stencil of two rows fewer differ by more than :math:`10^{-8}` in the second moment or :math:`10^{-7}` of the row's largest value pointwise, round by round, until no interval fails, a round no longer halves the worst, or no failing interval can take a row; the same estimate is taken in the gap between each true end of a stretch and its nearest row. An interval that cannot be closed is recorded with its reason, and a request inside it is refused with that reason stated. An interval of a stretch of fewer than four rows is not checked; the table counts it and reads it with the rows the stretch holds. The certificate of the table is its last estimate of its reading between rows, the worst over the intervals in units of their tolerances; the report states it times the larger tolerance, :math:`10^{-7}`, a bound on both estimates. The estimate in the end gaps is recorded beside it. The rows are computed in a pool of processes, as many as the cores set, each process computing one row at a time. The report of the table (:ref:`Section T:8.1 <tn-sec-reports>`) carries the worst norm and second moment over the rows, the rows that could not be judged, the rows whose depth rule did not converge, the certificate, and the floor of the layers at the point of zero angular momentum where :math:`\beta_\mathrm{c}` lies between :math:`1/2` and :math:`1`.

.. _tn-sec-aperture-losvd:

T:6.6 The distribution in an aperture
-------------------------------------

The distribution inside an aperture is the surface-density weighted average of the rows over the aperture, |paper|, Eqs. (B58) and (B59). No row is computed for it. The rows are read at the radii of the aperture's rule (:ref:`Section T:6.3 <tn-sec-aperture>`), with every break of a row followed as a function of radius, and each radius is scanned once. The result is formed on its own panels in :math:`V`, with breaks at zero, at the smallest and the largest value of each row break over the aperture, and at the top; a velocity that meets a break curve inside the aperture takes the radii where the curve equals it as breaks of its radial rule. The output rule refines itself, bisecting each sub-panel wider than half the widest, until its norm and its second moment agree with those of the rows' own rule (each row integrated on its own nodes) to one hundredth of :math:`10^{-6}`, until its widest sub-panel is no wider than the narrowest it began with, or until a round moves them by less than that; the end is recorded. At any velocity, the distribution is the polynomial of its panel through the panel's own nodes, namely the Legendre series of the antiderivative of the quadrature, so that it integrates exactly as the quadrature does. Its second moment is checked against the dispersion of the aperture from the projected tables, which reads no distribution.

.. _tn-sec-gauss-hermite:

T:6.7 Gauss-Hermite moments
---------------------------

With :math:`y = (V - V_*)/\sigma_*`, where :math:`V_*` and :math:`\sigma_*` are the mean and the dispersion of the distribution itself, the moments of |paper|, Eq. (B63) are

.. _tn-eq-gauss-hermite:

.. math::
   :nowrap:

   \begin{equation}
     h_n = 2\sqrt{\pi}\int D(V)\,H_n(y)\,\frac{e^{-y^2/2}}{\sqrt{2\pi}}\,\mathrm{d} V , \qquad H_3 = \frac{2y^3 - 3y}{\sqrt{3}}, \quad H_4 = \frac{4y^4 - 12y^2 + 3}{2\sqrt{6}} ,
     \tag{T:6.3}
   \end{equation}

with the Gauss-Hermite polynomials of |paper|, Eq. (B61). They are taken on the velocity nodes and weights of a row or of an aperture distribution, mirrored to both signs and normalized to unit integral. For one distribution the standardized moments by direct sums are computed beside them; to first order, :math:`s_3 = 4\sqrt{3}\,h_3` and :math:`s_4 = 3 + 8\sqrt{6}\,h_4`. The moments tabulated over radius and those over apertures are :math:`V_*`, :math:`\sigma_*`, :math:`h_3` and :math:`h_4`. Moments at projected radii are tabulated once over radius and read between rows; moments over apertures are taken on the distribution of each aperture.

.. _tn-sec-pdf-v:

T:6.8 The local velocity distribution
-------------------------------------

With :math:`t = \cos\eta`, where :math:`\eta` is the angle between the velocity and the radius, :math:`\xi = (R/R_\mathrm{a})^2`, and :math:`s` the sign of the anisotropy (:math:`+1` for Type I, :math:`-1` for Type II, and :math:`0` without :math:`R_\mathrm{a}`),

.. _tn-eq-pdf-v:

.. math::
   :nowrap:

   \begin{equation}
     p(V\,|\,R) = \frac{4\pi}{\varrho(R)}\,R^{-2b}\,V^{2-2b}\int_0^{1}(1 - t^2)^{-b}\,F_\mathcal{K}(Q)\,\mathrm{d} t , \qquad Q = P(R) - \frac{V^2}{2}(1 + s\xi) + \frac{s\xi V^2}{2}\,t^2 ,
     \tag{T:6.4}
   \end{equation}

where :math:`b` is :math:`\beta_\mathrm{c}` for the constant and COM classes and 0 otherwise. Since :math:`Q = c_0 + c_2 t^2`, with :math:`c_0` and :math:`c_2` independent of :math:`t`, is monotonic in :math:`t` on :math:`[0, 1]`, the support of the DF in :math:`t` is a single interval, :math:`[t_*, 1]` for Type I and :math:`[0, t_*]` for Type II, and the integral is taken over that interval, so the edge of the support is an end of the interval of integration and never an interior point of it. The interval is split where :math:`Q` crosses the first energy of the table, and for :math:`b > 0` the piece that ends at :math:`t = 1` is integrated in :math:`w`, with :math:`1 - t = w^{p}` and :math:`p = 1/(1 - b)`, so that :math:`(1 - t)^{-b}\,\mathrm{d} t = p\,\mathrm{d} w`. A Type II model holds unbound orbits, of negative relative energy :math:`P(R) - V^2/2 < 0`, and its velocities reach :math:`V_{\rm esc}/\sqrt{1 - R^2/R_\mathrm{a}^2}`; the share beyond the escape velocity is reported, and it was measured at :math:`9.3\times10^{-5}`, :math:`1.9\times10^{-3}` and :math:`5.4\times10^{-2}` at :math:`R/R_\mathrm{a} = 0.1`, :math:`0.21` and :math:`0.5`, about a fifth of :math:`(R/R_\mathrm{a})^2`. Each radius is computed with three checks: the norm against the density, the second moment against :math:`\sigma_r^2(3 - 2\beta_\sigma)`, and the change of the distribution on its velocity nodes, relative to its largest value and away from the velocity at which the support starts to cut the directions, when the nodes of its angle rule are doubled: starting from a stated number of nodes, the rule is doubled until that change falls below the tolerance, at most three times, and the distribution is kept on the coarser rule of the last pair compared.

.. _tn-sec-mixing:

T:6.9 Superposed models
-----------------------

A superposed model (:ref:`Section T:4 <tn-sec-sup>`) is projected one component at a time, each on its own tables, and the components are then combined with their weights. The members share one surface density, so the projected second moment is linear in the weights,

.. _tn-eq-sup-los:

.. math::
   :nowrap:

   \begin{equation}
     \sigma_{\rm los}^2(R_{\rm 2D}) = \sum_i c_i\,\sigma_{{\rm los},i}^2(R_{\rm 2D}) ,
     \tag{T:6.5}
   \end{equation}

and the same holds for the intrinsic second moments. With the projected moments of each member tabulated once, the composite profile for a new set of weights costs two weighted sums over the members, of :math:`\Sigma_i\sigma_{{\rm los},i}^2` and of :math:`\Sigma_i`, and their ratio, which is what the likelihood of :ref:`Section T:7.4 <tn-sec-likelihood>` evaluates. A distribution at a projected radius mixes by the surface densities, :math:`D = \sum_i c_i\Sigma_i D_i/\sum_i c_i\Sigma_i`; a distribution in an aperture mixes by the projected masses inside it; and the local velocity distribution mixes by the densities at that radius.
