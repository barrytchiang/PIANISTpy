.. _tn-sec-model:

T:1 The model: density and potential
====================================

.. _tn-sec-pair:

T:1.1 The two maps
------------------

The potential of the double power law is given in |paper|, Eq. (A3), and its inner and outer expansions in Eq. (A11) and Eq. (A13). Two consequences of those expansions are used below.

*Near the center*, :math:`P_0 - P \propto R^{2-\gamma}\,[1 + \mathcal{O}(R^{\alpha})]`. Since :math:`R^{\alpha} \propto (P_0-P)^{\delta}`, any function of :math:`R` that is a series in :math:`R^{\alpha}` is also a series in :math:`(P_0 - P)^{\delta}`:

.. _tn-eq-inner-map:

.. math::
   :nowrap:

   \begin{equation}
     R^{\alpha} \propto (P_0 - P)^{\delta}\left[1 + \mathcal{O}\!\left((P_0-P)^{\delta}\right)\right] .
     \tag{T:1.1}
   \end{equation}

The coefficients of such a series are exact, and no fit is needed. The depth :math:`P_0 - P` is itself a series in :math:`R^{\alpha}` with known coefficients, |paper|, Eq. (A11). Inverting that series order by order gives :math:`R = t\,[1 + S(t)]`, with :math:`t \propto (P_0-P)^{1/(2-\gamma)}` and :math:`S` a series in :math:`t^{\alpha}`. Substituting this inverse into the density and into the anisotropy factor of :math:`\varrho_\mathrm{aux}` yields the coefficient of every power :math:`s` of :math:`P_0 - P` as :math:`C\,c_u^{-s}`, with :math:`c_u = C/[(2-\gamma)(3-\gamma)]`, times a finite sum of products of generalized binomial coefficients, of the depth's relative coefficients :math:`(2-\gamma)(3-\gamma)\binom{e}{k}/[(2-\gamma+k\alpha)(3-\gamma+k\alpha)]`, :math:`e = (\gamma-\beta)/\alpha`, and, for the OM and COM factors, of powers of :math:`\pm R_\mathrm{a}^{-2}`. The package forms these coefficients in closed form by a power-series algebra, keeping every term up to the power :math:`3\max(\alpha, 2)` of :math:`t` beyond the leading power, that is, up to :math:`(P_0-P)^{3\max(\alpha,2)/(2-\gamma)}` beyond the leading power of :math:`P_0 - P`.

*At large radii*, the potential of an untruncated profile behaves as

.. _tn-eq-outer-map:

.. math::
   :nowrap:

   \begin{equation}
     P \propto \begin{cases} R^{-1} & (\beta > 3),\\
      R^{-1}\ln R & (\beta = 3),\\
      R^{-(\beta-2)} & (2 < \beta < 3). \end{cases}
     \tag{T:1.2}
   \end{equation}

.. _tn-sec-num-pair:

T:1.2 Derivatives of the auxiliary density
------------------------------------------

.. _tn-sec-derivatives:

T:1.2.1 The operator
~~~~~~~~~~~~~~~~~~~~

The inversion with the order pair :math:`(j, n')` of :ref:`Eq. (T:2.16) <tn-eq-unified-pair>` requires :math:`\mathrm{d}^k\varrho_\mathrm{aux}/\mathrm{d}P^k` up to :math:`k = j+1` (:math:`k = j = m` at :math:`n = 0`), that is up to :math:`m+1`, or :math:`m+2` for a replaced pair; Method I reads one order higher as well, as the slope of its source and as the source of the slope of the DF. In Method I the profile is analytical and these derivatives are exact. The operator

.. _tn-eq-dp-operator:

.. math::
   :nowrap:

   \begin{equation}
     \frac{\mathrm{d}}{\mathrm{d} P} = -\frac{R^2}{M(R)}\,\frac{\mathrm{d}}{\mathrm{d} R},
     \tag{T:1.3}
   \end{equation}

with the closed-form enclosed mass :math:`M(R)` of |paper|, Eq. (A7), is applied symbolically to the closed-form auxiliary density :math:`\varrho_\mathrm{aux}(R) = \varrho(R)\,h(R)`, where :math:`h = R^{2\beta_\mathrm{c}}(1 \pm R^2/R_\mathrm{a}^2)^{s}` is the inverse of the anisotropy factor (:math:`s = 1` for OM, :math:`s = 1 - \beta_\mathrm{c}` for COM, and :math:`s = 0` otherwise). The result is evaluated at :math:`R(P)`, which we obtain by bisection in :math:`\ln R` on the monotonic :math:`P(R)` and, near the center, on the subtraction-free depth. The potential :math:`P(R)` does not enter the derivatives' expressions, which hold :math:`M(R)` only; its closed form, |paper|, Eq. (A3), enters only through the radius :math:`R(P)` at which they are evaluated. A single expression of :math:`h` serves every closed-form evaluation of the factor in the package: the Method I sources, the Method II sources, the edge law, the density of the Poisson solve, and the parameter derivatives of :ref:`Section T:7 <tn-sec-gradients>`; the inner series of :ref:`Section T:1.1 <tn-sec-pair>` and its parameter derivatives expand the same factor as a binomial series in :math:`R^2/R_\mathrm{a}^2`, and the central deficit of :ref:`Section T:3.4 <tn-sec-fills>` writes the factor's logarithmic step and slope out by hand. In Method II the input is a tabulated pair, and the derivatives are integrals of the parent's DF (:ref:`Section T:2.6 <tn-sec-tabulated>`).

Our first design for the tabulated case was a spline of :math:`\ln{\varrho_\mathrm{aux}}` in :math:`P` with knots placed by the equidistribution principle (`de Boor 1978 <https://link.springer.com/book/9780387953663>`__). Its derivatives of second and higher order carried the conditioning of the spline, which grows as the square of the ratio of its largest to its smallest knot spacing, into the inversion. The exact and integral forms replaced it.

.. _tn-sec-series:

T:1.2.2 The inner series and where it is trusted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Near the center, the density is its closed form, and the enclosed mass and the depth are the series of |paper|, Eq. (A12) and Eq. (A11), carried to four terms in :math:`R^{\alpha}` and summed term by term with no subtraction from the central value; the potential is :math:`P_0` less that depth. The series is trusted out to the largest radius at which its next term, relative to the depth, stays within :math:`10^{-7}`. This bound follows from the parameters of the profile alone, and no fixed depth is assumed. The Poisson solve of a truncated model starts from the same series, and its inner rows are filled from the series out to that join (:ref:`Section T:3.4 <tn-sec-fills>`).

.. _tn-sec-profile-object:

T:1.3 The profile of a truncated model
--------------------------------------

The self-consistent profile of a truncated model is a table of rows (radius, potential, density, and enclosed mass), and a single object owns its reading: every stage and every reader evaluates the profile through the four functions of that object and holds no interpolant of its own. The potential and the mass are cubic Hermite interpolants in the logarithms, with the exact slopes that the rows state, :math:`\mathrm{d}\ln P/\mathrm{d}\ln r = -M/(rP)` and :math:`\mathrm{d}\ln M/\mathrm{d}\ln r = r^3\varrho/M` (the Poisson equation and the mass integral). The density is a cubic spline of :math:`\ln\varrho` in :math:`\ln r`, since the rows state no density slope. The radius at a given potential is the guarded inverse of the potential interpolant. A spatially finite profile carries its edge as its last row, where the density is zero and the radius is :math:`R_\mathrm{t}`. On the last interval the potential takes the Kepler form :math:`M_\mathrm{t}(1/r - 1/R_\mathrm{t})` and the density follows the edge law :math:`\varrho_L\,(P/P_L)^{q}`, with the exponent :math:`q` stated by the profile; beyond :math:`R_\mathrm{t}` the potential continues as :math:`M_\mathrm{t}(1/r - 1/R_\mathrm{t})` and the density is zero; the package serves no reading there. Inside the first row, the interpolants continue as the power laws of the end rows. Nothing outside the domain is clipped: a radius that lies beyond the edge by a rounding error is treated as the edge, and anything further out is refused with the reason stated.
