Code and paper notations
========================

The package's identifiers follow the notation of |paper|. A quantity with a symbol in |paper| or in the
Technical notes is called ``<family>_<symbol>``. The family prefix gives the kind of quantity, and the rest of the
name is the symbol or the symbol's subscript in lower case. A truncated quantity (hatted in |paper|) takes the
suffix ``_trunc``, as in ``p_trunc``, ``df_trunc``, ``rho_trunc`` and ``mass_trunc``. The untruncated quantity
keeps the name without the suffix. Every docstring gives the
symbol beside the name the first time the docstring describes the quantity.

.. list-table::
   :header-rows: 1
   :widths: 24 12 34 30

   * - Family
     - Prefix
     - Names
     - Paper
   * - density and its parameters
     - ``rho_``
     - ``rho``, ``rho_aux``, ``rho_alpha``, ``rho_beta``, ``rho_gamma``, ``rho_norm``
     - :math:`\varrho`, :math:`\varrho_{\rm aux}`, :math:`(\alpha, \beta, \gamma)`, :math:`C`
   * - anisotropy
     - ``beta_``
     - ``beta_c``, ``beta_sigma``
     - :math:`\beta_{\rm c}`, :math:`\beta_\sigma`
   * - radii
     - ``r_``
     - ``r``, ``r_a``, ``r_t``, ``r_2d``, ``r_off``, ``r_ape``
     - :math:`R`, :math:`R_{\rm a}`, :math:`R_{\rm t}`, :math:`R_{\rm 2D}`, :math:`R_{\rm off}`, :math:`R_{\rm ape}`
   * - relative potential
     - ``p_``
     - ``p``, ``p_0``
     - :math:`P`, :math:`P_0`
   * - energy-like variables
     - ``k_``, ``q_``
     - ``k``, ``k_t``, ``q``, ``q_min``, ``q_0``
     - :math:`\mathcal{K}`, :math:`\mathcal{K}_{\rm t}`, :math:`\mathcal{Q}`, :math:`\mathcal{Q}_{\rm min}`, :math:`\mathcal{Q}_0`
   * - distribution function
     - ``df_``
     - ``df``, ``df_k``
     - :math:`F`, :math:`F_\mathcal{K}`
   * - mass
     - ``mass_``
     - ``mass``, ``mass_t``, ``mass_0``
     - :math:`\mathcal{M}`, :math:`M_{\rm t}`, :math:`M_0`
   * - order of the inversion
     - ``order_``
     - ``order_m``, ``order_n``
     - :math:`(m, n)`
   * - velocities and dispersions
     - ``v_``, ``sigma_``
     - ``v_r``, ``v_t``, ``v_los``, ``v_esc``, ``sigma_r``, ``sigma_t``, ``sigma_los``
     - :math:`V_{\rm r}`, :math:`V_{\rm t}`, :math:`V_{\rm los}`, :math:`V_{\rm esc}`, :math:`\sigma_{\rm r}`, :math:`\sigma_{\rm t}`, :math:`\sigma_{\rm los}`
   * - probability densities
     - ``pdf_``
     - ``pdf_v``
     - :math:`p(V \,|\, R)`

Where the letter of |paper| cannot serve as a name, a fixed word replaces the letter. The normalization :math:`C` of the
density is ``rho_norm``, since ``rho_c`` would read as a central density. The concentration :math:`c` is
``concentration``. The letter ``c`` alone is never a name. As a subscript, ``c`` means central (``beta_c``).

Derived names
-------------

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - What
     - Form
     - Example
   * - logarithm of a quantity
     - ``ln_``, ``log10_`` prefix
     - ``ln_r``, ``log10_df``
   * - derivative
     - ``d<y>_d<x>``, the order as a digit
     - ``drho_dp``, ``d2rho_dp2``
   * - count
     - ``n_`` prefix
     - ``n_nodes``, ``n_rows``
   * - index
     - ``i_`` prefix
     - ``i_first``, ``i_last``
   * - quadrature nodes and weights
     - ``_nodes``, ``_weights`` suffix
     - ``k_nodes``, ``k_weights``
   * - tabulated values on a grid
     - ``_grid`` suffix
     - ``r_grid``, ``df_grid``
   * - boolean
     - ``is_``, ``has_`` prefix
     - ``is_physical``, ``has_edge``

The three unit systems are ``internal``, ``model`` and ``physical`` (:doc:`../guide/units`).
