PIANISTpy
=========

**P**\ article **I**\ nitial conditions with **ANIS**\ o\ **T**\ ropy.

PIANISTpy builds spherical, non-rotating, velocity-anisotropic equilibrium models, self-consistently truncated in
phase space, for dark matter (sub)halos, dwarf galaxies, stellar spheroids and star clusters. The package provides
distribution functions, particle initial conditions, radial and tangential velocity dispersion profiles and
line-of-sight observables, with JAX-based end-to-end differentiability for gradient-based Monte Carlo inference.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Get started
      :link: start/index
      :link-type: doc

      What the package does, installation and dependencies, a quick start, and the representative computing time.

   .. grid-item-card:: User guide
      :link: guide/index
      :link-type: doc

      One page per topic, each with short code examples.

   .. grid-item-card:: Examples
      :link: notebooks/index
      :link-type: doc

      Tutorial notebooks for generating particle initial conditions and for computing observables from a model.

   .. grid-item-card:: API reference
      :link: reference/index
      :link-type: doc

      Every function by subpackage, the numerical settings, the code and paper notations, the accuracy tables and the
      technical notes.

.. toctree::
   :hidden:

   start/index
   guide/index
   notebooks/index
   reference/index
   project/index
   citation/index
