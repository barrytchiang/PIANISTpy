What PIANISTpy does
===================

.. The landing page shows this list as written here: docs/tools/assemble_site.py copies it from the built page,
   links and all. Edit the list here only.

.. rst-class:: features

- :doc:`Density profiles. <../guide/profiles>` Analytical double-power-law density profiles with a finite central
  potential, including the commonly used Navarro-Frenk-White, Hernquist and Plummer profiles.
- :doc:`Velocity anisotropy. <../guide/anisotropy>` Isotropic, constant-anisotropy, Osipkov-Merritt and
  Cuddeford-Osipkov-Merritt models, the latter two radially (Type I) or tangentially (Type II) biased.
- :doc:`Distribution functions. <../guide/df>` Each anisotropy model's distribution function, inverted from the
  input density with exact symbolic expressions up to the final numerical integration, with
  :doc:`extensively validated numerical accuracy <../reference/accuracy>` and checks that it is non-negative
  (physical) and monotonic (dynamically stable).
- :doc:`Truncation in phase space. <../guide/truncation>` Model truncation performed in phase space, such that the
  truncation boundary is fully self-consistent and in dynamical equilibrium, by two methods: one preserves the
  model's anisotropy profile (Method I), the other gives different anisotropy models an identical truncated
  density (Method II).
- :doc:`Particles. <../guide/particles>` Particle initial conditions with a finite, dynamically stable truncation
  boundary, drawn from the truncated distribution function, in internal, model or physical units.
- :doc:`Observables. <../guide/observables>` Radial, tangential and line-of-sight velocity dispersions,
  line-of-sight velocity distributions and their Gauss-Hermite moments, and the local 3D velocity distribution at
  any radius; projected quantities averaged over centered or offset circular and rectangular apertures.
- :ref:`Superposition. <guide-superposition>` Distribution functions of different anisotropy models on one
  truncated density, linearly superposed into one equilibrium model whose anisotropy profile lies beyond the
  analytical families, such as one fitted to simulated halos.
- :ref:`Inference. <guide-inference>` End-to-end differentiable in JAX: the likelihood of a model against
  kinematic data, and every observable, with gradients in the superposition weights, the unit scales and the
  anisotropy parameters, for gradient-based Monte Carlo inference.
