What PIANISTpy does
===================

.. The landing page shows this list as written here: docs/tools/assemble_site.py copies it from the built page,
   links and all. Edit the list here only.

.. rst-class:: features

- :doc:`Density profiles. <../guide/profiles>` Analytical double-power-law density profiles with a finite central
  potential, including the Navarro-Frenk-White, Hernquist and Plummer profiles.
- :doc:`Velocity anisotropy. <../guide/anisotropy>` Isotropic, constant-anisotropy, Osipkov-Merritt and
  Cuddeford-Osipkov-Merritt (COM) models. The Osipkov-Merritt and COM models are radially (Type I) or tangentially
  (Type II) biased.
- :doc:`Distribution functions. <../guide/df>` Each anisotropy model's distribution function, inverted from the
  input density with exact symbolic expressions up to the final numerical integration, with extensively validated
  numerical accuracy (:doc:`Accuracy tables <../reference/accuracy>`) and checks that the distribution function is
  non-negative (physical) and monotonic (a sufficient condition for dynamical stability).
- :doc:`Truncation in phase space. <../guide/truncation>` Model truncation performed in phase space. The
  truncation boundary is self-consistent and in dynamical equilibrium. Method I preserves the model's anisotropy
  profile, and Method II gives different anisotropy models an identical truncated density.
- :doc:`Particles. <../guide/particles>` Particle initial conditions drawn from the truncated distribution
  function, with a finite, dynamically stable truncation boundary. The particles are given in internal, model or
  physical units.
- :doc:`Observables. <../guide/observables>` Radial, tangential and line-of-sight velocity dispersions,
  line-of-sight velocity distributions and their Gauss-Hermite moments, and the local velocity distribution at a 3D
  radius. Projected quantities are averaged over centered or offset circular and rectangular apertures.
- :ref:`Superposition. <guide-superposition>` Distribution functions of different anisotropy models on one
  truncated density, linearly superposed into one equilibrium model. The superposed model's anisotropy profile can
  differ from every analytical family, for example an anisotropy profile fitted to simulated halos.
- :ref:`Inference. <guide-inference>` End-to-end differentiable in JAX: the likelihood of a model against
  kinematic data, with gradients in the superposition weights, the mass and length scales (*M*\ :sub:`0` and
  *r*\ :sub:`0`) and, for Method II components, the anisotropy parameters, for gradient-based Monte Carlo inference.
