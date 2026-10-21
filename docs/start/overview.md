# What PIANISTpy does

Source: https://pianistpy.com/docs/start/overview

- [Density profiles.](https://pianistpy.com/docs/guide/profiles) Analytical double-power-law density profiles with a finite central
potential, including the Navarro-Frenk-White, Hernquist and Plummer profiles.

- [Velocity anisotropy.](https://pianistpy.com/docs/guide/anisotropy) Isotropic, constant-anisotropy, Osipkov-Merritt and
Cuddeford-Osipkov-Merritt (COM) models. The Osipkov-Merritt and COM models are radially (Type I) or tangentially
(Type II) biased.

- [Distribution functions.](https://pianistpy.com/docs/guide/df) Each anisotropy model’s distribution function, inverted from the
input density with exact symbolic expressions up to the final numerical integration, with extensively validated
numerical accuracy ([Accuracy tables](https://pianistpy.com/docs/reference/accuracy)) and checks that the distribution function is
non-negative (physical) and monotonic (a sufficient condition for dynamical stability).

- [Truncation in phase space.](https://pianistpy.com/docs/guide/truncation) Model truncation performed in phase space. The
truncation boundary is self-consistent and in dynamical equilibrium. Method I preserves the model’s anisotropy
profile, and Method II gives different anisotropy models an identical truncated density.

- [Particles.](https://pianistpy.com/docs/guide/particles) Particle initial conditions drawn from the truncated distribution
function, with a finite, dynamically stable truncation boundary. The particles are given in internal, model or
physical units.

- [Observables.](https://pianistpy.com/docs/guide/observables) Radial, tangential and line-of-sight velocity dispersions,
line-of-sight velocity distributions and their Gauss-Hermite moments, and the local velocity distribution at a 3D
radius. Projected quantities are averaged over centered or offset circular and rectangular apertures.

- [Superposition.](https://pianistpy.com/docs/guide/superposition#guide-superposition) Distribution functions of different anisotropy models on one
truncated density, linearly superposed into one equilibrium model. The superposed model’s anisotropy profile can
differ from every analytical family, for example an anisotropy profile fitted to simulated halos.

- [Inference.](https://pianistpy.com/docs/guide/superposition#guide-inference) End-to-end differentiable in JAX: the likelihood of a model against
kinematic data, with gradients in the superposition weights, the mass and length scales (*M*<sub>0</sub> and
*r*<sub>0</sub>) and, for Method II components, the anisotropy parameters, for gradient-based Monte Carlo inference.
