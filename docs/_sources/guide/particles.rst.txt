Particles
=========

:class:`pianistpy.Sampler`, :class:`pianistpy.particles.Particles`

``Sampler(model)`` builds the sampling tables of a model once. ``sampler.sample(n, seed)`` then returns ``n``
equal-mass particles in memory, as a ``Particles`` object that holds positions, velocities, masses and component
labels in the model's unit system. The draw is reproducible particle by particle. Each particle depends only on the
seed, the particle's component and the particle's index, not on the hardware or on ``n``. One call of ``sample``
returns at most :math:`2^{31} - 1` particles. A larger number of particles is drawn in several calls with distinct
seeds.

.. code-block:: python

   sampler = pp.Sampler(model)
   particles = sampler.sample(10_000_000, seed=1)
   particles.write("halo.h5")                     # the native HDF5 layout
   particles.write("halo_gamer.bin", layout="gamer")
   particles.write("halo_gadget.hdf5", layout="gadget")
   particles.write("halo.txt", layout="text")

The layouts are:

- ``"native"`` (the default): the package's own HDF5 file, read back by :func:`pianistpy.read`;
- ``"gamer"``: the particle file of `GAMER <https://doi.org/10.1093/mnras/sty2586>`_;
- ``"gadget"``: an HDF5 initial condition in the Gadget snapshot format 3;
- ``"text"``: fixed-width text, with the units in its header.

Every writer reads the file back before returning, and raises :class:`~pianistpy.ResultError` if the file differs
from the particles written. A layout for another simulation code is added by one function decorated with
``pianistpy.particles.register_particle_writer``.

A model that is not spatially finite needs an explicit sampling radius, ``Sampler(model, r_sample=...)``, in
internal units. The particles then carry the mass inside that radius (``Sampler.radius_table.mass_sample``, in
internal units).

Particles in a range of radii
-----------------------------

``Sampler(model, r_range=(r_lo, r_hi))`` draws every particle between two radii, in internal units, with
:math:`0 \le r_{\rm lo} < r_{\rm hi}` and :math:`r_{\rm hi}` at most the outer radius of the sampling. Each
particle's radius is drawn from the model's enclosed mass restricted to the range, not selected from a full draw.
Every particle lies between :math:`r_{\rm lo}` and :math:`r_{\rm hi}`, and the particles together carry the model's
mass between :math:`r_{\rm lo}` and :math:`r_{\rm hi}`. For a superposition, each component's share of the particles
between :math:`r_{\rm lo}` and :math:`r_{\rm hi}` is the component's weight, because the components share one
density.

Constrained velocity draws
--------------------------

``Sampler.velocities(r, seed, v_range=(v_lo, v_hi))`` draws one velocity at each given radius, in internal units, from the
model's velocity distribution at that radius conditioned on the range. ``Sampler.velocities`` returns the velocities and, at
each radius, the fraction of the distribution inside the range. Where the range holds none of the distribution, the velocity is NaN.

- ``v_lo`` and ``v_hi`` are floats, for one range at every radius, or arrays that broadcast against ``r``, for a
  range at each radius. For example, a range from 0.5 to 0.6 times the escape velocity at each radius is one call,
  with ``v_lo`` and ``v_hi`` given as arrays over ``r``. A grid of such ranges in units of the escape velocity takes
  one call per range.
- Every velocity lies in the range. A range from 0 to at least the escape velocity at every radius gives the velocities that
  ``sample`` draws from the same random numbers at the same seed.
- For a superposition, the component of each velocity is drawn with a probability proportional to the component's
  weight times the component's fraction of velocities in the range.
- ``Sampler.velocities`` takes isotropic and constant-anisotropy models only. A model with a finite anisotropy radius
  raises :class:`~pianistpy.InputError`, because with a finite anisotropy radius the velocity depends on the drawn
  velocity angle.

Sampling accuracy near the smallest anisotropy radius
-----------------------------------------------------

A model whose distribution function is non-negative cannot always be sampled to :math:`10^{-6}`. The velocity table
of a ``Sampler`` is refined until the table's error is below the setting ``sampling["row_error_goal"]``
(:math:`10^{-6}` with the default settings, :math:`10^{-4}` with the quick settings) or until the table holds
``sampling["n_rows_max"]`` potential values. The validation report ``Sampler.validation`` records the velocity table's
error, and a velocity-table error above :math:`10^{-2}` raises :class:`~pianistpy.AccuracyError` (:doc:`accuracy`).

Let :math:`R_{\rm a,min}` be the smallest anisotropy radius at which an Osipkov-Merritt (OM) or
Cuddeford-Osipkov-Merritt (COM) Type I distribution function is non-negative. Close to :math:`R_{\rm a,min}`, the
velocity table reaches ``sampling["n_rows_max"]`` potential values before the table's error falls below
``sampling["row_error_goal"]``, and the error grows steeply as :math:`R_{\rm a}` falls toward
:math:`R_{\rm a,min}`. For a truncated NFW density of concentration 10 (truncation radius 31.9 scale radii) with the
default settings, the velocity table's error crosses :math:`10^{-4}` and :math:`10^{-2}` at the following ratios
:math:`R_{\rm a}/R_{\rm a,min}`:

===========  =====================  ==========================================================  =============================================================
model        :math:`\beta_{\rm c}`  :math:`10^{-4}` at :math:`R_{\rm a}/R_{\rm a,min}`          :math:`10^{-2}` at :math:`R_{\rm a}/R_{\rm a,min}`
===========  =====================  ==========================================================  =============================================================
COM I        -0.75                  1.62                                                        1.16
COM I        -0.50                  1.84                                                        1.20
COM I        -0.25                  1.38                                                        1.10
OM I         0                      1.27                                                        1.07
COM I        +0.05                  1.24                                                        1.07
COM I        +0.25                  1.14                                                        1.04
COM I        +0.40                  1.09                                                        1.02
COM I        +0.45                  above 1.23                                                  1.02
===========  =====================  ==========================================================  =============================================================

Above the ratio for :math:`10^{-4}`, the velocity table's error is below :math:`10^{-4}`. Between the ratio for
:math:`10^{-4}` and the ratio for :math:`10^{-2}`, ``Sampler`` issues an :class:`~pianistpy.AccuracyWarning`. Below
the ratio for :math:`10^{-2}`, ``Sampler`` raises :class:`~pianistpy.AccuracyError`. At :math:`\beta_{\rm c} = 1/2`
the distribution function is non-negative at every :math:`R_{\rm a}`, but the velocity table's error is above
:math:`10^{-2}` at every :math:`R_{\rm a}` up to 0.32 scale radii. The velocity table of a constant-anisotropy model has an error below
:math:`10^{-4}` up to :math:`\beta_{\rm c} = 0.4` :math:`(6 \times 10^{-5})` and above :math:`10^{-4}` at
:math:`\beta_{\rm c} = 0.45` :math:`(4 \times 10^{-4})`. The package sets no limit on :math:`R_{\rm a}` or
:math:`\beta_{\rm c}` for sampling. The tier of each ``Sampler``'s validation report sets whether the ``Sampler``
issues an :class:`~pianistpy.AccuracyWarning` or raises :class:`~pianistpy.AccuracyError`.
