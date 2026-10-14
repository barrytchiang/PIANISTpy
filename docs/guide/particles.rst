Particles
=========

:class:`pianistpy.Sampler`, :class:`pianistpy.particles.Particles`

``Sampler(model)`` prepares the sampling tables of a model once; ``.sample(n, seed)`` then returns ``n``
equal-mass particles as a ``Particles`` result, with positions, velocities, masses and component labels in
memory. The draw is by inversion, with no rejection step, and is reproducible particle by particle: each
particle is determined by the seed, its component and its index, independently of the hardware and of the
number of particles requested.

.. code-block:: python

   sampler = pp.Sampler(model)
   particles = sampler.sample(10_000_000, seed=1)
   particles.write("halo.h5")                     # the native HDF5 layout
   particles.write("halo_gamer.bin", layout="gamer")
   particles.write("halo_gadget.hdf5", layout="gadget")
   particles.write("halo.txt", layout="text")

The layouts: the native HDF5 file, read back by :func:`pianistpy.read`; GAMER's particle file; a Gadget HDF5
initial condition; fixed-width text with the units in its header. Every writer reads its file back and compares
it with what it wrote before returning. A writer for another simulation code is one registered function.

A model that is not spatially finite needs an explicit sampling radius; the result records the mass inside it
and the fraction of the total it represents.

Where the speed table reaches its tier
--------------------------------------

A model whose distribution function is non-negative is not thereby samplable at every tier. The speed table's
rows are refined until the error across them is under the goal or their count reaches ``n_rows_max``; the table's
report carries that error, and a table beyond the hard limit is refused (:doc:`accuracy`). Close to the smallest
anisotropy radius ``r_a_min`` at which an Osipkov-Merritt or Cuddeford Type I distribution function is
non-negative, the rows reach ``n_rows_max`` first and the error grows steeply as ``r_a`` falls toward
``r_a_min``. Measured on a truncated NFW density of concentration 10 (truncation radius 31.9 scale radii) at the
default tier, the error crosses the ceiling and the hard limit at:

==============  ==========  =============================  ================================
model           ``beta_c``  ceiling 1e-4 at r_a/r_a_min    hard limit 1e-2 at r_a/r_a_min
==============  ==========  =============================  ================================
Cuddeford I     -0.75       1.62                           1.16
Cuddeford I     -0.50       1.84                           1.20
Cuddeford I     -0.25       1.38                           1.10
OM I            0           1.27                           1.07
Cuddeford I     +0.05       1.24                           1.07
Cuddeford I     +0.25       1.14                           1.04
Cuddeford I     +0.40       1.09                           1.02
Cuddeford I     +0.45       above 1.23                     1.02
==============  ==========  =============================  ================================

Above the first ratio the table is within the ceiling; between the two it is returned with a warning; below the
second it is refused. At ``beta_c = 1/2`` the distribution function is non-negative at every ``r_a``, and the
table is beyond the hard limit at every ``r_a`` up to 0.32 scale radii. A constant-anisotropy model is within the
ceiling up to ``beta_c = 0.4`` (row error 6e-5) and above it at ``beta_c = 0.45`` (4e-4). These bounds are
measured, not imposed: each table's own report is the verdict.
