# Particles

Source: https://pianistpy.com/docs/guide/particles.html

# Particles

pianistpy.Sampler, pianistpy.particles.Particles

Sampler(model) prepares the sampling tables of a model once; .sample(n, seed) then returns n
equal-mass particles as a Particles result, with positions, velocities, masses and component labels in
memory. The draw is by inversion, with no rejection step, and is reproducible particle by particle: each
particle is determined by the seed, its component and its index, independently of the hardware and of the
number of particles requested.

```
sampler = pp.Sampler(model)
particles = sampler.sample(10_000_000, seed=1)
particles.write("halo.h5")                     # the native HDF5 layout
particles.write("halo_gamer.bin", layout="gamer")
particles.write("halo_gadget.hdf5", layout="gadget")
particles.write("halo.txt", layout="text")

```

The layouts: the native HDF5 file, read back by pianistpy.read(); the particle file of
GAMER; a Gadget HDF5
initial condition; fixed-width text with the units in its header. Every writer reads its file back and compares
it with what it wrote before returning. A writer for another simulation code is one registered function.

A model that is not spatially finite needs an explicit sampling radius; the result records the mass inside it
and the fraction of the total it represents.

Sampler(model, r_range=(r_lo, r_hi)) draws every particle between two radii, in internal units, inside the
sampling. The draw is the model’s own conditioned on the range, not a selection after a full draw: each
particle’s radius number is taken between the mass fractions of r_lo and r_hi, so every particle lands in
the range, and the particles carry the range’s mass. A superposition is drawn in the range by its components’
weights, which the range leaves unchanged.

Sampler.speeds(r, seed, v_range=(v_lo, v_hi)) draws one speed at each given radius from the model’s speed
distribution at that radius conditioned on the window, in internal units, and returns with the speeds the fraction
of the distribution at each radius that the window holds. v_lo and v_hi are floats, one window for every
radius, or arrays that broadcast against r, a window at each radius (a grid in units of each radius’s escape
speed is one call per grid speed). Each speed number is taken between the speed table’s own
numbers for v_lo and v_hi, so every speed lands in the window; a window wider than the escape speed draws the
speeds of sample at the same seed. A superposition takes each component by its share of the window. The window is
drawn for a constant anisotropy only: with a finite anisotropy radius the speed carries a factor of the drawn angle,
so the window couples to the angle, and such a model is refused.

## Where the speed table reaches its tier

A model whose distribution function is non-negative is not thereby samplable at every tier. The speed table’s
rows are refined until the error across them is under the goal or their count reaches n_rows_max; the table’s
report carries that error, and a table beyond the hard limit is refused (Accuracy). Close to the smallest
anisotropy radius r_a_min at which an Osipkov-Merritt or Cuddeford Type I distribution function is
non-negative, the rows reach n_rows_max first and the error grows steeply as r_a falls toward
r_a_min. Measured on a truncated NFW density of concentration 10 (truncation radius 31.9 scale radii) at the
default tier, the error crosses the ceiling and the hard limit at:

model

beta_c

ceiling 1e-4 at r_a/r_a_min

hard limit 1e-2 at r_a/r_a_min

Cuddeford I

-0.75

1.62

1.16

Cuddeford I

-0.50

1.84

1.20

Cuddeford I

-0.25

1.38

1.10

OM I

0

1.27

1.07

Cuddeford I

+0.05

1.24

1.07

Cuddeford I

+0.25

1.14

1.04

Cuddeford I

+0.40

1.09

1.02

Cuddeford I

+0.45

above 1.23

1.02

Above the first ratio the table is within the ceiling; between the two it is returned with a warning; below the
second it is refused. At beta_c = 1/2 the distribution function is non-negative at every r_a, and the
table is beyond the hard limit at every r_a up to 0.32 scale radii. A constant-anisotropy model is within the
ceiling up to beta_c = 0.4 (row error 6e-5) and above it at beta_c = 0.45 (4e-4). These bounds are
measured, not imposed: each table’s own report is the verdict.

     On this page
