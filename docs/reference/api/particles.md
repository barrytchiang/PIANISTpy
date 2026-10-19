# Particles (pianistpy.particles)

Source: https://pianistpy.com/docs/reference/api/particles.html

# Particles (pianistpy.particles)

Particle sampling: the prepared sampler and the particle result.

class pianistpy.particles.Particles(position, velocity, mass, component, component_names, unit_system, units, identity, parent)

A particle realization of a model.

Parameters:

position (array, shape (n, 3)) – Cartesian positions.

velocity (array, shape (n, 3)) – Cartesian velocities.

mass (array, shape (n,)) – Particle masses.

component (array of int, shape (n,)) – Index, into component_names, of the component each particle
belongs to.

component_names (tuple of str) – Names of the components of the model.

unit_system ({"internal", "model", "physical"}) – The unit system of the arrays. Internal by default everywhere.

units (dict of str to str) – Unit of the position, the velocity and the mass, as text.

identity (str) – Identity string of the realization.

parent (str) – Identity string of the sampler or model the particles were drawn from.

Notes

The arrays are read-only. Two realizations are compared by their
identity strings.

write(path, layout='native', **options)

Write the particles to a file; see write_particles.

Parameters:

path (str | PathLike)

layout (str)

options (Any)

Return type:

None

class pianistpy.particles.Sampler(model, r_sample=None, r_range=None)

Everything needed to draw particles from a model, built once.

The sampler holds the tables that sampling needs. They are built when the
sampler is created and reused by every call of sample, so a user keeps
one object however many tables there are.

Parameters:

model (Model) – The model to draw from.

r_sample (float, optional) – Outer radius of the sampling, in internal units: the radius table
depends on it. Required for a model that is not spatially finite;
the truncation radius of one that is.

r_range (tuple of float, optional) – Radii (r_lo, r_hi), in internal units, between which every particle
is drawn: 0 <= r_lo < r_hi <= the outer radius of the sampling. The
draw is the model’s own conditioned on the range, not a selection:
each particle’s radius number is taken in the mass fractions of the
range (stages.radius_bounds), and the particles carry the range’s
mass. Its other numbers are those of the draw without the range.

Variables:

validation (ValidationReport) – The checks of the tables (those of SpeedTable.validation) and the
worst tier of the model. A sampler above the goal says so in a
warning; one beyond the hard limit is refused.

Raises:

InputError – If model is not a Model, or has more components than this release
    samples, or r_range is not a range of radii inside the sampling.

AccuracyError – If the model’s report reads "refused" and its settings do not ask
    for results from it anyway (Settings, write_anyway).

ResultError – If a speed table lies beyond the hard limit and the settings do not ask
    for it anyway.

Notes

A physical model is not thereby samplable at every tier. The speed
table’s rows are refined until the error across them is under the goal
or their count reaches n_rows_max; close to the smallest anisotropy
radius at which an Osipkov-Merritt or Cuddeford Type I DF is non-negative,
the rows reach n_rows_max first and the error grows steeply as r_a
falls toward that radius. Measured on a truncated NFW density of
concentration 10 at the default tier, the error crosses the hard limit at
1.02 to 1.2 times that radius and the ceiling at 1.09 to 1.84 times it
(up to beta_c = 0.4), the widest for the most negative beta_c; at
beta_c = 1/2, whose DF is non-negative at every r_a, the table is beyond
the hard limit at every r_a up to 0.32. A constant-anisotropy model is
within the ceiling up to beta_c = 0.4. The bound is measured, not
imposed: the report of each table is the verdict (the user guide,
Particles, gives the table per family).

sample(n, seed=0)

Draw particles from the model.

Parameters:

n (int) – Number of particles.

seed (int, optional) – Seed of the random streams.

Returns:

particles – The particles, in the model’s units (Model.units). Every draw is
made in internal units and converted at the output only.

Return type:

Particles

Raises:

InputError – If n is not a positive integer below 2^31.

ResultError – If a particle lies beyond the phase-space bound of its component.

speeds(r, seed=0, v_range=None)

Draw one speed at each given radius, within a window of speeds.

Parameters:

r (array-like, shape (n,)) – Radii, in internal units, inside the sampling: 0 < r <= its outer
radius.

seed (int, optional) – Seed of the random streams. The speed at the i-th radius takes the
speed number of particle i of sample at the same seed, mapped into
the window’s numbers (and, for a superposition, its component by
the window’s share of each component).

v_range (tuple of float or of array-like, optional) – The speeds (v_lo, v_hi), in internal units, 0 <= v_lo < v_hi. Each
bound is a float, the same window at every radius, or an array that
broadcasts against r, a window at each radius (as a grid in units
of each radius’s escape speed needs). Without it, the whole speed
distribution.

Returns:

v (array, shape (n,)) – The speed at each radius, drawn from the model’s speed distribution
at that radius conditioned on the window, in internal units; NaN at
a radius where the window holds none of the distribution.

fraction (array, shape (n,)) – The fraction of the speed distribution at each radius in the window
(the weighted sum of the components’ fractions for a
superposition).

Raises:

InputError – For a model with a finite anisotropy radius (Osipkov-Merritt or
    Cuddeford): there the speed is the scaled speed times a factor of
    the drawn angle, so the window couples to the angle, and no table
    holds the speed’s distribution at a radius with the angle
    integrated out. Also for radii outside the sampling, an invalid
    window, or bounds that do not broadcast against r.

Return type:

tuple[Any, Any]

pianistpy.particles.component_view(model, k)

Return component k of a superposed model as a model of one component.

Parameters:

model (Model) – A superposed model (is_superposition).

k (int) – Index of the component.

Returns:

view – The component on its own: its table on its own grid, its forms, the
shared truncated profile, and mass 1. The sampler and the observables
read each component of a superposition through it.

Return type:

Model

pianistpy.particles.is_superposition(model)

Return whether a model is a superposition (superpose).

Parameters:

model (Model) – The model.

Returns:

superposed – Whether every component is built on one truncated profile, as
superpose builds them.

Return type:

bool

pianistpy.particles.register_particle_writer(layout)

Make a function the writer of particles in a given file layout.

A writer takes the particles, the path and its own keyword options. The
layouts of simulation codes are added this way, without a change to the
sampling or to Particles.

Parameters:

layout (str) – Name of the layout, as passed to write_particles.

Returns:

decorator – Registers the decorated function and returns it unchanged.

Return type:

callable

pianistpy.particles.write_particles(particles, path, layout='native', **options)

Write particles to a file in the requested layout.

Parameters:

particles (Particles) – The particles to write.

path (path) – File to write.

layout (str, optional) – Name of a registered layout. "native" is the package’s own HDF5
result file, which _read_particles reads back.

**options – Options of the layout’s writer.

Return type:

None

     On this page
