# Particles

Source: https://pianistpy.com/docs/reference/api/particles

Module `pianistpy.particles`

Particle sampling: the sampler, the particle result and the particle writers.

A user builds a Sampler from a model once, then calls Sampler.sample for
a Particles object and Particles.write to write the particles to a file
in one of the layouts of `PARTICLE_WRITERS`.

Every particle is drawn by inversion, with no rejection step. The radius
\(r\) of a particle solves \(M(r)/M(r_{\rm s}) = u_r\), where
\(M(r)\) is the mass enclosed within \(r\), \(r_{\rm s}\) is the
outer radius of the sampling and \(u_r\) is a uniform random number
between 0 and 1. The velocity of a particle is drawn from the model’s
distribution of velocity at the potential \(\Psi(r)\) of the particle’s
radius. The derivation is in Notes Section T:5.

class pianistpy.particles.Particles(*position*, *velocity*, *mass*, *component*, *component_names*, *unit_system*, *units*, *identity*, *parent*)

A particle realization of a model: positions, velocities, masses and
component labels.

Sampler.sample returns a Particles object. Particles.write writes the
particles to a file, and read reads a file of the native layout back
into a Particles object. Each parameter below is kept as an attribute
with the same spelling as the parameter, and the four arrays are
read-only.

Parameters:

- **position** (*array-like**, **shape** (**n**, **3**)*) – Cartesian positions of the n particles, in the unit that
`units["position"]` gives.

- **velocity** (*array-like**, **shape** (**n**, **3**)*) – Cartesian velocities of the n particles, in the unit that
`units["velocity"]` gives.

- **mass** (*array-like**, **shape** (**n**,**)*) – Masses of the n particles, in the unit that `units["mass"]` gives.
The length of mass sets the number of particles n.

- **component** (*array-like** of *[*int*](https://docs.python.org/3/builtins/functions.html#int)*, **shape** (**n**,**)*) – Index into component_names of the component each particle belongs
to, from 0 to the number of names minus one.

- **component_names** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Names of the model’s components.

- **unit_system** (*{"internal"**, **"model"**, **"physical"}*) – Unit system of position, velocity and mass. Sampler.sample
writes the particles in the unit system of the model that the
sampler draws from (Model.to).

- **units** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to str*) – Units of the positions, the velocities and the masses as text, under
the keys `"position"`, `"velocity"` and `"mass"`.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the realization. Sampler.sample derives the
identity string from the sampler’s identity string, the number of
particles and the seed.

- **parent** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the sampler that the particles were drawn from.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If position or velocity does not have the shape (n, 3), with n
    the length of mass.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component does not have the shape (n,), or holds an index that
    is negative or not below the number of component_names.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If unit_system is not `"internal"`, `"model"` or
    `"physical"`.

See also

[`Sampler.sample`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Sampler.sample)
Draw a Particles object from a model.

[`write_particles`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.write_particles)
Write particles to a file in a registered layout.

`read`
Read a file of the native layout back into a Particles object.

Notes

Two realizations hold identical particles when the two identity strings
are equal.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> particles = pp.Particles(
...     position=np.zeros((2, 3)),
...     velocity=np.ones((2, 3)),
...     mass=np.full(2, 0.5),
...     component=np.zeros(2, dtype=int),
...     component_names=("halo",),
...     unit_system="internal",
...     units={
...         "position": "internal", "velocity": "internal", "mass": "internal"
...     },
...     identity="a" * 64,
...     parent="b" * 64,
... )
>>> len(particles), particles.position.shape
(2, (2, 3))
```

write(*path*, *layout='native'*, ***options*)

Write the particles to a file in a registered layout.

`particles.write(path, layout, **options)` is the call
`write_particles(particles, path, layout, **options)`.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the file to write.

- **layout** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – Name of a layout registered in `PARTICLE_WRITERS`: `"native"`,
`"gamer"`, `"gadget"`, `"text"`, or a name added by
register_particle_writer. Default: `"native"`, which means
the package’s own HDF5 result file.

- ****options** – Keyword options of the layout’s writer. write_particles lists
the options of every layout with their defaults.

Returns:

The file is written, read back and compared with the particles
before the call returns.

Return type:

None

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If layout is not registered in `PARTICLE_WRITERS`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If an option’s value is outside the values the layout’s writer
    takes (write_particles lists the conditions).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the file exists and the option `overwrite` is not true.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the file read back differs from the particles written.

See also

[`write_particles`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.write_particles)
The function that Particles.write calls.

`read`
Read a file of the native layout back into a Particles object.

Examples

```python
>>> import os
>>> import tempfile
>>> import numpy as np
>>> import pianistpy as pp
>>> particles = pp.Particles(
...     position=np.zeros((2, 3)),
...     velocity=np.ones((2, 3)),
...     mass=np.full(2, 0.5),
...     component=np.zeros(2, dtype=int),
...     component_names=("halo",),
...     unit_system="internal",
...     units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
...     identity="a" * 64,
...     parent="b" * 64,
... )
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "halo.txt")
...     particles.write(path, layout="text")
...     with open(path) as file:
...         n_lines = len(file.readlines())
>>> n_lines
9
```

class pianistpy.particles.Sampler(*model*, *r_sample=None*, *r_range=None*)

The sampling tables of a model, built once, from which particles are drawn.

A Sampler builds the radius table, the velocity table and the angle table
of a model when the Sampler is created. Every call of Sampler.sample
and of Sampler.velocities reuses the three tables. For many draws from one
model, build one Sampler and call Sampler.sample on the one Sampler
for each draw.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The model to draw from: a model of one component, or a superposition
returned by superpose.

- **r_sample** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Outer radius of the sampling, in internal units whatever the model’s
unit system. r_sample must lie inside the radii of the model’s
profile. A model that is not spatially finite needs r_sample, and a
spatially finite model takes no r_sample. Default: `None`, which
means the truncation radius of a spatially finite model.

- **r_range** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Radii (r_lo, r_hi) between which every particle is drawn, in internal
units, with r_lo and r_hi finite, `0 <= r_lo < r_hi` and r_hi at
most the outer radius of the sampling. Each particle’s radius is
drawn from the model’s enclosed mass restricted to the range, so
every particle lies in the range and the particles carry the mass of
the model between r_lo and r_hi. Each particle’s velocity and
directions take the random numbers of the draw without r_range.
Default: `None`, which means radii from the center to the outer
radius of the sampling.

Variables:

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The report of the sampler: the velocity table’s checks `row_error` and
`polish_residual`, and the model’s own report as the parent
report. The report’s tier is the band of the largest relative error:
`"silent"` at or below \(10^{-6}\), `"recorded"` above
\(10^{-6}\) and at or below \(10^{-4}\), `"warned"` above
\(10^{-4}\) and at or below \(10^{-2}\), and `"refused"`
above \(10^{-2}\). A sampler whose tier is `"recorded"` writes
one line at level INFO to the package’s log. A sampler whose tier is
`"warned"` or `"refused"` also issues an AccuracyWarning.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the sampler, derived from the model’s identity
string, r_sample, r_range, the settings and the package version.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is not a Model.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model has more than one component and is not a superposition.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_sample is given for a spatially finite model, is missing for a
    model that is not spatially finite, or lies outside the radii of the
    model’s profile.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_range is not two finite radii with `0 <= r_lo < r_hi`, if r_hi
    is beyond the outer radius of the sampling, if r_hi is at or below the
    innermost radius of the radius table, or if r_lo is at or above the
    outermost radius of the radius table.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the model’s tier is `"refused"` and the model’s settings do not
    set `write_anyway=True` (Settings).

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If a check of the velocity table exceeds \(10^{-2}\) and the settings
    do not set `write_anyway=True`. With `write_anyway=True`, the
    Sampler is built with the tier `"refused"` and issues an
    AccuracyWarning.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the velocity table cannot resolve the distribution of velocity at a
    potential with the resolution that the settings allow.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the DF below the DF table’s lowest energy falls so steeply toward
    \(\mathcal{K} = 0\) that the mass at energies below the DF table’s
    lowest energy is not finite.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a sampling table fails the table’s own consistency check: an
    inverse that is not finite and increasing, or a cumulative
    distribution that is not monotone.

See also

[`Sampler.sample`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Sampler.sample)
Draw particles from the model.

[`Sampler.velocities`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Sampler.velocities)
Draw one velocity at each user-specified 3D radius.

[`Particles`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Particles)
The result of Sampler.sample.

`superpose`
Build a superposition, which a Sampler also takes.

Notes

A model with a non-negative DF is not necessarily sampled within
\(10^{-4}\). The velocity table doubles the velocity table’s resolution
until the velocity table’s own error is below the setting
`sampling["row_error_goal"]` (\(10^{-6}\) with the default
settings, \(10^{-4}\) with the quick settings) or until the
resolution reaches the setting `sampling["n_rows_max"]`. For an
Osipkov-Merritt or Cuddeford-Osipkov-Merritt (COM) Type I DF whose
anisotropy radius \(R_{\rm a}\) is close to the smallest anisotropy
radius at which the DF is non-negative, the resolution reaches
`sampling["n_rows_max"]` first, and the velocity table’s error grows as
\(R_{\rm a}\) decreases toward the smallest anisotropy radius with a
non-negative DF. The report in
validation gives the velocity table’s error for each model. The page on
particles in the user guide lists, for each anisotropy model, the
anisotropy radius at which the velocity table’s error exceeds
\(10^{-4}\) and \(10^{-2}\).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> sampler = pp.Sampler(model, r_sample=100.0)
>>> particles = sampler.sample(1000, seed=1)
>>> particles.position.shape, particles.unit_system
((1000, 3), 'internal')
```

sample(*n*, *seed=0*)

Return n equal-mass particles drawn from the model.

Each particle is determined by the seed, the particle’s component and
the particle’s index, and not by the number of particles requested.

Parameters:

- **n** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of particles, from 1 to \(2^{31} - 1\). A larger
number of particles is drawn in several calls with distinct seeds.

- **seed** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – Seed of the random numbers. One seed gives identical particles for
every value of the setting `sampling["draw_block"]`
(Settings). Default: `0`.

Returns:

**particles** – The n particles, in the unit system of the model (Model.to).
Every particle has the mass \(M_{\rm s}/n\), where
\(M_{\rm s}\) is the model’s mass inside the outer radius of
the sampling, or between r_lo and r_hi when the Sampler has an
r_range. The draw is made in internal units, and the positions,
velocities and masses are converted to the model’s unit system
at the output.

Return type:

[Particles](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Particles)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If n is not a positive integer.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If n is \(2^{31}\) or more.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If a particle lies outside the phase-space support of the
    particle’s component, or if the velocity rebuilt from the
    particle’s arrays misses the support’s bound by more than the
    rounding of the particle’s own variables.

See also

[`Sampler.velocities`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Sampler.velocities)
Draw one velocity at each user-specified 3D radius.

[`write_particles`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.write_particles)
Write the particles to a file.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> particles = pp.Sampler(model, r_sample=100.0).sample(1000, seed=1)
>>> len(particles), bool(np.all(particles.mass == particles.mass[0]))
(1000, True)
```

velocities(*r*, *seed=0*, *v_range=None*)

Return one velocity drawn at each user-specified 3D radius, within a
user-specified range of velocities.

Parameters:

- **r** (*array-like**, **shape** (**n**,**)*) – User-specified 3D radii, in internal units whatever the model’s
unit system. Every radius is finite, positive and at most the outer
radius of the sampling.

- **seed** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – Seed of the random numbers. The velocity at the i-th radius takes the
velocity’s random number of particle i of Sampler.sample at the
same seed, mapped into the user-specified range. For a
superposition, the component of the i-th velocity is chosen with
probabilities proportional to each component’s weight times the
component’s fraction of velocities in the range. Default: `0`.

- **v_range** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*float*](https://docs.python.org/3/builtins/functions.html#float)* or **of array-like**, **optional*) – The user-specified range of velocities (v_lo, v_hi), in internal
units, with v_lo finite, v_hi finite or infinite, and
`0 <= v_lo < v_hi`. Each bound is a float, which applies at
every radius, or an array that broadcasts against r, which
gives a bound at each radius (for example, a bound in units of
each radius’s escape velocity). Default: `None`, which means the
velocities from 0 to infinity, the whole distribution of velocity.

Returns:

- **v** (*array, shape (n,)*) – The velocity at each radius of r, in internal units, drawn from the
model’s distribution of velocity at the radius, restricted to the
user-specified range. The velocity is NaN at a radius where the
user-specified range holds no part of the distribution of velocity.

- **fraction** (*array, shape (n,)*) – The fraction of the distribution of velocity at each radius that
lies in the user-specified range. For a superposition, the
fraction is the sum over components of each component’s weight
times the component’s fraction.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a component has a finite anisotropy radius (Osipkov-Merritt
    or Cuddeford-Osipkov-Merritt). For such a component the velocity is
    the scaled velocity times a factor of the drawn angle, so the
    user-specified range of velocities depends on the angle, and no table
    holds the distribution of velocity at a radius with the angle
    integrated out.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r is not one-dimensional, or a radius in r is not finite,
    not positive or beyond the outer radius of the sampling.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If v_range is not two bounds, or if v_lo is not finite, v_hi is
    NaN or `0 <= v_lo < v_hi` fails at a radius.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the bounds of v_range do not broadcast against r.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

See also

[`Sampler.sample`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Sampler.sample)
Draw particles, positions and velocities included.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> sampler = pp.Sampler(model, r_sample=100.0)
>>> v, frac = sampler.velocities([0.5, 1.0, 2.0], seed=1, v_range=(0.1, 0.3))
>>> v.shape, bool(np.all((frac > 0.0) & (frac < 1.0)))
((3,), True)
```

pianistpy.particles.component_view(*model*, *k*)

Return component k of a superposed model as a model of one component.

Sampler and the observables read each component of a superposition
through component_view.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – A superposed model, for which is_superposition returns True.

- **k** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Index of the component, from 0 to the number of components of
model minus one.

Returns:

**view** – A model of one component: component k with mass 1, component k’s DF
table on component k’s own energy grid, and the truncated profile
that the components of model share. component_view raises no
package exception for a superposed model and a valid k.

Return type:

[Model](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)

See also

[`is_superposition`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.is_superposition)
Test whether a model is a superposition.

`superpose`
Build a superposition.

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.particles import component_view
>>> parent = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()), pp.Truncation(k_t=0.05)
... )
>>> children = [
...     pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=beta_c)))
...     for beta_c in (0.0, 0.2)
... ]
>>> composite = pp.superpose({children[0]: 0.5, children[1]: 0.5})
>>> view = component_view(composite, 1)
>>> len(composite.components), len(view.components)
(2, 1)
```

pianistpy.particles.is_superposition(*model*)

Return whether every component of a model shares one truncated profile.

A model returned by superpose shares one truncated profile across the
model’s components. A model of one component on a truncated profile also
passes the test; Sampler treats a model as a superposition only when
the model has more than one component.

Parameters:

**model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The model to test.

Returns:

**superposed** – True if the first component’s profile is a truncated profile and
every other component’s profile equals the first component’s
profile; False otherwise. is_superposition raises no package
exception.

Return type:

[bool](https://docs.python.org/3/builtins/functions.html#bool)

See also

`superpose`
Build a superposition from models on one truncated profile.

[`component_view`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.component_view)
One component of a superposition as a model.

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.particles import is_superposition
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> is_superposition(model)
False
```

pianistpy.particles.register_particle_writer(*layout*)

Return a decorator that registers a function as the writer of a layout.

The decorated function becomes the writer that write_particles and
Particles.write call for the name layout. A writer takes the
particles, the path of the file and the writer’s own keyword options. A
file layout of another simulation code is added by one registered
function, with no change to Sampler or to Particles.

Parameters:

**layout** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the layout, as passed to write_particles.

Returns:

**decorator** – A decorator that adds the decorated function to `PARTICLE_WRITERS`
under the name layout and returns the function unchanged.

Return type:

callable

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If `PARTICLE_WRITERS` already holds a writer under the name
    layout. The decorator raises the error when the decorator is
    applied to a function.

See also

[`write_particles`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.write_particles)
Write particles in a registered layout.

Examples

```python
>>> from pianistpy.particles import PARTICLE_WRITERS, register_particle_writer
>>> @register_particle_writer("count_only")
... def write_count(particles, path, overwrite=False):
...     with open(path, "w") as file:
...         file.write(str(len(particles)))
>>> "count_only" in PARTICLE_WRITERS
True
>>> del PARTICLE_WRITERS["count_only"]
```

pianistpy.particles.write_particles(*particles*, *path*, *layout='native'*, ***options*)

Write particles to a file in a registered layout.

Every writer reads the file back after writing the file and compares the
values read with the particles. The comparison is bit for bit after the
cast to the file’s number type, and to the 14 digits written after the
decimal point for the `"text"` layout.

Parameters:

- **particles** ([*Particles*](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Particles)) – The particles to write.

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the file to write.

- **layout** (*{"native"**, **"gamer"**, **"gadget"**, **"text"}** or *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – Name of a layout registered in `PARTICLE_WRITERS`.

- `"native"`: the package’s own HDF5 result file, which read
reads back into a Particles object.

- `"gamer"`: the particle file that GAMER reads at its
initialization from file (`PAR_INIT 3`), a stream of the
particle attributes with no header.

- `"gadget"`: a Gadget-derivative HDF5 initial condition in
snapshot format 3, with the groups and attributes of the
OpenGadget3 reference example.

- `"text"`: fixed-width text with a header that states the unit
of every column.

- any other name: a layout added by register_particle_writer.

Default: `"native"`, which means the package’s own HDF5 result
file.

- ****options** – Keyword options of the layout’s writer. Every option is optional.

- Every layout takes `overwrite` (bool). Default: `False`, which
means an existing file at path is not replaced.

- `"gamer"` takes `order` (`"ATT_ID"` or `"ID_ATT"`): one
block per attribute (every particle’s mass, then every particle’s
x, and so on) for `"ATT_ID"`, or one record per particle for
`"ID_ATT"`, the two values that GAMER’s parameter
`PAR_IC_FORMAT` takes. Default: `"ATT_ID"`.

- `"gamer"` takes `double_precision` (bool): float64 values for a
GAMER build with `PAR_IC_FLOAT8`, float32 values otherwise.
Default: `True`.

- `"gamer"` takes `include_mass` (bool): false for a GAMER run
that sets the particle mass itself (`PAR_IC_MASS` at or above
zero). Default: `True`.

- `"gamer"` takes `offset` (array-like, shape (3,)): a vector
added to every position, because GAMER’s simulation box has its
corner at the origin while the sampler centers the model at the
origin. Default: `None`, which means no shift.

- `"gamer"` takes `attributes` (`"standard"` or `"legacy"`):
`"standard"` writes the mass, the position and the velocity.
`"legacy"` writes 13 float32 blocks, one per attribute, in the
layout of the GAMER build used for the runs of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing): the
mass, the position, the velocity, the type 1, the tag 1, the
energy 0, the angular momentum 0, the energy backup 0 and the
identifiers 0 to \(n - 1\). Default: `"standard"`.

- `"gadget"` takes `double_precision` (bool): float64 arrays and
the header flag `Flag_DoublePrecision` set to 1, or float32
arrays and the flag set to 0. Default: `False`.

- `"gadget"` takes `part_type` (int): the particle type slot,
from 0 to 5. Default: `1`.

- `"gadget"` takes `hubble_param` (float): the value written to
the header’s `HubbleParam`, which an isolated model does not
use. Default: `0.7`.

- `"gadget"` takes `offset` (array-like, shape (3,)): a vector
added to every position. Default: `None`, which means no shift.

- `"native"` and `"text"` take `overwrite` only.

The `"gadget"` writer puts equal particle masses into the header’s
mass table and writes unequal masses per particle. The `"text"`
header states the unit system, the unit of every column, the
identity strings and the column names. Each line of the `"text"`
file holds the mass, the position and the velocity of one particle,
followed by the particle’s component index when the particles have
more than one component.

Returns:

The file is written, read back and compared with the particles
before the call returns.

Return type:

None

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If layout is not registered in `PARTICLE_WRITERS`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the file exists and the option `overwrite` is not true.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – For `"gamer"`, if `order` is not `"ATT_ID"` or `"ID_ATT"`,
    or `attributes` is not `"standard"` or `"legacy"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – For `"gamer"` or `"gadget"`, if `offset` does not have the
    shape (3,).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – For `"gamer"` with `attributes="legacy"`, if `order`,
    `double_precision` or `include_mass` differs from its default.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – For `"gamer"` with `attributes="legacy"`, if the number of
    particles exceeds \(2^{24}\), the largest integer that a float32
    identifier holds exactly.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – For `"gadget"`, if `part_type` is not an integer from 0 to 5.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the file read back differs from the particles written.

See also

[`Particles.write`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.Particles.write)
write_particles as a method of Particles.

[`register_particle_writer`](https://pianistpy.com/docs/reference/api/particles#pianistpy.particles.register_particle_writer)
Add a layout to `PARTICLE_WRITERS`.

`read`
Read a file of the native layout back into a Particles object.

Examples

```python
>>> import os
>>> import tempfile
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.particles import write_particles
>>> particles = pp.Particles(
...     position=np.zeros((2, 3)),
...     velocity=np.ones((2, 3)),
...     mass=np.full(2, 0.5),
...     component=np.zeros(2, dtype=int),
...     component_names=("halo",),
...     unit_system="internal",
...     units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
...     identity="a" * 64,
...     parent="b" * 64,
... )
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "halo.txt")
...     write_particles(particles, path, layout="text")
...     with open(path) as file:
...         first_line = file.readline().strip()
>>> first_line
'# PIANISTpy particles'
```
