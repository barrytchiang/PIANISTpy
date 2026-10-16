# Distribution functions and models (pianistpy.df)

Source: https://pianistpy.com/docs/reference/api/df.html

# Distribution functions and models (pianistpy.df)

Distribution functions and equilibrium models.

class pianistpy.df.Component(profile, anisotropy=<factory>, mass=1.0, name='component')

One component of a model: a density profile with its anisotropy.

A model is a sequence of components that share one total potential. A
single component is a sequence of length one.

Given the truncated profile of another model (parent.profile) in
place of an analytic profile, the component is Method II of the paper:
the density of the parent in the parent’s potential, with a new
anisotropy. It is refused here where no distribution function exists
for that anisotropy at the edge of the density.

Parameters:

profile (DoublePowerLaw or TruncatedProfile) – Density profile of the component: an analytic profile, or the
truncated profile of a model of one component.

anisotropy (Anisotropy, optional) – Velocity anisotropy of the component. Isotropic by default.

mass (float, optional) – Mass of the component relative to the first component.

name (str, optional) – Label carried by the particles and the results of the component.

property unit_profile: DoublePowerLaw

The analytic profile that fixes the internal units of the component.

to_dict()

Return the component as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record, profile=None)

Build the component from its dictionary form.

Parameters:

record (mapping) – The dictionary form.

profile (TruncatedProfile, optional) – The profile, where the record states a truncated one: the
record identifies it and does not hold its arrays.

Return type:

Component

class pianistpy.df.DistributionFunction(components, settings, identity, grid, table, form, k_t_min, validation, timing, inversion, units=Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None), rows=None)

Untruncated distribution functions of the components of a model.

Returned by compute_df. It is a mathematical object: it may be negative or
non-monotonic, and states where. Only a Model can be sampled or
projected.

Parameters:

components (tuple of Component) – The components the DFs belong to.

settings (Settings) – The settings the DFs were computed with.

identity (str) – Identity string of the result.

grid (EnergyGrid) – The energies of the table and their radii.

table (DfTable) – The DFs on the grid.

form (AsymptoticForm) – The analytic form of each DF beyond the ends of its table.

k_t_min (float) – Lowest admissible truncation energy; zero where every DF is
non-negative and non-decreasing.

validation (ValidationReport) – The report of the inversion.

timing (dict of str to float) – Wall time of each stage, in seconds.

inversion (InversionRecord or None) – How the table was made; None where it was not made by an inversion
(a closed form). Every solved DF states it.

rows (InversionRows or None) – The inversion’s rows as made on its whole grid, which the truncation
reads; None where the table was not made by an inversion.

units (Units)

property is_physical: bool

Whether the untruncated DFs are a valid equilibrium as they stand.

property diverges_at_zero: tuple[bool, ...]

Whether the DF of each component diverges as the energy goes to zero.

See AsymptoticForm.diverges_at_zero.

to(system, concentration=None, mass_0=None, r_0=None, r_vir=None)

Return the same result presented in another unit system.

The numbers of the result are not recomputed: they stay in internal
units, and what is computed from the result inherits its units.

Parameters:

system ({"internal", "model", "physical"}) – The target unit system.

concentration (float, optional) – Concentration that defines M_0, the mass inside that many scale
radii. Taken from the truncation where it was given there.

mass_0 (float or astropy Quantity, optional) – M_0, for the physical system; a float is in solar masses.

r_0 (float or astropy Quantity, optional) – The scale radius, or the concentration radius in its place, for
the physical system; a float is in parsecs.

r_vir (float or astropy Quantity, optional) – The scale radius, or the concentration radius in its place, for
the physical system; a float is in parsecs.

Returns:

df

Return type:

DistributionFunction

write(path, overwrite=False)

Write the result to a file; pianistpy.read reads it back.

Parameters:

path (str | PathLike)

overwrite (bool)

Return type:

None

write_text(path, overwrite=False)

Write the DF table as fixed-width text, an export beside the file.

One row per node of each component’s support: the energy k (in
units of the central potential, internal units), log10_df and
the slope dlog10_df_dlnk of the table, and with more than one
component its index in a component column. The header names the
result, its components and the units. The file is read back and
compared with what was written, to the digits written
(pianistpy.io.write_table).

Parameters:

path (str | PathLike)

overwrite (bool)

Return type:

None

class pianistpy.df.Model(components, truncation, settings, identity, parent, table, form, profile, validation, timing, r_trust, inversion, units=Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None))

A valid equilibrium: truncated or untruncated, of one or more components.

Returned by truncate and solve. A model always has non-negative,
non-decreasing DFs. Whether its density ends at a finite radius is a
property of the model, spatially_finite.

Parameters:

components (tuple of Component) – The components of the model.

truncation (Truncation or None) – The truncation, or None for an untruncated model.

settings (Settings) – The settings the model was computed with.

identity (str) – Identity string of the result.

parent (str) – Identity string of the DistributionFunction the model was built from.

table (DfTable) – The DFs of the model.

form (AsymptoticForm) – The analytic form of each DF beyond the ends of its table.

profile (TruncatedProfile) – The self-consistent profile of the model.

validation (ValidationReport) – The report of the model.

timing (dict of str to float) – Wall time of each stage, in seconds.

r_trust (mapping of str to float) – The radius within which the model holds each accuracy tier, keyed by
the report’s tier names (“goal”, “ceiling”; “limit” where it is
computed): where the result is valid, not an error of it. Every tier
or none: the radii are measured on an untruncated model, from the
offset of its potential at large radii, and the mapping is empty on
a truncated model (a spatially finite one holds its table to its
edge) and where that offset is undefined. A tier the table’s first
outer row already misses has radius 0; infinite on a closed form.

inversion (InversionRecord or None) – How the model’s table was made; None where it was not made by an
inversion (a closed form). Every solved model states it.

units (Units)

property spatially_finite: bool

Whether the density reaches zero at a finite radius and mass.

property diverges_at_zero: tuple[bool, ...]

Whether the DF of each component diverges as the energy goes to zero.

See AsymptoticForm.diverges_at_zero. Such a model is a valid
equilibrium with a finite density; its DF is not monotonic.

to(system, concentration=None, mass_0=None, r_0=None, r_vir=None)

Return the same model presented in another unit system.

The numbers of the model are not recomputed: they stay in internal
units, and what is computed from the model inherits its units.

Parameters:

system ({"internal", "model", "physical"}) – The target unit system.

concentration (float, optional) – Concentration that defines M_0, the mass inside that many scale
radii. Taken from the truncation where it was given there.

mass_0 (float or astropy Quantity, optional) – M_0, for the physical system; a float is in solar masses.

r_0 (float or astropy Quantity, optional) – The scale radius, or the concentration radius in its place, for
the physical system; a float is in parsecs.

r_vir (float or astropy Quantity, optional) – The scale radius, or the concentration radius in its place, for
the physical system; a float is in parsecs.

Returns:

model

Return type:

Model

radial_profiles()

Return the radial profiles of the model in its unit system.

Returns:

profiles – "r", "potential" (the relative potential), "density"
(one row per component) and "mass" (the total enclosed
mass): plain arrays in the internal and the model systems,
Astropy quantities in the physical system.

Return type:

dict

property mass_t: Any

Total mass of the model, in its unit system.

property r_t: Any

Outer radius of the model, in its unit system.

write(path, overwrite=False)

Write the result to a file; pianistpy.read reads it back.

Parameters:

path (str | PathLike)

overwrite (bool)

Return type:

None

write_text(path, table='profile', overwrite=False)

Write a table of the model as fixed-width text, an export beside the file.

Parameters:

path (path) – File to write.

table ({"profile", "df"}) – "profile": the radial profiles in the model’s unit system
(radial_profiles): r, potential, the density of
each component (density_<name> with more than one) and the
enclosed mass; the header states the unit of each column.
"df": the DF table, as DistributionFunction.write_text
writes it.

overwrite (bool, optional) – Replace an existing file.

Raises:

InputError – For a table other than the two, or an existing file without
    overwrite.

ResultError – If the file does not read back as written.

Return type:

None

class pianistpy.df.Truncation(k_t=None, mass_t=None, r_t=None, concentration=None)

How the distribution function of a model is truncated.

This is Method I of the paper. Method II needs no truncation: it is a
component whose profile is the truncated profile of another model, given
a new anisotropy.

Exactly one of k_t, mass_t and r_t is given. A target mass is given
together with the concentration that defines its unit.

Parameters:

k_t (float, optional) – Truncation energy \(\mathcal{K}_{\rm t}\), in internal units (the
central potential of the untruncated profile is 1).

mass_t (float, optional) – Target total mass of the truncated model, in MODEL units: in units of
\(M_0\), the mass of the untruncated profile inside the
concentration radius, so that a value of 1 is the virial mass. This
is the one input that is not in internal units. The truncation energy
is solved for. Requires concentration.

r_t (float, optional) – Target truncation radius \(R_{\rm t}\), in internal units (the
scale radius \(r_0\) is their unit of length). The truncation
energy is solved for.

concentration (float, optional) – Concentration \(c\) of the first component: \(M_0\) is the
mass of its untruncated profile inside \(c\,r_0\). Required
with mass_t, and refused with k_t and r_t, which do not depend
on it.

to_dict()

Return the truncation as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build the truncation from its dictionary form.

Parameters:

record (Mapping[str, Any])

Return type:

Truncation

pianistpy.df.compute_df(components, settings=None)

Return the untruncated distribution function of each component.

The DF is returned whatever its sign, with the information needed to
truncate it: where it is negative, its minimum, and the lowest admissible
truncation energy.

Parameters:

components (Component or sequence of Component) – The component of the model. This release takes one; a sequence of
several, in one total potential, is refused.

settings (Settings, optional) – Numerical settings. The default tier if omitted.

Returns:

df

Return type:

DistributionFunction

Raises:

NotImplementedError – If the component’s anisotropy family is not served yet (served:
    isotropic, constant beta and Osipkov-Merritt Type I).

pianistpy.df.solve(components, truncation=None, settings=None)

Return the equilibrium model of the given components.

This runs compute_df and then truncate.

Parameters:

components (Component or sequence of Component) – The component of the model. This release takes one; a sequence of
several, in one total potential, is refused.

truncation (Truncation, optional) – The truncation. If omitted, the model is untruncated.

settings (Settings, optional) – Numerical settings. The default tier if omitted.

Returns:

model

Return type:

Model

pianistpy.df.truncate(df, truncation=None)

Return the equilibrium model built from a distribution function.

Parameters:

df (DistributionFunction) – Result of compute_df.

truncation (Truncation, optional) – The truncation. If omitted, the untruncated model is returned where
it is a valid equilibrium.

Returns:

model

Return type:

Model

Raises:

UnphysicalModelError – If no valid equilibrium exists for the request. The exception carries
    df as its result.

     On this page
