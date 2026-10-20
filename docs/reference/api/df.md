# Distribution functions and models

Source: https://pianistpy.com/docs/reference/api/df

# Distribution functions and models

Module pianistpy.df

Distribution functions and equilibrium models.

class pianistpy.df.Component(profile, anisotropy=<factory>, mass=1.0, name='component')

One component of a model: a density profile with its anisotropy.

A model is a sequence of components that share one total potential. A
single component is a sequence of length one.

Given the truncated profile of another model (parent.profile) in
place of an analytic profile, the component is Method II of Chiang+(2026d):
the density of the parent in the parent’s potential, with a new
anisotropy. It is refused here where no distribution function exists
for that anisotropy at the edge of the density.

A profile with an outer slope between 2 and 3 is refused here (OUTER
SLOPE): the outer end of the inversion and its tail exponent assume
Psi ~ 1/R, which that regime does not have.

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

class pianistpy.df.DistributionFunction(components, settings, identity, grid, table, form, k_t_min, validation, timing, inversion, units=Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None), rows=None, source='inversion', function_name=None)

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
non-negative and non-decreasing. For a Type II model whose bound is
its domain edge, the next float above \(P(R_{\rm a})\), the
smallest \(\mathcal{K}_{\rm t}\) the gates admit; the truncated
model’s edge must also lie inside \(R_{\rm a}\), which the
truncation’s solve judges (TYPE II EDGE).

validation (ValidationReport) – The report of the inversion.

timing (dict of str to float) – Wall time of each stage, in seconds.

inversion (InversionRecord or None) – How the table was made; None where it was not made by an inversion
(a closed form). Every solved DF states it.

rows (InversionRows or None) – The inversion’s rows as made on its whole grid, which the truncation
reads; None where the table was not made by an inversion.

source (str) – Where the table’s values come from: "inversion" (compute_df)
or "function" (from_function).

function_name (str or None) – The __name__ of the function the table was taken from; None for
an inversion.

units (Units)

classmethod from_function(component, f, settings=None)

Return the distribution function a function of the energy gives.

A user brings a DF (a fitting function, another code’s output) and
the package validates it against the density by its own integration.
The function is tabulated on the energy grid compute_df places for
the same component and settings, in place of the inversion’s values,
and everything after runs as for the inversion: the forms beyond the
table, the slope column, the sign report, the validation, the
identity, and the file. The table is stored, the function is not.

Parameters:

component (Component) – The component the DF belongs to, on an analytic profile.

f (callable) – f(k, depth): takes the array of the energies K (the table’s,
in internal units) and the array of their depths P_0 - K, and
returns the DF at each, F and not its logarithm. The depth is
the grid’s own P_0 - K of each node, exact, so a fit written in
the depth near the center reads it directly; a function of K
alone may ignore it.

settings (Settings, optional) – Numerical settings. The default tier if omitted.

Returns:

df – Its source is "function" and its function_name the
function’s __name__. The table has no derivative column, and
the report’s slope_column states that it is not judged. It
has no inversion record and no rows, so truncate refuses it.

Return type:

DistributionFunction

Raises:

InputError – If the function is not finite somewhere on the grid (the first
    such K is stated), returns another shape, or the component is
    built on a tabulated profile.

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

This is Method I of Chiang+(2026d). Method II needs no truncation: it is a
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

InputError – If the DF comes from a function (DistributionFunction.from_function):
    it carries no inversion rows, which truncation needs.

## Gradients

Module pianistpy.df.gradient

The distribution function’s rows and table as JAX functions of the anisotropy parameters, and their Jacobians
(End-to-end differentiability).

The inversion’s gradient in the anisotropy parameters at a fixed profile.

The inversion’s rows F = T_b[g](q) (invert._rows: the kernel and the
completion below the floor) on a fixed grid q. At a fixed central anisotropy
b the operator T_b is linear in its source g, the rows of d^N rho_aux / dP^N,
so, by that linearity:

dF/dR_a    = T_b[ d_Ra g ]
dF/dbeta_c = T_b[ d_b g ] + E_b[g]

with d_theta g the mixed derivatives at fixed P (ParameterDerivs) and E_b
the derivative of the operator itself in b with g held: the prefactor C(n, b)
and the kernel’s exponent, computed by the operator at b +- d (central,
Richardson over d and d/2) or, at the corners where the inversion’s form
changes on one side (half-integer \(\beta_{\rm c}\), the pair’s switch),
by the one-sided Richardson combinations (third order) of both sides and
their mean; the step d
balances the kernel’s measured accuracy over d against the quotient’s
truncation term (_exponent_part).

The table (invert._table_pieces) is a function of the rows and of closed
forms in theta, so its derivative is the chain rule through each piece, with
every discrete choice of the forward held (the grid, the anchor, the fit’s
window, the nodes below the written floor), table_jacobian:

the rows’ log10 F and their log slope, from dF and d(dF/dK), the rows’
part as the operator’s own derivative along the mixed rows (_along);

the form above the anchor (top_patch): its least-squares fit linearized
in the rows, and the center’s exponent kappa(b) and amplitude A(b) in
closed form (the regular branch);

the nodes below the written floor (below_floor): the integral linear in
its source, the source’s cubic in (ln P, ln |g|) linearized, and in b the
floor pair’s exponent and prefactor by the same difference as E_b;

the tail (A, a(b), k), from the first node below the floor or the first
written row.

On a component on a truncated profile (Method II, tabulated.tabulated_df)
the parent is held and the child’s sources are a Leibniz sum of the parent’s
density derivatives and the factor g(R(P); beta_c, R_a), so their derivative
is the same sum with d g / d theta in closed form; the rows follow as above
on the grid held at theta, and the leaves are log10 |F| on the table’s nodes,
the edge law (p, C_0, C_1) and the center the form states, each closed form
with its derivative written out.

Served: constant beta and Cuddeford-Osipkov-Merritt (COM) Type I (\(\beta_{\rm c}\),
\(R_{\rm a}\)), Osipkov-Merritt Type I and II (\(R_{\rm a}\)).
Refused: COM Type II, whose interior point at \(P(R_{\rm a})\) moves
with \(R_{\rm a}\) (and its exponent with \(\beta_{\rm c}\)), so the
rows of the mixed derivative are not integrable there; and a table whose
center is not on its regular branch (the critical band, where the branch
changes with \(\beta_{\rm c}\)).

pianistpy.df.gradient.inversion_nodes(component, settings)

Return the grid q the inversion places its rows on (inversion_df’s:
the pilot’s feature where the family has one, then top_limit, the
least depth at which the exact derivatives are finite in float64).

Parameters:

component (Component)

settings (Settings)

Return type:

ndarray

pianistpy.df.gradient.make_inversion_rows(component, settings, q=None)

Return (rows, params, q): rows(theta) is the inversion’s rows on
the fixed grid q as a JAX function of the component’s anisotropy
parameters (params, in that order) with a custom VJP. The forward is
invert._rows through a host callback (the forward’s numbers), the
backward the rule (rows_jacobian) at the same parameters.

Parameters:

component (Component)

settings (Settings)

q (ndarray | None)

Return type:

tuple[Any, tuple[str, …], ndarray]

pianistpy.df.gradient.make_inversion_table(component, settings, q=None)

Return (table, params, values): table(theta) is the DF table’s
values (table_jacobian’s, but the energies) as a JAX function of the
component’s anisotropy parameters with a custom VJP; values the
table at the component’s own parameters, energies included. The forward
is inversion_df’s table through a host callback, the backward the rule.
The grid and the forward’s discrete choices are those at the component’s
parameters.

Parameters:

component (Component)

settings (Settings)

q (ndarray | None)

Return type:

tuple[Any, tuple[str, …], dict[str, ndarray]]

pianistpy.df.gradient.rows_jacobian(component, settings, q=None, steps=None)

Return the inversion’s rows and their derivatives in the component’s
anisotropy parameters on a fixed grid.

Parameters:

component (Component) – An analytic Method I component.

settings (Settings)

q (array, optional) – The grid; inversion_nodes where None.

steps (dict[tuple, dict[str, Any]] | None)

Returns:

q, f (array) – The grid and the rows (the forward’s, unchanged).

jac (dict) – {"beta_c": dF/dbeta_c, "r_a": dF/dR_a}, the parameters the family
has.

record (dict) – How the exponent’s part was taken ("central" or "one-sided"),
its steps, the kernel’s accuracy and its Richardson residual max
|E(d) - E(d/2)| over the column’s scale max |dF/dbeta_c| and,
one-sided, the two sides’ spread over the same scale. The accuracy
is one maximum over the whole column, whose entries can span some
thirty decades (the largest near the center), so the steps suit the
largest entries; an entry far below them is differenced at the same
steps.

Raises:

InputError – Where the component is not analytic, or a COM Type II, or its
    family has no parameter.

Return type:

tuple[ndarray, ndarray, dict[str, ndarray], dict[str, Any]]

pianistpy.df.gradient.table_jacobian(component, settings, q=None, steps=None)

Return the DF table’s values and their derivatives in the component’s
anisotropy parameters, the forward’s grid and discrete choices held.

Parameters:

component (Component) – An analytic Method I component.

settings (Settings)

q (array, optional) – The grid; inversion_nodes where None.

steps (dict, optional) – The caller’s store of the exponent’s measured steps
(_table_difference): read where it has them, filled where it has
not; without one they are measured at this call.

Returns:

values (dict) – The table as inversion_df makes it, as arrays: k, log_df,
log_df_slope (the rows; zero derivative where a Type II DF is
outside its support), and where the table has them k_below,
log_df_below, log_df_slope_below; the form above the anchor
central (kappa, a_anchor, b, d) and the tail tail (A, a, k).

jac (dict) – {theta: {key: d value / d theta}} for the keys of values but the
energies.

record (dict) – The exponent’s part of the rows ("rows") and of the nodes below
the floor ("below"), as rows_jacobian states it.

Raises:

InputError – As rows_jacobian; and where the form above the anchor is used and
    the center is not on its regular branch (the critical band).

Return type:

tuple[dict[str, ndarray], dict[str, dict[str, ndarray]], dict[str, Any]]

     On this page
