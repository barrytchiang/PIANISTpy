# Density profiles (pianistpy.profiles)

Source: https://pianistpy.com/docs/reference/api/profiles.html

# Density profiles (pianistpy.profiles)

Density profiles.

class pianistpy.profiles.DoublePowerLaw(rho_alpha, rho_beta, rho_gamma)

Double-power-law density profile in internal units.

Parameters:

rho_alpha (float) – Sharpness \(\alpha\) of the transition between the inner and the
outer slope. At least 1/2.

rho_beta (float) – Outer logarithmic slope \(\beta\). Greater than 2.

rho_gamma (float) – Inner logarithmic slope \(\gamma\), with
\(0 \leq \gamma < 2\).

Notes

In internal units the scale radius is 1, the central potential is 1 and the
potential vanishes at infinity:

\[\varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha} .\]

The lower bound on \(\gamma\) excludes a density that decreases
towards the center; the upper bound keeps the central potential finite.
The bound on \(\beta\) keeps the potential finite at large radii.

The named profiles are parameter sets of this one profile, built by the
class methods nfw, hernquist and plummer. They are not separate
implementations. Where a profile has a closed form for its mass or its
potential, that form is used inside mass and p; the tests hold it to
the general incomplete-Beta form.

References

Paper I, Appendix A (labels eqn:double_power_law_MT,
eqn:double_power_law_mass, app:P_beta).

Examples

```
>>> DoublePowerLaw.nfw()
DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
>>> DoublePowerLaw.hernquist().mass_total
1.0
>>> float(DoublePowerLaw.plummer().p(0.0))
1.0

```

classmethod nfw()

Return the Navarro-Frenk-White profile, (1, 3, 1).

Return type:

DoublePowerLaw

classmethod hernquist()

Return the Hernquist profile, (1, 4, 1).

Return type:

DoublePowerLaw

classmethod plummer()

Return the Plummer profile, (2, 5, 0).

Return type:

DoublePowerLaw

property name: str

Name of the profile, or its parameters if it has none.

property rho_norm: float

Normalization \(C\) of the density, for a central potential of 1.

Notes

\[C = \frac{\alpha\,\Gamma[(\beta-\gamma)/\alpha]}
         {\Gamma[(\beta-2)/\alpha]\,\Gamma[(2-\gamma)/\alpha]} .\]

property mass_total: float

Total mass in internal units; infinite for an outer slope of 3 or less.

rho(r)

Return the density \(\varrho(R)\).

Parameters:

r (array-like) – Radius in internal units.

Returns:

rho – Density in internal units, with the shape of r.

Return type:

array

mass(r)

Return the enclosed mass \(\mathcal{M}(R)\).

Parameters:

r (array-like) – Radius in internal units.

Returns:

mass – Enclosed mass in internal units, with the shape of r.

Return type:

array

Notes

With \(z = R^\alpha/(1+R^\alpha)\),

\[\mathcal{M}(R) = \frac{C}{\alpha}\,
    B\!\left(z;\,\frac{3-\gamma}{\alpha},\,\frac{\beta-3}{\alpha}\right) .\]

p(r)

Return the relative potential \(P(R)\).

Parameters:

r (array-like) – Radius in internal units.

Returns:

p – Relative potential in internal units, with the shape of r. It is 1
at the center and vanishes at infinity.

Return type:

array

Notes

With \(z = R^\alpha/(1+R^\alpha)\), \(a_2 = (2-\gamma)/\alpha\)
and \(b_2 = (\beta-2)/\alpha\),

\[P(R) = \frac{\mathcal{M}(R)}{R}
     + \frac{C}{\alpha}\left[B(a_2, b_2) - B(z;\,a_2, b_2)\right] .\]

depth_inner(r, terms=4)

Return the depth \(u = P(0) - P(R)\) near the center, with no
subtraction, and the series’ own truncation estimate.

Parameters:

r (array-like) – Radius in internal units.

terms (int, optional) – Number of terms of the series.

Returns:

u (array) – The depth from the first terms terms.

bound (array) – The magnitude of the next term relative to u, per point;
infinite where u is not positive.

Return type:

tuple[Any, Any]

Notes

A density term \(C\binom{e}{k}R^{-\gamma+k\alpha}\),
\(e = (\gamma-\beta)/\alpha\), contributes
\(C\binom{e}{k}R^{2-\gamma+k\alpha}/[(2-\gamma+k\alpha)(3-\gamma+k\alpha)]\)
to the depth, term by term.

to_dict()

Return the profile as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build the profile from its dictionary form.

Parameters:

record (Mapping[str, Any])

Return type:

DoublePowerLaw

class pianistpy.profiles.ProfileTables(log_r, log_p, log_p_slope, log_r_all, log_mass, log_mass_slope, log_rho_coefficients, has_edge_row, r_last, p_last, rho_last, r_t, mass_t, edge_exponent)

The arrays a kernel reads a TruncatedProfile from.

They are the numbers TruncatedProfile.potential, density, mass and
radius evaluate, laid out for a kernel that takes them as traced
arguments: the cubic Hermite in the logarithms on the interior rows and
the edge’s closed forms beyond them.

Parameters:

log_r (array, shape (n_interior,)) – ln r, ln P and d ln P / d ln r at the interior rows (every row but
an edge row).

log_p (array, shape (n_interior,)) – ln r, ln P and d ln P / d ln r at the interior rows (every row but
an edge row).

log_p_slope (array, shape (n_interior,)) – ln r, ln P and d ln P / d ln r at the interior rows (every row but
an edge row).

log_r_all (array, shape (n,)) – ln r, ln M and d ln M / d ln r at every row.

log_mass (array, shape (n,)) – ln r, ln M and d ln M / d ln r at every row.

log_mass_slope (array, shape (n,)) – ln r, ln M and d ln M / d ln r at every row.

log_rho_coefficients (array, shape (n_components, 4, n_interior - 1)) – The coefficients of the cubic spline of ln rho in ln r on each
interval, highest power first, in the interval’s own ln r offset.

has_edge_row (bool) – Whether the profile carries its edge as its last row; the fields
below are read only then.

r_last (float) – Radius and potential of the last interior row.

p_last (float) – Radius and potential of the last interior row.

rho_last (array, shape (n_components,)) – Density of each component there.

r_t (float) – Truncation radius and total mass.

mass_t (float) – Truncation radius and total mass.

edge_exponent (array, shape (n_components,)) – Exponent q of each component’s density at the edge.

class pianistpy.profiles.TruncatedProfile(r, p_trunc, rho_trunc, mass_trunc, k_t, r_t, mass_t, spatially_finite, source, source_anisotropy, model, edge_exponent, u_trunc=<factory>, central_deficit=(), override_r=nan, override_r_end=nan, override_rows=0, override_rule='none', origin=<factory>, df_k=<factory>, df_log_f=<factory>, df_slope=<factory>, df_taylor=())

The self-consistent profile of a truncated model.

It is the profile of a model, and it can be the profile of a component
of another model: a component given the truncated profile of a parent
model and a new anisotropy is Method II of the paper.

Two profiles are equal where they belong to the same model: the identity
of a model determines its profile.

The profile evaluates itself: potential, density, mass and
radius are the one reading of the rows that every stage of the
package takes, in place of an interpolant of its own. Between the rows
the potential and the mass are cubic Hermite interpolants in the
logarithms with the exact slopes the rows state (the Poisson equation
gives \(\mathrm{d}\ln\mathcal{P}/\mathrm{d}\ln r = -M/(r\mathcal{P})\)
and \(\mathrm{d}\ln M/\mathrm{d}\ln r = r^3\rho/M\)); the density
of each component is a cubic spline in the logarithms, since the rows
state no density slope. On the last interval of a spatially finite
profile whose last row is its edge (\(\mathcal{P} = 0\) at r_t),
the potential is the Kepler form \(M_t(1/r - 1/r_t)\) and the
density follows the edge law \(\rho \propto \mathcal{P}^q\) with
the stated edge_exponent; beyond r_t the potential is \(M_t/r\)
and the density zero. Inside the first row and, for a profile without
an edge, beyond the last, the interpolants continue as the power laws
of the end rows.

Parameters:

r (array, shape (n,)) – Radius, strictly increasing.

p_trunc (array, shape (n,)) – Total relative potential of the truncated model.

rho_trunc (array, shape (n_components, n)) – Density of each component.

mass_trunc (array, shape (n,)) – Total enclosed mass, in the unit of DoublePowerLaw.mass (the
Hernquist profile’s total is 1); 4 pi times it is the integral of
the density over the volume.

k_t (float) – Truncation energy.

r_t (float) – Truncation radius and total mass; infinite where the model is not
spatially finite.

mass_t (float) – Truncation radius and total mass; infinite where the model is not
spatially finite.

spatially_finite (bool) – Whether the density reaches zero at a finite radius with a finite
total mass. A spatially finite profile whose last row has
p_trunc == 0 carries its edge as that row, and then the row’s
density is zero and its radius is r_t.

source (DoublePowerLaw) – The analytic profile of the first component of the model: its scale
radius and its central potential are the internal units the arrays
are in.

source_anisotropy (Anisotropy) – The anisotropy of that component in the model the profile belongs
to. A component built on this profile (Method II) reads here what
its parent was.

model (str) – Identity string of the model the profile belongs to.

edge_exponent (tuple of float) – Exponent \(q\) of each component’s density at the edge of the
model, \(\rho \propto \mathcal{P}^q\); empty where the model
is not spatially finite and has no edge.

u_trunc (array, shape (n,)) – The central potential less the potential, p_trunc[0] + u_trunc[0]
less p_trunc, kept exact where p_trunc is close to it: near the
center p_trunc alone has lost the digits of the difference that the
DF’s singular center is read in. Empty where not computed.

central_deficit (tuple of tuple of float) – For each component, (D_0, D_1, …): the central auxiliary density
of the orbits the truncation removed as its Taylor series in the
depth u, D_0 - D_1 u + D_2 u^2 - … (D_0 to D_4 as made; a record
of D_0 and D_1 alone reads as the linear deficit). Empty where not
recorded.

override_r (float) – Radii between which the closed-form profile replaced the rows of
the solve; NaN where no row was replaced.

override_r_end (float) – Radii between which the closed-form profile replaced the rows of
the solve; NaN where no row was replaced.

df_k (array, shape (n_components, m)) – The DF table of the model the profile belongs to: the energies,
log10 F and its slope d log10 F / d ln K (NaN where not exact), as
truncate made them. A component built on this profile (Method II)
reads its sources from the integral of this DF. Empty where not
recorded.

df_log_f (array, shape (n_components, m)) – The DF table of the model the profile belongs to: the energies,
log10 F and its slope d log10 F / d ln K (NaN where not exact), as
truncate made them. A component built on this profile (Method II)
reads its sources from the integral of this DF. Empty where not
recorded.

df_slope (array, shape (n_components, m)) – The DF table of the model the profile belongs to: the energies,
log10 F and its slope d log10 F / d ln K (NaN where not exact), as
truncate made them. A component built on this profile (Method II)
reads its sources from the integral of this DF. Empty where not
recorded.

df_taylor (tuple of tuple of float) – For each component, the model’s Taylor edge (F_1, F_2, K_switch):
F = F_1 K + F_2 K^2 / 2 below K_switch. Empty where not recorded.

override_rows (int) – Number of rows replaced there.

override_rule (str) – What replaced them, "none" where nothing did.

origin (array of int8, shape (n,)) – Where each row comes from, a code of
pianistpy.df.truncation.ORIGINS (the main solve, the edge
continuation, the floor event, the fills, the closed-form outer rows,
the edge row): the profile’s checks read each row’s tolerance from it.
Empty where not recorded.

property has_edge_row: bool

Whether the last row is the edge of a spatially finite profile.

property r_table_max: float

the outer end of the table, which is
the edge of a spatially finite model.

Type:

The radius of the last row

property mass_at_r_table_max: float

mass_trunc at the last row.

interpolation_tables()

Return the arrays a kernel reads this profile from.

A kernel that evaluates them as potential, density, mass and
radius do gives the same numbers; the arrays are traced arguments
of the kernel, never closed over.

Return type:

ProfileTables

potential(r)

Return the relative potential at radii r.

Parameters:

r (Any)

Return type:

Any

density(r)

Return the density of each component at radii r.

The result has one row per component and one column per radius.

Parameters:

r (Any)

Return type:

Any

mass(r)

Return the total enclosed mass at radii r.

Parameters:

r (Any)

Return type:

Any

potential_log_slope(r)

Return \(d\ln P / d\ln r\) at radii r, from the profile’s own
representation.

The slope is exact in the rows’ quantities: \(dP/dr = -M/r^2\)
(the Poisson equation in the profile’s unit), so
\(d\ln P/d\ln r = -M/(r P)\) with mass and potential read as
every consumer reads them. The knots of the representation are the
rows r; between two rows the slope is smooth.

Parameters:

r (Any)

Return type:

Any

potential_log_slope_derivative(r)

Return \(d(d\ln P/d\ln r)/dr\) at radii r, exact in the rows’
quantities.

With \(q = -M/(rP)\), \(dM/dr = r^2 \varrho\) (the total
density of the components) and \(dP/dr = -M/r^2\):
\(dq/dr = -(r^3 \varrho P - M P + M^2/r) / (r^2 P^2)\). Beyond the
edge the density is zero and the same form holds on the Kepler
potential.

Parameters:

r (Any)

Return type:

Any

radius(p)

Return the radii at which the potential takes the values p.

Values above the central potential are refused; a value of zero or
below is the truncation radius of a spatially finite profile.

Parameters:

p (Any)

Return type:

Any

to_dict()

Return what identifies the profile, as a plain dictionary.

The arrays are not part of it: the identity of the model determines
them, and a file holds them beside this record.

Return type:

dict[str, Any]

     On this page
