# Observables (pianistpy.observables)

Source: https://pianistpy.com/docs/reference/api/observables.html

# Observables (pianistpy.observables)

Observables: dispersions, line-of-sight velocity distributions and apertures.

class pianistpy.observables.Aperture(shape, r_off=0.0, r_ape=None, side_x=None, side_y=None, angle=0.0)

An aperture on the plane of the sky, circular or rectangular.

Build one with Aperture.circle or Aperture.rectangle. Lengths are
in internal units.

Parameters:

shape ({"circle", "rectangle"}) – Shape of the aperture.

r_off (float) – Projected distance \(R_{\rm off}\) of the center of the aperture
from the center of the model.

r_ape (float, optional) – Radius \(R_{\rm ape}\) of a circle.

side_x (float, optional) – Full sides of a rectangle.

side_y (float, optional) – Full sides of a rectangle.

angle (float, optional) – For a rectangle, the angle between its x side and the direction from
its center to the center of the model, in radians.

Examples

```
>>> Aperture.circle(1.0, r_off=10.0).shape
'circle'
>>> Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3).side_y
0.5

```

classmethod circle(r_ape, r_off=0.0)

Return a circular aperture of radius r_ape at offset r_off.

Parameters:

r_ape (float)

r_off (float)

Return type:

Aperture

classmethod rectangle(side_x, side_y, r_off=0.0, angle=0.0)

Return a rectangular aperture of full sides side_x by side_y.

Its center is at offset r_off, and angle is the angle between its
x side and the direction from its center to the center of the model.

Parameters:

side_x (float)

side_y (float)

r_off (float)

angle (float)

Return type:

Aperture

arc(r)

Return the angular measure of the circle of radius r inside the
aperture, in radians.

For a circle it is the closed form of the overlap of two circles:
\(2\pi\) where the whole circle is inside, zero where it is
outside, and the band
\(4 \arcsin\sqrt{x/2}\) with
\(x = [R_{\rm ape}^2 - (r - R_{\rm off})^2] / (2 r R_{\rm off})\)
between. For a rectangle, each of the four edges is a half-plane
constraint on the angle that becomes active at angles in closed form;
those angles cut the circle into intervals on which membership is
constant, and the measure is the length of the intervals whose
midpoint lies inside. One code path serves a rectangle that contains
the center of the model, touches it, or is clear of it.

Parameters:

r (array-like) – Projected radii, not negative.

Returns:

arc

Return type:

array, the shape of r

breaks()

Return the radii at which arc is not smooth, increasing.

For a circle: the edges of the band and, where the aperture contains
the center of the model, the radius at which the full circle ends.
For a rectangle: the perpendicular distances from the center of the
model to the four edge lines, kept where the foot of the perpendicular
lies on the edge, and the distances to the four corners; at most
eight radii, between two of which arc is smooth.

Return type:

list[float]

to_dict()

Return the aperture as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build the aperture from its dictionary form.

Parameters:

record (Mapping[str, Any])

Return type:

Aperture

class pianistpy.observables.LosTable(model)

The line-of-sight velocity distributions of a model, built once.

The object holds the table of line-of-sight velocity distributions over
projected radius that the aperture quantities read. It is built when the
object is created and reused by every observable computed from it.

Parameters:

model (Model) – The model to project: of one component, or a superposition
(superpose).

Variables:

model (Model) – The model, whose unit system the observables read from this object
are given in.

table (profile, projected, kept,) – The results of the stages dispersion_profile,
projected_dispersion, kept_ranges and losvd_table, at the
model’s settings; of the first component for a superposition.

validation (ValidationReport) – The checks of the table (those of validate_table) and the worst
tier of the model; for a superposition, each check at its worst over
the components. A table above the goal says so in a warning.

parts (tuple of LosTable) – One table per component of a superposition, each of the component
as a model of one component; (self,) for a model of one
component.

to(system, **kwargs)

Return the same tables presented in another unit system.

Nothing is recomputed: the tables stay in internal units, and the
observables read from the returned object are given in the units of
self.model.to(system, **kwargs) (Model.to).

Parameters:

system (str)

kwargs (Any)

Return type:

LosTable

class pianistpy.observables.ProjectedPairs(r_2d, v_los, mass, component, units, record=<factory>, validation=None)

Pairs drawn by ProjectedSampler.sample.

Variables:

mass (r_2d, v_los,) – Projected radius, line-of-sight velocity and mass of each pair, in
the source’s unit system (named in units).

component (array of int, shape (n,)) – The component each pair was drawn from.

units (dict) – The unit of each quantity.

record (dict) – The source, the ranges in internal units, the seed, the annulus’s
mass in the window and the components’ shares, the tables’ sizes.

validation (ValidationReport) – The sampler’s report.

Parameters:

r_2d (Any)

v_los (Any)

mass (Any)

component (Any)

units (dict[str, str])

record (dict[str, Any])

validation (ValidationReport | None)

class pianistpy.observables.ProjectedSampler(source, R_range, v_range=None)

Draw pairs of projected radius and line-of-sight velocity.

Pairs follow \(p(R, v) \propto 2\pi R\,\Sigma(R)\,D(v\,|\,R)\) on
the annulus R_range and in the window v_range, exactly conditioned:
the radius’s distribution carries the window’s share at each radius.
Exact for every anisotropy (the distribution is the line of sight’s
marginal, the angles integrated in the table).

Parameters:

source (Model or LosTable) – The model (its LosTable is built first) or its prepared table; of
one component or a superposition.

R_range ((float, float)) – The annulus, in the source’s unit system (Model.to): internal units
by default, parsecs (floats or Astropy quantities) in the physical
system. An end beyond the table’s rows is taken at the row where the
surface density beyond it is within INNER_DISC_TOLERANCE of the
annulus’s, and refused otherwise.

v_range ((float, float), optional) – The window of line-of-sight velocity, signed, in the source’s unit
system (km/s in the physical system); the whole distribution where
None.

Variables:

validation (ValidationReport) – v_table: the reading’s error in the cumulative, between grid
radii and between nodes; window: the window ends’ reading;
radius_quadrature: the mass integral against a finer rule,
relative; row_norm: the rows’ distance from unit integral;
annulus_cut: the share cut at the table’s ends.

mass (float) – The annulus’s mass in the window, internal units.

Raises:

InputError – Where the annulus lies outside the table, overlaps an interval the
    table left uncertified, or the window holds none of the
    distribution.

sample(n, seed=0)

Draw n pairs.

Pair i draws from its own key as Sampler does: the radius from the
channel radius, the velocity from speed and, of a
superposition, the component (by its share of the annulus’s mass in
the window) from component; so a draw is the same pair by pair
for any n.

Returns:

pairs

Return type:

ProjectedPairs

Parameters:

n (int)

seed (int)

pianistpy.observables.aperture_losvd(source, aperture, v)

Return the line-of-sight velocity distribution inside an aperture.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable. Given a model, its table is
built first (LosTable).

aperture (Aperture) – The aperture, built with Aperture.circle or Aperture.rectangle,
its lengths in the model’s unit system (parsecs in the physical
system).

v (array-like) – Line-of-sight velocity in the model’s unit system (km/s, a float or
an Astropy quantity, in the physical system).

Returns:

losvd – The probability density of the line-of-sight velocity of the
tracers inside the aperture, weighted by the surface density over
it, at each v, per unit velocity of the model’s unit system (an
Astropy quantity in the physical system), with the shape of v. It
is even in \(v\), integrates to one over all \(v\), and is
zero beyond the escape velocity along the line of sight.

Return type:

array

Notes

The distribution is the stage aperture_losvd, on Gauss-Legendre
panels in velocity refined until its norm and second moment agree with
the table’s rows; at any v it is read by each panel’s polynomial
through its own nodes, the polynomial its quadrature integrates. Its
second moment is sigma_los_aperture’s from the same LosTable, and
it is checked against the aperture’s dispersion (validate_aperture):
a distribution above the goal, or read from a table above it, says so
in a warning.

Of a superposition, each component’s distribution inside the aperture
is summed with its weight times its projected mass inside the
aperture: \(D = \sum_k w_k M_k D_k / \sum_k w_k M_k\).

pianistpy.observables.losvd(source, r_2d, v)

Return the line-of-sight velocity distribution at projected radii.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable. Given a model, its table is
built first (LosTable).

r_2d (array-like) – Projected radius \(R_{\rm 2D}\) in the model’s unit system
(Model.to): internal units by default, parsecs (a float or an
Astropy quantity) in the physical system.

v (array-like) – Line-of-sight velocity in the model’s unit system (km/s, a float or
an Astropy quantity, in the physical system). r_2d and v
broadcast against each other.

Returns:

losvd – The probability density of the line-of-sight velocity at each pair
\((R_{\rm 2D}, v)\), per unit velocity of the model’s unit
system (an Astropy quantity in the physical system), with the
broadcast shape of r_2d and v. It is even in \(v\), integrates
to one over all \(v\), and is zero beyond the escape velocity
along the line of sight.

Return type:

array

Notes

Between the rows of the table the distribution is read through the
table’s own stencil, the reading the table certified its rows for; a
projected radius inside an uncertified interval is refused by name.

Of a superposition, each component’s distribution is read from its own
table and the components are summed with their weights times their
surface densities: \(D = \sum_k w_k \Sigma_k D_k / \sum_k w_k
\Sigma_k\).

pianistpy.observables.pdf_v(source, r, v)

Return the local speed distribution at 3D radii.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable (whose intrinsic second
moments the distribution is checked against).

r (array-like) – Radius \(r\) in the model’s unit system (Model.to): internal
units by default, parsecs (a float or an Astropy quantity) in the
physical system.

v (array-like) – Speed in the model’s unit system (km/s, a float or an Astropy
quantity, in the physical system). r and v broadcast against
each other.

Returns:

pdf_v – The probability density of the speed \(V = |\mathbf{v}|\) at
each pair \((r, V)\), per unit velocity of the model’s unit
system (an Astropy quantity in the physical system), with the
broadcast shape of r and v. It integrates to one over
\(V \geq 0\) and is zero beyond the end of the DF’s support,
the escape speed but for a Type II model, whose speeds reach
\(V_{\rm esc}/\sqrt{1 - r^2/r_a^2}\) (its unbound share is
SpeedDistribution.unbound_fraction).

Return type:

array

Raises:

AccuracyError – Where the distribution at a radius is beyond the hard limit, unless
    the model’s settings ask for it anyway (write_anyway).

Notes

Each radius is the stage speed_distribution, certified by its norm
against the density, its second moment against the Jeans dispersion
and the change of its angle rule; the distribution at the speeds asked
is read by the rule it certified. Of a superposition, the components are
mixed by their density at the radius: \(p = \sum_k w_k \varrho_k
p_k / \sum_k w_k \varrho_k\).

References

Paper, Sect. 5.1 and [eqn:rho_f_anisotropic_scale].

pianistpy.observables.sigma_los_aperture(source, aperture)

Return the velocity dispersion inside an aperture.

Given a model, only the projected radii the aperture covers are computed.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable.

aperture (Aperture) – The aperture, built with Aperture.circle or Aperture.rectangle,
its lengths in the model’s unit system (parsecs in the physical
system).

Returns:

sigma_ape – Velocity dispersion inside the aperture, in the model’s unit system.

Return type:

float

Notes

Given a model, the second moment of the line-of-sight dispersion averaged
over the aperture with the surface density (aperture_dispersion), which
reads no velocity distribution. Given a LosTable, the second moment of
the aperture’s velocity distribution (aperture_losvd) by its own
weights. The two agree to the accuracy goal (validate_aperture); a
distribution above the goal, or read from a table above it, says so in a
warning.

Of a superposition, the components are summed with their weights under
the aperture: given a model, the surface-density weighted sums of
\(\sigma_{\rm los}^2\) and of the surface density, each component’s
own, on the aperture’s nodes of the first component (the components
share one density), mixed as for sigma_los; given a LosTable, the
second moment of the aperture distributions summed with the weights
times each component’s projected mass inside the aperture, each
distribution checked against its own component’s aperture dispersion.

pianistpy.observables.sigma_los(source, r_2d)

Return the line-of-sight velocity dispersion at projected radii.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable.

r_2d (array-like) – Projected radius \(R_{\rm 2D}\) in the model’s unit system
(Model.to): internal units by default, parsecs (a float or an
Astropy quantity) in the physical system.

Returns:

sigma_los – Line-of-sight velocity dispersion in the model’s unit system, with
the shape of r_2d.

Return type:

array

Notes

Of a superposition, \(\sigma_{\rm los}^2 = \sum_k w_k \Sigma_k
\sigma_{{\rm los},k}^2 / \sum_k w_k \Sigma_k\), each component read from
its own table; zero beyond the last row.

pianistpy.observables.sigma_r(source, r)

Return the radial velocity dispersion at radii.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable, whose intrinsic second
moments are read.

r (array-like) – Radius \(r\) in the model’s unit system (Model.to): internal
units by default, parsecs (a float or an Astropy quantity) in the
physical system.

Returns:

sigma_r – Radial velocity dispersion \(\sigma_r\) in the model’s unit
system, with the shape of r.

Return type:

array

Notes

The reading between the grid points of the intrinsic profile is that of
DispersionProfile.read. Of a superposition, whose components share one
density, \(\sigma_r^2 = \sum_k w_k \sigma_{r,k}^2\).

pianistpy.observables.sigma_t(source, r)

Return the tangential velocity dispersion at radii.

Parameters:

source (Model or LosTable) – The model, or its prepared LosTable.

r (array-like) – Radius \(r\) in the model’s unit system, as for sigma_r.

Returns:

sigma_t – Tangential velocity dispersion of both tangential components
together, \(\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2
= 2 (1 - \beta_\sigma) \sigma_r^2\), in the model’s unit system, with
the shape of r.

Return type:

array

Notes

Read as sigma_r is; of a superposition, \(\sigma_{\rm t}^2 =
\sum_k w_k \sigma_{{\rm t},k}^2\).

     On this page
