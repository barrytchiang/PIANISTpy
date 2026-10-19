# Settings tables

Source: https://pianistpy.com/docs/reference/settings

# Settings tables

Every numerical setting of the package is a declared field of one group, with its value in each tested tier and
one line on what it controls. The tables below are generated from the package’s own declaration at every build
of this manual, so they cannot lag the code. The settings container is pianistpy.Settings; a stage reads
its group and nothing else; an override of an undeclared field, of the wrong type or not positive is refused.

## inversion

Field

Default

Quick

What it controls

n_nodes

500

same

Gauss-Legendre nodes of the inversion integral.

n_grid_rows

1000

same

Rows of the linear middle zone of the inversion’s energy grid.

n_grid_tail_rows

3000

same

Rows of each logarithmic end zone of the inversion’s energy grid, toward the floor and toward the top.

grid_tail_extent

0.03

same

Extent of the energy grid’s end zones, as a fraction of the central potential.

table_nodes_per_decade

50

same

Nodes per decade of the energy grid below its floor, down to the table’s floor.

n_floor_nodes

240

same

Gauss-Legendre nodes in ln P of the completion below the table’s floor (a third of them, at least 40, near the floor).

## truncation

Field

Default

Quick

What it controls

n_source_nodes

500

same

Gauss-Legendre nodes of each of the two intervals of the quadrature of the auxiliary density on the source grid.

n_source_rows

12000

same

Rows of the source grid in the potential.

n_source_tail_rows

3000

same

Rows of each logarithmic tail of the source grid, toward the floor and toward the top (at most a third of the rows).

source_tail_extent

0.03

same

Largest extent of the source grid’s tails, as a fraction of the grid’s top (the tails take at most a hundredth of the top).

source_tail_floor

1.20401e-08

same

Smallest start of the source grid’s lower tail and smallest extent of its upper tail, as fractions of the grid’s top.

quadrature_split

0.9

same

Fraction of the potential where the first interval of the source quadrature ends.

quadrature_stretch

2

same

Power k of the second interval’s map Q = P - u^k, where the kernel’s exponent is not negative.

ode_rtol

1e-11

same

Relative tolerance of the Poisson integration.

ode_atol

1e-11

same

Absolute tolerance of the Poisson integration.

n_rows_inner

500

same

Rows of the Poisson solve’s grid inside the inner zone.

n_rows_outer

59500

same

Rows of the Poisson solve’s grid outside the inner zone.

log10_r_min

-9

same

Logarithm of the Poisson solve’s innermost radius.

log10_r_max

12

same

Logarithm of the Poisson solve’s outermost radius; a run raises it until the potential there is below half the floor, and the result states the value it used.

continuation_start

0.01

same

Fraction of the central potential below which a truncated model’s last rows are re-solved on a grid geometric in R_t - r.

continuation_rows_per_decade

1200

same

Rows per decade of R_t - r of the edge continuation.

continuation_floor_fraction

0.02

same

The edge continuation’s last row, as a fraction of the floor.

profile_bin_decades

0.1

same

Width in decades of the potential of the profile’s row-count bins.

profile_rows_per_decade

1000

same

Fewest rows per decade of the potential (and of the depth) that the profile keeps at its two ends; the fills supply the rest.

## sampling

Field

Default

Quick

What it controls

n_rows

512

same

Rows in potential of the speed table to start from: the rows are refined until the table’s own error across rows is under the goal.

n_rows_max

4075

same

Most rows of the speed table: no refinement passes this count.

n_nodes

2048

512

Nodes of each row of the speed table.

n_speed_intervals

200

same

Intervals in the scaled speed of the distribution integrated on each row of the speed table.

n_gauss_legendre

16

8

Gauss-Legendre nodes of each panel of the speed integrals.

panels_per_decade

8

same

Panels per decade of the local variable where the speed integrals refine toward the two ends of a row and inside each interval.

speed_low_refined

1e-09

same

Smallest scaled speed refined in panels; below it the density is its power law in the speed, integrated in closed form.

edge_refined

1e-12

same

Distance from the edge of the speed support, relative to the edge, down to which panels refine; below it the DF’s rule below its table is integrated in closed form.

radius_edge_refined

1e-10

same

Smallest enclosed-mass fraction beyond a radius that the radius table of a model with a sharp edge resolves in nodes; beyond it the mass follows its power law in the distance from the edge.

n_radius_nodes

2048

same

Nodes of the radius table.

n_angle_nodes

1024

same

Nodes of each half of the angle table, uniform in the logarithm of the cumulative probability from its end.

draw_block

262144

same

Particles drawn at once: a draw runs over blocks of this many, so its working memory is one block’s (about 0.75 kB a particle, 2**18 particles about 0.2 GB) above its outputs, whatever the count; the particles are the same at any block size.

row_error_goal

1e-06

0.0001

Error across rows at which the speed table’s refinement stops (its rows double until the table’s own error is under it, or n_rows_max is reached).

tail_mass_min

1e-18

same

Smallest tabulated cumulative probability at either end of a table: a uniform double reaches 2**-53, so a draw never reads beyond it.

## observables

Field

Default

Quick

What it controls

n_radii

10000

500

Projected radii of the projection table before its certificate adds rows, logarithmic from the innermost row to the edge.

profile_tolerance

1e-07

1e-06

Relative error of sigma_r at which the radial dispersion’s rows stop being added: every interval’s midpoint reads within it. A tenth of the goal, the profile’s share of the error of what is projected from it.

n_profile_start

100

same

Logarithmic radii the radial dispersion starts from, from the innermost row to the edge, before rows are added to its tolerance.

n_jeans_nodes

500

same

Gauss-Legendre nodes of a row’s own Jeans integral of the radial dispersion (the superposition’s); the observables’ profile sums panels between its rows instead, their nodes doubled until sigma_r moves by less than profile_tolerance.

n_projection_nodes

500

same

Gauss-Legendre nodes of each line-of-sight integral of the surface density and the projected dispersion.

n_aperture_nodes

100

same

Gauss-Legendre nodes of each panel of an aperture’s radial integral, at least.

aperture_nodes_per_decade

8

same

Nodes per decade of a panel’s span in the logarithm of the projected radius where that exceeds the minimum, rounded up to a multiple of 20.

rows_route

kernel

same

The angle rule of the line-of-sight velocity distributions’ rows: “kernel”, the JAX kernel, or “numpy”, the reference route.

n_speed_angle_nodes

16

same

Gauss-Legendre nodes of each panel of the angle integral of the local speed distribution (pdf_v), in t = cos(eta).

n_speed_angle_panels

16

8

Panels of that angle integral graded toward each end of each of its intervals, and of the speed integral of its norm and second moment toward each of their breaks.

speed_tolerance

1e-06

0.0001

Bound on the local speed distribution’s change when its angle rule’s nodes double: they double, at most three times, until the change is under it.

## losvd

Field

Default

Quick

What it controls

rows_per_decade

55

20

Base rows of the table of line-of-sight velocity distributions per decade of projected radius, from the innermost row of the profile.

n_energy_nodes

8

same

Gauss-Legendre nodes of each panel of the energy integral of a node.

n_floor_layers

4

2

Panels of the energy integral graded toward the energy of the DF’s floor.

energy_layer_ratio

0.2

same

Ratio of successive graded panels of the energy integral.

n_angle_nodes

64

same

Gauss-Legendre nodes of the angle integral of an L factor, the most a node takes.

n_sightline_nodes

12

8

Gauss-Legendre nodes of each panel of the line-of-sight integral.

n_sightline_panels

8

same

Panels of each interval of the support along the line of sight.

sightline_panels_refined

6

same

Doublings of the line-of-sight panels allowed before a row is recorded as not converged.

sightline_layer_ratio

0.3

same

Ratio of the line-of-sight panels graded toward the axis.

n_sightline_layers

4

same

Line-of-sight panels graded toward the axis added at each doubling.

n_crossing_scan

2001

same

Points of the scan for where the angular momentum vanishes.

n_cumulative_nodes

200001

same

Nodes of each of the three branches of the table of the DF’s cumulative integral.

n_velocity_nodes

16

8

Gauss-Legendre nodes of each velocity sub-panel of a row.

velocity_layer_ratio

0.3

same

Ratio of the velocity sub-panels graded toward a break.

n_velocity_layers_zero

12

same

Velocity sub-panels graded toward zero velocity.

n_velocity_layers_break

8

same

Velocity sub-panels graded toward each break of a row.

n_break_scan

4001

1001

Points of the coarse scan for a velocity break.

n_break_zooms

4

same

Zooms of the scan for a velocity break.

n_bisections

60

same

Bisections of an edge of the support or of a root along a line of sight.

n_radius_break_layers

16

same

Rows added toward each radius where the layout of the rows changes.

n_radius_break_bisections

40

same

Bisections in projected radius of a change of the layout.

n_radius_break_samples

3

same

Samples inside each interval of base rows for a change of the layout.

n_edge_samples

10

same

Samples beyond the last base row toward the end of the profile.

n_edge_layers

16

same

Rows added toward the end of the table.

n_stencil_rows

8

same

Rows of the stencil that reads the table between its rows.

n_stencil_test_rows

8

same

Rows of the stencil whose estimate certifies the table.

n_block_elements

25000

same

Elements of one block of node pairs, energy nodes and angle nodes, which bounds the memory of a row.

innermost_read_radius

0

same

The smallest projected radius the table will be read at, in the model’s internal length unit: the rows start at the last base row inside the radius whose disk holds the accuracy goal times the projected mass inside this one, and a read inside the first row is refused. 0 keeps the rows from the profile’s innermost row, the full-depth table.

outer_mass_share

1e-06

same

Share of the total projected mass that may lie beyond the table’s last row: the rows end at the first base row beyond the radius outside which this share lies, and a read beyond the last row is refused. 0 keeps the rows to the model’s edge.

Tiers: default, quick. A field is overridden one at a time, pp.Settings(inversion={'n_nodes': 800}); the result then records that it was made with custom settings and names the fields that differ.

     On this page
