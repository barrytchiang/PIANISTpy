# End-to-end differentiability

Source: https://pianistpy.com/docs/guide/gradient.html

# End-to-end differentiability

pianistpy.inference.likelihood(), pianistpy.df.gradient.make_inversion_table,
pianistpy.df.gradient.table_jacobian, pianistpy.df.gradient.make_inversion_rows,
pianistpy.df.gradient.rows_jacobian

## What is differentiable

The weights and the unit scales. The likelihood of a model against data, pianistpy.inference.likelihood(),
is a JAX function of the components’ weights and of the model’s mass and length scales, traceable and differentiable
in both. With free=, the anisotropy (beta_c, R_a) of the components named there is a parameter of it as well, for
a component on a truncated profile with a finite R_a; the inference page states how.
The two calls serve different members: the rule below serves an untruncated analytic component, while likelihood
refuses a member in free= that is not spatially finite, by name, since it reads the distribution function at
the model’s edge.

The observables. pianistpy.Differentiable serves a model’s observables as JAX functions of its
anisotropy (beta_c, R_a), each with the public call’s own value and the tangent rows’ derivative: Differentiable observables.

The anisotropy parameters. The distribution function of a component is differentiable in its central anisotropy
beta_c and its anisotropy radius R_a, for the families served:

constant anisotropy (beta_c),

Cuddeford Type I (beta_c and R_a),

Osipkov-Merritt Type I and Type II (R_a).

Two levels are offered. The inversion’s rows (the distribution function on the inversion’s energy grid) through
make_inversion_rows and rows_jacobian; the distribution function’s table, as the lookup reads it, through
make_inversion_table and table_jacobian. The table’s derivative covers every part of it: the rows, the nodes
below the written floor, the form above the anchor at the center and the tail at the outer end. On a component on
the truncated profile of another model (Method II), the table’s parts are the rows, the edge law below the floor and
the center’s stated form; the rows’ functions serve a component on its own analytic profile only.

make_inversion_rows and make_inversion_table return a JAX function of the parameter vector with a custom
derivative: its value is the package’s own solve, computed on the host, and its derivative is the rule. The
*_jacobian functions return the values and their derivatives as arrays. In both, the energy grid and every
discrete choice of the solve (the grid’s placement, the inversion’s form, the anchor, the nodes kept) are held at the
parameters the function was made at: the derivative is that of the solve with these choices fixed, which is the
derivative of the table the package returns there. Where a choice changes between two parameters, the solve at the
second one differs from the held one by its own tolerance.

Not served, each refused with an error that names it: Cuddeford Type II, a center on its critical branch (where
the branch itself changes with beta_c), and a component on a truncated profile that carries no distribution function
of its model.

Cuddeford Type II. The inversion’s kernel carries the factor (P - P(R_a))^s, with s = 1 - beta_c - j and j the
inversion’s order: its interior point P(R_a) moves with R_a and its exponent with beta_c. The derivative of that
factor in R_a is not integrable at the interior point for s at or below 0, so the rows of the mixed derivative do not
carry the derivative there, and the rule refuses the family for both parameters, naming the interior point. A Cuddeford
Type II model’s second moments take their derivative by the Jeans path (dispersion_profile’s tangents); its
distribution function, and the observables read from it (losvd, aperture_losvd, pdf_v), take none.

Where a model’s anisotropy is a parameter. The derivative of a model’s observables in (beta_c, R_a), through
likelihood with free= and through pp.Differentiable, is the full derivative for a Method II model: a
component whose distribution function is inverted on another model’s truncated profile, held fixed, so its profile
and its potential do not move with the anisotropy. A Method I truncated model is refused by name: its truncated
profile is that of its own distribution function, which moves with beta_c and R_a, so neither the Jeans path (a fixed
profile) nor the distribution function’s path at a fixed potential carries the full derivative, and the profile’s
own motion through the truncation is not served in this version. An untruncated model is refused on both paths,
since its edge terms read its distribution function.

On a GPU, the derivative of losvd and aperture_losvd through pp.Differentiable may refuse a row of
the line-of-sight table by name (a ResultError: the row recomputed level by level is not the row): its transpose
recomputes each row and holds it to the forward’s row bit for bit, which a GPU’s order of summation does not
promise. It never returns a wrong number. On the CPU every row is served.

## Per call

pianistpy.inference.likelihood: differentiable in the weights, the unit scales and, with free=, the
(beta_c, R_a) of the named components; the families and refusals as above.

The distribution function (make_inversion_rows, make_inversion_table): differentiable in (beta_c, R_a)
for the families served.

Each observable is differentiable in (beta_c, R_a) through pp.Differentiable(model), whose methods mirror the
public calls as JAX functions of theta, for the families above and with the same refusals. Pending: these seven
lines hold once the object is in the package (1.0.0).

sigma_r: differentiable in (beta_c, R_a) through pp.Differentiable.

sigma_t: differentiable in (beta_c, R_a) through pp.Differentiable.

sigma_los: differentiable in (beta_c, R_a) through pp.Differentiable.

sigma_los_aperture: differentiable in (beta_c, R_a) through pp.Differentiable.

losvd: differentiable in (beta_c, R_a) through pp.Differentiable.

aperture_losvd: differentiable in (beta_c, R_a) through pp.Differentiable.

pdf_v: differentiable in (beta_c, R_a) through pp.Differentiable.

## What the record carries

The rule is exact except for one part, the derivative of the inversion’s operator itself in beta_c at a fixed
source, which has no closed form and is taken from the operator’s values at nearby beta_c. The record returned with
the derivative states how:

central (a Richardson quotient) inside a form’s region, or one-sided at a corner, where the inversion’s form
changes on one side (integer and half-integer beta_c);

the step, set by measurement: the kernel’s accuracy balanced against the quotient’s truncation term;

the quotient’s Richardson residual and, at a corner, the spread of its two sides, both over the column’s scale;

for a component on a truncated profile, where its energy grid is placed.

The residual is the derivative’s accuracy, and it reads in the package’s tiers (Accuracy). Inside a form’s
region it is at the goal. At the isotropic corner, beta_c = 0, the quotient is one-sided, and its residual is about
1e-5 of the column: the ceiling tier, not the goal, which the record states. A gradient there is usable at that
accuracy; one at the goal takes a beta_c off the corner.

## Example

The derivative of log10 F at one node of a Hernquist component’s table, by the rule and by a central difference of
the same function:

```
import jax
import numpy as np

import pianistpy as pp
from pianistpy.df.gradient import make_inversion_table

halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(beta_c=0.25))
table, params, values = make_inversion_table(halo, pp.Settings())
node = int(np.searchsorted(values["k"], 0.5))  # the node nearest K = 0.5

def log_df(theta):
    return table(theta)["log_df"][node]

theta = np.array([0.25])  # params == ("beta_c",)
rule = float(jax.grad(log_df)(theta)[0])
h = 1e-3
difference = float(log_df(theta + h) - log_df(theta - h)) / (2 * h)
print(rule, difference)

```

The two agree to the difference’s own error (h squared times the third derivative). The table is made once; each
call of table solves the inversion again at the new parameters on the held grid.

     On this page
