# Superposition

Source: https://pianistpy.com/docs/reference/api/superposition

# Superposition

Module pianistpy.superposition

Linear superposition of models that share one density.

pianistpy.superposition.mixture(weights, values, xp=np)

Return the weighted sum of per-component values, the weights normalized.

The one way a superposed model combines its components: a density, a
moment or a table read per component. Linear in the values, and in the
weights up to their normalization; with xp=jax.numpy the weights may
be traced, so a quantity of the mixture is differentiable in them.

Parameters:

weights (array-like, shape (n_components,)) – Positive weights; normalized here.

values (array-like, shape (n_components, ...)) – The value of each component.

xp (module, optional) – numpy, or jax.numpy for traced weights.

Returns:

value

Return type:

array, shape (…)

pianistpy.superposition.moments(model, r, n_nodes=None)

Return the second moments of a superposed model at radii r.

Parameters:

model (Model) – A superposed model (superpose), its components sharing one density.

r (array-like, shape (n,)) – Radii, in internal units.

n_nodes (int, optional) – Gauss-Legendre nodes of each component’s Jeans integral; the model’s
observables.n_jeans_nodes if omitted.

Returns:

moments – sigma_r2 and sigma_t2 of the mixture, beta_sigma its
anisotropy, and sigma_r2_components and beta_components (one
row per component), with the shape of r.

Return type:

dict

Notes

Each component’s radial dispersion is the Jeans integral of its
auxiliary density on the shared profile (radial_dispersion, the
Gauss-Legendre rule in ln(1 - K / P_0)); its components share the density,
so the moments of the mixture are the weighted sums
\(\sigma_r^2 = \sum_k w_k\sigma_{r,k}^2\),
\(\sigma_t^2 = 2\sum_k w_k(1-\beta_k)\sigma_{r,k}^2\) and
\(\beta_\sigma = 1 - \sigma_t^2 / (2\sigma_r^2)\).

References

Chiang+(2026d), Eqs. (B42), (B43) [eqn:linearity_2, eqn:linearity_beta_1] and
eqn:beta_sigma_sup.

pianistpy.superposition.superpose(members)

Return the superposition of models that share one truncated pair.

Parameters:

members (mapping of Model to float) – Two or more models of one component each, every one built on the same
truncated profile (Method II children of one truncated parent), with
their weights. The weights must be positive; they are normalized to
sum to one, with a warning where they did not.

Returns:

model – One component per member, in the order given: the member’s component
with its mass the weight relative to the first member’s; its DF table
on its own energy grid (DfTable carries one grid per component); the
shared profile with the density split by weight; the members’ checks,
each at its worst over the members.

Return type:

Model

Raises:

InputError – If fewer than two members are given, a weight is not positive and
    finite, a member is not a single Method II component on a truncated
    profile, the members are built on different profiles, or their
    settings or units differ.

     On this page
