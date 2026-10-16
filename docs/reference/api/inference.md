# Inference (pianistpy.inference)

Source: https://pianistpy.com/docs/reference/api/inference.html

# Inference (pianistpy.inference)

The log-likelihood of a model against data.

class pianistpy.inference.Data(observable, values, r_2d=None, apertures=(), errors=None, covariance=None, unit_system='internal', v_edges=None, r=None)

Measurements of one observable, with their errors.

Parameters:

observable ({"sigma_los", "sigma_los_aperture", "losvd", "losvd_aperture", "pdf_v"}) – What was measured: the line-of-sight dispersion at projected radii,
or averaged over apertures (weighted by the surface density); or a
binned distribution of velocity: the line-of-sight velocity
distribution at one projected radius or in one aperture, or the
local speed distribution at one 3D radius.

values (array-like, shape (n,)) – The measured values. For a binned distribution, the share of the
distribution inside the window [v_edges[0], v_edges[-1]] that
falls in each bin.

r_2d (array-like, shape (n,), optional) – The projected radii, for "sigma_los"; one radius for
"losvd".

apertures (sequence of Aperture, optional) – The apertures, for "sigma_los_aperture", their lengths in the
units of the data; one aperture for "losvd_aperture".

errors (array-like, shape (n,), optional) – The standard deviations of independent errors.

covariance (array-like, shape (n, n), optional) – The covariance of the errors, in place of errors.

unit_system ({"internal", "model", "physical"}) – The units of the data: in the physical system radii in parsecs and
speeds in kilometres per second. A model is compared in the same
system.

v_edges (array-like, shape (n + 1,), optional) – The edges of the bins of a binned distribution, increasing; speeds
of "pdf_v" not negative.

r (array-like, shape (1,), optional) – The 3D radius of "pdf_v".

property identity: str

a digest of everything it holds.

Type:

The identity string of the data

write(path, overwrite=False)

Write the data to a file; pianistpy.read reads it back.

Parameters:

path (str | PathLike)

overwrite (bool)

Return type:

None

class pianistpy.inference.Likelihood(tables, blocks, weights, scale, length, velocity, r_min, r_max, free=(), theta=None, members=(), readings=(), binned=())

The log-likelihood of a model against data, as a JAX function.

Built by likelihood. Called with the weights of the components and the
scales of the units, it returns the Gaussian log-likelihood, traceable
and differentiable in both.

Variables:

tables (tuple) – The projected table of each component, read in JAX.

blocks (tuple) – Each Data, prepared.

weights (array, shape (n_components,)) – The model’s own weights (its components’ masses).

scale (array, shape (2,)) – The model’s own mass and length scales in the data’s units.

velocity (length,) – The factors of the data’s units at the model’s own scales.

r_max (r_min,) – The innermost and outermost projected radius of the tables, in
internal units.

free (tuple of int) – The components whose anisotropy (beta_c, R_a) is a parameter.

theta (array, shape (n_free, 2)) – Their own (beta_c, R_a), in internal units.

members (tuple) – Their tables at any theta, on the host.

readings (tuple) – Their sigma_los^2 readings as JAX functions of theta.

Parameters:

tables (tuple[_Table, ...])

blocks (tuple[_Block, ...])

weights (Any)

scale (Any)

length (float)

velocity (float)

r_min (float)

r_max (float)

free (tuple[int, ...])

theta (Any | None)

members (tuple[_Free, ...])

readings (tuple[Any, ...])

binned (tuple[_BinnedBlock, ...])

pianistpy.inference.likelihood(model, data, settings=None, free=())

Return the log-likelihood of a model against data, prepared.

The projected table of each component is computed here, once; the
returned function reads them in JAX. A model or data that cannot be
compared are refused here, with an exception.

Parameters:

model (Model) – A model of one component, or a superposition (superpose), in the
unit system of the data.

data (Data or sequence of Data) – The measurements; their log-likelihoods are summed.

settings (Settings, optional) – Numerical settings of the observables; the model’s by default.

free (sequence of int, optional) – The components whose anisotropy (beta_c, R_a) is a parameter of the
likelihood (its theta): each a spatially finite model on a
truncated profile (a DF inverted on its parent’s profile). An
isotropic or constant member’s beta_c is free and its R_a column is
held at its own value (infinite), with a zero gradient: the family
has no R_a.

Returns:

likelihood

Return type:

Likelihood

Notes

Each measurement is the surface-density weighted mean of
\(\sigma_{\rm los}^2\) over its nodes, sum_j q_ij Sigma(x_j)
sigma^2(x_j) / sum_j q_ij Sigma(x_j): for a projected radius one node,
for an aperture the nodes and weights of aperture_dispersion. On a
superposition each component’s table is its member’s own, on the shared
density, and the two sums are mixed by the weights (mixture): the
measurement is a ratio of two sums linear in the weights. A scale
of length other than the model’s own moves the nodes (x = R / (L l))
and keeps their weights, a change of variable to the data’s radius in
which the aperture is fixed.

A free component’s tables at a theta are built on the host when the
function first meets it: its DF on its truncated profile (refused there,
the function is minus infinity), then the two tables, their rows placed
at that theta. The gradient in theta is that of the reading on those rows
(their placement frozen); between placements the value moves within the
tables’ tolerance. An untruncated component is not free: its edge terms
read its DF.

pianistpy.inference.log_likelihood(model, data, settings=None)

Return the Gaussian log-likelihood of a model against data.

Parameters:

model (Model) – The model, in the unit system of the data.

data (Data or sequence of Data) – The measurements; their log-likelihoods are summed.

settings (Settings, optional) – Numerical settings of the observables; the model’s by default.

Returns:

log_likelihood

Return type:

float

Notes

With residuals r of the measured values against the model’s, and
errors e or a covariance C, -r^T C^-1 r / 2 - ln det(2 pi C) / 2.
The value of likelihood at the model’s own weights and scales.

     On this page
