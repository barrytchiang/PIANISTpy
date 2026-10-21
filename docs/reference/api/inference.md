# Inference

Source: https://pianistpy.com/docs/reference/api/inference

Module `pianistpy.inference`

The log-likelihood of a model against measured data.

A user stores each set of measurements in a Data and calls likelihood on
a model and the data. likelihood computes the projected tables of the
model once and returns a Likelihood, a JAX function of the component
weights, of the mass and length scales and, for the Method II components
chosen as free, of the anisotropy parameters. log_likelihood returns the value of the
log-likelihood at the model’s own weights and scales.

For measured values \(d\), model values \(m\) and the covariance
matrix \(C\) of the errors (diagonal, with \(C_{ii} = e_i^2\), for
independent errors \(e_i\)), the log-likelihood of one Data is

\[\ln \mathcal{L} = -\frac{1}{2} (d - m)^{\rm T} C^{-1} (d - m)
- \frac{1}{2} \ln \det (2 \pi C),\]

and the log-likelihoods of several Data are summed. Notes Section T:7.4
derives the likelihood.

class pianistpy.inference.Data(*observable*, *values*, *r_2d=None*, *apertures=()*, *errors=None*, *covariance=None*, *unit_system='internal'*, *v_edges=None*, *r=None*)

Measurements of one observable, with their errors.

A Data holds one set of measured values, the projected radii,
apertures or 3D radius at which the values were measured, and the
errors of the values. likelihood compares a model with one Data or
with several.

Parameters:

- **observable** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The measured quantity, one of seven observables. `"sigma_los"`
is the line-of-sight velocity dispersion \(\sigma_{\rm los}\)
at user-specified projected radii. `"sigma_los_aperture"` is
\(\sigma_{\rm los}\) averaged over each aperture with the
surface density as the weight.
`"losvd"` is the binned line-of-sight velocity distribution at one
projected radius, and `"losvd_aperture"` the binned line-of-sight
velocity distribution in one aperture. `"pdf_v"` is the binned
local velocity distribution at one 3D radius. `"h3"` and `"h4"` are
the Gauss-Hermite moments \(h_3\) and \(h_4\) of the
line-of-sight velocity distribution at user-specified projected
radii or in apertures.

- **values** (*array-like**, **shape** (**n**,**)*) – The measured values. A value of `"sigma_los"` or
`"sigma_los_aperture"` is a dispersion in the velocity unit of
unit_system. A value of `"losvd"`, `"losvd_aperture"` or
`"pdf_v"` is the fraction of the distribution inside the
user-specified range `[v_edges[0], v_edges[-1]]` that falls in one
bin. A value of `"h3"` or `"h4"` is dimensionless.

- **r_2d** (*array-like**, **shape** (**n**,**)**, **optional*) – User-specified projected radii \(R_{\rm 2D}\), positive, in the
length unit of unit_system: one projected radius per value for
`"sigma_los"`, `"h3"` and `"h4"`, and one projected radius for
`"losvd"`. Default: `None`, which means no projected radii, as
`"sigma_los_aperture"`, `"losvd_aperture"` and `"pdf_v"` take.

- **apertures** (*sequence** of *[*Aperture*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)*, **optional*) – The apertures, with lengths in the length unit of unit_system: one
aperture per value for `"sigma_los_aperture"`, `"h3"` and
`"h4"`, and one aperture for `"losvd_aperture"`. Default:
`()`, which means no apertures.

- **errors** (*array-like**, **shape** (**n**,**)**, **optional*) – The standard deviation of the error of each value, positive, for
independent errors. Exactly one of errors and covariance is
given. Default: `None`, which means covariance gives the errors.

- **covariance** (*array-like**, **shape** (**n**, **n**)**, **optional*) – The covariance matrix of the errors, symmetric. likelihood
requires the covariance matrix to be positive definite as well.
Default: `None`, which means errors gives the errors.

- **unit_system** (*{"internal"**, **"model"**, **"physical"}**, **optional*) – The unit system of the data. In the physical system a length is in
parsecs and a velocity in kilometers per second. likelihood
compares a model only in the unit system of the data. Default:
`"internal"`, which means internal units.

- **v_edges** (*array-like**, **shape** (**n + 1**,**)**, **optional*) – The bin edges of `"losvd"`, `"losvd_aperture"` or `"pdf_v"`,
increasing, in the velocity unit of unit_system. The first edge of
`"pdf_v"` is not negative. Default: `None`, which means no bins;
`"losvd"`, `"losvd_aperture"` and `"pdf_v"` require v_edges.

- **r** (*array-like**, **shape** (**1**,**)**, **optional*) – The 3D radius of `"pdf_v"`, positive, in the length unit of
unit_system. Default: `None`, which means no 3D radius;
`"pdf_v"` requires r.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If observable is not one of the seven observables, or
    unit_system is not one of the three unit systems.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If values is not a non-empty one-dimensional array.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a field that observable requires (r_2d, apertures,
    v_edges or r) is missing, has the wrong length or holds a
    radius that is not positive, or if a field that observable does
    not take is given.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If v_edges is not increasing, or the first edge of `"pdf_v"` is
    negative.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If both errors and covariance are given or neither is given, if
    an error is not positive, or if covariance is not a symmetric
    matrix of n rows and n columns.

See also

[`likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.likelihood)
The log-likelihood of a model against one or several Data.

[`log_likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.log_likelihood)
The log-likelihood at the model’s own weights and scales.

`Aperture`
An aperture of `"sigma_los_aperture"`, `"losvd_aperture"`, `"h3"` or `"h4"`.

Examples

```python
>>> import pianistpy as pp
>>> data = pp.Data(
...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
... )
>>> data.values.shape
(2,)
>>> data.unit_system
'internal'
```

property identity: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

The identity string of the data, a digest of every field of the data.

write(*path*, *overwrite=False*)

Write the data to an HDF5 file, which pianistpy.read reads back.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – The path of the HDF5 file to write.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default:
`False`, which means an existing file at path raises
InputError.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the written file does not read back as the data.

Return type:

None

See also

`pianistpy.read`
Read a result file, a file of Data included.

class pianistpy.inference.Likelihood(*tables*, *blocks*, *weights*, *scale*, *length*, *velocity*, *r_min*, *r_max*, *free=()*, *theta=None*, *members=()*, *readings=()*, *binned=()*)

The log-likelihood of a model against data, as a JAX function.

likelihood builds a Likelihood. A call of a Likelihood with the
component weights, the mass and length scales and, for free components,
the anisotropy parameters returns the Gaussian log-likelihood. JAX can
compile the call and differentiate the call in each argument.

Variables:

- **tables** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – The projected table of each component, read in JAX.

- **blocks** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Each Data of dispersions or Gauss-Hermite moments, prepared.

- **weights** (*array**, **shape** (**n_components**,**)*) – The model’s own component weights, which are the masses of the
components.

- **scale** (*array**, **shape** (**2**,**)*) – The model’s own mass scale and length scale, in the units of the
data.

- **length** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The length factor \(L\) of the unit system of the data at the
model’s own scales.

- **velocity** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The velocity factor \(V\) of the unit system of the data at the
model’s own scales.

- **r_min** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The innermost projected radius of the tables, in internal units.

- **r_max** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The outermost projected radius of the tables, in internal units.

- **free** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*int*](https://docs.python.org/3/builtins/functions.html#int)) – The indices of the components whose anisotropy parameters
(\(\beta_{\rm c}\), \(R_{\rm a}\)) are parameters of the
likelihood.

- **theta** (*array**, **shape** (**n_free**, **2**)*) – The model’s own (\(\beta_{\rm c}\), \(R_{\rm a}\)) of each
free component, in internal units.

- **members** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – The free components, each with the tables computed at every
anisotropy at which the Likelihood has been called.

- **readings** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – The squared line-of-sight dispersion \(\sigma_{\rm los}^2\) of
each free component, as a JAX function of the component’s
anisotropy parameters.

- **binned** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Each Data of `"losvd"`, `"losvd_aperture"` or `"pdf_v"`,
prepared.

Parameters:

- **tables** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[**_Table**, **...**]*)

- **blocks** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[**_Block** | **_Moments** | **_PointMoments**, **...**]*)

- **weights** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **scale** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **length** ([*float*](https://docs.python.org/3/builtins/functions.html#float))

- **velocity** ([*float*](https://docs.python.org/3/builtins/functions.html#float))

- **r_min** ([*float*](https://docs.python.org/3/builtins/functions.html#float))

- **r_max** ([*float*](https://docs.python.org/3/builtins/functions.html#float))

- **free** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[*[*int*](https://docs.python.org/3/builtins/functions.html#int)*, **...**]*)

- **theta** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any)* | **None*)

- **members** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[**_Free**, **...**]*)

- **readings** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any)*, **...**]*)

- **binned** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*[**_BinnedBlock**, **...**]*)

See also

[`likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.likelihood)
Build the Likelihood of a model against data.

[`log_likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.log_likelihood)
The log-likelihood at the model’s own weights and scales.

[`Data`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)
The measurements a Likelihood compares with.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> data = pp.Data(
...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
... )
>>> like = pp.likelihood(model, data)
>>> bool(float(like()) > float("-inf"))
True
>>> float(like(scale=[1.0, -1.0]))
-inf
```

pianistpy.inference.likelihood(*model*, *data*, *settings=None*, *free=()*)

Return the log-likelihood of a model against data, as a JAX function.

likelihood computes the projected table of each component once and
returns a Likelihood, which reads the tables in JAX at every call. A
model and data that cannot be compared raise an exception in
likelihood, never in a call of the returned Likelihood. For one
value at the model’s own weights and scales, call log_likelihood.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – A model of one component, or a superposition (superpose), in the
unit system of the data.

- **data** ([*Data*](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)* or **sequence** of *[*Data*](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)) – The measurements. The log-likelihoods of several Data are summed.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – The numerical settings of the observables. Default: `None`, which
means the model’s own settings.

- **free** (*sequence** of *[*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – The indices of the components whose anisotropy parameters
(\(\beta_{\rm c}\), \(R_{\rm a}\)) become the parameter
`theta` of the returned Likelihood. Each free component is a
spatially finite Method II model: a DF computed on the truncated
profile of another model, where the truncated profile carries the
DF of the other model.
For an isotropic or constant-anisotropy component,
\(\beta_{\rm c}\) is free and the \(R_{\rm a}\) column stays
at the component’s own value, infinity, with a zero gradient,
because the isotropic and constant-anisotropy models have no
anisotropy radius. Default: `()`, which means no anisotropy
parameter is a parameter of the likelihood.

Returns:

**likelihood** – The log-likelihood as a JAX function of the component weights, the
mass and length scales and, for the components in free, the
anisotropy parameters. A call of the Likelihood without arguments
returns the value at the model’s own weights and scales, the value
log_likelihood returns.

Return type:

[Likelihood](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Likelihood)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is not a Model, or model has several components and
    is not a superposition.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If data is empty or holds an object that is not a Data.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the data are in several unit systems, or in a unit system other
    than the unit system of model (Model.to presents a model in
    another unit system).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If free repeats a component index or lists a component index that
    model does not have.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a free component is not spatially finite, is not a DF on a
    truncated profile that carries the DF of another model, or has an
    anisotropy model with neither a \(\beta_{\rm c}\) nor an
    \(R_{\rm a}\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If free is given together with binned distributions or
    Gauss-Hermite moments.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the projected tables of the components differ in the innermost
    tabulated radius, the outermost tabulated radius or the edge radius.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a projected radius of `"sigma_los"` data lies inside the
    innermost or beyond the outermost projected radius of the model’s
    table. The likelihood does not extrapolate the table.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If Gauss-Hermite moments at projected radii are given on a
    superposition, or a projected radius of the moments lies outside the
    projected radii of the model’s table of moments. A superposition
    takes Gauss-Hermite moments in apertures only.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If an aperture encloses no tabulated surface density, or if the disk
    inside the innermost tabulated projected radius carries more than
    \(10^{-6}\) of the aperture’s weight.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the 3D radius of `"pdf_v"` data lies outside the profile or,
    for a Type II model, at or beyond the anisotropy radius.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the error of a component’s accuracy check exceeds
    \(10^{-2}\) and the component’s settings do not set
    `write_anyway=True`.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the projection gives a negative squared line-of-sight dispersion.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a table of velocity distributions fails the table’s own
    consistency check (for binned distributions and Gauss-Hermite
    moments).

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If a parallel process ends while a table of velocity distributions
    is computed (for binned distributions and Gauss-Hermite moments).

See also

[`log_likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.log_likelihood)
The log-likelihood at the model’s own weights and scales.

[`Likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Likelihood)
The JAX function likelihood returns.

[`Data`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)
The measurements a model is compared with.

`superpose`
A superposition of models that share one density.

Notes

Each dispersion measurement \(i\) is the mean of
\(\sigma_{\rm los}^2\) over the measurement’s nodes \(x_j\),
weighted by the node weights \(q_{ij}\) and the surface density
\(\Sigma\),

\[\sigma_i^2 = \frac{\sum_j q_{ij}\, \Sigma(x_j)\,
\sigma_{\rm los}^2(x_j)}{\sum_j q_{ij}\, \Sigma(x_j)}.\]

A projected radius has one node, the projected radius itself. An
aperture has the nodes and weights of sigma_los_aperture. On a
superposition, each component’s table is the component’s own table on
the shared density, and the component weights mix the numerator and
the denominator of \(\sigma_i^2\) (mixture), so each measurement
is a ratio of two sums linear in the weights. A length scale \(l\)
other than the model’s own length scale moves each node to
\(x_j = R_j / (L\,l)\) and keeps the node weights \(q_{ij}\):
the move is a change of variable to the projected radius of the data,
in which the aperture is fixed.

A free component’s tables at an anisotropy \(\theta\) are computed
at the first call of the Likelihood at \(\theta\): first the
component’s DF on the component’s truncated profile, then the
component’s two projected tables, with the tabulated radii placed for
\(\theta\). Where the DF cannot be built at \(\theta\), the
Likelihood returns minus infinity. The gradient in \(\theta\) is
the gradient of the reading at the placed radii, with the placed radii
held fixed. Between two placements of the radii, the value changes by
no more than the tolerance of the tables. An untruncated component
cannot be free, because the edge terms of an untruncated component read
the component’s DF.

A Gauss-Hermite moment (`"h3"` or `"h4"`) in an aperture is the
moment of the mixture’s velocity distribution in the aperture. The
mixture’s distribution is the sum of the components’ distributions
(aperture_losvd), each weighted by the component weight and the
component’s projected mass, and the moment is taken about the
mixture’s own mean velocity, in units of the mixture’s own dispersion.
A moment in an aperture is differentiable in the weights and does not
depend on the mass scale. The distributions in apertures are computed
at the model’s own radii, so the length scale stays at the model’s own
length scale. A Gauss-Hermite moment at a projected radius is read on a
model of one component from a table of the model’s moments over
projected radius, at the projected radius that the length scale moves,
so a moment at a projected radius is also differentiable in the length
scale. Neither kind of Gauss-Hermite moment is taken together with a
free component.

Each call of likelihood computes the projected table of every
component. For many evaluations on one model, call likelihood once
and call the returned Likelihood many times.

References

Notes Section T:7.4.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> data = pp.Data(
...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
... )
>>> like = pp.likelihood(model, data)
>>> like.weights.shape
(1,)
>>> bool(float(like()) > float("-inf"))
True
```

pianistpy.inference.log_likelihood(*model*, *data*, *settings=None*)

Return the Gaussian log-likelihood of a model against data.

log_likelihood returns one number, the value of likelihood at the
model’s own component weights and scales. For many evaluations, or for
gradients, call likelihood once and call the returned Likelihood.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – A model of one component, or a superposition (superpose), in the
unit system of the data.

- **data** ([*Data*](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)* or **sequence** of *[*Data*](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)) – The measurements. The log-likelihoods of several Data are summed.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – The numerical settings of the observables. Default: `None`, which
means the model’s own settings.

Returns:

**log_likelihood** – The Gaussian log-likelihood at the model’s own component weights
and scales, summed over the data. The value is minus infinity where
the nodes of a measurement carry no surface density or where a
projected radius of the data lies inside the innermost tabulated
projected radius.

Return type:

[float](https://docs.python.org/3/builtins/functions.html#float)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model and data cannot be compared, under each condition the
    Raises section of likelihood lists for InputError.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the error of a component’s accuracy check exceeds
    \(10^{-2}\) and the component’s settings do not set
    `write_anyway=True`, or if the projection gives a negative squared
    line-of-sight dispersion.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a table of velocity distributions fails the table’s own
    consistency check (for binned distributions and Gauss-Hermite
    moments).

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If a parallel process ends while a table of velocity distributions
    is computed (for binned distributions and Gauss-Hermite moments).

See also

[`likelihood`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.likelihood)
The log-likelihood as a JAX function of the parameters.

[`Data`](https://pianistpy.com/docs/reference/api/inference#pianistpy.inference.Data)
The measurements a model is compared with.

Notes

For residuals \(r\) of the measured values minus the model’s values
and the covariance matrix \(C\) of the errors (diagonal, with
\(C_{ii} = e_i^2\), for independent errors \(e_i\)),

\[\ln \mathcal{L} = -\frac{1}{2} r^{\rm T} C^{-1} r
- \frac{1}{2} \ln \det (2 \pi C).\]

References

Notes Section T:7.4.

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.inference import log_likelihood
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> data = pp.Data(
...     "sigma_los", [0.30, 0.25], r_2d=[0.5, 1.0], errors=[0.01, 0.01]
... )
>>> value = log_likelihood(model, data)
>>> isinstance(value, float)
True
```
