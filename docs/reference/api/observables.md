# Observables

Source: https://pianistpy.com/docs/reference/api/observables

Module `pianistpy.observables`

Velocity dispersions and velocity distributions of a model.

The functions of pianistpy.observables return the radial, tangential and
line-of-sight velocity dispersions, the line-of-sight velocity distribution,
and the local velocity distribution of a model, at user-specified radii or in a
user-specified aperture on the plane of the sky.

class pianistpy.observables.Aperture(*shape*, *r_off=0.0*, *r_ape=None*, *side_x=None*, *side_y=None*, *angle=0.0*)

An aperture on the plane of the sky, circular or rectangular.

The constructors Aperture.circle and Aperture.rectangle build an
aperture. The lengths of an aperture are in the unit system of the
model or LosTable that the aperture is used with. In the internal and
the model systems, the unit of length is the scale radius \(r_0\).
In the physical system, the unit of length is the parsec.

Parameters:

- **shape** (*{"circle"**, **"rectangle"}*) – Shape of the aperture. `"circle"` is a circle of radius r_ape.
`"rectangle"` is a rectangle of full sides side_x and side_y.

- **r_off** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Projected distance \(R_{\rm off}\) of the center of the aperture
from the center of the model. r_off must be zero or positive.
Default: `0.0`, which means the aperture is centered on the model.

- **r_ape** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Radius \(R_{\rm ape}\) of a circle. r_ape must be positive for
a circle and must be `None` for a rectangle. Default: `None`.

- **side_x** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Full length of the rectangle’s side along the rectangle’s x axis.
side_x must be positive for a rectangle and must be `None` for a
circle. Default: `None`.

- **side_y** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Full length of the rectangle’s side along the rectangle’s y axis.
side_y must be positive for a rectangle and must be `None` for a
circle. Default: `None`.

- **angle** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Angle, in radians, between the rectangle’s x axis and the line
through the center of the model and the center of the rectangle.
Default: `0.0`, which means the rectangle’s x axis lies on the
line through the center of the model and the center of the
rectangle. A circle accepts only `0.0`.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If shape is neither `"circle"` nor `"rectangle"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_off is negative.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a length the shape requires is missing or not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a length the shape does not use is given, or a circle is given a
    nonzero angle.

See also

[`sigma_los_aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los_aperture)
Line-of-sight velocity dispersion inside an aperture.

[`aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.aperture_losvd)
Line-of-sight velocity distribution inside an aperture.

Examples

```python
>>> Aperture.circle(1.0, r_off=10.0).shape
'circle'
>>> Aperture.rectangle(2.0, 0.5, r_off=10.0, angle=0.3).side_y
0.5
```

classmethod circle(*r_ape*, *r_off=0.0*)

Return a circular aperture of radius r_ape at offset r_off.

Parameters:

- **r_ape** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Radius \(R_{\rm ape}\) of the circle. r_ape must be
positive.

- **r_off** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Projected distance \(R_{\rm off}\) of the circle’s center
from the center of the model. r_off must be zero or positive.
Default: `0.0`, which means the circle is centered on the
model.

Returns:

**aperture** – The circular aperture. The lengths r_ape and r_off are in
the unit system of the model that the aperture is used with.

Return type:

[Aperture](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_ape is not positive or r_off is negative.

See also

[`Aperture.rectangle`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.rectangle)
A rectangular aperture.

classmethod rectangle(*side_x*, *side_y*, *r_off=0.0*, *angle=0.0*)

Return a rectangular aperture of full sides side_x by side_y.

Parameters:

- **side_x** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Full length of the rectangle’s side along the rectangle’s x
axis. side_x must be positive.

- **side_y** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Full length of the rectangle’s side along the rectangle’s y
axis. side_y must be positive.

- **r_off** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Projected distance \(R_{\rm off}\) of the rectangle’s center
from the center of the model. r_off must be zero or positive.
Default: `0.0`, which means the rectangle is centered on the
model.

- **angle** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Angle, in radians, between the rectangle’s x axis and the line
through the center of the model and the center of the rectangle.
Default: `0.0`, which means the rectangle’s x axis lies on the
line through the center of the model and the center of the
rectangle.

Returns:

**aperture** – The rectangular aperture. The lengths side_x, side_y and
r_off are in the unit system of the model that the aperture is
used with.

Return type:

[Aperture](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If side_x or side_y is not positive, or r_off is negative.

See also

[`Aperture.circle`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.circle)
A circular aperture.

arc(*r*)

Return the angle, in radians, of the circle of radius r inside
the aperture.

The circle of radius r is centered on the model. The angle
\({\rm arc}(r)\) is \(2\pi\) where the whole circle lies
inside the aperture. The angle is zero where no part of the circle
lies inside the aperture. The integral over the aperture of a
quantity \(f(R)\) of projected radius is
\(\int f(R)\,{\rm arc}(R)\,R\,{\rm d}R\).

Parameters:

**r** (*array-like*) – User-specified projected radii, zero or positive, in the unit of
length of the aperture.

Returns:

**arc** – The angle \({\rm arc}(r)\) in radians, from zero to
\(2\pi\), with the shape of r.

Return type:

array

See also

[`Aperture.breaks`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.breaks)
The projected radii at which \({\rm arc}(r)\) is not smooth.

Notes

For a circle of radius \(R_{\rm ape}\) at offset
\(R_{\rm off} > 0\), the angle is \(2\pi\) for
\(r \le R_{\rm ape} - R_{\rm off}\), zero for
\(r < |R_{\rm off} - R_{\rm ape}|\) or
\(r > R_{\rm off} + R_{\rm ape}\), and otherwise

\[{\rm arc}(r) = 4 \arcsin\sqrt{x/2}, \qquad
x = \frac{R_{\rm ape}^2 - (r - R_{\rm off})^2}{2 r R_{\rm off}}.\]

For a centered circle (\(R_{\rm off} = 0\)), the angle is
\(2\pi\) for \(r \le R_{\rm ape}\) and zero otherwise.

For a rectangle, each of the rectangle’s four edge lines crosses
the circle of radius r at angles given in closed form. These
crossing angles cut the circle into arcs. Each arc lies wholly
inside or wholly outside the rectangle. The angle
\({\rm arc}(r)\) is the sum of the arcs inside the rectangle.
The same closed form holds for a rectangle that contains the center
of the model, a rectangle whose edge passes through the center of
the model, and a rectangle that excludes the center of the model.

breaks()

Return the projected radii at which Aperture.arc is not smooth.

A quadrature in projected radius whose intervals end at the returned
projected radii integrates a smooth function on each interval.

Returns:

**breaks** – The positive projected radii, in increasing order and in the
unit of length of the aperture. For a centered circle, the
list holds \(R_{\rm ape}\). For an offset circle, the list
holds the positive values among
\(|R_{\rm off} - R_{\rm ape}|\),
\(R_{\rm ape} - R_{\rm off}\) and
\(R_{\rm off} + R_{\rm ape}\). For a rectangle, the list
holds at most eight projected radii. Four of the eight are the
distances from the center of the model to the rectangle’s four
edge lines. Each such distance is included only where the point
of the edge line nearest to the center of the model lies on the
edge. The other four are the distances from the center of the
model to the rectangle’s four corners.

Return type:

[list](https://docs.python.org/3/builtins/stdtypes.html#list) of [float](https://docs.python.org/3/builtins/functions.html#float)

See also

[`Aperture.arc`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.arc)
The angle of the circle of radius r inside the aperture.

to_dict()

Return the aperture as a plain dictionary.

Returns:

**record** – The six fields of the aperture by name: `shape`, `r_off`,
`r_ape`, `side_x`, `side_y` and `angle`.
Aperture.from_dict builds the same aperture from the
returned dictionary.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Aperture.from_dict`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.from_dict)
Build an aperture from a dictionary.

classmethod from_dict(*record*)

Return the aperture described by a dictionary of its fields.

Parameters:

**record** (*mapping*) – The fields of the aperture by name, as Aperture.to_dict
writes them. The key `shape` is required. A missing `r_off`
or `angle` takes the value `0.0`. A missing `r_ape`,
`side_x` or `side_y` takes the value `None`.

Returns:

**aperture** – The aperture with the fields of record.

Return type:

[Aperture](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the fields of record do not make a valid aperture, under
    the rules of the Aperture constructor.

See also

[`Aperture.to_dict`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture.to_dict)
Return the aperture as a plain dictionary.

class pianistpy.observables.Differentiable(*model*, *settings=None*)

The observables of a one-component model as JAX functions of the
component’s anisotropy.

The anisotropy is the pair \(\theta = (\beta_{\rm c}, R_{\rm a})\) of
the central anisotropy \(\beta_{\rm c}\) and the anisotropy radius
\(R_{\rm a}\). Each method of Differentiable takes \(\theta\) as
the first argument. Each method returns the value of the public function
of the same name as the method (pianistpy.sigma_los for
Differentiable.sigma_los) on the model rebuilt at \(\theta\). Each method is a
JAX function of \(\theta\) and works under `jax.jit` and
`jax.grad`. To differentiate the observables of a model with several
components, call likelihood.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – Model of one component. The model must meet three conditions: the
model is spatially finite (`model.spatially_finite` is True); the
component’s profile is the truncated profile of a parent model and
carries the parent’s distribution function (DF); and the component’s
anisotropy family is isotropic, constant, Osipkov-Merritt (OM) Type
I, OM Type II or Cuddeford-Osipkov-Merritt (COM) Type I.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – Numerical settings of the observables. Default: `None`, which
means the settings of model.

Variables:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The model passed to the constructor.

- **theta** (*array**, **shape** (**2**,**)*) – The anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the model’s
component. \(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\)
is in internal units in every unit system of the model (internal,
model and physical). An isotropic or constant component has an
infinite \(R_{\rm a}\).

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is not a Model, or if model has more than one
    component. The components of a model with several components are
    differentiated through likelihood.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the model is not spatially finite, or if the component’s profile
    is not a truncated profile that carries the DF of a parent model.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component’s anisotropy family is COM Type II. The observables
    of a COM Type II component are not differentiable in the anisotropy.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a setting of the `observables` group is outside the setting’s
    range: `n_projection_nodes` below 20.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the validation report of the model has tier `"refused"` (an
    error above \(10^{-2}\)) and the settings of the model do not set
    `write_anyway=True`.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the squared line-of-sight dispersion of the model is negative at
    a tabulated projected radius.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If an intermediate result fails a check that the next step of the
    computation needs, for example non-finite values from a degenerate
    model.

See also

`likelihood`
The likelihood of data given a model, differentiable in the anisotropy of each component.

`truncate`
The truncated model whose profile a differentiable component uses.

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
The line-of-sight velocity dispersion of a model.

[`losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.losvd)
The line-of-sight velocity distribution of a model.

[`pdf_v`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.pdf_v)
The local velocity distribution of a model.

Notes

Every radius, velocity, aperture length and \(R_{\rm a}\) that a
method reads is in internal units in every unit system of the model.
Every value that a method returns is in internal units.

The first call of a method at a new \(\theta\) inverts the DF of the
component at \(\theta\) on the component’s truncated profile. The
first call then builds the tables of the observable at \(\theta\). A
Differentiable keeps the tables of at most eight values of
\(\theta\): the model’s own \(\theta\) and the seven other values
of \(\theta\) first met most recently.

At a \(\theta\) where rebuilding the model raises InputError,
UnphysicalModelError, AccuracyError or ResultError, a method
returns NaN and a zero gradient. Any other package error raised during
the computation of a method reaches the caller as
`jax.errors.JaxRuntimeError`. The message of the
`jax.errors.JaxRuntimeError` carries the message of the package error.
Three checks run before the computation and raise the package error
directly: the check of a concrete radius against the tabulated radii in
sigma_r, sigma_t and sigma_los; the check that the component’s DF
is differentiable in losvd, pdf_v and aperture_losvd; and the
check that the aperture passed to aperture_losvd is an Aperture.

The derivative of a method holds fixed the radii and velocities at which
the tables at \(\theta\) are tabulated. The value of a method follows
the change of the tabulated radii and velocities from one \(\theta\)
to another. The derivative of a method omits the change of the tabulated
radii and velocities. A method is not differentiable in the radii or
velocities that the method reads. The gradient in a radius or a velocity
is NaN.

The parameters that carry a derivative depend on the method:

- sigma_r, sigma_t, sigma_los and sigma_los_aperture carry the
derivative in \(\beta_{\rm c}\) for every accepted family, and in
\(R_{\rm a}\) for the OM and COM families.

- losvd, pdf_v and aperture_losvd carry the derivative in
\(\beta_{\rm c}\) for the constant and COM Type I families, and in
\(R_{\rm a}\) for the OM and COM Type I families. The gradient of
losvd, pdf_v or aperture_losvd in \(\beta_{\rm c}\) is NaN
for an OM component.

- For an isotropic or constant component, every method holds
\(R_{\rm a}\) at infinity, and the gradient in \(R_{\rm a}\) is
zero.

References

Notes Section T:7.2. Notes Section T:7.3.

Examples

```python
>>> import jax
>>> import pianistpy as pp
>>> parent = pp.truncate(
...     pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy())),
...     pp.Truncation(k_t=0.05),
... )
>>> member = pp.truncate(
...     pp.compute_df(
...         pp.Component(
...             profile=parent.profile, anisotropy=pp.Anisotropy(beta_c=0.25)
...         )
...     )
... )
>>> d = pp.Differentiable(member)
>>> grad = jax.grad(lambda theta: d.sigma_los(theta, [0.1, 0.3]).sum())(d.theta)
>>> grad.shape
(2,)
```

sigma_r(*theta*, *r*)

Return the radial velocity dispersion at user-specified 3D radii
for the model at the anisotropy theta.

The value is pianistpy.sigma_r of the model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **r** (*array-like*) – User-specified 3D radii \(r\). The radii are in internal units
in every unit system of the model.

Returns:

**sigma_r** – Radial velocity dispersion \(\sigma_{\rm r}\) in internal
units, with the shape of r. Every value is NaN when rebuilding
the model at theta raises InputError, UnphysicalModelError,
AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    radius that the model tabulates. A radius inside a function
    compiled by `jax.jit` is not checked before the computation.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error.

See also

[`None`](https://docs.python.org/3/builtins/constants.html#None)
The radial velocity dispersion of a model.

[`None`](https://docs.python.org/3/builtins/constants.html#None)
The tangential velocity dispersion as a JAX function of the anisotropy.

sigma_t(*theta*, *r*)

Return the tangential velocity dispersion at user-specified 3D
radii for the model at the anisotropy theta.

The tangential velocity dispersion \(\sigma_{\rm t}\) combines
both tangential velocity components. The value is pianistpy.sigma_t
of the model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **r** (*array-like*) – User-specified 3D radii \(r\). The radii are in internal units
in every unit system of the model.

Returns:

**sigma_t** – Tangential velocity dispersion \(\sigma_{\rm t}\) in internal
units, with the shape of r. Every value is NaN when rebuilding
the model at theta raises InputError, UnphysicalModelError,
AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    radius that the model tabulates. A radius inside a function
    compiled by `jax.jit` is not checked before the computation.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error.

See also

[`None`](https://docs.python.org/3/builtins/constants.html#None)
The tangential velocity dispersion of a model.

[`None`](https://docs.python.org/3/builtins/constants.html#None)
The radial velocity dispersion as a JAX function of the anisotropy.

sigma_los(*theta*, *r_2d*)

Return the line-of-sight velocity dispersion at user-specified
projected radii for the model at the anisotropy theta.

The value is pianistpy.sigma_los of the model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **r_2d** (*array-like*) – User-specified projected radii \(R_{\rm 2D}\). The projected
radii are in internal units in every unit system of the model.

Returns:

**sigma_los** – Line-of-sight velocity dispersion \(\sigma_{\rm los}\) in
internal units, with the shape of r_2d. Every value is NaN when
rebuilding the model at theta raises InputError,
UnphysicalModelError, AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a projected radius lies inside the innermost or beyond the
    outermost projected radius that the model tabulates. A projected
    radius inside a function compiled by `jax.jit` is not checked
    before the computation.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error.

See also

[`None`](https://docs.python.org/3/builtins/constants.html#None)
The line-of-sight velocity dispersion of a model.

[`Differentiable.sigma_los_aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Differentiable.sigma_los_aperture)
The line-of-sight velocity dispersion over an aperture as a JAX function of the anisotropy.

sigma_los_aperture(*theta*, *aperture*)

Return the line-of-sight velocity dispersion over an aperture for
the model at the anisotropy theta.

The aperture dispersion \(\sigma_{\rm ape}\) is the root mean
square of \(\sigma_{\rm los}\) over the aperture, weighted by the
surface density. The value is pianistpy.sigma_los_aperture of the
model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **aperture** ([*Aperture*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)) – User-specified aperture. The lengths of the aperture are in
internal units in every unit system of the model.

Returns:

**sigma_los** – Aperture dispersion \(\sigma_{\rm ape}\) in internal units.
The value is NaN when rebuilding the model at theta raises
InputError, UnphysicalModelError, AccuracyError or ResultError.

Return type:

scalar array

Raises:

[**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If pianistpy.sigma_los_aperture raises InputError for the
    aperture on the model at theta, because the tables of the model
    do not cover the aperture. The message of the
    `jax.errors.JaxRuntimeError` carries the message of the
    InputError.

See also

`pianistpy.sigma_los_aperture`
The line-of-sight velocity dispersion of a model over an aperture.

[`Aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)
The apertures of the observables.

[`Differentiable.aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Differentiable.aperture_losvd)
The line-of-sight velocity distribution in an aperture as a JAX function of the anisotropy.

Notes

The derivative of \(\sigma_{\rm ape}\) in a parameter
\(\theta_i\) (\(\beta_{\rm c}\) or \(R_{\rm a}\)) is

\[\frac{\partial \sigma_{\rm ape}}{\partial \theta_i}
= \frac{1}{2\sigma_{\rm ape}}
\frac{\int_A \Sigma\, \partial \sigma_{\rm los}^2/\partial
\theta_i \, {\rm d}A}{\int_A \Sigma \, {\rm d}A},\]

with \(A\) the aperture and \(\Sigma\) the surface density of
the component. The surface density does not depend on
\(\theta\), since the component’s profile is fixed. Both
integrals use the quadrature nodes and weights of
pianistpy.sigma_los_aperture.

losvd(*theta*, *r_2d*, *v*)

Return the line-of-sight velocity distribution at the pairs
`(r_2d, v)` for the model at the anisotropy theta.

The value is pianistpy.losvd of the model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **r_2d** (*array-like*) – User-specified projected radii \(R_{\rm 2D}\). The projected
radii are in internal units in every unit system of the model.

- **v** (*array-like*) – Line-of-sight velocities \(V\), in internal units in every
unit system of the model. The arrays r_2d and v broadcast
against each other.

Returns:

**losvd** – Line-of-sight velocity distribution per unit internal velocity,
with the broadcast shape of r_2d and v. Every value is NaN
when rebuilding the model at theta raises InputError,
UnphysicalModelError, AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component’s DF is not differentiable in the anisotropy.
    The DF of an isotropic component is not differentiable in the
    anisotropy. The DF of a component whose DF center lies on a
    critical branch is not differentiable in the anisotropy. The
    message states the reason.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error. A projected radius outside the
    table, for which pianistpy.losvd raises a package error, gives
    a `jax.errors.JaxRuntimeError`.

See also

`pianistpy.losvd`
The line-of-sight velocity distribution of a model.

[`Differentiable.aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Differentiable.aperture_losvd)
The line-of-sight velocity distribution in an aperture as a JAX function of the anisotropy.

Notes

The derivative in a parameter \(\theta_i\) (\(\beta_{\rm c}\)
or \(R_{\rm a}\)) is the sum of two parts. The first part is the
derivative in \(\theta_i\) at a fixed DF table. The second part
is the derivative through the values of the DF table, with the
energies of the DF table held fixed. The method is not
differentiable in r_2d or v.

References

Notes Section T:7.1.

pdf_v(*theta*, *r*, *v*)

Return the local velocity distribution at the pairs `(r, v)` for the
model at the anisotropy theta.

The value is pianistpy.pdf_v of the model rebuilt at theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **r** (*array-like*) – User-specified 3D radii \(r\). The radii are in internal units
in every unit system of the model.

- **v** (*array-like*) – Velocities \(v\), in internal units in every unit system of the
model. The arrays r and v broadcast against each other.

Returns:

**pdf_v** – Local velocity distribution per unit internal velocity, with the
broadcast shape of r and v. Every value is NaN when rebuilding
the model at theta raises InputError, UnphysicalModelError,
AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component’s DF is not differentiable in the anisotropy.
    The DF of an isotropic component is not differentiable in the
    anisotropy. The DF of a component whose DF center lies on a
    critical branch is not differentiable in the anisotropy. The
    message states the reason.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error. A radius outside the radii of the
    model, for which pianistpy.pdf_v raises a package error, gives a
    `jax.errors.JaxRuntimeError`.

See also

`pianistpy.pdf_v`
The local velocity distribution of a model.

[`Differentiable.losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Differentiable.losvd)
The line-of-sight velocity distribution as a JAX function of the anisotropy.

Notes

The derivative in a parameter \(\theta_i\) (\(\beta_{\rm c}\)
or \(R_{\rm a}\)) is the sum of two parts. The first part is the
derivative in \(\theta_i\) at a fixed DF table and at fixed
velocities. The second part is the derivative through the values of the
DF table, with the energies of the DF table held fixed. The method is
not differentiable in r or v.

References

Notes Section T:7.1.

aperture_losvd(*theta*, *aperture*, *v*)

Return the line-of-sight velocity distribution in an aperture at
user-specified line-of-sight velocities for the model at the
anisotropy theta.

The value is pianistpy.aperture_losvd of the model rebuilt at
theta.

Parameters:

- **theta** (*array-like**, **shape** (**2**,**)*) – Anisotropy \((\beta_{\rm c}, R_{\rm a})\) of the component.
\(\beta_{\rm c}\) is dimensionless. \(R_{\rm a}\) is in
internal units in every unit system of the model.

- **aperture** ([*Aperture*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)) – User-specified aperture. The lengths of the aperture are in
internal units in every unit system of the model.

- **v** (*array-like*) – Line-of-sight velocities \(V\), in internal units in every
unit system of the model.

Returns:

**losvd** – Line-of-sight velocity distribution in the aperture per unit
internal velocity, with the shape of v. Every value is NaN when
rebuilding the model at theta raises InputError,
UnphysicalModelError, AccuracyError or ResultError.

Return type:

array

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component’s DF is not differentiable in the anisotropy.
    The DF of an isotropic component is not differentiable in the
    anisotropy. The DF of a component whose DF center lies on a
    critical branch is not differentiable in the anisotropy.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If aperture is not an Aperture.

- [**jax.errors.JaxRuntimeError**](https://docs.jax.dev/en/latest/101/errors.html#jax.errors.JaxRuntimeError) – If the computation raises a package error other than the four
    package errors that give NaN, for example
    pianistpy.exceptions.WorkerError when a parallel process ends.
    The message of the `jax.errors.JaxRuntimeError` carries the
    message of the package error. An aperture that the table does not
    cover, for which pianistpy.aperture_losvd raises a package
    error, gives a `jax.errors.JaxRuntimeError`.

See also

`pianistpy.aperture_losvd`
The line-of-sight velocity distribution of a model in an aperture.

[`Aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)
The apertures of the observables.

[`Differentiable.losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Differentiable.losvd)
The line-of-sight velocity distribution as a JAX function of the anisotropy.

Notes

The derivative in a parameter \(\theta_i\) (\(\beta_{\rm c}\)
or \(R_{\rm a}\)) is the sum of two parts. The first part is the
derivative in \(\theta_i\) at a fixed DF table. The second part
is the derivative through the values of the DF table, with the
energies of the DF table held fixed. The method is not
differentiable in v.

References

Notes Section T:7.1.

class pianistpy.observables.LosTable(*model*)

The projected tables of a model, built once for many calls.

A LosTable holds the intrinsic velocity dispersions of a model, the
projected velocity dispersion of the model, and the model’s table of
line-of-sight velocity distributions over projected radius. Every
observable function given a LosTable reads the tables of the LosTable.
A model passed to many observable calls through one LosTable is
therefore projected once. Building a LosTable starts parallel
processes, one per core set by set_cores.

Parameters:

**model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The model to project. The model has one component, or the model is
a superposition of components (superpose).

Variables:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The projected model. The observable functions given the LosTable
return values in the unit system of model.

- **profile** (*DispersionProfile*) – The intrinsic second moments of the velocity of the model, computed
at the model’s settings. For a superposition, profile belongs to
the first component.

- **projected** (*ProjectedDispersion*) – The surface density and the line-of-sight velocity dispersion over
projected radius. For a superposition, projected belongs to the
first component.

- **kept** (*KeptRanges*) – The ranges of projected radius over which the velocity quadrature
of the table keeps two of the table’s break velocities. For a
superposition, kept belongs to the first component.

- **table** (*LosvdTable*) – The line-of-sight velocity distributions over projected radius. For
a superposition, table belongs to the first component.

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The checks of the table. Each check holds its relative error and
the projected radius where the check’s relative error is largest. The
report’s tier (`validation.tier`) is the band of the worst
relative error among the table’s checks and the model’s own
validation report. For a superposition, each check holds its
largest relative error over the components.

- **parts** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – For a superposition, one LosTable per component, each built from
the component as a model of one component. For a model of one
component, parts is `(self,)`.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is not a Model, or model has several components in
    one potential.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a setting of the `observables` group is outside its valid
    range: `n_projection_nodes` below 20, or `rows_route` other than
    `"kernel"` or `"numpy"`.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the model’s or the table’s relative error is above
    \(10^{-2}\) and the settings do not set `write_anyway=True`.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If an intermediate result fails a check that the next step of the
    computation needs. Examples are a degenerate model, and a `losvd`
    setting that leaves the table no valid quadrature
    (`energy_layer_ratio` of one or more).

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If a parallel process of the build ends before finishing its task.
    The usual cause is a lack of memory.

Warns:

**AccuracyWarning** – If the table’s relative error is above \(10^{-4}\) (tier
`"warned"`).

See also

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
Line-of-sight velocity dispersion at projected radii.

[`losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.losvd)
Line-of-sight velocity distribution at projected radii.

[`aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.aperture_losvd)
Line-of-sight velocity distribution inside an aperture.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> table = pp.LosTable(model)
>>> pp.sigma_los(table, [0.5, 1.0, 2.0]).shape
(3,)
```

to(*system*, ***kwargs*)

Return the same tables presented in another unit system.

The call computes no table again. The tables stay in internal
units. The observable functions given the returned LosTable return
values in the unit system of `self.model.to(system, **kwargs)`.

Parameters:

- **system** (*{"internal"**, **"model"**, **"physical"}*) – The target unit system.

- ****kwargs** – The scales of the target unit system, as Model.to takes them:
`concentration`, `mass_0`, and `r_0` or `r_vir`.

Returns:

**table** – A LosTable with the same tables, whose model is presented in
system. The returned LosTable shares its parallel processes
with the original LosTable (LosTable.close).

Return type:

[LosTable](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system or the scales in kwargs are not valid, under the
    rules of Model.to.

See also

`Model.to`
Present a model in another unit system.

close()

Close the parallel processes kept open by the table’s aperture calls.

With more than one core set by set_cores, the first aperture call
on a LosTable starts parallel processes. Later aperture calls on the
same LosTable reuse the parallel processes of the first aperture
call. The parallel processes end at LosTable.close or when the
LosTable is deleted. A LosTable returned by LosTable.to shares the
parallel processes of the original LosTable. An aperture call after
LosTable.close starts new parallel processes.

See also

[`sigma_los_aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los_aperture)
Line-of-sight velocity dispersion inside an aperture.

[`aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.aperture_losvd)
Line-of-sight velocity distribution inside an aperture.

Return type:

None

class pianistpy.observables.ProjectedPairs(*r_2d*, *v_los*, *mass*, *component*, *units*, *record=<factory>*, *validation=None*)

Pairs of projected radius and line-of-sight velocity.

ProjectedSampler.sample returns a ProjectedPairs. Each array
attribute holds one value per pair.

Variables:

- **r_2d** (*array**, **shape** (**n**,**)*) – Projected radius of each pair. The unit depends on the source’s unit
system (Model.to): units of \(r_0\) in the internal and the
model systems, parsecs in the physical system.
`units["r_2d"]` gives the unit.

- **v_los** (*array**, **shape** (**n**,**)*) – Line-of-sight velocity of each pair. The unit depends on the
source’s unit system: internal units, units of
\((G M_0/r_0)^{1/2}\) in the model system, km/s in the physical
system. `units["v_los"]` gives the unit.

- **mass** (*array**, **shape** (**n**,**)*) – Mass of each pair. Every pair carries the mass
ProjectedSampler.mass divided by the number of pairs. The unit
depends on the source’s unit system: internal units, units of
\(M_0\) in the model system, solar masses in the physical
system. `units["mass"]` gives the unit.

- **component** (*array** of *[*int*](https://docs.python.org/3/builtins/functions.html#int)*, **shape** (**n**,**)*) – Index of the model component each pair was drawn from, in the order
of the model’s components.

- **units** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The unit of each of `"r_2d"`, `"v_los"` and `"mass"`. In the
physical system, the unit is the Astropy unit’s name, such as
`"pc"`. In the internal and the model systems, the unit is the
system’s name, `"internal"` or `"model"`.

- **record** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The draw’s record, with the keys `"source"` (the names of the
model’s components), `"R_range"` and `"v_range"` (the annulus
and the velocity range, in internal units), `"seed"`, `"mass"`
(ProjectedSampler.mass, in internal units), `"shares"` (each
component’s share of `"mass"`) and `"grids"` (the sizes of the
sampler’s tables).

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The validation report of the ProjectedSampler that drew the pairs.

Parameters:

- **r_2d** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **v_los** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **mass** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **component** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any))

- **units** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*]*)

- **record** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, *[*Any*](https://docs.python.org/3/library/typing.html#typing.Any)*]*)

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)* | **None*)

See also

[`ProjectedSampler`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.ProjectedSampler)
The sampler that draws the pairs.

[`LosTable`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)
The tables of the model that the sampler reads.

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.observables import ProjectedSampler
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> pairs = ProjectedSampler(pp.LosTable(model), (0.5, 2.0)).sample(10)
>>> sorted(pairs.units)
['mass', 'r_2d', 'v_los']
```

class pianistpy.observables.ProjectedSampler(*source*, *R_range*, *v_range=None*)

Sampler of pairs of projected radius and line-of-sight velocity.

The sampler draws pairs \((R, v)\) of projected radius \(R\)
and line-of-sight velocity \(v\) from the probability density
\(p(R, v) \propto 2\pi R\,\Sigma(R)\,D(v\,|\,R)\) on the annulus
R_range and in the user-specified range v_range of line-of-sight
velocity. \(\Sigma(R)\) is the surface density, and
\(D(v\,|\,R)\) is the line-of-sight velocity distribution at
\(R\). The distribution of \(R\) is weighted at each radius by
the share of \(D(v\,|\,R)\) inside v_range. \(D(v\,|\,R)\)
is the distribution that the model’s LosTable holds, for every
anisotropy model, including Cuddeford-Osipkov-Merritt (COM) models.
Call ProjectedSampler rather than Sampler to draw a projected sample
on an annulus without drawing 3D particles.

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, of one component or a superposition of components, or the
model’s LosTable. Given a Model, the sampler builds the model’s
LosTable first. For several samplers of one model, pass
`pp.LosTable(model)`.

- **R_range** (*(*[*float*](https://docs.python.org/3/builtins/functions.html#float)*, *[*float*](https://docs.python.org/3/builtins/functions.html#float)*)*) – The annulus \(R_{\rm lo} \leq R_{\rm 2D} \leq R_{\rm hi}\), with
\(0 \leq R_{\rm lo} < R_{\rm hi}\). The unit depends on the
model’s unit system (Model.to): units of the scale radius
\(r_0\) in the internal and the model systems, parsecs in the
physical system. In the physical system, Astropy quantities of any
length unit are also accepted. An end of the annulus inside the
table’s innermost projected radius moves to the innermost projected
radius. An end beyond the table’s outermost projected radius moves to
the outermost projected radius. The sampler moves an end only if the
annulus’s mass cut off is at most \(10^{-6}\) of the annulus’s
mass.

- **v_range** (*(*[*float*](https://docs.python.org/3/builtins/functions.html#float)*, *[*float*](https://docs.python.org/3/builtins/functions.html#float)*)**, **optional*) – The user-specified range \(v_{\rm lo} < v < v_{\rm hi}\) of
signed line-of-sight velocity. The unit depends on the model’s unit
system: internal units, units of \((G M_0/r_0)^{1/2}\) in the
model system, km/s in the physical system. Default: `None`, which
means the whole velocity distribution.

Variables:

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The sampler’s validation report. Each check is a dimensionless
error. `v_table` is the error of the cumulative distribution of
\(v\) that the draw interpolates. `window` is the error of the
cumulative probabilities at the ends of v_range.
`radius_quadrature` is the difference of the annulus’s mass from
the mass by a finer quadrature, relative to the annulus’s mass.
`row_norm` is the largest difference of a tabulated velocity
distribution’s integral from 1. `annulus_cut` is the share of the
annulus’s mass cut off outside the table’s projected radii.

- **mass** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The annulus’s mass in the velocity range, in internal units.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError)**, **[**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError)**, **[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError)**, **[**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – Given a Model, if LosTable raises the exception.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the LosTable’s validation report is above \(10^{-2}\) and
    the settings do not set `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If R_range is not \(0 \leq R_{\rm lo} < R_{\rm hi}\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If v_range is not \(v_{\rm lo} < v_{\rm hi}\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the annulus lies wholly inside the table’s innermost projected
    radius or wholly beyond the table’s outermost projected radius.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the annulus’s mass outside the table’s projected radii is more
    than \(10^{-6}\) of the annulus’s mass.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If R_range starts inside the table’s innermost projected radius
    and the surface density’s logarithmic slope between the table’s
    two innermost projected radii is 2 or steeper, so that the mass
    inside the innermost projected radius diverges.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the annulus overlaps an interval where the table did not meet the
    table’s tolerance (`table.table.r_uncertified`).

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the inverse cumulative distribution of \(|v|\) at a tabulated
    radius has no positive finite slope.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the velocity range holds none of the distribution on the
    annulus.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the sampler’s validation report is above \(10^{-2}\) and the
    settings do not set `write_anyway=True`.

Warns:

**AccuracyWarning** – If the sampler’s validation report, or the LosTable’s, is above
\(10^{-4}\) (`report.tier` is `"warned"`).

See also

[`ProjectedPairs`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.ProjectedPairs)
The pairs that ProjectedSampler.sample returns.

[`LosTable`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)
The tables of a model, built once for many calls.

[`losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.losvd)
The line-of-sight velocity distribution at projected radii.

`Sampler`
The sampler of 3D particles.

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.observables import ProjectedSampler
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> sampler = ProjectedSampler(pp.LosTable(model), (0.5, 2.0))
>>> pairs = sampler.sample(100, seed=1)
>>> pairs.r_2d.shape, pairs.v_los.shape
((100,), (100,))
>>> bool(((pairs.r_2d >= 0.5) & (pairs.r_2d <= 2.0)).all())
True
```

sample(*n*, *seed=0*)

Return n pairs drawn from the annulus and the velocity range.

Pair \(i\) is drawn from random numbers of pair \(i\) alone,
as Sampler draws its particles. So at one seed, pair \(i\) is
the same for every n above \(i\). For a superposition of
components, the component of each pair is drawn with probability
equal to the component’s share of the annulus’s mass in the velocity
range.

Parameters:

- **n** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of pairs.

- **seed** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – Seed of the random numbers. Default: `0`, which means that
every call with `seed=0` draws the same pairs.

Returns:

**pairs** – The pairs, in the source’s unit system. Each pair carries the
mass mass divided by n.

Return type:

[ProjectedPairs](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.ProjectedPairs)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the source’s unit system is not the internal system and the
    unit system’s scales do not form conversion factors.

See also

[`ProjectedPairs`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.ProjectedPairs)
The class of the returned pairs.

pianistpy.observables.aperture_losvd(*source*, *aperture*, *v*)

Return the line-of-sight velocity distribution inside an aperture.

The aperture distribution is the probability density of the
line-of-sight velocity of the model’s mass inside the aperture, weighted by
the surface density over the aperture. The second moment of the
aperture distribution is the square of sigma_los_aperture given the
same LosTable. The call compares the second moment of the aperture distribution
with the squared
aperture dispersion computed from the projected dispersion, without the
distribution. The relative difference enters the validation report of
the aperture distribution.

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. Given a Model, the call
builds the model’s LosTable at every call. For many calls on one
model, pass `pp.LosTable(model)`.

- **aperture** ([*Aperture*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)) – The aperture, built by Aperture.circle or Aperture.rectangle.
The unit of the aperture’s lengths is the scale radius \(r_0\)
in the internal and the model systems, and the parsec in the
physical system (Model.to). Each length is a float.

- **v** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – Line-of-sight velocity \(v\). The unit is the internal unit in
the internal system, \((G M_0/r_0)^{1/2}\) in the model system,
and km/s in the physical system. In the physical system, an Astropy
quantity of any velocity unit is also accepted.

Returns:

**losvd** – The aperture distribution at each v, with the shape of v. The
aperture distribution is even in \(v\), integrates to one over
all \(v\), and is zero beyond the support of the distribution
function along the line of sight (the escape velocity, except for a
Type II model, whose support extends above it). In the internal and
the model systems, a plain array per unit velocity of the system. In
the physical system, an Astropy quantity in s/km.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model and building the model’s LosTable raises
    InputError (see LosTable).

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model and building the model’s LosTable raises
    AccuracyError (see LosTable).

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and building the model’s LosTable raises
    ResultError (see LosTable).

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If source is a Model and building the model’s LosTable raises
    WorkerError (see LosTable).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the aperture fails one of four conditions: the aperture lies
    wholly outside the tabulated projected radii; a centered circle’s
    radius does not exceed the innermost tabulated projected radius; a
    centered circle reaches beyond the outermost tabulated projected
    radius; or the part of the aperture inside the innermost tabulated
    projected radius carries more than \(10^{-6}\) of the
    aperture’s weight.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the aperture distribution’s relative error is above
    \(10^{-2}\) and the settings do not set `write_anyway=True`.

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If more than one core is set by set_cores and a parallel process
    of the call ends before finishing its task. The usual cause is a
    lack of memory.

Warns:

**AccuracyWarning** – If the relative error of the aperture distribution, or of the
LosTable, is above \(10^{-4}\) (tier `"warned"`).

See also

[`sigma_los_aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los_aperture)
Line-of-sight velocity dispersion inside an aperture.

[`losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.losvd)
Line-of-sight velocity distribution at projected radii.

[`Aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)
A circular or rectangular aperture.

Notes

For a superposition of components \(k\) with weights \(w_k\),
aperture distributions \(D_k\) and projected masses \(M_k\)
inside the aperture,

\[D = \frac{\sum_k w_k M_k D_k}{\sum_k w_k M_k}.\]

References

Notes Section T:6.6 (the distribution in an aperture) and
Notes Section T:6.9 (superposed models).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> table = pp.LosTable(model)
>>> pp.aperture_losvd(table, pp.Aperture.circle(1.0), [0.0, 0.2, 0.4]).shape
(3,)
```

pianistpy.observables.losvd(*source*, *r_2d*, *v*)

Return the line-of-sight velocity distribution at projected radii.

The line-of-sight velocity distribution \(D(R_{\rm 2D}, v)\) is the
probability density of the line-of-sight velocity \(v\) of the
tracers at projected radius \(R_{\rm 2D}\).

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. Given a Model, the call
builds the model’s LosTable at every call. For many calls on one
model, pass `pp.LosTable(model)`.

- **r_2d** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – User-specified projected radii \(R_{\rm 2D}\). The unit is the
scale radius \(r_0\) in the internal and the model systems, and
the parsec in the physical system (Model.to). In the physical
system, an Astropy quantity of any length unit is also accepted.

- **v** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – Line-of-sight velocity \(v\). The unit is the internal unit in
the internal system, \((G M_0/r_0)^{1/2}\) in the model system,
and km/s in the physical system. In the physical system, an Astropy
quantity of any velocity unit is also accepted. The arrays r_2d
and v broadcast against each other.

Returns:

**losvd** – The probability density \(D\) at each pair
\((R_{\rm 2D}, v)\), with the broadcast shape of r_2d and v.
\(D\) is even in \(v\), integrates to one over all
\(v\), and is zero beyond the support of the distribution
function along the line of sight (the escape velocity, except for a
Type II model, whose support extends above it). In the internal and
the model systems, a plain array per unit velocity of the system. In
the physical system, an Astropy quantity in s/km.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model and building the model’s LosTable raises
    InputError (see LosTable).

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model and building the model’s LosTable raises
    AccuracyError (see LosTable).

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and building the model’s LosTable raises
    ResultError (see LosTable).

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If source is a Model and building the model’s LosTable raises
    WorkerError (see LosTable).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    projected radius of the table. The call does not extrapolate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a radius lies inside an interval of projected radius where the
    table does not meet its tolerance
    (the attribute `r_uncertified` of `LosTable.table`). The message gives the
    interval and the reason.

See also

[`aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.aperture_losvd)
Line-of-sight velocity distribution inside an aperture.

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
Line-of-sight velocity dispersion at projected radii.

[`LosTable`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)
The projected tables of a model, built once for many calls.

Notes

For a superposition of components \(k\) with weights \(w_k\),
surface densities \(\Sigma_k\) and line-of-sight velocity
distributions \(D_k\),

\[D = \frac{\sum_k w_k \Sigma_k D_k}{\sum_k w_k \Sigma_k}.\]

References

Notes Section T:6.4 (the line-of-sight velocity distribution) and
Notes Section T:6.9 (superposed models).

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()),
...     settings=pp.Settings(quick=True),
... )
>>> table = pp.LosTable(model)
>>> v = np.linspace(-1.0, 1.0, 5)
>>> pp.losvd(table, 1.0, v).shape
(5,)
```

pianistpy.observables.pdf_v(*source*, *r*, *v*)

Return the local velocity distribution at 3D radii.

The local velocity distribution \(p(V \,|\, r)\) is the probability
density of the velocity \(V = |\mathbf{v}|\) of the model’s mass at 3D
radius \(r\). The call checks the distribution at each radius in
three ways: the norm of the distribution against the density; the
second moment of the distribution against the velocity dispersion from
the Jeans equation; and the change of the distribution when the angular
quadrature is refined (pianistpy.observables.stages.velocity_distribution).

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. A Model and the
LosTable of the same model return the same values.

- **r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – User-specified 3D radii \(r\). The unit is the scale radius
\(r_0\) in the internal and the model systems, and the parsec
in the physical system (Model.to). In the physical system, an
Astropy quantity of any length unit is also accepted.

- **v** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – Velocity \(V\). The unit is the internal unit in the internal
system, \((G M_0/r_0)^{1/2}\) in the model system, and km/s in
the physical system. In the physical system, an Astropy quantity of
any velocity unit is also accepted. The arrays r and v broadcast
against each other.

Returns:

**pdf_v** – The probability density \(p(V \,|\, r)\) at each pair
\((r, V)\), with the broadcast shape of r and v. The
density integrates to one over \(V \geq 0\). The density is zero
beyond the end of the support of the distribution function (DF).
For a Type II model, the support ends at
\(V_{\rm esc}/\sqrt{1 - r^2/r_{\rm a}^2}\), with
\(V_{\rm esc}\) the escape velocity and \(r_{\rm a}\) the
anisotropy radius. For every other model, the support ends at the
escape velocity \(V_{\rm esc}\). In the internal and the model
systems, a plain array per unit velocity of the system. In the
physical system, an Astropy quantity in s/km.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model with several components in one potential.
    A model of one component and a superposition (superpose) are
    accepted.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model whose relative error is above
    \(10^{-2}\) and whose settings do not set
    `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies outside the radii that the model tabulates. For a
    Type II model, also if a radius lies at or beyond the anisotropy
    radius \(r_{\rm a}\).

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the relative error of the distribution at a radius is above
    \(10^{-2}\) and the model’s settings do not set
    `write_anyway=True`.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and an intermediate result fails a check
    that the next step of the computation needs, such as values that
    are not finite for a degenerate model.

Warns:

**AccuracyWarning** – If the relative error of the distribution at a radius is above
\(10^{-4}\) (tier `"warned"`).

See also

[`sigma_r`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_r)
Radial velocity dispersion at 3D radii.

[`sigma_t`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_t)
Tangential velocity dispersion at 3D radii.

Notes

For a superposition of components \(k\) with weights \(w_k\),
densities \(\varrho_k\) and velocity distributions \(p_k\),

\[p = \frac{\sum_k w_k \varrho_k p_k}{\sum_k w_k \varrho_k}.\]

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Section 5.1 and Eq. (B38).
Notes Section T:6.8 (the local velocity distribution).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> pp.pdf_v(model, 1.0, [0.1, 0.3, 0.5]).shape
(3,)
```

pianistpy.observables.sigma_los_aperture(*source*, *aperture*)

Return the line-of-sight velocity dispersion inside an aperture.

The line-of-sight velocity dispersion \(\sigma_{\rm ape}\) inside
an aperture is the square root of the second moment of the
line-of-sight velocity over the aperture, weighted by the surface
density. Given a Model, the call computes the projected dispersion
only at the projected radii that the aperture covers. Given a
LosTable, the call computes \(\sigma_{\rm ape}\) as the square
root of the second moment of the distribution that aperture_losvd
returns. The call then compares the second moment of the aperture
distribution with \(\sigma_{\rm ape}^2\) computed from the
projected dispersion. The relative difference enters the validation
report of the aperture distribution.

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model.

- **aperture** ([*Aperture*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)) – The aperture, built by Aperture.circle or Aperture.rectangle.
The unit of the aperture’s lengths is the scale radius \(r_0\)
in the internal and the model systems, and the parsec in the
physical system (Model.to). Each length is a float.

Returns:

**sigma_ape** – Line-of-sight velocity dispersion \(\sigma_{\rm ape}\) inside
the aperture. In the internal system, a float in internal units. In
the model system, a float in units of \((G M_0/r_0)^{1/2}\). In
the physical system, an Astropy quantity in km/s.

Return type:

[float](https://docs.python.org/3/builtins/functions.html#float) or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model with several components in one potential.
    A model of one component and a superposition (superpose) are
    accepted.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model whose relative error is above
    \(10^{-2}\) and whose settings do not set
    `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the aperture fails one of four conditions: the aperture lies
    wholly outside the tabulated projected radii; a centered circle’s
    radius does not exceed the innermost tabulated projected radius; a
    centered circle reaches beyond the outermost tabulated projected
    radius; or the part of the aperture inside the innermost tabulated
    projected radius carries more than \(10^{-6}\) of the
    aperture’s weight.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a LosTable, the aperture distribution’s relative
    error is above \(10^{-2}\), and the settings do not set
    `write_anyway=True`.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and an intermediate result fails a check
    that the next step of the computation needs, such as values that
    are not finite for a degenerate model.

- [**WorkerError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.WorkerError) – If source is a LosTable, more than one core is set by
    set_cores, and a parallel process of the call ends before
    finishing its task. The usual cause is a lack of memory.

Warns:

**AccuracyWarning** – If source is a LosTable and the relative error of the aperture
distribution, or of the LosTable, is above \(10^{-4}\) (tier
`"warned"`).

See also

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
Line-of-sight velocity dispersion at projected radii.

[`aperture_losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.aperture_losvd)
Line-of-sight velocity distribution inside an aperture.

[`Aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.Aperture)
A circular or rectangular aperture.

Notes

Given a Model, the call evaluates

\[\sigma_{\rm ape}^2 = \frac{\int_A \Sigma\,\sigma_{\rm los}^2\,{\rm d}A}
{\int_A \Sigma\,{\rm d}A},\]

with \(A\) the aperture, \(\Sigma\) the surface density and
\(\sigma_{\rm los}\) the line-of-sight velocity dispersion. For a
superposition of components \(k\) with weights \(w_k\), the
numerator integrates \(\sum_k w_k \Sigma_k \sigma_{{\rm los},k}^2\)
and the denominator integrates \(\sum_k w_k \Sigma_k\). Given a
LosTable, the call sums the second moments of the components’
aperture distributions with the weights \(w_k M_k\), with
\(M_k\) the projected mass of component \(k\) inside the
aperture.

References

Notes Section T:6.3 (apertures) and Notes Section T:6.6 (the
distribution in an aperture).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> sigma = pp.sigma_los_aperture(model, pp.Aperture.circle(1.0))
>>> bool(sigma > 0.0)
True
```

pianistpy.observables.sigma_los(*source*, *r_2d*)

Return the line-of-sight velocity dispersion at projected radii.

The line-of-sight velocity dispersion
\(\sigma_{\rm los}(R_{\rm 2D})\) at projected radius
\(R_{\rm 2D}\) is the square root of the second moment of the
line-of-sight velocity, weighted by the density along the line of
sight.

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. A Model and the
LosTable of the same model return the same values.

- **r_2d** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – User-specified projected radii \(R_{\rm 2D}\). The unit is the
scale radius \(r_0\) in the internal and the model systems, and
the parsec in the physical system (Model.to). In the physical
system, an Astropy quantity of any length unit is also accepted.

Returns:

**sigma_los** – Line-of-sight velocity dispersion \(\sigma_{\rm los}\), with
the shape of r_2d. In the internal system, a plain array in
internal units. In the model system, a plain array in units of
\((G M_0/r_0)^{1/2}\). In the physical system, an Astropy
quantity in km/s.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model with several components in one potential.
    A model of one component and a superposition (superpose) are
    accepted.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model whose relative error is above
    \(10^{-2}\) and whose settings do not set
    `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    projected radius that the model tabulates. The call does not
    extrapolate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and an intermediate result fails a check
    that the next step of the computation needs, such as values that
    are not finite for a degenerate model.

See also

[`sigma_los_aperture`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los_aperture)
Line-of-sight velocity dispersion inside an aperture.

[`losvd`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.losvd)
Line-of-sight velocity distribution at projected radii.

[`LosTable`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)
The projected tables of a model, built once for many calls.

Notes

For a superposition of components \(k\) with weights \(w_k\),
surface densities \(\Sigma_k\) and line-of-sight velocity
dispersions \(\sigma_{{\rm los},k}\),

\[\sigma_{\rm los}^2 = \frac{\sum_k w_k \Sigma_k
\sigma_{{\rm los},k}^2}{\sum_k w_k \Sigma_k}.\]

References

Notes Section T:6.2 (projection) and Notes Section T:6.9 (superposed
models).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> pp.sigma_los(model, [0.1, 1.0, 10.0]).shape
(3,)
```

pianistpy.observables.sigma_r(*source*, *r*)

Return the radial velocity dispersion at 3D radii.

The radial velocity dispersion \(\sigma_r(r)\) is the square root
of the second moment of the radial velocity at 3D radius \(r\).

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. A Model and the
LosTable of the same model return the same values.

- **r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – User-specified 3D radii \(r\). The unit is the scale radius
\(r_0\) in the internal and the model systems, and the parsec
in the physical system (Model.to). In the physical system, an
Astropy quantity of any length unit is also accepted.

Returns:

**sigma_r** – Radial velocity dispersion \(\sigma_r\), with the shape of r.
In the internal system, a plain array in internal units. In the
model system, a plain array in units of \((G M_0/r_0)^{1/2}\).
In the physical system, an Astropy quantity in km/s.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model with several components in one potential.
    A model of one component and a superposition (superpose) are
    accepted.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model whose relative error is above
    \(10^{-2}\) and whose settings do not set
    `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    radius that the model tabulates, or beyond the edge of a spatially
    finite model. The call does not extrapolate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and an intermediate result fails a check
    that the next step of the computation needs, such as values that
    are not finite for a degenerate model.

See also

[`sigma_t`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_t)
Tangential velocity dispersion at 3D radii.

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
Line-of-sight velocity dispersion at projected radii.

[`pdf_v`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.pdf_v)
Local velocity distribution at 3D radii.

Notes

For a superposition of components \(k\) with weights \(w_k\),
densities \(\rho_k\) and radial velocity dispersions
\(\sigma_{r,k}\),

\[\sigma_r^2 = \frac{\sum_k w_k \rho_k \sigma_{r,k}^2}
{\sum_k w_k \rho_k}.\]

References

Notes Section T:6.1 (intrinsic moments) and Notes Section T:6.9
(superposed models).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> pp.sigma_r(model, [0.1, 1.0, 10.0]).shape
(3,)
```

pianistpy.observables.sigma_t(*source*, *r*)

Return the tangential velocity dispersion at 3D radii.

The tangential velocity dispersion \(\sigma_{\rm t}(r)\) combines
both tangential components of the velocity at 3D radius \(r\):
\(\sigma_{\rm t}^2 = \sigma_\theta^2 + \sigma_\phi^2\).

Parameters:

- **source** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)* or *[*LosTable*](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.LosTable)) – The model, or the LosTable of the model. A Model and the
LosTable of the same model return the same values.

- **r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **array-like** or **Quantity*) – User-specified 3D radii \(r\). The unit is the scale radius
\(r_0\) in the internal and the model systems, and the parsec
in the physical system (Model.to). In the physical system, an
Astropy quantity of any length unit is also accepted.

Returns:

**sigma_t** – Tangential velocity dispersion \(\sigma_{\rm t}\), with the
shape of r. The square \(\sigma_{\rm t}^2\) equals
\(2 (1 - \beta_\sigma) \sigma_r^2\), with \(\beta_\sigma\)
the velocity anisotropy and \(\sigma_r\) the radial velocity
dispersion. In the internal system, a plain array in internal
units. In the model system, a plain array in units of
\((G M_0/r_0)^{1/2}\). In the physical system, an Astropy
quantity in km/s.

Return type:

array or Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is neither a Model nor a LosTable.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If source is a Model with several components in one potential.
    A model of one component and a superposition (superpose) are
    accepted.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If source is a Model whose relative error is above
    \(10^{-2}\) and whose settings do not set
    `write_anyway=True`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a radius lies inside the innermost or beyond the outermost
    radius that the model tabulates, or beyond the edge of a spatially
    finite model. The call does not extrapolate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If source is a Model and an intermediate result fails a check
    that the next step of the computation needs, such as values that
    are not finite for a degenerate model.

See also

[`sigma_r`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_r)
Radial velocity dispersion at 3D radii.

[`sigma_los`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.sigma_los)
Line-of-sight velocity dispersion at projected radii.

Notes

For a superposition of components \(k\) with weights \(w_k\),
densities \(\rho_k\) and tangential velocity dispersions
\(\sigma_{{\rm t},k}\),

\[\sigma_{\rm t}^2 = \frac{\sum_k w_k \rho_k \sigma_{{\rm t},k}^2}
{\sum_k w_k \rho_k}.\]

References

Notes Section T:6.1 (intrinsic moments) and Notes Section T:6.9
(superposed models).

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> pp.sigma_t(model, [0.1, 1.0, 10.0]).shape
(3,)
```

## Velocity distribution at one radius

Module `pianistpy.observables.stages`

`velocity_distribution` computes the local velocity distribution at one radius, with its accuracy report;
[`pianistpy.observables.pdf_v()`](https://pianistpy.com/docs/reference/api/observables#pianistpy.observables.pdf_v) computes the same distribution at many radii ([Local velocity distribution](https://pianistpy.com/docs/guide/pdf_v)).

pianistpy.observables.stages.velocity_distribution(*model*, *r*, *settings*, *profile=None*)

Return the local velocity distribution of a model at one 3D radius.

The distribution \(p(V\,|\,R)\) is the probability density of the
velocity \(V\) of the model’s particles at the 3D radius \(R\). The
call returns the distribution at one radius, together with the checks
of the distribution and their validation report. pdf_v evaluates the
same distribution at many radii and velocities, in the model’s unit system.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – The model. The model has one component.

- **r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – 3D radius \(R\). The radius is in internal units in all three
unit systems: a model set to the model system or to the physical
system (Model.to) still takes r in internal units. An Astropy
quantity is not accepted. The radius must lie at or beyond the
innermost radius of the model’s tabulated profile and below the
outermost radius of the model’s tabulated profile. For a Type II
model, the radius must also lie below the anisotropy radius
\(R_{\rm a}\). A radius outside that range raises InputError.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – The numerical settings. The call reads three fields of the
`observables` group: `n_velocity_angle_nodes`,
`n_velocity_angle_panels` and `velocity_tolerance`. The settings table
([Numerical settings](https://pianistpy.com/docs/reference/settings)) describes the three fields.

- **profile** (*DispersionProfile**, **optional*) – The intrinsic velocity dispersions of model, as the attribute
`profile` of `pp.LosTable(model)` holds them. The second moment
of the distribution is checked against these dispersions. Default:
`None`, which means the call computes the intrinsic velocity
dispersions of model.

Returns:

**distribution** – The distribution at r. Every value is in internal units in all
three unit systems of the model: velocities in the internal unit of
velocity, and the distribution per unit internal velocity. The
object holds the 3D radius `r`; the velocity nodes `v_nodes` and
their quadrature weights `v_weights`; the distribution `pdf` at
the nodes; the escape velocity `v_esc`; the largest velocity of the
distribution’s support `v_max`; and the share of the distribution
above the escape velocity `unbound_fraction`. The method
`read(v)` returns the distribution at any velocities v, and zero at
negative velocities and at velocities above `v_max`. The integral of the
distribution over velocity is 1 to within `record["norm"]`.
`record` holds the relative errors of three checks (`"norm"`,
`"second_moment"` and `"halving"`), their validation report
`record["report"]` and the report’s tier `record["tier"]`. The
call raises no error for a large check error: a check error above
\(10^{-2}\) gives the tier `"refused"` in `record["tier"]`.

Return type:

VelocityDistribution

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model has more than one component.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r lies inside the innermost radius or at or beyond the
    outermost radius of the model’s tabulated profile.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is a Type II model and r lies at or beyond the
    anisotropy radius \(R_{\rm a}\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r lies outside the radii at which the intrinsic velocity
    dispersions, given as profile or computed by the call, are
    tabulated.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If profile is not given and the call cannot form the model’s
    radial velocity dispersion: the contribution of the DF below the
    tabulated potential is not finite, or the dispersion is positive at
    no tabulated radius.

See also

`pdf_v`
The local velocity distribution at many radii and velocities, in the model’s unit system.

`sigma_r`
The radial velocity dispersion, which the second-moment check reads.

`LosTable`
The tables of a model, whose attribute `profile` can be passed as profile.

Notes

The distribution is

\[p(V\,|\,R) = \frac{4\pi}{\varrho(R)}\, R^{-2b} V^{2-2b}
\int_0^1 (1-t^2)^{-b} F(\mathcal{Q})\,{\rm d}t,\]

with \(\varrho\) the density, \(F\) the distribution function
(DF), \(t = \cos\eta\) and \(\eta\) the angle between the
velocity and the radius vector. The exponent \(b\) is the central
anisotropy \(\beta_{\rm c}\) for a constant-anisotropy model and for
a Cuddeford-Osipkov-Merritt (COM) model, and \(b = 0\) for an
isotropic model and for an Osipkov-Merritt model. The argument of the
DF is

\[\mathcal{Q} = P(R) - \frac{V^2}{2}(1 + s\xi)
+ \frac{s\xi V^2}{2}\,t^2,\]

with \(P\) the relative potential, \(\xi = R^2/R_{\rm a}^2\),
and \(s = +1\) for a Type I model, \(s = -1\) for a Type II
model and \(s = 0\) for a model without an anisotropy radius. The
integral runs over the DF’s support, \(\mathcal{Q} > K_{\rm c}\),
with \(K_{\rm c}\) the DF’s floor energy (0 where the DF has no
floor).

The support excludes part of the directions above the velocity
\(V_*\): \(V_* = \sqrt{2(P - K_{\rm c})/(1 + \xi)}\) for a
Type I model, and \(V_* = \sqrt{2(P - K_{\rm c})}\) for a Type II
model. The support ends at
\(V_{\rm max} = \sqrt{2(P - K_{\rm c})/(1 - \xi)}\) for a Type II
model and at \(V_{\rm max} = \sqrt{2(P - K_{\rm c})}\) for every
other model. The escape velocity is \(V_{\rm esc} = \sqrt{2P}\). For a
Type II model, \(V_{\rm max}\) is above \(V_{\rm esc}\) where
\(K_{\rm c}\) is below \(\xi P(R)\). unbound_fraction is the
share of the distribution between \(V_{\rm esc}\) and
\(V_{\rm max}\).

The angle integral starts with `n_velocity_angle_nodes` nodes per
interval. The nodes double, at most three times, until the largest
change of the distribution between the integral and the integral with
doubled nodes is at most `velocity_tolerance` times the largest absolute
value of the distribution with doubled nodes. `record["halving"]` is
the ratio of the two maxima for the kept distribution.
`velocity_tolerance` is \(10^{-6}\) in the default settings and
\(10^{-4}\) in the quick settings. Velocities within a
relative distance of \(10^{-9}\) from \(V_*\) are left out of
that comparison. On a DF that diverges at its floor energy, the
distribution diverges at \(V_*\), and the divergence is integrable.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (B38). [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (B39). [Chiang+(2026d)](https://pianistpy.com/docs/project/citing),
Eq. (B40). Notes Eq. (T:6.4).

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.observables.stages import velocity_distribution
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> distribution = velocity_distribution(model, 1.0, model.settings)
>>> distribution.pdf.shape == distribution.v_nodes.shape
True
```
