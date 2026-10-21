# Superposition

Source: https://pianistpy.com/docs/reference/api/superposition

Module `pianistpy.superposition`

Linear superposition of models that share one truncated density.

superpose combines models built on one truncated potential-density pair
into one model, moments returns the velocity moments of the combined model,
and mixture combines any per-component quantity by the weights.

pianistpy.superposition.mixture(*weights*, *values*, *xp=np*)

Return the weighted sum of per-component values, with the weights normalized.

A superposed model combines a density, a velocity moment or any other
quantity computed per component by the weighted sum mixture returns.
The sum is linear in the values and, through the normalization, a smooth
function of the
weights. With `xp=jax.numpy` the sum is differentiable in the weights
by JAX.

Parameters:

- **weights** (*array-like**, **shape** (**n_components**,**)*) – The weights \(w_k\), positive. The call divides the weights by
the weights’ sum and does not check the weights.

- **values** (*array-like**, **shape** (**n_components**, **...**)*) – The value \(v_k\) of each component, along the first axis.

- **xp** (*module**, **optional*) – The array module, `numpy` or `jax.numpy`. Default: `np`, which
means NumPy arrays in and out.

Returns:

**value** – The sum \(\sum_k w_k v_k / \sum_k w_k\), with the shape of one
component’s value, as an array of xp.

Return type:

array, shape (…)

See also

[`superpose`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.superpose)
The model that combines components by the weights.

[`moments`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.moments)
The velocity moments of a superposed model.

Examples

```python
>>> from pianistpy.superposition import mixture
>>> mixture([1.0, 3.0], [[2.0, 4.0], [6.0, 8.0]])
array([5., 7.])
```

pianistpy.superposition.moments(*model*, *r*, *n_nodes=None*)

Return the second velocity moments of a superposed model at 3D radii.

Parameters:

- **model** ([*Model*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)) – A superposed model (superpose) whose components share one spatially
finite density.

- **r** (*array-like**, **shape** (**n**,**)*) – User-specified 3D radii, in internal units whatever the model’s unit
system.

- **n_nodes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – The number of Gauss-Legendre nodes of each component’s Jeans
integral. Default: `None`, which means the value of the model’s
setting `observables.n_jeans_nodes`.

Returns:

**moments** – Arrays in internal units, whatever the model’s unit system:
`sigma_r2` and `sigma_t2`, the squared radial and tangential
dispersions of the superposed model, and `beta_sigma`, the
superposed model’s velocity anisotropy, each with the shape of r;
`sigma_r2_components` and `beta_components`, the squared radial
dispersion and the anisotropy of each component, one row per
component along the first axis, each row with the shape of r.
`beta_sigma` is NaN at a radius where `sigma_r2` is not positive.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If model is not a Model.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the density of model is not spatially finite.

See also

[`superpose`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.superpose)
The superposed model.

[`mixture`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.mixture)
The weighted sum the moments are formed by.

Notes

The squared radial dispersion \(\sigma_{r,k}^2\) of component
\(k\) is the Jeans integral of the component’s augmented density on
the shared profile, evaluated by a Gauss-Legendre rule in the variable
\(\ln(1 - \mathcal{K}/P_0)\), with \(\mathcal{K}\) the binding
energy and \(P_0\) the central potential. Because the components
share one density, the moments of the superposed model are the weighted
sums

\[\sigma_r^2 = \sum_k w_k \sigma_{r,k}^2, \qquad
\sigma_t^2 = 2 \sum_k w_k (1 - \beta_k) \sigma_{r,k}^2, \qquad
\beta_\sigma = 1 - \frac{\sigma_t^2}{2 \sigma_r^2},\]

with \(w_k\) the normalized weight and \(\beta_k\) the
velocity anisotropy of component \(k\).

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (B43), [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (B44) and
[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (B74).

Examples

```python
>>> import pianistpy as pp
>>> settings = pp.Settings(quick=True)
>>> profile = pp.DoublePowerLaw.nfw()
>>> parent = pp.solve(pp.Component(profile), pp.Truncation(k_t=0.05), settings)
>>> children = [
...     pp.solve(
...         pp.Component(parent.profile, pp.Anisotropy(beta_c=b)), None, settings
...     )
...     for b in (0.0, 0.2)
... ]
>>> model = pp.superpose({children[0]: 0.5, children[1]: 0.5})
>>> result = pp.superposition.moments(model, [0.5, 1.0])
>>> result["sigma_r2"].shape, result["sigma_r2_components"].shape
((2,), (2, 2))
```

pianistpy.superposition.superpose(*members*)

Return the superposition of models built on one truncated profile.

The returned model is an equilibrium of the shared pair, because every
member’s distribution function was inverted from the shared density.

Parameters:

**members** (*mapping** of **Model to float*) – Two or more models, each of one component and each built on the same
truncated profile (for example, Method II children of one truncated
parent), mapped to the models’ weights. Each weight is positive and
finite. The weights are normalized to sum to 1, with a logged warning
where the weights’ sum differs from 1 by more than \(10^{-10}\).

Returns:

**model** – A model with one component per member, in the order of members.
Each component is the member’s component, with a mass equal to the
member’s weight divided by the first member’s weight, and with the
member’s DF table on the member’s own energy grid. The model’s
profile is the shared truncated profile with the density split among
the components by weight. Each check of the model’s
ValidationReport holds the largest error of the check over the
members.

Return type:

[Model](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If members is not a mapping.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If fewer than two members are given.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a member is not a Model, or has more than one component.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a member’s profile is not a truncated profile with
    \(\mathcal{K}_{\rm t} > 0\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a weight is not positive and finite.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the members are built on different truncated profiles.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the members were computed with different settings, or are in
    different units.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a member’s DF is signed.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the members’ central values of the potential differ.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If some members’ distribution functions come from closed forms
    (directly or through a closed-form parent) and the other members’
    distribution functions were computed by inversion. Build every member
    the same way, from closed forms or by inversion, to superpose the
    members.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the members’ validation reports hold different sets of checks.

See also

[`moments`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.moments)
The velocity moments of the superposed model.

[`mixture`](https://pianistpy.com/docs/reference/api/superposition#pianistpy.superposition.mixture)
The weighted sum of per-component quantities.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Appendix B5; Notes Section T:4.

Examples

```python
>>> import pianistpy as pp
>>> settings = pp.Settings(quick=True)
>>> profile = pp.DoublePowerLaw.nfw()
>>> parent = pp.solve(pp.Component(profile), pp.Truncation(k_t=0.05), settings)
>>> children = [
...     pp.solve(
...         pp.Component(parent.profile, pp.Anisotropy(beta_c=b)), None, settings
...     )
...     for b in (0.0, 0.2)
... ]
>>> model = pp.superpose({children[0]: 0.5, children[1]: 0.5})
>>> len(model.components), [c.mass for c in model.components]
(2, [1.0, 1.0])
```
