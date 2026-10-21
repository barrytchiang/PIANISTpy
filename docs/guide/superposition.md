# Superposition and inference

Source: https://pianistpy.com/docs/guide/superposition

`pianistpy.superpose()`, [`pianistpy.inference`](https://pianistpy.com/docs/reference/api/inference#module-pianistpy.inference)

## Superposition

Method II models built on the truncated density of one Method I model
([Truncation in phase space](https://pianistpy.com/docs/guide/truncation)) share that density. The distribution functions of those Method II models, combined with
positive weights that sum to one, form one equilibrium model in that truncated density. `superpose` returns the combined model as a
`Model`, which `Sampler`, the observables, `likelihood` and `write` take as any other model.
`Differentiable` raises `InputError` for a combined model, whose components are differentiated
through `likelihood`.

```python
members = {model_a: 0.6, model_b: 0.3, model_c: 0.1}   # Method II models on one truncated density
composite = pp.superpose(members)
pp.Sampler(composite).sample(1_000_000, seed=1)
```

`superpose` takes a mapping of two or more Method II models on one truncated density, the members, to the members’
weights.
`superpose` raises `InputError` when a weight is not positive and finite, or when the members
are built on different truncated profiles or with different settings or units. The weights are normalized to sum to
one, and a warning goes to the package’s log when the sum of the weights differs from one by more than
\(10^{-10}\).

## Inference

[`pianistpy.inference`](https://pianistpy.com/docs/reference/api/inference#module-pianistpy.inference) gives the log-likelihood of a model against data as a JAX function. The function’s arguments
are the weights of the components, the mass scale and the length scale in the units of the data, and, for the
components listed in `free=`, the anisotropy \((\beta_{\rm c}, R_{\rm a})\). Each modeled dispersion is a
ratio of two sums that are linear in the weights, so the gradient with respect to the weights is exact and needs no
new table.

```python
from pianistpy.inference import Data, likelihood

data = Data("sigma_los", values=sigma_obs, r_2d=radii, errors=sigma_err, unit_system="physical")
loglike = likelihood(composite.to("physical", concentration=..., mass_0=..., r_0=...), data)
loglike(weights)                       # a JAX scalar; jax.grad(loglike)(weights) is its gradient
```

`likelihood` computes the tables of the observables of each component once. Each later call of the returned
function takes a fraction of a millisecond, except a call at a new anisotropy of a free component, which first
rebuilds the free component’s tables. An invalid argument of the returned function, such as a negative weight, a
scale that is not positive, or a length scale that moves a data radius inside the innermost tabulated radius,
gives minus infinity, not an exception, so a Monte Carlo sampler over the weights always receives a value.
A model and data that cannot be compared raise `InputError` in `likelihood` itself, never in a
call of the returned function.

[End-to-end differentiability](https://pianistpy.com/docs/guide/differentiability) states what the likelihood is differentiable in, through which observables, and at what accuracy.
