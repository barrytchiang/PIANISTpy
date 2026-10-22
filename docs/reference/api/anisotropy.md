# Velocity anisotropy

Source: https://pianistpy.com/docs/reference/api/anisotropy

Module `pianistpy.anisotropy`

Velocity anisotropy models of a component’s distribution function (DF).

A user builds the anisotropy of a component with Anisotropy and passes
the Anisotropy object to Component. One class covers four models: the
isotropic model, the constant-anisotropy model, the Osipkov-Merritt (OM)
model and the Cuddeford-Osipkov-Merritt (COM) model. The COM model of central
anisotropy \(\beta_{\rm c}\) and anisotropy radius \(R_{\rm a}\) has
the anisotropy profile

\[\beta(R) = \frac{\beta_{\rm c} \pm R^2/R_{\rm a}^2}{1 \pm R^2/R_{\rm a}^2},\]

with the upper sign for Type I and the lower sign for Type II, and
\(R\) the radius. The OM model is the COM model with
\(\beta_{\rm c} = 0\). The constant-anisotropy model is the COM model
with an infinite \(R_{\rm a}\), and the isotropic model is the
constant-anisotropy model with \(\beta_{\rm c} = 0\). The factor
\(g(R)\) that relates the augmented density to the auxiliary density is
given in [Chiang+(2026d)](https://arxiv.org/abs/2610.28758), Eq. (B44).

class pianistpy.anisotropy.Anisotropy(*beta_c=0.0*, *r_a=inf*, *kind='I'*)

Velocity anisotropy of one component.

The isotropic, constant-anisotropy, Osipkov-Merritt (OM) and
Cuddeford-Osipkov-Merritt (COM) models are all instances of
Anisotropy, since the COM model contains the other three models as
special cases. The anisotropy profile of the COM model is

\[\beta(R) = \frac{\beta_{\rm c} \pm R^2/R_{\rm a}^2}
{1 \pm R^2/R_{\rm a}^2},\]

with the upper sign for Type I and the lower sign for Type II.

Parameters:

- **beta_c** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Central anisotropy \(\beta_{\rm c}\). beta_c must be smaller
than 1. Default: `0.0`, which means an isotropic model or an OM
model.

- **r_a** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Anisotropy radius \(R_{\rm a}\), in units of \(r_0\), the
unit of length of the internal and the model unit systems. r_a
is a plain float, and an Astropy quantity is not accepted. r_a
must be positive. Default: `math.inf`, which means an isotropic model or a
constant-anisotropy model.

- **kind** (*{"I"**, **"II"}**, **optional*) – Type of the OM or COM model. The anisotropy of a Type I model rises
outward to \(\beta = 1\) (radial orbits) as \(R\) grows. The
anisotropy of a Type II model falls to \(\beta \to -\infty\)
(tangential orbits) as \(R\) approaches \(R_{\rm a}\), and a
Type II model is defined only at \(R < R_{\rm a}\). When r_a
is infinite, the type does not enter the model, and the type is
stored as `"I"` so that each model has one identity. Default:
`"I"`, which means a Type I model.

Variables:

- **beta_c** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Central anisotropy \(\beta_{\rm c}\).

- **r_a** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Anisotropy radius \(R_{\rm a}\), in units of \(r_0\);
`math.inf` for an isotropic or a constant-anisotropy model.

- **kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Type of the model, `"I"` or `"II"`.

- **family** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model and type together: `"isotropic"`, `"constant"`,
`"OM_I"`, `"OM_II"`, `"COM_I"` or `"COM_II"`.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If beta_c is 1 or larger.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_a is zero or negative.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If kind is neither `"I"` nor `"II"`.

See also

`Component`
A density profile with its anisotropy.

[`Anisotropy.factor`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.factor)
The factor \(g(R)\) of the augmented density.

Notes

Anisotropy decides the model by exact comparison of the given
parameters with `beta_c == 0` and with an infinite r_a. A value such
as
\(\beta_{\rm c} = 10^{-12}\) gives a constant-anisotropy model and
is not rounded to zero. The results of the package are continuous
across \(\beta_{\rm c} = 0\) and across an infinite
\(R_{\rm a}\).

References

[Chiang+(2026d)](https://arxiv.org/abs/2610.28758), Eq. (B44).

Examples

```python
>>> from pianistpy import Anisotropy
>>> Anisotropy().model
'isotropic'
>>> Anisotropy(beta_c=0.3).model
'constant'
>>> Anisotropy(r_a=10.0).model
'om'
>>> Anisotropy(beta_c=-0.25, r_a=10.0, kind="II").model
'com'
```

factor(*r*, *xp=np*)

Return the factor \(g(R)\) of the augmented density.

The factor \(g(R)\) relates the augmented density to the
auxiliary density, \(\varrho_{\rm aug} = g(R)\,\varrho_{\rm aux}\).
The package uses the factor \(g(R)\) in the inversion of the
density, at the truncation edge and in the solution of the Poisson
equation.

Parameters:

- **r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **array-like*) – User-specified 3D radius \(R\), in internal units (units of
\(r_0\)). r is a plain float or array, and an Astropy
quantity is not accepted.

- **xp** (*module**, **optional*) – The array namespace, `numpy` or `jax.numpy`. Default:
`numpy`, which means the factor is a NumPy array.

Returns:

**g** – The factor \(g(R)\), dimensionless, with the shape of r:
\(1\) for the isotropic model,
\(R^{-2\beta_{\rm c}}\) for the constant-anisotropy model,
\((1 \pm R^2/R_{\rm a}^2)^{-1}\) for the OM model and
\(R^{-2\beta_{\rm c}} (1 \pm R^2/R_{\rm a}^2)^{-(1-\beta_{\rm c})}\)
for the COM model, with the upper sign for Type I and the lower
sign for Type II.

Return type:

array

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the model is Type II and a radius in r is equal to or larger
    than \(R_{\rm a}\), where the factor is not defined. Inside a
    JAX transformation such as `jax.jit`, the radius is not
    checked.

See also

[`Anisotropy.domain_edge`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.domain_edge)
The largest radius where the factor is defined.

References

[Chiang+(2026d)](https://arxiv.org/abs/2610.28758), Eq. (B44).

Examples

```python
>>> from pianistpy import Anisotropy
>>> float(Anisotropy(beta_c=0.5).factor(4.0))
0.25
>>> Anisotropy(r_a=1.0).factor([1.0]).tolist()
[0.5]
```

property model: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

`"isotropic"`, `"constant"`, `"om"` or `"com"`.

The names are those of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758). The name does not include the
type, which is in kind.

See also

[`Anisotropy.to_dict`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.to_dict)
The dictionary form, which records the name.

Examples

```python
>>> from pianistpy import Anisotropy
>>> Anisotropy(beta_c=0.25, r_a=10.0).model
'com'
```

Type:

Name of the model

property domain_edge: [float](https://docs.python.org/3/builtins/functions.html#float)

Largest radius of the anisotropy’s domain, in units of \(r_0\).

For a Type II model with a finite \(R_{\rm a}\), the domain edge
is \(R_{\rm a}\). The factor \(g(R)\) of a Type II model is
real, finite and positive only at \(R < R_{\rm a}\). For every
other model, the
domain edge is infinite.

See also

[`Anisotropy.factor`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.factor)
The factor \(g(R)\).

Examples

```python
>>> from pianistpy import Anisotropy
>>> Anisotropy(r_a=10.0, kind="II").domain_edge
10.0
>>> Anisotropy(r_a=10.0).domain_edge
inf
```

to_dict()

Return the anisotropy as a plain dictionary.

The dictionary holds only JSON types, so that a model file can store
the anisotropy as text.

Returns:

**record** – The keys `"model"` (as model), `"beta_c"`, `"r_a"` and
`"kind"`. The value of `"r_a"` is None for an infinite
anisotropy radius.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Anisotropy.from_dict`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.from_dict)
The anisotropy built back from the dictionary.

Examples

```python
>>> from pianistpy import Anisotropy
>>> Anisotropy(beta_c=0.3).to_dict()
{'model': 'constant', 'beta_c': 0.3, 'r_a': None, 'kind': 'I'}
```

classmethod from_dict(*record*)

Return the anisotropy that a dictionary describes.

Parameters:

**record** (*mapping*) – The dictionary form of the anisotropy, as to_dict returns it.
The key `"model"` is not read. A missing key takes the default
of the matching parameter of Anisotropy, and an `"r_a"` of
None means an infinite anisotropy radius.

Returns:

**anisotropy** – The anisotropy that record describes.

Return type:

[Anisotropy](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If `"beta_c"` is 1 or larger, `"r_a"` is zero or negative,
    or `"kind"` is neither `"I"` nor `"II"`.

See also

[`Anisotropy.to_dict`](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy.to_dict)
The dictionary form of the anisotropy.

Examples

```python
>>> from pianistpy import Anisotropy
>>> Anisotropy.from_dict({"beta_c": 0.3}) == Anisotropy(beta_c=0.3)
True
```
