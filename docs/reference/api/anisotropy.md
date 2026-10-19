# Velocity anisotropy (pianistpy.anisotropy)

Source: https://pianistpy.com/docs/reference/api/anisotropy

# Velocity anisotropy (pianistpy.anisotropy)

Velocity anisotropy models.

class pianistpy.anisotropy.Anisotropy(beta_c=0.0, r_a=inf, kind='I')

Velocity anisotropy of one component.

The isotropic, constant-anisotropy, Osipkov-Merritt (OM) and
Cuddeford-Osipkov-Merritt (COM) models are regions of the parameters of
this one class, since the COM model contains the other three.

Parameters:

beta_c (float, optional) – Central anisotropy \(\beta_{\rm c}\), smaller than 1. The default,
exactly 0, gives an isotropic or an OM model.

r_a (float, optional) – Anisotropy radius \(R_{\rm a}\) in internal units, positive. The
default, infinity, gives an isotropic or a constant-anisotropy model.

kind ({"I", "II"}, optional) – Type of the OM or COM model. Ignored when r_a is infinite.

Notes

The model is identified by exact comparison of the parameters as given:
beta_c == 0 and an infinite r_a. A value such as 1e-12 is a
constant-anisotropy model and is not rounded to zero. The package is
tested to give results that are continuous through these special values.

Examples

```
>>> Anisotropy().model
'isotropic'
>>> Anisotropy(beta_c=0.3).model
'constant'
>>> Anisotropy(r_a=10.0).model
'om'
>>> Anisotropy(beta_c=-0.25, r_a=10.0, kind="II").model
'com'

```

factor(r, xp=np)

Return the factor \(g(R)\) of the augmented density.

\(g = 1/h\), with \(h\) from inverse_factor, the package’s
one form of the factor, which the inversion, the edge law and the
Poisson solve read too.

Parameters:

r (array-like) – Radius, in internal units.

xp (module, optional) – The array namespace, NumPy by default; JAX’s inside a kernel.

Returns:

g – \(g(R)\): 1 (isotropic), \(R^{-2\beta_{\rm c}}\)
(constant), \((1 \pm R^2/R_{\rm a}^2)^{-1}\) (OM) and
\(R^{-2\beta_{\rm c}} (1 \pm R^2/R_{\rm a}^2)^{-(1-\beta_{\rm c})}\)
(COM), the upper sign for Type I.

Return type:

array

Raises:

InputError – For a Type II model at a radius at or beyond \(R_{\rm a}\),
    where it is not defined (a traced radius is not checked).

References

Paper Eq. (B43) [eqn:linearity_beta_1].

property model: str

isotropic, constant, om or com.

Type:

Label of the model in the paper

property domain_edge: float

R_a
for a Type II model (its factor is real, finite and positive only
inside it), infinite otherwise.

Type:

The outer edge of the radii where the anisotropy is defined

to_dict()

Return the anisotropy as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build the anisotropy from its dictionary form.

Parameters:

record (Mapping[str, Any])

Return type:

Anisotropy

     On this page
