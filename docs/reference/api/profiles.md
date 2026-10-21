# Density profiles

Source: https://pianistpy.com/docs/reference/api/profiles

Module `pianistpy.profiles`

Density profiles: the analytic input profile and the profile of a model.

A user builds the input profile of a component with DoublePowerLaw, for
example `pp.DoublePowerLaw.nfw()`. The self-consistent profile of a model
is a TruncatedProfile, read as `model.profile`.
TruncatedProfile.interpolation_tables returns the ProfileTables of a
TruncatedProfile.

The input profile is the double power law

\[\varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha},\]

where \(R\) is the radius in units of the scale radius \(r_0\),
\(\varrho\) is the density in internal units, \(\alpha\) is the
sharpness of the transition between the inner and the outer slope,
\(\beta\) is the outer logarithmic slope, \(\gamma\) is the inner
logarithmic slope, and \(C\) is the normalization that sets the central
relative potential \(P(0)\) to 1. Near the center, a model’s profile
takes the depth \(P(0) - P(R)\) from the depth’s series in
\(R^{\alpha}\), out to the largest radius at which the series’ next
term is at most \(10^{-7}\) of the depth. The formulas are in Notes
Section T:1.

class pianistpy.profiles.DoublePowerLaw(*rho_alpha*, *rho_beta*, *rho_gamma*)

Double-power-law density profile, in internal units.

The profile is the input density of a Component. Its methods return
the density, the enclosed mass and the relative potential at
user-specified 3D radii, in closed form. The class methods nfw,
hernquist and plummer return three parameter sets that have names.

Parameters:

- **rho_alpha** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Sharpness \(\alpha\) of the transition between the inner and the
outer slope. Must be at least 1/2.

- **rho_beta** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Outer logarithmic slope \(\beta\). Must be greater than 2. A
Component takes \(\beta \geq 3\) only.

- **rho_gamma** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Inner logarithmic slope \(\gamma\). Must satisfy
\(0 \leq \gamma < 2\).

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If \(\alpha < 1/2\), \(\beta \leq 2\), \(\gamma < 0\) or
    \(\gamma \geq 2\).

See also

`Component`
A profile with its anisotropy, the input of a model.

[`TruncatedProfile`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile)
The self-consistent profile of a model.

Notes

In internal units the scale radius \(r_0\) is 1, the central
relative potential \(P(0)\) is 1, and \(P(R)\) vanishes as
\(R \to \infty\). The density is

\[\varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha},\]

with \(R\) the radius in units of \(r_0\) and \(C\) the
normalization rho_norm. The bound \(\gamma \geq 0\) excludes a
density that decreases toward the center. The bound \(\gamma < 2\)
keeps the central potential finite. The bound \(\beta > 2\) keeps
the integral of \(\varrho R\) over \(R\) from 0 to infinity
finite, so that \(P(0)\) is finite.

The profiles with names are parameter sets of the double-power-law
profile. Where a profile with a name has a closed form for the profile’s mass or
potential in elementary functions, mass and p evaluate the
elementary closed form.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (5), defines the profile. [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (A6),
gives \(C\). [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (A7), gives the enclosed mass.

Examples

```python
>>> import pianistpy as pp
>>> pp.DoublePowerLaw.nfw()
DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
>>> pp.DoublePowerLaw.hernquist().mass_total
1.0
>>> float(pp.DoublePowerLaw.plummer().p(0.0))
1.0
```

classmethod nfw()

Return the Navarro-Frenk-White profile.

Returns:

**profile** – The profile with
\((\alpha, \beta, \gamma) = (1, 3, 1)\).

Return type:

[DoublePowerLaw](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)

See also

[`hernquist`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.hernquist)
The Hernquist profile.

[`plummer`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.plummer)
The Plummer profile.

Examples

```python
>>> import pianistpy as pp
>>> pp.DoublePowerLaw.nfw()
DoublePowerLaw(rho_alpha=1.0, rho_beta=3.0, rho_gamma=1.0)
```

classmethod hernquist()

Return the Hernquist profile.

Returns:

**profile** – The profile with
\((\alpha, \beta, \gamma) = (1, 4, 1)\).

Return type:

[DoublePowerLaw](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)

See also

[`nfw`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.nfw)
The Navarro-Frenk-White profile.

[`plummer`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.plummer)
The Plummer profile.

Examples

```python
>>> import pianistpy as pp
>>> pp.DoublePowerLaw.hernquist()
DoublePowerLaw(rho_alpha=1.0, rho_beta=4.0, rho_gamma=1.0)
```

classmethod plummer()

Return the Plummer profile.

Returns:

**profile** – The profile with
\((\alpha, \beta, \gamma) = (2, 5, 0)\).

Return type:

[DoublePowerLaw](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)

See also

[`nfw`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.nfw)
The Navarro-Frenk-White profile.

[`hernquist`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.hernquist)
The Hernquist profile.

Examples

```python
>>> import pianistpy as pp
>>> pp.DoublePowerLaw.plummer()
DoublePowerLaw(rho_alpha=2.0, rho_beta=5.0, rho_gamma=0.0)
```

property name: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

Name of the parameter set, or the three slopes if the set has no name.

property rho_norm: [float](https://docs.python.org/3/builtins/functions.html#float)

Normalization \(C\) of the density that sets \(P(0) = 1\).

Notes

\[C = \frac{\alpha\,\Gamma[(\beta-\gamma)/\alpha]}
         {\Gamma[(\beta-2)/\alpha]\,\Gamma[(2-\gamma)/\alpha]} .\]

property mass_total: [float](https://docs.python.org/3/builtins/functions.html#float)

Total mass in internal units, infinite for an outer slope of 3 or less.

rho(*r*)

Return the density \(\varrho(R)\).

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), in internal units (units of
the scale radius \(r_0\)).

Returns:

**rho** – Density \(\varrho(R)\) in internal units, with the shape of
r.

Return type:

array

See also

[`mass`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.mass)
The enclosed mass.

[`p`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.p)
The relative potential.

Notes

\[\varrho(R) = C\,R^{-\gamma}\,(1 + R^{\alpha})^{(\gamma-\beta)/\alpha}.\]

Examples

```python
>>> import pianistpy as pp
>>> float(pp.DoublePowerLaw.hernquist().rho(1.0))
0.25
```

mass(*r*)

Return the enclosed mass \(\mathcal{M}(R)\).

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), in internal units (units of
the scale radius \(r_0\)).

Returns:

**mass** – Enclosed mass \(\mathcal{M}(R)\) in internal units, with the
shape of r. The mass is the integral of
\(\varrho R^2\) over \(R\) from 0 to r, that is,
\(1/(4\pi)\) times the integral of the density over the
volume.

Return type:

array

See also

[`mass_total`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.mass_total)
The total mass.

[`rho`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.rho)
The density.

[`p`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.p)
The relative potential.

Notes

With \(z = R^\alpha/(1+R^\alpha)\) and the incomplete Beta
function \(B(z; a, b)\),

\[\mathcal{M}(R) = \frac{C}{\alpha}\,
    B\!\left(z;\,\frac{3-\gamma}{\alpha},\,\frac{\beta-3}{\alpha}\right) .\]

For \(\beta = 3\) the second argument of \(B\) is zero, and
the mass is evaluated from a logarithm and a series. For
\(\beta < 3\) the second argument is negative, and the analytic
continuation of the formula is evaluated in arbitrary precision.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (A7).

Examples

```python
>>> import pianistpy as pp
>>> float(pp.DoublePowerLaw.hernquist().mass(1.0))
0.25
```

p(*r*)

Return the relative potential \(P(R)\).

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), in internal units (units of
the scale radius \(r_0\)).

Returns:

**p** – Relative potential \(P(R)\) in internal units, with the shape
of r. \(P(0) = 1\), and \(P(R)\) vanishes as
\(R \to \infty\).

Return type:

array

See also

[`depth_inner`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.depth_inner)
The depth \(P(0) - P(R)\) near the center.

[`mass`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.mass)
The enclosed mass.

[`rho`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.rho)
The density.

Notes

With \(z = R^\alpha/(1+R^\alpha)\), \(a_2 = (2-\gamma)/\alpha\),
\(b_2 = (\beta-2)/\alpha\), the Beta function \(B(a, b)\) and
the incomplete Beta function \(B(z; a, b)\),

\[P(R) = \frac{\mathcal{M}(R)}{R}
     + \frac{C}{\alpha}\left[B(a_2, b_2) - B(z;\,a_2, b_2)\right] .\]

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (A3).

Examples

```python
>>> import pianistpy as pp
>>> float(pp.DoublePowerLaw.hernquist().p(1.0))
0.5
```

depth_inner(*r*, *terms=4*)

Return the depth \(u = P(0) - P(R)\) near the center, from its series.

The depth is summed term by term from its series in
\(R^{\alpha}\), with no subtraction from \(P(0)\), so the
depth keeps its relative precision where \(u\) is small.
depth_inner also returns the size of the series’ next term
relative to the depth, as an estimate of the truncation error of
the sum.

Parameters:

- **r** (*array-like*) – User-specified 3D radii \(R\), in internal units (units of
the scale radius \(r_0\)).

- **terms** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – Number of terms of the series summed. Must be at least 1.
Default: `4`, which means the first four terms.

Returns:

- **u** (*array*) – Depth \(u\) from the first terms terms, in internal units,
with the shape of r. Infinite at \(R \geq 1\), where the
series in \(R^{\alpha}\) does not converge.

- **bound** (*array*) – Magnitude of the series’ next term divided by u, with the shape
of r. Infinite where u is not positive or not finite.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

See also

[`p`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.p)
The relative potential in closed form.

Notes

The density term \(C\binom{e}{k}R^{-\gamma+k\alpha}\), with
\(e = (\gamma-\beta)/\alpha\), contributes the term
\(C\binom{e}{k}R^{2-\gamma+k\alpha}/[(2-\gamma+k\alpha)(3-\gamma+k\alpha)]\)
to the depth.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (A11), gives the first two terms of the series.

Examples

```python
>>> import pianistpy as pp
>>> u, bound = pp.DoublePowerLaw.hernquist().depth_inner([1e-3, 2.0])
>>> round(float(u[0]), 9)
0.000999001
>>> bool(bound[1] == float("inf"))
True
```

to_dict()

Return the profile as a plain dictionary.

Returns:

**record** – The key `"profile"` with the value `"double_power_law"`, and
the keys `"rho_alpha"`, `"rho_beta"` and `"rho_gamma"` with
the three slopes. from_dict builds the profile back from the
dictionary.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`from_dict`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.from_dict)
The profile from its dictionary.

Examples

```python
>>> import pianistpy as pp
>>> record = pp.DoublePowerLaw.hernquist().to_dict()
>>> record["profile"], record["rho_beta"]
('double_power_law', 4.0)
```

classmethod from_dict(*record*)

Return the profile that a dictionary from to_dict describes.

Parameters:

**record** (*mapping*) – The dictionary that to_dict returns. The keys `"rho_alpha"`,
`"rho_beta"` and `"rho_gamma"` are read; other keys are
ignored.

Returns:

**profile** – The profile with the slopes of record.

Return type:

[DoublePowerLaw](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a slope in record lies outside its admissible range (see
    DoublePowerLaw).

See also

[`to_dict`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.to_dict)
The dictionary of a profile.

Examples

```python
>>> import pianistpy as pp
>>> record = {"rho_alpha": 1.0, "rho_beta": 3.0, "rho_gamma": 1.0}
>>> pp.DoublePowerLaw.from_dict(record).name
'NFW'
```

class pianistpy.profiles.ProfileTables(*log_r*, *log_p*, *log_p_slope*, *log_r_all*, *log_mass*, *log_mass_slope*, *log_rho_coefficients*, *has_edge_row*, *r_last*, *p_last*, *rho_last*, *r_t*, *mass_t*, *edge_exponent*)

Arrays from which a JAX function evaluates a TruncatedProfile.

The arrays are the numbers from which TruncatedProfile.potential,
TruncatedProfile.density, TruncatedProfile.mass and
TruncatedProfile.radius evaluate the profile, laid out so that a JAX
function takes them as arguments and is differentiable in them. A node
is one tabulated radius of the profile. An interior node is any node
other than the edge node of a spatially finite profile.

Parameters:

- **log_r** (*array**, **shape** (**n_interior**,**)*) – \(\ln R\) at the interior nodes, with \(R\) the radius in
internal units.

- **log_p** (*array**, **shape** (**n_interior**,**)*) – \(\ln P\) at the interior nodes, with \(P\) the relative
potential in internal units.

- **log_p_slope** (*array**, **shape** (**n_interior**,**)*) – \(\mathrm{d}\ln P/\mathrm{d}\ln R\) at the interior nodes.

- **log_r_all** (*array**, **shape** (**n**,**)*) – \(\ln R\) at every node.

- **log_mass** (*array**, **shape** (**n**,**)*) – \(\ln \mathcal{M}\) at every node, with \(\mathcal{M}\) the
enclosed mass in internal units.

- **log_mass_slope** (*array**, **shape** (**n**,**)*) – \(\mathrm{d}\ln \mathcal{M}/\mathrm{d}\ln R\) at every node.

- **log_rho_coefficients** (*array**, **shape** (**n_components**, **4**, **n_interior - 1**)*) – Coefficients of the cubic spline of \(\ln\varrho\) in
\(\ln R\) for each component and each interval between interior
nodes, highest power first, as polynomials in the offset of
\(\ln R\) from the interval’s lower node.

- **has_edge_row** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether the last node of the profile is its edge node.

- **r_last** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Radius of the last interior node, in internal units.

- **p_last** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Relative potential at the last interior node, in internal units.

- **rho_last** (*array**, **shape** (**n_components**,**)*) – Density of each component at the last interior node, in internal
units.

- **r_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Truncation radius \(R_{\rm t}\), in internal units. Infinite
where has_edge_row is False.

- **mass_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Total mass \(\mathcal{M}_{\rm t}\), in internal units. Infinite
where has_edge_row is False.

- **edge_exponent** (*array**, **shape** (**n_components**,**)*) – Exponent \(q\) of each component’s density at the edge,
\(\varrho \propto P^q\). Empty where the model has no edge.

See also

[`TruncatedProfile.interpolation_tables`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.interpolation_tables)
The arrays of a profile.

[`TruncatedProfile`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile)
The profile the arrays evaluate.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> tables = profile.interpolation_tables()
>>> tables.log_r.shape, tables.log_rho_coefficients.shape
((601,), (1, 4, 600))
>>> tables.has_edge_row
False
```

class pianistpy.profiles.TruncatedProfile(*r*, *p_trunc*, *rho_trunc*, *mass_trunc*, *k_t*, *r_t*, *mass_t*, *spatially_finite*, *source*, *source_anisotropy*, *model*, *edge_exponent*, *u_trunc=<factory>*, *central_deficit=()*, *override_r=nan*, *override_r_end=nan*, *override_rows=0*, *override_rule='none'*, *origin=<factory>*, *df_k=<factory>*, *df_log_f=<factory>*, *df_slope=<factory>*, *df_taylor=()*, *model_level=None*)

Self-consistent profile of a model, tabulated at radii called nodes.

The profile of a model is `model.profile`. A component given the
profile of a parent model and a new anisotropy is Method II of
[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Appendix B2.2. The methods potential, density,
mass and radius evaluate the profile between the nodes.

Two profiles are equal if the identity strings model of the two
profiles are equal, because the identity of a model determines its
profile.

Parameters:

- **r** (*array**, **shape** (**n**,**)*) – Radius \(R\) of each node, in internal units. The radii are
positive and strictly increasing, and there are at least two.

- **p_trunc** (*array**, **shape** (**n**,**)*) – Total relative potential \(P\) of the model at each node, in
internal units.

- **rho_trunc** (*array**, **shape** (**n_components**, **n**)*) – Density \(\varrho\) of each component at each node, in internal
units.

- **mass_trunc** (*array**, **shape** (**n**,**)*) – Total enclosed mass \(\mathcal{M}\) at each node, in internal
units, the unit of DoublePowerLaw.mass, in which the total mass of
the Hernquist profile is 1. \(4\pi\) times mass_trunc is the
integral of the density over the volume.

- **k_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Truncation energy \(\mathcal{K}_{\rm t}\), in internal units.

- **r_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Truncation radius \(R_{\rm t}\), in internal units. Infinite
where the model is not spatially finite.

- **mass_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Total mass \(\mathcal{M}_{\rm t}\), in internal units. Infinite
where the model is not spatially finite.

- **spatially_finite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether the density reaches zero at a finite radius with a finite
total mass. A spatially finite profile whose last node has
`p_trunc == 0` holds its edge as that last node, the edge node.
The edge node’s density is zero and its radius is r_t.

- **source** ([*DoublePowerLaw*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)) – The analytic profile of the first component of the model. The scale
radius and the central potential of source define the internal
units of the arrays.

- **source_anisotropy** ([*Anisotropy*](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy)) – The anisotropy of the first component in the model the profile
belongs to. A component built on the profile (Method II) reads its
parent’s anisotropy from source_anisotropy.

- **model** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the model the profile belongs to.

- **edge_exponent** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*float*](https://docs.python.org/3/builtins/functions.html#float)) – Exponent \(q\) of each component’s density at the edge of the
model, \(\varrho \propto P^q\). Empty where the model is not
spatially finite.

- **u_trunc** (*array**, **shape** (**n**,**)**, **optional*) – Depth \(u\), the central potential less the potential, at each
node, in internal units. Near the center p_trunc has lost the
digits of the depth, and u_trunc keeps them. Default: an empty
array, which means the depth was not stored.

- **central_deficit** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – For each component, the coefficients \((D_0, D_1, \ldots, D_4)\)
of the Taylor series \(D_0 - D_1 u + D_2 u^2 - \ldots\) in the
depth \(u\) of the central augmented density of the orbits that
the truncation removed. A record that holds only \(D_0\) and
\(D_1\) is the linear form of the deficit. Default: `()`,
which means the deficit was not recorded.

- **override_r** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Inner radius of the range in which a closed form of the profile
replaced the solution of the Poisson equation, in internal units.
Default: NaN, which means no node was replaced.

- **override_r_end** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Outer radius of the range in which a closed form replaced the
solution of the Poisson equation, in internal units. Default: NaN,
which means no node was replaced.

- **override_rows** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – Number of nodes that a closed form replaced. Default: `0`, which
means no node was replaced.

- **override_rule** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – Name of the closed form that replaced the nodes. Default:
`"none"`, which means no node was replaced.

- **origin** (*array** of **int8**, **shape** (**n**,**)**, **optional*) – Source of each node, as a code from 0 to 6. The code is 0 for the
solution of the Poisson equation, 1 for the continuation of the
Poisson solution to the edge, 2 for the radius at which the
potential reaches zero, 3 for a node filled in at the inner end, 4
for a node filled in at the outer end, 5 for a closed-form outer
node and 6 for the edge node. Default: an empty array, which means
the sources were not recorded.

- **df_k** (*array**, **shape** (**n_components**, **m**)**, **optional*) – Energies \(\mathcal{K}\) of the DF table of the model the
profile belongs to, in internal units. A component built on the
profile (Method II) computes its own DF from the parent’s DF table.
Default: an empty array, which means the table was not recorded.

- **df_log_f** (*array**, **shape** (**n_components**, **m**)**, **optional*) – \(\log_{10} F\) at the energies df_k, with \(F\) the DF in
internal units. Default: an empty array, which means the table was
not recorded.

- **df_slope** (*array**, **shape** (**n_components**, **m**)**, **optional*) – \(\mathrm{d}\log_{10} F/\mathrm{d}\ln\mathcal{K}\) at the
energies df_k. Default: an empty array, which means the table was
not recorded.

- **df_taylor** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – For each component, the coefficients \((F_1, F_2,
\mathcal{K}_{\rm s})\) of the model’s DF near the edge:
\(F = F_1 \mathcal{K} + F_2 \mathcal{K}^2/2\) for
\(\mathcal{K}\) below \(\mathcal{K}_{\rm s}\). Default:
`()`, which means the coefficients were not recorded.

- **model_level** ([*int*](https://docs.python.org/3/builtins/functions.html#int)* or **None**, **optional*) – Level of the validation report of the model the profile belongs to
(ValidationReport.level), set by the call that made the model. A
component built on the profile (Method II) takes model_level as
the level of its parent. Default: `None`, which means the level
was not recorded and is read as level 0.

Raises:

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the radii are fewer than two, not positive or not strictly
    increasing.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If p_trunc or mass_trunc does not have one value per radius, or
    rho_trunc does not have the shape (n_components, n).

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a spatially finite profile does not give the edge exponent of
    every component.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the edge node does not have zero density or does not lie at
    r_t.

See also

`truncate`
The call that makes a model and its profile.

`Component`
A component built on a profile (Method II).

[`DoublePowerLaw`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)
The analytic profile of a component.

Notes

Between the nodes, \(\ln P\) and \(\ln \mathcal{M}\) are cubic
polynomials in \(\ln R\) that match the values and the slopes at
the two nodes of each interval. The slopes are exact, from the Poisson
equation and the mass integral:

\[\frac{\mathrm{d}\ln P}{\mathrm{d}\ln R} = -\frac{\mathcal{M}}{R P},
\qquad
\frac{\mathrm{d}\ln \mathcal{M}}{\mathrm{d}\ln R}
= \frac{R^3\varrho}{\mathcal{M}},\]

with \(\varrho\) the total density. The logarithm of each
component’s density is a cubic spline in \(\ln R\). On the last
interval of a profile with an edge node, the potential is the Kepler
form \(\mathcal{M}_{\rm t}(1/R - 1/R_{\rm t})\), and each
component’s density is \(\varrho_L (P/P_L)^q\), with
\(\varrho_L\) and \(P_L\) the values at the last interior node
and \(q\) from edge_exponent. At and beyond \(R_{\rm t}\), the
relative potential and the density are zero, and the enclosed mass is
\(\mathcal{M}_{\rm t}\). Inside the first node, and beyond the last
node of a profile without an edge node, \(P\) and
\(\mathcal{M}\) continue as the power laws of the end nodes.

References

Notes Section T:1.3.

Examples

The example builds the untruncated Hernquist profile on 601 nodes
directly; a model’s profile is `model.profile`.

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.potential(1.5)), 6)
0.4
```

property has_edge_row: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the last node is the edge node of a spatially finite profile.

property r_table_max: [float](https://docs.python.org/3/builtins/functions.html#float)

Radius of the last node, in internal units; r_t at an edge node.

property mass_at_r_table_max: [float](https://docs.python.org/3/builtins/functions.html#float)

Enclosed mass mass_trunc at the last node, in internal units.

interpolation_tables()

Return the arrays from which a JAX function evaluates the profile.

A JAX function that applies the formulas of potential, density,
mass and radius to the arrays returns the values of those four
methods. Such a JAX function takes the arrays as arguments and is
differentiable in the arrays.

Returns:

**tables** – The logarithms of the potential and of the mass and their slopes
at the nodes, the coefficients of each component’s density
spline, and the values at the last interior node and at the edge.

Return type:

[ProfileTables](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.ProfileTables)

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the potential or a density at an interior node is not
    positive, or the enclosed mass at a node is not positive.

See also

[`ProfileTables`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.ProfileTables)
The arrays and their meaning.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> profile.interpolation_tables().log_r.shape
(601,)
```

potential(*r*)

Return the relative potential \(P(R)\).

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), positive, in internal units.

Returns:

**p** – Relative potential \(P(R)\) in internal units, with the
shape of r. For a profile with an edge node, \(P\) is the
Kepler form \(\mathcal{M}_{\rm t}(1/R - 1/R_{\rm t})\) beyond
the last interior node and zero at and beyond \(R_{\rm t}\).

Return type:

ndarray

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the potential at an interior node is not positive.

See also

[`radius`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.radius)
The radius at a given potential.

[`potential_log_slope`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential_log_slope)
The logarithmic slope of the potential.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.potential(1.0)), 6)
0.5
```

density(*r*)

Return the density \(\varrho(R)\) of each component.

Parameters:

**r** (*array-like**, **shape** (**m**,**)*) – User-specified 3D radii \(R\), positive, in internal units.

Returns:

**rho** – Density of each component in internal units. For a profile with
an edge node, the density is zero at and beyond
\(R_{\rm t}\).

Return type:

ndarray, shape (n_components, m)

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a density or, for a profile with an edge node, the potential
    at an interior node is not positive.

See also

[`mass`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.mass)
The enclosed mass.

[`potential`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential)
The relative potential.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> rho = profile.density([0.5, 1.0])
>>> rho.shape, round(float(rho[0, 1]), 6)
((1, 2), 0.25)
```

mass(*r*)

Return the total enclosed mass \(\mathcal{M}(R)\).

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), positive, in internal units.

Returns:

**mass** – Enclosed mass \(\mathcal{M}(R)\) in internal units (the unit
of mass_trunc), with the shape of r. For a spatially finite
profile, the mass is mass_t at and beyond r_t.

Return type:

ndarray

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the enclosed mass at a node is not positive.

See also

[`density`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.density)
The density of each component.

[`potential`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential)
The relative potential.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.mass(1.0)), 6)
0.25
```

potential_log_slope(*r*)

Return the logarithmic slope \(\mathrm{d}\ln P/\mathrm{d}\ln R\).

The Poisson equation \(\mathrm{d}P/\mathrm{d}R =
-\mathcal{M}/R^2\) gives \(\mathrm{d}\ln P/\mathrm{d}\ln R =
-\mathcal{M}/(R P)\). The slope is evaluated from mass and
potential with no further approximation. The slope is continuous
in \(R\) between the nodes.

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), positive, in internal units.

Returns:

**slope** – \(\mathrm{d}\ln P/\mathrm{d}\ln R\), dimensionless, with the
shape of r.

Return type:

ndarray

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the potential at an interior node or the enclosed mass at a
    node is not positive.

See also

[`potential_log_slope_derivative`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential_log_slope_derivative)
The derivative of the slope in \(R\).

[`potential`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential)
The relative potential.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.potential_log_slope(1.0)), 6)
-0.5
```

potential_log_slope_derivative(*r*)

Return the derivative in \(R\) of the potential’s logarithmic slope.

With \(s = -\mathcal{M}/(R P)\) the logarithmic slope of the
potential, \(\mathrm{d}\mathcal{M}/\mathrm{d}R = R^2 \varrho\)
(\(\varrho\) the total density of the components) and
\(\mathrm{d}P/\mathrm{d}R = -\mathcal{M}/R^2\), the derivative is

\[\frac{\mathrm{d}s}{\mathrm{d}R} = -\frac{R^3 \varrho P
- \mathcal{M} P + \mathcal{M}^2/R}{R^2 P^2},\]

evaluated from mass, potential and density with no further
approximation.

Parameters:

**r** (*array-like*) – User-specified 3D radii \(R\), positive, in internal units.

Returns:

**derivative** – \(\mathrm{d}s/\mathrm{d}R\) in internal units (inverse units
of \(r_0\)), with the shape of r.

Return type:

ndarray

Raises:

[**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the potential or a density at an interior node, or the
    enclosed mass at a node, is not positive.

See also

[`potential_log_slope`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential_log_slope)
The logarithmic slope of the potential.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.potential_log_slope_derivative(1.0)), 6)
-0.25
```

radius(*p*)

Return the radius \(R\) at which the potential takes the value p.

Parameters:

**p** (*array-like*) – User-specified relative potentials \(P\), in internal units,
at most p_trunc at the first node. For a profile with an edge
node, a value of zero or below gives r_t. For a profile
without an edge node, a value of zero or below gives the radius
of the last node.

Returns:

**r** – Radius \(R\) in internal units, with the shape of p.

Return type:

ndarray

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a value of p is above p_trunc at the first node.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the potential at an interior node is not positive.

See also

[`potential`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile.potential)
The potential at a given radius.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> round(float(profile.radius(0.5)), 6)
1.0
```

to_dict()

Return the record that identifies the profile, as a plain dictionary.

The arrays of the profile are not in the record. The identity of
the model determines the arrays, and a file stores the arrays
beside the record.

Returns:

**record** – The key `"profile"` with the value `"truncated"`, the key
`"model"` with the identity string of the model, and the keys
`"source"` and `"source_anisotropy"` with the dictionaries of
source and source_anisotropy.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`DoublePowerLaw.to_dict`](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw.to_dict)
The dictionary of the source profile.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.profiles import TruncatedProfile
>>> hernquist = pp.DoublePowerLaw.hernquist()
>>> r = np.geomspace(1e-3, 1e3, 601)
>>> profile = TruncatedProfile(
...     r=r, p_trunc=hernquist.p(r), rho_trunc=hernquist.rho(r)[np.newaxis, :],
...     mass_trunc=hernquist.mass(r), k_t=0.0, r_t=np.inf, mass_t=1.0,
...     spatially_finite=False, source=hernquist,
...     source_anisotropy=pp.Anisotropy(), model="hernquist", edge_exponent=(),
... )
>>> sorted(profile.to_dict())
['model', 'profile', 'source', 'source_anisotropy']
```
