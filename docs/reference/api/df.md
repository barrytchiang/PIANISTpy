# Distribution functions and models

Source: https://pianistpy.com/docs/reference/api/df

Module `pianistpy.df`

Distribution functions and equilibrium models of spherical systems.

The pianistpy.df module computes the distribution function (DF) of a
spherical density profile with a given velocity anisotropy, and the
equilibrium model that the DF supports. A Component holds one density
profile with its anisotropy. solve returns the model of a component.
compute_df returns the untruncated DF of a component, and truncate
builds the model from that DF, with or without a Truncation.

The DF of every anisotropy model of the package has the separable form

\[F(\mathcal{K}, \mathcal{L}) = F_{\mathcal{K}}(\mathcal{K})\,
\mathcal{L}^{-2\beta_{\rm c}}\]

([Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (8)). \(\mathcal{K}\) is the energy-like
variable in internal units, with \(0 \le \mathcal{K} \le 1\).
\(\mathcal{K}\) is the relative energy \(Z\) for a constant
anisotropy, and \(\mathcal{K}\) is \(\mathcal{Q}_\pm\) for an
Osipkov-Merritt (OM) or Cuddeford-Osipkov-Merritt (COM) model.
\(\mathcal{L}\) is the angular momentum in internal units, and
\(\beta_{\rm c}\) is the central anisotropy. A truncation at the
energy \(\mathcal{K}_{\rm t}\) (Method I of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing)) gives the truncated DF

\[\hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
- F(\mathcal{K}_{\rm t}, \mathcal{L})\]

for \(\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0\),
and \(\hat{F} = 0\) for \(\hat{\mathcal{K}} < 0\).

Each DF and each model carries a validation report. truncate and solve
raise AccuracyError if a check of the model’s validation report lies
above \(10^{-2}\) (the band “refused”), the round trip of the model’s
DF against the model’s density included, unless the settings set
`write_anyway=True`. compute_df, truncate and solve issue an
AccuracyWarning if a check of the returned result’s validation report
lies above \(10^{-4}\).

The derivation is in [Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Section 3.1, and the numerical method
in Notes Section T:2.

class pianistpy.df.Component(*profile*, *anisotropy=<factory>*, *mass=1.0*, *name='component'*)

One component of a model: a density profile with its velocity anisotropy.

A model is a sequence of components that share one total potential. A
model of one component is a sequence of length one. compute_df and
solve accept a model of one component.

A component whose profile is the TruncatedProfile of another model
(`parent.profile`) is Method II of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing). The Method II
component has the density of the parent model in the parent model’s
potential, with the component’s own anisotropy.

Parameters:

- **profile** ([*DoublePowerLaw*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)* or *[*TruncatedProfile*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile)) – Density profile of the component. A DoublePowerLaw is an analytic
profile and needs an outer slope \(\beta \ge 3\). A
TruncatedProfile must be the profile of a model of one component.

- **anisotropy** ([*Anisotropy*](https://pianistpy.com/docs/reference/api/anisotropy#pianistpy.anisotropy.Anisotropy)*, **optional*) – Velocity anisotropy of the component. Default: `Anisotropy()`,
which means an isotropic velocity distribution.

- **mass** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Mass of the component in units of the mass of the first component.
Must be positive. Default: `1.0`, which means the mass of the
first component.

- **name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – Name of the component, carried by the component’s results and
particles. Default: `"component"`, which means the results and
particles carry the name `"component"`.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If mass is not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the outer slope \(\beta\) of the analytic profile lies above
    2 and below 3. The inversion assumes a relative potential
    \(P \propto 1/R\) at large radius \(R\), which holds for
    \(\beta \ge 3\) only.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If profile is the TruncatedProfile of a model of more than one
    component.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If profile is a TruncatedProfile with an edge and the central
    anisotropy \(\beta_{\rm c}\) is not above \(1/2 - q\), with
    \(q\) the exponent of the density at the edge,
    \(\varrho \propto \hat{P}^{\,q}\). No distribution function of
    the given anisotropy exists at the edge of such a profile.

See also

[`Truncation`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)
The truncation of a model’s distribution function (Method I).

[`solve`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.solve)
The equilibrium model of a component.

`Anisotropy`
The velocity anisotropy of a component.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Section 3.1 and Appendix B2.2.

Examples

```python
>>> import pianistpy as pp
>>> halo = pp.Component(
...     pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25), name="halo"
... )
>>> halo.anisotropy.model, halo.mass, halo.name
('constant', 1.0, 'halo')
```

property unit_profile: [DoublePowerLaw](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)

Analytic profile that defines the internal units of the component.

to_dict()

Return the component as a plain dictionary.

Returns:

**record** – Dictionary with the keys `"name"`, `"mass"`, `"profile"`
(the dictionary form of the profile) and `"anisotropy"` (the
dictionary form of the anisotropy). Component.from_dict
builds the component back from the dictionary.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Component.from_dict`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component.from_dict)
The component built from its dictionary.

Examples

```python
>>> import pianistpy as pp
>>> sorted(pp.Component(pp.DoublePowerLaw.nfw()).to_dict())
['anisotropy', 'mass', 'name', 'profile']
```

classmethod from_dict(*record*, *profile=None*)

Return the component that a dictionary describes.

Parameters:

- **record** (*mapping*) – Dictionary form of the component, as Component.to_dict
returns it.

- **profile** ([*TruncatedProfile*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile)*, **optional*) – The TruncatedProfile of the component, where record states a
TruncatedProfile. The dictionary of a TruncatedProfile identifies
the profile’s model and does not hold the profile’s arrays.
Default: `None`, which means that record states an analytic
profile.

Returns:

**component** – The component that record describes.

Return type:

[Component](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If record states a TruncatedProfile and profile is not
    given or belongs to a model other than the one record identifies.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component that record describes is not admissible. The
    Raises section of Component lists the conditions.

See also

[`Component.to_dict`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component.to_dict)
The dictionary of a component.

Examples

```python
>>> import pianistpy as pp
>>> halo = pp.Component(pp.DoublePowerLaw.nfw(), name="halo")
>>> pp.Component.from_dict(halo.to_dict()) == halo
True
```

class pianistpy.df.DistributionFunction(*components*, *settings*, *identity*, *grid*, *table*, *form*, *k_t_min*, *validation*, *timing*, *inversion*, *units=Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None)*, *rows=None*, *source='inversion'*, *function_name=None*)

Untruncated distribution functions of the components of a model.

compute_df and DistributionFunction.from_function return a
DistributionFunction. An untruncated distribution function (DF) may
be negative or non-monotonic, and the DistributionFunction states
where. A DistributionFunction is not sampled or projected: truncate
builds the Model that is.

Parameters:

- **components** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Components to which the DFs belong.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings with which the DFs were computed.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the result.

- **grid** (*EnergyGrid*) – Energies of the DF table and the radii at which the untruncated
relative potential equals each energy.

- **table** (*DfTable*) – The DF of each component at the energies of grid.

- **form** (*AsymptoticForm*) – Analytic form of each DF below and above the energies of the DF
table.

- **k_t_min** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Lowest admissible truncation energy \(\mathcal{K}_{\rm t}\), in
internal units. `k_t_min` is zero where every untruncated DF is a
valid equilibrium as it stands (is_physical). Otherwise
`k_t_min` is the largest of the highest energy at which a DF is
negative, the energy of a DF’s interior minimum and, for a Type II
model, the relative potential \(P(R_{\rm a})\) at the anisotropy
radius \(R_{\rm a}\). Where \(P(R_{\rm a})\) is the largest
of these energies, `k_t_min` is the next double above
\(P(R_{\rm a})\). truncate also checks
that the edge of a truncated Type II model lies inside
\(R_{\rm a}\).

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – Validation report of the DFs.

- **timing** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to float*) – Elapsed time of each step of the computation, in seconds.

- **inversion** (*InversionRecord** or **None*) – Record of how the inversion made the DF table. None where the DF
table has a closed form. Every DistributionFunction from
compute_df holds an InversionRecord.

- **units** ([*Units*](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units)*, **optional*) – Unit system in which the result is presented (to). Default:
`Units()`, which means internal units.

- **rows** (*InversionRows** or **None**, **optional*) – Values of the inversion at every energy of the inversion’s grid,
which truncate reads. Default: `None`, which means that the DF
table was not made by an inversion.

- **source** (*{"inversion"**, **"function"}**, **optional*) – Origin of the DF table’s values: `"inversion"` (compute_df) or
`"function"` (DistributionFunction.from_function). Default:
`"inversion"`.

- **function_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **None**, **optional*) – The `__name__` of the function from which the DF table was
tabulated. Default: `None`, which means that the DF table comes
from an inversion.

See also

[`compute_df`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.compute_df)
The untruncated DF of a component.

[`truncate`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.truncate)
The equilibrium model built from a DistributionFunction.

[`Model`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)
An equilibrium model.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (8). Notes Section T:2.

Examples

```python
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> df.source, df.function_name, len(df.components)
('inversion', None, 1)
```

classmethod from_function(*component*, *f*, *settings=None*)

Return the distribution function that a user-specified function gives.

The user-specified function is a distribution function (DF) of the
energy, such as a fitting function. The package tabulates the
function on the energy grid that compute_df uses for the given
component and settings, and validates the tabulated DF against the
component’s density. Like a DistributionFunction from
compute_df, the result holds analytic forms below and above the
DF table, a validation report and an identity string, and
DistributionFunction.write writes the result to an HDF5 file. The
result stores the DF table and does not store the function.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – The component to which the DF belongs. The component’s profile
must be an analytic profile (a DoublePowerLaw).

- **f** (*callable*) – Function `f(k, depth)` of two arrays of equal shape: the
energies \(\mathcal{K}\) of the DF table in internal units,
and the depth \(P_0 - \mathcal{K}\) at each energy, with
\(P_0 = 1\) the central relative potential. `f` returns
the DF \(F\) (not its logarithm) at each energy, as an array
of the shape of `k`. The package passes `depth` as
`1 - k` in double precision, which is exact for every energy
\(\mathcal{K} \ge 1/2\). A function of \(\mathcal{K}\)
alone may ignore `depth`.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – Numerical settings. Default: `None`, which means
`Settings()`, the default settings.

Returns:

**df** – The tabulated DF. The attribute source is `"function"` and
the attribute function_name is the `__name__` of f. The DF
table has no column of the derivative of the DF, and the check
`slope_column` of the validation report states that the
derivative is not judged. The result holds no inversion record
and no inversion values, so truncate does not accept it.

Return type:

[DistributionFunction](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If f is not callable, returns an array whose shape differs
    from the shape of `k`, or returns a value that is not finite
    at an energy of the grid. For a value that is not finite, the
    message states the first energy with such a value.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component is built on a TruncatedProfile.

Warns:

**AccuracyWarning** – If the DF is non-negative on its table and a check of the DF’s
validation report lies above \(10^{-4}\).

See also

[`compute_df`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.compute_df)
The DF computed by inversion of the density.

[`DistributionFunction.is_physical`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction.is_physical)
Whether the DF is a valid equilibrium as it stands.

References

Notes Section T:2.7.

Examples

The isotropic Plummer DF in internal units is
\(F = A\,\mathcal{K}^{7/2}\) with \(A \approx 1.965117\):

```python
>>> import pianistpy as pp
>>> def plummer(k, depth):
...     return 1.965117148288 * k**3.5
>>> component = pp.Component(pp.DoublePowerLaw.plummer())
>>> df = pp.DistributionFunction.from_function(component, plummer)
>>> df.source, df.function_name
('function', 'plummer')
```

property is_physical: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the untruncated DFs are a valid equilibrium without a truncation.

is_physical is true if k_t_min is zero.

property diverges_at_zero: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[bool](https://docs.python.org/3/builtins/functions.html#bool), ...]

Whether the DF of each component diverges as the energy goes to zero.

A DF that goes as \(\mathcal{K}^p\) with \(-1 < p < 0\) as
the energy \(\mathcal{K}\) goes to zero is integrable and gives a
finite density, but such a DF is not monotonic.

to(*system*, *concentration=None*, *mass_0=None*, *r_0=None*, *r_vir=None*)

Return the distribution function presented in another unit system.

The values of the result are not recomputed and stay in internal
units. A quantity computed from the returned result is presented in
the returned result’s unit system.

Parameters:

- **system** (*{"internal"**, **"model"**, **"physical"}*) – Target unit system: `"internal"` (\(G = \rho_0/C = r_0 =
1\)), `"model"` (\(G = M_0 = r_0 = 1\)) or `"physical"`
(Astropy quantities).

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Concentration \(c\), positive. The mass unit \(M_0\) is
the mass of the untruncated profile inside \(c\,r_0\). The
model and the physical systems need concentration. Default:
`None`, which means that no concentration is given.

- **mass_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Mass unit \(M_0\), positive, for the physical system. A float
is in solar masses, and a Quantity of any mass unit is accepted.
Default: `None`, which means that no mass unit is given.

- **r_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Scale radius \(r_0\), positive, for the physical system. A
float is in parsecs, and a Quantity of any length unit is
accepted. Default: `None`, which means that r_vir gives the
length scale.

- **r_vir** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Concentration radius \(c\,r_0\), positive, given in place of
r_0 for the physical system. A float is in parsecs, and a
Quantity of any length unit is accepted. Default: `None`,
which means that r_0 gives the length scale.

Returns:

**df** – The DF with unchanged values, with the attribute units set to
the target unit system.

Return type:

[DistributionFunction](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is not `"internal"`, `"model"` or
    `"physical"`, if a scale that system needs is missing, if a
    scale is not positive, or if both r_0 and r_vir are given.

See also

[`Model.to`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.to)
A model presented in another unit system.

Examples

```python
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> df.to("model", concentration=10.0).units.system
'model'
```

write(*path*, *overwrite=False*)

Write the distribution function to an HDF5 file.

pianistpy.read reads the file back into a DistributionFunction
with the same identity and the same arrays.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the HDF5 file to write.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default:
`False`, which means that an existing file is kept and
InputError is raised.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the written file does not read back as written.

Return type:

None

See also

[`DistributionFunction.write_text`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction.write_text)
The DF table as a text file.

Examples

```python
>>> import os, tempfile
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "hernquist_df.h5")
...     df.write(path)
...     same = pp.read(path).identity == df.identity
>>> same
True
```

write_text(*path*, *overwrite=False*)

Write the DF table to a fixed-width text file.

The text file is an export for reading outside the package, and
pianistpy.read does not read it. The file holds one line per
energy of each component’s support, with the columns `k` (the
energy \(\mathcal{K}\) in internal units, in which the central
relative potential of the untruncated profile is 1), `log10_df`
(the base-10 logarithm of the DF) and `dlog10_df_dlnk` (the slope
\({\rm d}\log_{10} F / {\rm d}\ln \mathcal{K}\)). A model of more
than one component has a fourth column, `component`, with the
index of the component. The header states the result, the result’s
components and the units. The written file is read back and
compared with the written values, to the digits written.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the text file to write.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default:
`False`, which means that an existing file is kept and
InputError is raised.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the written file does not read back as written.

Return type:

None

See also

[`DistributionFunction.write`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction.write)
The full result as an HDF5 file.

Examples

```python
>>> import os, tempfile
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "hernquist_df.txt")
...     df.write_text(path)
...     with open(path) as file:
...         first = file.readline().strip()
>>> first
'# PIANISTpy distribution function table'
```

class pianistpy.df.Model(*components*, *truncation*, *settings*, *identity*, *parent*, *table*, *form*, *profile*, *validation*, *timing*, *r_trust*, *inversion*, *units=Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None)*)

An equilibrium model of one or more components, truncated or untruncated.

truncate and solve return a Model. The distribution function (DF)
of every component of a model is non-negative. A DF that diverges as
the energy goes to zero is not monotonic (diverges_at_zero). Whether
the model’s density reaches zero at a finite radius is the property
spatially_finite.

Parameters:

- **components** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Components of the model.

- **truncation** ([*Truncation*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)* or **None*) – Truncation of the model. None for an untruncated model and for a
model whose component is built on a TruncatedProfile.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings with which the model was computed.

- **identity** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the result.

- **parent** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identity string of the DistributionFunction from which the model
was built.

- **table** (*DfTable*) – The DF of each component of the model.

- **form** (*AsymptoticForm*) – Analytic form of each DF below and above the energies of the DF
table.

- **profile** ([*TruncatedProfile*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.TruncatedProfile)) – Self-consistent density, mass and relative potential of the model.

- **validation** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – Validation report of the model.

- **timing** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to float*) – Elapsed time of each step of the computation, in seconds.

- **r_trust** (*mapping** of **str to float*) – Radius, in internal units, inside which the model’s values stay
within the bound of a tier of the validation report, keyed by the
tier’s name. The key `"silent"` holds the radius for a relative
error of at most \(10^{-6}\), and the key `"recorded"` the
radius for a relative error of at most \(10^{-4}\)
([Accuracy tables](https://pianistpy.com/docs/reference/accuracy)). Each radius states how far out the
model meets the tier’s bound, and is not itself an error of the
model. On an untruncated model of an
analytic profile, both radii are infinite. On every other model
that truncate and solve return, the mapping is empty. A model
read from a file holds the radii that the file records.

- **inversion** (*InversionRecord** or **None*) – Record of how the inversion made the model’s DF table. None where
the DF table has a closed form. Every Model from solve holds an
InversionRecord.

- **units** ([*Units*](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units)*, **optional*) – Unit system in which the model is presented (to). Default:
`Units()`, which means internal units.

See also

[`solve`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.solve)
The equilibrium model of a component.

[`truncate`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.truncate)
The equilibrium model built from a DistributionFunction.

[`DistributionFunction`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction)
The untruncated DF of a component.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Section 3.1. Notes Section T:3.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(
...     pp.Component(pp.DoublePowerLaw.hernquist()), pp.Truncation(k_t=0.05)
... )
>>> model.spatially_finite, model.truncation.k_t
(True, 0.05)
```

property spatially_finite: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the model’s density reaches zero at a finite radius.

property diverges_at_zero: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[bool](https://docs.python.org/3/builtins/functions.html#bool), ...]

Whether the DF of each component diverges as the energy goes to zero.

A DF that goes as \(\mathcal{K}^p\) with \(-1 < p < 0\) as
the energy \(\mathcal{K}\) goes to zero is integrable. A model
with such a DF is a valid equilibrium with a finite density, but the
model’s DF is not monotonic.

to(*system*, *concentration=None*, *mass_0=None*, *r_0=None*, *r_vir=None*)

Return the model presented in another unit system.

The values of the model are not recomputed and stay in internal
units. A quantity computed from the returned model is presented in
the returned model’s unit system.

Parameters:

- **system** (*{"internal"**, **"model"**, **"physical"}*) – Target unit system: `"internal"` (\(G = \rho_0/C = r_0 =
1\)), `"model"` (\(G = M_0 = r_0 = 1\)) or `"physical"`
(Astropy quantities).

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Concentration \(c\), positive. The mass unit \(M_0\) is
the mass of the untruncated profile inside \(c\,r_0\). The
model and the physical systems need a concentration. Default:
`None`, which means the concentration of the model’s
Truncation where the Truncation gives one, and no
concentration otherwise.

- **mass_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Mass unit \(M_0\), positive, for the physical system. A float
is in solar masses, and a Quantity of any mass unit is accepted.
Default: `None`, which means that no mass unit is given.

- **r_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Scale radius \(r_0\), positive, for the physical system. A
float is in parsecs, and a Quantity of any length unit is
accepted. Default: `None`, which means that r_vir gives the
length scale.

- **r_vir** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **Quantity**, **optional*) – Concentration radius \(c\,r_0\), positive, given in place of
r_0 for the physical system. A float is in parsecs, and a
Quantity of any length unit is accepted. Default: `None`,
which means that r_0 gives the length scale.

Returns:

**model** – The model with unchanged values, with the attribute units set
to the target unit system.

Return type:

[Model](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If concentration differs from the concentration of the
    model’s Truncation.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is not `"internal"`, `"model"` or
    `"physical"`, if a scale that system needs is missing, if a
    scale is not positive, or if both r_0 and r_vir are given.

See also

[`Model.radial_profiles`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.radial_profiles)
The radial profiles in the model’s unit system.

[`DistributionFunction.to`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction.to)
A DF presented in another unit system.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> model.to("model", concentration=10.0).units.system
'model'
```

radial_profiles()

Return the radial profiles of the model in the model’s unit system.

Returns:

**profiles** – Dictionary of four arrays tabulated at the radii of `"r"`:
`"r"` (the radii, shape `(n,)`), `"potential"` (the
relative potential, shape `(n,)`), `"density"` (the density
of each component, shape `(n_components, n)`) and `"mass"`
(the total enclosed mass, shape `(n,)`). The arrays are plain
arrays in the internal and the model systems, and Astropy
quantities in the physical system (Model.to).

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Model.to`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.to)
The model presented in another unit system.

[`Model.write_text`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.write_text)
The radial profiles as a text file.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> profiles = model.radial_profiles()
>>> sorted(profiles), profiles["density"].shape[0]
(['density', 'mass', 'potential', 'r'], 1)
```

property mass_t: [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Total mass of the model, in the model’s unit system.

property r_t: [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Outer radius of the model, in the model’s unit system.

Infinite for a model whose density does not reach zero at a finite
radius.

write(*path*, *overwrite=False*)

Write the model to an HDF5 file.

pianistpy.read reads the file back into a Model with the same
identity and the same arrays.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the HDF5 file to write.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default:
`False`, which means that an existing file is kept and
InputError is raised.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the written file does not read back as written.

Return type:

None

See also

[`Model.write_text`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.write_text)
A table of the model as a text file.

Examples

```python
>>> import os, tempfile
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "hernquist.h5")
...     model.write(path)
...     same = pp.read(path).identity == model.identity
>>> same
True
```

write_text(*path*, *table='profile'*, *overwrite=False*)

Write a table of the model to a fixed-width text file.

The text file is an export for reading outside the package, and
pianistpy.read does not read it. The written file is read back and
compared with the written values, to the digits written.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – Path of the text file to write.

- **table** (*{"profile"**, **"df"}**, **optional*) – Table to write. `"profile"` writes the radial profiles in the
model’s unit system (Model.radial_profiles), with the columns
`r`, `potential`, `density` and `mass` (the total
enclosed mass). A model of more than one component has one
density column per component, called `density_<name>` after
the component’s name. The header states the unit of each
column. `"df"` writes the DF table in internal units, with the
columns that DistributionFunction.write_text writes. Default:
`"profile"`, which means the radial profiles.

- **overwrite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether to replace an existing file at path. Default:
`False`, which means that an existing file is kept and
InputError is raised.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If table is neither `"profile"` nor `"df"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a file exists at path and overwrite is false.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the written file does not read back as written.

Return type:

None

See also

[`Model.write`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.write)
The full model as an HDF5 file.

[`Model.radial_profiles`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model.radial_profiles)
The radial profiles as arrays.

Examples

```python
>>> import os, tempfile
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "hernquist_profile.txt")
...     model.write_text(path)
...     with open(path) as file:
...         first = file.readline().strip()
>>> first
'# PIANISTpy radial profile'
```

class pianistpy.df.Truncation(*k_t=None*, *mass_t=None*, *r_t=None*, *concentration=None*)

Truncation of a model’s distribution function (Method I).

A truncation shifts the distribution function (DF) in energy by the
truncation energy \(\mathcal{K}_{\rm t}\) and lowers the DF by the
DF’s value at \(\mathcal{K}_{\rm t}\), which gives a model of finite
mass and finite radius (Method I of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing)). The truncation is given
by exactly one of three inputs: the truncation energy k_t, a target
mass mass_t or a target radius r_t. A target mass is given together
with the concentration that defines the unit of the target mass.
Method II needs no Truncation: a Method II model is a Component
whose profile is the TruncatedProfile of another model.

Parameters:

- **k_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Truncation energy \(\mathcal{K}_{\rm t}\), positive, in internal
units, in which the central relative potential of the untruncated
profile is 1. Default: `None`, which means that mass_t or r_t
is given.

- **mass_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Target total mass of the truncated model, positive, in units of
\(M_0\), the mass of the untruncated profile inside the radius
\(c\,r_0\) (model units). A value of 1 is the virial mass.
mass_t is the one input of Truncation that is not in internal
units. The truncation energy is solved for. mass_t requires
concentration. Default: `None`, which means that k_t or r_t
is given.

- **r_t** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Target truncation radius \(R_{\rm t}\), positive, in units of
the scale radius \(r_0\), the unit of length of the internal
and the model systems. The truncation energy is solved for.
Default: `None`, which means that k_t or mass_t is given.

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – Concentration \(c\) of the first component, positive. The mass
unit \(M_0\) of mass_t is the mass of the untruncated profile
inside \(c\,r_0\). concentration is given with mass_t only,
since k_t and r_t do not depend on the concentration. Default:
`None`, which means that no target mass is given.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If not exactly one of k_t, mass_t and r_t is given, or if the
    one given is not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If mass_t is given without concentration, if concentration is
    given without mass_t, or if concentration is not positive.

See also

[`truncate`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.truncate)
The model that a truncation gives.

[`solve`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.solve)
The model of a component, with or without a truncation.

[`Component`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)
A component on the TruncatedProfile of another model (Method II).

Notes

The truncated DF is
\(\hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
- F(\mathcal{K}_{\rm t}, \mathcal{L})\) for
\(\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0\), and
\(\hat{F} = 0\) for \(\hat{\mathcal{K}} < 0\), with \(F\)
the untruncated DF, \(\mathcal{K}\) the energy-like variable and
\(\mathcal{L}\) the angular momentum.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (9). Notes Section T:3.6.

Examples

```python
>>> import pianistpy as pp
>>> pp.Truncation(mass_t=1.0, concentration=10.0).to_dict()
{'k_t': None, 'mass_t': 1.0, 'r_t': None, 'concentration': 10.0}
```

to_dict()

Return the truncation as a plain dictionary.

Returns:

**record** – Dictionary with the keys `"k_t"`, `"mass_t"`, `"r_t"` and
`"concentration"`, each None where the field is not given.
Truncation.from_dict builds the truncation back from the
dictionary.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Truncation.from_dict`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation.from_dict)
The truncation built from its dictionary.

Examples

```python
>>> import pianistpy as pp
>>> pp.Truncation(k_t=0.05).to_dict()
{'k_t': 0.05, 'mass_t': None, 'r_t': None, 'concentration': None}
```

classmethod from_dict(*record*)

Return the truncation that a dictionary describes.

Parameters:

**record** (*mapping*) – Dictionary form of the truncation, as Truncation.to_dict
returns it. A key missing from record means None for that
field.

Returns:

**truncation** – The truncation that record describes.

Return type:

[Truncation](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the fields of record do not make an admissible truncation.
    The Raises section of Truncation lists the conditions.

See also

[`Truncation.to_dict`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation.to_dict)
The dictionary of a truncation.

Examples

```python
>>> import pianistpy as pp
>>> pp.Truncation.from_dict({"r_t": 20.0})
Truncation(k_t=None, mass_t=None, r_t=20.0, concentration=None)
```

pianistpy.df.compute_df(*components*, *settings=None*)

Return the untruncated distribution function of a component.

The distribution function (DF) is returned whatever its sign. The
result states where the DF is negative, and the lowest truncation
energy \(\mathcal{K}_{\rm t}\) that gives a valid equilibrium. Call
solve for the model itself. Call compute_df to inspect the DF before
choosing a truncation, or to truncate one DF in several ways.

Parameters:

- **components** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)* or **sequence** of *[*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – The component of the model, or a sequence that holds exactly one
Component.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – Numerical settings. Default: `None`, which means `Settings()`,
the default settings.

Returns:

**df** – The DF of the component, tabulated on an energy grid, with its
analytic forms below and above the energies of the table, the
lowest admissible truncation energy `k_t_min` in internal units,
and the validation report of the DF.

Return type:

[DistributionFunction](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If components is an empty sequence, a sequence with an item that
    is not a Component, or a sequence of more than one component.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component is built on a TruncatedProfile whose model has an
    anisotropy other than isotropic or Osipkov-Merritt Type I, or whose
    model carries no DF.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component is built on a TruncatedProfile, has a Type II
    anisotropy, and the edge of the TruncatedProfile does not lie
    inside the component’s anisotropy radius \(R_{\rm a}\).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the component is built on a TruncatedProfile and the
    component’s central anisotropy \(\beta_{\rm c}\) is below
    \(-1\) and is not \(-3/2\).

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the component has a Cuddeford-Osipkov-Merritt (COM) Type II
    anisotropy and the integrand of the DF’s inversion is not
    integrable at the anisotropy radius \(R_{\rm a}\).

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the component is built on a TruncatedProfile and the central
    density slope \(\gamma\) is below \(2\beta_{\rm c}\).

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the component is built on the TruncatedProfile of a model that
    was truncated at \(\mathcal{K}_{\rm t} > 0\) but has no edge.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the inversion cannot resolve the DF at the resolution of
    settings. `Settings(write_anyway=True)` does not turn this error
    into a warning.

Warns:

**AccuracyWarning** – If the DF is non-negative on its table and a check of the DF’s
validation report lies above \(10^{-4}\).

See also

[`solve`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.solve)
The equilibrium model of a component in one call.

[`truncate`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.truncate)
The equilibrium model built from a distribution function.

[`DistributionFunction.from_function`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction.from_function)
A DF given as a function of the energy.

Notes

The DF has the separable form
\(F(\mathcal{K}, \mathcal{L}) = F_{\mathcal{K}}(\mathcal{K})\,
\mathcal{L}^{-2\beta_{\rm c}}\), with \(\mathcal{K}\) the
energy-like variable and \(\mathcal{L}\) the angular momentum, both
in internal units. The result tabulates
\(F_{\mathcal{K}}(\mathcal{K})\).

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eq. (8). Notes Section T:2.1.

Examples

```python
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> df.is_physical, df.k_t_min
(True, 0.0)
```

pianistpy.df.solve(*components*, *truncation=None*, *settings=None*)

Return the equilibrium model of a component.

solve calls compute_df and then truncate. Call compute_df and
truncate separately to inspect the distribution function (DF) before
the truncation, or to truncate one DF in several ways.

Parameters:

- **components** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)* or **sequence** of *[*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – The component of the model, or a sequence that holds exactly one
Component.

- **truncation** ([*Truncation*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)*, **optional*) – The truncation. Default: `None`, which means the untruncated
model, as in truncate.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)*, **optional*) – Numerical settings. Default: `None`, which means `Settings()`,
the default settings.

Returns:

**model** – The equilibrium model, as truncate returns it.

Return type:

[Model](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If compute_df or truncate raises InputError for the inputs.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If no valid equilibrium exists for the inputs. The Raises sections
    of compute_df and truncate list the conditions.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the DF or the model misses the accuracy that compute_df and
    truncate require.

Warns:

**AccuracyWarning** – If a check of the DF’s or of the model’s validation report lies
above \(10^{-4}\).

See also

[`compute_df`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.compute_df)
The untruncated distribution function.

[`truncate`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.truncate)
The equilibrium model built from a distribution function.

[`Component`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)
A density profile with its velocity anisotropy.

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Section 3.1.

Examples

```python
>>> import pianistpy as pp
>>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> model.truncation is None, model.spatially_finite
(True, False)
```

pianistpy.df.truncate(*df*, *truncation=None*)

Return the equilibrium model built from a distribution function.

With a Truncation, the model’s DF is the untruncated DF shifted in
energy by the truncation energy \(\mathcal{K}_{\rm t}\) and lowered
by the untruncated DF’s value at \(\mathcal{K}_{\rm t}\), which
gives a model of finite mass and finite radius (Method I of
[Chiang+(2026d)](https://pianistpy.com/docs/project/citing)). Without a truncation, the untruncated DF is the
model’s DF. An untruncated DF is a valid equilibrium only if
DistributionFunction.is_physical is true.

Parameters:

- **df** ([*DistributionFunction*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.DistributionFunction)) – The untruncated DF, as compute_df returns it.

- **truncation** ([*Truncation*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)*, **optional*) – The truncation. Default: `None`, which means the untruncated
model. For a component built on a TruncatedProfile, `None`
means the model that ends at the edge of that TruncatedProfile.

Returns:

**model** – The equilibrium model: its non-negative DF, its self-consistent
density and potential (a TruncatedProfile), and its validation
report.

Return type:

[Model](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Model)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If df was built by DistributionFunction.from_function. Such a DF
    holds the function’s values only, and a truncation needs the values
    of the inversion.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the truncation energy \(\mathcal{K}_{\rm t}\) is below
    `df.k_t_min`, or no admissible truncation energy reaches the
    target mass or the target radius of truncation.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the central density slope \(\gamma\) is below
    \(2\beta_{\rm c}(0)\), with \(\beta_{\rm c}(0)\) the
    central anisotropy (0 for the isotropic and Osipkov-Merritt
    models), or if the DF is negative or tends to minus infinity at the
    center.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If \(\gamma = 2\beta_{\rm c}(0)\) and the DF has no closed form
    at the center to join the table of the DF.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the truncated model has no edge, or the edge of a Type II model
    lies at or beyond the anisotropy radius \(R_{\rm a}\).

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If truncation is `None` and the untruncated DF is not a valid
    equilibrium (`df.is_physical` is false), or the untruncated mass
    diverges.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If the component is built on a TruncatedProfile and the DF is
    negative at an energy inside the DF’s support.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If a check of the model’s validation report lies above
    \(10^{-2}\) (the band “refused”), the round trip of the model’s
    DF against the model’s density included. With
    `Settings(write_anyway=True)` in the settings of df, the model is
    returned with an AccuracyWarning instead.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the truncated DF, or the potential solved from the truncated DF,
    is not finite at the resolution of the settings.
    `write_anyway=True` does not turn this error into a warning.

Warns:

**AccuracyWarning** – If a check of the model’s validation report lies above
\(10^{-4}\).

See also

[`compute_df`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.compute_df)
The untruncated distribution function.

[`solve`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.solve)
compute_df and truncate in one call.

[`Truncation`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Truncation)
The truncation energy, target mass or target radius.

Notes

An UnphysicalModelError raised by truncate holds a result in its
`result` attribute. The `result` attribute holds the model without
an edge where the truncation gives no edge, and df otherwise.

The truncated DF is

\[\hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
- F(\mathcal{K}_{\rm t}, \mathcal{L})\]

for \(\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0\),
and \(\hat{F} = 0\) for \(\hat{\mathcal{K}} < 0\). Here
\(F\) is the untruncated DF, \(\mathcal{K}\) the energy-like
variable and \(\mathcal{L}\) the angular momentum, all in internal
units. The truncated relative potential \(\hat{P}(R)\) solves
Poisson’s equation with the density of \(\hat{F}\), from
\(\hat{P}(0) = 1 - \mathcal{K}_{\rm t}\) to the truncation radius
\(R_{\rm t}\), where \(\hat{P}(R_{\rm t}) = 0\).

References

[Chiang+(2026d)](https://pianistpy.com/docs/project/citing), Eqs. (9), (10) and (11). Notes Section T:3.

Examples

```python
>>> import pianistpy as pp
>>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
>>> model = pp.truncate(df, pp.Truncation(k_t=0.05))
>>> model.spatially_finite
True
```

## Gradients

Module `pianistpy.df.gradient`

JAX functions that return the tabulated distribution function in terms of the anisotropy parameters, and the
Jacobians of those functions ([End-to-end differentiability](https://pianistpy.com/docs/guide/differentiability)).

Derivatives of a component’s distribution function in its anisotropy.

This module returns the distribution function (DF) \(F(Q)\) that
compute_df makes for a component, at fixed relative energies \(Q\),
together with the derivatives of \(F\) in the anisotropy parameters
\(\theta = (\beta_{\rm c}, R_{\rm a})\). inversion_nodes returns the
energies at which compute_df evaluates \(F\). rows_jacobian returns
\(F\) and its derivatives at the energies of inversion_nodes as
arrays, and table_jacobian returns every array of the DF’s table and its
derivatives. make_inversion_rows returns \(F\), and
make_inversion_table returns the arrays of the DF’s table, as JAX
functions of \(\theta\), which `jax.grad` and `jax.vjp`
differentiate.

The DF is \(F = T_b[g]\). Here \(b = \beta_{\rm c}\) is the central
anisotropy. The source \(g\) is the set of derivatives
\({\rm d}^N \varrho_{\rm aux} / {\rm d}P^N\) of the auxiliary density
\(\varrho_{\rm aux}\) in the relative potential \(P\). The operator
\(T_b\) is the inversion of the auxiliary density at
\(\beta_{\rm c} = b\), and \(T_b\) is linear in \(g\). The
derivatives of \(F\) are

\[\frac{\partial F}{\partial R_{\rm a}} = T_b\left[\partial_{R_{\rm a}} g\right],
\qquad
\frac{\partial F}{\partial \beta_{\rm c}} = T_b\left[\partial_b g\right]
+ E_b[g].\]

The derivative \(\partial_\theta g\) of the source in \(\theta\) at
a fixed \(P\) is in closed form. The term \(E_b[g]\) is the
derivative of the operator \(T_b\) in \(b\) at a fixed source
\(g\). \(E_b[g]\) has no closed form and is taken by finite
differences of \(T_b[g]\) in \(b\). The record returned with each
derivative in \(\beta_{\rm c}\) states the residual of the finite
differences in \(b\), for comparison with the accuracy bounds
\(10^{-6}\), \(10^{-4}\) and \(10^{-2}\) of the package’s
validation reports.

The derivative of the DF’s table follows by the chain rule through each
array of the table. The energies and every discrete choice of the table are
held at the parameters the call is given. For a component on a
TruncatedProfile (Method II), the profile is held, and the derivative of
the source is the derivative of the inverse anisotropy factor in closed
form.

The derivatives are available for constant anisotropy
(\(\beta_{\rm c}\)), Cuddeford-Osipkov-Merritt (COM) Type I
(\(\beta_{\rm c}\) and \(R_{\rm a}\)), and Osipkov-Merritt Type I
and Type II (\(R_{\rm a}\)). The derivatives are not available for COM
Type II, whose interior point at the potential \(P(R_{\rm a})\) moves
with both parameters. The derivatives are not available for a DF whose
center lies on the
critical branch, where the inner slope \(\gamma\) of the density equals
\(2\beta_{\rm c}\) and the form of the center changes with
\(\beta_{\rm c}\). The derivation is Notes Eq. (T:7.1).

pianistpy.df.gradient.inversion_nodes(*component*, *settings*)

Return the energies at which compute_df inverts a component.

The energies are the relative energies \(Q\) at which compute_df
evaluates the DF \(F(Q)\) for the given component and settings. The
other public functions of pianistpy.df.gradient hold the energies of
inversion_nodes fixed when the anisotropy parameters change.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Component on an analytic profile (not a TruncatedProfile).

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings, the same as those given to compute_df.

Returns:

**q** – Relative energies \(Q\), strictly increasing, in internal
units.

Return type:

ndarray, shape (n,)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the energies that settings places include two energies that
    float64 cannot separate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If the component is Osipkov-Merritt Type II or COM, whose energies
    are placed around a feature of a first coarse inversion, and the
    coarse inversion finds a derivative of the auxiliary density that
    is not finite.

See also

[`rows_jacobian`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.rows_jacobian)
The DF and its derivatives at the energies of inversion_nodes.

[`make_inversion_rows`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_rows)
The DF at the energies of inversion_nodes as a JAX function.

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.df.gradient import inversion_nodes
>>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
>>> q = inversion_nodes(halo, pp.Settings())
>>> bool(np.all(np.diff(q) > 0.0))
True
```

pianistpy.df.gradient.make_inversion_rows(*component*, *settings*, *q=None*)

Return the DF at fixed energies as a JAX function of the anisotropy.

The returned function maps the anisotropy parameters \(\theta\) to
the DF \(F(Q)\) that compute_df makes at the relative energies
q. The reverse-mode derivative of the returned function at each
\(\theta\) is the derivative that rows_jacobian returns at that
\(\theta\). For the derivatives as arrays, call rows_jacobian
instead.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Component on an analytic profile (not a TruncatedProfile), with
constant, isotropic, Osipkov-Merritt Type I or Type II, or COM Type
I anisotropy. The anisotropy parameters of component are the
point at which the steps of the finite differences in
\(\beta_{\rm c}\) are measured.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings, the same as those given to compute_df.

- **q** (*array-like**, **shape** (**n**,**)**, **optional*) – Relative energies \(Q\), increasing, in internal units.
Default: `None`, which means the energies that inversion_nodes
returns for component and settings.

Returns:

- **rows** (*callable*) – Function `rows(theta)`. The argument `theta` is an array of the
parameter values in the order of params. The value is a JAX array
of shape (n,) that holds \(F(Q)\) at q, in internal units.
`jax.grad` and `jax.vjp` differentiate `rows`.

- **params** (*tuple of str*) – Names of the anisotropy parameters in `theta`: `("beta_c",)`,
`("r_a",)` or `("beta_c", "r_a")`.

- **q** (*ndarray, shape (n,)*) – Relative energies \(Q\), in internal units: q as given, or the
energies of inversion_nodes.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the anisotropy is COM Type II.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If q is `None` and the energies that settings places include
    two energies that float64 cannot separate.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If q is `None`, the component is Osipkov-Merritt Type II or
    COM, and the coarse inversion that places the energies finds a
    derivative of the auxiliary density that is not finite.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), …], [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)]

See also

[`rows_jacobian`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.rows_jacobian)
The DF at fixed energies and its derivatives as arrays.

[`make_inversion_table`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_table)
Every array of the DF’s table as a JAX function.

[`inversion_nodes`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.inversion_nodes)
The energies at which compute_df evaluates the DF.

Notes

Each call of `rows` inverts the component again at the new
parameters, at the energies q. Each derivative of `rows` calls
rows_jacobian, which raises InputError for a component on a
TruncatedProfile.

The steps of the finite differences in \(\beta_{\rm c}\) are
measured once, at the parameters of component, and are used at every
`theta`. The derivative at one `theta` therefore does not depend on
the values of `theta` in earlier calls.

References

Notes Eq. (T:7.1).

Examples

```python
>>> import jax
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.df.gradient import make_inversion_rows
>>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
>>> rows, params, q = make_inversion_rows(halo, pp.Settings())
>>> params
('r_a',)
>>> theta = np.array([10.0])
>>> rows(theta).shape == q.shape
True
>>> jax.grad(lambda t: rows(t)[-1])(theta).shape
(1,)
```

pianistpy.df.gradient.make_inversion_table(*component*, *settings*, *q=None*)

Return the arrays of the DF’s table as a JAX function of the anisotropy.

The returned function maps the anisotropy parameters \(\theta\) to
the arrays of the table that compute_df makes. The energies of the
table and the set of its arrays are held at the component’s own
parameters. The other discrete choices of the table, the anchor of the
form at the center and the nodes kept, are made again at each
\(\theta\), except that a component on a TruncatedProfile also
holds the nodes kept at the component’s own parameters. The
reverse-mode derivative of the returned function at
each \(\theta\) is the derivative that table_jacobian returns at
that \(\theta\). For the derivatives as arrays, call
table_jacobian instead.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Component with constant, isotropic, Osipkov-Merritt Type I or Type
II, or COM Type I anisotropy, on an analytic profile or on a
TruncatedProfile (Method II).

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings, the same as those given to compute_df.

- **q** (*array-like**, **shape** (**n**,**)**, **optional*) – Relative energies \(Q\), increasing, in internal units, for a
component on an analytic profile. Default: `None`, which means
the energies that inversion_nodes returns for component and
settings. A component on a TruncatedProfile takes the energies
of its own solve, and q is not used.

Returns:

- **table** (*callable*) – Function `table(theta)`. The argument `theta` is an array of the
parameter values in the order of params. The value is a dict of
JAX arrays with every key of table_jacobian’s `values` except
the energies `k` and `k_below`. `jax.grad` and `jax.vjp`
differentiate `table`.

- **params** (*tuple of str*) – Names of the anisotropy parameters in `theta`: `("beta_c",)`,
`("r_a",)` or `("beta_c", "r_a")`.

- **values** (*dict of ndarray*) – Arrays of the table at the component’s own parameters, energies
included, as table_jacobian returns them.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the anisotropy is COM Type II, or the DF’s center lies on a
    branch that table_jacobian rejects.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component is on a TruncatedProfile that Method II does not
    invert, as in table_jacobian.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If a value of the DF inside the table is not finite, is 0 or
    exceeds \(10^{300}\), or the tail’s logarithmic exponent below
    the floor is not finite.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If component is on a TruncatedProfile whose model is truncated
    but not spatially finite.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a derivative of the auxiliary density is not finite at the
    energies, or if the DF of a component on a TruncatedProfile has a
    defective band at the top of its energies.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), …], [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)]]

See also

[`table_jacobian`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.table_jacobian)
The arrays of the DF’s table and their derivatives as arrays.

[`make_inversion_rows`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_rows)
The DF at the energies of the inversion as a JAX function.

Notes

The call computes the table and its derivatives once, at the
component’s own parameters. Each call of `table` inverts the
component again at the new parameters, with the held energies. Each
derivative of `table` calls table_jacobian. The steps of the finite
differences in \(\beta_{\rm c}\) are measured once, at the
component’s own parameters, and are used at every `theta`.

References

Notes Eq. (T:7.1).

Examples

```python
>>> import numpy as np
>>> import pianistpy as pp
>>> from pianistpy.df.gradient import make_inversion_table
>>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
>>> table, params, values = make_inversion_table(halo, pp.Settings())
>>> params
('r_a',)
>>> "k" in values, "k" in table(np.array([10.0]))
(True, False)
```

pianistpy.df.gradient.rows_jacobian(*component*, *settings*, *q=None*, *steps=None*)

Return the DF at fixed energies and its derivatives in the anisotropy.

The values are the DF \(F(Q)\) that compute_df makes, at the
relative energies q. The derivatives are those of \(F\) in the
component’s anisotropy parameters at fixed q. For the derivatives of
every array of the DF’s table (the nodes below the table, the form at
the center and the tail), call table_jacobian instead.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Component on an analytic profile (not a TruncatedProfile), with
constant, isotropic, Osipkov-Merritt Type I or Type II, or COM Type
I anisotropy.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings, the same as those given to compute_df.

- **q** (*array-like**, **shape** (**n**,**)**, **optional*) – Relative energies \(Q\), increasing, in internal units.
Default: `None`, which means the energies that inversion_nodes
returns for component and settings.

- **steps** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*, **optional*) – Dictionary that keeps the steps of the finite differences in
\(\beta_{\rm c}\) between calls. A call reads the steps from
steps where steps holds steps for the call’s profile, anisotropy
family, anisotropy kind, settings and form of the inversion. A call
otherwise measures the steps and writes them into steps. Passing
one dictionary to a sequence of calls makes every later call use the
steps that the first call measured. Default: `None`, which means
each call measures its own steps and keeps them nowhere.

Returns:

- **q** (*ndarray, shape (n,)*) – Relative energies \(Q\), in internal units: q as given, or
the energies of inversion_nodes.

- **f** (*ndarray, shape (n,)*) – DF \(F(Q)\) at q, as compute_df makes it, in internal units.

- **jac** (*dict of ndarray*) – Derivatives of \(F\) at q, each with the shape of q, in
internal units. The key `"beta_c"` holds
\(\partial F/\partial\beta_{\rm c}\) and the key `"r_a"` holds
\(\partial F/\partial R_{\rm a}\), for the parameters that the
anisotropy family has.

- **record** (*dict*) – Description of the finite differences in \(\beta_{\rm c}\). The
record is an empty dict for an Osipkov-Merritt component, which has
no parameter \(\beta_{\rm c}\). The entry `"exponent"` is
`"central"` where the form of the inversion on one side of
\(\beta_{\rm c}\) equals the form on the other side, and
`"one-sided"` at a \(\beta_{\rm c}\)
where the form changes (such as \(\beta_{\rm c} = 0\) or a
half-integer \(\beta_{\rm c}\)). The entry `"steps"` holds the
steps, `"pilots"` the trial steps from which the steps were
derived, `"regions"` the widths in \(\beta_{\rm c}\) over which
the form does not change, and `"measured"` whether the call
measured the steps instead of reading them from steps. The entry
`"accuracy"` is the inversion’s own accuracy, the change of the
operator’s values when the number of quadrature nodes doubles. The
entry `"residual"` is the Richardson residual
\(\max|E(d) - E(d/2)|\) of the finite differences at the step
\(d\). A one-sided record also holds `"sides"`, the number of
sides (2), and `"spread"`, the largest difference between the
derivatives from the two sides. The entries `"accuracy"`,
`"residual"` and `"spread"` are relative to
\(\max|\partial F/\partial\beta_{\rm c}|\) over q, so the
relative error of an entry of \(\partial F/\partial\beta_{\rm c}\)
far below that maximum can exceed `"residual"`.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component is on a TruncatedProfile.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the anisotropy is COM Type II, whose interior point at the
    potential \(P(R_{\rm a})\) moves with both parameters.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a derivative of the auxiliary density is not finite at q.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)], [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

See also

[`inversion_nodes`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.inversion_nodes)
The energies at which compute_df evaluates the DF.

[`make_inversion_rows`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_rows)
The DF at fixed energies as a JAX function of the parameters.

[`table_jacobian`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.table_jacobian)
The derivatives of every array of the DF’s table.

Notes

Each derivative is \(T_b\) applied to the closed-form derivative of
the source \(g\). The derivative in \(\beta_{\rm c}\) adds the
finite differences of the operator \(T_b\) in \(b\).

A call inverts the component once for \(F\) and once per parameter.
In \(\beta_{\rm c}\), a call also inverts the component at every
point of the finite differences, at settings and at twice the
quadrature nodes of settings.

References

Notes Eq. (T:7.1).

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.df.gradient import rows_jacobian
>>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
>>> q, f, jac, record = rows_jacobian(halo, pp.Settings())
>>> sorted(jac), jac["r_a"].shape == q.shape, record
(['r_a'], True, {})
```

pianistpy.df.gradient.table_jacobian(*component*, *settings*, *q=None*, *steps=None*)

Return the arrays of the DF’s table and their derivatives in the anisotropy.

The values are the arrays of the table that compute_df makes for the
component. The derivatives are those of each array in the component’s
anisotropy parameters, with the energies and every discrete choice of
the table held at the component’s own parameters. The discrete choices
are the placement of the energies, the form of the inversion, the
anchor of the form at the center and the nodes kept. For the DF at the
energies of the inversion alone, call rows_jacobian instead.

Parameters:

- **component** ([*Component*](https://pianistpy.com/docs/reference/api/df#pianistpy.df.Component)) – Component with constant, isotropic, Osipkov-Merritt Type I or Type
II, or COM Type I anisotropy, on an analytic profile or on a
TruncatedProfile (Method II).

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – Numerical settings, the same as those given to compute_df.

- **q** (*array-like**, **shape** (**n**,**)**, **optional*) – Relative energies \(Q\), increasing, in internal units, for a
component on an analytic profile. Default: `None`, which means
the energies that inversion_nodes returns for component and
settings. A component on a TruncatedProfile takes the energies
of its own solve and accepts only `None`.

- **steps** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*, **optional*) – Dictionary that keeps the steps of the finite differences in
\(\beta_{\rm c}\) between calls, as in rows_jacobian.
Default: `None`, which means each call measures its own steps and
keeps them nowhere.

Returns:

- **values** (*dict of ndarray*) – Arrays of the table, in internal units. For a component on an
analytic profile, `k` holds the energies of the table,
`log_df` the base-10 logarithm of the DF at `k`, and
`log_df_slope` the slope \({\rm d}\log_{10} F/{\rm d}\ln Q\)
at `k`. Where the table has nodes below its first energy,
`k_below`, `log_df_below` and `log_df_slope_below` hold the
energy, the logarithm of the DF and the slope at those nodes. Where
the table has a closed form above its anchor,
\(F = a_{\rm anchor} (u/u_{\rm anchor})^\kappa + b + d\,u\) in
the depth \(u = 1 - Q\), `central` holds the array
\((\kappa, a_{\rm anchor}, b, d)\). Where the table has a tail
below its lowest node, `tail` holds the tail’s exponent,
amplitude and logarithmic exponent. For a component on a
TruncatedProfile, the keys are `k`, `log_df`, `edge_law`
(the power and the two coefficients of the edge law below the
floor, where the component has one) and `center` (the exponent
and the amplitude of the center’s closed form, and the next term’s
exponent and coefficient where the center’s form states them).

- **jac** (*dict of dict of ndarray*) – Derivatives `jac[name][key]` of `values[key]` in the parameter
`name` (`"beta_c"` or `"r_a"`), for every key of values
except the energies `k` and `k_below`, each with the shape of
`values[key]`. Where a Type II DF is outside its support, the
derivatives of `log_df` and `log_df_slope` are 0.

- **record** (*dict*) – Description of the finite differences in \(\beta_{\rm c}\),
with the entries of the record of rows_jacobian in absolute
values, not relative ones. For a component on an analytic profile,
`record["rows"]` describes the finite differences of the DF and
its slope at the energies of the inversion, and `record["below"]`,
where present, describes the finite differences at the nodes below
the table. For a component
on a TruncatedProfile, `record["rows"]` describes the finite
differences of the DF, and `record["grid"]` states the number of
energies, the number of energies kept, and the energy around which
the energies are placed.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the anisotropy is COM Type II, whose interior point at the
    potential \(P(R_{\rm a})\) moves with both parameters.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component is on an analytic profile and the DF’s center lies
    on the critical branch, where the inner slope \(\gamma\) of the
    density equals \(2\beta_{\rm c}\), with a divergent center of a
    Type I model or with a closed form above the anchor.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component is on a TruncatedProfile and the DF’s center is
    not on the regular branch \(\gamma > 2\beta_{\rm c}\), or the
    solve of component finds the center in the band of
    \(\beta_{\rm c}\) around the critical branch.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If q is given for a component on a TruncatedProfile.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If component is on a TruncatedProfile that Method II does not
    invert. Method II does not invert a profile that carries no DF of
    its own model, a profile whose model’s anisotropy is neither
    isotropic nor Osipkov-Merritt Type I, a component whose
    \(\beta_{\rm c}\) needs sources of an order above 3, or a Type
    II component whose \(R_{\rm a}\) does not lie outside the
    profile’s edge.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If a value of the DF inside the table is not finite or is 0, or
    exceeds \(10^{300}\) where no closed form covers the center.

- [**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the tail’s logarithmic exponent below the floor is not finite.

- [**UnphysicalModelError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.UnphysicalModelError) – If component is on a TruncatedProfile whose model is truncated
    but not spatially finite.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If a derivative of the auxiliary density is not finite at the
    energies, or if the DF of a component on a TruncatedProfile has a
    defective band at the top of its energies.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)], [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)]], [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

See also

[`make_inversion_table`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.make_inversion_table)
The arrays of the DF’s table as a JAX function of the parameters.

[`rows_jacobian`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.rows_jacobian)
The DF and its derivatives at the energies of the inversion.

[`inversion_nodes`](https://pianistpy.com/docs/reference/api/df#pianistpy.df.gradient.inversion_nodes)
The energies at which compute_df evaluates the DF.

Notes

The derivative of each array of the table follows by the chain rule
from the derivatives of the DF at the energies of the inversion and
from the closed-form derivatives of the center’s exponent and
amplitude.

A call costs several inversions of the component, as rows_jacobian
does, and for a component on a TruncatedProfile one solve of the
component in addition.

References

Notes Eq. (T:7.1).

Examples

```python
>>> import pianistpy as pp
>>> from pianistpy.df.gradient import table_jacobian
>>> halo = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy(r_a=10.0))
>>> values, jac, record = table_jacobian(halo, pp.Settings())
>>> sorted(jac)
['r_a']
>>> jac["r_a"]["log_df"].shape == values["log_df"].shape
True
```
