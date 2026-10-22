# Units

Source: https://pianistpy.com/docs/reference/api/units

Module `pianistpy.units`

Unit systems of results and conversion factors out of internal units.

A result presents the result’s quantities in another unit system through the result’s
`to` method, which builds a Units object. unit_factors returns the
conversion factors of a density profile as a UnitFactors object.

class pianistpy.units.UnitFactors(*rho_norm*, *mass_q*, *model_v*, *model_mass*, *model_rho*, *model_p*, *physical_r_pc=None*, *physical_v_kms=None*, *physical_mass_msun=None*, *physical_rho_msun_pc3=None*, *physical_p_kms2=None*, *physical_rho_0_msun_pc3=None*)

Factors that convert quantities from internal units to model and physical units.

A quantity in internal units is multiplied by the factor of the
quantity’s dimension to obtain the quantity in the target units. The
model units have \(G = M_0 = r_0 = 1\), with \(M_0\) the mass
inside the radius \(c\,r_0\) and \(c\) the concentration given to
unit_factors. A length has no model factor, because internal units and
model units both measure a length in units of \(r_0\).

Parameters:

- **rho_norm** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Normalization \(C\) of the density profile
(DoublePowerLaw.rho_norm).

- **mass_q** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The number \(q = \mathcal{M}(c)/C\): the mass inside the radius
\(c\,r_0\) in units of \(4\pi\rho_0 r_0^3\).

- **model_v** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Factor that converts a velocity to model units, \((C q)^{-1/2}\).

- **model_mass** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Factor that converts a mass to model units, \((C q)^{-1}\).

- **model_rho** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Factor that converts a density to model units,
\((4\pi C q)^{-1}\).

- **model_p** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Factor that converts a potential to model units, \((C q)^{-1}\).

- **physical_r_pc** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – Factor that converts a length to parsecs, equal to \(r_0\) in
parsecs. Default: `None`, which means unit_factors received no
physical scales.

- **physical_v_kms** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – Factor that converts a velocity to kilometers per second. Default:
`None`, which means unit_factors received no physical scales.

- **physical_mass_msun** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – Factor that converts a mass to solar masses. Default: `None`, which
means unit_factors received no physical scales.

- **physical_rho_msun_pc3** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – Factor that converts a density to solar masses per cubic parsec.
Default: `None`, which means unit_factors received no physical
scales.

- **physical_p_kms2** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – Factor that converts a potential to the square of kilometers per
second. Default: `None`, which means unit_factors received no
physical scales.

- **physical_rho_0_msun_pc3** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **None**, **optional*) – The scale density \(\rho_0\) in solar masses per cubic parsec.
Default: `None`, which means unit_factors received no physical
scales.

See also

[`unit_factors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.unit_factors)
The function that builds the factors of a density profile.

[`Units`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units)
The unit system a result carries.

Notes

Every physical factor is None unless unit_factors received both
physical scales, \(M_0\) in solar masses and \(r_0\) in parsecs.

Examples

```python
>>> from pianistpy import DoublePowerLaw
>>> from pianistpy.units import unit_factors
>>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
>>> round(factors.model_mass, 6)
0.67168
>>> factors.physical_r_pc is None
True
```

class pianistpy.units.Units(*system='internal'*, *concentration=None*, *mass_0_msun=None*, *r_0_pc=None*)

The unit system of a result and the scales that define the unit system.

A result holds the result’s numbers in internal units and hands the
numbers over in the unit system that the result’s Units object sets.
Units.build builds a Units object from scales in any unit; the
constructor takes the scales in solar masses and parsecs.

Parameters:

- **system** (*{"internal"**, **"model"**, **"physical"}**, **optional*) – The unit system. `"internal"`: \(G = \rho_0/C = r_0 = 1\), with
\(\rho_0\) the scale density and \(C\) the normalization of
the density profile. `"model"`: \(G = M_0 = r_0 = 1\), with
\(M_0\) the mass inside the radius \(c\,r_0\).
`"physical"`: Astropy quantities in parsecs, kilometers per second,
solar masses, solar masses per cubic parsec and
\({\rm km^2\,s^{-2}}\). Default: `"internal"`, which means the
result hands the result’s quantities over in internal units.

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The concentration \(c\), which defines \(M_0\) as the mass
inside the radius \(c\,r_0\). Required by the model units and the
physical units. Default: `None`, which means no concentration, as
the internal units require.

- **mass_0_msun** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The mass \(M_0\) in solar masses. Required by the physical units.
Default: `None`, which means no physical mass scale.

- **r_0_pc** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The scale radius \(r_0\) in parsecs. Required by the physical
units. Default: `None`, which means no physical length scale.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is not `"internal"`, `"model"` or `"physical"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is `"model"` or `"physical"` and concentration is
    None.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is `"physical"` and mass_0_msun or r_0_pc is None.

See also

[`Units.build`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units.build)
The units of a unit system from scales in any unit.

[`unit_factors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.unit_factors)
The conversion factors out of internal units.

[`UnitFactors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.UnitFactors)
The conversion factors and their definitions.

Examples

```python
>>> from pianistpy.units import Units
>>> Units().system
'internal'
>>> Units.build("model", concentration=10.0).concentration
10.0
>>> units = Units.build("physical", concentration=10.0, mass_0=1e12, r_vir=2e5)
>>> units.r_0_pc
20000.0
```

classmethod build(*system*, *concentration=None*, *mass_0=None*, *r_0=None*, *r_vir=None*)

Return the units of a unit system from user-specified scales in any unit.

The internal units ignore every scale. The model units keep the
concentration and ignore the physical scales. The physical units keep
the concentration, \(M_0\) in solar masses and \(r_0\) in
parsecs.

Parameters:

- **system** (*{"internal"**, **"model"**, **"physical"}*) – The unit system of the returned units.

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The concentration \(c\), which defines \(M_0\) as the mass
inside the radius \(c\,r_0\). Positive. Required by the model
units and the physical units. Default: `None`, which means no
concentration.

- **mass_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **astropy Quantity**, **optional*) – The mass \(M_0\). A float is in solar masses; a quantity of any
mass unit is converted to solar masses. Positive. Required by the
physical units. Default: `None`, which means no mass scale.

- **r_0** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **astropy Quantity**, **optional*) – The scale radius \(r_0\). A float is in parsecs; a quantity of
any length unit is converted to parsecs. Positive. Default:
`None`, which means the scale radius is taken from r_vir.

- **r_vir** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **astropy Quantity**, **optional*) – The radius \(c\,r_0\), given in place of r_0. A float is in
parsecs; a quantity of any length unit is converted to parsecs.
Positive. Default: `None`, which means the scale radius is
taken from r_0.

Returns:

**units** – The units of system, with \(M_0\) in solar masses and
\(r_0\) in parsecs.

Return type:

[Units](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If both r_0 and r_vir are given.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If concentration, r_0, r_vir or, for the physical units,
    mass_0 is given and is not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If r_vir is given and concentration is not.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is not `"internal"`, `"model"` or `"physical"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is `"model"` or `"physical"` and concentration
    is None.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If system is `"physical"` and mass_0 is None, or neither
    r_0 nor r_vir is given.

See also

`Model.to`
The model presented in the unit system that a Units object sets.

Examples

```python
>>> from pianistpy.units import Units
>>> Units.build("internal", concentration=10.0)
Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None)
>>> Units.build("physical", concentration=10.0, mass_0=1e12, r_0=2e4).r_0_pc
20000.0
```

factors(*profile*)

Return the conversion factors of the units’ scales for a density profile.

Parameters:

**profile** ([*DoublePowerLaw*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)) – The density profile, which fixes the normalization \(C\) of
the density.

Returns:

**factors** – The factors unit_factors returns for the profile, the units’
concentration, and the units’ mass_0_msun and r_0_pc.

Return type:

[UnitFactors](https://pianistpy.com/docs/reference/api/units#pianistpy.units.UnitFactors)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the units’ concentration is None, as in the internal units.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the units’ concentration is not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If one of mass_0_msun and r_0_pc is set and the other is None.

See also

[`unit_factors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.unit_factors)
The function that builds the factors.

Examples

```python
>>> from pianistpy import DoublePowerLaw
>>> from pianistpy.units import Units
>>> units = Units.build("model", concentration=10.0)
>>> round(units.factors(DoublePowerLaw.nfw()).model_mass, 6)
0.67168
```

convert(*kind*, *value*, *profile*)

Return a quantity given in internal units in the unit system of the units.

Parameters:

- **kind** (*{"length"**, **"velocity"**, **"mass"**, **"density"**, **"potential"}*) – The physical dimension of the quantity.

- **value** (*array-like*) – The quantity, in internal units.

- **profile** ([*DoublePowerLaw*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)) – The density profile, which fixes the normalization \(C\) of
the density.

Returns:

**converted** – The quantity, with the shape of value. A float64 array in the
internal units and in the model units; an Astropy quantity in the
physical units. A length has the same value in internal units and
in model units, because both unit systems measure a length in
units of \(r_0\).

Return type:

array or astropy Quantity

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If kind is not `"length"`, `"velocity"`, `"mass"`,
    `"density"` or `"potential"`.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the unit system is not the internal one and factors raises
    InputError for the units’ scales.

See also

[`Units.factors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units.factors)
The conversion factors the conversion multiplies by.

Examples

```python
>>> from pianistpy import DoublePowerLaw
>>> from pianistpy.units import Units
>>> units = Units.build("model", concentration=10.0)
>>> units.convert("length", [1.0, 2.0], DoublePowerLaw.nfw())
array([1., 2.])
>>> round(float(units.convert("mass", 1.0, DoublePowerLaw.nfw())), 6)
0.67168
```

to_dict()

Return the units as a dictionary of plain values.

Returns:

**record** – The keys `system`, `concentration`, `mass_0_msun` and
`r_0_pc`, with the values of the units’ fields, in the form
from_dict reads.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`Units.from_dict`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units.from_dict)
The units rebuilt from the dictionary.

Examples

```python
>>> from pianistpy.units import Units
>>> record = Units.build("model", concentration=10.0).to_dict()
>>> record["system"], record["concentration"], record["r_0_pc"]
('model', 10.0, None)
```

classmethod from_dict(*record*)

Return the units that a dictionary written by to_dict records.

Parameters:

**record** (*mapping** or **None*) – The dictionary to_dict writes. None or an empty mapping stands
for the internal units.

Returns:

**units** – The units the dictionary records.

Return type:

[Units](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units)

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the recorded unit system and scales raise InputError in the
    Units constructor.

See also

[`Units.to_dict`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units.to_dict)
The dictionary of the units.

Examples

```python
>>> from pianistpy.units import Units
>>> units = Units.build("model", concentration=10.0)
>>> Units.from_dict(units.to_dict()) == units
True
>>> Units.from_dict(None).system
'internal'
```

pianistpy.units.unit_factors(*profile*, *concentration*, *mass_0_msun=None*, *r_0_pc=None*)

Return the factors that convert a profile’s quantities out of internal units.

The model-unit factors depend on the density profile and the
concentration \(c\) alone. The physical factors also need the two
physical scales, \(M_0\) in solar masses and \(r_0\) in parsecs.

Parameters:

- **profile** ([*DoublePowerLaw*](https://pianistpy.com/docs/reference/api/profiles#pianistpy.profiles.DoublePowerLaw)) – The density profile. The profile fixes the normalization \(C\)
and the mass \(\mathcal{M}(c)\) inside the radius \(c\,r_0\).

- **concentration** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The concentration \(c\): the radius, in units of the scale radius
\(r_0\), inside which the mass \(M_0\) is defined. Positive.

- **mass_0_msun** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The mass \(M_0\) in solar masses. Positive. Given together with
r_0_pc or not at all. Default: `None`, which means only the
model-unit factors are formed.

- **r_0_pc** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The scale radius \(r_0\) in parsecs. Positive. Given together with
mass_0_msun or not at all. Default: `None`, which means only the
model-unit factors are formed.

Returns:

**factors** – The model-unit factors, and the physical factors when both
mass_0_msun and r_0_pc are given. Every physical factor is None
when neither mass_0_msun nor r_0_pc is given.

Return type:

[UnitFactors](https://pianistpy.com/docs/reference/api/units#pianistpy.units.UnitFactors)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If concentration is not positive.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If one of mass_0_msun and r_0_pc is given and the other is not.

See also

[`UnitFactors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.UnitFactors)
The factors and their definitions.

[`Units.factors`](https://pianistpy.com/docs/reference/api/units#pianistpy.units.Units.factors)
The factors for the scales a result carries.

Notes

With \(\rho_{\rm w} = \rho_0/C\), the internal unit of mass is
\(M_{\rm W} = 4\pi\rho_{\rm w} r_0^3\), the internal unit of velocity is
\(v_{\rm W} = (4\pi G\rho_{\rm w} r_0^2)^{1/2}\), and the mass inside
the radius \(c\,r_0\) is \(M_0 = 4\pi\rho_0 r_0^3\,q\). A velocity in
internal units is therefore multiplied by \((C q)^{-1/2}\) to reach
model units, a mass and a potential by \((C q)^{-1}\), and a density
by \((4\pi C q)^{-1}\).

References

[Chiang+(2026d)](https://arxiv.org/abs/2610.28758), Section 2.1 and Eq. (6).

Examples

```python
>>> import math
>>> from pianistpy import DoublePowerLaw
>>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
>>> factors.rho_norm
1.0
>>> round(factors.mass_q, 12) == round(math.log(11.0) - 10.0 / 11.0, 12)
True
>>> physical = unit_factors(
...     DoublePowerLaw.nfw(), 10.0, mass_0_msun=1e12, r_0_pc=2e4
... )
>>> physical.physical_r_pc
20000.0
```
