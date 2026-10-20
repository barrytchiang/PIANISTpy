# Units

Source: https://pianistpy.com/docs/reference/api/units

# Units

Module pianistpy.units

Conversion from internal units to model and physical units.

class pianistpy.units.UnitFactors(rho_norm, mass_q, model_v, model_mass, model_rho, model_p, physical_r_pc=None, physical_v_kms=None, physical_mass_msun=None, physical_rho_msun_pc3=None, physical_p_kms2=None, physical_rho_0_msun_pc3=None)

Factors that convert internal units to model and to physical units.

A quantity in internal units is multiplied by the factor to obtain it in
the target units. The model units have \(G = M_0 = r_0 = 1\), with
\(M_0\) the mass inside the radius concentration times
\(r_0\).

Parameters:

rho_norm (float) – Normalization \(C\) of the density profile.

mass_q (float) – \(q = \mathcal{M}(c)/C\), the mass inside the concentration
radius in units of \(4\pi\rho_0 r_0^3\).

model_v (float) – Factors to the model units, for a speed, a mass, a density and a
potential.

model_mass (float) – Factors to the model units, for a speed, a mass, a density and a
potential.

model_rho (float) – Factors to the model units, for a speed, a mass, a density and a
potential.

model_p (float) – Factors to the model units, for a speed, a mass, a density and a
potential.

physical_r_pc (float or None) – Factors to parsecs, kilometers per second and solar masses.

physical_v_kms (float or None) – Factors to parsecs, kilometers per second and solar masses.

physical_mass_msun (float or None) – Factors to parsecs, kilometers per second and solar masses.

physical_rho_msun_pc3 (float or None) – Factors to solar masses per cubic parsec, and to the square of
kilometers per second.

physical_p_kms2 (float or None) – Factors to solar masses per cubic parsec, and to the square of
kilometers per second.

physical_rho_0_msun_pc3 (float or None) – The scale density \(\rho_0\) in solar masses per cubic parsec.

Notes

The physical factors are None unless both physical scales were given to
unit_factors.

class pianistpy.units.Units(system='internal', concentration=None, mass_0_msun=None, r_0_pc=None)

The unit system of a result and the scales that define it.

Parameters:

system ({"internal", "model", "physical"}) – "internal": \(G = \rho_0/C = r_0 = 1\). "model":
\(G = M_0 = r_0 = 1\). "physical": Astropy quantities.

concentration (float, optional) – Concentration \(c\) that defines \(M_0\) as the mass inside
\(c\,r_0\). Needed by the model and the physical systems.

mass_0_msun (float, optional) – \(M_0\) in solar masses. Needed by the physical system.

r_0_pc (float, optional) – Scale radius \(r_0\) in parsecs. Needed by the physical system.

Examples

```
>>> Units().system
'internal'
>>> Units.build("model", concentration=10.0).concentration
10.0

```

classmethod build(system, concentration=None, mass_0=None, r_0=None, r_vir=None)

Return the units of a system from the scales a user gives.

Parameters:

system ({"internal", "model", "physical"}) – The target system.

concentration (float, optional) – Concentration that defines \(M_0\).

mass_0 (float or astropy Quantity, optional) – \(M_0\); a float is in solar masses.

r_0 (float or astropy Quantity, optional) – The scale radius, or the radius \(c\,r_0\) in its place; a
float is in parsecs. One of the two for the physical system.

r_vir (float or astropy Quantity, optional) – The scale radius, or the radius \(c\,r_0\) in its place; a
float is in parsecs. One of the two for the physical system.

Returns:

units

Return type:

Units

factors(profile)

Return the conversion factors of these units for a density profile.

Parameters:

profile (DoublePowerLaw)

Return type:

UnitFactors

convert(kind, value, profile)

Return a quantity in internal units in the units of this object.

Parameters:

kind ({"length", "velocity", "mass", "density", "potential"}) – What the quantity is.

value (array-like) – The quantity, in internal units.

profile (DoublePowerLaw) – The density profile that fixes the normalization.

Returns:

converted – A plain array for the internal and the model systems, an Astropy
quantity for the physical system.

Return type:

array or astropy Quantity

to_dict()

Return the units as a plain dictionary.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build the units from their dictionary form; internal if absent.

Parameters:

record (Mapping[str, Any] | None)

Return type:

Units

pianistpy.units.unit_factors(profile, concentration, mass_0_msun=None, r_0_pc=None)

Return the conversion factors of a profile.

Parameters:

profile (DoublePowerLaw) – The density profile, which fixes the normalization \(C\).

concentration (float) – Radius, in units of the scale radius \(r_0\), inside which the
mass \(M_0\) is defined. Positive.

mass_0_msun (float, optional) – \(M_0\) in solar masses.

r_0_pc (float, optional) – Scale radius \(r_0\) in parsecs.

Returns:

factors – The model-unit factors, and the physical factors if both mass_0_msun
and r_0_pc are given.

Return type:

UnitFactors

Notes

With \(\rho_{\rm w} = \rho_0/C\) the internal units are
\(M_{\rm W} = 4\pi\rho_{\rm w} r_0^3\) for mass and
\(v_{\rm W} = \sqrt{4\pi G\rho_{\rm w} r_0^2}\) for speed, and
\(M_0 = 4\pi\rho_0 r_0^3\,q\). Hence a speed is multiplied by
\((C q)^{-1/2}\), a mass and a potential by \((C q)^{-1}\), and a
density by \((4\pi C q)^{-1}\) to reach the model units.

References

Chiang+(2026d), Section on units (labels ssec:Units, eqn:M_nor_1).

Examples

```
>>> factors = unit_factors(DoublePowerLaw.nfw(), concentration=10.0)
>>> factors.rho_norm
1.0
>>> round(factors.mass_q, 12) == round(math.log(11.0) - 10.0 / 11.0, 12)
True

```

     On this page
