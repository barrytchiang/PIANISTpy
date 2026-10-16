# Density profiles

Source: https://pianistpy.com/docs/guide/profiles.html

# Density profiles

pianistpy.DoublePowerLaw

Any double power law in its three slopes, DoublePowerLaw(rho_alpha, rho_beta, rho_gamma): the inner slope,
the outer slope and the sharpness of the transition between them, with the scale radius and the central
potential of the untruncated profile set to one (the internal units, Units). The named profiles are
presets of the same class, not classes of their own:

```
pp.DoublePowerLaw.nfw()        # Navarro-Frenk-White
pp.DoublePowerLaw.hernquist()
pp.DoublePowerLaw.plummer()
pp.DoublePowerLaw(1.0, 3.0, 1.0)   # the same NFW profile by its slopes

```

A profile holds its density, enclosed mass and potential as closed forms; every quantity of the package is
computed from them by one path, and the named profiles’ textbook forms are test references only.

A profile is one half of a pianistpy.Component; the other half is its anisotropy.
