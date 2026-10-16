# Truncation in phase space

Source: https://pianistpy.com/docs/guide/truncation.html

# Truncation in phase space

pianistpy.Truncation, pianistpy.truncate(), pianistpy.solve(), pianistpy.Model

## Method I

truncate(df, truncation) returns a Model: the distribution function truncated at an energy,
with the self-consistent potential-density pair that follows from it. A Model is by construction a physical
and stable equilibrium; a request for which none exists is refused with UnphysicalModelError,
which carries the distribution function for inspection.

One limit is stated rather than solved: for a steep cusp at beta_c = -1/2 the self-consistent center of the
truncated model is not reached (measured on (alpha, beta) = (2, 4) at K_t = 0.3, from gamma 1.735 on), and the call
refuses it with an AccuracyError that names the iteration’s response.

Truncation takes exactly one of three targets:

```
pp.Truncation(k_t=0.2)                          # the truncation energy, in internal units
pp.Truncation(mass_t=1.0, concentration=10.0)   # a target mass, in units of M_0 (model units)
pp.Truncation(r_t=5.0)                          # a target radius, in units of the scale radius

```

A target mass is given in units of M_0, so it needs the concentration that defines M_0; a target radius and a
truncation energy do not, and a concentration given with them is refused.

solve(component, truncation) runs compute_df and truncate at once.

What a model holds: the truncated distribution function, the truncated density, potential and enclosed mass on
one radial grid (model.profile.r, .rho_trunc, .p_trunc, .mass_trunc), the truncation radius and
the total mass, the validation report and the identity. A model that is not spatially finite is still a model:
it says so (spatially_finite is false), states its radius and mass as not finite, and what needs finiteness
asks for it by name (Flags and refusals).

A Type II model (Osipkov-Merritt or Cuddeford) must end inside its anisotropy radius R_a, where its density
rho_aux / (1 - r^2/R_a^2) has a pole. The lower bound K_t > P(R_a) keeps the untruncated radius of the truncation
energy inside R_a, but the truncated model’s own edge lies farther out; where that edge would reach R_a the request
is refused by name (TYPE II EDGE, with the potential at R_a stated), and a larger k_t brings the edge
inside.

## Method II

Method II gives a new anisotropy to a density that is already truncated: a component whose profile is the
truncated profile of another model, through the same functions.

```
parent = pp.solve(pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy()), pp.Truncation(k_t=0.2))
child = pp.solve(pp.Component(parent.profile, pp.Anisotropy(beta_c=-0.25)))

```

A Method II model exists only when its anisotropy is compatible with the edge of the parent’s density; the
condition is checked at the call and a violation is refused by name, with the admissible range stated. The
technical notes derive it (Technical notes).

     On this page
