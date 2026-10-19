# Truncation in phase space

Source: https://pianistpy.com/docs/guide/truncation

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
is refused by name (TYPE II EDGE, with the potential at R_a stated), and a larger K_t brings the edge
inside. The bound is strict, so where it sets a Type II DF’s k_t_min that is the next float above P(R_a), the
smallest K_t the gates admit: necessary, not sufficient, since only the truncated solve places the edge.

An Osipkov-Merritt Type I model with a small R_a and a low K_t can leave no edge within the solve’s radius;
the request is refused by name (NO EDGE). On the Hernquist density at the quick settings, R_a = 1 is refused at
every K_t from 0.005 to 0.3, R_a = 2 below K_t = 0.2, R_a = 5 below 0.03 and R_a = 10 at 0.005; the parents that
build are at the ceiling tier, set by their distribution function’s high_k_slope_mismatch (3.4e-6, at K = 1).

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
