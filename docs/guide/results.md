# Flags and refusals

Source: https://pianistpy.com/docs/guide/results

# Flags and refusals

What the package returns, what it flags and what it refuses, in one place.

## The distribution function is always returned

compute_df returns the distribution function of any component the inversion defines, whatever its sign. Where
it is negative the result says so, with the zero crossing, the minimum and the smallest truncation value above
which the truncated function is non-negative.

## A model is always a valid equilibrium

truncate and solve return a Model only where one exists. A request for which there is
no physical, stable equilibrium raises UnphysicalModelError; the exception carries the
distribution function, so nothing is lost and nothing has to be recomputed. A Method II request whose anisotropy
is incompatible with the parent’s edge is refused at the call by the existence condition, with the admissible
range stated.

## A model that is not spatially finite is a model

A truncated model whose density does not reach zero at a finite radius, or whose mass does not converge, is
returned with spatially_finite false and its truncation radius and total mass stated as not finite. This is a
property, not an error, and comes with no accuracy warning. What needs finiteness asks for it by name: a
truncation by target mass or target radius has no solution there and is refused with the reachable range;
sampling requires an explicit sampling radius and records the mass inside it; observables state the radius to
which they hold.

## A distribution function that diverges at the truncation energy

A Method II child whose anisotropy is more than one unit below its parent’s has a distribution function that
diverges, integrably, at the truncation energy. It is returned, with a warning that names the exponent and says
the model is a valid equilibrium of finite density whose stability is not established.

## The accuracy tiers

Every result carries a validation report: a short and fixed summary of the checks it passed, one record per
check, with its tier. A check within the accuracy goal (a relative error of \(10^{-6}\)) is silent. Above the goal,
within the ceiling (\(10^{-4}\)) or within the hard limit (\(10^{-2}\)), the result is returned with a warning in the package’s
log and as an AccuracyWarning (Python’s standard filter silences or escalates it), and the
flag stays with the result and with everything computed from it. Beyond the hard limit the result is refused.

## One component

The first release covers one component per model. compute_df and solve refuse more than one; several
components in one potential, and tracers in a fixed potential, are the planned next extension
(Upcoming features).

## The keys to read first in a result

validation (the report and its tier), spatially_finite, r_t and mass_t (the truncation radius
and the total mass), identity (what the result was computed from), units (the system it is presented in),
timing (the wall time of each stage) and, on a file, the backend that made it.

     On this page
