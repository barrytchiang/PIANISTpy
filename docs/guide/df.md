# Distribution functions

Source: https://pianistpy.com/docs/guide/df.html

# Distribution functions

pianistpy.compute_df(), pianistpy.DistributionFunction

compute_df(component) returns the untruncated distribution function of a component, from the analytic
inversion of its density, as a DistributionFunction. This is a mathematical object: it is
returned whether or not it is everywhere non-negative and monotonic, and it records where it is not, together
with the smallest truncation value above which it is. It is also the base of Method I truncation, so a scan over
truncations of one model reuses one inversion.

```
halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))
df = pp.compute_df(halo)

```

What the result holds: the tabulated distribution function on its energy grid with its logarithmic slope, the
asymptotic form at the center and at the outer end (how the function continues beyond the table), the sign
report, the inputs and the settings it was computed from, its validation report and its identity. A result can
be written to a file and read back equal (Input and output).

A distribution function is not yet a model: only a truncated model can be sampled or
projected.
