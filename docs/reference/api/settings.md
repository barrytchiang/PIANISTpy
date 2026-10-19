# Settings (pianistpy.settings)

Source: https://pianistpy.com/docs/reference/api/settings

# Settings (pianistpy.settings)

Numerical settings: the tested tiers and expert overrides.

class pianistpy.settings.Field(default, doc, quick=None)

One numerical setting: its value in each tier and what it controls.

Parameters:

default (bool, int, float or str) – Value in the default tier. Its type is the type of the field.

doc (str) – What the field controls.

quick (bool, int, float or str, optional) – Value in the quick tier. If omitted, the quick tier keeps the default
value: a cheaper value enters only once it has passed the tests.

value(tier)

Return the value of the field in a tier.

Parameters:

tier (str)

Return type:

Any

class pianistpy.settings.Settings(quick=False, write_anyway=False, **overrides)

Numerical settings of a computation.

A tier is a complete set of numerical settings that has been run through
the test battery and for which the package states the accuracy it
delivers. An expert may override single fields; the result then records
that it was not made with a tested tier and names the fields that differ.

Parameters:

quick (bool, optional) – Use the quick tier instead of the default tier.

write_anyway (bool, optional) – Return a result whose own check lies beyond the hard limit instead of
refusing it. The result carries the check’s verdict in its validation
report (tier "refused") and an AccuracyWarning is issued in
place of the refusal. For debugging a case the package cannot serve
at its stated accuracy; off by default. A physically invalid model
is refused regardless.

**overrides (mapping) – One mapping per group (inversion, truncation, sampling,
observables) of the fields to change within that group.

Variables:

tier (str) – Name of the tier the settings start from.

write_anyway (bool) – Whether a result beyond the hard limit is returned instead of refused.

overrides (mapping) – The fields changed from the tier, by group. Read only.

Examples

```
>>> Settings().tier
'default'
>>> Settings(quick=True).is_tested_tier
True
>>> Settings(inversion={"n_nodes": 256}).is_tested_tier
False
>>> Settings(inversion={"n_nodes": 256}).inversion["n_nodes"]
256
>>> Settings().truncation["ode_rtol"]
1e-11

```

property tier: str

Name of the tier the settings start from.

property write_anyway: bool

Whether a result beyond the hard limit is returned instead of refused.

property overrides: Mapping[str, Mapping[str, Any]]

The fields changed from the tier, by group.

property is_tested_tier: bool

Whether the settings are a tested tier with no field changed.

group(name)

Return the value of every field of one group.

Parameters:

name (str) – One of the groups inversion, truncation, sampling and
observables.

Returns:

values – The fields of the group with the values of the tier, each
replaced by its override where one was given. Read only.

Return type:

mapping

property inversion: Mapping[str, Any]

Settings of the inversion.

property truncation: Mapping[str, Any]

Settings of the truncation.

property sampling: Mapping[str, Any]

Settings of the particle sampling.

property observables: Mapping[str, Any]

Settings of the observables.

property losvd: Mapping[str, Any]

The fields of the LOSVD table and the aperture LOSVD.

resolved()

Return every field of every group with the value in force.

Return type:

dict[str, dict[str, Any]]

to_dict()

Return the settings as a plain dictionary.

Returns:

record – The tier and the overrides, in the form written to files, with
"write_anyway": True where that was asked. The numerical
settings alone determine a result, so numerical is what the
identity hashes.

Return type:

dict

numerical()

Return the tier and the overrides, the settings that determine a result.

Return type:

dict[str, Any]

classmethod from_dict(record)

Build settings from their dictionary form.

Parameters:

record (mapping) – A dictionary as returned by to_dict.

Returns:

settings

Return type:

Settings

     On this page
