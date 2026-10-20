# Validation

Source: https://pianistpy.com/docs/reference/api/validation

# Validation

Module pianistpy.validation

Accuracy tiers and the validation report of a result.

class pianistpy.validation.Check(error, where=None, reason=None, source=None, excluded=None)

One check of a result: what was measured and where.

Parameters:

error (float) – The relative error the check measured.

where (float, optional) – The place of the largest error, in the variable the check runs over
(a radius or an energy). Absent where that has no meaning.

reason (str, optional) – Why the check was not judged, where it was not: such a check has no
error (NaN), its tier reads "not judged", and the result’s tier
is the worst of the checks that were.

source (str, optional) – Where a judged check read its input from another source than its
own, what it read and why; the error and the tier are as measured.

excluded (str, optional) – Where a judged check leaves part of its input unjudged, which part
and why; the error and the tier are those of the rest.

property level: int

Tier of the error, as returned by tier_level; 0 for a check not
judged, which leaves the result’s tier to the others.

property tier: str

Name of the tier of the error, or "not judged".

class pianistpy.validation.ValidationReport(keys, checks, parents_level=0)

The validation report of one result.

The report has a fixed set of keys for each kind of result and one
Check per key. It is not a log: nothing is appended to it, and a
derived result holds its own checks and the worst tier of its parents,
never a copy of their reports.

Parameters:

keys (tuple of str) – The keys this kind of result reports, in order.

checks (mapping of str to Check) – One check per key.

parents_level (int, optional) – Worst tier among the parents of the result.

property level: int

Worst tier of the result’s own checks and of its parents.

property tier: str

Name of the worst tier.

classmethod from_dict(record, key_sets=None)

Build the report from its dictionary form.

Parameters:

record (mapping) – The report as to_dict writes it.

key_sets (sequence of tuple of str, optional) – The key sets a report of this kind of result has. Given, a check
no set holds is refused by name, and the report is read against
the smallest set that holds all its checks: a check of that set
the record lacks (a file written before the check was made) is
read as not judged, with that reason.

Raises:

ResultError – If a check of the record is in none of the key sets, or no one
    set holds all its checks.

Return type:

ValidationReport

to_dict()

Return the report as a plain dictionary, in the order of its keys.

Return type:

dict[str, Any]

pianistpy.validation.announce(what, report, detail)

Say that a result lies above the goal; silent within it.

The one channel for every block: the package’s log (the pianistpy
logger, printed by default) and Python’s own warnings, as an
AccuracyWarning, so that a user can silence or escalate it with the
standard filter. Nothing is said within the goal; beyond the hard limit
the result is refused by its stage, not announced here.

Parameters:

what (str) – The result, named: "The sampler of 'nfw'".

report (ValidationReport) – Its report.

detail (str) – The checks above the goal with their errors and where they occur.

Return type:

None

pianistpy.validation.refuse_unless_asked(what, report, settings)

Refuse a result beyond the hard limit as the input of another.

A result whose report reads "refused" exists only because it was asked
for anyway (Settings(write_anyway=True)) or was read from a file that
holds one. It is consumed only under the same request: a sampler, a table
or a projection built on it carries its tier and says so.

Parameters:

what (str) – The input, named: "The model 'nfw'".

report (ValidationReport) – Its report.

settings (Settings) – The settings of the result being built from it.

Raises:

AccuracyError – If the report’s tier is "refused" and the settings do not ask for
    the result anyway.

Return type:

None

pianistpy.validation.tier_level(error)

Return the tier of a relative error.

Parameters:

error (float) – Relative error, non-negative. A value that is not a number counts as
beyond the hard limit.

Returns:

level – 0 within the goal, 1 within the ceiling, 2 within the hard limit,
3 beyond it.

Return type:

int

Examples

```
>>> [tier_level(e) for e in (1e-7, 1e-5, 1e-3, 1e-1, float("nan"))]
[0, 1, 2, 3, 3]

```

     On this page
