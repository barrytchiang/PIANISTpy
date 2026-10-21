# Validation

Source: https://pianistpy.com/docs/reference/api/validation

Module `pianistpy.validation`

The accuracy bands of a result and the validation report that states them.

Every result of the package (a model, a distribution function, a sampler, a
projected table) carries a ValidationReport as its `validation`
attribute. A user reads the band of a result from `result.validation.tier`
and the measured relative errors from `result.validation.checks`.
tier_level maps one relative error \(\epsilon\) to its band:

\[\begin{split}\text{band}(\epsilon) =
\begin{cases}
\text{silent}, & \epsilon \le 10^{-6},\\
\text{recorded}, & 10^{-6} < \epsilon \le 10^{-4},\\
\text{warned}, & 10^{-4} < \epsilon \le 10^{-2},\\
\text{refused}, & \epsilon > 10^{-2}.
\end{cases}\end{split}\]

The technical notes, Section T:0.5, state how each band is reported.

class pianistpy.validation.Check(*error*, *where=None*, *reason=None*, *source=None*, *excluded=None*)

One check of a result: the relative error measured and where.

A check is either judged or not judged. A judged check has a measured
relative error and the band of that error. A check that is not judged
has the error NaN, a stated reason, and the name `"not judged"` in
place of a band; a check that is not judged does not change the band of
the result.

Parameters:

- **error** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The relative error the check measured. NaN for a check that is not
judged.

- **where** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*, **optional*) – The position of the largest relative error, in the variable the
check runs over (a radius or an energy, in internal units).
Default: `None`, which means the check has no single position of
its largest error.

- **reason** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – The reason the check is not judged. Default: `None`, which means
the check is judged.

- **source** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – For a judged check that read its input from a source other than the
result’s own data: the source read, and the reason. The error and
the band are those measured on that source. Default: `None`,
which means the check read the result’s own data.

- **excluded** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – For a judged check that leaves part of its input unjudged: the part
left out, and the reason. The error and the band are those of the
remaining input. Default: `None`, which means the check judged
its whole input.

Variables:

- **level** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The band of the check’s error as returned by tier_level; 0 for a
check that is not judged.

- **tier** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the check’s band, one of TIER_NAMES, or
`"not judged"`.

See also

[`ValidationReport`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)
The checks of one result together.

[`tier_level`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.tier_level)
The band of a relative error.

Examples

```python
>>> check = Check(error=3e-5, where=12.5)
>>> check.tier
'recorded'
>>> Check(error=float("nan"), reason="the table has no outer rows").tier
'not judged'
```

property level: [int](https://docs.python.org/3/builtins/functions.html#int)

The band of the check’s error as a level from 0 to 3; 0 for a check
that is not judged.

property tier: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

The name of the check’s band, or `"not judged"`.

class pianistpy.validation.ValidationReport(*keys*, *checks*, *parents_level=0*)

The validation report of one result: its checks and its band.

Each kind of result reports a fixed set of keys, one Check per key.
The band of the result is the worst band among the result’s own checks
and the worst band among the results the result was built from (its
parents). A report holds only the result’s own checks: a derived result
carries the worst level of its parents as one number, not the parents’
reports.

Parameters:

- **keys** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The keys the kind of result reports, in the order the report lists
them.

- **checks** (*mapping** of **str to Check*) – One Check for each entry of keys.

- **parents_level** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*, **optional*) – The worst level, from 0 to 3, among the parents of the result.
Default: `0`, which means the result has no parent above
\(10^{-6}\).

Variables:

- **level** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – The worst level among the result’s own checks and parents_level.

- **tier** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the worst band, `TIER_NAMES[level]`.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the keys of checks are not the entries of keys.

See also

[`Check`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.Check)
One check of the report.

[`tier_level`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.tier_level)
The band of a relative error.

Examples

```python
>>> report = ValidationReport(
...     keys=("round_trip", "top_form"),
...     checks={"round_trip": Check(2e-7), "top_form": Check(3e-5)},
... )
>>> report.tier
'recorded'
>>> ValidationReport(report.keys, report.checks, parents_level=2).tier
'warned'
```

property level: [int](https://docs.python.org/3/builtins/functions.html#int)

The worst level among the result’s own checks and its parents.

property tier: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

The name of the worst band of the result.

classmethod from_dict(*record*, *key_sets=None*)

Build a report from the dictionary that to_dict writes.

Parameters:

- **record** (*mapping*) – The report in the form to_dict returns: the entries
`"parents_tier"` and `"checks"`.

- **key_sets** (*sequence** of *[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)* of *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – Every key set that a report of the record’s kind of result can
have. With key_sets given, the report takes the smallest key
set that holds every check of the record, and a key of that set
that the record lacks becomes a check that is not judged, with
the reason that the file does not hold the check. Default:
`None`, which means the report takes the record’s own checks
as its keys.

Returns:

**report** – The report the record describes.

Return type:

[ValidationReport](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)

Raises:

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If key_sets is given and a check of the record belongs to no
    key set of key_sets.

- [**ResultError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.ResultError) – If key_sets is given and no single key set of key_sets holds
    every check of the record.

See also

[`to_dict`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport.to_dict)
The dictionary form of a report.

Examples

```python
>>> report = ValidationReport(("a",), {"a": Check(3e-5)})
>>> record = report.to_dict()
>>> ValidationReport.from_dict(record) == report
True
>>> wider = ValidationReport.from_dict(record, [("a", "b")])
>>> wider.checks["b"].tier
'not judged'
```

to_dict()

Return the report as a plain dictionary.

Returns:

**record** – The entries `"tier"` (the report’s band), `"parents_tier"`
(the name of the parents’ worst band) and `"checks"` (one
dictionary for each key, in the order of keys, with the
entries `"error"`, `"where"` and `"tier"`, and
`"reason"`, `"source"` or `"excluded"` where the check
states one).

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`from_dict`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport.from_dict)
The report built from the dictionary.

Examples

```python
>>> report = ValidationReport(("a",), {"a": Check(3e-5, where=1.0)})
>>> record = report.to_dict()
>>> record["tier"], record["parents_tier"]
('recorded', 'silent')
>>> record["checks"]["a"]
{'error': 3e-05, 'where': 1.0, 'tier': 'recorded'}
```

pianistpy.validation.announce(*what*, *report*, *detail*)

Report a result whose relative error is above \(10^{-6}\).

The announcement depends on the band of report. A result in the band
`"silent"` (at most \(10^{-6}\)) is not announced. A result in the
band `"recorded"` (above \(10^{-6}\), at most \(10^{-4}\)) is
written as one line to the `pianistpy` logger at level INFO, and no
warning is issued. A result in the band `"warned"` (above
\(10^{-4}\), at most \(10^{-2}\)) is written to the
`pianistpy` logger at level WARNING and issued as an
AccuracyWarning, which the standard `warnings` filter can silence or
turn into an error. A result in the band `"refused"` (above
\(10^{-2}\)) reaches announce only if its settings set
`write_anyway=True`; that result is announced as a result in the band
`"warned"` is.

Parameters:

- **what** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The result’s name as messages write it, for example
`"The sampler of 'nfw'"`.

- **report** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The validation report of the result.

- **detail** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The checks above \(10^{-6}\), each with its relative error and
the position of its largest error.

Warns:

**AccuracyWarning** – If the band of report is `"warned"` or `"refused"`.

Return type:

None

See also

[`refuse_unless_asked`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.refuse_unless_asked)
The error for an input in the band `"refused"`.

`set_verbosity`
The lowest level of the messages printed on the screen.

Examples

```python
>>> silent = ValidationReport(("a",), {"a": Check(1e-8)})
>>> announce("The sampler of 'x'", silent, "a 1.0e-08")
```

pianistpy.validation.refuse_unless_asked(*what*, *report*, *settings*)

Raise if an input result is in the band `"refused"` and the settings
do not ask for the result anyway.

A result in the band `"refused"` (relative error above
\(10^{-2}\)) exists only if the settings that built the result set
`write_anyway=True`, or if the result was read from a file that holds
such a result. A computation that takes such a result as input proceeds
only if its own settings also set `write_anyway=True`; the new result
then carries the band `"refused"`.

Parameters:

- **what** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The input result’s name as messages write it, for example
`"The model 'nfw'"`.

- **report** ([*ValidationReport*](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.ValidationReport)) – The validation report of the input result.

- **settings** ([*Settings*](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)) – The settings of the computation that takes the input result.

Raises:

[**AccuracyError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.AccuracyError) – If the band of report is `"refused"` and settings does not set
    `write_anyway=True`.

Return type:

None

See also

[`announce`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.announce)
The message for a result above \(10^{-6}\).

`Settings`
The setting `write_anyway`.

Examples

```python
>>> from pianistpy import Settings
>>> report = ValidationReport(("a",), {"a": Check(0.5)})
>>> report.tier
'refused'
>>> refuse_unless_asked("The model 'x'", report, Settings(write_anyway=True))
>>> refuse_unless_asked("The model 'x'", report, Settings())
Traceback (most recent call last):
...
pianistpy.exceptions.AccuracyError: The model 'x' is beyond 0.01 (tier ...
```

pianistpy.validation.tier_level(*error*)

Return the band of a relative error as a level from 0 to 3.

The level indexes TIER_NAMES: level 0 is `"silent"`, level 1 is
`"recorded"`, level 2 is `"warned"` and level 3 is `"refused"`.

Parameters:

**error** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – Relative error \(\epsilon\), a non-negative number. A NaN error
has level 3, the level of an error above \(10^{-2}\).

Returns:

**level** – 0 if error is at most \(10^{-6}\); 1 if error is above
\(10^{-6}\) and at most \(10^{-4}\); 2 if error is above
\(10^{-4}\) and at most \(10^{-2}\); 3 if error is above
\(10^{-2}\) or is NaN.

Return type:

[int](https://docs.python.org/3/builtins/functions.html#int)

See also

`TIER_NAMES`
The name of each level.

`TIER_BOUNDS`
The three relative errors that separate the levels.

[`Check`](https://pianistpy.com/docs/reference/api/validation#pianistpy.validation.Check)
One measured relative error and its level.

Examples

```python
>>> [tier_level(e) for e in (1e-7, 1e-5, 1e-3, 1e-1, float("nan"))]
[0, 1, 2, 3, 3]
>>> TIER_NAMES[tier_level(3e-5)]
'recorded'
```
