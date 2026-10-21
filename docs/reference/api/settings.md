# Settings

Source: https://pianistpy.com/docs/reference/api/settings

Module `pianistpy.settings`

The numerical settings of a computation: the default settings, the quick
settings, and single fields changed by an expert.

A user passes a Settings object to a computation, for example
`pp.solve(component, settings=pp.Settings(quick=True))`. The default
settings and the quick settings are the two tested settings levels: the
package states the accuracy of each level from its test suite. The quick
settings use fewer nodes and rows than the default settings and hold every
check of a result to the same bounds, \(10^{-6}\), \(10^{-4}\) and
\(10^{-2}\). The manual’s settings tables list every field with its
value at each level.

class pianistpy.settings.Field(*default*, *doc*, *quick=None*)

One numerical setting: its value at each settings level and a
description of what the setting controls.

Parameters:

- **default** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, *[*int*](https://docs.python.org/3/builtins/functions.html#int)*, *[*float*](https://docs.python.org/3/builtins/functions.html#float)* or *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The field’s value in the default settings. The type of default is
the type every value of the field must have.

- **doc** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – A description of what the field controls, as the manual’s settings
tables print it.

- **quick** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, *[*int*](https://docs.python.org/3/builtins/functions.html#int)*, *[*float*](https://docs.python.org/3/builtins/functions.html#float)* or *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*, **optional*) – The field’s value in the quick settings. Default: `None`, which
means the quick settings use the value default.

See also

[`Settings`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)
The settings of a computation.

`FIELDS`
Every field of every group.

Examples

```python
>>> field = FIELDS["sampling"]["n_nodes"]
>>> field.value("default"), field.value("quick")
(2048, 512)
```

value(*level*)

Return the field’s value at a settings level.

Parameters:

**level** (*{"default"**, **"quick"}*) – The settings level.

Returns:

**value** – The field’s value in the quick settings if level is
`"quick"` and the field has a quick value; the field’s value
in the default settings otherwise.

Return type:

[bool](https://docs.python.org/3/builtins/functions.html#bool), [int](https://docs.python.org/3/builtins/functions.html#int), [float](https://docs.python.org/3/builtins/functions.html#float) or [str](https://docs.python.org/3/builtins/stdtypes.html#str)

class pianistpy.settings.Settings(*quick=False*, *write_anyway=False*, *check_rows=False*, ***overrides*)

The numerical settings of a computation.

A Settings object starts from one of the two tested settings levels,
the default settings or the quick settings, and may change single
fields of a group. A result records its settings; a result made with a
changed field records that its settings are not a tested settings level
and lists each changed field.

Parameters:

- **quick** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether the settings start from the quick settings. Default:
`False`, which means the settings start from the default settings.

- **write_anyway** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether a computation returns a result whose own check is above
\(10^{-2}\) instead of raising AccuracyError. The returned
result’s validation report has the band `"refused"`, and an
AccuracyWarning is issued in place of the error. A model that
is not a physical equilibrium raises UnphysicalModelError
whatever the value of write_anyway. Default: `False`, which
means a result above \(10^{-2}\) raises AccuracyError.

- **check_rows** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*, **optional*) – Whether a gradient through a LosTable recomputes each
line-of-sight velocity distribution the gradient reads and compares
the recomputed distribution with the stored distribution. A
recomputed distribution must equal the stored distribution bit for
bit, and a JAX copy of the distribution’s computation (the twin
that carries the derivative) must agree with the stored
distribution to \(10^{-12}\) of the distribution’s largest
value; otherwise the gradient raises ResultError. The gradient’s value is
the same with and without the comparison; the comparison computes
each distribution read a second time. Default: `False`, which
means no distribution is recomputed.

- ****overrides** (*mapping*) – For each group to change (`inversion`, `truncation`,
`sampling`, `observables` or `losvd`), a mapping from field
name to new value. The manual’s settings tables list every field.
Default: no override, which means every field has its value at the
settings level.

Variables:

- **level** (*{"default"**, **"quick"}*) – The settings level the settings start from.

- **write_anyway** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether a result above \(10^{-2}\) is returned instead of
raising AccuracyError.

- **check_rows** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether a gradient through a LosTable recomputes and compares
each distribution the gradient reads.

- **overrides** (*mapping*) – For each changed group, the changed fields and their values. Read
only.

- **is_tested_level** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether no field is changed from the settings level.

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If write_anyway or check_rows is not a bool.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a keyword of overrides is not a group name.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a field of an override is not a field of the override’s group.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If an override’s value does not have the type of the field’s
    default value (an int is accepted for a float field).

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If a number override is zero or negative, except a field whose name
    starts with `log10_` and the value 0 of `innermost_read_radius`
    and `outer_mass_share`.

See also

`FIELDS`
Every field of every group with its value at each level.

`LEVELS`
The names of the two tested settings levels.

Examples

```python
>>> Settings().level
'default'
>>> Settings(quick=True).is_tested_level
True
>>> Settings(inversion={"n_nodes": 256}).is_tested_level
False
>>> Settings(inversion={"n_nodes": 256}).inversion["n_nodes"]
256
>>> Settings().truncation["ode_rtol"]
1e-11
```

property level: [str](https://docs.python.org/3/builtins/stdtypes.html#str)

The settings level the settings start from, `"default"` or
`"quick"`.

property write_anyway: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether a result above \(10^{-2}\) is returned instead of
raising AccuracyError.

property check_rows: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether a gradient through a LosTable recomputes and compares
each distribution the gradient reads.

property overrides: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

For each changed group, the changed fields and their values.

property is_tested_level: [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether no field is changed from the settings level.

group(*name*)

Return the value in force of every field of one group.

Parameters:

**name** (*{"inversion"**, **"truncation"**, **"sampling"**, **"observables"**, **"losvd"}*) – The group’s name.

Returns:

**values** – Each field of the group mapped to the field’s override if the
settings change the field, and to the field’s value at the
settings level otherwise. Read only.

Return type:

mapping

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If name is not a group name.

See also

[`resolved`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings.resolved)
The values in force of every group.

Examples

```python
>>> Settings(quick=True).group("sampling")["n_nodes"]
512
```

property inversion: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

The values in force of the group `inversion`.

property truncation: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

The values in force of the group `truncation`.

property sampling: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

The values in force of the group `sampling`.

property observables: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

The values in force of the group `observables`.

property losvd: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

The values in force of the group `losvd`.

resolved()

Return the value in force of every field of every group.

Returns:

**values** – Each group name mapped to a dictionary of the group’s fields and
their values in force.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`group`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings.group)
The values in force of one group.

Examples

```python
>>> sorted(Settings().resolved())
['inversion', 'losvd', 'observables', 'sampling', 'truncation']
```

to_dict()

Return the settings as the dictionary written to result files.

Returns:

**record** – The entries of numerical (`"level"` and `"overrides"`),
and `"write_anyway": True` and `"check_rows": True` if the
settings set those options.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

See also

[`from_dict`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings.from_dict)
The settings built from the dictionary.

[`numerical`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings.numerical)
The entries that determine a result’s values.

Examples

```python
>>> Settings(quick=True, write_anyway=True).to_dict()
{'level': 'quick', 'overrides': {}, 'write_anyway': True}
```

numerical()

Return the settings that determine a result’s values: the
settings level and the overrides.

A result’s identity is computed from this dictionary, so that two
results computed with the same level and overrides share an
identity whatever their write_anyway and check_rows.

Returns:

**record** – The entries `"level"` and `"overrides"` (each changed group
mapped to its changed fields and their values).

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

Examples

```python
>>> Settings(inversion={"n_nodes": 256}).numerical()
{'level': 'default', 'overrides': {'inversion': {'n_nodes': 256}}}
```

classmethod from_dict(*record*)

Build settings from the dictionary that to_dict writes.

Parameters:

**record** (*mapping*) – A dictionary in the form to_dict returns. A missing
`"level"` entry is read from the entry `"tier"`, the
entry’s former name, and else as `"default"`.

Returns:

**settings** – The settings the dictionary describes.

Return type:

[Settings](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings)

Raises:

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If the dictionary’s settings level is not one of LEVELS.

- [**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If an override in the dictionary is not accepted by Settings.

See also

[`to_dict`](https://pianistpy.com/docs/reference/api/settings#pianistpy.settings.Settings.to_dict)
The dictionary form of settings.

Examples

```python
>>> settings = Settings(sampling={"n_rows_max": 1000})
>>> Settings.from_dict(settings.to_dict()) == settings
True
```
