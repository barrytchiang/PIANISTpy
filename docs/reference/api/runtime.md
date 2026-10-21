# Runtime options

Source: https://pianistpy.com/docs/reference/api/runtime

Module `pianistpy.runtime`

Options of the Python session: the messages printed, the stage times, the
number of cores, the hardware JAX runs on, and JAX’s compilation cache.

A user sets each option once for the session, by a call such as
`pp.set_verbosity("info")` or `pp.set_cores(8)`, or by an environment
variable read when the package is imported: `PIANISTPY_VERBOSITY`,
`PIANISTPY_TIMING`, `PIANISTPY_CORES`, `PIANISTPY_PLATFORM` and
`PIANISTPY_COMPILATION_CACHE`. No option changes a computed number, and no
option is part of a result’s identity. No computation of the package takes
a runtime option as an argument.

class pianistpy.runtime.Timing(*stages=<factory>*, *memory=<factory>*)

The elapsed time of each stage of one computation, and the largest
memory of the process at the end of each stage.

Parameters:

- **stages** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to float**, **optional*) – Each stage’s name mapped to the seconds spent in the stage, in the
order in which the stages first ran. A stage that runs more than
once has the sum of its times. Default: an empty dictionary.

- **memory** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)* of **str to int**, **optional*) – Each stage’s name mapped to the largest resident memory of the
process, in bytes, at the last end of the stage (peak_memory).
The largest memory never decreases from one stage to the next, so
a stage raised the largest memory if the stage’s value exceeds the
value of the stage before. Default: an empty dictionary.

Variables:

**total** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) – The sum of the stage times, in seconds.

See also

[`timing_record`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.timing_record)
Collect a Timing for a block of code.

[`set_timing`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_timing)
Print each result’s stage times.

Examples

```python
>>> timing = Timing(stages={"inversion": 1.5, "truncation": 0.25})
>>> timing.total
1.75
```

property total: [float](https://docs.python.org/3/builtins/functions.html#float)

The sum of the stage times, in seconds.

to_dict()

Return the stage times as a plain dictionary.

Returns:

**stages** – Each stage’s name mapped to the seconds spent in the stage.

Return type:

[dict](https://docs.python.org/3/builtins/stdtypes.html#dict) of str to float

Examples

```python
>>> Timing(stages={"inversion": 1.5}).to_dict()
{'inversion': 1.5}
```

pianistpy.runtime.allowed_cores()

Return the number of cores the Python process may use.

The number is the first one available of three: the job’s number of
cores as a batch scheduler states the number in its environment
variable; the number of cores the operating system lets the process run
on; the number of cores of the machine.

Returns:

**cores** – The number of cores, at least 1.

Return type:

[int](https://docs.python.org/3/builtins/functions.html#int)

See also

[`set_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_cores)
Choose the number of cores the package uses.

[`get_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.get_cores)
The number of cores the package uses.

Examples

```python
>>> allowed_cores() >= 1
True
```

pianistpy.runtime.card_memory()

Return the free memory and the total memory of the GPU that JAX
runs on, in bytes, as the CUDA driver states them.

The free memory counts the memory that every process holds on the GPU,
including each process’s CUDA context; JAX’s allocator statistics do
not count the contexts. The call uses the GPU’s primary context, which
JAX also uses, so the call creates no CUDA context.

Returns:

**memory** – The free memory and the total memory, in bytes; None if JAX does
not run on a GPU, if the CUDA driver library cannot be loaded, or
if the driver reports an error.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) of ([int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)) or None

See also

[`platform`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.platform)
The hardware JAX runs on.

[`peak_memory`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.peak_memory)
The largest memory of the process on the host.

Examples

```python
>>> memory = card_memory()
>>> memory is None or memory[0] <= memory[1]
True
```

pianistpy.runtime.configure_jax()

Configure JAX for the package: 64-bit floating point, the hardware,
and the compilation cache.

The package calls configure_jax once, when the package is imported.
The environment variable `PIANISTPY_PLATFORM`, if set, chooses the
hardware (set_platform). The compilation cache is turned on in the
folder of default_compilation_cache, unless
`PIANISTPY_COMPILATION_CACHE` turns the cache off.

See also

[`set_platform`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_platform)
Choose the hardware JAX runs on.

[`set_compilation_cache`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_compilation_cache)
Turn the compilation cache on or off.

Examples

```python
>>> import jax
>>> jax.config.jax_enable_x64
True
```

Return type:

None

pianistpy.runtime.cores_explicit()

Return whether the number of cores was set as an integer.

Returns:

**explicit** – True if the last call of set_cores, or `PIANISTPY_CORES`, gave
an integer; False if the value is `"auto"`, `"full"` or
`"half"`.

Return type:

[bool](https://docs.python.org/3/builtins/functions.html#bool)

See also

[`set_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_cores)
Choose the number of cores the package uses.

Examples

```python
>>> set_cores(2)
>>> cores_explicit()
True
>>> set_cores("auto")
>>> cores_explicit()
False
```

pianistpy.runtime.default_compilation_cache()

Return the folder that the compilation cache uses by default.

Returns:

**folder** – None if the environment variable `PIANISTPY_COMPILATION_CACHE` is
`0`, `false`, `no` or `off`, which turns the cache off; the
variable’s value, a folder’s path, if the variable has any other
value;
`~/.cache/pianistpy/jax` if the variable is unset.

Return type:

[str](https://docs.python.org/3/builtins/stdtypes.html#str) or None

See also

[`set_compilation_cache`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_compilation_cache)
Turn the compilation cache on or off.

Examples

```python
>>> folder = default_compilation_cache()
>>> folder is None or isinstance(folder, str)
True
```

pianistpy.runtime.get_cores()

Return the number of cores the package uses.

Returns:

**cores** – The integer last given to set_cores; for `"full"`, the count of
allowed_cores; for `"half"`, half of that count, at least 1;
for `"auto"`, the count of allowed_cores inside a batch job and
half of that count, at least 1, elsewhere.

Return type:

[int](https://docs.python.org/3/builtins/functions.html#int)

See also

[`set_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_cores)
Choose the number of cores the package uses.

Examples

```python
>>> set_cores(3)
>>> get_cores()
3
>>> set_cores("auto")
```

pianistpy.runtime.log_to_file(*path*, *level='debug'*)

Write the package’s messages during a `with` block to a file.

The file receives the messages in addition to the screen; the screen
keeps the session’s verbosity.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like*) – The file the messages are appended to. The file is created if
path does not exist.

- **level** (*{"quiet"**, **"info"**, **"debug"}**, **optional*) – The verbosity of the file, with the meanings of set_verbosity.
Default: `"debug"`, which means every message is written.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If level is not `"quiet"`, `"info"` or `"debug"`.

Return type:

[*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[None]

See also

[`set_verbosity`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_verbosity)
Choose the messages printed on the screen.

Examples

```python
>>> import os, tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "run.log")
>>> with log_to_file(path):
...     log.debug("A message for the file.")
>>> open(path).read()
'pianistpy DEBUG: A message for the file.\n'
```

pianistpy.runtime.peak_memory()

Return the largest resident memory of the Python process, in bytes.

The value counts the memory of the Python process on the host. Memory
held on a GPU is not counted.

Returns:

**peak** – The largest resident memory since the process started, in bytes;
None if the operating system does not report the value.

Return type:

[int](https://docs.python.org/3/builtins/functions.html#int) or None

See also

[`Timing`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.Timing)
The stage times and memory of one computation.

Examples

```python
>>> peak = peak_memory()
>>> peak is None or peak > 0
True
```

pianistpy.runtime.recent_messages()

Return the package’s most recent messages, oldest first.

The package keeps its 50 most recent messages at every verbosity, so
that the messages that led to an error can be read after a run at the
verbosity `"quiet"`.

Returns:

**messages** – Each message as printed: the logger’s name, the message’s level and
the message text.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) of [str](https://docs.python.org/3/builtins/stdtypes.html#str)

See also

[`set_verbosity`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_verbosity)
Choose the messages printed on the screen.

Examples

```python
>>> log.debug("A message kept at every verbosity.")
>>> recent_messages()[-1]
'pianistpy DEBUG: A message kept at every verbosity.'
```

pianistpy.runtime.platform()

Return the hardware JAX runs on.

A call of platform makes JAX choose its hardware if no computation
has chosen the hardware yet.

Returns:

**name** – JAX’s name of the hardware: `"cpu"`, `"gpu"`, or another name
that JAX reports.

Return type:

[str](https://docs.python.org/3/builtins/stdtypes.html#str)

See also

[`set_platform`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_platform)
Choose the hardware JAX runs on.

Examples

```python
>>> isinstance(platform(), str)
True
```

pianistpy.runtime.set_compilation_cache(*cache=True*, *max_size=None*)

Turn JAX’s persistent compilation cache on or off.

With the cache on, a later Python session reuses the code that an
earlier session compiled, which shortens every session after the first.
The cache changes no computed number. When the package is imported,
the cache is on, in the folder of default_compilation_cache, with no
bound on the folder’s size, unless `PIANISTPY_COMPILATION_CACHE`
turns the cache off.

A bound on the folder’s size slows every compilation: with a bound,
JAX locks the folder at every read and every write, and scans every
entry of the folder at every write. On a shared or network file
system, keep the folder unbounded.

Parameters:

- **cache** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)* or *[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)* or **path-like**, **optional*) – True for the folder of default_compilation_cache, or for
`~/.cache/pianistpy/jax` if the environment variable turns the
cache off; a path for another folder; False to turn the cache off.
Default: `True`.

- **max_size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)* or **None**, **optional*) – The largest size of the folder, in bytes; above max_size, JAX
removes the oldest entries. Default: `None`, which means the
folder has no bound.

Returns:

**folder** – The folder the cache uses; None if the cache is off.

Return type:

[str](https://docs.python.org/3/builtins/stdtypes.html#str) or None

See also

[`default_compilation_cache`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.default_compilation_cache)
The folder the cache uses by default.

Examples

```python
>>> on = default_compilation_cache() is not None
>>> set_compilation_cache(on) == default_compilation_cache()
True
```

pianistpy.runtime.set_cores(*value*)

Set the number of cores the package uses for parallel computations.

The number of cores changes no computed number: a result is the same,
bit for bit, for every value. Each parallel process of the package runs
one thread, so the number of cores is also the largest number of
parallel processes. The environment variable `PIANISTPY_CORES` sets
the value when the package is imported; with the variable unset, the
value is `"auto"`.

Parameters:

**value** (*{"auto"**, **"full"**, **"half"}** or *[*int*](https://docs.python.org/3/builtins/functions.html#int)) – `"auto"` uses every core allowed_cores counts inside a batch job
and half of those cores, at least 1, elsewhere. `"full"` uses
every core allowed_cores counts. `"half"` uses half of those
cores, at least 1. A positive integer, or a string of digits, uses
that number of cores.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If value is none of `"auto"`, `"full"`, `"half"` and a
    positive integer.

Return type:

None

See also

[`get_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.get_cores)
The number of cores the package uses.

[`allowed_cores`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.allowed_cores)
The number of cores the process may use.

Examples

```python
>>> set_cores(2)
>>> get_cores()
2
>>> set_cores("auto")
```

pianistpy.runtime.set_platform(*name*)

Choose the hardware JAX runs on, before the session’s first
computation.

Every function of the package runs on the CPU alone (`"cpu"`), and
the CPU alone gives the same numbers on every machine. On a machine
with a GPU, results computed on the GPU and results computed on the CPU
differ at the level of rounding; each result file records the hardware
that computed the result.

JAX chooses its hardware at the first computation of the Python process
and keeps the choice for the rest of the process. A call of
set_platform after the first computation does not change the
hardware. To choose the hardware before the package is imported, set
the environment variable `PIANISTPY_PLATFORM`.

Parameters:

**name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"auto"` leaves JAX’s own order: a usable GPU first, the CPU
otherwise. `"cpu"` runs on the CPU alone. `"gpu"` runs on a
CUDA GPU first and on the CPU second. Any other string is passed to
JAX as JAX’s platform list, for example `"rocm,cpu"`.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If name is not a non-empty string.

Return type:

None

See also

[`platform`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.platform)
The hardware JAX runs on.

Examples

```python
>>> set_platform("")
Traceback (most recent call last):
...
pianistpy.exceptions.InputError: ...
```

pianistpy.runtime.set_timing(*enabled*)

Turn the printing of each computation’s stage times on or off.

The package measures the stage times of every computation and keeps
the times with the result, whether the printing is on or off. The
printing does not depend on the verbosity. The environment variable
`PIANISTPY_TIMING` turns the printing on when the package is imported
if the variable is `1`, `true`, `yes` or `on`.

Parameters:

**enabled** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) – Whether the stage times are printed at the end of each computation.

Return type:

None

See also

[`timing_enabled`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.timing_enabled)
Whether the stage times are printed.

[`Timing`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.Timing)
The stage times of one computation.

Examples

```python
>>> set_timing(True)
>>> timing_enabled()
True
>>> set_timing(False)
```

pianistpy.runtime.set_verbosity(*level*)

Set which of the package’s messages are printed on the screen.

The verbosity holds for the rest of the session. The verbosity changes
no computed number. The environment variable `PIANISTPY_VERBOSITY`
sets the verbosity when the package is imported; with the variable
unset, the verbosity is `"quiet"`.

Parameters:

**level** (*{"quiet"**, **"info"**, **"debug"}*) – `"quiet"` prints warnings and errors only. `"info"` also prints
one line when each stage of a computation starts, and the messages
of results with relative error above \(10^{-6}\) and at most
\(10^{-4}\). `"debug"` also prints the details of each stage.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If level is not `"quiet"`, `"info"` or `"debug"`.

Return type:

None

See also

[`log_to_file`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.log_to_file)
Write the messages of a block to a file.

[`recent_messages`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.recent_messages)
The most recent messages at every verbosity.

Examples

```python
>>> set_verbosity("info")
>>> set_verbosity("quiet")
```

pianistpy.runtime.timed(*stage*)

Time one stage of a computation in a `with` block.

The stage’s start is written to the log at level INFO and the stage’s
time at level DEBUG. Inside a timing_record block, the stage’s time
and the process’s largest memory are added to the record.

Parameters:

**stage** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The stage’s name.

Return type:

[*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[None]

See also

[`timing_record`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.timing_record)
Collect the stage times of a block.

Examples

```python
>>> with timed("example"):
...     pass
```

pianistpy.runtime.timing_enabled()

Return whether the stage times are printed at the end of each
computation.

Returns:

**enabled** – The value last given to set_timing.

Return type:

[bool](https://docs.python.org/3/builtins/functions.html#bool)

See also

[`set_timing`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.set_timing)
Turn the printing of stage times on or off.

Examples

```python
>>> isinstance(timing_enabled(), bool)
True
```

pianistpy.runtime.timing_record()

Collect the stage times of the computation in a `with` block.

With the printing of stage times on (set_timing), the times are
printed when the block ends.

Yields:

**timing** (*Timing*) – The record to which each timed stage inside the block adds its
time and memory.

Return type:

[*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[*Timing*](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.Timing)]

See also

[`timed`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.timed)
Time one stage.

[`Timing`](https://pianistpy.com/docs/reference/api/runtime#pianistpy.runtime.Timing)
The stage times of one computation.

Examples

```python
>>> with timing_record() as record:
...     with timed("example"):
...         pass
>>> list(record.stages)
['example']
```
