# Runtime options (pianistpy.runtime)

Source: https://pianistpy.com/docs/reference/api/runtime

# Runtime options (pianistpy.runtime)

Runtime options of the session: verbosity, timing and workers.

None of these changes a number, and none is part of the identity of a
result. Each is set once for the session, by a call or by an environment
variable, and no function of the package takes it as an argument.

class pianistpy.runtime.Timing(stages=<factory>, memory=<factory>)

Wall times of the stages of one computation, and the memory they reached.

Parameters:

stages (dict of str to float) – Seconds spent in each stage, in the order the stages first ran. A
stage that runs more than once accumulates.

memory (dict of str to int) – Peak resident memory of the process, in bytes, as it stood when each
stage last ended. The value never decreases from one stage to the
next: a stage raised the peak if its value exceeds the one before.

property total: float

Sum of the stage times, in seconds.

to_dict()

Return the stage times as a plain dictionary.

Return type:

dict[str, float]

pianistpy.runtime.allowed_cores()

Return the number of cores this process is allowed to use.

Inside a batch job the scheduler’s own statement of the job’s cores is
taken where it gives one; otherwise the cores the operating system lets
the process run on; otherwise the cores of the machine.

Return type:

int

pianistpy.runtime.card_memory()

Return the free and the total memory of the GPU JAX runs on, in
bytes, as the CUDA driver states them; None off a CUDA GPU.

The driver is asked in the device’s primary context, the one JAX uses,
so asking makes no context of its own. Its free memory counts what
every process holds on the GPU, a CUDA context included, which the
allocator’s own statistics do not.

Return type:

tuple[int, int] | None

pianistpy.runtime.configure_jax()

Set JAX as the package needs it: 64 bits, the platform and the
compilation cache.

Called once when the package is imported. The environment variable
PIANISTPY_PLATFORM names the platform (set_platform), and
PIANISTPY_COMPILATION_CACHE set to off leaves the cache off.

Return type:

None

pianistpy.runtime.default_compilation_cache()

Return the default folder of the compilation cache.

The environment variable PIANISTPY_COMPILATION_CACHE if it is set,
None if it is set to off (the cache is off), otherwise
~/.cache/pianistpy/jax.

Return type:

str | None

pianistpy.runtime.get_workers()

Return the number of workers the current setting resolves to.

Return type:

int

pianistpy.runtime.log_to_file(path, level='debug')

Write the messages of the enclosed block to a file as well.

Parameters:

path (path) – File to append to.

level ({"quiet", "info", "debug"}, optional) – Lowest level written to the file.

Return type:

Iterator[None]

pianistpy.runtime.peak_memory()

Return the largest resident memory this process has held, in bytes.

This is the memory of the host process. Memory held on a GPU is not
part of it.

Returns:

peak – The peak resident memory since the process started, or None where
the operating system does not report it.

Return type:

int or None

pianistpy.runtime.recent_messages()

Return the most recent messages of the package, oldest first.

The messages are kept whatever the verbosity, so that a quiet run that
fails can still say what it was doing.

Return type:

tuple[str, …]

pianistpy.runtime.platform()

Return the backend JAX runs on: "cpu", "gpu" or another.

Asking initializes the backend if it is not yet.

Return type:

str

pianistpy.runtime.set_compilation_cache(cache=True, max_size=None)

Turn the persistent compilation cache of JAX on or off.

With the cache on, the code compiled in one run is reused by the next,
which shortens every run after the first. It changes no number. The
cache is on by default, its folder unbounded in size, as JAX’s own is.

A bound costs time: JAX then takes a lock on the folder at every read
and write, and scans every entry at every write. Measured on a folder of
22,000 entries on a network file system, a test that takes 5.6 minutes
with no bound (4.3 to 5.2 minutes with eight runs at once on the folder,
as with the cache off) took 2 hours 8 minutes alone with a bound of
512 MiB.

Parameters:

cache (bool or path, optional) – True for the default folder (default_compilation_cache, or the
package’s own folder where the environment variable turns the cache
off), a path for another folder, False to turn the cache off.

max_size (int or None, optional) – Bound on the folder in bytes, above which JAX evicts the oldest
entries; None, the default, leaves the folder unbounded.

Returns:

folder – The folder in use, or None if the cache is off.

Return type:

str or None

pianistpy.runtime.set_platform(name)

Choose the hardware JAX runs on, before the first computation.

The CPU-only mode ("cpu") is first class: every function of the
package runs on it and gives the same numbers on every machine. It
changes no number on a machine without a GPU; on one with a GPU the
CPU and GPU results differ at rounding, as the provenance of each
file records (pianistpy.io.backend_provenance).

Parameters:

name ({"auto", "cpu", "gpu"}) – "auto" leaves JAX its own order (a GPU where one is usable);
"cpu" is the CPU-only mode; "gpu" asks for the CUDA backend
with the CPU behind it. Any other string is handed to JAX as its
own platform list (for example "rocm,cpu").

Return type:

None

Notes

JAX picks its backend at the first computation of the process and
keeps it: call this before any result is computed, or set the
environment variable PIANISTPY_PLATFORM before the package is
imported. A later call has no effect and is reported at info.

pianistpy.runtime.set_timing(enabled)

Switch the printing of stage times on or off.

The times are always measured and kept with the result. This switch
controls printing only and is independent of the verbosity. The
environment variable PIANISTPY_TIMING sets it at import.

Parameters:

enabled (bool)

Return type:

None

pianistpy.runtime.set_verbosity(level)

Set what the package prints for the rest of the session.

Parameters:

level ({"quiet", "info", "debug"}) – "quiet" shows warnings and errors only and is the default;
"info" adds one line per stage; "debug" adds the detail of
each stage.

Return type:

None

Notes

The verbosity never changes what is computed. The environment variable
PIANISTPY_VERBOSITY sets the level at import.

pianistpy.runtime.set_workers(value)

Set how many cores the package uses for parallel work.

Parameters:

value ({"auto", "full", "half"} or int) – "auto", the default, uses every allowed core inside a batch job
and half of them anywhere else; "full" every allowed core;
"half" half of them; an integer exactly that many.

Return type:

None

Notes

The number of workers never changes a number: a result is the same, to
the bit, for any value. The environment variable PIANISTPY_WORKERS
sets the value at import.

pianistpy.runtime.timed(stage)

Time one stage and announce it.

Parameters:

stage (str) – Name of the stage, from the fixed list of its computation.

Return type:

Iterator[None]

pianistpy.runtime.timing_enabled()

Return whether stage times are printed.

Return type:

bool

pianistpy.runtime.timing_record()

Collect the stage times of the enclosed computation.

Yields:

timing (Timing) – The record that every timed stage of the block adds to.

Return type:

Iterator[Timing]

pianistpy.runtime.workers_explicit()

Whether the workers were set to an exact number (set_workers).

Return type:

bool

     On this page
