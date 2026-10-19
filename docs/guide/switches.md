# Runtime options

Source: https://pianistpy.com/docs/guide/switches

# Runtime options

None of these changes a number, except the platform: a GPU and a CPU give results that differ at rounding, and
each file records which made it.

```
pp.set_verbosity("info")   # "quiet" (the default), "info" or "debug"
pp.set_timing(True)        # print the wall time of each stage
pp.set_workers("half")     # "auto" (the default), "full", "half" or a number of cores
pp.set_compilation_cache(False)  # the compilation cache is on by default; a path moves it
pp.set_platform("cpu")     # "auto" (the default), "cpu" (the CPU-only mode) or "gpu"; before the first computation

```

The CPU-only mode gives the same numbers on every machine; the environment variable PIANISTPY_PLATFORM=cpu
sets it before the package is imported. Each file records the backend that made it.

The workers’ switch sets how many cores the package’s parallel work uses; "auto" uses every allowed core
inside a batch job and half of them anywhere else, and no value changes a number. The verbosity, the timing and
the workers can also be set by the environment variables PIANISTPY_VERBOSITY, PIANISTPY_TIMING and
PIANISTPY_WORKERS, in any case; PIANISTPY_TIMING reads 1, true, yes or on and 0,
false, no or off, and any other value is refused by name.

The compilation cache keeps JAX’s compiled kernels between sessions in one folder: ~/.cache/pianistpy/jax
by default, or the folder the environment variable PIANISTPY_COMPILATION_CACHE names (off, 0,
false or no turns the cache off). The switch returns the folder in use, and the cache changes no number.
