# Runtime options

Source: https://pianistpy.com/docs/guide/switches.html

# Runtime options

None of these changes a number.

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
PIANISTPY_WORKERS.

The compilation cache keeps JAX’s compiled kernels between sessions in one folder of the user’s cache directory,
named for the package and its version, with a text file inside saying what it is; the switch reports where it
is and clears it.
