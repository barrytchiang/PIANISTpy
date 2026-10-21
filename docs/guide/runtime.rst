Runtime options
===============

The runtime options below set how PIANISTpy runs. Every option except the platform leaves every result identical bit
for bit. CPU-only and GPU-enabled results differ by less than :math:`10^{-10}` in relative terms.

.. code-block:: python

   pp.set_verbosity("info")   # "quiet" (the default), "info" or "debug"
   pp.set_timing(True)        # print the wall-clock time of each step of the computation
   pp.set_cores("half")       # "auto" (the default), "full", "half" or a number of cores
   pp.set_compilation_cache(False)  # the compilation cache is on by default; a path moves it
   pp.set_platform("cpu")     # "auto" (the default), "cpu" (the CPU-only mode) or "gpu", set before the first computation

The CPU-only mode gives the same numbers on every machine. The environment variable ``PIANISTPY_PLATFORM=cpu``,
set before the package is imported, also selects the CPU-only mode. Each result file records the platform (CPU or
GPU) and the devices that computed the result.

On shared clusters, a GPU in the Exclusive_Process compute mode serves one process at a time. If another process
holds the GPU, JAX stops with ``Unable to initialize backend 'cuda'``. Free the GPU, share the GPU through an MPS
daemon (``nvidia-cuda-mps-control -d``), or use the CPU-only mode.

``set_cores`` sets how many cores the package uses for parallel work. ``"auto"`` uses every core the process may use
inside a batch job, and half of those cores, at least one, anywhere else. No value of ``set_cores`` changes a computed
number. Each parallel process of the package runs one thread, so the number of cores is also the largest number of
parallel processes.

Environment variables set the verbosity, the timing and the number of cores at import: ``PIANISTPY_VERBOSITY``,
``PIANISTPY_TIMING`` and ``PIANISTPY_CORES``. The values of the three variables are read in any letter case.
``PIANISTPY_TIMING`` takes ``1``, ``true``, ``yes`` or ``on`` to switch the printing on, and ``0``, ``false``, ``no``
or ``off`` to switch the printing off. Any other value of ``PIANISTPY_TIMING`` raises :class:`~pianistpy.InputError`
when the package is imported, and the message gives the variable.

The compilation cache keeps JAX's compiled kernels between sessions, in one folder. The folder is
``~/.cache/pianistpy/jax`` by default, or the folder the environment variable ``PIANISTPY_COMPILATION_CACHE``
sets. Setting ``PIANISTPY_COMPILATION_CACHE`` to ``off``, ``0``, ``false`` or ``no`` turns the cache off.
``set_compilation_cache`` returns the folder in use. The cache changes no number.
