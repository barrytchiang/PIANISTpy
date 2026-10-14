Accuracy and session switches
=============================

Accuracy
--------

Every result carries a validation report, a short and fixed summary of the checks it passed, with its tier: a
check within the accuracy goal (a relative error of 1e-6) is silent; above the goal, within the ceiling (1e-4) or
within the hard limit (1e-2), the result is returned with a warning in the package's log and as an
``AccuracyWarning`` (Python's standard filter silences or escalates it), and the flag stays with the result and with
everything computed from it; beyond the hard limit the result is refused.

The numerical settings come in two tested tiers, both held to these checks by the test suite:

.. code-block:: python

   pp.Settings()            # the default tier
   pp.Settings(quick=True)  # a faster tier for exploration

Session switches
----------------

None of these changes a number.

.. code-block:: python

   pp.set_verbosity("info")   # "quiet" (the default), "info" or "debug"
   pp.set_timing(True)        # print the wall time of each stage
   pp.set_workers("half")     # "auto" (the default), "full", "half" or a number of cores
   pp.set_compilation_cache(False)  # the compilation cache is on by default; a path moves it
