Contributing
============

Please use the `issue tracker <https://github.com/barrytchiang/PIANISTpy/issues>`_ of the GitHub repository for
bug reports, questions and proposed changes. A bug report should include the version (``pianistpy.__version__``),
the inputs that reproduce the problem, and the full error message with the last log lines before the error. A
report of an accuracy warning also includes the result's validation report.

Before proposing a change
-------------------------

1. Read the `style guide <https://github.com/barrytchiang/PIANISTpy/blob/main/STYLE_GUIDE.md>`_. The style guide covers the names, the docstrings, the units, the
   numerical rules and the checklist that every change is held to.
2. Run the formatter, the linter and the type checker with ``python -m ruff format pianistpy``,
   ``python -m ruff check pianistpy`` and ``python -m mypy``.
3. Add a test for the change. A new function comes with a test of its result against an independent
   reference: a closed form, a limit, or an identity that the result must satisfy.

Rules for pull requests
-----------------------

- **Validate before filing.** A contributor should validate the changes with the test suite and the accuracy
  checks, under both the default and the quick settings, and, for a change to a computation that runs on the GPU,
  separately in the CPU-only mode and with the GPU enabled. The runtime options (verbosity, timing, number of
  cores, compilation cache) must leave every result unchanged.
- **State every changed result.** If a code edit alters any result of the validation tests beyond rounding, the
  pull request should state explicitly each result that changed and by how much.
- **LLM-assisted development.** Code written by or with the assistance of large language models is permitted. The
  contributor must review every code change, every test added or changed, and the pull request description. A
  pull request whose code, tests and description the contributor has not reviewed may be rejected.
