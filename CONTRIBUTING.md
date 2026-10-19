# Contributing to PIANISTpy

Bug reports, questions and proposed changes are welcome on the
[issue tracker](https://github.com/barrytchiang/PIANISTpy/issues).

## Reporting a problem

Please include the version (`pianistpy.__version__`), the inputs that reproduce the problem (the profile, the
anisotropy, the truncation and the settings) and the full error message. An error raised by the package ends with
its most recent log messages; please keep them in the report. For a result that came with an accuracy warning,
please include its validation report.

## Setting up

```
git clone https://github.com/barrytchiang/PIANISTpy.git
cd PIANISTpy
python -m pip install -e .
python -m pip install pytest ruff mypy
```

## Before proposing a change

1. Read `STYLE_GUIDE.md`. It covers the names, the docstrings, the units, the numerical rules and the checklist
   that every change is held to.
2. Run the formatter and the linter:

   ```
   python -m ruff format pianistpy
   python -m ruff check pianistpy
   python -m mypy
   ```

3. Run the tests, which include the examples in the docstrings:

   ```
   python -m pytest
   ```

4. Add a test for the change. A new function comes with a test of its result against an independent
   reference: a closed form, a limit, or an identity that the result must satisfy.

## Rules for pull requests

- **One quantity, one function.** Each quantity is computed by exactly one function; a second route to the same
  number replaces the first rather than sitting beside it.
- **No silent changes to results.** A change that moves a result beyond rounding states the size of the move and
  its reason.
- **Runtime options leave results unchanged.** Verbosity, timing and the number of workers never alter what is
  computed.
- **Accuracy is verified, not assumed.** A numerical setting joins a tested tier only after passing that tier's
  checks.
- **LLM-assisted development.** Code written with the help of large language models is permitted and
  encouraged. The contributor must review every change, and the pull request description itself, thoroughly
  before submitting; a pull request that shows it was not reviewed this way is rejected outright.
