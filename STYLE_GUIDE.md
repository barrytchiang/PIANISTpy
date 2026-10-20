# PIANISTpy style guide

This guide sets how PIANISTpy code is organized, written and documented, for its authors and for every future
contributor, human or LLM-assisted. It is modeled on Astropy.

Compiled 2026-10-04 from two kinds of source, and each rule below comes from one of them:

- **official:** Astropy's published coding, documentation and testing guidelines (linked below);
- **observed:** conventions read in Astropy's own source (`astropy/modeling/physical_models.py`,
  `astropy/cosmology/_src/funcs/optimize.py`) where the published guidelines are silent: the file header, the
  placement of `__all__`, the wording of docstrings, how messages are built.

Read it before writing code, and run the checklist of Section 15 on every touched file before a commit.

**Precedence.** Follow, in this order:

1. this guide, where it states a rule;
2. the [Astropy coding guidelines](https://docs.astropy.org/en/latest/development/codeguide.html), the
   [Astropy documentation guidelines](https://docs.astropy.org/en/latest/development/docguide.html) and the
   [Astropy testing guidelines](https://docs.astropy.org/en/latest/development/testguide.html);
3. [PEP 8](https://peps.python.org/pep-0008/) and the
   [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html).

Section 13 lists every place where PIANISTpy departs from Astropy, with the reason. Section 14 covers code
brought in from the working scripts.

## 1. Package layout

The package is `pianistpy`, all lower case. It is split into subpackages by subject, as Astropy is
(`astropy.coordinates`, `astropy.units`, ...). Each subpackage has the same internal shape:

```
pianistpy/
    __init__.py            # version, public names, nothing else
    profiles/
        __init__.py        # imports the public names of the subpackage, defines __all__
        core.py            # the main classes and functions
        <topic>.py         # further modules, one subject each
        diagnostics.py     # measurements a developer or user calls on purpose (Section 10.4)
        tests/
            __init__.py
            test_core.py
            test_<topic>.py
    inversion/
    ...
    utils/                 # helpers shared by several subpackages
docs/                      # narrative documentation (Sphinx, reStructuredText)
examples/                  # notebooks of the user guide, comparisons with other codes, benchmarks
pyproject.toml
```

- Every source file opens the same way (observed): the module docstring (one sentence saying what the module
  holds), `__all__`, then the imports. There is no license line in the files; the license is the one `LICENSE`
  file at the root of the repository.

```python
"""Asymptotic forms of the distribution function at the center and the outer end."""

__all__ = ["central_form", "tail_exponent"]

import numpy as np

from ..numerics import incomplete_beta
from .core import order_pair
```

- `__init__.py` files contain no significant implementation code. They import the public names of their
  subpackage and define `__all__`.
- A helper used by more than one subpackage lives in `pianistpy.utils` (or `pianistpy.numerics` for quadrature,
  interpolation and special functions). It is never copied.
- Module and subpackage names are short, lower case, without underscores where that stays readable.
- Tests live in the `tests` directory of the subpackage they test, in files named `test_<module>.py`.
- Command-line entry points are thin: they parse arguments and call the library.

## 2. Formatting

- Formatting is done by `ruff format` (the Black code style) and is not argued about: 4 spaces per indentation
  level, no tabs, 88 characters per line, double quotes.
- Imports are sorted by `ruff` (isort rules).
- A list, dictionary or call that is written one item per line ends with a trailing comma.
- A formatter exception (`# fmt: off` ... `# fmt: on`) is allowed only around a table of numbers or an aligned
  mathematical expression whose layout carries meaning.
- One statement per line. No semicolons. No alignment of `=` signs across lines with extra spaces.
- Two blank lines between top-level functions and classes, one between methods, single blank lines between the
  logical steps inside a function (observed).
- No trailing whitespace. Files end with one newline.

```python
# WRONG: aligned assignments, compound statement, single quotes
n_sub   = 12;  k_map = 1.0/(1.0 - n)
label   = 'regular'

# RIGHT
n_sub = 12
k_map = 1.0 / (1.0 - n)
label = "regular"
```

A `pre-commit` configuration runs both tools before each commit.

## 3. Naming

### 3.1 Case and form

| Kind | Style | Example |
|---|---|---|
| package, subpackage, module | lower case | `pianistpy.inversion.quadrature` |
| function, method, variable, argument | lower case with underscores | `central_form`, `beta_c` |
| class, exception, warning | CapWords | `DoublePowerLaw`, `UnphysicalModelError` |
| constant (never changes at run time) | upper case with underscores | `GOLDEN_RATIO` |

- A function is named for what it returns or does (`enclosed_mass`, `solve_truncation`), a boolean for the
  statement it makes (`is_truncated`).
- No dates, initials, version tags or issue numbers in names.
- An acronym is treated as an ordinary word, and takes the case the rule for that kind of name gives it: `Losvd`,
  `LosvdTable`, `DfTable` in class names; `losvd`, `sigma_los`, `compute_df` in functions, arguments and variables;
  `LOSVD_KIND` in constants. One rule, no exception for acronyms, and the words of a name are always told apart
  (`LosvdTable`, not `LOSVDTable`). This departs from PEP 8 and Astropy, which write an acronym in all capitals
  inside a class name (`WCS`, `ICRS`). In prose and docstrings the acronym is written as usual (LOSVD, DF).

### 3.2 Physical quantities: family prefix plus the paper's symbol

**Notation matches the paper and the technical notes.** A quantity that has a symbol there is named
`<family>_<symbol>`: the family says what kind of quantity it is, the rest is the paper's symbol or subscript,
lower case. The paper separates symbols by case, script letters and context; lower-case names cannot, and the
family prefix does it instead.

| Family | Prefix | Names | Paper |
|---|---|---|---|
| density and its parameters | `rho_` | `rho`, `rho_aux`, `rho_alpha`, `rho_beta`, `rho_gamma`, `rho_norm` | $\varrho$, $\varrho_{\rm aux}$, $(\alpha, \beta, \gamma)$, $C$ |
| anisotropy | `beta_` | `beta_c`, `beta_sigma`, `beta_sup` | $\beta_{\rm c}$, $\beta_\sigma$, $\beta_{\rm sup}$ |
| radii | `r_` | `r`, `r_a`, `r_t`, `r_2d`, `r_off`, `r_ape` | $R$, $R_{\rm a}$, $R_{\rm t}$, $R_{\rm 2D}$, ... |
| relative potential | `p_` | `p`, `p_0` | $P$, $P_0$ |
| energy-like variables | `k_`, `q_` | `k`, `k_t`, `q`, `q_min`, `q_0` | $\mathcal{K}$, $\mathcal{K}_{\rm t}$, $\mathcal{Q}_\pm$, ... |
| distribution function | `df_` | `df`, `df_k` | $F$, $F_\mathcal{K}$ |
| mass | `mass_` | `mass`, `mass_t`, `mass_vir` | $\mathcal{M}$, $M_{\rm t}$, $M_{\rm vir}$ |
| order of the inversion | `order_` | `order_m`, `order_n` | $(m, n)$ |
| velocities, dispersions | `v_`, `sigma_` | `v_r`, `v_t`, `v_los`, `v_esc`, `sigma_r`, `sigma_los` | $V_r$, ..., $\sigma_{\rm los}$ |
| probability densities | `pdf_` | `pdf_x`, `pdf_phi` | $p_x$, $p_\varphi$ |

- **Where the paper's letter cannot be used, the table fixes a word.** The normalization $C$ of the density is
  `rho_norm`, because `rho_c` would read as a central density; the concentration $c$ is `concentration`. The
  letter `c` alone is never a name, and as a subscript it means central (`beta_c`).
- **Truncated quantities** (hatted in the paper) take the suffix `_trunc`: `p_trunc`, `df_trunc`, `rho_trunc`.
  The untruncated quantity has the bare name.
- **A bare letter is used only for the head of its family** (`r`, `p`, `k`, `q`, `rho`, `df`). Mass and the order
  pair never appear as `m` or `n`.
- **Names of the table are reserved in the whole package.** `k`, `p`, `q` and `r` are never loop indices or
  scratch variables; indices are `i` and `j`.
- A symbol not in the table is added here before it is used in a second module. A quantity with no symbol in the
  paper or the notes gets a descriptive name in full words (`truncation_energy_min`), never a new letter.
- Every docstring gives the paper's symbol beside the name the first time it describes it.

### 3.3 Derived names

| What | Form | Example |
|---|---|---|
| logarithm of a quantity | `ln_`, `log10_` prefix | `ln_r`, `log10_df` |
| derivative | `d<y>_d<x>`, order as a digit | `drho_dp`, `d2rho_dp2` |
| count | `n_` prefix | `n_nodes`, `n_rows` |
| index | `i_` prefix | `i_split` |
| quadrature nodes and weights | `_nodes`, `_weights` suffix | `k_nodes`, `k_weights` |
| tabulated values on a grid | `_grid` suffix | `r_grid`, `df_grid` |
| boolean | `is_`, `has_` prefix | `is_truncated`, `has_tail` |

### 3.4 The leading underscore

The leading underscore marks what is not part of the public interface, at the level of a module or a class:
private functions, private classes, private module-level names, private attributes and methods, private modules
(`_quadrature.py`).

- Local variables inside a function and function arguments never start with an underscore. Everything inside a
  function is private already.
- A name without an underscore at module level is public, is listed in `__all__`, and has a docstring.

### 3.5 One name, one meaning

- A name means the same quantity everywhere in the package. A name of Section 3.2 is never reused for something
  else, however small the scope.
- **No shadowing.** A local name does not repeat a module-level name, an imported name, a built-in (`sum`, `max`,
  `input`, `type`), or a name of an enclosing function.
- **No rebinding to a different thing.** Within a function a name keeps its meaning and its unit: write
  `ln_r = np.log(r)`, not `r = np.log(r)`. Reassigning a name to a converted input at the top of a function
  (`r = np.asarray(r)`) is the one exception.
- Short generic names (`x`, `t`, `u`, `w`, `i`, `j`) are for scopes of a few lines, such as a substitution
  variable inside one integral, and only where they are not in the table of Section 3.2. They are introduced by
  a comment or the docstring's `Notes` when they stand for a mathematical substitution.
- The linter enforces what it can (redefinition, shadowed built-ins, overwritten loop variables); the table of
  Section 3.2 is checked in review.

## 4. Imports

- Standard library first, then third-party packages, then `pianistpy`, separated by blank lines (ruff sorts them).
- Inside the package, use relative imports: `from .core import DoublePowerLaw`, `from ..numerics import gauss_legendre`.
- Never `from module import *`.
- Fixed aliases: `import numpy as np`, `import jax.numpy as jnp`, `import astropy.units as u`.
- Every module defines `__all__` listing its public names.
- An optional dependency is imported inside the function that needs it, with a plain `import` so that a missing
  package raises `ImportError`.

## 5. Docstrings

Every public module, class, method and function has a docstring in the numpydoc format. Private functions have
one whenever what they do is not obvious from the name and five lines of code.

Sections appear in this order, and only those that apply: summary line, extended summary, `Parameters`,
`Returns` (or `Yields`), `Raises`, `Warns`, `See Also`, `Notes`, `References`, `Examples`.

- The summary is one line, in the imperative ("Return the enclosed mass ..."), ending with a period.
- Each parameter gives its type, its shape if it is an array, and its unit. Internal functions state "internal
  units" (Section 8); functions of the public interface that accept quantities use
  `` `~astropy.units.Quantity` ['length'] ``.
- `Notes` holds the formula the function evaluates and where it comes from. `References` cites it:
  - an equation of the paper by its number and label;
  - the technical notes by number and fixed key, exactly as `Notes Eq. (I.5) [conditions]`,
    `Notes Section II.3.1 [df-form]` or `Notes Table I.1 [critical]`. The key is what stays fixed; a tool checks
    and updates the numbers.
- Formulas in `Notes` use the `.. math::` directive with LaTeX (observed); short expressions in running text use
  double backticks.
- A class docstring documents the constructor's parameters; `__init__` has no docstring of its own (observed).
- `Examples` are short, reproducible and written as doctests, so that the test suite runs them.
- Docstrings describe the code as it is. They contain no history, no dates and no account of earlier versions.

```python
def tail_exponent(rho_beta: float, beta_c: float) -> float:
    r"""Return the exponent of the low-energy tail of a (C)OM Type I DF.

    Parameters
    ----------
    rho_beta : float
        Outer slope :math:`\beta` of the double-power-law density. Must exceed 3.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`.

    Returns
    -------
    a : float
        Exponent of the tail, ``df ~ q**(-a)`` as ``q -> 0``. The tail
        diverges for ``a > 0`` and is integrable for ``a < 1``.

    Notes
    -----
    ``a = 7/2 - beta_c - rho_beta``.

    References
    ----------
    Notes Eq. (I.13) [tail-exponent].

    Examples
    --------
    >>> tail_exponent(4.0, 0.0)
    -0.5
    """
    return 3.5 - beta_c - rho_beta
```

### 5a. Manual pages (docs/)

- Symbols, relations and equations are math, in the initial release paper's notation (its macros; the
  "Names and paper symbols" page maps each name to its symbol): :math:`G = M_0 = r_0 = 1`,
  :math:`\mathcal{K}_{\rm t}`, :math:`R_{\rm a}`, :math:`P(R_{\rm a})`, :math:`\beta_{\rm c}`,
  :math:`\sigma_{\rm r}`, :math:`\gamma \ge 2\beta_{\rm c}`; subscripts that are labels are upright. Names a
  user types (`k_t`, `mass_t`, `pp.truncate`, `beta_c=0.25`) are literals, and a code object's name is shown
  exactly as the code spells it, never as math, in titles and menu labels too ("Radial velocity dispersion
  (sigma_r)"). No symbol in plain text with an
  underscore, no `>=` or `|x - 1|` in prose. The same holds in docstrings (the API pages and `help()` show
  them) and in generated text (the accuracy and settings tables). README is read on GitHub and on PyPI, and
  PyPI renders no `$...$` math: there a symbol is HTML (`<i>M</i><sub>0</sub>`), and every link and image is
  absolute.
- Numbers in prose are math: an accuracy level or a measured error reads :math:`10^{-6}`,
  :math:`6.7 \times 10^{-6}`, never `1e-6`. A value typed into Python stays a literal in Python's form
  (settings' defaults, code, a call's argument).
- A symbol is defined in words at its first use on a page. The density roundtrip is
  :math:`|(\varrho_{\rm inv} - \varrho)/\varrho|`: :math:`\varrho` the input profile,
  :math:`\varrho_{\rm inv}` the density computed back from the inverted distribution function. One symbol,
  one meaning: :math:`\theta` is the parameter vector, `theta` the argument that passes it.
- The release paper is "Chiang+(2026d)", with no space (Barry, 2026-10-19; the letter stays even where no
  other 2026 paper is cited), always as a link to `PAPER_URL` in `docs/conf.py`, the one target. The landing
  page (at its first mention), the citing page and the README's status line give the full form once,
  "Chiang, van den Bosch and Schive (2026d), MNRAS, submitted (henceforth Chiang+(2026d))"; every other
  mention is the short form. Pages write `|paper|`, `|paper-full|` or `|paper-name|` (plain, for headings),
  defined from `PAPER`, `PAPER_FULL` and `PAPER_STATUS` in `docs/conf.py`. Docstrings write the short form
  as plain text, an equation as "Chiang+(2026d), Eq. (B42)". Never "Chiang et al. (2026d)", "Paper I", "C26",
  "C26d" or "Chiang+26d", and "the paper" alone does not stand for it. "Chiang+(2026d)" is a protected string
  for the release paper only (Barry, 2026-10-19): every other paper keeps its usual form ("Chiang et al. 2025",
  "2026a"), and no other citation is rewritten to the "+" form.
- An external package is linked at its first mention on a page (intersphinx where it has an
  inventory, else its home page: Astropy, JAX, NumPy, galpy, AGAMA).
- A page's title is its menu label; a renamed page is renamed in its toctree in the same commit.
- Every number on a page (a cost, a timing, a tier count) is read from a product of the
  version's tip and says what it was measured on; none is copied from a draft.
- A statement about devices, the Python floor or the public set matches README and the
  install page in substance.
- American spelling; one name per thing, as in section 3.5. The device is "the GPU" (or "a GPU"), never "card"
  or "accelerator", in the docs, README, messages and docstrings alike.
- No "you" or "your" in any public text (docs, README, CONTRIBUTING, docstrings, messages, GitHub text): the
  package or the call is the subject ("PIANISTpy builds ...", "the call returns ..."), or the imperative ("Please
  cite ..."); where a person must be named, "user-specified" or "the user(s)".
- The landing page states nothing a later release could make untrue (no per-version stage lists, counts or
  timings); those live on the pages that are checked at each release.

## 6. Type hints

- Every public function, method and class attribute carries type hints, and so does every field of the settings
  object. Private helpers carry them where they help the reader.
- Arrays are annotated with the package's own aliases (`Array` for a JAX or NumPy array, `ArrayLike` for anything
  that converts to one), defined once in `pianistpy.typing`. A hint says what kind of object an argument is; its
  shape, its unit and its meaning stay in the docstring, which remains the specification.
- The numpydoc `Parameters` section still gives the type in words. Where the hint and the docstring disagree,
  that is an error to fix, not a choice.
- Modern syntax: `float | None`, `tuple[Array, Array]`, no `typing.Optional` or `typing.List`. Imports needed
  only for hints go under `if TYPE_CHECKING:`.
- Hints are checked by a static type checker in the same run as the tests. A hint is never satisfied by
  `Any` or silenced with an ignore comment without a reason on the same line.

```python
def tail_exponent(rho_beta: float, beta_c: float) -> float:
    ...


def enclosed_mass(r: ArrayLike, settings: Settings) -> Array:
    ...
```

## 7. Comments

- A comment says why, or states a fact the code cannot show (a derivation step, the reason for an order of
  operations, the source of a constant). It does not restate the line below it.
- Comments are complete sentences, start with a capital letter and stay within the line limit. Block comments
  sit above the code they describe, at its indentation; inline comments are rare and short.
- No history in comments: no dates, initials, bracketed change tags, "was X before", or commented-out code. That
  record is in the version control history and the change log.
- A constant that controls accuracy carries one comment naming what it is derived from and the tier it serves.
- An unfinished point is marked `# TODO: <what and why>` and has an entry in the issue tracker.

## 8. Units

- There are three unit systems, named as follows everywhere (code, docstrings, files, documentation):

  | Name | Definition |
  |---|---|
  | `internal` | $G = \rho_0/C = r_0 = 1$, the natural normalization of Widrow (2000); $P_0 = 1$ for an untruncated profile |
  | `model` | $G = M_0 = r_0 = 1$, as in the paper |
  | `physical` | physical units through `astropy.units` |

- Inside the package every quantity is a plain array in internal units. No function below the output layer
  accepts or returns a `Quantity`.
- **An input that is naturally given in another system says so in its docstring**, in the first sentence of the
  parameter: the target mass of a truncation is in units of $M_0$ (model units, given with the concentration that defines
  $M_0$ and refused without it), the target radius in units of
  $r_0$ (the unit of length of both systems), the truncation energy in internal units.
- **Internal units are the default of every output.** Model and physical units are produced only when the user
  gives the scales they need (a concentration; a mass and a length) and asks for them. One default for every
  kind of result, so that nobody has to remember which file is in which units.
- Conversion to physical units happens in one place, the output layer, through one helper built on
  `astropy.units`. A product file states the unit of every dataset.
- A docstring always states the unit of each argument and return value, by the names above. Factors of the unit
  layer carry the target system and unit in their name (`model_v`, `physical_v_kms`).

## 9. Settings and constants

- Settings that a user may change are fields of one settings object with defaults, passed explicitly to the
  functions that need them. A function does not read a setting from a module-level variable.
- Module-level upper-case names are for true constants only.
- Every accuracy setting states, in its docstring line, what it is derived from. A threshold that was tuned by
  trial has no place in the package; a tolerance follows from the accuracy tiers (goal `1e-6`, ceiling `1e-4`,
  hard limit `1e-2`).
- No numerical literal appears twice. A repeated value is a named constant or a setting.

## 10. Output, warnings and errors

### 10.1 Messages

- No bare `print()` in library code, except in functions whose stated purpose is to print. Messages go through
  the package logger (`from pianistpy import log`), which writes each one as it is emitted, so a user who turns
  the verbosity on watches the stages pass and, if a run fails, sees where.
- Levels: `log.info` for progress, one line per stage; `log.debug` for detail, still per stage, never per row or
  per iteration of a loop. A loop reports once, with counts.
- Every stage announces itself on entry. A short memory of the most recent debug messages is kept even when the
  session is quiet, and is shown when the package raises one of its own errors.
- A message is one line, a complete sentence, and names the quantity with its value. No bracketed tags, banners
  or tables. It is written so that it costs nothing when it is not shown (`log.debug("... %s", value)`, never
  an f-string built in advance).
- No message inside a function that JAX compiles.
- Bulk diagnostic data goes to a file, and only on request; never to the screen.
- **Verbosity never changes what is computed.** A message reports a number the calculation already has; no
  work is done for a message.
- The verbosity is set once for the session (one call, or an environment variable for batch jobs); functions do
  not take a `verbose` argument. A log file is available on request.

### 10.2 Timing

- Every stage of a computation runs inside the package's timing helper, permanently: the checkpoints are part of
  the code, not something added for one optimization and removed after it.
- Timing has its own switch, independent of the verbosity, so that a run can be timed with nothing else turned
  on. The switch controls printing only; the times are always measured.
- The stages are a fixed, short list per computation (five to ten), nested one level at most. Compilation time is
  kept apart from run time, and peak memory is recorded beside them.
- The times of a result are an attribute of the result (`result.timing`) and are stored in the provenance of
  its file with the processor model. They are not part of the identity or of the data digest.

### 10.3 Checks

- **The essential checks always run and cannot be switched off:** those that decide a tier or a refusal. They
  are kept to the minimum that the guarantee of a result needs, and their cost is part of the cost of the result.
- **No other check runs with a result.** A check beyond the essential ones belongs to the test suite (Section
  10.4). A result records which checks ran.
- The report of a result is compact and fixed: a documented set of keys per kind of result, one record per key
  (the measured error, its tier, where), nothing appended over time. A derived result holds its own checks and
  the worst tier of its parents.
- Verbosity, timing and the worker count (Section 11.3) are independent switches of the session, and none is
  part of the identity.

### 10.4 Where verification code lives

| What it is | Where it goes |
|---|---|
| A check of the main path that decides a tier or a refusal | the validation of the subpackage, always run |
| A deeper, independent measurement of a result, called on purpose | `diagnostics.py` of the subpackage |
| An invariant of the code | a test, in `tests/` of the subpackage |
| Notebooks of the user guide, comparisons with other codes, performance benchmarks | `examples/` at the root |

- A function of `diagnostics.py` returns numbers (it does not only print), has a numpydoc docstring, and has its
  own test. The opt-in checks of Section 10.3 call these functions, so that each measurement exists once.
- No diagnostic branch inside a function of the main path, and no diagnostic that is switched by the verbosity.

### 10.5 Warnings and errors

- Errors are raised with a built-in exception or a subclass of `PianistpyError`. Never raise or catch a bare
  `Exception`. A model that the code refuses is refused by a named exception whose message states the condition
  that failed and the values involved.
- Warnings use `warnings.warn(message, PianistpyWarning)` or a subclass. A result between the goal and the
  ceiling warns; a result beyond the hard limit raises unless the caller overrides the check explicitly.
- What the user must act on is a warning or an exception, never only a log line.
- Messages are complete sentences and name the quantities with their values, built with f-strings (observed).

```python
# WRONG
print("WARNING: bad model!!")
raise Exception("failed")

# RIGHT
raise UnphysicalModelError(
    f"The central DF diverges to negative values for delta = {delta:.3f} "
    f"between 1 and m + n = {order:.2f}."
)
```

## 11. Numerical code

### 11.1 Stages

- A computation is a pipeline of stages. A stage is a pure function: it takes its inputs and the settings and
  returns a frozen stage result. The public functions (`compute_df`, `truncate`, ...) compose stages and do nothing
  else.
- A stage has a contract: its signature, its docstring (inputs, outputs, units, the equations it evaluates), and a
  contract test (shapes, units, invariants). A stage is filled against its contract and can be tested alone.
- No state outside the signatures: no module-level variable is read or written by a stage, and nothing is passed
  between stages except their results.
- The timing checkpoints and the checks of Section 10 attach to stages.

### 11.2 What runs in JAX

- **Something moves to JAX only if it gains from it.** Work that is small, heterogeneous, or needs arbitrary
  precision or symbolic algebra stays in NumPy, SciPy or mpmath on the host.
- It does move in three cases: where gradients are wanted (the observables built from precomputed members, which
  are linear in the superposition weights); where it is a bottleneck; and where a small piece sits between two
  JAX blocks and merging the three is simpler to run and to maintain.
- A bottleneck gains from JAX only as batched array work. It is made batchable first, then moved.
- A gradient needs every step it crosses to be in JAX. A stage that is to be differentiable says so in its
  contract.
- **Host code and kernels are kept apart.** Host code reads settings, builds grids, chooses the form of a
  calculation by its analytic criterion, logs and checks. A kernel (a function JAX compiles) does array arithmetic
  on whole batches and nothing else: no branching in Python on computed values, no logging, no file access, no
  global state.
- A function needed both in a kernel and on the host is written once.
- Double precision everywhere. The package switches JAX to 64 bits when it is imported and refuses to run if it
  finds it switched off.
- The backend is the GPU where the installed JAX supports one, and the CPU otherwise; a documented switch forces
  either. Every result records the backend.

### 11.3 Workers

- One mechanism for parallel work, never nested (no process pool on top of JAX's own threads).
- One switch for the session, `set_workers`, also settable by an environment variable. No function takes a
  worker count as an argument; the stages that have batched work read the switch through the one helper that
  runs it, and the others ignore it.

  | Value | Meaning |
  |---|---|
  | `"auto"` (default) | inside a scheduler's job, every core of the allocation; anywhere else, half the cores |
  | `"full"` | every core the process is allowed to use |
  | `"half"` | half of them |
  | an integer | exactly that many |

- The count is taken from what the process is allowed to use, not from what the machine has.
- **Parallelism never changes a number.** A result is the same, to the bit, for any worker count.
- The worker count is recorded in the provenance and is not part of the identity.

### 11.4 Randomness

- Random numbers come from a counter-based generator keyed by the seed, the component and the particle's index.
  They are bit-identical on every machine and backend, for any batch size and worker count; the first n particles
  of a larger draw are the same particles.
- The transformation from random numbers to particles runs in one host code path whatever the backend. Results are
  bit-identical as far as the code controls; differences in the last bit between processors' mathematical
  libraries are accepted.
- A seed that the user did not give is drawn and recorded in the result.

### 11.5 Non-finite values

- Tables keep a fixed size for a given set of settings, so that the results of a sweep stack and kernels can
  batch over models.
- **NaN has one meaning: no value at this entry.** It marks entries outside the support of a table, and the
  bounds of the support are recorded beside the table, so that no consumer has to test for NaN to find them.
- **Minus infinity has one meaning: the logarithm of an exact zero.**
- No plus infinity and no other marker values in a stored array. A missing scalar is an absent field.
- A NaN inside the recorded support is a fault: every stage checks its result and raises by name. This is one
  of the essential checks of Section 10.3.

### 11.6 General

- The form of a calculation is chosen from the model parameters by an analytic criterion, never by inspecting
  the numbers it produced.
- One implementation of each formula. If two modules need it, it moves to a shared module.
- **One result, one path.** The public interface offers exactly one way to obtain each result, and every model
  goes through the same general code.
  - No class or function per special case. A named profile (`NFW`, `Hernquist`) is a set of parameter values of
    the one general profile, never a separate implementation.
  - A closed form that exists for a special case is used in two places only: in the tests, as the reference, and
    inside the general path as a branch chosen by an analytic criterion, invisible to the user.
  - An alternative algorithm for the same quantity may exist only behind the same function, selected by an
    analytic criterion or by the accuracy settings, and a test holds the two to the accuracy goal against each
    other.
  - Methods of objects are thin calls to the functional layer, so the two cannot differ.
  - Accuracy modes are the same code at different settings.
- Classes expose attributes directly or through properties. `get_`/`set_` methods are used only where the
  operation is expensive.
- `super()` is used for every call to a parent class. Multiple inheritance is avoided.

## 12. Tests

- Tests use `pytest`. Each public function has at least one test.
- A test compares against an analytic result: a closed form, an exact identity, or a round trip of the input.
  Another code is a sanity check, not a reference.
- Tolerances in tests are the accuracy tiers or tighter, with `numpy.testing.assert_allclose` and an explicit
  `rtol`.
- Cases are listed with `pytest.mark.parametrize`. Tests do not depend on each other or on the order they run in.
- A change that is meant to leave results unchanged is checked to leave the stored reference products unchanged
  bit for bit. A change that moves a number is a separate commit that says so.

## 13. Where PIANISTpy departs from Astropy

| Topic | Astropy | PIANISTpy | Reason |
|---|---|---|---|
| Units inside the library | `Quantity` throughout | plain arrays in internal units, `Quantity` only at the output layer | compiled JAX code cannot carry `Quantity` objects |
| Configuration | `astropy.config` items at module level | one settings object passed explicitly | settings must reach compiled functions as arguments and be recorded in each product |
| Array library | NumPy | JAX for compiled paths, NumPy elsewhere | performance and automatic differentiation |
| Type hints | not required by the guidelines; used in newer modules | required on the public interface | the interface is new, so it starts typed |
| References in docstrings | literature | literature, the paper's equations, and the technical notes by number and key | the notes are the specification of the numerical methods |
| License line at the top of each file | required | none; one `LICENSE` file at the root (BSD 3-clause) | the authors' choice |

## 14. Bringing code in from the working scripts

The package is assembled from long working scripts. When a function moves into the package it is brought up to
this guide in the same commit, and the commit changes no number.

- Bracketed change tags, dates and initials in comments are removed; what they explain, if still true, becomes a
  plain comment or a `Notes` entry.
- Module-level settings read inside the function become arguments or fields of the settings object.
- `print` calls become `log.info` or `log.debug`, or go.
- Names are brought to Section 3 (the working scripts' underscore on local variables goes); the old name is not
  kept as an alias inside the package.
- Diagnostic branches that no setting reaches are deleted, not carried.
- The docstring is written in full (Section 5), with its citation of the technical notes.
- The stored reference products are reproduced bit for bit before and after the move.

## 15. Checklist before a commit

1. `ruff format` and `ruff check` pass on every touched file; no trailing whitespace.
2. Every new or changed public name has type hints and a numpydoc docstring with units, and is in `__all__`.
3. Every formula cites the paper or the technical notes; the notes' reference checker passes.
4. No `print`, no `verbose` argument, no diagnostic branch in a main-path function, no bare `Exception`, no setting read from a module-level variable, no numerical literal repeated.
5. No dates, tags, initials or commented-out code in comments or docstrings.
6. Names follow Section 3: family prefixes, no shadowing, no rebinding; new symbols are in the table of Section 3.2.
7. Tests exist for each new public function and compare against an analytic result; the test suite passes.
8. A commit meant to leave results unchanged leaves the reference products unchanged bit for bit; a commit that
   moves a number says so in its message and does nothing else.

## 16. Open points

- The copyright line of the `LICENSE` file (the license is BSD 3-clause).
- The symbol table of Section 3.2 is completed as modules are assembled.
