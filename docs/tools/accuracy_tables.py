"""The accuracy tables of the manual and the downloadable accuracy files, from the
accuracy grid of each release.

    python docs/tools/accuracy_tables.py import <version> <members.csv> \\
        <grid.csv> [<rerun.csv> ...]

The accuracy grid is a designed set of models, four tables of rows by columns, rerun
unchanged at every release with each settings level; its CSV is the release's accuracy
record. ``import`` writes one release's files into ``docs/_extra/accuracy/<version>/``:
the model list (``members.csv``: the table, row and column of every model and its
expected outcome) and the grid of the settings level the grid file states
(``grid_default.csv`` or ``grid_quick.csv``). Rows of the rerun files replace the grid's
rows of the same model. Every published file is self-contained: it opens with a comment
block that states what the file is, what one row is, the address of the column
definitions on the manual, and one line per column; its columns are the ones a reader
can act on, under names in words (``COLUMNS`` and ``MEMBER_COLUMNS`` are the one list
the files and the manual's section are written from). The ledger's internal
diagnostics (certificates, reading rules, timings) are not published.

At every build ``conf.py`` calls :func:`write`, which renders the newest release's
four tables (``reference/_accuracy_tables.rst``, included by the API reference's
accuracy page), the section on the downloadable files
(``reference/_accuracy_files.rst``, included by the same page) and the list of
releases (``project/_accuracy_releases.rst``, included by the Development tab's page).
A cell is the relative error of the density computed back from the inverted DF
against the model's density, over the grid's radial range: its
maximum, its median and the tier the result reports, with a note where the grid
expects the model unphysical; or the refusal by name; or "excluded" where the model
violates the Cusp Slope-Central Anisotropy theorem (An & Evans 2006b).
Symbols and numbers are written as math in the paper's
notation (MathJax spans), as the accuracy page defines them; tooltips stay plain text.
"""

import csv
import html
import io
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "_extra" / "accuracy"
FILES_URL = "https://pianistpy.com/docs/reference/accuracy#the-downloadable-files"
TIERS = ("silent", "recorded", "warned", "refused")
TIER_WORDS = (
    "silent, every check at most 1e-6; recorded, at most 1e-4; warned, at most 1e-2; "
    "refused, beyond 1e-2"
)
RANGES = (
    ("1e-4..1e4", "1e-4 < R < 1e4"),
    ("1e-4..1e5", "1e-4 < R < 1e5"),
    ("1e-8..1e8", "1e-8 < R < 1e8"),
)
RANGE = RANGES[2][0]
# The range the tables and the releases list read, under its published name.
RANGE_MAX = f"maximum relative error, {RANGES[2][1]}"
RANGE_MEDIAN = f"median relative error, {RANGES[2][1]}"

# The published columns of a grid file: (published name, the ledger's key, the
# definition the file's header and the manual's section state).
COLUMNS = [
    (
        "model",
        "member",
        "the model's key in the grid: its table (T1 the main grid, T2 anisotropy "
        "radius, T3 truncation energy, T4 limiting cases, the sections of the "
        "Accuracy tables page), its profile, its anisotropy model and its truncation",
    ),
    (
        "settings level",
        "settings",
        'the settings level of the run: "default" or "quick" (the two tested levels '
        "of pianistpy.Settings)",
    ),
    (
        "truncation method",
        "method",
        "1 where the distribution function is inverted from the model's own density "
        "(Method I, with or without a truncation); 2 where the model is a Method II "
        "model, the distribution function inverted from the truncated density of a "
        "Method I model",
    ),
    (
        "anisotropy model",
        "family",
        'the velocity anisotropy model: "iso" (isotropic), "constant" (constant '
        'anisotropy beta_c), "OM_I" and "OM_II" (Osipkov-Merritt of Type I and Type '
        'II), "COM_I" and "COM_II" (Cuddeford-Osipkov-Merritt of Type I and Type II)',
    ),
    ("alpha", "alpha", "the transition sharpness of the double power law"),
    ("beta", "beta", "the outer slope of the double power law"),
    ("gamma", "gamma", "the inner slope of the double power law"),
    (
        "beta_c",
        "beta_c",
        "the central velocity anisotropy; 0 for the isotropic and the Osipkov-Merritt "
        "models",
    ),
    (
        "r_a",
        "r_a",
        "the anisotropy radius in scale radii; blank for the isotropic and the "
        "constant-anisotropy models",
    ),
    (
        "k_t",
        "z_t",
        "the truncation energy as a fraction of the central potential; blank for an "
        "untruncated model",
    ),
    (
        "source model",
        "parent",
        "for a Method II model, the Method I model whose truncated density it is "
        "built on (its profile, anisotropy model and truncation energy); blank "
        "otherwise",
    ),
    (
        "expected outcome",
        "expected",
        'what the grid expects of the model: "value" (a result is served), "value: not '
        'physical" (a result is served for a model that is not a physical '
        "equilibrium), "
        "\"value: the parent's DF within 1e-6, read by the package's rule\" (a "
        "Method II model whose distribution function equals its Method I model's), "
        "or a refusal with the error the package raises (its kind, then the opening "
        "words of its message)",
    ),
    (
        "physical",
        "physical",
        "whether the model is a physical equilibrium by the grid's own criteria: True, "
        "False, or blank where the model is refused before the criteria apply; a "
        "result can be served for a model that is not physical",
    ),
    (
        "outcome",
        "outcome",
        'what the package did: "value" (the model was computed and the error of the '
        "density computed back from its distribution function measured); "
        '"refused: the package\'s tier (checks)" (the model was '
        "computed and its validation report is in the tier refused, with the checks "
        'beyond 1e-2 listed); "refusal: <error>: <message>" (the package raised that '
        "error; no model exists)",
    ),
    (
        "outcome as expected",
        "as_expected",
        "True where the outcome is the expected outcome, False where it is not (the "
        "Accuracy tables page marks such a cell)",
    ),
    (
        "tier",
        "tier",
        f"the tier of the model's validation report: {TIER_WORDS}; blank where the "
        "model is refused before a report exists",
    ),
    (
        "tier of the distribution function",
        "df_tier",
        "the tier of the distribution function's own report, read as the model's tier "
        "(the model's tier is at least this one, since a model's report carries its "
        "distribution function's checks)",
    ),
    (
        "lowest energy of the support",
        "p_cut",
        "for a Type II model without a truncation, the energy P(R_a), as a fraction of "
        "the central potential, above which its distribution function is defined; the "
        "errors are read over that support only; blank otherwise",
    ),
]
for _key, _words in RANGES:
    COLUMNS += [
        (
            f"median relative error, {_words}",
            f"{_key} median",
            f"the median over the radii {_words} (R in scale radii) of the relative "
            "error of the density computed back from the distribution function against "
            "the model's density; blank where no result exists",
        ),
        (
            f"maximum relative error, {_words}",
            f"{_key} max",
            f"the maximum of the same relative error over the radii {_words}",
        ),
        (
            f"radius of the maximum, {_words}",
            f"{_key} at",
            f"the radius, in scale radii, at which the maximum over {_words} is taken",
        ),
        (
            f"number of radii, {_words}",
            f"{_key} n",
            f"the number of radii over which the error is read in {_words}",
        ),
        (
            f"radii with a non-finite error, {_words}",
            f"{_key} nonfinite",
            "the number of those radii at which the computed density is not finite; "
            "0 where every radius is finite",
        ),
    ]
COLUMNS += [
    (
        "refusal of the distribution function",
        "compute_df refused",
        "the package's message where it refused to compute the model's distribution "
        "function; blank where it did not",
    ),
    (
        "refusal of the truncation",
        "truncate refused",
        "the package's message where it refused a truncation of the model: for a "
        "truncated model, its own truncation; for an untruncated model, the grid's "
        "probe that asks the package to truncate it (a profile whose total mass "
        "diverges, outer slope at most 3, and a Type II model whose support ends at "
        "the anisotropy's domain edge are refused by design); blank where nothing "
        "was refused",
    ),
    (
        "refusal of the density check",
        "roundtrip refused",
        "the package's message where the density could not be computed back from the "
        "distribution function (a distribution function that is negative somewhere has "
        "no logarithm); blank otherwise",
    ),
]
MEMBER_COLUMNS = [
    ("model", "member", "the model's key, as in the grid files"),
    (
        "table",
        "table",
        "the table of the Accuracy tables page the model is in: 1 the main grid, 2 "
        "anisotropy radius, 3 truncation energy, 4 limiting cases (the page's "
        "sections)",
    ),
    ("row", "row", "the key of the model's row in that table, as the page labels it"),
    (
        "column",
        "column",
        "the key of the model's column in that table, as the page labels it",
    ),
    ("alpha", "alpha", "the transition sharpness of the double power law"),
    ("beta", "beta", "the outer slope of the double power law"),
    ("gamma", "gamma", "the inner slope of the double power law"),
    (
        "anisotropy model",
        "family",
        "the velocity anisotropy model, as in the grid files",
    ),
    ("beta_c", "beta_c", "the central velocity anisotropy"),
    (
        "r_a",
        "r_a",
        "the anisotropy radius in scale radii; blank where the model has none",
    ),
    ("truncation method", "method", "1 or 2, as in the grid files"),
    (
        "k_t",
        "k_t",
        "the truncation energy as a fraction of the central potential; blank for an "
        "untruncated model",
    ),
    (
        "source model",
        "parent",
        "for a Method II model, the Method I model it is built on; blank otherwise",
    ),
    (
        "expected outcome",
        "expected",
        "what the grid expects of the model, as in the grid files",
    ),
]
PUBLISHED = {src: name for name, src, _ in COLUMNS}

# Each table's section: its heading, the heading of the section that holds it (None
# for a top-level section), and one line on what its rows and columns vary.
SECTIONS = {
    "1": (
        "The main grid",
        None,
        "Untruncated models: each density profile (rows), the double power law "
        "at inner slope \x00\\gamma\x01 and outer slope \x00\\beta\x01 with "
        "\x00\\alpha = 1\x01 and the Plummer profile, under each anisotropy model "
        "(columns), with \x00R_{\\rm a} = 10\x01 for the Osipkov-Merritt (OM) and "
        "Cuddeford-Osipkov-Merritt (COM) models.",
    ),
    "2": (
        "Anisotropy radius",
        "Variations of the main grid",
        "The OM and COM models of the main grid (COM at "
        "\x00\\beta_{\\rm c} = -0.25\x01), Type I and Type II, on three profiles "
        "(rows), with the anisotropy radius from \x00R_{\\rm a} = 0.3\x01 to 1000 "
        "(columns).",
    ),
    "3": (
        "Truncation energy",
        "Variations of the main grid",
        "Four profiles as Method I models under six anisotropy models of the main "
        "grid, "
        "each truncated by Method I and by Method II (rows), with the truncation "
        "energy from \x00\\mathcal{K}_{\\rm t} = 0.5\x01 to 0.002 (columns).",
    ),
    "4": (
        "Limiting cases",
        None,
        "Anisotropies and truncations at the edges of the supported range "
        "(columns): the limit of the Cusp Slope-Central Anisotropy theorem "
        "(An & Evans 2006b), the degenerate and steep constant anisotropies, and a "
        "small anisotropy radius inside a deep truncation. The rows include an "
        "outer slope of 2.5, which the package refuses.",
    ),
}
PROFILES = {
    "nfw": "NFW",
    "hern": "Hernquist",
    "cored": "cored \x00(\\gamma = 0, \\beta = 4)\x01",
    "steep": "steep \x00(\\gamma = 1.5, \\beta = 4)\x01",
    "plummer": "Plummer",
}
BETA_C = {"gamma/2": "\\gamma/2", "-0.5+1e-7": "-1/2 + 10^{-7}", "-0.5": "-1/2"}
# Math in a label: TeX between \x00 and \x01, set as a MathJax span by _html.
MATH = re.compile("\x00(.*?)\x01")


def _read(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = [line for line in lines if line.startswith("#")]
    body = (line for line in lines if not line.startswith("#"))
    return head, list(csv.DictReader(body))


def _math(tex: str) -> str:
    """TeX as a MathJax span, as Sphinx writes :math: (escaped for HTML); braced, so
    MathJax does not break it across lines in a narrow cell."""
    tex = html.escape("{" + tex + "}")
    return f'<span class="math notranslate nohighlight">\\({tex}\\)</span>'


def _num(text: str) -> str:
    mant, _, exp = f"{float(text):.1e}".partition("e")
    return _math(f"{mant} \\times 10^{{{int(exp)}}}")


def _words(key: str) -> str:
    """A key of the model list in words, its symbols marked as math:
    ``COMI_b-0.25_Ra10`` is ``COM I (beta_c = -0.25, R_a = 10)``."""
    out = []
    for tok in key.split("_"):
        if tok == "iso":
            out.append("isotropic")
        elif m := re.fullmatch(r"b([+-][\d.]+)", tok):
            out.append(f"\x00\\beta_{{\\rm c}} = {m.group(1).lstrip('+')}\x01")
        elif m := re.fullmatch(r"(C?OM)(I{1,2})", tok):
            out.append(f"{m.group(1)} {m.group(2)}")
        elif m := re.fullmatch(r"Ra([\d.]+)", tok):
            out.append(f"\x00R_{{\\rm a}} = {m.group(1)}\x01")
        elif m := re.fullmatch(r"kt([\d.]+)", tok):
            out.append(f"\x00\\mathcal{{K}}_{{\\rm t}} = {m.group(1)}\x01")
        elif m := re.fullmatch(r"M([12])", tok):
            out.append("Method " + "I" * int(m.group(1)))
        elif m := re.fullmatch(r"bc=(.+)", tok):
            value = BETA_C.get(m.group(1), m.group(1))
            out.append(f"\x00\\beta_{{\\rm c}} = {value}\x01")
        else:
            out.append(PROFILES.get(tok, tok))
    text = " ".join(out)
    # The COM models write their central anisotropy and radius together.
    pair = (
        "(C?OM I{1,2}) \x00\\\\beta_{\\\\rm c} = ([^\x01]*)\x01 "
        "\x00R_{\\\\rm a} = ([^\x01]*)\x01"
    )
    joined = "\\1 \x00(\\\\beta_{\\\\rm c}, R_{\\\\rm a}) = (\\2, \\3)\x01"
    return re.sub(pair, joined, text)


def _row_label(key: str) -> str:
    if m := re.fullmatch(r"g([\d.]+)_b([\d.]+)", key):
        return f"\x00(\\gamma, \\beta) = ({m.group(1)}, {m.group(2)})\x01"
    if key == "plummer":
        # brackets inside the math: the label breaks only before them, never after "("
        return "Plummer \x00(\\alpha = 2, \\beta = 5, \\gamma = 0)\x01"
    profile, _, rest = key.partition("_")
    if profile in PROFILES and rest:
        return f"{PROFILES[profile]} · {_words(rest)}".replace(" Method", " · Method")
    return _words(key)


def _html(label: str) -> str:
    """A label as HTML, its marked TeX set as math."""
    parts = MATH.split(label)
    return "".join(_math(p) if i % 2 else html.escape(p) for i, p in enumerate(parts))


def _breakable(name: str) -> str:
    """A refusal's name with a break allowed inside a long class name."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "<wbr>", html.escape(name))


def _refusal(outcome: str) -> str:
    """The class of the exception an error cell shows; the message key and the full
    message stay in the cell's tooltip."""
    kind, _, _message = outcome.removeprefix("refusal:").strip().partition(":")
    return kind.strip()


def _cusp_excluded(member: dict) -> bool:
    return float(member["gamma"]) < 2.0 * float(member["beta_c"] or 0.0)


def _cell(member: dict, result: dict) -> str:
    outcome = result["outcome"]
    flag = ""
    if result.get("outcome as expected") != "True":
        flag = '<sup class="acc-flag" title="not the outcome the grid expects">†</sup>'
    if outcome.startswith("refusal"):
        return (
            f'<td class="acc-refusal" title="{html.escape(outcome)}">'
            '<span class="acc-word">error</span>'
            f'<span class="acc-name">{_breakable(_refusal(outcome))}</span>{flag}</td>'
        )
    if _cusp_excluded(member):
        return (
            '<td class="acc-none" title="γ &lt; 2β_c: violates the Cusp '
            'Slope-Central Anisotropy theorem (An &amp; Evans 2006b)">'
            f'<span class="acc-word">excluded</span>{flag}</td>'
        )
    # A value is shown with what the grid expects of it. An untruncated Type II model
    # has its support above P(R_a): the density's error is no measure there, so
    # the cell gives the tier the package reports over the support, and the note.
    note = ""
    mx, md = result.get(RANGE_MAX, ""), result.get(RANGE_MEDIAN, "")
    support = result.get("lowest energy of the support")
    if "not physical" in member.get("expected", ""):
        words = "not physical"
        if support:
            words = "support above \x00P(R_{\\rm a})\x01"
        note = f'<span class="acc-note">{_html(words)}</span>'
        if support:
            mx = ""
    tier = result.get("tier", "")
    # The most frequent tier, silent, is the cell without a tier word (the legend
    # says so); the cell keeps its class and its shade.
    word = "" if tier == "silent" else f'<span class="acc-tier">{tier}</span>'
    nums = ""
    if mx:
        nums = (
            f'<span class="acc-max">{_num(mx)}</span>'
            f'<span class="acc-med">{_num(md)}</span>'
        )
    return f'<td class="acc-{tier}">{nums}{word}{note}{flag}</td>'


def _lead_html(lead: str) -> str:
    """A table's lead as HTML, the theorem's citation linked to the paper (rule 15b)."""
    return _html(lead).replace(
        "(An &amp; Evans 2006b)",
        '(<a href="https://doi.org/10.1086/501040">An &amp; Evans 2006b</a>)',
    )


def _table(number: str, members: list[dict], grid: dict) -> str:
    cells = [m for m in members if m["table"] == number]
    rows = list(dict.fromkeys(m["row"] for m in cells))
    cols = list(dict.fromkeys(m["column"] for m in cells))
    at = {(m["row"], m["column"]): m for m in cells}
    lead = SECTIONS[number][2]
    out = [
        f'<p class="acc-lead">{_lead_html(lead)}</p>',
        '<div class="acc-wrap"><table class="acc"><thead><tr><th></th>',
    ]
    out += [f'<th scope="col">{_html(_words(c))}</th>' for c in cols]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append(f'<tr><th scope="row">{_html(_row_label(r))}</th>')
        for c in cols:
            member = at.get((r, c))
            if member is None:
                out.append('<td class="acc-empty"></td>')
            else:
                out.append(_cell(member, grid[member["model"]]))
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _meta(head: list[str]) -> dict:
    if not head:
        return {}
    found = dict(
        re.findall(r"(package|settings|date|platform|run) ([^,.\n]+)", head[0])
    )
    found.setdefault("date", found.get("run", ""))
    return found


def _versions() -> list[str]:
    if not DATA.is_dir():
        return []
    found = [p.name for p in DATA.iterdir() if (p / "grid_default.csv").is_file()]
    return sorted(found, key=lambda v: tuple(int(p) for p in re.findall(r"\d+", v)))


def _indent(block: str) -> str:
    return "\n".join("   " + line if line else "" for line in block.splitlines())


def _download(version: str, name: str, text: str) -> str:
    return f":download:`{text} <../_extra/accuracy/{version}/{name}>`"


def _rst(words: str) -> str:
    """A definition as the page writes it: numbers and symbols as math (the files,
    plain text, keep them as words)."""
    words = re.sub(
        r"1e(-?\d+) < R < 1e(-?\d+)",
        lambda m: f":math:`10^{{{int(m.group(1))}}} < R < 10^{{{int(m.group(2))}}}`",
        words,
    )
    words = re.sub(
        r"\b1e(-?\d+)\b", lambda m: f":math:`10^{{{int(m.group(1))}}}`", words
    )
    words = words.replace("(R in scale radii)", "(:math:`R` in scale radii)")
    words = words.replace("beta_c", ":math:`\\beta_{\\rm c}`")
    return words.replace("P(R_a)", ":math:`P(R_{\\rm a})`")


def _files_section(version: str) -> str:
    """The section on the downloadable files, from the one list the files are written
    from: what each file is, what one row is, and every column with its definition."""
    lines = [
        f"The grid of ver. {version} is published as three files, each of which opens "
        "with a comment block that repeats the definitions below (``#`` lines, which "
        "``pandas.read_csv(path, comment='#')`` skips):",
        "",
        f"- {_download(version, 'grid_default.csv', 'grid_default.csv')}: the grid "
        "with the default settings level. One row is one model of the grid.",
        f"- {_download(version, 'grid_quick.csv', 'grid_quick.csv')}: the same grid "
        "with the quick settings level, in the same columns.",
        f"- {_download(version, 'members.csv', 'members.csv')}: the list of the "
        "grid's models with the place of each in the tables above and the "
        "outcome the grid expects of it. One row is one model.",
        "",
        "The columns of the grid files, in their order:",
        "",
    ]
    lines += [f"``{name}``" + "\n   " + _rst(words) for name, _, words in COLUMNS]
    lines += ["", "The columns of the model list, in their order:", ""]
    lines += [
        f"``{name}``" + "\n   " + _rst(words) for name, _, words in MEMBER_COLUMNS
    ]
    return "\n".join(lines) + "\n"


RELEASE_HEADS = [
    "Release",
    "Silent",
    "Recorded",
    "Warned",
    "Refused (above :math:`10^{-2}`)",
    "Refused with an error",
    "Excluded (Cusp Slope-Central Anisotropy theorem)",
    "Files",
]


def write(docs: str) -> None:
    """Render the newest release's four tables, the section on its downloadable files
    and the list of releases."""
    docs_dir = Path(docs)
    tables_rst = docs_dir / "reference" / "_accuracy_tables.rst"
    files_rst = docs_dir / "reference" / "_accuracy_files.rst"
    releases_rst = docs_dir / "project" / "_accuracy_releases.rst"
    versions = _versions()
    if not versions:
        for path in (tables_rst, files_rst, releases_rst):
            path.write_text(
                "No release's accuracy grid is in this tree yet.\n",
                encoding="utf-8",
                newline="\n",
            )
        return
    newest = versions[-1]
    head, rows = _read(DATA / newest / "grid_default.csv")
    grid = {r["model"]: r for r in rows}
    _, members = _read(DATA / newest / "members.csv")
    # One rst section per table, so that the page's contents show which table is the
    # main grid and which tables vary it; the tables themselves are raw HTML.
    parts = [
        f"The grid of ver. {newest} with the default settings level; the files that "
        "hold it, with the quick level's grid, are listed under "
        "`The downloadable files`_ below.\n"
    ]
    opened = None
    for t in sorted({m["table"] for m in members}):
        heading, parent, _ = SECTIONS[t]
        if parent is not None and parent != opened:
            parts.append(f"{parent}\n{'-' * len(parent)}\n")
            opened = parent
        mark = "-" if parent is None else "~"
        parts.append(f"{heading}\n{mark * len(heading)}\n")
        parts.append(".. raw:: html\n\n" + _indent(_table(t, members, grid)) + "\n")
    tables_rst.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    files_rst.write_text(_files_section(newest), encoding="utf-8", newline="\n")
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + RELEASE_HEADS[0]]
    lines += [f"     - {h}" for h in RELEASE_HEADS[1:]]
    for v in reversed(versions):
        _, rs = _read(DATA / v / "grid_default.csv")
        mem = {m["model"]: m for m in _read(DATA / v / "members.csv")[1]}
        # As the tables show them: an error first, then the theorem's exclusion, then
        # a value.
        refusals = [r for r in rs if r["outcome"].startswith("refusal")]
        excluded = [
            r for r in rs if r not in refusals and _cusp_excluded(mem[r["model"]])
        ]
        values = [r for r in rs if r not in excluded and r not in refusals]
        files = [_download(v, "grid_default.csv", "default")]
        if (DATA / v / "grid_quick.csv").is_file():
            files.append(_download(v, "grid_quick.csv", "quick"))
        files.append(_download(v, "members.csv", "models"))
        lines += [f"   * - Ver. {v}"]
        lines += [
            f"     - {sum(1 for r in values if r.get('tier') == t)}" for t in TIERS
        ]
        lines += [
            f"     - {len(refusals)}",
            f"     - {len(excluded)}",
            f"     - {', '.join(files)}",
        ]
    releases_rst.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _source_row(row: dict) -> dict:
    """A grid row under the ledger's keys, whether it comes from the ledger (the
    settings level under ``tier_requested``) or from a published file (the columns
    under their published names)."""
    back = {name: src for name, src, _ in COLUMNS}
    out = {}
    for k, v in row.items():
        if k == "tier_requested":
            out["settings"] = v
        else:
            out[back.get(k, k)] = v
    return out


def _header(kind: str, meta: dict, columns: list[tuple[str, str, str]]) -> list[str]:
    version, level = meta.get("package", ""), meta.get("settings", "")
    if kind == "grid":
        lines = [
            f"# PIANISTpy accuracy grid, ver. {version}, settings level {level}, "
            f"platform {meta.get('platform', '')}, run {meta.get('date', '')}.",
            "# One row is one model of the grid: the model's parameters, what the grid "
            "expects of it, what the package did, the tier of the result's validation "
            "report, and the relative error of the density computed back from the "
            "distribution function against the model's density, over three ranges of "
            "radius R (in scale radii): its median, its maximum and where the maximum "
            "lies. Blank cells mean not applicable.",
        ]
    else:
        lines = [
            f"# PIANISTpy accuracy grid, ver. {version}: the list of the grid's "
            "models.",
            "# One row is one model: its place in the tables of the Accuracy "
            "tables page (table, row, column), its parameters, and the outcome the "
            "grid expects of it.",
        ]
    lines.append(f"# Columns: {FILES_URL}")
    lines += [f"# {name}: {words}" for name, _, words in columns]
    return lines


def import_grid(version: str, members: str, grid: str, *reruns: str) -> int:
    """Write one release's model list and one settings level's grid into the manual,
    in the published columns, each file with its header block."""
    head, rows = _read(Path(grid))
    meta = _meta(head)
    meta["package"] = version
    level = meta.get("settings", "")
    if level not in ("default", "quick"):
        print(f"the grid's settings level is {level!r}, not 'default' or 'quick'")
        return 1
    order = [r["member"] for r in (_source_row(r) for r in rows)]
    by_member = {r["member"]: r for r in (_source_row(r) for r in rows)}
    for path in reruns:
        for raw in _read(Path(path))[1]:
            r = _source_row(raw)
            by_member[r["member"]] = r
    missing = [src for _, src, _ in COLUMNS if src not in by_member[order[0]]]
    if missing:
        print(f"the grid lacks the columns {missing}")
        return 1
    out = DATA / version
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    buf.write("\n".join(_header("grid", meta, COLUMNS)) + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([name for name, _, _ in COLUMNS])
    for m in order:
        w.writerow([by_member[m].get(src, "") for _, src, _ in COLUMNS])
    (out / f"grid_{level}.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="\n"
    )
    _, mem = _read(Path(members))
    back = {name: src for name, src, _ in MEMBER_COLUMNS}
    mem = [{back.get(k, k): v for k, v in m.items()} for m in mem]
    buf = io.StringIO()
    buf.write("\n".join(_header("members", meta, MEMBER_COLUMNS)) + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([name for name, _, _ in MEMBER_COLUMNS])
    for m in mem:
        w.writerow([m.get(src, "") for _, src, _ in MEMBER_COLUMNS])
    (out / "members.csv").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    unmatched = set(order) ^ {m["member"] for m in mem}
    print(f"{version}, {level}: {len(order)} models, {len(unmatched)} unmatched")
    return 0 if not unmatched else 1


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "import":
        sys.exit(import_grid(*sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
