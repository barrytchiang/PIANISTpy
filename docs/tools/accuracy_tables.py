"""The accuracy tables of the manual, from the accuracy grid of each release.

    python docs/tools/accuracy_tables.py import <version> <members.csv> \\
        <grid.csv> [<rerun.csv> ...]

The accuracy grid is a designed set of members, four tables of rows by columns,
rerun unchanged at every release; its CSV is the release's accuracy record.
``import`` copies one release's grid into ``docs/_extra/accuracy/<version>/``: the
member list (``members.csv``: the table, row and column of every member and its
expected outcome) and the grid at the default tier (``grid_default.csv``). Rows of
the rerun files replace the grid's rows of the same member. The header keeps what
the grid measured and drops where it ran.

At every build ``conf.py`` calls :func:`write`, which renders the newest release's
four tables (``reference/_accuracy_tables.rst``, included by the API reference's
accuracy page) and the list of releases (``project/_accuracy_releases.rst``,
included by the Development tab's page). A cell is the relative roundtrip error of
the density computed back from the inverted DF, over the grid's radial range: its
maximum, its median and the tier the result reports, with a note where the grid
expects the model unphysical; or the refusal by name; or "excluded" where the model
breaks the cusp bound. Symbols and numbers are written as math in the paper's
notation (MathJax spans), as the accuracy page defines them; tooltips stay plain text.
"""

import csv
import html
import io
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "_extra" / "accuracy"
RANGE = "1e-4..1e5"
TIERS = ("goal", "ceiling", "warning", "refused")

TITLES = {
    "1": (
        "Table 1. Profiles by anisotropy",
        "The double power law at inner slope \x00\\gamma\x01 and outer slope "
        "\x00\\beta\x01 \x00(\\alpha = 1)\x01, and the Plummer profile, under ten "
        "anisotropy models; untruncated.",
    ),
    "2": (
        "Table 2. The anisotropy radius",
        "Osipkov-Merritt (OM) and Cuddeford-Osipkov-Merritt (COM, "
        "\x00\\beta_{\\rm c} = -0.25\x01) models of Type I and Type II on three "
        "profiles, from \x00R_{\\rm a} = 0.3\x01 to 1000.",
    ),
    "3": (
        "Table 3. Truncations",
        "Method I and Method II truncations at energy \x00\\mathcal{K}_{\\rm t}\x01 on "
        "four parents under six anisotropy models.",
    ),
    "4": (
        "Table 4. Edges",
        "Profiles at the edges of the conditions: the cusp bound, the degenerate and "
        "steep constant anisotropies, a small anisotropy radius inside a deep "
        "truncation, and the outer slope 2.5 profile, which is refused.",
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
# A refusal's name: the capitals the package prints before a colon or a semicolon.
CODE = re.compile(r"(?:^|[:;(]\s*)([A-Z][A-Z0-9]+(?: [A-Z0-9]+)*)(?=\s*[:;.)]|$)")


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
    """A key of the member list in words, its symbols marked as math:
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
    # The COM models name their central anisotropy and radius together.
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
        # the brackets inside the math: the label can break only before them, never after "("
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
    kind, _, message = outcome.removeprefix("refusal:").strip().partition(":")
    codes = list(dict.fromkeys(CODE.findall(message.strip())))
    return " · ".join(codes) if codes else kind.strip()


def _cusp_excluded(member: dict) -> bool:
    return float(member["gamma"]) < 2.0 * float(member["beta_c"] or 0.0)


def _cell(member: dict, result: dict) -> str:
    outcome = result["outcome"]
    flag = ""
    if result.get("as_expected") != "True":
        flag = '<sup class="acc-flag" title="not the outcome the grid expects">†</sup>'
    if outcome.startswith("refusal"):
        return (
            f'<td class="acc-refusal" title="{html.escape(outcome)}">'
            '<span class="acc-word">refused</span>'
            f'<span class="acc-name">{_breakable(_refusal(outcome))}</span>{flag}</td>'
        )
    if _cusp_excluded(member):
        return (
            '<td class="acc-none" title="γ &lt; 2β_c: below the cusp bound">'
            f'<span class="acc-word">excluded</span>{flag}</td>'
        )
    # A value is shown with what the grid expects of it. An untruncated Type II model
    # has its support above P(R_a): the density's roundtrip is no measure there, so
    # the cell gives the tier the package reports over the support, and the note.
    note = ""
    mx, md = result.get(f"{RANGE} max", ""), result.get(f"{RANGE} median", "")
    if "not physical" in member.get("expected", ""):
        words = "not physical"
        if result.get("p_cut"):
            words = "support above \x00P(R_{\\rm a})\x01"
        note = f'<span class="acc-note">{_html(words)}</span>'
        if result.get("p_cut"):
            mx = ""
    tier = result.get("tier", "")
    nums = ""
    if mx:
        nums = (
            f'<span class="acc-max">{_num(mx)}</span>'
            f'<span class="acc-med">{_num(md)}</span>'
        )
    return (
        f'<td class="acc-{tier}">{nums}<span class="acc-tier">{tier}</span>'
        f"{note}{flag}</td>"
    )


def _table(number: str, members: list[dict], grid: dict) -> str:
    cells = [m for m in members if m["table"] == number]
    rows = list(dict.fromkeys(m["row"] for m in cells))
    cols = list(dict.fromkeys(m["column"] for m in cells))
    at = {(m["row"], m["column"]): m for m in cells}
    title, lead = TITLES[number]
    out = [
        f'<h3 class="acc-title">{title}</h3>',
        f'<p class="acc-lead">{_html(lead)}</p>',
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
                out.append(_cell(member, grid[member["member"]]))
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _meta(head: list[str]) -> dict:
    if not head:
        return {}
    return dict(
        re.findall(r"(package|commit|numerics|tier|date|platform) ([^,]+)", head[0])
    )


def _versions() -> list[str]:
    if not DATA.is_dir():
        return []
    found = [p.name for p in DATA.iterdir() if (p / "grid_default.csv").is_file()]
    return sorted(found, key=lambda v: tuple(int(p) for p in re.findall(r"\d+", v)))


def _indent(block: str) -> str:
    return "\n".join("   " + line if line else "" for line in block.splitlines())


def _download(version: str, name: str, text: str) -> str:
    return f":download:`{text} <../_extra/accuracy/{version}/{name}>`"


def write(docs: str) -> None:
    """Render the newest release's four tables and the list of releases."""
    docs_dir = Path(docs)
    tables_rst = docs_dir / "reference" / "_accuracy_tables.rst"
    releases_rst = docs_dir / "project" / "_accuracy_releases.rst"
    versions = _versions()
    if not versions:
        for path in (tables_rst, releases_rst):
            path.write_text(
                "No release's accuracy grid is in this tree yet.\n",
                encoding="utf-8",
                newline="\n",
            )
        return
    newest = versions[-1]
    head, rows = _read(DATA / newest / "grid_default.csv")
    grid = {r["member"]: r for r in rows}
    _, members = _read(DATA / newest / "members.csv")
    meta = _meta(head)
    blocks = [_table(t, members, grid) for t in sorted({m["table"] for m in members})]
    # Where the grid ran: the commit, and the commit whose numerical code it carries.
    run = f", run at {meta['commit'][:8]}" if "commit" in meta else ""
    if meta.get("numerics", meta.get("commit")) != meta.get("commit"):
        run += f", whose numerical code is {meta['numerics'][:8]}'s"
    tables_rst.write_text(
        f"The grid of ver. {newest}{run} ({meta.get('platform', 'cpu')}), "
        "the default tier: "
        f"{_download(newest, 'grid_default.csv', 'the full grid as CSV')} and "
        f"{_download(newest, 'members.csv', 'its member list')}.\n\n"
        ".. raw:: html\n\n" + _indent("\n\n".join(blocks)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    head_cells = ["Release", "Goal", "Ceiling", "Warning", "Refused tier"]
    head_cells += ["Refusals", "Excluded", "The grid"]
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + head_cells[0]]
    lines += [f"     - {h}" for h in head_cells[1:]]
    for v in reversed(versions):
        _, rs = _read(DATA / v / "grid_default.csv")
        mem = {m["member"]: m for m in _read(DATA / v / "members.csv")[1]}
        # As the tables show them: a refusal first, then the cusp bound, then a value.
        refusals = [r for r in rs if r["outcome"].startswith("refusal")]
        excluded = [
            r for r in rs if r not in refusals and _cusp_excluded(mem[r["member"]])
        ]
        values = [r for r in rs if r not in excluded and r not in refusals]
        lines += [f"   * - Ver. {v}"]
        lines += [
            f"     - {sum(1 for r in values if r.get('tier') == t)}" for t in TIERS
        ]
        lines += [
            f"     - {len(refusals)}",
            f"     - {len(excluded)}",
            f"     - {_download(v, 'grid_default.csv', 'CSV')}, "
            f"{_download(v, 'members.csv', 'members')}",
        ]
    releases_rst.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def import_grid(version: str, members: str, grid: str, *reruns: str) -> int:
    """Copy one release's member list and default-tier grid into the manual.

    ``--numerics <commit>`` among the reruns names the commit whose numerical code
    the grid's commit carries, where the two differ; the page states both.
    """
    reruns_l = list(reruns)
    numerics = ""
    if "--numerics" in reruns_l:
        i = reruns_l.index("--numerics")
        numerics = reruns_l[i + 1]
        del reruns_l[i : i + 2]
    reruns = tuple(reruns_l)
    head, rows = _read(Path(grid))
    meta = _meta(head)
    if meta.get("tier") != "default":
        print(f"the grid's tier is {meta.get('tier')!r}, not 'default'")
        return 1
    order = [r["member"] for r in rows]
    by_member = {r["member"]: r for r in rows}
    columns = list(rows[0])
    for path in reruns:
        for r in _read(Path(path))[1]:
            by_member[r["member"]] = r
            columns += [c for c in r if c not in columns]
    out = DATA / version
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    date = meta.get("date", "")
    where = f"commit {meta['commit']}, " if "commit" in meta else ""
    if numerics:
        where += f"numerics {numerics}, "
    buf.write(
        f"# PIANISTpy accuracy grid: package {version}, {where}"
        f"tier default, date {date}, platform {meta.get('platform', '')}\n"
    )
    buf.writelines(line + "\n" for line in head[1:])
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(by_member[m] for m in order)
    (out / "grid_default.csv").write_text(
        buf.getvalue(), encoding="utf-8", newline="\n"
    )
    _, mem = _read(Path(members))
    buf = io.StringIO()
    keep = [c for c in mem[0] if c != "basis"]
    w = csv.DictWriter(buf, fieldnames=keep, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(mem)
    (out / "members.csv").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    unmatched = set(order) ^ {m["member"] for m in mem}
    print(f"{version}: {len(order)} cells, {len(unmatched)} unmatched")
    return 0 if not unmatched else 1


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "import":
        sys.exit(import_grid(*sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
