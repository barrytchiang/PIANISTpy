"""The technical notes as pages of the manual, transcribed from their LaTeX source.

    python docs/tools/notes_pages.py <PIANISTpy_Technical_Notes.tex> [--aux <file.aux>]

Writes ``docs/reference/notes.rst`` (the notes' opening, their version and the list of
sections) and one page per section under ``docs/reference/technical-notes/``. Without
``--aux`` the notes are compiled twice with ``pdflatex`` in a temporary folder, so every
number on the pages is LaTeX's own: the transcription counts sections, equations and
tables itself and refuses to write anything where a count differs from the ``.aux``.

The transcription covers the LaTeX the notes use and nothing else: any other command
or environment stops it with the line it is on, so nothing is dropped silently. The
rules it follows (and how to extend it) are in ``docs/tools/NOTES_TRANSCRIPTION.md``.
"""

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
OUT_DIR = DOCS / "reference" / "technical-notes"
INDEX = DOCS / "reference" / "notes.rst"
PAPER = "|paper|"
NBSP = "\u00a0"

# Text macros of the notes' preamble, as the site writes them.
TEXT_MACROS = {
    "PIANISTpy": "PIANISTpy",
    "JAX": "JAX",
    "galpy": "galpy",
    "agama": "AGAMA",
}
# Commands that carry no content on a web page.
DROP = {
    "maketitle",
    "tableofcontents",
    "noindent",
    "centering",
    "toprule",
    "midrule",
    "bottomrule",
    "par",
}
ENVS = ("equation", "table", "itemize", "enumerate", "list")
HEAD = {"section": "=", "subsection": "-", "subsubsection": "~"}


class TranscriptionError(Exception):
    pass


# ---- reading the source ----------------------------------------------------------


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def braced(text: str, i: int) -> tuple[str, int]:
    """The group that opens at text[i] == '{', and the index after it."""
    if i >= len(text) or text[i] != "{":
        raise TranscriptionError(f"expected '{{' at: {text[i : i + 40]!r}")
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{" and text[j - 1] != "\\":
            depth += 1
        elif text[j] == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    raise TranscriptionError(f"unbalanced group at: {text[i : i + 40]!r}")


def math_macros(preamble: str) -> dict[str, str]:
    """The notes' argument-free math macros (\\betac, \\Ra, \\K, ...), for expansion."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", preamble):
        body, _ = braced(preamble, m.end() - 1)
        if m.group(1) in TEXT_MACROS or "#" in body or "\\paper" in body:
            continue
        if re.search(r"\\(mathrm|mathcal|beta|varrho)", body):
            out[m.group(1)] = body
    return out


def link_macros(preamble: str) -> dict[str, tuple[str, str]]:
    """The notes' citation macros, \\href{url}{text} without arguments."""
    out = {}
    for m in re.finditer(
        r"\\newcommand\{\\(\w+)\}\{\\href\{([^}]*)\}\{(.*)\}\}", preamble
    ):
        out[m.group(1)] = (m.group(2), m.group(3))
    return out


def read_aux(aux: str) -> dict[str, str]:
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))


def compile_aux(tex: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(2):
            run = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={tmp}",
                    tex.name,
                ],
                cwd=tex.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if run.returncode:
                raise TranscriptionError("pdflatex failed:\n" + run.stdout[-2000:])
        return (Path(tmp) / tex.with_suffix(".aux").name).read_text(encoding="utf-8")


# ---- the transcription -----------------------------------------------------------


class Notes:
    def __init__(self, tex: str, aux: dict[str, str]):
        self.source = tex
        pre, body = tex.split(r"\begin{document}", 1)
        self.body = strip_comments(body.split(r"\end{document}", 1)[0])
        pre = strip_comments(pre)
        self.math = math_macros(pre)
        self.links = link_macros(pre)
        url = re.search(r"\\newcommand\{\\paperurl\}\{([^}]*)\}", pre)
        self.paper_url = url.group(1) if url else ""
        self.version = re.search(r"\\date\{([^}]*?),", pre).group(1).strip()
        self.aux = aux
        self.numbers: dict[str, str] = {}  # label -> number, counted here
        self.sec = [0, 0, 0]  # \setcounter{section}{n} in the body resets it
        self.eq = 0
        self.tab = 0

    # -- numbers --

    def number(self, kind: str) -> str:
        if kind == "section":
            self.sec = [self.sec[0] + 1, 0, 0]
            self.eq = self.tab = 0
            return f"T:{self.sec[0]}"
        if kind == "subsection":
            self.sec[1] += 1
            self.sec[2] = 0
            return f"T:{self.sec[0]}.{self.sec[1]}"
        if kind == "subsubsection":
            self.sec[2] += 1
            return f"T:{self.sec[0]}.{self.sec[1]}.{self.sec[2]}"
        if kind == "equation":
            self.eq += 1
            return f"T:{self.sec[0]}.{self.eq}"
        self.tab += 1
        return f"T:{self.sec[0]}.{self.tab}"

    def label(self, key: str, number: str) -> str:
        if self.aux.get(key) != number:
            raise TranscriptionError(
                f"{key}: counted {number}, LaTeX has {self.aux.get(key)}"
            )
        self.numbers[key] = number
        return ".. _" + anchor(key) + ":\n\n"

    # -- math --

    def expand(self, math: str) -> str:
        # A macro is one token in LaTeX: after a command that takes an argument
        # (\hat\K) its body is braced, or MathJax reads only the body's first token.
        for name in sorted(self.math, key=len, reverse=True):
            math = re.sub(
                r"(\\[A-Za-z]+\s*)?\\" + name + r"(?![A-Za-z])",
                lambda m, b=self.math[name]: f"{m.group(1)}{{{b}}}" if m.group(1) else b,
                math,
            )
        if re.search(r"\\(label|ref|eref|sref|tref|peq)\b", math):
            raise TranscriptionError(f"a reference inside math: {math[:60]!r}")
        return " ".join(math.split())

    # -- inline text --

    def inline(self, text: str) -> str:
        """LaTeX running text as reST inline markup."""
        parts = self._inline(text)
        out = []
        for kind, s in parts:
            if kind == "role":
                # reST needs a boundary before markup: after a word, or after other
                # markup that ends right here (an escaped space joins them invisibly).
                if out and re.search(r"[\w)\]`*]$", out[-1]):
                    out.append("\\ ")
                out.append(s)
            else:
                if (
                    out
                    and re.match(r"\w", s)
                    and out[-1].endswith(("`", "`_", "`__", "*"))
                ):
                    out.append("\\ ")
                out.append(s)
        return re.sub(r"[ \t\n]+", " ", "".join(out)).strip()

    def _inline(self, text: str) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        i, buf = 0, []

        def flush():
            if buf:
                parts.append(("text", esc("".join(buf))))
                buf.clear()

        def role(s):
            flush()
            parts.append(("role", s))

        while i < len(text):
            c = text[i]
            if c == "$":
                j = text.index("$", i + 1)
                role(f":math:`{self.expand(text[i + 1 : j])}`")
                i = j + 1
            elif c == "~":
                buf.append(NBSP)
                i += 1
            elif text.startswith("``", i):
                buf.append("\u201c")
                i += 2
            elif text.startswith("''", i):
                buf.append("\u201d")
                i += 2
            elif text.startswith("--", i):
                buf.append("-")
                i += 3 if text.startswith("---", i) else 2
            elif c == "\\":
                m = re.match(r"\\([A-Za-z]+)\*?|\\(.)", text[i:])
                name, sym = m.group(1), m.group(2)
                i += m.end()
                if sym is not None:
                    if sym in "&%_#$":
                        buf.append(sym)
                    elif sym in " ,":
                        buf.append(" ")
                    else:
                        raise TranscriptionError(f"unknown symbol \\{sym}")
                    continue
                if name in DROP:
                    continue
                if name in TEXT_MACROS:
                    buf.append(TEXT_MACROS[name])
                    continue
                if name in ("textit", "emph", "textbf"):
                    inner, i = braced(text, i)
                    flush()
                    parts.extend(
                        emphasis(self._inline(inner), "**" if name == "textbf" else "*")
                    )
                    continue
                if name == "textsc":
                    inner, i = braced(text, i)
                    buf.append(inner)
                    continue
                if name == "href":
                    url, i = braced(text, i)
                    label, i = braced(text, i)
                    url = self.paper_url if url == r"\paperurl" else url
                    role(f"`{plain(self, label)} <{url}>`__")
                    continue
                if name in self.links:
                    url, label = self.links[name]
                    role(f"`{plain(self, label)} <{url}>`__")
                    continue
                if name == "paper":
                    role(PAPER)
                    continue
                if name in ("peq", "peqn", "psec", "papp", "pfig"):
                    arg, i = braced(text, i)
                    what = {
                        "peq": "Eq.",
                        "peqn": "Eq.",
                        "psec": "Section",
                        "papp": "Appendix",
                        "pfig": "Fig.",
                    }[name]
                    arg = f"({arg})" if what == "Eq." else arg
                    if name == "peqn":
                        buf.append(f"Eq.{NBSP}{arg}")
                    else:
                        role(PAPER)
                        buf.append(f", {what}{NBSP}{arg}")
                    continue
                if name in ("eref", "sref", "tref", "ref"):
                    key, i = braced(text, i)
                    number = self.aux.get(key)
                    if number is None:
                        raise TranscriptionError(f"reference to an unknown label {key}")
                    shown = {
                        "eref": f"Eq. ({number})",
                        "sref": f"Section {number}",
                        "tref": f"Table {number}",
                        "ref": number,
                    }[name]
                    role(f":ref:`{shown} <{anchor(key)}>`")
                    continue
                raise TranscriptionError(
                    f"unknown command \\{name} near: {text[max(0, i - 40) : i + 20]!r}"
                )
            elif c in "{}":
                i += 1
            else:
                buf.append(c)
                i += 1
        flush()
        return parts

    # -- blocks --

    def paragraphs(self, text: str) -> list[str]:
        out = []
        for para in re.split(r"\n\s*\n", text):
            if para.strip():
                out.append(self.inline(para) + "\n\n")
        return out

    def equation(self, content: str) -> str:
        key = re.search(r"\\label\{([^}]+)\}", content)
        number = self.number("equation")
        content = re.sub(r"\\label\{[^}]+\}", "", content)
        math = self.expand(content).replace("\\\\", "\\\\\n     ")
        head = self.label(key.group(1), number) if key else ""
        return (
            f"{head}.. math::\n   :nowrap:\n\n   \\begin{{equation}}\n     {math}"
            f"\n     \\tag{{{number}}}\n   \\end{{equation}}\n\n"
        )

    def table(self, content: str) -> str:
        cap, key = (
            re.search(r"\\caption\{", content),
            re.search(r"\\label\{([^}]+)\}", content),
        )
        caption, _ = braced(content, cap.end() - 1)
        number = self.number("table")
        head = self.label(key.group(1), number) if key else ""
        tab = re.search(
            r"\\begin\{tabular\}\{[^}]*\}(.*)\\end\{tabular\}", content, re.S
        ).group(1)
        tab = re.sub(r"\\(top|mid|bottom)rule", "", tab)
        rows = [r for r in re.split(r"\\\\", tab) if r.strip()]
        cells = [[self.inline(c) for c in r.split("&")] for r in rows]
        lines = [
            f"{head}.. list-table:: Table {number}. {self.inline(caption)}",
            "   :header-rows: 1",
            "",
        ]
        for r in cells:
            lines.append(f"   * - {r[0]}")
            lines += [f"     - {c}" for c in r[1:]]
        return "\n".join(lines) + "\n\n"

    def items(self, env: str, content: str) -> str:
        if env == "list":  # \begin{list}{label}{spacing}: both arguments are layout
            content = content.strip()
            for _ in range(2):
                _, j = braced(content, 0)
                content = content[j:].lstrip()
        mark = "#." if env == "enumerate" else "-"
        pad = " " * (len(mark) + 1)
        out = []
        for item in re.split(r"\\item\b", content)[1:]:
            body = "".join(self.flow(item)).rstrip("\n")
            lines = body.split("\n")
            out.append(
                f"{mark} {lines[0]}\n"
                + "".join((pad + line if line else "") + "\n" for line in lines[1:])
            )
        return "\n".join(out) + "\n"

    def flow(self, text: str) -> list[str]:
        """Running text and its blocks: paragraphs, equations, tables, lists."""
        out, i = [], 0
        begin = re.compile(r"\\begin\{(\w+)\}")
        while True:
            m = begin.search(text, i)
            out += self.paragraphs(text[i : m.start() if m else len(text)])
            if not m:
                return out
            env = m.group(1)
            if env not in ENVS:
                near = text[m.start() : m.start() + 60]
                raise TranscriptionError(f"unknown environment {env} near: {near!r}")
            end = text.index(f"\\end{{{env}}}", m.end())
            content = text[m.end() : end]
            if env == "equation":
                out.append(self.equation(content))
            elif env == "table":
                out.append(self.table(content))
            else:
                out.append(self.items(env, content))
            i = end + len(f"\\end{{{env}}}")

    def heading(self, kind: str, title: str, key: str | None, star: bool) -> str:
        if star:
            text = self.inline(title)
        else:
            number = self.number(kind)
            text = f"{number} {self.inline(title)}"
        head = self.label(key, number) if key and not star else ""
        return f"{head}{text}\n{HEAD[kind] * len(text)}\n\n"

    # -- the whole document --

    def pages(self) -> tuple[str, list[tuple[str, str]]]:
        """The opening text and one (slug, rst) per section."""
        body = self.body
        token = re.compile(
            r"\\(section|subsection|subsubsection)(\*?)\{"
            r"|\\setcounter\{section\}\{(-?\d+)\}"
        )
        opening, pages, cur = [], [], None
        i = 0
        while True:
            m = token.search(body, i)
            (cur[1] if cur else opening).extend(
                self.flow(body[i : m.start() if m else len(body)])
            )
            if not m:
                break
            if m.group(3) is not None:
                self.sec[0] = int(m.group(3))
                i = m.end()
                continue
            kind, star = m.group(1), bool(m.group(2))
            title, j = braced(body, m.end() - 1)
            key = None
            lab = re.match(r"\s*\\label\{([^}]+)\}", body[j:])
            if lab:
                key, j = lab.group(1), j + lab.end()
            if kind == "section":
                slug = (
                    key.split(":")[-1]
                    if key
                    else re.sub(r"\W+", "-", title.lower()).strip("-")
                )
                cur = (slug, [])
                pages.append(cur)
            cur[1].append(self.heading(kind, title, key, star))
            i = j
        missing = set(self.aux) - set(self.numbers)
        if missing:
            raise TranscriptionError(
                f"labels LaTeX numbers that were not met: {sorted(missing)}"
            )
        return "".join(opening), [(slug, "".join(parts)) for slug, parts in pages]


def anchor(key: str) -> str:
    return key.replace(":", "-")


def esc(text: str) -> str:
    text = re.sub(r"[ \t\n]+", " ", text)
    return re.sub(r"([*`|\\_])", r"\\\1", text)


def plain(notes: Notes, text: str) -> str:
    """Link text: no markup survives inside it."""
    out = "".join(s for kind, s in notes._inline(text))
    if ":math:" in out or "`" in out.replace("\\`", ""):
        raise TranscriptionError(f"markup inside a link's text: {text[:60]!r}")
    return out.replace("\\", "")


def emphasis(parts: list[tuple[str, str]], mark: str) -> list[tuple[str, str]]:
    """Emphasis over text runs only: reST does not nest a role inside emphasis."""
    out = []
    for kind, s in parts:
        if kind == "text" and s.strip():
            lead, core, trail = re.match(r"(\s*)(.*?)(\s*)$", s, re.S).groups()
            if lead:
                out.append(("text", lead))
            out.append(("role", f"{mark}{core}{mark}"))
            if trail:
                out.append(("text", trail))
        else:
            out.append((kind, s))
    return out


GENERATED = (
    ".. Generated by docs/tools/notes_pages.py from the technical notes ({version}, "
    "source sha256 {sha}). Do not edit; rerun the tool.\n\n"
)


def write(tex_path: Path, aux_text: str) -> list[Path]:
    tex = tex_path.read_text(encoding="utf-8")
    notes = Notes(tex, read_aux(aux_text))
    opening, pages = notes.pages()
    stamp = GENERATED.format(
        version=notes.version, sha=hashlib.sha256(tex.encode("utf-8")).hexdigest()[:12]
    )
    OUT_DIR.mkdir(exist_ok=True)
    written = []
    for slug, rst in pages:
        path = OUT_DIR / f"{slug}.rst"
        path.write_text(stamp + rst.rstrip() + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    toc = "\n".join(f"   technical-notes/{slug}" for slug, _ in pages)
    INDEX.write_text(
        stamp
        + "Technical notes\n===============\n\n"
        + opening
        + f"This is {notes.version.lower()} of the notes. Their numbers carry the "
        "prefix T: (sections T:2.3, equations (T:2.4), tables T:2.1); the package's "
        "docstrings cite them in the form the :doc:`API reference <api/index>` states. "
        "The notes "
        "are also published as a PDF with each release; the link is added with "
        "ver. 1.0.0.\n\n"
        f".. toctree::\n   :maxdepth: 2\n\n{toc}\n",
        encoding="utf-8",
        newline="\n",
    )
    return [INDEX, *written]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tex", type=Path)
    ap.add_argument("--aux", type=Path)
    a = ap.parse_args(argv)
    try:
        aux = (
            a.aux.read_text(encoding="utf-8") if a.aux else compile_aux(a.tex.resolve())
        )
        for p in write(a.tex, aux):
            print(p.relative_to(DOCS).as_posix())
    except TranscriptionError as err:
        print(f"transcription refused: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
