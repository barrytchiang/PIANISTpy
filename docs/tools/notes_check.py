"""Check the rendered technical-notes pages for anything the transcription let through.

    python docs/tools/notes_check.py <built manual>

Reads ``reference/notes.html`` and ``reference/technical-notes/*.html`` of a built
manual and reports, outside the math, any LaTeX command, dollar sign, unrendered reST
role or stray backtick or asterisk, and counts the equation tags against the
``.. math::`` blocks of the sources. Exit status 1 on any finding.
"""

import html
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
MATH = re.compile(r'<(span|div) class="math[^>]*>.*?</\1>', re.S)
LEFTOVER = re.compile(r"\\[A-Za-z]+|\$|:math:|:ref:|`|\*\S")


def main(built: Path) -> int:
    root = built / "reference"
    pages = [root / "notes.html", *sorted((root / "technical-notes").glob("*.html"))]
    findings, tags = 0, 0
    for page in pages:
        article = page.read_text(encoding="utf-8").split("<article", 1)[1]
        article = article.split("</article>", 1)[0]
        tags += article.count("\\tag{T:")
        text = html.unescape(re.sub(r"<[^>]+>", " ", MATH.sub("", article)))
        for m in LEFTOVER.finditer(text):
            findings += 1
            near = " ".join(text[max(0, m.start() - 50) : m.end() + 30].split())
            print(f"{page.name}: {near}")
    sources = sorted((DOCS / "reference" / "technical-notes").glob("*.rst"))
    blocks = sum(s.read_text(encoding="utf-8").count(".. math::") for s in sources)
    if tags != blocks:
        findings += 1
        print(f"{tags} equation tags rendered, {blocks} equations transcribed")
    print(f"{len(pages)} pages, {tags} equations, {findings} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
