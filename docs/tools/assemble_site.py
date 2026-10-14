"""Assemble pianistpy.com from the built manual and the landing page.

    python docs/tools/assemble_site.py <docs source> <site root>

After ``sphinx -b html docs <site root>/docs`` this puts the landing page at
the site's root, copies the logo beside it, writes ``robots.txt``, the
``.nojekyll`` marker, ``llms.txt`` and, for every page of the manual, a
markdown twin (``page.html`` to ``page.md``) plus one ``llms-full.txt`` that
concatenates them. The twins are for readers that fetch text rather than
render HTML; nothing on the human-facing pages changes.
"""

import html
import re
import shutil
import sys
from pathlib import Path

# The manual's pages that llms.txt lists first, in order, with one line each.
# Paths are relative to /docs/; the list is edited by hand with the manual.
PRIMARY = [
    ("start/index.html", "Install, requirements, quick start"),
    ("guide/index.html", "The user guide, one page per keyword"),
    ("guide/results.html", "What the package returns, flags and refuses"),
    ("notebooks/index.html", "Notebook tutorials and the paper's figures"),
    ("reference/api/index.html", "The API reference, by subpackage"),
    ("reference/notes.html", "The technical notes"),
    ("project/citing.html", "How to cite"),
]

SUMMARY = (
    "PIANISTpy builds equilibrium models of spherical systems with a chosen "
    "velocity anisotropy: distribution functions by analytic inversion, "
    "truncation in phase space, particle initial conditions, line-of-sight "
    "observables, superposition and a differentiable likelihood. Python, JAX."
)

TAG = re.compile(r"<[^>]+>")
SCRIPT = re.compile(r"<(script|style|nav|footer|header)\b.*?</\1>", re.S | re.I)
MAIN = re.compile(r"<main\b.*?</main>|<article\b.*?</article>", re.S | re.I)
WS = re.compile(r"[ \t]+\n")
BLANKS = re.compile(r"\n{3,}")
PRE = re.compile(r"<pre\b.*?</pre>", re.S | re.I)
HEAD = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.S | re.I)
TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
HEADERLINK = re.compile(r'<a class="headerlink"[^>]*>.*?</a>', re.S | re.I)


def page_to_markdown(text: str) -> tuple[str, str]:
    """A plain-text rendering of one built page: the title and the body."""
    title = html.unescape(TAG.sub("", (TITLE.search(text) or [None, ""])[1])).strip()
    title = re.split(r"\s+[—–-]\s+PIANISTpy", title)[0].strip()
    body = (MAIN.search(text) or [text])[0]
    body = SCRIPT.sub("", body)
    body = HEADERLINK.sub("", body)
    body = PRE.sub(lambda m: "\n```\n" + TAG.sub("", m.group(0)) + "\n```\n", body)
    body = HEAD.sub(lambda m: "\n" + "#" * int(m.group(1)) + " " + TAG.sub("", m.group(2)) + "\n", body)
    body = re.sub(r"</(p|div|li|tr|dd|dt|blockquote)>", "\n", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = TAG.sub("", body)
    body = html.unescape(body)
    body = WS.sub("\n", body)
    body = BLANKS.sub("\n\n", body).strip()
    return title, body


def main(source: str, root: str) -> int:
    docs_src, site = Path(source), Path(root)
    manual = site / "docs"
    if not manual.is_dir():
        print(f"no built manual at {manual}")
        return 1
    # The landing page and the logo at the root.
    shutil.copy(docs_src / "landing" / "index.html", site / "index.html")
    (site / "static").mkdir(exist_ok=True)
    shutil.copy(docs_src / "_static" / "logo.png", site / "static" / "logo.png")
    (site / ".nojekyll").write_text("")
    # myst-nb's execution folder is written beside the manual; nothing serves it.
    if (site / "jupyter_execute").is_dir():
        shutil.rmtree(site / "jupyter_execute")
    (site / "robots.txt").write_text("User-agent: *\nAllow: /\nSitemap: https://pianistpy.com/docs/sitemap.xml\n")
    # The markdown twins and the full text.
    full = []
    for page in sorted(manual.rglob("*.html")):
        rel = page.relative_to(manual)
        if rel.parts[0] in ("_static", "_sources", "_modules", "_images") or rel.name in (
            "genindex.html",
            "search.html",
            "py-modindex.html",
        ):
            continue
        title, body = page_to_markdown(page.read_text(encoding="utf-8"))
        md = f"# {title}\n\nSource: https://pianistpy.com/docs/{rel.as_posix()}\n\n{body}\n"
        page.with_suffix(".md").write_text(md, encoding="utf-8")
        full.append(md)
    (manual / "llms-full.txt").write_text("\n\n---\n\n".join(full), encoding="utf-8")
    # llms.txt at the root.
    lines = ["# PIANISTpy", "", f"> {SUMMARY}", "",
             "Every page of the manual has a markdown twin at the same address with `.md` in",
             "place of `.html`; the whole manual as one file is /docs/llms-full.txt.", "",
             "## Documentation", ""]
    for path, note in PRIMARY:
        lines.append(f"- [{note}](https://pianistpy.com/docs/{Path(path).with_suffix('.md').as_posix()})")
    lines += ["", "## Optional", "",
              "- [The package on GitHub](https://github.com/barrytchiang/PIANISTpy)",
              "- [The notebooks on GitHub](https://github.com/barrytchiang/PIANISTpy-notebooks)",
              "- [The full manual as one text file](https://pianistpy.com/docs/llms-full.txt)", ""]
    (site / "llms.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"site assembled at {site}: {len(full)} pages twinned")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
