"""Assemble pianistpy.com from the built manual and the landing page.

    python docs/tools/assemble_site.py <docs source> <site root>

After ``sphinx -b html docs <site root>/docs`` this puts the landing page at
the site's root, copies the logo beside it, writes ``robots.txt``, the
``.nojekyll`` marker, ``llms.txt``, the sitemap and, for every page of the manual, a
markdown twin (``page.html`` to ``page.md``) plus one ``llms-full.txt`` that
concatenates them. The twins are for readers that fetch text rather than
render HTML; nothing on the human-facing pages changes.
"""

import html
import posixpath
import re
import shutil
import sys
import urllib.parse
from pathlib import Path

# The manual's pages that llms.txt lists first, in order, with one line each.
# Paths are relative to /docs/; the list is edited by hand with the manual.
PRIMARY = [
    (
        "start/index.html",
        "Get started: what the package does, installation, quick start and "
        "representative computing time",
    ),
    ("guide/index.html", "User guide: one page per topic"),
    (
        "guide/flags-refusals.html",
        "Flags and refusals: what the package returns, flags and refuses",
    ),
    (
        "notebooks/index.html",
        "Examples: tutorial notebooks and the result figures of @PAPER@",
    ),
    ("reference/api/index.html", "Functions by subpackage: the API reference"),
    ("reference/notes.html", "Technical notes"),
    ("project/citing.html", "How to cite"),
]

SUMMARY = (
    "PIANISTpy builds equilibrium models of spherical systems with a chosen "
    "velocity anisotropy: distribution functions by analytic inversion, "
    "truncation in phase space, particle initial conditions, line-of-sight "
    "observables, superposition and a differentiable likelihood. Python, JAX."
)

# What /docs/index.html becomes once the landing page has taken its band.
REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PIANISTpy</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="https://pianistpy.com/">
<meta http-equiv="refresh" content="0; url=../index.html">
</head>
<body><p><a href="../index.html">pianistpy.com</a></p></body>
</html>
"""

TAG = re.compile(r"<[^>]+>")
SCRIPT = re.compile(r"<(script|style|nav|footer|header)\b.*?</\1>", re.S | re.I)
MAIN = re.compile(r"<main\b.*?</main>|<article\b.*?</article>", re.S | re.I)
WS = re.compile(r"[ \t]+\n")
BLANKS = re.compile(r"\n{3,}")
PRE = re.compile(r"<pre\b.*?</pre>", re.S | re.I)
HEAD = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.S | re.I)
TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
HEADERLINK = re.compile(r'<a class="headerlink"[^>]*>.*?</a>', re.S | re.I)
ARTICLE = re.compile(r"<article\b.*?</article>", re.S | re.I)
LINK = re.compile(r'<a\b[^>]*\bhref="([^"]*)"[^>]*>(.*?)</a>', re.S | re.I)
CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.S | re.I)
STRONG = re.compile(r"<(strong|b)\b[^>]*>(.*?)</\1>", re.S | re.I)
EMPH = re.compile(r"<(em|i)\b[^>]*>(.*?)</\1>", re.S | re.I)
SUBSUP = re.compile(r"<(/?)(sub|sup)>", re.I)


def page_to_markdown(text: str, url: str = "") -> tuple[str, str]:
    """A markdown rendering of one built page: the title and the body. Links are
    made absolute against the page's address ``url``; code, emphasis, list items
    and sub- and superscripts keep their markdown or HTML form."""
    title = html.unescape(TAG.sub("", (TITLE.search(text) or [None, ""])[1])).strip()
    title = re.split(r"\s+[—–-]\s+PIANISTpy", title)[0].strip()
    body = (ARTICLE.search(text) or MAIN.search(text) or [text])[0]
    body = SCRIPT.sub("", body)
    body = HEADERLINK.sub("", body)
    # Code blocks are set aside first, so nothing below rewrites their text.
    blocks: list[str] = []

    def keep(m: re.Match) -> str:
        # The block's language is the highlight-<language> class Sphinx puts on
        # the div that encloses the <pre>; "default" is Sphinx's name for Python.
        before = body[max(0, m.start() - 300) : m.start()]
        found = re.findall(r"highlight-([a-z0-9]+)", before)
        language = found[-1] if found else ""
        language = "python" if language == "default" else language
        blocks.append(
            f"\n```{language}\n"
            + html.unescape(TAG.sub("", m.group(0))).strip("\n")
            + "\n```\n"
        )
        return f"\x00{len(blocks) - 1}\x00"

    body = PRE.sub(keep, body)
    body = HEAD.sub(
        lambda m: "\n" + "#" * int(m.group(1)) + " " + TAG.sub("", m.group(2)) + "\n",
        body,
    )
    body = CODE.sub(lambda m: "`" + TAG.sub("", m.group(1)) + "`", body)
    body = STRONG.sub(lambda m: "**" + m.group(2) + "**", body)
    body = EMPH.sub(lambda m: "*" + m.group(2) + "*", body)
    body = LINK.sub(
        lambda m: (
            "["
            + TAG.sub("", m.group(2)).strip()
            + "]("
            + urllib.parse.urljoin(url, m.group(1))
            + ")"
            if url
            else m.group(2)
        ),
        body,
    )
    body = re.sub(r"<li\b[^>]*>\s*", "\n- ", body, flags=re.I)
    body = SUBSUP.sub(lambda m: f"\x01{m.group(1)}{m.group(2)}\x02", body)
    body = re.sub(r"</(p|div|li|tr|dd|dt|blockquote|ul|ol)>", "\n", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = TAG.sub("", body)
    body = body.replace("\x01", "<").replace("\x02", ">")
    body = html.unescape(body)
    body = re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], body)
    body = WS.sub("\n", body)
    body = re.sub(r"\n- \n+", "\n- ", body)
    body = BLANKS.sub("\n\n", body).strip()
    return title, body


def page_address(rel: Path) -> str:
    """A page's address as the links write it: no ".html", an index as its folder."""
    path = rel.with_suffix("").as_posix()
    if path == "index" or path.endswith("/index"):
        return path[: -len("index")]
    return path


NEXT = re.compile(r'<link rel="next"[^>]*\bhref="([^"#]+)"', re.I)


def reading_order(pages: list[Path]) -> list[Path]:
    """The pages in the manual's reading order: from each page that no page
    names as its next, along the rel="next" links; pages off every chain last."""
    known = {p.resolve(): p for p in pages}
    following = {}
    for page in pages:
        match = NEXT.search(page.read_text(encoding="utf-8"))
        if match is None:
            continue
        target = (page.parent / match.group(1)).resolve()
        for candidate in (
            target,
            target.with_name(target.name + ".html"),
            target / "index.html",
        ):
            if candidate in known:
                following[page.resolve()] = candidate
                break
    order, seen = [], set()
    heads = [p for p in known if p not in set(following.values())]
    for start in heads + list(known):
        page = start
        while page is not None and page not in seen:
            seen.add(page)
            order.append(known[page])
            page = following.get(page)
    return order


ANCHOR = re.compile(r"<a\b[^>]*>", re.I)
EXTERNAL = re.compile(r'href="https?://(?!(?:www\.)?pianistpy\.com[/"])', re.I)
REL = re.compile(r'\brel="([^"]*)"', re.I)


def external_in_new_tab(text: str) -> str:
    """Every link that leaves pianistpy.com opens in a new tab (``target="_blank"``,
    with ``rel="noopener"`` so the new page cannot reach this one)."""

    def tab(m: re.Match) -> str:
        tag = m.group(0)
        if not EXTERNAL.search(tag) or "target=" in tag:
            return tag
        rel = REL.search(tag)
        if rel is None:
            tag = tag[:-1] + ' rel="noopener">'
        elif "noopener" not in rel.group(1).split():
            tag = REL.sub(lambda r: f'rel="{r.group(1)} noopener"', tag, count=1)
        return tag[:-1] + ' target="_blank">'

    return ANCHOR.sub(tab, text)


INLINE_MATH = re.compile(
    r'([(\[]?)(<span class="math notranslate nohighlight">.*?</span>)([,.;:!?)\]]*)',
    re.S,
)


def glue_math(text: str) -> str:
    """An inline formula and the bracket before it or the punctuation after it stay on
    one line: MathJax renders each formula as a box, and a line may otherwise break
    between the box and a following comma or semicolon (span ``nobr``, custom.css)."""

    def glue(m: re.Match) -> str:
        before, formula, after = m.groups()
        if not before and not after:
            return formula
        return f'<span class="nobr">{before}{formula}{after}</span>'

    return INLINE_MATH.sub(glue, text)


BAND = re.compile(r"<!-- band -->.*?<!-- /band -->", re.S)
MANUAL_ONLY = re.compile(r"\s*<!-- manual-only -->.*?<!-- /manual-only -->", re.S)
HREF = re.compile(r'href="([^"]*)"')


def with_band(landing: str, manual_home: str) -> str:
    """The landing page with the manual home's site band in place of its band markers.

    The band's links are relative to /docs/; on the landing page they are rebased to
    the site's root (``start/index.html`` to ``docs/start/index.html``,
    ``../index.html`` to ``index.html``), and the manual-only controls (the button
    that opens the sidebar) are dropped.
    """
    band = BAND.search(manual_home)
    if band is None or BAND.search(landing) is None:
        raise SystemExit(
            "assemble_site: the band markers are missing from the manual's home "
            "or the landing page"
        )

    def rebase(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http:", "https:", "#", "mailto:")):
            return m.group(0)
        return f'href="{posixpath.normpath("docs/" + href)}"'

    text = HREF.sub(rebase, MANUAL_ONLY.sub("", band.group(0)))
    return BAND.sub(lambda _: text, landing, count=1)


TOCTREE = re.compile(r"^\.\. toctree::\n((?:[ \t]+.*\n|[ \t]*\n)*)", re.M)
ENTRY = re.compile(r"^[ \t]+(?!:)(?:.*<(\S+)>|(\S+))[ \t]*$", re.M)


def list_only_tabs(docs_src: Path) -> dict[str, str]:
    """The tabs whose index page holds nothing but its title and its list of pages,
    each mapped to its first page: ``{"start/index.html": "start/overview.html"}``.

    Such a page repeats the tab's menu and the sidebar, so its tab, and every link to
    it, opens the first page instead.
    """
    home = (docs_src / "index.rst").read_text(encoding="utf-8")
    tabs = {}
    for tree in TOCTREE.finditer(home):
        for entry in ENTRY.finditer(tree.group(1)):
            name = entry.group(1) or entry.group(2)
            if not name.endswith("/index"):
                continue
            text = (docs_src / f"{name}.rst").read_text(encoding="utf-8")
            trees = list(TOCTREE.finditer(text))
            rest = TOCTREE.sub("", text)
            rest = re.sub(r"^(\S.*)\n[=\-~^]+\n", "", rest, count=1, flags=re.M)
            rest = re.sub(r"^\.\. _[^:]+:\s*$", "", rest, flags=re.M)
            if rest.strip() or not trees:
                continue
            first = ENTRY.search(trees[0].group(1))
            if first is None:
                continue
            target = posixpath.join(
                posixpath.dirname(name), first.group(1) or first.group(2)
            )
            tabs[f"{name}.html"] = f"{target}.html"
    return tabs


def tab_redirect(index: str, target: str) -> str:
    """What a list-only tab's index page becomes: a redirect to its first page."""
    name = posixpath.relpath(target, posixpath.dirname(index))
    return REDIRECT.replace("../index.html", name).replace(
        "https://pianistpy.com/",
        f"https://pianistpy.com/docs/{target[: -len('.html')]}",
    )


def retarget(text: str, here: str, tabs: dict[str, str]) -> str:
    """Links of a page at ``here`` (relative to /docs/) that open a list-only tab's
    index page open its first page instead."""

    def sub(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http:", "https:", "#", "mailto:")):
            return m.group(0)
        path, sep, anchor = href.partition("#")
        if not path:
            return m.group(0)
        # the links write an address as "x/index.html", "x/index" or "x/"
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(here), path))
        if path.endswith("/"):
            resolved += "/index"
        bare = not resolved.endswith(".html")
        key = resolved + ".html" if bare else resolved
        if key not in tabs:
            return m.group(0)
        new = posixpath.relpath(tabs[key], posixpath.dirname(here) or ".")
        if bare:
            new = new[: -len(".html")]
        return f'href="{new}{sep if anchor else ""}{anchor}"'

    return HREF.sub(sub, text)


FEATURES = re.compile(r'<ul class="[^"]*\bfeatures\b[^"]*">(.*?)</ul>', re.S)
ITEM = re.compile(r"<li>\s*(?:<p>)?(.*?)(?:</p>)?\s*</li>", re.S)
FIRST_LINK = re.compile(
    r'^<a class="[^"]*" href="([^"]*)"><span class="[^"]*">(.*?)</span></a>', re.S
)
HREF = re.compile(r'href="([^"]*)"')


def rebase(href: str) -> str:
    """A link of the overview page (/docs/start/) as the site's root reads it."""
    if href.startswith("#") or "://" in href:
        return href
    path, _, anchor = href.partition("#")
    return posixpath.normpath("docs/start/" + path) + (f"#{anchor}" if anchor else "")


def features_from_overview(overview: str) -> str:
    """The landing page's feature list, from the built "What PIANISTpy does" page.

    That page (docs/start/overview.rst) is the one source: each item's name links to
    its guide page and is set in bold as the landing page's styles expect; every link
    of the item is rebased from /docs/start/ to the site's root.
    """
    found = FEATURES.search(overview)
    if found is None:
        raise SystemExit("assemble_site: the overview page has no features list")
    items = []
    for raw in ITEM.findall(found.group(1)):
        body = " ".join(raw.split())
        link = FIRST_LINK.match(body)
        if link is None:
            raise SystemExit(
                "assemble_site: a feature item does not start with its link: "
                + body[:80]
            )
        rest = HREF.sub(lambda m: f'href="{rebase(m.group(1))}"', body[link.end() :])
        name = f'<a href="{rebase(link.group(1))}"><b>{link.group(2)}</b></a>'
        items.append(f"      <li>{name}{rest}</li>")
    return "\n".join(items)


def conf_value(conf: str, name: str) -> str:
    """A plain string assignment of conf.py, ``NAME = "..."``."""
    found = re.search(rf'^{name} = "([^"]+)"', conf, re.M)
    if found is None:
        raise SystemExit(f"assemble_site: conf.py has no {name}")
    return found.group(1)


def main(source: str, root: str) -> int:
    docs_src, site = Path(source), Path(root)
    manual = site / "docs"
    if not manual.is_dir():
        print(f"no built manual at {manual}")
        return 1
    # The landing page and the logo at the root. The landing page gets the manual's
    # site band; the release paper's citation from conf.py (as the pages); the
    # release from the package's _version.py (as conf.py's `release`); and the
    # feature list from the built overview page.
    landing = (docs_src / "landing" / "index.html").read_text(encoding="utf-8")
    conf = (docs_src / "conf.py").read_text(encoding="utf-8")
    paper = {
        name: conf_value(conf, name)
        for name in ("PAPER", "PAPER_FULL", "PAPER_STATUS", "PAPER_URL")
    }
    version = conf_value(
        (docs_src.parent / "pianistpy" / "_version.py").read_text(encoding="utf-8"),
        "__version__",
    )
    # The release notes on the landing page are those of the release being built.
    notes = re.search(r'<section id="releases">(.*?)</section>', landing, re.S)
    if notes is None or f"<b>Ver. {version}</b>" not in notes.group(1):
        raise SystemExit(
            f"assemble_site: the landing page's release notes do not describe {version}"
        )
    # The full citation comes first on the landing page; every later mention is short.
    first = re.search(r"@PAPER(_FULL)?@", landing)
    if first is None or first.group(1) is None:
        raise SystemExit(
            "assemble_site: the landing page's first citation is not the full form"
        )
    landing = landing.replace("@RELEASE@", version)
    for name in ("PAPER_FULL", "PAPER_STATUS", "PAPER_URL", "PAPER"):
        landing = landing.replace(f"@{name}@", html.escape(paper[name]))
    landing = landing.replace(
        "@FEATURES@",
        features_from_overview(
            (manual / "start" / "overview.html").read_text(encoding="utf-8")
        ),
    )
    leftover = re.search(r"@[A-Z_]+@", landing)
    if leftover is not None:
        raise SystemExit(f"assemble_site: the landing page keeps {leftover.group(0)}")
    (site / "index.html").write_text(
        with_band(landing, (manual / "index.html").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    # The reading order, read before the home and the list-only tab pages become
    # redirects (their rel="next" links join the tabs into one chain).
    order = reading_order(sorted(manual.rglob("*.html")))
    # The manual's home only holds the toctree that drives the tabs; the address
    # sends readers to the landing page.
    (manual / "index.html").write_text(REDIRECT, encoding="utf-8")
    # A tab whose index page is only its list of pages opens its first page.
    tabs = list_only_tabs(docs_src)
    for index, target in tabs.items():
        (manual / index).write_text(tab_redirect(index, target), encoding="utf-8")
    (site / "static").mkdir(exist_ok=True)
    shutil.copy(docs_src / "_static" / "logo.png", site / "static" / "logo.png")
    (site / ".nojekyll").write_text("")
    # myst-nb's execution folder is written beside the manual; nothing serves it.
    if (site / "jupyter_execute").is_dir():
        shutil.rmtree(site / "jupyter_execute")
    (site / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://pianistpy.com/docs/sitemap.xml\n"
    )
    # The markdown twins and the full text, in the manual's reading order.
    full, urls = [], []
    for page in order:
        rel = page.relative_to(manual)
        if (
            rel == Path("index.html")
            or rel.as_posix() in tabs
            or rel.parts[0] in ("_static", "_sources", "_modules", "_images")
            or rel.name
            in (
                "genindex.html",
                "search.html",
                "py-modindex.html",
            )
        ):
            continue
        url = f"https://pianistpy.com/docs/{page_address(rel)}"
        title, body = page_to_markdown(page.read_text(encoding="utf-8"), url)
        # The body opens with the page's own title; the twin keeps it once.
        body = body.removeprefix(f"# {title}").lstrip("\n")
        md = f"# {title}\n\nSource: {url}\n\n{body}\n"
        page.with_suffix(".md").write_text(md, encoding="utf-8")
        full.append(md)
        urls.append(url)
    head = (
        f"# PIANISTpy ver. {version}: the manual as one text file\n\n"
        "Every page of https://pianistpy.com/docs/ as markdown, in the manual's "
        "reading order (the order of the tabs and their menus). Each page opens with "
        "its title and its address; a line of three dashes separates the pages.\n"
    )
    (manual / "llms-full.txt").write_text(
        "\n\n---\n\n".join([head, *full]), encoding="utf-8"
    )
    # The sitemap that robots.txt names: the landing page and every page of the
    # manual, at the addresses the links use.
    (manual / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(
            f"  <url><loc>{html.escape(u)}</loc></url>\n"
            for u in ["https://pianistpy.com/", *urls]
        )
        + "</urlset>\n",
        encoding="utf-8",
    )
    # llms.txt at the root.
    lines = [
        "# PIANISTpy",
        "",
        f"> {SUMMARY}",
        "",
        "Every page of the manual has a markdown twin at the same address with",
        "`.md` in place of `.html`; the whole manual as one file is",
        "/docs/llms-full.txt.",
        "",
        "## Documentation",
        "",
    ]
    for listed, raw in PRIMARY:
        path = tabs.get(listed, listed)
        note = raw.replace("@PAPER@", paper["PAPER"])
        lines.append(
            f"- [{note}](https://pianistpy.com/docs/{Path(path).with_suffix('.md').as_posix()})"
        )
    lines += [
        "",
        "## Optional",
        "",
        "- [The package on GitHub](https://github.com/barrytchiang/PIANISTpy)",
        "- [The notebooks on GitHub](https://github.com/barrytchiang/PIANISTpy-notebooks)",
        "- [The full manual as one text file](https://pianistpy.com/docs/llms-full.txt)",
        "",
    ]
    (site / "llms.txt").write_text("\n".join(lines), encoding="utf-8")
    # External links open in a new tab, on the landing page and every manual page;
    # links to a list-only tab's index page open its first page.
    at_root = {f"docs/{k}": f"docs/{v}" for k, v in tabs.items()}
    for page in [site / "index.html", *manual.rglob("*.html")]:
        text = page.read_text(encoding="utf-8")
        if page == site / "index.html":
            tabbed = retarget(text, "index.html", at_root)
        else:
            tabbed = retarget(text, page.relative_to(manual).as_posix(), tabs)
        tabbed = glue_math(external_in_new_tab(tabbed))
        if tabbed != text:
            page.write_text(tabbed, encoding="utf-8")
    print(f"site assembled at {site}: {len(full)} pages twinned")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
