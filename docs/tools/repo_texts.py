"""Write the repository's README.md and CONTRIBUTING.md from the website, so that the
website is the one source of their text.

    python docs/tools/repo_texts.py <site> <repo> [--check]

``<site>`` is the folder that assemble_site.py wrote: the landing page at its root and
the manual under ``docs/``, with a markdown twin beside every page. ``<repo>`` is the
repository's root. Each section of the README is copied from one page of the website,
as SECTIONS below lists; only the logo, the badges and the section headings are kept
here. The README is also the project's page on PyPI. CONTRIBUTING.md is the
Contributing page in full. With ``--check`` nothing is written, and the exit status is
1 if either file differs from the text this script would write.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from assemble_site import page_to_markdown  # noqa: E402

SITE = "https://pianistpy.com"
NOTE = (
    "<!-- Written by docs/tools/repo_texts.py from {source}. "
    "Edit the website, not this file. -->\n\n"
)
LOGO = "https://raw.githubusercontent.com/barrytchiang/PIANISTpy/main/docs/_static/logo.png"
HEADER = f"""<p align="center">
<img src="{LOGO}" alt="PIANISTpy" width="320">
</p>

# PIANISTpy

[![Tests](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml/badge.svg)](https://github.com/barrytchiang/PIANISTpy/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/barrytchiang/PIANISTpy/branch/main/graph/badge.svg)](https://codecov.io/gh/barrytchiang/PIANISTpy)
[![Documentation](https://img.shields.io/badge/docs-pianistpy.com-00356B)](https://pianistpy.com)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/barrytchiang/PIANISTpy/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pianistpy)](https://pypi.org/project/pianistpy/)
[![Python](https://img.shields.io/pypi/pyversions/pianistpy)](https://pypi.org/project/pianistpy/)

**P**article **I**nitial conditions with **ANIS**o**T**ropy.
"""
# The pages the Documentation section links, in the order of the website's tabs.
DOC_PAGES = [
    "start/overview",
    "guide/index",
    "notebooks/index",
    "reference/api/index",
    "project/release-notes",
]


def twin(site: Path, page: str) -> tuple[str, str, str]:
    """The title, the address and the body of one page's markdown twin."""
    text = (site / "docs" / f"{page}.md").read_text(encoding="utf-8")
    title = text.split("\n", 1)[0].removeprefix("# ").strip()
    url = re.search(r"^Source: (\S+)$", text, re.M).group(1)
    body = text.split(f"Source: {url}", 1)[1].strip()
    return title, url, body


def landing(site: Path, pattern: str) -> str:
    """The markdown of the first element of the landing page that ``pattern`` finds."""
    found = re.search(pattern, (site / "index.html").read_text(encoding="utf-8"), re.S)
    return page_to_markdown(f"<article>{found.group(0)}</article>", SITE + "/")[1]


def code_blocks(body: str) -> list[str]:
    return re.findall(r"```[a-z0-9]*\n.*?\n```", body, re.S)


def readme(site: Path) -> str:
    lead = landing(site, r'<p class="lead">.*?</p>')
    release = landing(site, r'<section id="releases">.*?<p>.*?</p>')
    release = re.sub(r"^#.*\n+", "", release).strip()
    status = re.split(r"(?<=\.)\s+(?=[A-Z])", release, maxsplit=1)[0]
    _, overview_url, features = twin(site, "start/overview")
    install_title, install_url, install = twin(site, "start/install")
    quick_title, quick_url, quick = twin(site, "start/quickstart")
    _, _, cite = twin(site, "project/citing")
    _, _, license_ = twin(site, "project/license")
    docs = "\n".join(
        f"- [{title}]({url})" for title, url, _ in (twin(site, p) for p in DOC_PAGES)
    )
    return (
        NOTE.format(source=SITE)
        + HEADER
        + f"\n{lead}\n\n{status}\n\n"
        + f"## What PIANISTpy does\n\n{features}\n\n"
        + "## Getting started\n\n"
        + f"{code_blocks(install)[0]}\n\n{code_blocks(quick)[0]}\n\n"
        + f"[{install_title}]({install_url}) · [{quick_title}]({quick_url})\n\n"
        + f"## Documentation\n\n{docs}\n\n"
        + f"## How to cite\n\n{cite}\n\n"
        + f"## License\n\n{license_}\n"
    )


def contributing(site: Path) -> str:
    title, url, body = twin(site, "project/contributing")
    return NOTE.format(source=url) + f"# {title} to PIANISTpy\n\n{body}\n"


def main(site_dir: str, repo_dir: str, *flags: str) -> int:
    site, repo = Path(site_dir), Path(repo_dir)
    texts = {"README.md": readme(site), "CONTRIBUTING.md": contributing(site)}
    stale = [
        name
        for name, text in texts.items()
        if not (repo / name).exists()
        or (repo / name).read_text(encoding="utf-8") != text
    ]
    if "--check" in flags:
        for name in stale:
            print(f"repo_texts: {name} differs from the website; rerun repo_texts.py")
        return 1 if stale else 0
    for name in stale:
        (repo / name).write_text(texts[name], encoding="utf-8", newline="\n")
        print(f"repo_texts: wrote {name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
