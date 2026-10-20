"""Sphinx configuration of the PIANISTpy manual (pianistpy.com/docs/)."""

import os
import sys

# The package is documented from this tree, not from an installed copy.
sys.path.insert(0, os.path.abspath(".."))
# Importing the package for its docstrings needs no GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pianistpy import __version__  # noqa: E402

# The settings tables of the API reference tab are written from the package's
# declared fields before each build (docs/tools/settings_tables.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
import settings_tables  # noqa: E402

settings_tables.write(
    os.path.join(os.path.dirname(__file__), "reference", "_settings_tables.rst")
)
# The accuracy tables, from each release's accuracy grid
# (docs/tools/accuracy_tables.py).
import accuracy_tables  # noqa: E402

accuracy_tables.write(os.path.dirname(__file__))

project = "PIANISTpy"
author = "Barry T. Chiang"
copyright = "2026, Barry T. Chiang"  # noqa: A001 (the name Sphinx reads)
version = __version__
release = __version__

# Until 1.0.0 the site says so on every page and asks search engines to skip it.
PRE_RELEASE = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "myst_nb",
]
# The preview of the next release, built from the development tree before that
# tree is public: PIANISTPY_DOCS_PREVIEW=1 leaves out the pages of the source code
# (viewcode's _modules), so the preview publishes the manual and not the package.
if os.environ.get("PIANISTPY_DOCS_PREVIEW") == "1":
    extensions.remove("sphinx.ext.viewcode")

# The package's docstrings are in the numpydoc format.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# An "Attributes" section is rendered as variables, so that a dataclass field
# is not described twice.
napoleon_use_ivar = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Defaults as written in the source (``xp=np``), not their repr, which for a module
# would print the path of the machine that built the manual.
autodoc_preserve_defaults = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# The notebooks are stored with their outputs, run on a compute node against
# the release: the build renders them and runs nothing.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "colon_fence"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "landing",
    "tools",
    "requirements.txt",
    # Included by reference/settings.rst, not a page of its own.
    "reference/_settings_tables.rst",
    # Written by docs/tools/accuracy_tables.py and included, like the settings tables.
    "reference/_accuracy_tables.rst",
    "project/_accuracy_releases.rst",
    "_extra",
]
templates_path = ["_templates"]

# The theme: pydata-sphinx-theme with the site's palette and fonts in
# _static/css/custom.css (shared with the landing page).
html_theme = "pydata_sphinx_theme"
# No image in the header band (the word alone); the logo sits in the sidebar.
html_logo = None
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["css/fonts.css", "css/custom.css", "css/band.css"]
# A displayed equation wider than the column scrolls sideways (MathJax 4 would otherwise
# break or shrink it); on a phone the technical notes' equations need it.
mathjax4_config = {"output": {"displayOverflow": "scroll"}}
html_title = f"PIANISTpy ver. {release}"
html_short_title = "PIANISTpy"
html_show_sourcelink = False
# Links between pages carry no ".html": GitHub Pages serves page.html at /page, and
# the files keep their names, so an address with ".html" still works.
html_link_suffix = ""
html_context = {"pre_release": PRE_RELEASE, "default_mode": "light"}

html_theme_options = {
    "logo": {"text": "PIANISTpy", "alt_text": "PIANISTpy"},
    "header_links_before_dropdown": 6,
    # The header is the site band (_templates/site-band.html), which replaces the
    # theme's navbar; empty here so the narrow screen's sidebar does not repeat the
    # tabs.
    "navbar_start": [],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": [],
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": False,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "native",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/barrytchiang/PIANISTpy",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pianistpy/",
            "icon": "fa-brands fa-python",
        },
    ],
}
# The release paper, cited the same way on every page, always as a link to PAPER_URL.
# The citation page and the landing page give the full form once, "PAPER_FULL, PAPER_STATUS
# (henceforth PAPER)"; every other mention is the short form. Pages write |paper| (the
# linked short form), |paper-full| (the linked full form and its status) or |paper-name|
# (the short form as plain text, for headings); the landing page's @PAPER...@ fields are
# set by assemble_site from these lines. Until the arXiv entry exists the link is the
# citation page.
PAPER = "Chiang+(2026d)"
PAPER_FULL = "Chiang, van den Bosch and Schive (2026d), MNRAS"
PAPER_STATUS = "submitted"
PAPER_URL = "https://pianistpy.com/docs/project/citing"
rst_epilog = (
    f".. |paper| replace:: `{PAPER} <{PAPER_URL}>`__\n"
    f".. |paper-full| replace:: `{PAPER_FULL} <{PAPER_URL}>`__, {PAPER_STATUS}\n"
    f".. |paper-name| replace:: {PAPER}\n"
)

if PRE_RELEASE:
    html_theme_options["announcement"] = (
        "Pre-release: ver. 0.9.9 is the code of "
        f'<a href="{PAPER_URL}">{PAPER}</a>; ver. 1.0.0 follows with the citation entry, the DOI of the software release and the changes its release notes list.'
    )

# The manual's home has no section sidebar; every other page keeps its tab's,
# with the logo at the sidebar's full width above it (it goes with the sidebar
# when that is collapsed; the small logo stays in the header band).
html_sidebars = {"index": [], "**": ["sidebar-logo.html", "sidebar-nav-bs.html"]}
