"""Sphinx configuration of the PIANISTpy manual (pianistpy.com/docs/)."""

import os
import sys

# The package is documented from this tree, not from an installed copy.
sys.path.insert(0, os.path.abspath(".."))
# Importing the package for its docstrings needs no accelerator.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pianistpy import __version__  # noqa: E402

# The settings tables of the Reference tab are written from the package's
# declared fields before each build (docs/tools/settings_tables.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
import settings_tables  # noqa: E402

settings_tables.write(os.path.join(os.path.dirname(__file__), "reference", "_settings_tables.rst"))

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

# The package's docstrings are in the numpydoc format.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# An "Attributes" section is rendered as variables, so that a dataclass field
# is not described twice.
napoleon_use_ivar = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

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
]
templates_path = ["_templates"]

# The theme: pydata-sphinx-theme with the site's palette and fonts in
# _static/css/custom.css (shared with the landing page).
html_theme = "pydata_sphinx_theme"
# No image in the header band (the word alone); the logo sits in the sidebar.
html_logo = None
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["css/fonts.css", "css/custom.css"]
html_title = f"PIANISTpy {release}"
html_short_title = "PIANISTpy"
html_show_sourcelink = False
html_context = {"pre_release": PRE_RELEASE, "default_mode": "light"}

html_theme_options = {
    "logo": {"text": "PIANISTpy", "alt_text": "PIANISTpy"},
    "header_links_before_dropdown": 6,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
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
if PRE_RELEASE:
    html_theme_options["announcement"] = (
        "Pre-release: version 0.9.0 is a preview of the package and of this manual; "
        "version 1.0.0 is released with the paper that describes the method."
    )

# The manual's home has no section sidebar; every other page keeps its tab's,
# with the logo at the sidebar's full width above it (it goes with the sidebar
# when that is collapsed; the small logo stays in the header band).
html_sidebars = {"index": [], "**": ["sidebar-logo.html", "sidebar-nav-bs.html"]}
