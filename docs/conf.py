"""Sphinx configuration of the lenscluster documentation (Markdown through MyST).  The API pages use autodoc with the
heavy dependencies mocked: the package itself, with JAX and the lenstronomy forks, is not installed on Read the Docs."""

import os
import sys

ROOT = os.path.abspath("..")
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = "lenscluster"
author = "Isaque Dutra"
copyright = "2026, Isaque Dutra"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: $...$ and $$...$$ math, ::: fenced directives, heading anchors for cross-references.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}
autodoc_mock_imports = [
    "astropy",
    "blackjax",
    "cmcrameri",
    "contourpy",
    "corner",
    "graphviz",
    "jax",
    "jax_cosmo",
    "jaxlib",
    "jaxtronomy",
    "lenstronomy",
    "matplotlib",
    "numpyro",
    "optax",
    "pandas",
    "photutils",
    "rich",
    "scipy",
    "skimage",
    "threadpoolctl",
    "tqdm",
    "yaml",
]

html_theme = "sphinx_book_theme"
html_title = "lenscluster Documentation"
html_static_path = []
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dutra/lenscluster",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
