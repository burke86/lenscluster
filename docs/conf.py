"""Sphinx configuration of the lenscluster documentation (Markdown through MyST; no autodoc, the package is not
installed on Read the Docs)."""

project = "lenscluster"
author = "Isaque Dutra"
copyright = "2026, Isaque Dutra"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: $...$ and $$...$$ math, ::: fenced directives, heading anchors for cross-references.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "lenscluster"
html_static_path = []
