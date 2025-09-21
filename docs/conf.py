"""Sphinx configuration file for NeuroScope documentation."""

import os
import sys
from importlib import metadata

# Path setup
sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "NeuroScope"
author = "Ahmad Raza"
copyright = "2025, Ahmad Raza"

# Version information
try:
    release = metadata.version("neuroscope")
    version = release.split("+")[0]
except metadata.PackageNotFoundError:
    release = "0.1.0.dev"
    version = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxext.opengraph",
]

# General configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autosummary_generate = True
autoclass_content = "both"

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST Parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# MathJax configuration (optimized for MyST parser)
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    },
}

# Let MyST parser handle MathJax class configuration automatically
myst_update_mathjax = True

# Site configuration
site_url = "https://ahmadrazacdx.github.io/neuroscope/"

# HTML output configuration
html_theme = "furo"
html_title = f"{project} {release}"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_templates_path = ["_templates"]

# Ensure templates are found
templates_path = ["_templates"]

# Theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/ahmadrazacdx/neuro-scope/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-foreground": "#1a1a1a",
        "color-background": "#ffffff",
        "color-background-secondary": "#f8fafc",
        "color-background-hover": "#f1f5f9",
        "color-background-border": "#e2e8f0",
        "color-sidebar-background": "#ffffff",
        "color-sidebar-background-border": "#e2e8f0",
        "color-brand-primary": "#0f172a",
        "color-brand-content": "#334155",
        "color-accent": "#3b82f6",
        "color-accent-2": "#1e40af",
        "color-link": "#2563eb",
        "color-link--hover": "#1d4ed8",
        "color-inline-code-background": "#f1f5f9",
        "color-highlighted-background": "#fef3c7",
        "color-highlighted-text": "#92400e",
        "color-admonition-background": "#f8fafc",
    },
    "dark_css_variables": {
        "color-foreground": "#e2e8f0",
        "color-background": "#0f172a",
        "color-background-secondary": "#1e293b",
        "color-background-hover": "#334155",
        "color-background-border": "#475569",
        "color-sidebar-background": "#1e293b",
        "color-sidebar-background-border": "#334155",
        "color-brand-primary": "#f1f5f9",
        "color-brand-content": "#cbd5e1",
        "color-accent": "#60a5fa",
        "color-accent-2": "#3b82f6",
        "color-link": "#60a5fa",
        "color-link--hover": "#93c5fd",
        "color-inline-code-background": "#334155",
        "color-highlighted-background": "#374151",
        "color-highlighted-text": "#fbbf24",
        "color-admonition-background": "#1e293b",
    },
}

# Output format configurations
htmlhelp_basename = "neuroscopedoc"

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "figure_align": "htbp",
}

latex_documents = [
    (root_doc, "NeuroScope.tex", "NeuroScope Documentation", author, "manual"),
]

man_pages = [(root_doc, "neuroscope", "NeuroScope Documentation", [author], 1)]

texinfo_documents = [
    (
        root_doc,
        "NeuroScope",
        "NeuroScope Documentation",
        author,
        "NeuroScope",
        "A microscope for neural networks - comprehensive framework for building, training, and diagnosing multi-layer perceptrons.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]
