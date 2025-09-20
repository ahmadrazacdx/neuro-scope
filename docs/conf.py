# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib import metadata

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Add the project root (src) to the path so autodoc can find your modules
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "neuroscope"
author = "Ahmad Raza"
copyright = "2025, Ahmad Raza"


# The short X.Y version
version = metadata.version("neuroscope").split("+")[0]  # e.g., 1.0.0
# The full version, including alpha/beta/rc tags
release = metadata.version("neuroscope")  # e.g., 1.0.0.dev1+gabc1234

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.todo",  # Support for todo items (useful during dev)
    "sphinx.ext.mathjax",  # Render math via MathJax
    # Markdown support
    "myst_parser",
    # Third-party extensions for modern features
    "sphinx_copybutton",  # Add copy buttons to code blocks
    "sphinx_design",  # Bootstrap-style components (cards, tabs, grids)
    "sphinxext.opengraph",  # Generate OpenGraph metadata for social previews
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for Extensions -------------------------------------------------

# -- Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",  # Order members by source order
    "special-members": "__init__",  # Include __init__ docstrings
    "undoc-members": True,  # Show undocumented members
    "exclude-members": "__weakref__",  # Exclude specific members
    "show-inheritance": True,  # Show class inheritance
}
autosummary_generate = True  # Automatically generate summary pages
autoclass_content = (
    "both"  # Include both class docstring and __init__ docstring in class description
)

# -- Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Napoleon settings (Google/NumPy docstrings)
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
napoleon_preprocess_types = True  # More accurate type hints
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MyST Parser settings
myst_enable_extensions = [
    "amsmath",  # LaTeX-like math
    "colon_fence",  # ::: blocks for directives
    "deflist",  # Definition lists
    "dollarmath",  # $...$ and $$...$$ math
    "fieldlist",  # :key: value lists
    "html_admonition",  # HTML-style admonitions
    "html_image",  # HTML <img> tags
    "linkify",  # Auto-linkify URLs and email addresses (requires linkify-it-py)
    "replacements",  # Simple text replacements
    "smartquotes",  # Typographic quotes
    "strikethrough",  # ~~strikethrough~~
    "substitution",  # Variable substitution
    "tasklist",  # GitHub-style task lists
]
# Configure MyST heading anchors if needed (e.g., for internal linking)
myst_heading_anchors = 3  # Generate anchors for h1, h2, h3

# -- Todo extension
todo_include_todos = False

# -- OpenGraph settings
ogp_site_url = "https://ahmadrazacdx.github.io/neuroscope/"
ogp_image = "_static/logo.png"
# ogp_description_length = 200 # Optional: Limit description length
# ogp_type = "website" # Optional: Set og:type

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-foreground": "#000000",
        "color-background": "#ffffff",
        "color-border": "#dddddd",
        "color-accent": "#ff0000",
        "color-link": "#1a0dab",
        "color-link--hover": "#551a8b",
    },
    "dark_css_variables": {
        "color-foreground": "#ffffff",
        "color-background": "#000000",
        "color-border": "#333333",
        "color-accent": "#ff0000",
        "color-link": "#ff0000",
        "color-link--hover": "#ff5555",
    },
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Options for HTML Help output --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-help-output

htmlhelp_basename = "neuroscopedoc"

# -- Options for LaTeX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
root_doc = "index"  # Standard since Sphinx 2.0
latex_documents = [
    (root_doc, "Neuroscope.tex", "Neuroscope Documentation", "Ahmad Raza", "manual"),
]

# -- Options for manual page output ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-manual-page-output

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "neuroscope", "Neuroscope Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-texinfo-output

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "Neuroscope",
        "Neuroscope Documentation",
        author,
        "Neuroscope",
        "A  microscope for informed training of multi-layer perceptron, diagnosing training issues at granular level and accelerating learning and rapid prototyping.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-epub-output

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Root Doc (Master Doc) ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-root_doc
root_doc = "index"  # Standard since Sphinx 2.0

# -- Ensure metadata.version works ------------------------------------------
# Make sure your package is installed in the environment where Sphinx runs,
# or the importlib.metadata call will fail.
try:
    release = metadata.version("neuroscope")
    version = release.split("+")[0]
except metadata.PackageNotFoundError:
    print("Warning: Package 'neuroscope' not found. Using placeholder version.")
    release = "0.1.0.dev"
    version = "0.1.0"
