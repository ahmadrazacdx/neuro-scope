"""Sphinx configuration."""

project = "neuroscope"
author = "Ahmad Raza"
copyright = "2025, Ahmad Raza"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
