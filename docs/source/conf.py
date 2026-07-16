"""Sphinx configuration for Sci-Fi-Parser."""

project = "Sci-Fi-Parser"
copyright = "2026, Sci-Fi-Parser"
author = "Sci-Fi-Parser"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
