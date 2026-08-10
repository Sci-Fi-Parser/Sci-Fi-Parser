"""Sphinx configuration for Sci-Fi-Parser."""

from importlib.metadata import version

project = "Sci-Fi-Parser"
copyright = "2026, Sci-Fi-Parser"
author = "Sci-Fi-Parser"
release = version("sci-fi-parser")

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
