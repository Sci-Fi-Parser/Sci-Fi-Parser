"""Sphinx configuration for Sci-Fi-Parser."""

from importlib.metadata import version as _pkg_version

project = "Sci-Fi-Parser"
copyright = "2026, Sci-Fi-Parser"
author = "Sci-Fi-Parser"
release = _pkg_version("sci-fi-parser")
version = release

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
