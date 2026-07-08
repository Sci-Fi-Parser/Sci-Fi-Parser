"""Sphinx configuration for Sci-Fi-Parser."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
    "undoc-members": False,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["generated"]

html_theme = "alabaster"
html_static_path = ["_static"]
