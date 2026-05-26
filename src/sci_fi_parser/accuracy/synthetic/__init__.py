"""Synthetic chart generator with pixel-accurate ground-truth labels.

Why this exists
---------------
Real chart labels in ``train_data/dataset.sqlite3`` are model-generated (Gemma ->
Haiku -> Opus), so they carry noise and a circularity risk when benchmarking
Claude/Gemma-family models. Charts rendered here are labelled *by construction*.

Catalog of types (pick via ``output_types`` in the TOML)
-------------------------------------------------------
    simple      single-series vertical bars, one colour
    multicolor  single-series vertical bars, colormap across bars
    grouped     multi-series bars side-by-side
    stacked     multi-series bars stacked
    horizontal  single-series horizontal bars
    line        single line
    2-line      two lines        3-line  three lines
    multiline   2-4 lines

Modes
-----
* (default) controlled density series per type: fix the style, vary only the bar
  count across ``density_steps``; each density is a matched labels-OFF/ON pair.
* random: N fully-random charts.        * preview: one small sample per type.

Modules: :mod:`config` (knobs + catalog), :mod:`style` (sampling), :mod:`render`
(one chart -> image + label), :mod:`generate` (the modes), :mod:`output`
(augment/overlay/PNG), :mod:`cli` (command line).
"""

from . import _backend  # noqa: F401  set the headless matplotlib backend first
from .config import CATALOG, GenConfig, load_config
from .generate import (Sample, generate_preview, generate_random,
                       generate_series)
from .render import render_chart
from .style import Style, make_categories, sample_style, series_values

__all__ = [
    "CATALOG", "GenConfig", "load_config",
    "Style", "sample_style", "make_categories", "series_values",
    "render_chart",
    "Sample", "generate_series", "generate_random", "generate_preview",
]
