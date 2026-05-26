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
