"""Generator configuration: the chart-type catalog and tunable knobs.

Every :class:`GenConfig` field is overridable from a TOML file via :func:`load_config`
(see ``config/synthetic_bars.toml``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

# alias -> preset. ``series``: int, or "multi" (count sampled from cfg.n_series).
# color: "single" (one colour), "colormap" (gradient across bars), "series" (per series).
CATALOG: dict[str, dict] = {
    "simple": {"chart_type": "vertical_bar", "family": "bar", "orientation": "v",
               "stacked": False, "color": "single", "series": 1},
    "multicolor": {"chart_type": "vertical_bar", "family": "bar", "orientation": "v",
                    "stacked": False, "color": "colormap", "series": 1},
    "grouped": {"chart_type": "grouped_bar", "family": "bar", "orientation": "v",
                 "stacked": False, "color": "series", "series": "multi"},
    "stacked": {"chart_type": "stacked_bar", "family": "bar", "orientation": "v",
                 "stacked": True, "color": "series", "series": "multi"},
    "horizontal": {"chart_type": "horizontal_bar", "family": "bar",
                   "orientation": "h", "stacked": False, "color": "single", "series": 1},
    "line": {"chart_type": "line", "family": "line", "orientation": "v",
              "stacked": False, "color": "series", "series": 1},
    "2-line": {"chart_type": "line", "family": "line", "orientation": "v",
                "stacked": False, "color": "series", "series": 2},
    "3-line": {"chart_type": "line", "family": "line", "orientation": "v",
                "stacked": False, "color": "series", "series": 3},
    "multiline": {"chart_type": "line", "family": "line", "orientation": "v",
                  "stacked": False, "color": "series", "series": "multi"},
}


@dataclass(slots=True)
class GenConfig:
    """Knobs for the generator. Override any field from a TOML file via ``--config``.

    In the default density-series mode the *style* (colours, figure, DPI, value
    range) is sampled ONCE per series and held fixed; only density varies.
    """

    # Which catalog types to generate, by alias (see CATALOG). Each gets its own
    # fixed-style density series.
    output_types: tuple[str, ...] = ("simple",)

    # The density ladder (bars/points per step). Each value -> one off + one on chart.
    density_steps: tuple[int, ...] = (4, 5, 7, 9, 10, 12, 15, 20, 25, 30)

    # Independent fixed-style series PER selected type. 1 -> 20 images/type.
    per_type: int = 1

    # (min, max) INCLUSIVE series for "multi" presets (grouped/stacked/multiline).
    n_series: tuple[int, int] = (2, 4)

    # Fraction [0..1] of charts emitting full geometry (else values only).
    geometry_full_prob: float = 0.5

    # Figure size (inches), sampled once per series; pixel size comes from resolution.
    fig_w_in: tuple[float, float] = (5.0, 9.0)
    fig_h_in: tuple[float, float] = (3.5, 5.5)
    dpi_choices: tuple[int, ...] = (90, 100, 120, 150)  # fallback dpi (preview/no-res)

    # Output image HEIGHTS in pixels (aspect ratio preserved). Each chart is
    # rendered at every resolution and the value is appended to the filename
    # (e.g. simple_s0_d25_on_240). Lower = harder for extractors.
    resolutions: tuple[int, ...] = (480,)

    # Bar width as a category-unit fraction (0..1); fixed per series so bars thin
    # as density rises.
    bar_width: tuple[float, float] = (0.6, 0.85)

    allow_negative: float = 0.15
    grid_prob: float = 0.5
    title_prob: float = 0.7

    # Category-label fitting: constant rotation; tick font shrinks with density so
    # labels always fit without overlapping.
    label_rotation: int = 90
    tick_fontsize: tuple[int, int] = (6, 11)

    # --random mode only: probability a random chart prints value labels.
    value_labels_prob: float = 0.0


_TUPLE_FIELDS = frozenset({
    "output_types", "density_steps", "n_series", "fig_w_in", "fig_h_in",
    "dpi_choices", "bar_width", "tick_fontsize", "resolutions",
})


def load_config(path: Path) -> GenConfig:
    """Build a :class:`GenConfig` from a TOML file, validating keys and types."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = GenConfig()
    valid = {f.name for f in fields(cfg)}
    for key, value in raw.items():
        if key not in valid:
            raise SystemExit(
                f"unknown config key {key!r} in {path}; valid: {sorted(valid)}")
        if key in _TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    for alias in cfg.output_types:
        if alias not in CATALOG:
            raise SystemExit(f"unknown output type {alias!r}; valid: {sorted(CATALOG)}")
    return cfg
