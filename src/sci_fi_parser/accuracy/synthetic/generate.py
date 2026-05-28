"""Generation modes that stream ``(filename, image, label)`` samples.

* :func:`generate_series` -- the default controlled-density experiment.
* :func:`generate_random` -- fully-random charts for variety/volume.
* :func:`generate_preview` -- one small sample per catalog type.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .config import CATALOG, GenConfig
from .render import render_chart
from .style import Style, make_categories, sample_style, series_values

Sample = tuple[str, np.ndarray, dict]   # (filename, RGB image, label record)


def _emit_density(cfg: GenConfig, style: Style, sid: str, d: int, cats: list[str],
                  svals: list[np.ndarray], geo: bool) -> Iterator[Sample]:
    """One density step -> one chart per ``cfg.labels_modes`` entry at every
    resolution (uses no RNG)."""
    for res in cfg.resolutions:
        for mode in cfg.labels_modes:
            labels_on = mode == "on"
            name = f"{sid}_d{d:02d}_{mode}_{res}.png"
            pair = f"{sid}_d{d:02d}_{res}"        # shared by every labels variant
            image, label = render_chart(
                cfg, style, cats, svals, labels_on, geo,
                {"series_id": sid, "pair_id": pair, "density_step": d}, resolution=res)
            yield name, image, label


def _series_charts(rng: np.random.Generator, cfg: GenConfig, alias: str,
                   k: int) -> Iterator[Sample]:
    """One fixed-style series: sample the style once, then vary only the density."""
    style = sample_style(rng, cfg, alias)
    sid = f"{alias}_s{k}"
    for d in cfg.density_steps:
        cats = make_categories(rng, d)
        svals = series_values(rng, style, d)       # shared by the off/on pair
        geo = rng.random() < cfg.geometry_full_prob
        yield from _emit_density(cfg, style, sid, d, cats, svals, geo)


def generate_series(rng: np.random.Generator, cfg: GenConfig) -> Iterator[Sample]:
    """Default mode: one controlled density series per selected type."""
    for alias in cfg.output_types:
        for k in range(cfg.per_type):
            yield from _series_charts(rng, cfg, alias, k)


def generate_random(rng: np.random.Generator, cfg: GenConfig,
                    n_charts: int) -> Iterator[Sample]:
    """N fully-random charts (variety/volume), each at a random resolution."""
    aliases = list(cfg.output_types) or list(CATALOG)
    for i in range(n_charts):
        style = sample_style(rng, cfg, str(rng.choice(aliases)))
        d = int(rng.integers(cfg.density_steps[0], cfg.density_steps[-1] + 1))
        cats = make_categories(rng, d)
        svals = series_values(rng, style, d)
        res = int(rng.choice(cfg.resolutions))
        image, label = render_chart(
            cfg, style, cats, svals,
            rng.random() < cfg.value_labels_prob,
            rng.random() < cfg.geometry_full_prob,
            {"series_id": f"random_{i}", "pair_id": None, "density_step": None},
            resolution=res)
        yield f"random_{i:05d}_{res}.png", image, label


def generate_preview(cfg: GenConfig) -> Iterator[Sample]:
    """One small sample per catalog type -> ``(alias.png, image, label)``."""
    rng = np.random.default_rng(7)
    for alias in CATALOG:
        style = sample_style(rng, cfg, alias)
        style.w_in, style.h_in, style.dpi = 3.4, 2.6, 90   # small thumbnail
        cats = make_categories(rng, 8)
        svals = series_values(rng, style, 8)
        image, label = render_chart(
            cfg, style, cats, svals, False, False,
            {"series_id": alias, "pair_id": None, "density_step": None})
        yield f"{alias}.png", image, label
