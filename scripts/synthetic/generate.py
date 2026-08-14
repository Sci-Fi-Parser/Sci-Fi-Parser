"""Generation modes that stream ``(filename, image, truth_json, metadata)`` samples.

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

Sample = tuple[str, np.ndarray, dict, dict]


def _emit_density(
    cfg: GenConfig, style: Style, sid: str, d: int, cats: list[str], svals: list[np.ndarray], geo: bool
) -> Iterator[Sample]:
    """One density step -> a labels-off/on chart at every resolution (uses no RNG)."""
    for res in cfg.resolutions:
        for labels_on in (False, True):
            onoff = "on" if labels_on else "off"
            name = f"{sid}_d{d:02d}_{onoff}_{res}.png"
            pair = f"{sid}_d{d:02d}_{res}"  # links the off/on twin at this res
            image, truth, metadata = render_chart(
                cfg,
                style,
                cats,
                svals,
                labels_on,
                geo,
                {"series_id": sid, "pair_id": pair, "density_step": d},
                resolution=res,
            )
            yield name, image, truth, metadata


def _series_charts(rng: np.random.Generator, cfg: GenConfig, alias: str, k: int) -> Iterator[Sample]:
    """One fixed-style series: sample the style once, then vary only the density."""
    style = sample_style(rng, cfg, alias)
    sid = f"{alias}_s{k}"
    for d in cfg.density_steps:
        cats = make_categories(rng, d)
        svals = series_values(rng, style, d)  # shared by the off/on pair
        geo = rng.random() < cfg.geometry_full_prob
        yield from _emit_density(cfg, style, sid, d, cats, svals, geo)


def generate_series(rng: np.random.Generator, cfg: GenConfig) -> Iterator[Sample]:
    """Default mode: one controlled density series per selected type."""
    for alias in cfg.output_types:
        for k in range(cfg.per_type):
            yield from _series_charts(rng, cfg, alias, k)


def generate_random(rng: np.random.Generator, cfg: GenConfig, n_charts: int) -> Iterator[Sample]:
    """N fully-random charts (variety/volume), each at a random resolution."""
    aliases = list(cfg.output_types) or list(CATALOG)
    for i in range(n_charts):
        style = sample_style(rng, cfg, str(rng.choice(aliases)))
        d = int(rng.integers(cfg.density_steps[0], cfg.density_steps[-1] + 1))
        cats = make_categories(rng, d)
        svals = series_values(rng, style, d)
        res = int(rng.choice(cfg.resolutions))
        image, truth, metadata = render_chart(
            cfg,
            style,
            cats,
            svals,
            rng.random() < cfg.value_labels_prob,
            rng.random() < cfg.geometry_full_prob,
            {"series_id": f"random_{i}", "pair_id": None, "density_step": None},
            resolution=res,
        )
        yield f"random_{i:05d}_{res}.png", image, truth, metadata


def generate_preview(cfg: GenConfig) -> Iterator[Sample]:
    """One small sample per catalog type."""
    rng = np.random.default_rng(7)
    for alias in CATALOG:
        style = sample_style(rng, cfg, alias)
        style.w_in, style.h_in, style.dpi = 3.4, 2.6, 90  # small thumbnail
        cats = make_categories(rng, 8)
        svals = series_values(rng, style, 8)
        image, truth, metadata = render_chart(
            cfg, style, cats, svals, False, False, {"series_id": alias, "pair_id": None, "density_step": None}
        )
        yield f"{alias}.png", image, truth, metadata


def generate_axis_challenges(rng: np.random.Generator, cfg: GenConfig) -> Iterator[Sample]:
    """Generate controlled pairs where one Y-axis factor changes at a time."""
    style = sample_style(rng, cfg, "simple")
    style.grid = True
    style.value_lo = 0.0
    cats = make_categories(rng, 6)
    values = series_values(rng, style, 6)
    baseline = {
        "axis_visible": True,
        "axis_width": 1.5,
        "axis_color": "black",
        "tick_marks": True,
        "numeric_labels": True,
        "grid": True,
        "grid_color": "0.8",
        "bar_distance": "normal",
        "image_quality": "clean",
    }
    variants = [
        ("axis_absent", {"axis_visible": False}),
        ("axis_05px", {"axis_width": 0.5}),
        ("axis_3px", {"axis_width": 3.0}),
        ("axis_5px", {"axis_width": 5.0}),
        ("axis_medium_gray", {"axis_color": "0.5"}),
        ("axis_light_gray", {"axis_color": "0.8"}),
        ("ticks_off", {"tick_marks": False}),
        ("numeric_labels_off", {"numeric_labels": False}),
        ("grid_off", {"grid": False}),
        ("grid_dark", {"grid_color": "0.25"}),
        ("bars_near_axis", {"bar_distance": "near"}),
        ("jpeg", {"image_quality": "jpeg"}),
        ("blur", {"image_quality": "blur"}),
        ("internal_vertical_rule", {"vertical_rule": True}),
    ]
    for pair_index, (name, change) in enumerate(variants):
        for side, settings in (("base", baseline), ("changed", {**baseline, **change})):
            challenge = dict(settings)
            metadata = {
                "series_id": "axis_challenge",
                "pair_id": f"axis_{pair_index:02d}_{name}",
                "density_step": None,
                "challenge": name,
                "challenge_side": side,
                "image_quality": challenge["image_quality"],
                "axis_factors": challenge,
            }
            image, truth, report_metadata = render_chart(
                cfg,
                style,
                cats,
                values,
                False,
                True,
                metadata,
                resolution=cfg.resolutions[0],
                axis_challenge=challenge,
            )
            yield f"axis_{pair_index:02d}_{name}_{side}.png", image, truth, report_metadata
