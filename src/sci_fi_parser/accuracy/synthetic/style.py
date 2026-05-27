"""The per-series ``Style`` (held constant within a density series) and the
random sampling of styles, category labels, and values.

All randomness flows through a single ``numpy`` ``Generator`` so a fixed ``--seed``
reproduces a dataset byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .config import CATALOG, GenConfig

# Small vocabularies so category/axis/series text varies like real charts.
_CATEGORY_STYLES = ("letters", "words", "quarters", "years", "numbers")
_WORDS = ("North", "South", "East", "West", "Alpha", "Beta", "Gamma", "Delta",
          "Retail", "Online", "Energy", "Banking", "Pharma", "Tech", "EU", "US",
          "UK", "Japan", "Brazil", "India", "China", "Canada")
_SERIES_NAMES = ("2021", "2022", "2023", "2024", "Actual", "Forecast",
                 "Domestic", "Export", "Group A", "Group B", "Group C")
_Y_AXES = (("Revenue", "$M"), ("Count", None), ("Value", None), ("Share", "%"),
           ("Sales", "$"), ("Population", None), ("Score", None), ("Profit", "$B"))
_TITLES = ("Quarterly Results", "Regional Breakdown", "Performance by Group",
           "Annual Summary", "Comparison", "Distribution", None)
_CMAPS = ("viridis", "plasma", "tab10", "Set2", "cividis", "coolwarm")


@dataclass(slots=True)
class Style:
    """Everything held fixed within one density series (only the bar count varies)."""

    alias: str
    chart_type: str
    family: str
    orientation: str
    stacked: bool
    color_mode: str            # "single" | "colormap" | "series"
    base_color: tuple
    cmap_name: str
    series_colors: list
    n_series: int
    series_names: list
    y_name: str
    y_unit: str | None
    title: str | None
    w_in: float
    h_in: float
    dpi: int
    bar_w: float
    value_lo: float
    value_hi: float
    rotation: int
    grid: bool

    @property
    def horizontal(self) -> bool:
        return self.orientation == "h"


def sample_style(rng: np.random.Generator, cfg: GenConfig, alias: str) -> Style:
    """Sample a fixed style for a catalog ``alias`` (style is reused across densities)."""
    p = CATALOG[alias]
    multi = p["series"] == "multi"
    n_ser = (int(rng.integers(cfg.n_series[0], cfg.n_series[1] + 1))
             if multi else int(p["series"]))
    y_name, y_unit = _Y_AXES[int(rng.integers(len(_Y_AXES)))]
    if n_ser > 1:
        names = [str(x) for x in rng.choice(_SERIES_NAMES, size=n_ser, replace=False)]
    else:
        names = [y_name]
    cmap10 = plt.get_cmap("tab10")
    series_colors = [cmap10(s % 10) for s in range(n_ser)]
    scale = float(rng.choice([10.0, 100.0, 1_000.0, 1e6]))
    lo = -0.4 * scale if (not p["stacked"] and rng.random() < cfg.allow_negative) else 0.0
    return Style(
        alias=alias, chart_type=p["chart_type"], family=p["family"],
        orientation=p["orientation"], stacked=p["stacked"], color_mode=p["color"],
        base_color=tuple(rng.random(3) * 0.7 + 0.1), cmap_name=str(rng.choice(_CMAPS)),
        series_colors=series_colors, n_series=n_ser, series_names=names,
        y_name=y_name, y_unit=y_unit,
        title=(_TITLES[int(rng.integers(len(_TITLES)))]
               if rng.random() < cfg.title_prob else None),
        w_in=float(rng.uniform(*cfg.fig_w_in)), h_in=float(rng.uniform(*cfg.fig_h_in)),
        dpi=int(rng.choice(cfg.dpi_choices)), bar_w=float(rng.uniform(*cfg.bar_width)),
        value_lo=lo, value_hi=scale, rotation=cfg.label_rotation,
        grid=rng.random() < cfg.grid_prob)


def _col_label(i: int) -> str:
    """Spreadsheet-style column name: 0->A, 25->Z, 26->AA, 27->AB ..."""
    label, k = "", i
    while True:
        label = chr(ord("A") + k % 26) + label
        k = k // 26 - 1
        if k < 0:
            return label


def make_categories(rng: np.random.Generator, n: int) -> list[str]:
    """Sample one category-label style and emit ``n`` labels in it."""
    style = rng.choice(_CATEGORY_STYLES)
    if style == "letters":
        return [_col_label(i) for i in range(n)]
    if style == "words":
        pool = list(_WORDS) * (n // len(_WORDS) + 1)
        return [w if i < len(_WORDS) else f"{w}{i}" for i, w in enumerate(pool[:n])]
    if style == "quarters":
        out, y, q = [], int(rng.integers(10, 24)), 1
        for _ in range(n):
            out.append(f"Q{q}'{y:02d}")
            q = q + 1 if q < 4 else 1
            y += q == 1
        return out
    if style == "years":
        start = int(rng.integers(1990, 2020))
        return [str(start + i) for i in range(n)]
    return [str(i + 1) for i in range(n)]


def series_values(rng: np.random.Generator, style: Style, n: int) -> list[np.ndarray]:
    """Per-series values within the style's fixed range (stacked totals stay under hi)."""
    if style.stacked:
        totals = rng.uniform(0.35 * style.value_hi, style.value_hi, n)
        fr = rng.random((style.n_series, n))
        fr /= fr.sum(axis=0, keepdims=True)
        return [np.round(fr[s] * totals, 2) for s in range(style.n_series)]
    return [np.round(rng.uniform(style.value_lo, style.value_hi, n), 2)
            for _ in range(style.n_series)]


def fit_fontsize(cfg: GenConfig, n: int) -> int:
    """Shrink the tick font as density rises so labels never overlap."""
    lo, hi = cfg.tick_fontsize
    return int(max(lo, hi - 2.0 * np.log2(max(1.0, n / 4.0))))
