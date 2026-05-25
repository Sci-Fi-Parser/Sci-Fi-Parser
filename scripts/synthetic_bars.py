"""Synthetic bar-chart generator with pixel-accurate ground-truth labels.

Why this exists
---------------
Real chart labels in ``train_data/dataset.sqlite3`` are model-generated (Gemma ->
Haiku -> Opus), so they carry noise and a circularity risk when benchmarking
Claude/Gemma-family models. Charts rendered here are labelled *by construction*: we
know the values we plotted, and matplotlib's ``ax.transData`` transform gives the
exact pixel location of every bar, axis line, and tick. That yields three label
levels for free:

  * ``label2``   - the value table (axes + per-series points)
  * ``geometry`` - bar bounding boxes + value-axis ticks (detection + calibration
                   ground truth that the real dataset lacks)
  * ``label1``   - the chart sub-type ("bar_chart", "grouped_bar_chart", ...)

Two generation modes
--------------------
* (default) CONTROLLED DENSITY SERIES -- a designed experiment. For each enabled
  type, fix the style once (colours, titles, figure, DPI, value range) and vary
  ONLY the number of bars across ``density_steps``. Each density is rendered as a
  matched pair: labels OFF and labels ON with identical data (50/50), so you can
  measure density and printed-label effects in isolation. Labels are auto-fitted
  (rotated + font-shrunk) so they never overlap, like a real chart.
* ``--random N``  N fully-random charts (the old behaviour), for variety/volume.

Usage
-----
    .venv/bin/python scripts/synthetic_bars.py --out train_data/synthetic --overlay --sqlite
    .venv/bin/python scripts/synthetic_bars.py --config scripts/synthetic_bars.toml
    .venv/bin/python scripts/synthetic_bars.py --random 200 --out /tmp/synth

Knobs live in :class:`GenConfig` (documented inline) and override via ``--config``
(TOML) -- see ``scripts/synthetic_bars.toml``.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render (no display needed)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BAR_TYPES = (
    "bar_chart",             # single series, vertical
    "grouped_bar_chart",     # multiple series side-by-side, vertical
    "stacked_bar_chart",     # multiple series stacked, vertical
    "horizontal_bar_chart",  # single series, horizontal (value axis = x)
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GenConfig:
    """Knobs for the generator. Override any field from a TOML file via ``--config``.

    In the default density-series mode, the *style* knobs (colours, figure, DPI,
    value range) are sampled ONCE per series and held fixed; only density varies.
    """

    # Which types to generate a density series for. Toggle each true/false; every
    # enabled type gets its own fixed-style series. (TOML: a [type_enabled] table.)
    type_enabled: dict = field(default_factory=lambda: {
        "bar_chart": True,
        "grouped_bar_chart": False,
        "stacked_bar_chart": False,
        "horizontal_bar_chart": False,
    })

    # The density ladder (number of bars per step). Tight at the low end where
    # degradation begins, widening toward the max.
    density_steps: tuple[int, ...] = (4, 5, 7, 9, 10, 12, 15, 20, 25, 30)

    # Independent fixed-style series to emit PER enabled type. 1 = a single series
    # (10 densities x off/on = 20 images). Bump for K colour/style variants.
    per_type: int = 1

    # (min, max) INCLUSIVE series count for grouped/stacked charts.
    n_series: tuple[int, int] = (2, 4)

    # Fraction [0..1] of charts that emit the FULL geometry block (else values only).
    geometry_full_prob: float = 0.5

    # Figure width / height in INCHES and DPI options (sampled once per series).
    fig_w_in: tuple[float, float] = (5.0, 9.0)
    fig_h_in: tuple[float, float] = (3.5, 5.5)
    dpi_choices: tuple[int, ...] = (90, 100, 120, 150)

    # Bar width as a CATEGORY-unit fraction (0..1). Fixed per series, so bars thin
    # automatically as density rises (that's the intended density effect).
    bar_width: tuple[float, float] = (0.6, 0.85)

    # Probability a (non-stacked) series uses a negative value range.
    allow_negative: float = 0.15
    # Probability of value-axis gridlines / a title (decided once per series).
    grid_prob: float = 0.5
    title_prob: float = 0.7

    # Category-label fitting: constant rotation, with the tick font shrinking from
    # max -> min as density grows so labels always fit without overlapping.
    label_rotation: int = 90
    tick_fontsize: tuple[int, int] = (6, 11)  # (min floor, max at low density)

    # --random mode only: probability a random chart prints value labels.
    value_labels_prob: float = 0.0


_TUPLE_FIELDS = frozenset({
    "density_steps", "n_series", "fig_w_in", "fig_h_in", "dpi_choices",
    "bar_width", "tick_fontsize",
})


def load_config(path: Path) -> GenConfig:
    """Build a GenConfig from a TOML file, overriding only the keys present."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = GenConfig()
    valid = {f.name for f in fields(cfg)}
    for key, value in raw.items():
        if key not in valid:
            raise SystemExit(f"unknown config key {key!r} in {path}; valid: {sorted(valid)}")
        if key in _TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


# Small vocabularies so category/axis/series text varies like real charts.
_CATEGORY_STYLES = ("letters", "words", "quarters", "years", "numbers")
_WORDS = (
    "North", "South", "East", "West", "Alpha", "Beta", "Gamma", "Delta",
    "Retail", "Online", "Energy", "Banking", "Pharma", "Tech", "EU", "US",
    "UK", "Japan", "Brazil", "India", "China", "Canada",
)
_SERIES_NAMES = ("2021", "2022", "2023", "2024", "Actual", "Forecast",
                 "Domestic", "Export", "Group A", "Group B", "Group C")
_Y_AXES = (
    ("Revenue", "$M"), ("Count", None), ("Value", None), ("Share", "%"),
    ("Sales", "$"), ("Population", None), ("Score", None), ("Profit", "$B"),
)
_TITLES = ("Quarterly Results", "Regional Breakdown", "Performance by Group",
           "Annual Summary", "Comparison", "Distribution", None)


# --------------------------------------------------------------------------- #
# Style: the part held CONSTANT within a density series
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Style:
    chart_type: str
    horizontal: bool
    multi: bool
    stacked: bool
    n_series: int
    series_names: list
    series_colors: list      # one per series (multi) or [rgb] (single)
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


def sample_style(rng: np.random.Generator, cfg: GenConfig, chart_type: str) -> Style:
    horizontal = chart_type == "horizontal_bar_chart"
    stacked = chart_type == "stacked_bar_chart"
    multi = chart_type in ("grouped_bar_chart", "stacked_bar_chart")
    n_ser = int(rng.integers(cfg.n_series[0], cfg.n_series[1] + 1)) if multi else 1
    y_name, y_unit = _Y_AXES[int(rng.integers(len(_Y_AXES)))]

    if multi:
        names = [str(x) for x in rng.choice(_SERIES_NAMES, size=n_ser, replace=False)]
        cmap = plt.get_cmap("tab10")
        colors = [cmap(s % 10) for s in range(n_ser)]
    else:
        names = [y_name]
        colors = [tuple(rng.random(3) * 0.7 + 0.1)]  # one fixed colour

    scale = float(rng.choice([10.0, 100.0, 1_000.0, 1e6]))
    lo = -0.4 * scale if (not stacked and rng.random() < cfg.allow_negative) else 0.0
    return Style(
        chart_type=chart_type, horizontal=horizontal, multi=multi, stacked=stacked,
        n_series=n_ser, series_names=names, series_colors=colors,
        y_name=y_name, y_unit=y_unit,
        title=_TITLES[int(rng.integers(len(_TITLES)))] if rng.random() < cfg.title_prob else None,
        w_in=float(rng.uniform(*cfg.fig_w_in)), h_in=float(rng.uniform(*cfg.fig_h_in)),
        dpi=int(rng.choice(cfg.dpi_choices)), bar_w=float(rng.uniform(*cfg.bar_width)),
        value_lo=lo, value_hi=scale, rotation=cfg.label_rotation,
        grid=rng.random() < cfg.grid_prob,
    )


# --------------------------------------------------------------------------- #
# Category / value sampling
# --------------------------------------------------------------------------- #
def make_categories(rng: np.random.Generator, n: int) -> list[str]:
    style = rng.choice(_CATEGORY_STYLES)
    if style == "letters":
        out = []
        for i in range(n):
            label, k = "", i
            while True:
                label = chr(ord("A") + k % 26) + label
                k = k // 26 - 1
                if k < 0:
                    break
            out.append(label)
        return out
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


def series_values(rng: np.random.Generator, style: Style, n_cats: int) -> list[np.ndarray]:
    """Values within the style's fixed value range (stacked totals fit under hi)."""
    if style.stacked:
        totals = rng.uniform(0.35 * style.value_hi, style.value_hi, n_cats)
        fracs = rng.random((style.n_series, n_cats))
        fracs /= fracs.sum(axis=0, keepdims=True)
        return [np.round(fracs[s] * totals, 2) for s in range(style.n_series)]
    return [np.round(rng.uniform(style.value_lo, style.value_hi, n_cats), 2)
            for _ in range(style.n_series)]


def _fit_fontsize(cfg: GenConfig, n_cats: int) -> int:
    """Shrink tick font from max->min as density grows, so labels always fit."""
    lo, hi = cfg.tick_fontsize
    return int(max(lo, hi - 2.0 * np.log2(max(1.0, n_cats / 4.0))))


# --------------------------------------------------------------------------- #
# Core renderer (fixed style + given density) -> image + exact labels
# --------------------------------------------------------------------------- #
def render_chart(cfg: GenConfig, style: Style, cats: list[str],
                 svals: list[np.ndarray], labels_on: bool, geometry_full: bool,
                 meta_extra: dict) -> tuple[np.ndarray, dict]:
    n = len(cats)
    fig, ax = plt.subplots(figsize=(style.w_in, style.h_in), dpi=style.dpi)
    pos = np.arange(n)
    entries: list[tuple[int, int, object]] = []  # (series_idx, cat_idx, Rectangle)

    if style.horizontal:
        cont = ax.barh(pos, svals[0], height=style.bar_w, color=style.series_colors[0])
        entries = [(0, j, r) for j, r in enumerate(cont)]
        ax.set_yticks(pos)
        ax.set_yticklabels(cats, fontsize=_fit_fontsize(cfg, n))
        ax.set_xlabel(f"{style.y_name} ({style.y_unit})" if style.y_unit else style.y_name)
        ax.set_xlim(style.value_lo - 0.02 * style.value_hi, style.value_hi * 1.05)
        containers = [cont]
    else:
        containers = []
        if style.stacked:
            bottom = np.zeros(n)
            for s in range(style.n_series):
                cont = ax.bar(pos, svals[s], width=style.bar_w, bottom=bottom,
                              color=style.series_colors[s], label=style.series_names[s])
                entries += [(s, j, r) for j, r in enumerate(cont)]
                bottom = bottom + svals[s]
                containers.append(cont)
        elif style.multi:  # grouped
            sub = style.bar_w / style.n_series
            for s in range(style.n_series):
                off = (s - (style.n_series - 1) / 2) * sub
                cont = ax.bar(pos + off, svals[s], width=sub * 0.9,
                              color=style.series_colors[s], label=style.series_names[s])
                entries += [(s, j, r) for j, r in enumerate(cont)]
                containers.append(cont)
        else:  # simple
            cont = ax.bar(pos, svals[0], width=style.bar_w, color=style.series_colors[0])
            entries = [(0, j, r) for j, r in enumerate(cont)]
            containers.append(cont)
        ax.set_xticks(pos)
        ax.set_xticklabels(cats, rotation=style.rotation,
                           ha="right" if 0 < style.rotation < 90 else "center",
                           fontsize=_fit_fontsize(cfg, n))
        ax.set_ylabel(f"{style.y_name} ({style.y_unit})" if style.y_unit else style.y_name)
        ax.set_ylim(style.value_lo - 0.02 * style.value_hi, style.value_hi * 1.05)
        if style.multi:
            ax.legend(fontsize=7)

    if style.title:
        ax.set_title(style.title)
    if style.grid:
        ax.grid(axis="x" if style.horizontal else "y", alpha=0.4)
    if labels_on:
        for cont in containers:
            ax.bar_label(cont, fmt="%.0f", fontsize=max(5, _fit_fontsize(cfg, n) - 1), padding=1)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h_px, w_px = rgba.shape[:2]
    image = rgba[:, :, :3].copy()

    def to_img(x, y):
        xd, yd = ax.transData.transform((x, y))
        return [float(xd), float(h_px - yd)]

    geometry = None
    if geometry_full:
        ext = ax.get_window_extent()
        plot_area = [float(ext.x0), float(h_px - ext.y1), float(ext.x1), float(h_px - ext.y0)]
        bars = []
        for s_idx, c_idx, rect in entries:
            p0 = to_img(rect.get_x(), rect.get_y())
            p1 = to_img(rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height())
            bars.append({"series": style.series_names[s_idx], "category": cats[c_idx],
                         "value": float(svals[s_idx][c_idx]),
                         "bbox_px": [min(p0[0], p1[0]), min(p0[1], p1[1]),
                                     max(p0[0], p1[0]), max(p0[1], p1[1])]})
        if style.horizontal:
            lo, hi = ax.get_xlim()
            ticks = [{"value": float(t), "px": to_img(float(t), ax.get_ylim()[0])[0]}
                     for t in ax.get_xticks() if lo <= t <= hi]
        else:
            lo, hi = ax.get_ylim()
            ticks = [{"value": float(t), "px": to_img(ax.get_xlim()[0], float(t))[1]}
                     for t in ax.get_yticks() if lo <= t <= hi]
        geometry = {"image_size": [int(w_px), int(h_px)],
                    "orientation": "h" if style.horizontal else "v",
                    "plot_area_px": plot_area, "value_ticks": ticks, "bars": bars}

    val_lo, val_hi = (ax.get_xlim() if style.horizontal else ax.get_ylim())
    cat_axis = {"title": None, "scale": "category"}
    val_axis = {"title": style.y_name, "unit": style.y_unit, "scale": "linear",
                "range": [float(val_lo), float(val_hi)]}
    label = {
        "label1": style.chart_type,
        "label2": {
            "chart_type": style.chart_type,
            "orientation": "h" if style.horizontal else "v",
            "axes": {"x": val_axis, "y": cat_axis} if style.horizontal
                    else {"x": cat_axis, "y": val_axis},
            "value_range": [float(val_lo), float(val_hi)],
            "series": [{"name": style.series_names[s],
                        "points": [[cats[c], float(svals[s][c])] for c in range(n)]}
                       for s in range(style.n_series)],
            "confidence": 1.0,
        },
        "geometry": geometry,
        "meta": {"type": style.chart_type, "orientation": "h" if style.horizontal else "v",
                 "density": n, "n_series": style.n_series, "labels_on": bool(labels_on),
                 "geometry_full": bool(geometry_full), **meta_extra},
        "render": {"figsize_in": [style.w_in, style.h_in], "dpi": style.dpi,
                   "title": style.title},
    }
    plt.close(fig)
    return image, label


# --------------------------------------------------------------------------- #
# Generation modes
# --------------------------------------------------------------------------- #
def generate_series(rng: np.random.Generator, cfg: GenConfig):
    """Yield (name, image, label) for the controlled density series (default mode)."""
    enabled = [t for t in BAR_TYPES if cfg.type_enabled.get(t)]
    for ctype in enabled:
        for k in range(cfg.per_type):
            style = sample_style(rng, cfg, ctype)
            sid = f"{ctype}_s{k}"
            for d in cfg.density_steps:
                cats = make_categories(rng, d)
                svals = series_values(rng, style, d)          # shared by the off/on pair
                pair = f"{sid}_d{d:02d}"
                geo = rng.random() < cfg.geometry_full_prob
                for labels_on in (False, True):
                    name = f"{pair}_{'on' if labels_on else 'off'}.png"
                    image, label = render_chart(
                        cfg, style, cats, svals, labels_on, geo,
                        {"series_id": sid, "pair_id": pair, "density_step": d})
                    yield name, image, label


def generate_random(rng: np.random.Generator, cfg: GenConfig, n: int):
    """Yield (name, image, label) for N fully-random charts (variety/volume)."""
    enabled = [t for t in BAR_TYPES if cfg.type_enabled.get(t)] or list(BAR_TYPES)
    for i in range(n):
        ctype = str(rng.choice(enabled))
        style = sample_style(rng, cfg, ctype)
        d = int(rng.integers(cfg.density_steps[0], cfg.density_steps[-1] + 1))
        cats = make_categories(rng, d)
        svals = series_values(rng, style, d)
        labels_on = rng.random() < cfg.value_labels_prob
        geo = rng.random() < cfg.geometry_full_prob
        image, label = render_chart(cfg, style, cats, svals, labels_on, geo,
                                    {"series_id": f"random_{i}", "pair_id": None,
                                     "density_step": None})
        yield f"random_{i:05d}.png", image, label


# --------------------------------------------------------------------------- #
# Augmentation + overlay + writers
# --------------------------------------------------------------------------- #
def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """JPEG/noise/blur/brightness only -- no spatial transforms, so labels stay exact."""
    import cv2
    out = image
    if rng.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < 0.4:
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(3, 12), out.shape),
                      0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def make_overlay(image: np.ndarray, label: dict) -> np.ndarray:
    """Draw ground-truth boxes/ticks back on the image to verify pixel accuracy."""
    import cv2
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    geo = label.get("geometry")
    if geo is None:
        return out
    horizontal = geo.get("orientation") == "h"
    for bar in geo["bars"]:
        x0, y0, x1, y1 = (int(round(v)) for v in bar["bbox_px"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    pa = [int(round(v)) for v in geo["plot_area_px"]]
    for tick in geo["value_ticks"]:
        c = int(round(tick["px"]))
        if horizontal:
            cv2.line(out, (c, pa[3] - 6), (c, pa[3] + 6), (0, 0, 255), 2)
        else:
            cv2.line(out, (pa[0] - 6, c), (pa[0] + 6, c), (0, 0, 255), 2)
    cv2.rectangle(out, (pa[0], pa[1]), (pa[2], pa[3]), (255, 0, 0), 1)
    return out


_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY, source TEXT, source_ref TEXT,
    label1 TEXT, label2 TEXT, geometry TEXT, meta TEXT,
    img BLOB, mime_type TEXT, width INTEGER, height INTEGER
);
"""


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=None, metavar="N",
                    help="random mode: N fully-random charts (default: density series)")
    ap.add_argument("--out", type=Path, default=Path("train_data/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=None,
                    help="TOML overriding GenConfig (see scripts/synthetic_bars.toml)")
    ap.add_argument("--augment", action="store_true", help="add JPEG/noise/blur realism")
    ap.add_argument("--overlay", action="store_true", help="write _debug overlays (verification)")
    ap.add_argument("--sqlite", action="store_true", help="also write dataset.sqlite3")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_config(args.config) if args.config else GenConfig()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = args.out / "_debug"
    if args.overlay:
        dbg_dir.mkdir(exist_ok=True)
    con = sqlite3.connect(args.out / "dataset.sqlite3") if args.sqlite else None
    if con is not None:
        con.executescript(_SQLITE_DDL)

    stream = (generate_random(rng, cfg, args.random) if args.random is not None
              else generate_series(rng, cfg))

    counts: dict[str, int] = {}
    n_total = 0
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as jsonl:
        from PIL import Image
        for name, image, label in stream:
            if args.augment:
                image = augment(image, rng)
            Image.fromarray(image).save(img_dir / name)
            label["image"] = name
            jsonl.write(json.dumps(label) + "\n")
            counts[label["label1"]] = counts.get(label["label1"], 0) + 1
            n_total += 1
            if args.overlay:
                import cv2
                cv2.imwrite(str(dbg_dir / f"overlay_{name}"), make_overlay(image, label))
            if con is not None:
                con.execute(
                    "INSERT INTO dataset(source,source_ref,label1,label2,geometry,meta,"
                    "img,mime_type,width,height) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    ("synthetic", f"seed{args.seed}/{name}", label["label1"],
                     json.dumps(label["label2"]),
                     json.dumps(label["geometry"]) if label["geometry"] else None,
                     json.dumps(label["meta"]), _png_bytes(image), "image/png",
                     image.shape[1], image.shape[0]))
    if con is not None:
        con.commit()
        con.close()

    print(f"generated {n_total} charts -> {img_dir}")
    for ctype, c in sorted(counts.items()):
        print(f"  {ctype:22s} {c}")
    print(f"labels -> {args.out / 'labels.jsonl'}")
    if args.overlay:
        print(f"overlays -> {dbg_dir}")
    if args.sqlite:
        print(f"sqlite  -> {args.out / 'dataset.sqlite3'}")


if __name__ == "__main__":
    main()
