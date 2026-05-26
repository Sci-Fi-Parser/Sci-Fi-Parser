"""Render one chart to an RGB image plus its pixel-accurate label record.

A chart is labelled *by construction*: we know the values we plotted, and matplotlib's
``ax.transData`` transform gives the exact pixel of every bar/marker, axis line, and
tick. Three label levels come for free -- ``label2`` (values), ``geometry`` (pixel
boxes/points + ticks), ``label1`` (chart type).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import GenConfig
from .style import Style, fit_fontsize


# --------------------------------------------------------------------------- #
# Drawing helpers (one chart family each, no RNG)
# --------------------------------------------------------------------------- #
def _value_label(style: Style) -> str:
    """Value-axis label, e.g. ``Revenue ($M)`` or just ``Count``."""
    return f"{style.y_name} ({style.y_unit})" if style.y_unit else style.y_name


def _tick_ha(rotation: int) -> str:
    """Horizontal alignment for rotated category ticks."""
    return "right" if 0 < rotation < 90 else "center"


def _bar_colors(style: Style, n: int):
    """Per-bar colour spec; ``None`` lets matplotlib colour each series itself."""
    if style.color_mode == "single":
        return style.base_color
    if style.color_mode == "colormap":
        return plt.get_cmap(style.cmap_name)(np.linspace(0.1, 0.9, n))
    return None


def _annotate_points(ax, pos: np.ndarray, vals: np.ndarray, fs: int) -> None:
    """Print value labels above each point of a line series."""
    for c, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (pos[c], v), fontsize=max(5, fs - 1),
                    ha="center", va="bottom")


def _draw_line(ax, style: Style, pos: np.ndarray, svals: list[np.ndarray],
               cats: list[str], fs: int, labels_on: bool) -> None:
    """Draw the line family (one plot per series); markers shrink as density rises."""
    n = len(cats)
    ms = max(2.0, 7.0 - 1.5 * np.log2(max(1.0, n / 4.0)))
    for s in range(style.n_series):
        ax.plot(pos, svals[s], marker="o", markersize=ms, linewidth=1.6,
                color=style.series_colors[s], label=style.series_names[s])
        if labels_on:
            _annotate_points(ax, pos, svals[s], fs)
    ax.set_xticks(pos)
    ax.set_xticklabels(cats, rotation=style.rotation,
                       ha=_tick_ha(style.rotation), fontsize=fs)
    ax.set_ylabel(_value_label(style))
    ax.set_ylim(style.value_lo - 0.02 * style.value_hi, style.value_hi * 1.05)
    if style.n_series > 1:
        ax.legend(fontsize=7)


def _draw_vertical_bars(ax, style: Style, pos: np.ndarray, svals: list[np.ndarray],
                        col) -> tuple[list, list]:
    """Draw stacked/grouped/simple vertical bars -> (geometry items, containers)."""
    n = len(svals[0])
    items: list = []
    containers: list = []
    if style.stacked:
        bottom = np.zeros(n)
        for s in range(style.n_series):
            cont = ax.bar(pos, svals[s], width=style.bar_w, bottom=bottom,
                          color=style.series_colors[s], label=style.series_names[s])
            items += [(s, j, r) for j, r in enumerate(cont)]
            bottom = bottom + svals[s]
            containers.append(cont)
    elif style.n_series > 1:  # grouped: offset each series by a sub-width
        sub = style.bar_w / style.n_series
        for s in range(style.n_series):
            off = (s - (style.n_series - 1) / 2) * sub
            cont = ax.bar(pos + off, svals[s], width=sub * 0.9,
                          color=style.series_colors[s], label=style.series_names[s])
            items += [(s, j, r) for j, r in enumerate(cont)]
            containers.append(cont)
    else:  # simple / multicolor: a single series
        cont = ax.bar(pos, svals[0], width=style.bar_w, color=col)
        items = [(0, j, r) for j, r in enumerate(cont)]
        containers.append(cont)
    return items, containers


def _draw_bars(ax, style: Style, pos: np.ndarray, svals: list[np.ndarray],
               cats: list[str], fs: int, labels_on: bool) -> list:
    """Draw the bar family and return matched bars as ``(series_idx, cat_idx, rect)``."""
    col = _bar_colors(style, len(cats))
    if style.horizontal:
        cont = ax.barh(pos, svals[0], height=style.bar_w, color=col)
        items, containers = [(0, j, r) for j, r in enumerate(cont)], [cont]
        ax.set_yticks(pos)
        ax.set_yticklabels(cats, fontsize=fs)
        ax.set_xlabel(_value_label(style))
        ax.set_xlim(style.value_lo - 0.02 * style.value_hi, style.value_hi * 1.05)
    else:
        items, containers = _draw_vertical_bars(ax, style, pos, svals, col)
        ax.set_xticks(pos)
        ax.set_xticklabels(cats, rotation=style.rotation,
                           ha=_tick_ha(style.rotation), fontsize=fs)
        ax.set_ylabel(_value_label(style))
        ax.set_ylim(style.value_lo - 0.02 * style.value_hi, style.value_hi * 1.05)
    if labels_on:
        for cont in containers:
            ax.bar_label(cont, fmt="%.0f", fontsize=max(5, fs - 1), padding=1)
    if style.n_series > 1:
        ax.legend(fontsize=7)
    return items


def _draw(ax, style: Style, pos: np.ndarray, svals: list[np.ndarray],
          cats: list[str], fs: int, labels_on: bool) -> list:
    """Dispatch to the line or bar renderer; return bar rects (``[]`` for lines)."""
    if style.family == "line":
        _draw_line(ax, style, pos, svals, cats, fs, labels_on)
        return []
    return _draw_bars(ax, style, pos, svals, cats, fs, labels_on)


def _decorate(ax, style: Style) -> None:
    """Optional title + value-axis gridlines."""
    if style.title:
        ax.set_title(style.title)
    if style.grid:
        ax.grid(axis="x" if style.horizontal else "y", alpha=0.4)


def _rasterize(fig, ax):
    """Render the figure to an RGB array + a data->image-pixel mapper ``to_img``."""
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    h_px = rgba.shape[0]
    image = rgba[:, :, :3].copy()

    def to_img(x, y):
        """Data coords -> top-left-origin image pixels (matplotlib y is bottom-up)."""
        xd, yd = ax.transData.transform((x, y))
        return [float(xd), float(h_px - yd)]

    return image, to_img


# --------------------------------------------------------------------------- #
# Ground-truth geometry + label assembly
# --------------------------------------------------------------------------- #
def _line_marks(style: Style, cats: list[str], svals: list[np.ndarray],
                pos: np.ndarray, to_img) -> list[dict]:
    """Per-point image-pixel positions for line charts."""
    return [{"series": style.series_names[s], "category": cats[c],
             "value": float(svals[s][c]), "point_px": to_img(pos[c], svals[s][c])}
            for s in range(style.n_series) for c in range(len(cats))]


def _bar_marks(style: Style, cats: list[str], svals: list[np.ndarray],
               items: list, to_img) -> list[dict]:
    """Per-bar image-pixel bounding boxes for bar charts."""
    marks = []
    for s_idx, c_idx, rect in items:
        p0 = to_img(rect.get_x(), rect.get_y())
        p1 = to_img(rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height())
        marks.append({"series": style.series_names[s_idx], "category": cats[c_idx],
                      "value": float(svals[s_idx][c_idx]),
                      "bbox_px": [min(p0[0], p1[0]), min(p0[1], p1[1]),
                                  max(p0[0], p1[0]), max(p0[1], p1[1])]})
    return marks


def _value_ticks(ax, style: Style, to_img) -> list[dict]:
    """Value-axis tick positions in image pixels (axis-calibration ground truth)."""
    if style.horizontal:
        lo, hi = ax.get_xlim()
        return [{"value": float(t), "px": to_img(float(t), ax.get_ylim()[0])[0]}
                for t in ax.get_xticks() if lo <= t <= hi]
    lo, hi = ax.get_ylim()
    return [{"value": float(t), "px": to_img(ax.get_xlim()[0], float(t))[1]}
            for t in ax.get_yticks() if lo <= t <= hi]


def _collect_geometry(ax, style: Style, cats: list[str], svals: list[np.ndarray],
                      pos: np.ndarray, items: list, to_img, image: np.ndarray) -> dict:
    """Pixel-space ground truth: plot box, value ticks, and per-element marks."""
    h_px, w_px = image.shape[:2]
    ext = ax.get_window_extent()
    plot_area = [float(ext.x0), float(h_px - ext.y1), float(ext.x1), float(h_px - ext.y0)]
    marks = (_line_marks(style, cats, svals, pos, to_img) if style.family == "line"
             else _bar_marks(style, cats, svals, items, to_img))
    return {"image_size": [int(w_px), int(h_px)], "family": style.family,
            "orientation": style.orientation, "plot_area_px": plot_area,
            "value_ticks": _value_ticks(ax, style, to_img), "items": marks}


def _build_label(style: Style, cats: list[str], svals: list[np.ndarray], labels_on: bool,
                 geometry_full: bool, geometry: dict | None, meta_extra: dict,
                 resolution: int | None, dpi: int, val_range) -> dict:
    """Assemble the full label record: label1 / label2 / geometry / meta / render."""
    n = len(cats)
    val_lo, val_hi = float(val_range[0]), float(val_range[1])
    cat_axis = {"title": None, "scale": "category"}
    val_axis = {"title": style.y_name, "unit": style.y_unit, "scale": "linear",
                "range": [val_lo, val_hi]}
    return {
        "label1": style.chart_type,
        "label2": {
            "chart_type": style.chart_type, "preset": style.alias,
            "orientation": style.orientation,
            "axes": {"x": val_axis, "y": cat_axis} if style.horizontal
                    else {"x": cat_axis, "y": val_axis},
            "value_range": [val_lo, val_hi],
            "series": [{"name": style.series_names[s],
                        "points": [[cats[c], float(svals[s][c])] for c in range(n)]}
                       for s in range(style.n_series)],
            "confidence": 1.0,
        },
        "geometry": geometry,
        "meta": {"type": style.chart_type, "preset": style.alias,
                 "orientation": style.orientation, "density": n,
                 "n_series": style.n_series, "labels_on": bool(labels_on),
                 "geometry_full": bool(geometry_full), "resolution": resolution,
                 **meta_extra},
        "render": {"figsize_in": [style.w_in, style.h_in], "dpi": dpi,
                   "title": style.title},
    }


def render_chart(cfg: GenConfig, style: Style, cats: list[str], svals: list[np.ndarray],
                 labels_on: bool, geometry_full: bool, meta_extra: dict,
                 resolution: int | None = None) -> tuple[np.ndarray, dict]:
    """Render one chart (fixed style, given density) -> (RGB image, label record)."""
    fs = fit_fontsize(cfg, len(cats))
    # resolution = target image height in px -> derive dpi (keeps the figure aspect).
    dpi = style.dpi if resolution is None else max(40, round(resolution / style.h_in))
    fig, ax = plt.subplots(figsize=(style.w_in, style.h_in), dpi=dpi)
    pos = np.arange(len(cats))
    items = _draw(ax, style, pos, svals, cats, fs, labels_on)
    _decorate(ax, style)
    image, to_img = _rasterize(fig, ax)
    geometry = (_collect_geometry(ax, style, cats, svals, pos, items, to_img, image)
                if geometry_full else None)
    val_range = ax.get_xlim() if style.horizontal else ax.get_ylim()
    label = _build_label(style, cats, svals, labels_on, geometry_full, geometry,
                         meta_extra, resolution, dpi, val_range)
    plt.close(fig)
    return image, label
