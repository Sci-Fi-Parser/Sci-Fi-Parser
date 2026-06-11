"""Report rendering for the accuracy benchmark — *decoupled* from scoring.

Everything that draws the HTML report lives here. It takes **plain data**
(dicts / lists of primitives, an aggregate dict, and pre-computed breakdown
rows) — it imports nothing from :mod:`benchmark`, so the scoring side can be
rewritten freely as long as it feeds the contract below.

Data contract (``write_html``):

* ``agg``       — the aggregate stats dict (keys read by :func:`_summary_cards`).
* ``charts``    — one dict per chart::

      {"image", "preset", "density", "labels_on",
       "type_true", "type_pred", "type_matched",
       "n_true", "matched", "missed", "extra",
       "mean_pct", "max_pct", "errors_pct", "span",
       "seconds", "confidence",
       "truth": [[series, cat, true_value], ...],   # ordered
       "pred":  [[series, cat, pred_value], ...]}

* ``breakdowns`` — ``[{"title": str, "rows": [(group, n, err, recall, sec), ...]}]``.
* ``img_dir``    — folder holding the chart PNGs (for the thumbnails).

The one *graph* (the deviation distribution) is drawn with **matplotlib**
(headless Agg backend) and embedded as a base64 PNG; the rest of the report is
plain HTML string-building.
"""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # headless: render to a buffer, never a window
import matplotlib.pyplot as plt    # noqa: E402  (must follow matplotlib.use)
import numpy as np                 # noqa: E402
from PIL import Image              # noqa: E402


# --------------------------------------------------------------------------- #
# Inline styling + the click-to-sort script
# --------------------------------------------------------------------------- #
_CSS = """
 body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a}
 h1{font-size:20px}
 h2{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
 .stats{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
 .stat{background:#f4f6f8;border-radius:10px;padding:12px 16px;
       min-width:90px;text-align:center;cursor:help}
 .stat .v{font-size:22px;font-weight:700} .stat .k{font-size:12px;color:#666}
 .tables{display:flex;flex-wrap:wrap;gap:24px}
 .grid{display:flex;flex-wrap:wrap;gap:14px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:10px;width:340px}
 .card img{width:100%;border-radius:6px} .meta{font-size:12px;margin-top:6px}
 .devwrap{max-width:880px;margin:10px 0}
 .devbell{width:100%;height:auto;border:1px solid #eee;border-radius:8px;background:#fff}
 table{border-collapse:collapse;width:100%;font-size:13px}
 .kv td,.kv th{border-bottom:1px solid #eee;padding:2px 6px;text-align:right}
 .kv td:first-child,.kv th:first-child{text-align:left}
 table.full th,table.full td{border-bottom:1px solid #eee;
                             padding:6px 8px;text-align:right}
 table.full th:first-child,table.full td:first-child{text-align:left}
 table.full th{cursor:pointer;background:#f4f6f8;position:sticky;top:0}
"""

_SORT_JS = r"""
document.querySelectorAll('#t th').forEach((h,i)=>h.onclick=()=>{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows];
 const num=h.dataset.num==='1', dir=h.dataset.d=h.dataset.d==='1'?'':'1';
 rows.sort((a,b)=>{const x=a.cells[i],y=b.cells[i];
   const va=num?+(x.dataset.v??x.textContent.replace(/[^0-9.\-]/g,'')||0):x.textContent;
   const vb=num?+(y.dataset.v??y.textContent.replace(/[^0-9.\-]/g,'')||0):y.textContent;
   return (va>vb?1:va<vb?-1:0)*(dir?-1:1);});
 rows.forEach(r=>tb.appendChild(r));});
"""


# --------------------------------------------------------------------------- #
# Tiny self-contained helpers (private copies so we don't import benchmark)
# --------------------------------------------------------------------------- #
def _normalize_key(series: str, category: str) -> tuple[str, str]:
    return (str(series).strip().casefold(), str(category).strip().casefold())


def _pct_of_span(pred: float, true: float, span: float) -> float:
    return abs(pred - true) / (span or 1.0) * 100.0


def _pct(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}%"


def _thumb_b64(path: Path, width: int = 360) -> str | None:
    """Base64 PNG thumbnail, or None if the image file is missing/unreadable
    (so one bad path doesn't sink the whole report)."""
    try:
        im = Image.open(path)
    except (FileNotFoundError, OSError):
        return None
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pred_lookup(chart: dict) -> dict[tuple[str, str], float]:
    return {_normalize_key(ps, pc): pv for ps, pc, pv in chart["pred"]}


def _signed_devs(chart: dict) -> list[float]:
    """Signed per-bar deviation ((pred-true)/axis_span %) for a chart's matched
    bars, clamped to [-100, +100]."""
    look = _pred_lookup(chart)
    span = chart.get("span", 1.0) or 1.0
    devs: list[float] = []
    for s, c, tv in chart["truth"]:
        pv = look.get(_normalize_key(s, c))
        if pv is None:
            continue
        d = (pv - tv) / span * 100.0
        devs.append(max(-100.0, min(100.0, d)))
    return devs


def _rank_score(chart: dict) -> float:
    """Composite ranking score; higher = worse. No matched bars -> +inf (sorts
    last). Mirrors the previous benchmark ranking."""
    if not chart["errors_pct"]:
        return float("inf")
    value_error = chart["mean_pct"] + chart["max_pct"]
    denom = chart["n_true"] + chart["extra"] or 1
    label_loss = (chart["missed"] + chart["extra"]) / denom * 5.0
    return value_error + label_loss


# --------------------------------------------------------------------------- #
# The one graph — matplotlib (deviation distribution + Laplace/normal fits)
# --------------------------------------------------------------------------- #
def _dev_bell_png(devs: list[float]) -> str:
    """Distribution of signed deviation as a matplotlib figure -> base64 PNG.

    Histogram of the per-bar deviations (clamped ±100 %) with two fitted
    curves scaled to the histogram's counts: a **Laplace** (MLE = median +
    mean-absolute-deviation -- robust, matches the sharp-peak/heavy-tail shape)
    and a **normal** (mean + std) for the familiar comparison.
    """
    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=120)
    if devs:
        arr = np.asarray(devs, dtype=float)
        n = arr.size
        med = float(np.median(arr))
        b = max(float(np.mean(np.abs(arr - med))), 3.0)
        mu = float(arr.mean())
        sd_raw = float(arr.std())
        sigma = max(sd_raw, 4.0)

        bins = np.linspace(-100, 100, 21)
        bw = bins[1] - bins[0]
        ax.hist(arr, bins=bins, color="#6366f1", alpha=0.18, edgecolor="none")

        xs = np.linspace(-100, 100, 400)
        nrm = n * bw * np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2)) \
            / (sigma * math.sqrt(2 * math.pi))
        lap = n * bw * np.exp(-np.abs(xs - med) / b) / (2 * b)
        ax.plot(xs, nrm, color="#94a3b8", lw=1.6, ls="--",
                label=f"Normal  (mean {mu:+.1f}%, sd {sd_raw:.1f})")
        ax.plot(xs, lap, color="#4f46e5", lw=2.3,
                label=f"Laplace  (median {med:+.1f}%, b {b:.1f})")
        ax.axvline(0, color="#cbd5e1", lw=1, ls=":")
        ax.axvline(med, color="#4f46e5", lw=1, ls="--", alpha=0.6)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"n = {n} matched bars", fontsize=9, loc="right", color="#64748b")
    else:
        ax.text(0.5, 0.5, "no matched bars", ha="center", va="center",
                color="#94a3b8", transform=ax.transAxes)

    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["-100%", "-50%", "0%", "+50%", "+100%"])
    ax.set_xlabel("deviation  (pred − true) / axis range", fontsize=9)
    ax.set_ylabel("bars", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#eef2f7", lw=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return ('<img class="devbell" src="data:image/png;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}"/>')


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #
def _group_table(title: str, rows: list[tuple]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{n}</td>"
        f"<td>{_pct(err)}</td><td>{rec*100:.1f}%</td><td>{sec:.1f} s</td></tr>"
        for k, n, err, rec, sec in rows
    )
    return (f"<table class='full' style='max-width:560px'><thead><tr><th>{title}</th>"
            f"<th>charts</th><th>mean err</th><th>recall</th>"
            f"<th>mean time</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _pred_cell(look: dict, tval: float, series: str, cat: str, span: float) -> str:
    pv = look.get(_normalize_key(series, cat))
    if pv is None:
        return "—"
    err = _pct_of_span(pv, tval, span)
    return f"{pv:.2f} <span style='color:#888'>({err:.1f}%)</span>"


def _type_chip(chart: dict) -> str:
    tt, tp = chart["type_true"], chart["type_pred"]
    if tt is None:
        return ""
    if tp is None:
        return (f" · type={html.escape(str(tt))} "
                f"<span style='color:#a00'>(no pred)</span>")
    ok = "✓" if chart["type_matched"] else "✗"
    colour = "#0a0" if chart["type_matched"] else "#a00"
    return (f" · type=<span style='color:{colour}'>{ok}</span> "
            f"{html.escape(str(tp))} (true: {html.escape(str(tt))})")


def _conf_chip(chart: dict) -> str:
    c = chart["confidence"]
    return f" · conf {c:.2f}" if c is not None else ""


def _detail_card(chart: dict, img_dir: Path) -> str:
    preset = html.escape(str(chart["preset"]))
    look = _pred_lookup(chart)
    span = chart.get("span", 1.0)
    kv_head = "<tr><th>series</th><th>cat</th><th>true</th><th>pred (err)</th></tr>"
    rows = "".join(
        f"<tr><td>{html.escape(str(s))}</td><td>{html.escape(str(c))}</td>"
        f"<td>{tv:.2f}</td><td>{_pred_cell(look, tv, s, c, span)}</td></tr>"
        for s, c, tv in chart["truth"]
    )
    thumb = _thumb_b64(img_dir / chart["image"])
    img_html = (f'<img src="data:image/png;base64,{thumb}"/>' if thumb else
                '<div style="height:120px;display:flex;align-items:center;'
                'justify-content:center;background:#f4f6f8;border-radius:6px;'
                'color:#94a3b8;font-size:12px">image missing</div>')
    return f"""
    <div class="card">
      {img_html}
      <div class="meta">
        <b>{html.escape(str(chart["image"]))}</b> · {preset}
        · d{chart["density"]} · labels={chart["labels_on"]}
        {_type_chip(chart)}{_conf_chip(chart)}<br/>
        mean {_pct(chart["mean_pct"])} · max {_pct(chart["max_pct"])}
        · {chart["seconds"]:.1f} s · missed {chart["missed"]} · extra {chart["extra"]}
        <table class="kv">{kv_head}{rows}</table>
      </div>
    </div>"""


def _summary_cards(agg: dict) -> str:
    """Top-of-report stat strip; each card has a hover description."""
    type_acc = agg["type_accuracy"]
    type_str = "-" if math.isnan(type_acc) else f"{type_acc*100:.0f}%"
    conf = agg["mean_confidence"]
    conf_str = "-" if math.isnan(conf) else f"{conf:.2f}"
    cards: list[tuple[str, object, str]] = [
        ("Charts", agg["n_charts"],
         "Number of chart images scored in this run."),
        ("Mean error", _pct(agg["mean_pct"]),
         "Mean per-bar error across all matched bars. "
         "Error = |predicted − true| as a percentage of the value-axis range."),
        ("Median", _pct(agg["median_pct"]),
         "Median per-bar error. Less sensitive to outliers than the mean."),
        ("p95", _pct(agg["p95_pct"]),
         "95th-percentile per-bar error: 95% of matched bars are at or below this."),
        ("Recall", f"{agg['recall']*100:.1f}%",
         "Fraction of true bars the extractor returned and matched. "
         "100% = no bars missed."),
        ("Precision", f"{agg['precision']*100:.1f}%",
         "Fraction of predicted bars that matched a true bar. "
         "100% = no hallucinated extras."),
        ("≤1%", f"{agg['within_1pct']*100:.0f}%",
         "Share of matched bars whose error is within 1% of the axis range."),
        ("≤5%", f"{agg['within_5pct']*100:.0f}%",
         "Share of matched bars whose error is within 5% of the axis range."),
        ("Type acc", type_str,
         "Fraction of charts where the extractor's chart_type matches truth. "
         "Only counts charts where both sides reported a type."),
        ("Mean conf", conf_str,
         "Average self-reported confidence (0–1). "
         "Not necessarily calibrated against actual error."),
        ("Missed", agg["missed_total"],
         "Total true bars the extractor failed to return across all charts."),
        ("Extra", agg["extra_total"],
         "Total predicted bars with no matching true bar (hallucinations)."),
        ("Mean time", f"{agg['mean_sec']:.1f} s",
         "Average extractor wall-clock time per chart."),
        ("p95 time", f"{agg['p95_sec']:.1f} s",
         "95th-percentile per-chart time: 95% of charts finished within this."),
        ("Total time", f"{agg['total_sec']:.1f} s",
         "Total wall-clock time across all charts."),
    ]
    return "".join(
        f'<div class="stat" title="{html.escape(desc)}">'
        f'<div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v, desc in cards)


def _chart_row(chart: dict) -> str:
    mean_pct, max_pct = chart["mean_pct"], chart["max_pct"]
    mean_v = 0 if math.isnan(mean_pct) else mean_pct
    max_v = 0 if math.isnan(max_pct) else max_pct
    tt, tp = chart["type_true"], chart["type_pred"]
    if tt is None:
        type_cell = "—"
    elif tp is None:
        type_cell = "<span style='color:#a00'>—</span>"
    else:
        type_cell = ("✓" if chart["type_matched"]
                     else f"<span style='color:#a00'>{html.escape(str(tp))}</span>")
    conf = chart["confidence"]
    conf_v = "" if conf is None else f"{conf:.2f}"
    conf_sort = conf if conf is not None else 0
    return (
        f"<tr><td>{html.escape(str(chart['image']))}</td>"
        f"<td>{html.escape(str(chart['preset']))}</td>"
        f"<td>{type_cell}</td>"
        f"<td>{chart['density']}</td><td>{chart['labels_on']}</td>"
        f"<td>{chart['n_true']}</td><td>{chart['matched']}</td>"
        f"<td>{chart['missed']}</td><td>{chart['extra']}</td>"
        f'<td data-v="{mean_v}">{_pct(mean_pct)}</td>'
        f'<td data-v="{max_v}">{_pct(max_pct)}</td>'
        f'<td data-v="{conf_sort}">{conf_v}</td>'
        f'<td data-v="{chart["seconds"]}">{chart["seconds"]:.1f} s</td></tr>')


def _pane_heading(title: str, shown: int, total: int, desc: str) -> str:
    return (f"<h2 title=\"{html.escape(desc)}\" style=\"cursor:help\">"
            f"{title} <small>· {shown} of {total} charts</small></h2>")


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def write_html(path: Path, extractor: str, agg: dict, charts: list[dict],
               breakdowns: list[dict], img_dir: Path) -> None:
    """Render the benchmark report to ``path`` from plain data (see module doc)."""
    by_score = sorted(charts, key=lambda c: (_rank_score(c), c["image"]))

    summary = _summary_cards(agg)
    table_rows = "".join(_chart_row(c) for c in charts)
    breakdown_html = "".join(_group_table(b["title"], b["rows"]) for b in breakdowns)
    cards_html = "".join(_detail_card(c, img_dir) for c in by_score)
    dev_img = _dev_bell_png([d for c in charts for d in _signed_devs(c)])

    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Extractor benchmark · {html.escape(extractor)}</title>
<style>{_CSS}</style>
<h1>Extractor benchmark — <code>{html.escape(extractor)}</code></h1>
<div class="stats">{summary}</div>
<p>Error = |predicted − true| as a percentage of the <b>value-axis range</b>
(max − min), matching how a point is read off the axis regardless of its own
magnitude. Recall = bars found / true bars. Precision = correct
(series,category) / predicted. Type acc = correct chart_type / charts with a
type prediction. Mean conf = average self-reported confidence (VLM).</p>

<h2>Breakdowns</h2>
<div class="tables">{breakdown_html}</div>

<h2 title="Distribution of signed per-bar deviation (pred − true) / axis range, across every matched bar. Centred near 0 if the extractor is unbiased; right tail = over-estimates, left tail = under-estimates." style="cursor:help">Deviation distribution</h2>
<div class="devwrap">{dev_img}</div>

{_pane_heading("Charts — best to worst", len(charts), len(charts),
    "Every chart as a card, ordered from lowest combined value error (best) "
    "to highest (worst). Score = mean + max per-bar error (% of axis range) "
    "plus a small label-match penalty; charts that matched no bars sort last.")}
<div class="grid">{cards_html}</div>

<h2>All charts <small>(click a header to sort)</small></h2>
<table class="full" id="t"><thead><tr>
 <th>image</th><th>preset</th><th>type</th>
 <th data-num="1">d</th><th>labels</th>
 <th data-num="1">true</th><th data-num="1">matched</th>
 <th data-num="1">missed</th><th data-num="1">extra</th>
 <th data-num="1">mean err</th><th data-num="1">max err</th>
 <th data-num="1">conf</th><th data-num="1">time</th></tr></thead>
 <tbody>{table_rows}</tbody></table>
<script>{_SORT_JS}</script>
""", encoding="utf-8")
