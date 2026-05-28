"""HTML report rendering for the accuracy benchmark.

Split out of :mod:`sci_fi_parser.accuracy.benchmark` so the scoring/aggregation
layer stays free of presentation code. CSS and the sort-table snippet live next
to this module as ``assets/report.css`` and ``assets/report.sort.js`` and are
loaded at import time via :mod:`importlib.resources` -- no runtime templating
dependency.
"""

from __future__ import annotations

import base64
import html
import io
import math
from importlib import resources
from pathlib import Path

from PIL import Image

from sci_fi_parser.accuracy.scoring import (
    ChartResult, _pct_of_true, fmt_pct, group_summary,
)
from sci_fi_parser.schema import normalize_key


def _load_asset(name: str) -> str:
    # Explicit package string (vs ``__package__``) for clarity: the assets
    # directory is data, not a Python sub-package, so naming the anchor
    # avoids any ambiguity about what's being resolved.
    return (resources.files("sci_fi_parser.accuracy")
            / "assets" / name).read_text(encoding="utf-8")


_CSS = _load_asset("report.css")
# Click-to-sort for the all-charts table. Headers with data-num="1" sort
# numerically (using each cell's data-v if present, else its text).
_SORT_JS = _load_asset("report.sort.js")


def _thumb_b64(path: Path, width: int = 260) -> str:
    im = Image.open(path)
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _group_table(title: str, rows: list[tuple]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{n}</td>"
        f"<td>{fmt_pct(err)}</td><td>{rec*100:.1f}%</td><td>{sec:.1f} s</td></tr>"
        for k, n, err, rec, sec in rows
    )
    return (f"<table class='full' style='max-width:560px'><thead><tr><th>{title}</th>"
            f"<th>charts</th><th>mean err</th><th>recall</th>"
            f"<th>mean time</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _pred_cell(r: ChartResult, series: str, cat: str) -> str:
    """Pred value with the % error vs truth (or — if no match)."""
    norm = normalize_key(series, cat)
    for (ps, pc), pv in r.pred.items():
        if normalize_key(ps, pc) == norm:
            err = _pct_of_true(pv, r.truth[(series, cat)])
            return f"{pv:.2f} <span style='color:#888'>({err:.1f}%)</span>"
    return "—"


def _type_chip(r: ChartResult) -> str:
    if r.type_true is None:
        return ""
    if r.type_pred is None:
        return (f" · type={html.escape(r.type_true)} "
                f"<span style='color:#a00'>(no pred)</span>")
    ok = "✓" if r.type_matched else "✗"
    colour = "#0a0" if r.type_matched else "#a00"
    return (f" · type=<span style='color:{colour}'>{ok}</span> "
            f"{html.escape(r.type_pred)} (true: {html.escape(r.type_true)})")


def _conf_chip(r: ChartResult) -> str:
    return f" · conf {r.confidence:.2f}" if r.confidence is not None else ""


def _detail_card(r: ChartResult, img_dir: Path) -> str:
    preset = html.escape(str(r.meta.get("preset", r.meta.get("type", ""))))
    kv_head = "<tr><th>series</th><th>cat</th><th>true</th><th>pred (err)</th></tr>"
    rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{html.escape(c)}</td><td>{tv:.2f}</td>"
        f"<td>{_pred_cell(r, s, c)}</td></tr>"
        for (s, c), tv in r.truth.items()
    )
    return f"""
    <div class="card">
      <img src="data:image/png;base64,{_thumb_b64(img_dir / r.image)}"/>
      <div class="meta">
        <b>{html.escape(r.image)}</b> · {preset}
        · d{r.meta.get('density', '?')} · labels={r.meta.get('labels_on')}
        {_type_chip(r)}{_conf_chip(r)}<br/>
        mean {fmt_pct(r.mean_pct)} · max {fmt_pct(r.max_pct)} · {r.seconds:.1f} s
        · missed {r.missed} · extra {r.extra}
        <table class="kv">{kv_head}{rows}</table>
      </div>
    </div>"""


def _summary_cards(agg: dict) -> str:
    """Build the top-of-report stat strip. Each card has a hover description."""
    type_acc = agg["type_accuracy"]
    type_str = "-" if math.isnan(type_acc) else f"{type_acc*100:.0f}%"
    conf = agg["mean_confidence"]
    conf_str = "-" if math.isnan(conf) else f"{conf:.2f}"
    cards: list[tuple[str, object, str]] = [
        ("Charts", agg["n_charts"],
         "Number of chart images scored in this run."),
        ("Mean error", fmt_pct(agg["mean_pct"]),
         "Mean per-bar error across all matched bars. "
         "Error = |predicted − true| as a percentage of the true value."),
        ("Median", fmt_pct(agg["median_pct"]),
         "Median per-bar error. Less sensitive to outliers than the mean."),
        ("p95", fmt_pct(agg["p95_pct"]),
         "95th-percentile per-bar error: 95% of matched bars are at or below this."),
        ("Recall", f"{agg['recall']*100:.1f}%",
         "Fraction of true bars the extractor returned and matched. "
         "100% = no bars missed."),
        ("Precision", f"{agg['precision']*100:.1f}%",
         "Fraction of predicted bars that matched a true bar. "
         "100% = no hallucinated extras."),
        ("≤1%", f"{agg['within_1pct']*100:.0f}%",
         "Share of matched bars whose error is within 1% of the true value."),
        ("≤5%", f"{agg['within_5pct']*100:.0f}%",
         "Share of matched bars whose error is within 5% of the true value."),
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


def _chart_row(r: ChartResult) -> str:
    mean_v = 0 if math.isnan(r.mean_pct) else r.mean_pct
    max_v = 0 if math.isnan(r.max_pct) else r.max_pct
    if r.type_true is None:
        type_cell = "—"
    elif r.type_pred is None:
        type_cell = "<span style='color:#a00'>—</span>"
    else:
        type_cell = ("✓" if r.type_matched
                     else f"<span style='color:#a00'>{html.escape(r.type_pred)}</span>")
    conf_v = "" if r.confidence is None else f"{r.confidence:.2f}"
    conf_sort = r.confidence if r.confidence is not None else 0
    return (
        f"<tr><td>{html.escape(r.image)}</td>"
        f"<td>{html.escape(str(r.meta.get('preset','')))}</td>"
        f"<td>{type_cell}</td>"
        f"<td>{r.meta.get('density','')}</td><td>{r.meta.get('labels_on')}</td>"
        f"<td>{r.n_true}</td><td>{r.matched}</td><td>{r.missed}</td><td>{r.extra}</td>"
        f'<td data-v="{mean_v}">{fmt_pct(r.mean_pct)}</td>'
        f'<td data-v="{max_v}">{fmt_pct(r.max_pct)}</td>'
        f'<td data-v="{conf_sort}">{conf_v}</td>'
        f'<td data-v="{r.seconds}">{r.seconds:.1f} s</td></tr>')


def _rank_score(r: ChartResult) -> float:
    """Composite score for ranking charts; higher = worse.

    Priority order (per researcher feedback):
    1. Per-bar value accuracy: ``mean_pct + max_pct`` (% of true value).
       Max is included so a single 50%-off bar doesn't get washed out by a
       bunch of near-perfect ones.
    2. Label match: small penalty scaled by the fraction of bars that
       missed or were hallucinated.

    Zero-match charts get ``+inf`` so a catastrophic failure (recall 0%)
    bubbles to the top of Worst even though it has no value-error data.
    """
    if not r.errors_pct:
        return float("inf")
    value_error = r.mean_pct + r.max_pct
    denom = r.n_true + r.extra or 1
    label_loss = (r.missed + r.extra) / denom * 5.0
    return value_error + label_loss


def _pane_heading(title: str, shown: int, total: int, desc: str) -> str:
    return (f"<h2 title=\"{html.escape(desc)}\" style=\"cursor:help\">"
            f"{title} <small>· {shown} of {total} charts</small></h2>")


def write_html(path: Path, extractor: str, agg: dict, results: list[ChartResult],
               img_dir: Path, n_show: int = 6) -> None:
    by_score = sorted(results, key=lambda r: (_rank_score(r), r.image))
    n = min(n_show, len(by_score))
    best = by_score[:n]
    # Avoid overlap on small runs: worst takes from the *other* end and skips
    # anything already in best.
    worst = [r for r in reversed(by_score) if r not in best][:n]

    summary = _summary_cards(agg)
    table_rows = "".join(_chart_row(r) for r in results)
    by_preset = _group_table("by preset", group_summary(results, "preset"))
    by_density = _group_table("by density", group_summary(results, "density"))
    by_labels = _group_table("by labels-on", group_summary(results, "labels_on"))
    worst_html = "".join(_detail_card(r, img_dir) for r in worst)
    best_html = "".join(_detail_card(r, img_dir) for r in best)

    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Extractor benchmark · {html.escape(extractor)}</title>
<style>{_CSS}</style>
<h1>Extractor benchmark — <code>{html.escape(extractor)}</code></h1>
<div class="stats">{summary}</div>
<p>Error = |predicted − true| as a percentage of the <b>true value</b>
(clamped to 100 % when the true value is zero, so "pred 400 vs true 300" is
~33 %). Recall = bars found / true bars. Precision = correct
(series,category) / predicted. Type acc = correct chart_type / charts with a
type prediction. Mean conf = average self-reported confidence (VLM).</p>

<h2>Breakdowns</h2>
<div class="tables">{by_preset}{by_density}{by_labels}</div>

{_pane_heading("Worst", len(worst), len(results),
    "Ranked by combined value error (mean + max, % of true value) "
    "plus a small label-match penalty. "
    "Charts that matched no bars (recall 0%) are listed first.")}
<div class="grid">{worst_html}</div>
{_pane_heading("Best", len(best), len(results),
    "Same composite score as Worst, ascending. "
    "Charts with the lowest combined value error and the cleanest "
    "label match are ranked first.")}
<div class="grid">{best_html}</div>

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
