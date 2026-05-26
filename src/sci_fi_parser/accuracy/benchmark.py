"""Accuracy benchmark for chart extractors against synthetic ground truth.

Pipeline role -- the MEASUREMENT LAYER
--------------------------------------
Synthetic charts (:mod:`sci_fi_parser.accuracy.synthetic`) have values known *by
construction*, so for any extractor we can compute exact per-chart error. Aggregated,
those errors become the empirical **error margin** we attach to real-world extractions
(which have no ground truth) before they go to SQL.

One canonical schema, many extractors
-------------------------------------
Every extractor -- VLM, CV+OCR pipeline, chart-specialized model -- is adapted to
a single :class:`ChartData` schema, so comparisons are apples-to-apples. Add an
extractor by implementing ``Extractor.extract(image_path) -> ChartData``.

Note: pure OCR is *not* a standalone value extractor (it reads text, not data
points) -- benchmark it as part of a CV+OCR pipeline.

Usage
-----
    python scripts/benchmark.py --data train_data/synthetic --out reports/run1
    python scripts/benchmark.py --data /tmp/sweep --extractor noisy-oracle
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from sci_fi_parser.schema import (
    ChartData, ChartType, Extractor, Point, Series,
    normalize_key, parse_chartdata,
)


def truth_to_map(series: list[dict]) -> dict[tuple[str, str], float]:
    """Ground-truth label2 series -> {(series_name, category): value}."""
    out: dict[tuple[str, str], float] = {}
    for s in series:
        for cat, val in s["points"]:
            out[(s["name"], str(cat))] = float(val)
    return out


# --------------------------------------------------------------------------- #
# Test double + Ollama skeleton (the Extractor protocol lives in sci_fi_parser.schema)
# --------------------------------------------------------------------------- #
class NoisyOracle:
    """TEST double: returns true values perturbed by noise + occasional miss/extra.

    Lets us exercise the harness and report without a real model. Noise is scaled
    by the value-axis span, matching the %-of-span error metric. Not a real model.
    """

    def __init__(self, truth_by_image: dict, rng: np.random.Generator,
                 rel_noise: float = 0.03, miss_p: float = 0.05, extra_p: float = 0.06):
        self.name = "noisy-oracle"
        self._truth = truth_by_image
        self._rng = rng
        self._rel_noise = rel_noise
        self._miss_p = miss_p
        self._extra_p = extra_p

    def _perturb(self, series: dict, lo: float, hi: float, span: float) -> Series:
        """One ground-truth series -> a noisy prediction (some points dropped/added)."""
        pts = []
        for cat, val in series["points"]:
            if self._rng.random() < self._miss_p:
                continue  # simulate a missed bar
            noisy = float(val) + self._rng.normal(0, self._rel_noise * span)
            pts.append(Point(x=str(cat), y=round(noisy, 3)))
        if self._rng.random() < self._extra_p:  # simulate a hallucinated bar
            pts.append(Point(x="GHOST", y=round(self._rng.uniform(lo, hi), 3)))
        return Series(name=series["name"], points=pts)

    def extract(self, image_path: Path) -> ChartData:
        entry = self._truth[image_path.name]
        lo, hi = entry["value_range"]
        span = abs(hi - lo) or 1.0
        out_series = [self._perturb(s, lo, hi, span) for s in entry["series"]]
        conf = float(np.clip(self._rng.normal(0.9, 0.05), 0, 1))
        return ChartData(chart_type=entry.get("chart_type"),
                         series=out_series, confidence=conf)


class OllamaVLM:
    """Skeleton: local VLM via Ollama with schema-ENFORCED JSON output.

    Ollama's ``format=`` accepts a JSON schema and guarantees the reply conforms
    to it -- standardization-at-source for the VLM path. Needs ``pip install
    ollama`` and the model pulled (``ollama pull <model>``).
    """

    PROMPT = (
        "Extract the data from this chart.\n"
        "Rules:\n"
        "- chart_type: one of bar_chart, grouped_bar_chart, stacked_bar_chart, "
        "horizontal_bar_chart, line_chart.\n"
        "- Use the x-axis category labels EXACTLY as printed. Do not invent "
        "dates, years, or names.\n"
        "- Series naming: if there is a legend, use the legend labels. "
        "If there is NO legend (single-series chart), use the y-axis title "
        "as the series name. Never leave the name blank.\n"
        "- Read each y-value from the y-axis scale and the bar / marker height. "
        "If numeric labels are printed on the bars, prefer those.\n"
        "- Watch y-axis units: '200K' = 200000, '1.5M' = 1500000, "
        "'2.3B' = 2300000000. Return plain numbers, no suffixes, no extra zeros.\n"
        "- confidence: a number from 0.0 to 1.0 reflecting how certain you are "
        "that the extracted values are correct. Lower it for charts without "
        "printed value labels or with hard-to-read axes.\n"
        "- Do not output series, categories, or values that do not appear on "
        "the chart."
    )

    def __init__(self, model: str = "qwen2.5vl:7b"):
        self.name = model
        self._model = model
        # tune via env: BENCH_NUM_CTX (context), BENCH_NUM_GPU (layers on GPU)
        self._options = {"num_ctx": int(os.environ.get("BENCH_NUM_CTX", "2048"))}
        if os.environ.get("BENCH_NUM_GPU") is not None:
            self._options["num_gpu"] = int(os.environ["BENCH_NUM_GPU"])

    def extract(self, image_path: Path) -> ChartData:
        import ollama  # pylint: disable=import-outside-toplevel,import-error
        resp = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": self.PROMPT,
                       "images": [str(image_path)]}],
            format=ChartData.model_json_schema(),   # <- guarantees schema-valid JSON
            options=self._options,
        )
        return parse_chartdata(resp["message"]["content"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChartResult:
    image: str
    n_true: int
    n_pred: int
    matched: int
    missed: int                       # true bars the extractor did not return
    extra: int                        # predicted bars with no matching (series,cat)
    errors_pct: list[float] = field(default_factory=list)  # per matched bar, % of |true|
    truth: dict = field(default_factory=dict)
    pred: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    seconds: float = 0.0               # extractor wall-clock time for this image
    type_true: ChartType | None = None
    type_pred: ChartType | None = None
    confidence: float | None = None    # VLM self-reported confidence, if any

    @property
    def mean_pct(self) -> float:
        return mean(self.errors_pct) if self.errors_pct else float("nan")

    @property
    def max_pct(self) -> float:
        return max(self.errors_pct) if self.errors_pct else float("nan")

    @property
    def type_matched(self) -> bool:
        return (self.type_true is not None and self.type_pred is not None
                and self.type_true == self.type_pred)


def _align_series_names(tmap: dict[tuple[str, str], float],
                        pmap: dict[tuple[str, str], float]
                        ) -> dict[tuple[str, str], float]:
    """Rekey `pmap` to use truth series names when the alignment is unambiguous.

    Reason: single-series charts often have no legend, so the VLM can't read off
    a series name and falls back to a default. Demanding an exact series-name
    match in that case throws away all the (correct) per-category values.
    Rule: if BOTH sides have exactly one series, treat them as the same series.
    Multi-series alignment stays strict — order matters when series are distinct.
    """
    truth_names = {s for s, _ in tmap}
    pred_names = {s for s, _ in pmap}
    if len(truth_names) == 1 and len(pred_names) == 1 and truth_names != pred_names:
        (truth_name,) = truth_names
        return {(truth_name, cat): v for (_, cat), v in pmap.items()}
    return pmap


def _pct_of_true(pred: float, true: float) -> float:
    """|pred - true| as a percentage of |true|.

    What researchers expect: pred 400 vs true 300 -> 33%. Zero-truth case is
    bounded (avoids division by zero exploding the aggregate): 0% if pred is
    also 0, 100% otherwise.
    """
    if abs(true) < 1e-12:
        return 0.0 if abs(pred) < 1e-12 else 100.0
    return abs(pred - true) / abs(true) * 100.0


def score_chart(image: str, entry: dict, pred: ChartData) -> ChartResult:
    """Match predicted to true bars; error = |pred-true| / |true| as a percentage.

    Matching is whitespace- and case-insensitive on the (series, category) key,
    so 'Region A' / 'region a' / 'Region A ' all line up. For single-series
    charts where the VLM couldn't read off a series name we already rekey via
    `_align_series_names`; this normalization layers on top.
    """
    tmap = truth_to_map(entry["series"])
    pmap = _align_series_names(tmap, pred.series_map())
    pnorm = {normalize_key(s, c): k for k in pmap for s, c in [k]}

    errors, matched = [], 0
    for tkey, tv in tmap.items():
        pkey = pnorm.get(normalize_key(*tkey))
        if pkey is not None:
            matched += 1
            errors.append(_pct_of_true(pmap[pkey], tv))
    matched_pkeys = {pnorm[normalize_key(*k)] for k in tmap
                     if normalize_key(*k) in pnorm}
    extra = sum(1 for k in pmap if k not in matched_pkeys)
    return ChartResult(image, len(tmap), len(pmap), matched, len(tmap) - matched,
                       extra, errors, tmap, pmap, entry.get("meta", {}),
                       type_true=entry.get("chart_type"),
                       type_pred=pred.chart_type,
                       confidence=pred.confidence)


def aggregate(results: list[ChartResult]) -> dict:
    all_err = np.array([e for r in results for e in r.errors_pct], dtype=float)
    secs = np.array([r.seconds for r in results], dtype=float)
    confs = np.array([r.confidence for r in results
                      if r.confidence is not None], dtype=float)
    total_true = sum(r.n_true for r in results) or 1
    total_pred = sum(r.n_pred for r in results) or 1
    total_matched = sum(r.matched for r in results)
    n_type_known = sum(1 for r in results
                       if r.type_true is not None and r.type_pred is not None)
    n_type_match = sum(1 for r in results if r.type_matched)
    return {
        "n_charts": len(results),
        "n_bars_true": int(total_true),
        "mean_pct": float(all_err.mean()) if all_err.size else float("nan"),
        "median_pct": float(np.median(all_err)) if all_err.size else float("nan"),
        "p95_pct": float(np.percentile(all_err, 95)) if all_err.size else float("nan"),
        "recall": total_matched / total_true,
        "precision": total_matched / total_pred,
        "missed_total": sum(r.missed for r in results),
        "extra_total": sum(r.extra for r in results),
        "within_1pct": float((all_err <= 1).mean()) if all_err.size else float("nan"),
        "within_5pct": float((all_err <= 5).mean()) if all_err.size else float("nan"),
        "type_accuracy": (n_type_match / n_type_known) if n_type_known else float("nan"),
        "mean_confidence": float(confs.mean()) if confs.size else float("nan"),
        "mean_sec": float(secs.mean()) if secs.size else 0.0,
        "median_sec": float(np.median(secs)) if secs.size else 0.0,
        "p95_sec": float(np.percentile(secs, 95)) if secs.size else 0.0,
        "total_sec": float(secs.sum()),
    }


def group_summary(results: list[ChartResult], key: str) -> list[tuple]:
    """(group, n_charts, mean %err, recall, mean_sec) rows grouped by a meta key."""
    groups: dict = defaultdict(list)
    for r in results:
        groups[r.meta.get(key)].append(r)
    rows = []
    for k in sorted(groups, key=lambda x: (x is None, x)):
        rs = groups[k]
        errs = [e for r in rs for e in r.errors_pct]
        tt = sum(r.n_true for r in rs) or 1
        rows.append((k, len(rs), mean(errs) if errs else float("nan"),
                     sum(r.matched for r in rs) / tt,
                     mean([r.seconds for r in rs]) if rs else 0.0))
    return rows


# --------------------------------------------------------------------------- #
# HTML report (dependency-free templating + base64 thumbnails)
# --------------------------------------------------------------------------- #
_CSS = """
 body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a}
 h1{font-size:20px}
 h2{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
 .stats{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
 .stat{background:#f4f6f8;border-radius:10px;padding:12px 16px;
       min-width:90px;text-align:center}
 .stat .v{font-size:22px;font-weight:700} .stat .k{font-size:12px;color:#666}
 .tables{display:flex;flex-wrap:wrap;gap:24px}
 .grid{display:flex;flex-wrap:wrap;gap:14px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:10px;width:320px}
 .card img{width:100%;border-radius:6px} .meta{font-size:12px;margin-top:6px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 .kv td,.kv th{border-bottom:1px solid #eee;padding:2px 6px;text-align:right}
 .kv td:first-child,.kv th:first-child{text-align:left}
 table.full th,table.full td{border-bottom:1px solid #eee;
                             padding:6px 8px;text-align:right}
 table.full th:first-child,table.full td:first-child{text-align:left}
 table.full th{cursor:pointer;background:#f4f6f8;position:sticky;top:0}
"""

# Click-to-sort for the all-charts table. Headers with data-num="1" sort
# numerically (using each cell's data-v if present, else its text).
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


def _thumb_b64(path: Path, width: int = 260) -> str:
    im = Image.open(path)
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pct(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}%"


def _group_table(title: str, rows: list[tuple]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{n}</td>"
        f"<td>{_pct(err)}</td><td>{rec*100:.1f}%</td><td>{sec*1000:.0f} ms</td></tr>"
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
        mean {_pct(r.mean_pct)} · max {_pct(r.max_pct)} · {r.seconds*1000:.0f} ms
        · missed {r.missed} · extra {r.extra}
        <table class="kv">{kv_head}{rows}</table>
      </div>
    </div>"""


def _summary_cards(agg: dict) -> str:
    type_acc = agg["type_accuracy"]
    type_str = "-" if math.isnan(type_acc) else f"{type_acc*100:.0f}%"
    conf = agg["mean_confidence"]
    conf_str = "-" if math.isnan(conf) else f"{conf:.2f}"
    cards = [
        ("Charts", agg["n_charts"]), ("Mean error", _pct(agg["mean_pct"])),
        ("Median", _pct(agg["median_pct"])), ("p95", _pct(agg["p95_pct"])),
        ("Recall", f"{agg['recall']*100:.1f}%"),
        ("Precision", f"{agg['precision']*100:.1f}%"),
        ("≤1%", f"{agg['within_1pct']*100:.0f}%"),
        ("≤5%", f"{agg['within_5pct']*100:.0f}%"),
        ("Type acc", type_str), ("Mean conf", conf_str),
        ("Missed", agg["missed_total"]), ("Extra", agg["extra_total"]),
        ("Mean time", f"{agg['mean_sec']*1000:.0f} ms"),
        ("p95 time", f"{agg['p95_sec']*1000:.0f} ms"),
        ("Total time", f"{agg['total_sec']:.1f} s"),
    ]
    return "".join(
        f'<div class="stat"><div class="v">{v}</div>'
        f'<div class="k">{k}</div></div>'
        for k, v in cards)


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
        f'<td data-v="{mean_v}">{_pct(r.mean_pct)}</td>'
        f'<td data-v="{max_v}">{_pct(r.max_pct)}</td>'
        f'<td data-v="{conf_sort}">{conf_v}</td>'
        f'<td data-v="{r.seconds}">{r.seconds*1000:.0f} ms</td></tr>')


def write_html(path: Path, extractor: str, agg: dict, results: list[ChartResult],
               img_dir: Path, n_show: int = 6) -> None:
    ok = [r for r in results if r.errors_pct]
    best = sorted(ok, key=lambda r: r.mean_pct)[:n_show]
    worst = sorted(ok, key=lambda r: r.mean_pct, reverse=True)[:n_show]

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

<h2>Worst {len(worst)}</h2><div class="grid">{worst_html}</div>
<h2>Best {len(best)}</h2><div class="grid">{best_html}</div>

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


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def load_truth(data_dir: Path) -> dict:
    """image name -> {chart_type, series, value_range, meta} from labels.jsonl.

    `value_range` is clipped to ``[max(0, lo), hi]``: matplotlib's y-axis lower
    bound often pads below zero (e.g. -20 on a positive-only chart), which
    inflates any span-relative metric. We carry the corrected version forward.
    """
    truth = {}
    with (data_dir / "labels.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            l2 = rec["label2"]
            lo, hi = l2["value_range"]
            truth[rec["image"]] = {
                "chart_type": rec.get("label1"),
                "series": l2["series"],
                "value_range": [max(0.0, float(lo)), float(hi)],
                "meta": rec.get("meta", {}),
            }
    return truth


def build_extractor(name: str, truth: dict, rng: np.random.Generator) -> Extractor:
    if name == "noisy-oracle":
        return NoisyOracle(truth, rng)
    if name.startswith("ollama:"):
        return OllamaVLM(name.split(":", 1)[1])
    raise SystemExit(f"unknown extractor {name!r} (try: noisy-oracle, ollama:<model>)")


def _run_extractor(extractor: Extractor, truth: dict, images: list[str],
                   img_dir: Path) -> list[ChartResult]:
    """Run the extractor over every image, timing each, scoring against truth."""
    results: list[ChartResult] = []
    for name in images:
        entry = truth[name]
        t0 = time.perf_counter()
        try:
            pred = extractor.extract(img_dir / name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
            pred = ChartData(chart_type=None, series=[], confidence=None)
        r = score_chart(name, entry, pred)
        r.seconds = time.perf_counter() - t0
        results.append(r)
    return results


def _write_results_json(out: Path, extractor: Extractor, agg: dict,
                        results: list[ChartResult]) -> None:
    payload = {
        "extractor": extractor.name, "aggregate": agg,
        "by_preset": group_summary(results, "preset"),
        "by_density": group_summary(results, "density"),
        "per_chart": [{"image": r.image, "meta": r.meta, "mean_pct": r.mean_pct,
                       "max_pct": r.max_pct, "matched": r.matched,
                       "missed": r.missed, "extra": r.extra,
                       "type_true": r.type_true, "type_pred": r.type_pred,
                       "type_matched": r.type_matched,
                       "confidence": r.confidence} for r in results],
    }
    (out / "results.json").write_text(
        json.dumps(payload, indent=2,
                   default=lambda o: None if isinstance(o, float) and math.isnan(o)
                   else o),
        encoding="utf-8")


def _print_summary(extractor: Extractor, agg: dict, results: list[ChartResult],
                   out: Path) -> None:
    errs = f"{_pct(agg['mean_pct'])} / {_pct(agg['median_pct'])} / {_pct(agg['p95_pct'])}"
    type_acc = agg["type_accuracy"]
    type_str = "-" if math.isnan(type_acc) else f"{type_acc*100:.0f}%"
    conf = agg["mean_confidence"]
    conf_str = "-" if math.isnan(conf) else f"{conf:.2f}"
    print(f"\n  extractor : {extractor.name}")
    print(f"  charts    : {agg['n_charts']}  ({agg['n_bars_true']} bars)")
    print(f"  mean/med/p95 error : {errs}  (% of true value)")
    print(f"  recall/precision   : "
          f"{agg['recall']*100:.1f}% / {agg['precision']*100:.1f}%")
    print(f"  within 1% / 5%     : "
          f"{agg['within_1pct']*100:.0f}% / {agg['within_5pct']*100:.0f}%")
    print(f"  type accuracy      : {type_str}")
    print(f"  mean confidence    : {conf_str}")
    print(f"  missed/extra bars  : {agg['missed_total']} / {agg['extra_total']}")
    print(f"  time per chart     : mean {agg['mean_sec']*1000:.0f} ms · "
          f"p95 {agg['p95_sec']*1000:.0f} ms · total {agg['total_sec']:.1f} s")
    print("\n  by preset (err / recall / time):")
    for k, n, err, rec, sec in group_summary(results, "preset"):
        print(f"    {str(k):14s} {_pct(err):>8}  recall {rec*100:3.0f}%  "
              f"{sec*1000:6.0f} ms  ({n})")
    print(f"\n  report -> {out / 'report.html'}")
    print(f"  json   -> {out / 'results.json'}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="synthetic dataset dir (images/ + labels.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("reports/latest"))
    ap.add_argument("--extractor", default="noisy-oracle",
                    help="noisy-oracle | ollama:<model> (see build_extractor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    truth = load_truth(args.data)
    images = sorted(truth)[: args.limit] if args.limit else sorted(truth)
    rng = np.random.default_rng(args.seed)
    extractor = build_extractor(args.extractor, truth, rng)
    img_dir = args.data / "images"

    results = _run_extractor(extractor, truth, images, img_dir)
    agg = aggregate(results)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_results_json(args.out, extractor, agg, results)
    write_html(args.out / "report.html", extractor.name, agg, results, img_dir)
    _print_summary(extractor, agg, results, args.out)


if __name__ == "__main__":
    main()
