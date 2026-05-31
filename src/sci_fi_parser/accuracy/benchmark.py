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
    benchmark --data train_data/synthetic --out reports/run1
    benchmark --data /tmp/sweep --extractor noisy-oracle

(Installed as a ``[project.scripts]`` entry point via ``uv sync``. The module
also exposes ``main`` so ``python -m sci_fi_parser.accuracy.benchmark`` works.)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from sci_fi_parser.vlm.vlm import ChatCompletionsVLM, OllamaVLM
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile
from sci_fi_parser.schema import (
    ChartData, ChartType, Extractor, Point, Series,
)


def _cat_key(x: str | float) -> str:
    """String key for category matching; integer-valued floats lose the .0."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def normalize_key(series: str, category: str) -> tuple[str, str]:
    """Whitespace- and case-insensitive form of a (series, category) match key."""
    return (series.strip().casefold(), category.strip().casefold())


def series_map(chart: ChartData) -> dict[tuple[str, str], float]:
    """Flatten ChartData to ``{(series_name, category): value}`` for matching.

    Normalises integer-valued category keys (``2018.0`` -> ``"2018"``) so a
    VLM returning JSON numbers matches truth labels stored as strings.
    """
    return {(s.name, _cat_key(p.x)): float(p.y)
            for s in chart.series for p in s.points}


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
    """Harness sanity-check — run before any real model to confirm the pipeline works.

    Returns true values perturbed by Gaussian noise plus occasional missed/extra
    bars. If the oracle scores ~3% error (rel_noise default), the scoring,
    report, and JSONL output are all wired up correctly. A score of 0% or
    wildly off means a harness bug, not a model problem. Not a substitute for a
    real extractor.

    Noise is scaled by the value-axis span so the oracle degrades consistently
    across charts regardless of their units. ``value_range`` in labels.jsonl
    must reflect the actual data range — if it doesn't, the oracle looks
    artificially perfect and the check is useless.
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

    def extract(self, image_path: Path, prompt_suffix: str = "") -> ChartData:
        entry = self._truth[image_path.name]
        lo, hi = entry["value_range"]
        span = abs(hi - lo) or 1.0
        out_series = [self._perturb(s, lo, hi, span) for s in entry["series"]]
        conf = float(np.clip(self._rng.normal(0.9, 0.05), 0, 1))
        return ChartData(chart_type=entry.get("chart_type"),
                         series=out_series, confidence=conf)


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
    # Positional-matching diagnostics (pair true[i] with pred[i] by emission
    # order, regardless of label correctness). Lets us see value-reading skill
    # independently of label-reading skill -- a model that reads heights
    # perfectly but mis-transcribes categories will have low value_errors_pos
    # AND high misaligned. Both pad against min(n_true, n_pred); bar_count_err
    # captures the size mismatch separately so it isn't conflated with value
    # error (see docs/discussion: padded-100% would re-create the conflation).
    value_errors_pos: list[float] = field(default_factory=list)  # per paired position
    misaligned: int = 0                # paired positions whose labels disagree
    n_paired_pos: int = 0              # min(n_true, n_pred); paired-bar count
    bar_count_err: int = 0             # abs(n_pred - n_true): missed + extra
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


def _positional_score(tmap: dict[tuple[str, str], float],
                      pmap: dict[tuple[str, str], float]
                      ) -> tuple[list[float], int, int]:
    """Pair true[i] with pred[i] by emission order (== visual order for VLMs
    reading left to right). Returns (value_errors_per_position, misaligned,
    n_paired). Misaligned counts paired positions where the label tuples
    disagree -- height read correct but attached to the wrong category.
    """
    titems = list(tmap.items())
    pitems = list(pmap.items())
    n_paired = min(len(titems), len(pitems))
    errors_pos: list[float] = []
    misaligned = 0
    for i in range(n_paired):
        (tk, tv), (pk, pv) = titems[i], pitems[i]
        errors_pos.append(_pct_of_true(pv, tv))
        if normalize_key(*tk) != normalize_key(*pk):
            misaligned += 1
    return errors_pos, misaligned, n_paired


def score_chart(image: str, entry: dict, pred: ChartData) -> ChartResult:
    """Match predicted to true bars; error = |pred-true| / |true| as a percentage.

    Matching is whitespace- and case-insensitive on the (series, category) key,
    so 'Region A' / 'region a' / 'Region A ' all line up. For single-series
    charts where the VLM couldn't read off a series name we already rekey via
    `_align_series_names`; this normalization layers on top.

    A second, *positional* scoring also runs (pair-by-index) and feeds the
    value_*/misaligned fields. Identity-based stats answer "given the model
    found this bar, was the value right"; positional stats answer "ignoring
    label-reading, was the value right." The two diverge when a model reads
    heights well but mis-transcribes the x-axis -- see ChartResult docs.
    """
    tmap = truth_to_map(entry["series"])
    pmap = _align_series_names(tmap, series_map(pred))
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
    errors_pos, misaligned, n_paired = _positional_score(tmap, pmap)
    return ChartResult(image, len(tmap), len(pmap), matched, len(tmap) - matched,
                       extra, errors,
                       value_errors_pos=errors_pos,
                       misaligned=misaligned,
                       n_paired_pos=n_paired,
                       bar_count_err=abs(len(pmap) - len(tmap)),
                       truth=tmap, pred=pmap, meta=entry.get("meta", {}),
                       type_true=entry.get("chart_type"),
                       type_pred=pred.chart_type,
                       confidence=pred.confidence)


def aggregate(results: list[ChartResult]) -> dict:
    all_err = np.array([e for r in results for e in r.errors_pct], dtype=float)
    all_val = np.array([e for r in results for e in r.value_errors_pos],
                       dtype=float)
    secs = np.array([r.seconds for r in results], dtype=float)
    confs = np.array([r.confidence for r in results
                      if r.confidence is not None], dtype=float)
    total_true = sum(r.n_true for r in results) or 1
    total_pred = sum(r.n_pred for r in results) or 1
    total_matched = sum(r.matched for r in results)
    total_paired_pos = sum(r.n_paired_pos for r in results) or 1
    total_misaligned = sum(r.misaligned for r in results)
    n_type_known = sum(1 for r in results
                       if r.type_true is not None and r.type_pred is not None)
    n_type_match = sum(1 for r in results if r.type_matched)
    return {
        "n_charts": len(results),
        "n_bars_true": int(total_true),
        "mean_pct": float(all_err.mean()) if all_err.size else float("nan"),
        "median_pct": float(np.median(all_err)) if all_err.size else float("nan"),
        "p95_pct": float(np.percentile(all_err, 95)) if all_err.size else float("nan"),
        # Positional (value-only) scoring — ignores label correctness.
        "value_mean_pct": float(all_val.mean()) if all_val.size else float("nan"),
        "value_max_pct": float(all_val.max()) if all_val.size else float("nan"),
        "value_median_pct": float(np.median(all_val)) if all_val.size else float("nan"),
        "misalignment_pct": total_misaligned / total_paired_pos,
        "bar_count_err_total": int(sum(r.bar_count_err for r in results)),
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
       min-width:90px;text-align:center;cursor:help}
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
        f"<td>{_pct(err)}</td><td>{rec*100:.1f}%</td><td>{sec:.1f} s</td></tr>"
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
        mean {_pct(r.mean_pct)} · max {_pct(r.max_pct)} · {r.seconds:.1f} s
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
        ("Mean error", _pct(agg["mean_pct"]),
         "Mean per-bar error across all matched bars. "
         "Error = |predicted − true| as a percentage of the true value."),
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
        f'<td data-v="{mean_v}">{_pct(r.mean_pct)}</td>'
        f'<td data-v="{max_v}">{_pct(r.max_pct)}</td>'
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


def build_extractor(name: str, truth: dict, rng: np.random.Generator,
                    profile: VLMProfile | None = None) -> Extractor:
    if name == "noisy-oracle":
        return NoisyOracle(truth, rng)
    if name == "ollama":
        return OllamaVLM(profile=profile)
    if name.startswith("ollama:"):
        return OllamaVLM(profile=profile, model_override=name.split(":", 1)[1])
    if name == "api":
        return ChatCompletionsVLM(profile=profile)
    if name.startswith("api:"):
        return ChatCompletionsVLM(profile=profile, model_override=name.split(":", 1)[1])
    raise SystemExit(
        f"unknown extractor {name!r} (try: noisy-oracle, ollama, ollama:<model>, "
        "api, api:<model>)")


def _resolve_profile(config_arg: Path | None) -> VLMProfile:
    """CLI flag > config/vlm.toml in cwd > built-in default."""
    if config_arg is not None:
        return load_profile(config_arg)
    default_path = Path("config/vlm.toml")
    if default_path.exists():
        return load_profile(default_path)
    return VLMProfile()


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
    print(f"  time per chart     : mean {agg['mean_sec']:.1f} s · "
          f"p95 {agg['p95_sec']:.1f} s · total {agg['total_sec']:.1f} s")
    print("\n  by preset (err / recall / time):")
    for k, n, err, rec, sec in group_summary(results, "preset"):
        print(f"    {str(k):14s} {_pct(err):>8}  recall {rec*100:3.0f}%  "
              f"{sec:6.1f} s  ({n})")
    print(f"\n  report -> {out / 'report.html'}")
    print(f"  json   -> {out / 'results.json'}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="synthetic dataset dir (images/ + labels.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("reports/latest"))
    ap.add_argument("--extractor", default="noisy-oracle",
                    help="noisy-oracle | ollama | ollama:<model> | api | api:<model> "
                         "(ollama/api use the profile's model; :<tag> overrides it; "
                         "api needs backend/base_url set in the profile)")
    ap.add_argument("--vlm-config", type=Path, default=None,
                    help="VLM profile TOML (default: config/vlm.toml if present)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    return ap.parse_args()


def run_benchmark(*, data: Path, out: Path, extractor_name: str = "noisy-oracle",
                  profile: VLMProfile | None = None,
                  seed: int = 0, limit: int | None = None,
                  print_summary: bool = True) -> dict:
    """End-to-end run: load truth, score, write report.html + results.json.

    Returns the aggregate dict. Public entry point so other tools (e.g. the
    cross-model comparison runner) can drive it without going through argparse.
    """
    truth = load_truth(data)
    images = sorted(truth)[:limit] if limit else sorted(truth)
    rng = np.random.default_rng(seed)
    extractor = build_extractor(extractor_name, truth, rng, profile=profile)
    img_dir = data / "images"

    results = _run_extractor(extractor, truth, images, img_dir)
    agg = aggregate(results)
    out.mkdir(parents=True, exist_ok=True)
    _write_results_json(out, extractor, agg, results)
    write_html(out / "report.html", extractor.name, agg, results, img_dir)
    if print_summary:
        _print_summary(extractor, agg, results, out)
    return agg


def main() -> None:
    args = _parse_args()
    profile = _resolve_profile(args.vlm_config)
    run_benchmark(
        data=args.data, out=args.out,
        extractor_name=args.extractor,
        profile=profile,
        seed=args.seed, limit=args.limit,
    )


if __name__ == "__main__":
    main()
