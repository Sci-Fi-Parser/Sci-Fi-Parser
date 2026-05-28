"""Per-chart scoring and aggregation for the accuracy benchmark.

Split out of :mod:`sci_fi_parser.accuracy.benchmark` so the scoring layer is
importable without dragging in the runner/CLI, and so the HTML layer in
:mod:`sci_fi_parser.accuracy.report` can depend on it without creating a cycle
back to ``benchmark`` (which itself wires the runner on top of these helpers).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from sci_fi_parser.schema import ChartData, ChartType, normalize_key


def truth_to_map(series: list[dict]) -> dict[tuple[str, str], float]:
    """Ground-truth label2 series -> {(series_name, category): value}."""
    out: dict[tuple[str, str], float] = {}
    for s in series:
        for cat, val in s["points"]:
            out[(s["name"], str(cat))] = float(val)
    return out


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


def fmt_pct(x: float) -> str:
    """Format a percentage for human display; NaN -> '-'."""
    return "-" if math.isnan(x) else f"{x:.2f}%"
