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
extractor by implementing ``extract(image_path) -> (chartdata_dict, raw)`` plus a
``name``, and adding the class to the ``Extractor`` union.

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

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np

from sci_fi_parser.benchmark.truth import ChartTruth
from sci_fi_parser.vlm.vlm import ChatCompletionsVLM
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile
from sci_fi_parser.vlm.vlm_schema import (
    ChartData,
    ChartType,
    Point,
    Series,
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
    return {(s.name, _cat_key(p.x)): float(p.y) for s in chart.series for p in s.points}


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #
class NoisyOracle:
    """Harness sanity-check — run before any real model to confirm the pipeline works.

    Returns true values perturbed by Gaussian noise plus occasional missed/extra
    bars. If the oracle scores ~3% error (rel_noise default), the scoring,
    report, and JSONL output are all wired up correctly. A score of 0% or
    wildly off means a harness bug, not a model problem. Not a substitute for a
    real extractor.

    Noise is scaled by the value-axis span so the oracle degrades consistently
    across charts regardless of their units. ``data_range`` in truth.jsonl
    must reflect the actual data range -- if it doesn't, the oracle looks
    artificially perfect and the check is useless.
    """

    def __init__(
        self,
        truth_by_image: dict[str, ChartTruth],
        rng: np.random.Generator,
        rel_noise: float = 0.03,
        miss_p: float = 0.05,
        extra_p: float = 0.06,
    ):
        self._model = "noisy-oracle"
        self._truth = truth_by_image
        self._rng = rng
        self._rel_noise = rel_noise
        self._miss_p = miss_p
        self._extra_p = extra_p
        # Mock-data mode: BENCH_ORACLE_MOCK=1 makes the oracle ignore the small
        # noise and instead fake wildly-varying predictions -- each chart gets its
        # own random bias + spread (plus rare blow-out outliers), so the report
        # shows many different error patterns to analyse (see _mock_series).
        self._mock = bool(os.environ.get("BENCH_ORACLE_MOCK"))

    def _perturb(self, series: Series, lo: float, hi: float, span: float) -> Series:
        """One ground-truth series -> a noisy prediction (some points dropped/added)."""
        pts = []
        for point in series.points:
            if self._rng.random() < self._miss_p:
                continue  # simulate a missed bar
            noisy = float(point.y) + self._rng.normal(0, self._rel_noise * span)
            pts.append(Point(x=str(point.x), y=round(noisy, 3)))
        if self._rng.random() < self._extra_p:  # simulate a hallucinated bar
            pts.append(Point(x="GHOST", y=round(self._rng.uniform(lo, hi), 3)))
        return Series(name=series.name, points=pts)

    def _mock_series(self, series: Series, bias: float, spread: float) -> Series:
        """Fake a widely-varying prediction for one series: each bar's signed
        deviation is drawn from N(bias, spread) %, with ~10% of bars getting an
        extra blow-out outlier. pred = true * (1 + dev/100). Occasional dropped /
        hallucinated bars are kept so recall/precision vary too.
        """
        vals = [float(p.y) for p in series.points]
        scale = (sum(abs(v) for v in vals) / len(vals)) if vals else 1.0
        pts = []
        for point in series.points:
            if self._rng.random() < self._miss_p:
                continue  # dropped bar
            dev = self._rng.normal(bias, spread)
            if self._rng.random() < 0.10:  # rare blow-out outlier
                dev += self._rng.normal(0, 250)
            pts.append(
                Point(
                    x=str(point.x),
                    y=round(float(point.y) * (1.0 + dev / 100.0), 3),
                )
            )
        if self._rng.random() < self._extra_p:  # hallucinated bar
            pts.append(Point(x="GHOST", y=round(scale * self._rng.uniform(0.2, 1.5), 3)))
        return Series(name=series.name, points=pts)

    def extract(self, image_path: Path, prompt_suffix: str = "") -> tuple[dict, dict]:
        entry = self._truth[image_path.name]
        if self._mock:
            # Each chart gets its own personality: a random overall bias and a
            # random spread, so different charts read tight / noisy / skewed.
            bias = float(self._rng.uniform(-50, 50))
            spread = float(self._rng.uniform(5, 80))
            out_series = [self._mock_series(s, bias, spread) for s in entry.series]
            chart = ChartData(chart_type=entry.chart_type or "none", log_scale=False, series=out_series)
            return chart.model_dump(), {}
        lo, hi = entry.data_range
        span = abs(hi - lo) or 1.0
        out_series = [self._perturb(s, lo, hi, span) for s in entry.series]
        chart = ChartData(chart_type=entry.chart_type or "none", log_scale=False, series=out_series)
        return chart.model_dump(), {}


Extractor = ChatCompletionsVLM | NoisyOracle


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChartResult:
    image: str
    n_true: int
    n_pred: int
    matched: int
    missed: int  # true bars the extractor did not return
    extra: int  # predicted bars with no matching (series,cat)
    errors_pct: list[float] = field(default_factory=list)  # per matched bar, % of axis span
    # Positional-matching diagnostics (pair true[i] with pred[i] by emission
    # order, regardless of label correctness). Lets us see value-reading skill
    # independently of label-reading skill -- a model that reads heights
    # perfectly but mis-transcribes categories will have low value_errors_pos
    # AND high misaligned. Both pad against min(n_true, n_pred); bar_count_err
    # captures the size mismatch separately so it isn't conflated with value
    # error (see docs/discussion: padded-100% would re-create the conflation).
    value_errors_pos: list[float] = field(default_factory=list)  # per paired position
    misaligned: int = 0  # paired positions whose labels disagree
    n_paired_pos: int = 0  # min(n_true, n_pred); paired-bar count
    bar_count_err: int = 0  # abs(n_pred - n_true): missed + extra
    span: float = 1.0  # value-axis range (max - min), error denominator
    truth: dict = field(default_factory=dict)
    pred: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    seconds: float = 0.0  # extractor wall-clock time for this image
    type_true: ChartType | None = None
    type_pred: ChartType | None = None

    @property
    def mean_pct(self) -> float:
        return mean(self.errors_pct) if self.errors_pct else float("nan")

    @property
    def max_pct(self) -> float:
        return max(self.errors_pct) if self.errors_pct else float("nan")

    @property
    def type_matched(self) -> bool:
        return self.type_true is not None and self.type_pred is not None and self.type_true == self.type_pred


def _align_series_names(
    tmap: dict[tuple[str, str], float], pmap: dict[tuple[str, str], float]
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


def _pct_of_span(pred: float, true: float, span: float) -> float:
    """|pred - true| as a percentage of the value-axis span (max - min).

    Reading a point off a chart is a perceptual error that's roughly constant in
    axis pixels, so it scales with the drawn axis range, not with the point's own
    value. Normalising by span (rather than |true|) keeps small-valued bars from
    showing huge errors for being small and removes the divide-by-zero near zero.
    pred 190 vs true 200 on a 0-220 axis -> 4.55%. ``span`` is guaranteed nonzero
    by the caller.
    """
    return abs(pred - true) / span * 100.0


def _positional_score(
    tmap: dict[tuple[str, str], float], pmap: dict[tuple[str, str], float], span: float
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
        errors_pos.append(_pct_of_span(pv, tv, span))
        if normalize_key(*tk) != normalize_key(*pk):
            misaligned += 1
    return errors_pos, misaligned, n_paired


def aggregate(results: list[ChartResult]) -> dict:
    all_err = np.array([e for r in results for e in r.errors_pct], dtype=float)
    all_val = np.array([e for r in results for e in r.value_errors_pos], dtype=float)
    secs = np.array([r.seconds for r in results], dtype=float)
    total_true = sum(r.n_true for r in results) or 1
    total_pred = sum(r.n_pred for r in results) or 1
    total_matched = sum(r.matched for r in results)
    total_paired_pos = sum(r.n_paired_pos for r in results) or 1
    total_misaligned = sum(r.misaligned for r in results)
    n_type_known = sum(1 for r in results if r.type_true is not None and r.type_pred is not None)
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
        rows.append(
            (
                k,
                len(rs),
                mean(errs) if errs else float("nan"),
                sum(r.matched for r in rs) / tt,
                mean([r.seconds for r in rs]) if rs else 0.0,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Console-summary formatter (the HTML report rendering now lives in draw_lap.py)
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}%"


def build_extractor(
    name: str, truth: dict[str, ChartTruth], rng: np.random.Generator, profile: VLMProfile | None = None
) -> Extractor:
    if name == "noisy-oracle":
        return NoisyOracle(truth, rng)
    if name == "vlm":
        return ChatCompletionsVLM(profile=profile)
    if name.startswith("vlm:"):
        return ChatCompletionsVLM(profile=profile, model_override=name.split(":", 1)[1])
    raise SystemExit(f"unknown extractor {name!r} (try: noisy-oracle, vlm, vlm:<model>)")


def _resolve_profile(config_arg: Path | None) -> VLMProfile:
    """CLI flag > config/vlm.toml in cwd > built-in default."""
    if config_arg is not None:
        return load_profile(config_arg)
    default_path = Path("config/vlm.toml")
    if default_path.exists():
        return load_profile(default_path)
    return VLMProfile()


def _write_results_json(out: Path, extractor: Extractor, agg: dict, results: list[ChartResult]) -> None:
    payload = {
        "extractor": extractor._model,
        "aggregate": agg,
        "by_preset": group_summary(results, "preset"),
        "by_density": group_summary(results, "density"),
        "per_chart": [
            {
                "image": r.image,
                "meta": r.meta,
                "mean_pct": r.mean_pct,
                "max_pct": r.max_pct,
                "matched": r.matched,
                "missed": r.missed,
                "extra": r.extra,
                "type_true": r.type_true,
                "type_pred": r.type_pred,
                "type_matched": r.type_matched,
            }
            for r in results
        ],
    }
    (out / "results.json").write_text(
        json.dumps(
            payload, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o
        ),
        encoding="utf-8",
    )


def _print_summary(extractor: Extractor, agg: dict, results: list[ChartResult], out: Path) -> None:
    errs = f"{_pct(agg['mean_pct'])} / {_pct(agg['median_pct'])} / {_pct(agg['p95_pct'])}"
    type_acc = agg["type_accuracy"]
    type_str = "-" if math.isnan(type_acc) else f"{type_acc * 100:.0f}%"
    print(f"\n  extractor : {extractor._model}")
    print(f"  charts    : {agg['n_charts']}  ({agg['n_bars_true']} bars)")
    print(f"  mean/med/p95 error : {errs}  (% of axis range)")
    print(f"  recall/precision   : {agg['recall'] * 100:.1f}% / {agg['precision'] * 100:.1f}%")
    print(f"  within 1% / 5%     : {agg['within_1pct'] * 100:.0f}% / {agg['within_5pct'] * 100:.0f}%")
    print(f"  type accuracy      : {type_str}")
    print(f"  missed/extra bars  : {agg['missed_total']} / {agg['extra_total']}")
    print(
        f"  time per chart     : mean {agg['mean_sec']:.1f} s · "
        f"p95 {agg['p95_sec']:.1f} s · total {agg['total_sec']:.1f} s"
    )
    print("\n  by preset (err / recall / time):")
    for k, n, err, rec, sec in group_summary(results, "preset"):
        print(f"    {str(k):14s} {_pct(err):>8}  recall {rec * 100:3.0f}%  {sec:6.1f} s  ({n})")
    print(f"\n  report -> {out / 'report.html'}")
    print(f"  json   -> {out / 'results.json'}")


def run_benchmark(
    *,
    data: Path,
    out: Path,
    extractor_name: str = "noisy-oracle",
    profile: VLMProfile | None = None,
    seed: int = 0,
    limit: int | None = None,
    dataset: str = "synthetic",
    prompt_suffixes: dict[str, str] | None = None,
    print_summary: bool = True,
) -> dict:
    """Compatibility wrapper for callers that still import ``run_benchmark``."""
    if prompt_suffixes is not None:
        raise ValueError("prompt_suffixes are only supported by bench_pipeline stages")
    from sci_fi_parser.benchmark.bench_pipeline import run_pipeline

    return run_pipeline(
        data=data,
        out=out,
        extractor_name=extractor_name,
        profile=profile,
        seed=seed,
        limit=limit,
        dataset=dataset,
        print_summary=print_summary,
    )


def main() -> None:
    from sci_fi_parser.benchmark.bench_pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
