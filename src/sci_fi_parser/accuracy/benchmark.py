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

Module layout
-------------
* :mod:`.scoring`  -- per-chart scoring + aggregation (``score_chart``, ``aggregate``).
* :mod:`.report`   -- HTML output.
* this module      -- extractors, CLI, runner that glues the other two together.

Usage
-----
    benchmark --data train_data/synthetic --out reports/run1
    benchmark --data /tmp/sweep --extractor noisy-oracle

(Installed as a ``[project.scripts]`` entry point via ``uv sync``. The module
also exposes ``main`` so ``python -m sci_fi_parser.accuracy.benchmark`` works.)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from sci_fi_parser.accuracy.report import write_html
from sci_fi_parser.accuracy.scoring import (
    ChartResult, aggregate, fmt_pct, group_summary, score_chart,
)
from sci_fi_parser.accuracy.vlm import OllamaVLM
from sci_fi_parser.accuracy.vlm_config import VLMProfile, load_profile
from sci_fi_parser.schema import ChartData, Extractor, Point, Series

# Re-exported so external callers (tests, vlm_compare) keep working after the
# scoring extraction. Keep this list in sync with the public surface.
__all__ = [
    "ChartResult", "NoisyOracle", "aggregate", "group_summary", "main",
    "run_benchmark", "score_chart",
]


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

    def extract(self, image_path: Path) -> ChartData:
        entry = self._truth[image_path.name]
        lo, hi = entry["value_range"]
        span = abs(hi - lo) or 1.0
        out_series = [self._perturb(s, lo, hi, span) for s in entry["series"]]
        conf = float(np.clip(self._rng.normal(0.9, 0.05), 0, 1))
        return ChartData(chart_type=entry.get("chart_type"),
                         series=out_series, confidence=conf)


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
    raise SystemExit(
        f"unknown extractor {name!r} (try: noisy-oracle, ollama, ollama:<model>)")


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
    errs = (f"{fmt_pct(agg['mean_pct'])} / {fmt_pct(agg['median_pct'])} / "
            f"{fmt_pct(agg['p95_pct'])}")
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
        print(f"    {str(k):14s} {fmt_pct(err):>8}  recall {rec*100:3.0f}%  "
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
                    help="noisy-oracle | ollama | ollama:<model> "
                         "(ollama uses the profile's model; ollama:<tag> overrides it)")
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
