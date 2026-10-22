"""End-to-end benchmark: run the parsing pipeline on annotated chart images and score the VLM output.

The images go through the same stage functions as ``parse_folder`` (classification,
OCR/CV, VLM), so prompts and stage behaviour match a real parsing run. With
classification disabled, the chart class is taken from the truth instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sci_fi_parser.api import DEFAULT_VLM_CONFIG
from sci_fi_parser.benchmark.dataset import DatasetKind, Sample, load_samples
from sci_fi_parser.benchmark.report import write_report
from sci_fi_parser.benchmark.scoring import ChartScore, breakdowns, score_chart, summarize
from sci_fi_parser.classifier.classifier_pipeline import start_classification
from sci_fi_parser.classifier.image_classifier import DoclingClassifier
from sci_fi_parser.object_detection.detection_pipeline import start_ocr
from sci_fi_parser.object_detection.ocr import Ocr
from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm import ChatCompletionsVLM
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile
from sci_fi_parser.vlm.vlm_pipeline import start_vlm
from sci_fi_parser.vlm.vlm_schema import ChartData

DEFAULT_PROFILE_PATH = Path(str(DEFAULT_VLM_CONFIG))

# Synthetic metadata keys that get their own breakdown table.
BREAKDOWN_KEYS = ("preset", "density", "labels_on")

# Classifier label the pipeline would need to send a chart of this true type to the VLM.
_CLASS_BY_TRUTH = {
    "vertical_bar": "bar_chart",
    "grouped_bar": "bar_chart",
    "stacked_bar": "bar_chart",
    "horizontal_bar": "bar_chart",
    "line": "line_chart",
}


def run_stages(
    samples: list[Sample], profile: VLMProfile, *, classify: bool, ocr: bool
) -> tuple[ImageSet, dict]:
    """Run the pipeline stages over the samples.

    Returns:
        The populated image set, keyed by image file name, and wall-clock seconds per stage.
    """

    image_set = ImageSet()
    for sample in samples:
        image_set.add_extracted_image(sample.image_path.name, sample.image_path)

    seconds = {}
    started = time.perf_counter()
    if classify:
        start_classification(image_set, DoclingClassifier())
        seconds["classification"] = time.perf_counter() - started
    else:
        for sample in samples:
            label = _CLASS_BY_TRUTH.get(sample.chart_type or "", "")
            image_set.add_classification_result(sample.image_path.name, label)

    if ocr:
        started = time.perf_counter()
        start_ocr(image_set, Ocr())
        seconds["ocr"] = time.perf_counter() - started

    started = time.perf_counter()
    start_vlm(image_set, ChatCompletionsVLM(profile))
    seconds["vlm"] = time.perf_counter() - started
    return image_set, seconds


def score_image_set(samples: list[Sample], image_set: ImageSet) -> list[ChartScore]:
    """Score each sample's VLM result. Charts the VLM skipped or failed on count as no prediction."""

    scores = []
    for sample in samples:
        try:
            pred = ChartData.model_validate(image_set.get_vlm_result(sample.image_path.name))
        except ValidationError:
            pred = None
        scores.append(score_chart(sample, pred))
    return scores


def run_benchmark(
    data: Path,
    out: Path,
    *,
    dataset: DatasetKind = "synthetic",
    profile: VLMProfile | None = None,
    limit: int | None = None,
    classify: bool = True,
    ocr: bool = True,
) -> dict[str, Any]:
    """Run, score, and write ``results.json`` and ``report.html`` to ``out``.

    Returns:
        The results payload written to ``results.json``.
    """

    profile = profile or load_profile(DEFAULT_PROFILE_PATH)
    samples = load_samples(data, dataset)[:limit]
    if not samples:
        raise ValueError(f"no annotated images found in {data}")

    run = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": version("sci-fi-parser"),
        "data": str(data),
        "dataset": dataset,
        "limit": limit,
        "stages": {"classify": classify, "ocr": ocr},
        "vlm": {"model": profile.model, "base_url": profile.base_url, "prompt": profile.prompt},
    }
    image_set, run["seconds"] = run_stages(samples, profile, classify=classify, ocr=ocr)
    scores = score_image_set(samples, image_set)
    summary = summarize(scores)
    groups = breakdowns(scores, BREAKDOWN_KEYS)

    payload = {
        "run": run,
        "summary": summary,
        "breakdowns": groups,
        "charts": [
            {
                "image": score.image_path.name,
                "type_true": score.type_true,
                "type_pred": score.type_pred,
                "points": len(score.points),
                "matched": score.matched,
                "missed": score.missed,
                "extra": score.extra,
                "mean_pct": score.mean_pct,
                "max_pct": score.max_pct,
                "metadata": score.metadata,
            }
            for score in scores
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_report(out / "report.html", run, summary, groups, scores)
    return payload


def _print_summary(payload: dict[str, Any], out: Path) -> None:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}%"

    def share(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    s = payload["summary"]
    print(f"\n  model              : {payload['run']['vlm']['model']}")
    print(f"  charts             : {s['charts']} ({s['no_prediction']} without prediction)")
    print(f"  true points        : {s['points']}")
    print(f"  mean/med/p95 error : {pct(s['mean_pct'])} / {pct(s['median_pct'])} / {pct(s['p95_pct'])}")
    print(f"  recall/precision   : {share(s['recall'])} / {share(s['precision'])}")
    print(f"  missed/extra       : {s['missed']} / {s['extra']}")
    print(f"  type accuracy      : {share(s['type_accuracy'])}")
    print(f"\n  report -> {out / 'report.html'}")
    print(f"  json   -> {out / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench-pipeline", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset dir with images/")
    parser.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    parser.add_argument("-o", "--out", type=Path, default=Path("reports/benchmark"))
    parser.add_argument("--vlm-config", type=Path, default=DEFAULT_PROFILE_PATH, help="VLM profile TOML")
    parser.add_argument("--model", help="override the model named in the VLM profile")
    parser.add_argument("--limit", type=int, help="only the first N charts, by file name")
    parser.add_argument("--no-classify", action="store_true", help="take chart classes from the truth")
    parser.add_argument("--no-ocr", action="store_true", help="send no OCR/CV context to the VLM")
    args = parser.parse_args()

    profile = load_profile(args.vlm_config)
    if args.model:
        profile = dataclasses.replace(profile, model=args.model)
    payload = run_benchmark(
        args.data,
        args.out,
        dataset=args.dataset,
        profile=profile,
        limit=args.limit,
        classify=not args.no_classify,
        ocr=not args.no_ocr,
    )
    _print_summary(payload, args.out)


if __name__ == "__main__":
    main()
