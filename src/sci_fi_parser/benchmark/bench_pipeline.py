"""MVP staged benchmark pipeline.

This module only orchestrates stages. Scoring, aggregation, extractor building,
and report formatting are reused from the legacy benchmark where possible.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from sci_fi_parser.benchmark import benchmark
from sci_fi_parser.benchmark.report_adapter import (
    value_results_to_breakdowns,
    value_results_to_draw_lap_charts,
)
from sci_fi_parser.benchmark.scoring import (
    aggregate_value_results,
    score_vlm_outputs,
)
from sci_fi_parser.benchmark.truth import (
    ChartTruth,
    load_benetech_truth,
    load_synthetic_truth,
)
from sci_fi_parser.image_extraction.image_loader import load_images_from_folder
from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.vlm.vlm_schema import ChartData


@dataclass(slots=True)
class PipelineInputs:
    image_set: ImageSet
    truth_by_image_id: dict[str, ChartTruth]
    image_dir: Path


DatasetKind = Literal["synthetic", "benetech"]
BenchmarkTarget = Literal["vlm", "ocr-cv"]
OcrModeArgument = Literal["oracle", "detected", "both"]


def load_inputs(
    data: Path,
    limit: int | None = None,
    dataset: DatasetKind = "synthetic",
) -> PipelineInputs:
    def truth_key(path: Path) -> str:
        if dataset == "synthetic":
            return path.name
        if dataset == "benetech":
            return path.stem

    if dataset == "synthetic":
        truth_by_key, metadata_by_key = load_synthetic_truth(data)
    elif dataset == "benetech":
        truth_by_key, metadata_by_key = load_benetech_truth(data)
    else:
        raise ValueError(f"unknown benchmark dataset: {dataset}")

    img_dir = data / "images"

    image_set = ImageSet()
    load_images_from_folder(img_dir, image_set)

    image_ids = [
        image_id
        for image_id, _ in sorted(image_set.items(), key=lambda item: image_set.get_image_path(item[0]).name)
        if truth_key(image_set.get_image_path(image_id)) in truth_by_key
    ]
    if limit:
        image_ids = image_ids[:limit]

    truth_by_image_id: dict[str, ChartTruth] = {}
    for image_id in image_ids:
        key = truth_key(image_set.get_image_path(image_id))
        truth_by_image_id[image_id] = truth_by_key[key]
        image_set.get(image_id)["metadata"]["benchmark"] = metadata_by_key.get(key, {})

    return PipelineInputs(
        image_set=image_set,
        truth_by_image_id=truth_by_image_id,
        image_dir=img_dir,
    )


def run_classification_stage(inputs: PipelineInputs) -> None:
    from sci_fi_parser.classifier.classifier_pipeline import start_classification
    from sci_fi_parser.classifier.image_classifier import DoclingClassifier

    start_classification(inputs.image_set, DoclingClassifier())


def run_ocr_cv_context_stage(inputs: PipelineInputs) -> None:
    """Populate detected OCR/CV context for eligible VLM benchmark images."""

    from sci_fi_parser.object_detection.detection_pipeline import start_ocr
    from sci_fi_parser.object_detection.ocr import Ocr

    scoped = ImageSet()
    for image_id in inputs.truth_by_image_id:
        truth = inputs.truth_by_image_id[image_id]
        record = inputs.image_set.get(image_id)
        if not record["classification"]["result"] and truth.chart_type == "vertical_bar":
            record["classification"]["result"] = "bar_chart"
        scoped.add(image_id, inputs.image_set.get(image_id))
    start_ocr(scoped, Ocr())


def run_ocr_cv_stage(
    *,
    data: Path,
    out: Path,
    dataset: DatasetKind,
    ocr_mode: OcrModeArgument,
    limit: int | None,
) -> dict:
    """Run the component benchmark without classification or VLM construction."""

    from sci_fi_parser.benchmark.ocr_cv import OcrMode, run_benchmark

    modes: tuple[OcrMode, ...] = ("oracle", "detected") if ocr_mode == "both" else (cast(OcrMode, ocr_mode),)
    return run_benchmark(data, out, dataset=dataset, modes=modes, limit=limit)


def _prompt_suffix(inputs: PipelineInputs, image_id: str) -> str:
    value = inputs.image_set.get(image_id).get("ocrcv", {}).get("result", "")
    return f"OCR/CV context:\n{value}" if isinstance(value, str) and value else ""


def run_vlm_stage(inputs: PipelineInputs, extractor: benchmark.Extractor) -> None:
    from tqdm import tqdm

    for image_id in tqdm(inputs.truth_by_image_id):
        record = inputs.image_set.get(image_id)
        path = inputs.image_set.get_image_path(image_id)
        t0 = time.perf_counter()
        try:
            parsed, raw = extractor.extract(
                path,
                prompt_suffix=_prompt_suffix(inputs, image_id),
            )
            parsed_payload = ChartData.model_validate(parsed).model_dump()
        except Exception as exc:
            print(f"  ! {path.name}: {type(exc).__name__}: {exc}")
            parsed_payload = ChartData(chart_type="none", series=[], log_scale=False).model_dump()
            raw = {"error": f"{type(exc).__name__}: {exc}"}
        inputs.image_set.add_vlm_result(image_id, parsed_payload)
        inputs.image_set.add_vlm_result_raw(image_id, raw)
        record["metadata"]["vlm"]["seconds"] = time.perf_counter() - t0


def write_outputs(
    out: Path,
    extractor: benchmark.Extractor,
    agg: dict,
    results: list[benchmark.ChartResult],
    img_dir: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    benchmark._write_results_json(out, extractor, agg, results)
    from sci_fi_parser.benchmark import draw_lap

    draw_lap.write_html(
        out / "report.html",
        extractor._model,
        agg,
        value_results_to_draw_lap_charts(results),
        value_results_to_breakdowns(results),
        img_dir,
    )


def run_pipeline(
    *,
    data: Path,
    out: Path,
    extractor_name: str = "noisy-oracle",
    profile: VLMProfile | None = None,
    seed: int = 0,
    limit: int | None = None,
    dataset: DatasetKind = "synthetic",
    classification: bool = False,
    ocr_cv: bool = False,
    target: BenchmarkTarget = "vlm",
    ocr_mode: OcrModeArgument = "both",
    print_summary: bool = True,
) -> dict:
    if target == "ocr-cv":
        return run_ocr_cv_stage(
            data=data,
            out=out,
            dataset=dataset,
            ocr_mode=ocr_mode,
            limit=limit,
        )

    inputs = load_inputs(data, limit, dataset)
    if classification:
        run_classification_stage(inputs)
    if ocr_cv:
        run_ocr_cv_context_stage(inputs)

    extractor = benchmark.build_extractor(
        extractor_name,
        {
            inputs.image_set.get_image_path(image_id).name: truth
            for image_id, truth in inputs.truth_by_image_id.items()
        },
        np.random.default_rng(seed),
        profile=profile,
    )
    run_vlm_stage(inputs, extractor)

    results = score_vlm_outputs(
        inputs.image_set,
        inputs.truth_by_image_id,
    )
    agg = aggregate_value_results(results)
    write_outputs(out, extractor, agg, results, inputs.image_dir)
    if print_summary:
        benchmark._print_summary(extractor, agg, results, out)
    return agg


def _parse_args(*, default_target: BenchmarkTarget = "vlm") -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="dataset dir containing images/ plus truth.jsonl or annotations/",
    )
    ap.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--target", choices=("vlm", "ocr-cv"), default=default_target)
    ap.add_argument("--ocr-mode", choices=("oracle", "detected", "both"), default="both")
    ap.add_argument(
        "--extractor",
        default="noisy-oracle",
        help="noisy-oracle | vlm | vlm:<model>",
    )
    ap.add_argument("--vlm-config", type=Path, default=None, help="VLM profile TOML")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    ap.add_argument("--classification", action="store_true", help="run classification stage")
    ap.add_argument("--ocr-cv", action="store_true", help="run OCR/CV stage")
    return ap.parse_args()


def _run_args(args: argparse.Namespace) -> None:
    target = cast(BenchmarkTarget, args.target)
    out = args.out or (
        Path("reports/ocr_cv_stage/benchmark") if target == "ocr-cv" else Path("reports/pipeline")
    )
    payload = run_pipeline(
        data=args.data,
        out=out,
        extractor_name=args.extractor,
        profile=benchmark._resolve_profile(args.vlm_config) if target == "vlm" else None,
        seed=args.seed,
        limit=args.limit,
        dataset=args.dataset,
        classification=args.classification,
        ocr_cv=args.ocr_cv,
        target=target,
        ocr_mode=cast(OcrModeArgument, args.ocr_mode),
    )
    if target == "ocr-cv":
        print(json.dumps(payload["summary_by_mode"], indent=2, allow_nan=False))
        print(f"OCR/CV component report: {out / 'report.html'}")


def main() -> None:
    _run_args(_parse_args())


def ocr_cv_main() -> None:
    """Compatibility entry point for the former standalone component command."""

    _run_args(_parse_args(default_target="ocr-cv"))


if __name__ == "__main__":
    main()
