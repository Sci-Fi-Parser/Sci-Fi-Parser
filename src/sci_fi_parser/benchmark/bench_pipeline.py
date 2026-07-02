"""MVP staged benchmark pipeline.

This module only orchestrates stages. Scoring, aggregation, extractor building,
and report formatting are reused from the legacy benchmark where possible.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from sci_fi_parser.schema import ImageSet
from sci_fi_parser.image_extraction.image_loader import load_images_from_folder
from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.vlm.vlm_schema import ChartData, Extractor, parse_chartdata


@dataclass(slots=True)
class PipelineInputs:
    image_set: ImageSet
    truth_by_image_id: dict[str, ChartTruth]
    image_dir: Path


DatasetKind = Literal["synthetic", "benetech"]


def load_inputs(
    data: Path,
    limit: int | None = None,
    dataset: DatasetKind = "synthetic",
) -> PipelineInputs:
    if dataset == "synthetic":
        truth_by_key, metadata_by_key = load_synthetic_truth(data)
        truth_key = lambda path: path.name
    elif dataset == "benetech":
        truth_by_key, metadata_by_key = load_benetech_truth(data)
        truth_key = lambda path: path.stem
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
        image_set.get(image_id).setdefault("metadata", {})["benchmark"] = metadata_by_key.get(key, {})

    return PipelineInputs(
        image_set=image_set,
        truth_by_image_id=truth_by_image_id,
        image_dir=img_dir,
    )


def run_classification_stage(inputs: PipelineInputs) -> None:
    from sci_fi_parser.classifier.classifier_pipeline import start_classification

    start_classification(inputs.image_set)


def run_ocr_cv_stage(inputs: PipelineInputs) -> None:
    from sci_fi_parser.object_detection.pipeline import start_ocr

    scoped = ImageSet()
    for image_id in inputs.truth_by_image_id:
        scoped.add(image_id, inputs.image_set.get(image_id))
    start_ocr(scoped)


def _prompt_suffix(inputs: PipelineInputs, image_id: str) -> str:
    value = inputs.image_set.get(image_id).get("ocrcv", {}).get("result", "")
    return f"OCR/CV context:\n{value}" if isinstance(value, str) and value else ""


def run_vlm_stage(inputs: PipelineInputs, extractor: Extractor) -> None:
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
            parsed_payload = parse_chartdata(parsed).model_dump()
        except Exception as exc:
            print(f"  ! {path.name}: {type(exc).__name__}: {exc}")
            parsed_payload = ChartData(
                chart_type=None,
                series=[],
                confidence=None,
            ).model_dump()
            raw = {"error": f"{type(exc).__name__}: {exc}"}
        inputs.image_set.add_vlm_result(image_id, parsed_payload)
        inputs.image_set.add_vlm_result_raw(image_id, raw)
        record.setdefault("metadata", {}).setdefault("vlm", {})["seconds"] = time.perf_counter() - t0


def write_outputs(
    out: Path,
    extractor: Extractor,
    agg: dict,
    results: list[benchmark.ChartResult],
    img_dir: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    benchmark._write_results_json(out, extractor, agg, results)
    from sci_fi_parser.benchmark import draw_lap

    draw_lap.write_html(
        out / "report.html",
        extractor.name,
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
    print_summary: bool = True,
) -> dict:
    inputs = load_inputs(data, limit, dataset)
    if classification:
        run_classification_stage(inputs)
    if ocr_cv:
        run_ocr_cv_stage(inputs)

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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="dataset dir containing images/ plus truth.jsonl or annotations/",
    )
    ap.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    ap.add_argument("--out", type=Path, default=Path("reports/pipeline"))
    ap.add_argument(
        "--extractor",
        default="noisy-oracle",
        help="noisy-oracle | ollama | ollama:<model> | api | api:<model>",
    )
    ap.add_argument("--vlm-config", type=Path, default=None, help="VLM profile TOML")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    ap.add_argument("--classification", action="store_true", help="run classification stage")
    ap.add_argument("--ocr-cv", action="store_true", help="run OCR/CV stage")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        data=args.data,
        out=args.out,
        extractor_name=args.extractor,
        profile=benchmark._resolve_profile(args.vlm_config),
        seed=args.seed,
        limit=args.limit,
        dataset=args.dataset,
        classification=args.classification,
        ocr_cv=args.ocr_cv,
    )


if __name__ == "__main__":
    main()
