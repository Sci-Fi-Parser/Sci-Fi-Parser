"""Benchmark experimental Y-axis selection with synthetic geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import cv2

from sci_fi_parser.object_detection.computer_vision.axis import NumericAxisLabel, detect_y_axis
from sci_fi_parser.object_detection.computer_vision.axis_ocr import numeric_labels_from_ocr
from sci_fi_parser.object_detection.computer_vision.bars import (
    BarCandidate,
    BoundingBox,
    detect_bars,
)
from sci_fi_parser.object_detection.computer_vision.config import CvConfig, YAxisConfig
from sci_fi_parser.object_detection.computer_vision.geometry import interval_overlap
from sci_fi_parser.schema import ImageSet

Mode = Literal["oracle", "detected"]


def _oracle_inputs(geometry: dict[str, Any]) -> tuple[list[NumericAxisLabel], list[BarCandidate]]:
    axis_x = float(geometry["y_axis"]["p1_px"][0])
    labels = [
        NumericAxisLabel(
            float(tick["value"]),
            (axis_x - 45, float(tick["px"]) - 6, axis_x - 5, float(tick["px"]) + 6),
        )
        for tick in geometry.get("value_ticks", [])
    ]
    bars = []
    for item in geometry.get("items", []):
        if "bbox_px" not in item:
            continue
        left, top, right, bottom = item["bbox_px"]
        bars.append(
            BarCandidate(
                BoundingBox(
                    round(left),
                    round(top),
                    max(1, round(right - left)),
                    max(1, round(bottom - top)),
                )
            )
        )
    return labels, bars


def _detected_inputs(
    image,
    stem: str,
    cache_dir: Path,
    refresh: bool,
    ocr_holder: list,
    minimum_confidence: float,
) -> tuple[list[NumericAxisLabel], list[BarCandidate]]:
    cache_path = cache_dir / f"{stem}.json"
    if cache_path.exists() and not refresh:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        ocr = raw.get("ocr", raw)
        labels = numeric_labels_from_ocr(ocr, minimum_confidence)
        if "bars" in raw:
            bars = [BarCandidate(BoundingBox(**bbox)) for bbox in raw["bars"]]
        else:
            bar_image = (
                cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.ndim == 3 and image.shape[2] == 4 else image
            )
            bars = detect_bars(bar_image)
        return labels, bars
    if not ocr_holder:
        from sci_fi_parser.object_detection.ocr import Ocr

        ocr_holder.append(Ocr())
    ocr_holder[0].read_image(image)
    ocr = ocr_holder[0].run_ocr()
    serializable_ocr = {
        key: value.tolist() if hasattr(value, "tolist") else value for key, value in ocr.items()
    }
    bar_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.ndim == 3 and image.shape[2] == 4 else image
    bars = detect_bars(bar_image)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"ocr": serializable_ocr, "bars": [asdict(bar.bbox) for bar in bars]}),
        encoding="utf-8",
    )
    return numeric_labels_from_ocr(serializable_ocr, minimum_confidence), bars


def _score(result: Any, geometry: dict[str, Any], image_width: int) -> dict[str, Any]:
    truth = geometry["y_axis"]
    present = bool(truth.get("visible", False))
    selected = result.selected_candidate
    correct = False
    error = None
    coverage = None
    if present and selected is not None:
        truth_x = (float(truth["p1_px"][0]) + float(truth["p2_px"][0])) / 2
        truth_top, truth_bottom = sorted((float(truth["p1_px"][1]), float(truth["p2_px"][1])))
        error = abs(selected.x - truth_x)
        coverage = interval_overlap(selected.top, selected.bottom, truth_top, truth_bottom) / max(
            1.0, truth_bottom - truth_top
        )
        correct = error <= max(2.0, image_width * 0.005) and coverage >= 0.5
    if present and correct:
        outcome = "TP"
    elif present and selected is None:
        outcome = "FN"
    elif present:
        outcome = "wrong_selection"
    elif selected is None:
        outcome = "TN"
    else:
        outcome = "FP"
    return {
        "outcome": outcome,
        "axis_present": present,
        "selected_candidate": selected.id if selected else None,
        "coordinate_error_px": error,
        "span_coverage": coverage,
        "best_score": result.best_evidence.score if result.best_evidence else None,
        "score_margin": result.score_margin,
        "failure_reason": result.failure_reason,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["outcome"] == "TP" for row in rows)
    fp = sum(row["outcome"] in {"FP", "wrong_selection"} for row in rows)
    fn = sum(row["outcome"] in {"FN", "wrong_selection"} for row in rows)
    tn = sum(row["outcome"] == "TN" for row in rows)
    errors = [row["coordinate_error_px"] for row in rows if row["outcome"] == "TP"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "mean_coordinate_error_px": sum(errors) / len(errors) if errors else None,
    }


def run_axis_benchmark(
    image_set: ImageSet,
    truth_by_image_id: dict[str, Any],
    output_path: Path,
    modes: tuple[Mode, ...] = ("oracle", "detected"),
    cache_dir: Path = Path("temp/axis_benchmark_cache"),
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Compare oracle labels with the current minimal OCR adapter."""
    config = YAxisConfig()
    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    ocr_holder: list = []
    for image_id, truth in truth_by_image_id.items():
        geometry = truth.geometry
        if truth.chart_type == "horizontal_bar" or not isinstance(geometry, dict) or "y_axis" not in geometry:
            continue
        image_path = image_set.get_image_path(image_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        for mode in modes:
            if mode == "oracle":
                labels, bars = _oracle_inputs(geometry)
            else:
                labels, bars = _detected_inputs(
                    image,
                    image_path.stem,
                    cache_dir,
                    refresh_cache,
                    ocr_holder,
                    config.numeric_ocr_min_confidence,
                )
            result = detect_y_axis(image, labels, bars, axis_config=config)
            mode_rows[mode].append(
                {
                    "image": image_path.name,
                    "numeric_label_count": len(labels),
                    **_score(result, geometry, image.shape[1]),
                }
            )
    report = {
        "benchmark": "y_axis_selection",
        "configuration": {"cv": asdict(CvConfig()), "y_axis": asdict(config)},
        "modes": {mode: {"metrics": _aggregate(rows), "images": rows} for mode, rows in mode_rows.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/axis_benchmark.json"))
    parser.add_argument("--mode", choices=("oracle", "detected", "both"), default="both")
    parser.add_argument("--ocr-cache", type=Path, default=Path("temp/axis_benchmark_cache"))
    parser.add_argument("--refresh-ocr", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from sci_fi_parser.benchmark.bench_pipeline import load_inputs

    inputs = load_inputs(args.data, args.limit, "synthetic")
    modes: tuple[Mode, ...] = ("oracle", "detected") if args.mode == "both" else (args.mode,)
    report = run_axis_benchmark(
        inputs.image_set,
        inputs.truth_by_image_id,
        args.output,
        modes,
        args.ocr_cache,
        args.refresh_ocr,
    )
    for mode, data in report["modes"].items():
        print(f"{mode}: {json.dumps(data['metrics'], indent=2)}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
