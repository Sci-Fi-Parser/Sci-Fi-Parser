"""Standalone component benchmark for OCR/CV milestones 2-4 and initial bars."""

from __future__ import annotations

import argparse
import html
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal, cast

import cv2
import numpy as np

from sci_fi_parser.object_detection.axis_analysis import analyze_ocr_cv
from sci_fi_parser.object_detection.computer_vision.bars import detect_bars
from sci_fi_parser.object_detection.models import OcrOutput
from sci_fi_parser.object_detection.normalization import normalize_ocr_output

OcrMode = Literal["oracle", "detected"]


@dataclass(slots=True)
class Sample:
    image_path: Path
    truth: dict


def _load_samples(data: Path, dataset: str) -> list[Sample]:
    if dataset == "synthetic":
        records = [
            json.loads(line) for line in (data / "truth.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return [Sample(data / "images" / record["image"], record) for record in records]
    annotation_dir = data / "annotations"
    samples = []
    for path in sorted(annotation_dir.glob("*.json")):
        image_path = next(
            (
                candidate
                for suffix in (".jpg", ".png", ".jpeg")
                if (candidate := data / "images" / f"{path.stem}{suffix}").exists()
            ),
            None,
        )
        if image_path is not None:
            samples.append(Sample(image_path, json.loads(path.read_text(encoding="utf-8"))))
    return samples


def _eligible(sample: Sample, dataset: str) -> bool:
    chart_type = sample.truth.get("chart_type") if dataset == "synthetic" else sample.truth.get("chart-type")
    if chart_type not in {"vertical_bar", "grouped_bar"} and not sample.truth.get("metadata", {}).get(
        "bar_negative_control", False
    ):
        return False
    geometry = sample.truth.get("geometry")
    return dataset == "benetech" or (
        geometry is not None and geometry.get("family") == "bar" and geometry.get("orientation") == "v"
    )


def _oracle_ocr(truth: dict) -> OcrOutput:
    geometry = truth["geometry"]
    tokens = geometry.get("ocr_tokens", [])
    return normalize_ocr_output(
        [token["text"] for token in tokens],
        [1.0] * len(tokens),
        [token["bbox_px"] for token in tokens],
    )


def _box_iou(first, second) -> float:
    left = max(first.left, float(second[0]))
    top = max(first.top, float(second[1]))
    right = min(first.right, float(second[2]))
    bottom = min(first.bottom, float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first.width) * max(0.0, first.height)
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _text_key(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum() or character in ".-%")


def _match_ocr_tokens(result, truth_tokens: list[dict]) -> dict[str, int]:
    matches = {}
    used = set()
    for token in result.ocr.tokens:
        options = []
        for index, truth in enumerate(truth_tokens):
            if index in used:
                continue
            iou = _box_iou(token.box, truth["bbox_px"])
            text_equal = _text_key(token.original_text) == _text_key(truth["text"])
            if iou >= 0.1 or (text_equal and iou > 0.0):
                options.append((iou + (0.25 if text_equal else 0.0), index))
        if options:
            _, index = max(options)
            matches[token.token_id] = index
            used.add(index)
    return matches


def _role_counts(result, truth_tokens: list[dict]) -> dict[str, dict[str, int]]:
    matches = _match_ocr_tokens(result, truth_tokens)
    assignments = {assignment.token_id: assignment.role for assignment in result.roles.assignments}
    counts = {role: {"tp": 0, "fp": 0, "fn": 0} for role in ("y_tick", "x_label")}
    correctly_matched: dict[str, set[int]] = {role: set() for role in counts}
    for token_id, predicted_role in assignments.items():
        if predicted_role not in counts:
            continue
        truth_index = matches.get(token_id)
        if truth_index is not None and truth_tokens[truth_index]["role"] == predicted_role:
            counts[predicted_role]["tp"] += 1
            correctly_matched[predicted_role].add(truth_index)
        else:
            counts[predicted_role]["fp"] += 1
    for index, truth in enumerate(truth_tokens):
        role = truth["role"]
        if role in counts and index not in correctly_matched[role]:
            counts[role]["fn"] += 1
    return counts


def _tick_recall(result, truth_tokens: list[dict]) -> tuple[int, int]:
    matches = _match_ocr_tokens(result, truth_tokens)
    tokens = {token.token_id: token for token in result.ocr.tokens}
    recognized = {
        truth_index
        for token_id, truth_index in matches.items()
        if _text_key(tokens[token_id].original_text) == _text_key(truth_tokens[truth_index]["text"])
    }
    y_tick_indices = {index for index, token in enumerate(truth_tokens) if token["role"] == "y_tick"}
    return len(recognized.intersection(y_tick_indices)), len(y_tick_indices)


def _calibration_metrics(result, geometry: dict) -> dict:
    ticks = geometry.get("value_ticks", [])
    if not result.calibration.succeeded or not ticks:
        return {
            "fit_produced": False,
            "success": False,
            "normalized_mae": None,
            "pixel_reprojection_mae": None,
        }
    slope = result.calibration.slope
    intercept = result.calibration.intercept
    values = [float(tick["value"]) for tick in ticks]
    span = max(values) - min(values)
    value_errors = [abs(slope * float(tick["px"]) + intercept - float(tick["value"])) for tick in ticks]
    pixel_errors = [abs((float(tick["value"]) - intercept) / slope - float(tick["px"])) for tick in ticks]
    normalized_mae = mean(value_errors) / span if span else None
    return {
        "fit_produced": True,
        "success": normalized_mae is not None and normalized_mae < 0.01,
        "normalized_mae": normalized_mae,
        "pixel_reprojection_mae": mean(pixel_errors),
    }


def _bar_box(candidate):
    box = candidate.bbox
    return [float(box.x), float(box.y), float(box.right), float(box.bottom)]


def _raw_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _bar_metrics(bars, geometry: dict | None, expected_count: int) -> dict:
    metrics = {
        "detected_count": len(bars),
        "expected_count": expected_count,
        "exact_count": len(bars) == expected_count,
        "tp": None,
        "fp": None,
        "fn": None,
        "center_error": None,
        "top_error": None,
    }
    if geometry is None:
        return metrics
    truth_boxes = [list(map(float, item["bbox_px"])) for item in geometry.get("items", [])]
    predictions = [_bar_box(bar) for bar in bars]
    pairs = sorted(
        (
            (_raw_iou(prediction, truth), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(truth_boxes)
        ),
        reverse=True,
    )
    matched_predictions = set()
    matched_truth = set()
    center_errors = []
    top_errors = []
    for iou, pred_index, truth_index in pairs:
        if iou < 0.5 or pred_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(pred_index)
        matched_truth.add(truth_index)
        prediction, truth = predictions[pred_index], truth_boxes[truth_index]
        center_errors.append(abs((prediction[0] + prediction[2]) / 2 - (truth[0] + truth[2]) / 2))
        top_errors.append(abs(prediction[1] - truth[1]))
    metrics.update(
        {
            "tp": len(matched_truth),
            "fp": len(predictions) - len(matched_predictions),
            "fn": len(truth_boxes) - len(matched_truth),
            "center_error": mean(center_errors) if center_errors else None,
            "top_error": mean(top_errors) if top_errors else None,
        }
    )
    return metrics


def _expected_count(truth: dict, dataset: str) -> int:
    if truth.get("metadata", {}).get("bar_negative_control", False):
        return 0
    if dataset == "synthetic":
        return sum(len(series.get("points", [])) for series in truth.get("series", []))
    return len(truth.get("data-series", []))


def _run_sample(sample: Sample, dataset: str, mode: OcrMode, ocr_engine) -> dict:
    image = cv2.imread(str(sample.image_path))
    if image is None:
        raise ValueError(f"could not read image: {sample.image_path}")
    started = time.perf_counter()
    if mode == "oracle":
        ocr = _oracle_ocr(sample.truth)
        ocr_ms = 0.0
    else:
        ocr_started = time.perf_counter()
        ocr_engine.read_image(image)
        ocr = ocr_engine.run_ocr()
        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0
    bar_started = time.perf_counter()
    bars = detect_bars(image)
    bar_ms = (time.perf_counter() - bar_started) * 1000.0
    result = analyze_ocr_cv(image, ocr, "bar_chart", bars)
    geometry = sample.truth.get("geometry")
    metadata = sample.truth.get("metadata", {})
    expected_count = _expected_count(sample.truth, dataset)
    record = {
        "image": sample.image_path.name,
        "mode": mode,
        "label_rotation": sample.truth.get("metadata", {}).get("label_rotation"),
        "metadata": metadata,
        "role_counts": None,
        "y_tick_recognized": None,
        "y_tick_total": None,
        "calibration": None,
        "bars": _bar_metrics(bars, geometry, expected_count),
        "role_abstention": result.roles.abstention_reason,
        "calibration_failure": result.calibration.failure_reason,
        "both_axes_selected": (
            result.roles.selected_x_candidate_id is not None
            and result.roles.selected_y_candidate_id is not None
        ),
        "role_confidence": result.roles.confidence,
        "bar_slices": _bar_slice_metadata(image, sample.truth, geometry),
        "runtime_ms": {
            "ocr": ocr_ms,
            "initial_bars": bar_ms,
            **result.stage_runtime_ms,
            "wall_total": (time.perf_counter() - started) * 1000.0,
        },
    }
    if geometry is not None:
        truth_tokens = geometry.get("ocr_tokens", [])
        record["role_counts"] = _role_counts(result, truth_tokens)
        recognized, total = _tick_recall(result, truth_tokens)
        record["y_tick_recognized"] = recognized
        record["y_tick_total"] = total
        record["calibration"] = _calibration_metrics(result, geometry)
    return record


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _bar_slice_metadata(image, truth: dict, geometry: dict | None) -> dict[str, str]:
    height, width = image.shape[:2]
    metadata = truth.get("metadata", {})
    boxes = geometry.get("items", []) if geometry else []
    widths = [float(item["bbox_px"][2]) - float(item["bbox_px"][0]) for item in boxes]
    heights = [float(item["bbox_px"][3]) - float(item["bbox_px"][1]) for item in boxes]
    if image.ndim == 3 and boxes:
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        samples = []
        for item in boxes:
            left, top, right, bottom = (round(float(value)) for value in item["bbox_px"])
            region = saturation[
                max(0, top) : min(height, bottom),
                max(0, left) : min(width, right),
            ]
            samples.extend(region.flat)
        percentile = float(np.percentile(samples, 75)) if samples else 0.0
    else:
        percentile = 0.0
    saturation_bucket = "achromatic" if percentile <= 3 else "low" if percentile <= 32 else "high"

    def dimension_bucket(values: list[float], small: float, medium: float) -> str:
        value = float(np.median(values)) if values else 0.0
        return f"<={small:g}" if value <= small else f"<={medium:g}" if value <= medium else f">{medium:g}"

    return {
        "saturation_bucket": saturation_bucket,
        "outline_type": str(metadata.get("outline_type", metadata.get("bar_fill", "filled"))),
        "outline_width": str(metadata.get("outline_width", "0")),
        "resolution": f"{width}x{height}",
        "bar_width": dimension_bucket(widths, 4, 12),
        "bar_height": dimension_bucket(heights, height * 0.03, height * 0.15),
        "density": str(metadata.get("density", "unknown")),
        "series_count": str(len(truth.get("series", []))),
        "group_gap": str(metadata.get("group_gap", "not_grouped")),
        "grid": str(metadata.get("grid", metadata.get("grid_on", "unknown"))),
        "value_labels": str(metadata.get("value_labels", metadata.get("labels_on", "unknown"))),
    }


def _summarize_bar_group(records: list[dict]) -> dict:
    metrics = [record["bars"] for record in records]
    true_positive = sum(metric["tp"] or 0 for metric in metrics)
    false_positive = sum(metric["fp"] or 0 for metric in metrics)
    false_negative = sum(metric["fn"] or 0 for metric in metrics)
    return {
        "sample_count": len(records),
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "exact_count_accuracy": mean(metric["exact_count"] for metric in metrics) if metrics else None,
    }


def _bar_breakdowns(records: list[dict]) -> dict:
    breakdowns = {}
    for key in (
        "saturation_bucket",
        "outline_type",
        "outline_width",
        "resolution",
        "bar_width",
        "bar_height",
        "density",
        "series_count",
        "group_gap",
        "grid",
        "value_labels",
    ):
        values = sorted({record["bar_slices"][key] for record in records})
        breakdowns[key] = {
            value: _summarize_bar_group([record for record in records if record["bar_slices"][key] == value])
            for value in values
        }
    return breakdowns


def _paired_axis_junction_summary(records: list[dict]) -> dict:
    paired = [
        record
        for record in records
        if record["metadata"].get("axis_junction_pair")
        and record["metadata"].get("axis_junction_variant") in {"separated", "shared"}
    ]
    summary = {}
    for pair_id in sorted({record["metadata"]["axis_junction_pair"] for record in paired}):
        variants = {}
        for variant in ("separated", "shared"):
            subset = [
                record
                for record in paired
                if record["metadata"]["axis_junction_pair"] == pair_id
                and record["metadata"]["axis_junction_variant"] == variant
            ]
            counts = {"tp": 0, "fp": 0, "fn": 0}
            x_counts = {"tp": 0, "fn": 0}
            for record in subset:
                if record["role_counts"] is None:
                    continue
                for role_counts in record["role_counts"].values():
                    for key in counts:
                        counts[key] += role_counts[key]
                for key in x_counts:
                    x_counts[key] += record["role_counts"]["x_label"][key]
            precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
            recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
            variants[variant] = {
                "sample_count": len(subset),
                "role_precision": precision,
                "role_recall": recall,
                "role_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and precision + recall
                    else None
                ),
                "x_label_recall": _ratio(x_counts["tp"], x_counts["tp"] + x_counts["fn"]),
                "both_axis_selection_rate": mean(record["both_axes_selected"] for record in subset)
                if subset
                else None,
                "confidence": mean(record["role_confidence"] for record in subset) if subset else None,
            }
        separated_recall = variants["separated"]["x_label_recall"]
        shared_recall = variants["shared"]["x_label_recall"]
        summary[str(pair_id)] = {
            **variants,
            "shared_x_label_recall_no_regression": (
                separated_recall is not None
                and shared_recall is not None
                and shared_recall >= separated_recall
            ),
        }
    return summary


def _summarize(records: list[dict]) -> dict:
    role_totals = {role: {"tp": 0, "fp": 0, "fn": 0} for role in ("y_tick", "x_label")}
    for record in records:
        if record["role_counts"]:
            for role, counts in record["role_counts"].items():
                for key in counts:
                    role_totals[role][key] += counts[key]
    role_metrics = {}
    aggregate_tp = aggregate_fp = aggregate_fn = 0
    for role, counts in role_totals.items():
        precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        role_metrics[role] = {**counts, "precision": precision, "recall": recall, "f1": f1}
        aggregate_tp += counts["tp"]
        aggregate_fp += counts["fp"]
        aggregate_fn += counts["fn"]
    micro_precision = _ratio(aggregate_tp, aggregate_tp + aggregate_fp)
    micro_recall = _ratio(aggregate_tp, aggregate_tp + aggregate_fn)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision is not None and micro_recall is not None and micro_precision + micro_recall
        else None
    )
    calibration_records = [record["calibration"] for record in records if record["calibration"] is not None]
    successful_calibrations = [metric for metric in calibration_records if metric["success"]]
    produced_calibrations = [metric for metric in calibration_records if metric["fit_produced"]]
    bars = [record["bars"] for record in records]
    bar_tp = sum(metric["tp"] or 0 for metric in bars)
    bar_fp = sum(metric["fp"] or 0 for metric in bars)
    bar_fn = sum(metric["fn"] or 0 for metric in bars)
    runtime_keys = sorted({key for record in records for key in record["runtime_ms"]})
    rotations = {}
    for rotation in sorted(
        {record["label_rotation"] for record in records if record["label_rotation"] is not None}
    ):
        counts = [
            record["role_counts"]["x_label"]
            for record in records
            if record["label_rotation"] == rotation and record["role_counts"] is not None
        ]
        true_positive = sum(count["tp"] for count in counts)
        false_negative = sum(count["fn"] for count in counts)
        rotations[str(rotation)] = {
            "sample_count": sum(record["label_rotation"] == rotation for record in records),
            "x_label_recall": _ratio(true_positive, true_positive + false_negative),
        }
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "ocr_y_tick_recall": _ratio(
            sum(record["y_tick_recognized"] or 0 for record in records),
            sum(record["y_tick_total"] or 0 for record in records),
        ),
        "role_metrics": {**role_metrics, "micro_f1": micro_f1},
        "both_axis_selection_rate": mean(record["both_axes_selected"] for record in records)
        if records
        else None,
        "role_confidence_mean": mean(record["role_confidence"] for record in records) if records else None,
        "calibration": {
            "eligible": len(calibration_records),
            "fit_produced_rate": _ratio(len(produced_calibrations), len(calibration_records)),
            "success_rate": _ratio(len(successful_calibrations), len(calibration_records)),
            "normalized_mae": mean(
                metric["normalized_mae"]
                for metric in produced_calibrations
                if metric["normalized_mae"] is not None
            )
            if any(metric["normalized_mae"] is not None for metric in produced_calibrations)
            else None,
            "pixel_reprojection_mae": mean(
                metric["pixel_reprojection_mae"]
                for metric in produced_calibrations
                if metric["pixel_reprojection_mae"] is not None
            )
            if any(metric["pixel_reprojection_mae"] is not None for metric in produced_calibrations)
            else None,
            "abstention_rate": _ratio(
                len(calibration_records) - len(produced_calibrations), len(calibration_records)
            ),
            "wrong_scale_rate": _ratio(
                len(produced_calibrations) - len(successful_calibrations), len(calibration_records)
            ),
        },
        "initial_bars": {
            "precision": _ratio(bar_tp, bar_tp + bar_fp),
            "recall": _ratio(bar_tp, bar_tp + bar_fn),
            "exact_count_accuracy": mean(metric["exact_count"] for metric in bars) if bars else None,
            "center_pixel_mae": mean(
                metric["center_error"] for metric in bars if metric["center_error"] is not None
            )
            if any(metric["center_error"] is not None for metric in bars)
            else None,
            "top_pixel_mae": mean(metric["top_error"] for metric in bars if metric["top_error"] is not None)
            if any(metric["top_error"] is not None for metric in bars)
            else None,
        },
        "runtime_ms_mean": {
            key: mean(record["runtime_ms"][key] for record in records) for key in runtime_keys
        },
        "x_label_recognition_by_rotation": rotations,
        "initial_bars_by": _bar_breakdowns(records),
    }
    role_f1 = summary["role_metrics"]["micro_f1"]
    calibration = summary["calibration"]
    exact_count = summary["initial_bars"]["exact_count_accuracy"]
    summary["provisional_gates"] = {
        "label_role_f1_at_least_0_95": role_f1 is not None and role_f1 >= 0.95,
        "calibration_success_at_least_0_95": (
            calibration["success_rate"] is not None and calibration["success_rate"] >= 0.95
        ),
        "calibration_normalized_mae_below_0_01": (
            calibration["normalized_mae"] is not None and calibration["normalized_mae"] < 0.01
        ),
        "exact_bar_count_at_least_0_95": exact_count is not None and exact_count >= 0.95,
        "bar_value_mae_below_0_02": None,
    }
    return summary


def _write_report(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    rows = []
    for mode, summary in payload["summary_by_mode"].items():
        calibration = summary["calibration"]
        bars = summary["initial_bars"]
        rows.append(
            f"<tr><td>{html.escape(mode)}</td><td>{summary['sample_count']}</td>"
            f"<td>{summary['role_metrics']['micro_f1']}</td><td>{calibration['success_rate']}</td>"
            f"<td>{calibration['normalized_mae']}</td><td>{calibration['pixel_reprojection_mae']}</td>"
            f"<td>{bars['precision']}</td><td>{bars['recall']}</td><td>{bars['exact_count_accuracy']}</td></tr>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>OCR/CV component benchmark</title>
<style>body{{font:14px sans-serif;margin:24px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #aaa;padding:7px}}
th{{background:#eee}}</style></head><body><h1>OCR/CV component benchmark</h1>
<p>This report is independent of VLM end-result scoring. <a href="results.json">Full JSON</a></p>
<table><tr><th>OCR mode</th><th>N</th><th>Role micro F1</th><th>Calibration success</th>
<th>Calibration normalized MAE</th><th>Reprojection px MAE</th><th>Bar precision</th><th>Bar recall</th>
<th>Exact bar count</th></tr>{"".join(rows)}</table></body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")


def run_benchmark(
    data: Path,
    out: Path,
    *,
    dataset: str = "synthetic",
    modes: tuple[OcrMode, ...] = ("oracle", "detected"),
    limit: int | None = None,
) -> dict:
    """Run component scoring without constructing or invoking a VLM extractor."""

    all_samples = _load_samples(data, dataset)
    samples = [sample for sample in all_samples if _eligible(sample, dataset)]
    eligible_count = len(samples)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("no eligible vertical bar samples with required truth")
    if dataset == "benetech" and "oracle" in modes:
        raise ValueError("oracle OCR requires synthetic geometry with ocr_tokens")

    ocr_engine = None
    if "detected" in modes:
        from sci_fi_parser.object_detection.ocr import Ocr

        ocr_engine = Ocr()
    records: list[dict] = []
    for mode in modes:
        records.extend(_run_sample(sample, dataset, mode, ocr_engine) for sample in samples)
    payload = {
        "schema_version": "ocr-cv-component-benchmark-v1",
        "dataset": dataset,
        "data": str(data),
        "scope": "OCR/CV stages only; no VLM invocation or ChartData scoring",
        "benchmark_thresholds": {
            "bar_match_iou": 0.5,
            "calibration_success_normalized_mae": 0.01,
        },
        "grouped_bar_scope": {
            "separated": "initial proposals are scored when pixel gaps remain visible",
            "touching": "merged/unsupported; no generic splitting is attempted",
        },
        "eligibility": {
            "total_samples": len(all_samples),
            "eligible_vertical_bars": eligible_count,
            "evaluated_after_limit": len(samples),
            "skipped": len(all_samples) - eligible_count,
        },
        "summary_by_mode": {
            mode: _summarize([record for record in records if record["mode"] == mode]) for mode in modes
        },
        "axis_junction_pairs_by_mode": {
            mode: _paired_axis_junction_summary([record for record in records if record["mode"] == mode])
            for mode in modes
        },
        "samples": records,
    }
    _write_report(out, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    parser.add_argument("--out", type=Path, default=Path("reports/ocr_cv_stage/benchmark"))
    parser.add_argument("--ocr-mode", choices=("oracle", "detected", "both"), default="both")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes: tuple[OcrMode, ...] = (
        ("oracle", "detected") if args.ocr_mode == "both" else (cast(OcrMode, args.ocr_mode),)
    )
    payload = run_benchmark(args.data, args.out, dataset=args.dataset, modes=modes, limit=args.limit)
    print(json.dumps(payload["summary_by_mode"], indent=2, allow_nan=False))
    print(f"OCR/CV component report: {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
