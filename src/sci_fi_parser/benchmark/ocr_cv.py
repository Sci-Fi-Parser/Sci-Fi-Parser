"""OCR/CV component benchmark for vertical bar charts; no VLM is involved.

Scores axis-label role assignment, Y-axis tick recognition and calibration, and
initial bar detection. ``oracle`` mode feeds the synthetic ground-truth OCR tokens
to the analysis so role and calibration errors can be told apart from OCR errors;
``detected`` mode runs PaddleOCR. Role, tick, and calibration metrics need
synthetic geometry; Benetech data only gets bar counts.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import cv2

from sci_fi_parser.benchmark.dataset import DatasetKind, Sample, load_samples
from sci_fi_parser.object_detection.axis_analysis import analyze_ocr_cv
from sci_fi_parser.object_detection.computer_vision.bars import detect_bars
from sci_fi_parser.object_detection.models import OcrOutput
from sci_fi_parser.object_detection.normalization import normalize_ocr_output
from sci_fi_parser.object_detection.ocr import Ocr

OcrMode = Literal["oracle", "detected"]

BAR_MATCH_IOU = 0.5
CALIBRATION_SUCCESS_MAE = 0.01  # mean tick value error as a fraction of the tick span


def _eligible(sample: Sample, dataset: DatasetKind) -> bool:
    if sample.chart_type not in {"vertical_bar", "grouped_bar"} and not sample.metadata.get(
        "bar_negative_control"
    ):
        return False
    geometry = sample.geometry
    return dataset == "benetech" or (
        geometry is not None and geometry.get("family") == "bar" and geometry.get("orientation") == "v"
    )


def _oracle_ocr(geometry: dict) -> OcrOutput:
    tokens = geometry.get("ocr_tokens", [])
    return normalize_ocr_output(
        [token["text"] for token in tokens],
        [1.0] * len(tokens),
        [token["bbox_px"] for token in tokens],
    )


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _text_key(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum() or character in ".-%")


def _match_ocr_tokens(result, truth_tokens: list[dict]) -> dict[str, int]:
    """Greedily match each detected token to an unused truth token by box overlap and text."""

    matches = {}
    used = set()
    for token in result.ocr.tokens:
        box = [token.box.left, token.box.top, token.box.right, token.box.bottom]
        options = []
        for index, truth in enumerate(truth_tokens):
            if index in used:
                continue
            iou = _iou(box, [float(value) for value in truth["bbox_px"]])
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
    counts = {role: {"tp": 0, "fp": 0, "fn": 0} for role in ("y_tick", "x_label")}
    found: dict[str, set[int]] = {role: set() for role in counts}
    for assignment in result.roles.assignments:
        if assignment.role not in counts:
            continue
        truth_index = matches.get(assignment.token_id)
        if truth_index is not None and truth_tokens[truth_index]["role"] == assignment.role:
            counts[assignment.role]["tp"] += 1
            found[assignment.role].add(truth_index)
        else:
            counts[assignment.role]["fp"] += 1
    for index, truth in enumerate(truth_tokens):
        if truth["role"] in counts and index not in found[truth["role"]]:
            counts[truth["role"]]["fn"] += 1
    return counts


def _tick_recall(result, truth_tokens: list[dict]) -> tuple[int, int]:
    """Count Y-axis ticks whose truth token was matched by a token with the same text."""

    matches = _match_ocr_tokens(result, truth_tokens)
    tokens = {token.token_id: token for token in result.ocr.tokens}
    recognized = {
        truth_index
        for token_id, truth_index in matches.items()
        if _text_key(tokens[token_id].original_text) == _text_key(truth_tokens[truth_index]["text"])
    }
    y_ticks = {index for index, token in enumerate(truth_tokens) if token["role"] == "y_tick"}
    return len(recognized & y_ticks), len(y_ticks)


def _calibration_mae(result, geometry: dict) -> float | None:
    """Mean tick value error of the fitted calibration, normalised by the tick span."""

    ticks = geometry.get("value_ticks", [])
    if not result.calibration.succeeded or not ticks:
        return None
    values = [float(tick["value"]) for tick in ticks]
    span = max(values) - min(values)
    if not span:
        return None
    slope, intercept = result.calibration.slope, result.calibration.intercept
    errors = [abs(slope * float(tick["px"]) + intercept - float(tick["value"])) for tick in ticks]
    return float(mean(errors)) / span


def _bar_counts(bars, sample: Sample) -> dict[str, Any]:
    expected = 0 if sample.metadata.get("bar_negative_control") else sum(len(s.points) for s in sample.series)
    counts: dict[str, Any] = {"exact_count": len(bars) == expected, "tp": None, "fp": None, "fn": None}
    if sample.geometry is None:
        return counts
    truth_boxes = [[float(v) for v in item["bbox_px"]] for item in sample.geometry.get("items", [])]
    predictions = [
        [float(bar.bbox.x), float(bar.bbox.y), float(bar.bbox.right), float(bar.bbox.bottom)] for bar in bars
    ]
    pairs = sorted(
        (
            (_iou(prediction, truth), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(truth_boxes)
        ),
        reverse=True,
    )
    matched_pred: set[int] = set()
    matched_truth: set[int] = set()
    for iou, pred_index, truth_index in pairs:
        if iou >= BAR_MATCH_IOU and pred_index not in matched_pred and truth_index not in matched_truth:
            matched_pred.add(pred_index)
            matched_truth.add(truth_index)
    counts.update(
        tp=len(matched_truth),
        fp=len(predictions) - len(matched_pred),
        fn=len(truth_boxes) - len(matched_truth),
    )
    return counts


def _run_sample(sample: Sample, mode: OcrMode, ocr_engine: Ocr | None) -> dict[str, Any]:
    image = cv2.imread(str(sample.image_path))
    if image is None:
        raise ValueError(f"could not read image: {sample.image_path}")
    if mode == "oracle":
        assert sample.geometry is not None
        ocr = _oracle_ocr(sample.geometry)
    else:
        assert ocr_engine is not None
        ocr_engine.read_image(image)
        ocr = ocr_engine.run_ocr()
    bars = detect_bars(image)
    result = analyze_ocr_cv(image, ocr, "bar_chart", bars)
    record: dict[str, Any] = {
        "image": sample.image_path.name,
        "mode": mode,
        "bars": _bar_counts(bars, sample),
        "both_axes_selected": (
            result.roles.selected_x_candidate_id is not None
            and result.roles.selected_y_candidate_id is not None
        ),
        "calibration_failure": result.calibration.failure_reason,
        "role_counts": None,
        "y_ticks": None,
        "calibration_mae": None,
    }
    if sample.geometry is not None:
        truth_tokens = sample.geometry.get("ocr_tokens", [])
        record["role_counts"] = _role_counts(result, truth_tokens)
        record["y_ticks"] = _tick_recall(result, truth_tokens)
        record["calibration_mae"] = _calibration_mae(result, sample.geometry)
    return record


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _summarize(records: list[dict]) -> dict[str, Any]:
    with_truth = [r for r in records if r["role_counts"] is not None]
    role_metrics: dict[str, Any] = {}
    for role in ("y_tick", "x_label"):
        tp, fp, fn = (sum(r["role_counts"][role][k] for r in with_truth) for k in ("tp", "fp", "fn"))
        role_metrics[role] = _precision_recall_f1(tp, fp, fn)
    all_counts = [c for r in with_truth for c in r["role_counts"].values()]
    role_metrics["micro_f1"] = _precision_recall_f1(
        *(sum(c[k] for c in all_counts) for k in ("tp", "fp", "fn"))
    )["f1"]
    maes = [r["calibration_mae"] for r in with_truth if r["calibration_mae"] is not None]
    scored_bars = [r["bars"] for r in records if r["bars"]["tp"] is not None]
    bar_tp, bar_fp, bar_fn = (sum(b[k] for b in scored_bars) for k in ("tp", "fp", "fn"))
    return {
        "samples": len(records),
        "both_axes_selected": _ratio(sum(r["both_axes_selected"] for r in records), len(records)),
        "role_metrics": role_metrics,
        "y_tick_recall": _ratio(
            sum(r["y_ticks"][0] for r in with_truth), sum(r["y_ticks"][1] for r in with_truth)
        ),
        "calibration": {
            "fit_rate": _ratio(len(maes), len(with_truth)),
            "success_rate": _ratio(sum(m < CALIBRATION_SUCCESS_MAE for m in maes), len(with_truth)),
            "normalized_mae": mean(maes) if maes else None,
        },
        "bars": {
            **_precision_recall_f1(bar_tp, bar_fp, bar_fn),
            "exact_count_accuracy": _ratio(sum(r["bars"]["exact_count"] for r in records), len(records)),
        },
    }


def _write_report(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    rows = "".join(
        f"<tr><td>{html.escape(mode)}</td><td>{s['samples']}</td><td>{cell(s['role_metrics']['micro_f1'])}</td>"
        f"<td>{cell(s['y_tick_recall'])}</td><td>{cell(s['calibration']['success_rate'])}</td>"
        f"<td>{cell(s['calibration']['normalized_mae'])}</td><td>{cell(s['bars']['precision'])}</td>"
        f"<td>{cell(s['bars']['recall'])}</td><td>{cell(s['bars']['exact_count_accuracy'])}</td></tr>"
        for mode, s in payload["summary_by_mode"].items()
    )
    (out / "report.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>OCR/CV component benchmark</title>
<style>body{{font:14px sans-serif;margin:24px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #aaa;padding:7px}}th{{background:#eee}}</style>
<h1>OCR/CV component benchmark</h1><p><a href="results.json">Full JSON</a></p>
<table><tr><th>OCR mode</th><th>N</th><th>Role micro F1</th><th>Y-tick recall</th>
<th>Calibration success</th><th>Calibration normalized MAE</th><th>Bar precision</th>
<th>Bar recall</th><th>Exact bar count</th></tr>{rows}</table>""",
        encoding="utf-8",
    )


def run_benchmark(
    data: Path,
    out: Path,
    *,
    dataset: DatasetKind = "synthetic",
    modes: tuple[OcrMode, ...] = ("oracle", "detected"),
    limit: int | None = None,
) -> dict[str, Any]:
    """Score OCR/CV on eligible samples and write ``results.json`` and ``report.html`` to ``out``."""

    if dataset == "benetech" and "oracle" in modes:
        raise ValueError("oracle OCR requires synthetic geometry with ocr_tokens")
    samples = [sample for sample in load_samples(data, dataset) if _eligible(sample, dataset)][:limit]
    if not samples:
        raise ValueError("no eligible vertical bar samples with required truth")

    ocr_engine = Ocr() if "detected" in modes else None
    records = {mode: [_run_sample(sample, mode, ocr_engine) for sample in samples] for mode in modes}
    payload = {
        "data": str(data),
        "dataset": dataset,
        "summary_by_mode": {mode: _summarize(mode_records) for mode, mode_records in records.items()},
        "samples": [record for mode_records in records.values() for record in mode_records],
    }
    _write_report(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench-ocr-cv", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset dir with images/")
    parser.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    parser.add_argument("-o", "--out", type=Path, default=Path("reports/ocr_cv"))
    parser.add_argument("--ocr-mode", choices=("oracle", "detected", "both"), default="both")
    parser.add_argument("--limit", type=int, help="only the first N eligible charts, by file name")
    args = parser.parse_args()

    modes: tuple[OcrMode, ...] = ("oracle", "detected") if args.ocr_mode == "both" else (args.ocr_mode,)
    payload = run_benchmark(args.data, args.out, dataset=args.dataset, modes=modes, limit=args.limit)
    print(json.dumps(payload["summary_by_mode"], indent=2, allow_nan=False))
    print(f"report -> {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
