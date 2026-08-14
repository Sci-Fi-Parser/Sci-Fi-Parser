"""Visual reports for experimental Y-axis candidate evidence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.axis import (
    CandidateEvidence,
    NumericAxisLabel,
    YAxisDetectionResult,
    detect_y_axis,
)
from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate
from sci_fi_parser.object_detection.computer_vision.config import CvConfig, YAxisConfig


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _label(canvas: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_sources_overlay(image: np.ndarray, result: YAxisDetectionResult) -> np.ndarray:
    canvas = _to_bgr(image)
    for observation in result.detection.hough_vertical:
        cv2.line(canvas, observation.line.p1, observation.line.p2, (0, 140, 255), 2)
    for observation in result.detection.morphology_vertical:
        cv2.line(canvas, observation.line.p1, observation.line.p2, (255, 100, 0), 2)
    for line in result.detection.hough_lines:
        if line.orientation == "horizontal":
            cv2.line(canvas, line.p1, line.p2, (150, 255, 150), 1)
    _label(canvas, "Hough: orange  morphology: blue  horizontal: green", (8, 18), (30, 30, 30))
    return canvas


def draw_fusion_overlay(image: np.ndarray, result: YAxisDetectionResult) -> np.ndarray:
    canvas = draw_sources_overlay(image, result)
    colors = {"hough": (0, 140, 255), "morphology": (255, 100, 0), "both": (180, 40, 180)}
    for candidate in result.detection.candidates:
        color = colors[candidate.sources]
        cv2.line(canvas, candidate.p1, candidate.p2, color, 3)
        _label(canvas, candidate.id, (candidate.p1[0] + 3, max(13, candidate.p1[1] + 12)), color)
    return canvas


def draw_evidence_overlay(
    image: np.ndarray,
    result: YAxisDetectionResult,
    labels: list[NumericAxisLabel],
    bars: list[BarCandidate],
    candidate_id: str | None = None,
) -> np.ndarray:
    canvas = _to_bgr(image)
    for item in labels:
        label_box = item.box
        cv2.rectangle(
            canvas,
            (round(label_box.left), round(label_box.top)),
            (round(label_box.right), round(label_box.bottom)),
            (0, 180, 0),
            1,
        )
        _label(
            canvas,
            f"{item.value:g}",
            (round(label_box.left), max(12, round(label_box.top) - 2)),
            (0, 140, 0),
        )
    for bar in bars:
        bar_box = bar.bbox
        cv2.rectangle(
            canvas,
            (bar_box.x, bar_box.y),
            (bar_box.right, bar_box.bottom),
            (200, 80, 30),
            1,
        )
    ranked = sorted(result.evidence, key=lambda item: item.score, reverse=True)
    if candidate_id:
        ranked = [item for item in ranked if item.candidate.id == candidate_id]
    for rank, evidence in enumerate(ranked):
        color = (130, 130, 130) if evidence.rejection_reasons else (0, 80 + min(170, rank * 15), 230)
        cv2.line(canvas, evidence.candidate.p1, evidence.candidate.p2, color, 2)
        _label(
            canvas,
            f"{evidence.candidate.id} {evidence.score:.2f}",
            (evidence.candidate.p1[0] + 3, max(13, evidence.candidate.p1[1] + 12)),
            color,
        )
    return canvas


def draw_selection_overlay(image: np.ndarray, result: YAxisDetectionResult) -> np.ndarray:
    canvas = _to_bgr(image)
    if result.runner_up_evidence:
        cv2.line(
            canvas,
            result.runner_up_evidence.candidate.p1,
            result.runner_up_evidence.candidate.p2,
            (0, 220, 255),
            2,
        )
    if result.selected_candidate:
        cv2.line(canvas, result.selected_candidate.p1, result.selected_candidate.p2, (0, 200, 0), 4)
        status = f"SELECTED {result.selected_candidate.id}"
        color = (0, 140, 0)
    else:
        if result.best_evidence:
            cv2.line(
                canvas, result.best_evidence.candidate.p1, result.best_evidence.candidate.p2, (0, 0, 230), 3
            )
        status = "NO AXIS SELECTED"
        color = (0, 0, 220)
    cv2.rectangle(canvas, (4, 4), (min(image.shape[1] - 4, 460), 58), (255, 255, 255), -1)
    _label(canvas, status, (9, 21), color)
    _label(canvas, f"margin={result.score_margin} reason={result.failure_reason}", (9, 42), color)
    return canvas


def _evidence_json(evidence: CandidateEvidence) -> dict[str, Any]:
    return {
        "candidate_id": evidence.candidate.id,
        "score": evidence.score,
        "features": evidence.features,
        "contributions": evidence.contributions,
        "has_numeric_scale": evidence.has_numeric_scale,
        "rejection_reasons": evidence.rejection_reasons,
        "associated_labels": [asdict(label) for label in evidence.associated_labels],
    }


def write_y_axis_debug_outputs(
    image: np.ndarray,
    labels: list[NumericAxisLabel],
    bars: list[BarCandidate],
    output_directory: str | Path,
    stem: str,
    cv_config: CvConfig | None = None,
    axis_config: YAxisConfig | None = None,
    candidate_id: str | None = None,
) -> tuple[YAxisDetectionResult, tuple[Path, ...]]:
    cv_config = cv_config or CvConfig()
    axis_config = axis_config or YAxisConfig()
    started = time.perf_counter()
    result = detect_y_axis(image, labels, bars, cv_config, axis_config)
    runtime_ms = (time.perf_counter() - started) * 1000
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        output / f"{stem}_sources.png",
        output / f"{stem}_fusion.png",
        output / f"{stem}_evidence.png",
        output / f"{stem}_selection.png",
        output / f"{stem}_axis_report.json",
    )
    overlays = (
        draw_sources_overlay(image, result),
        draw_fusion_overlay(image, result),
        draw_evidence_overlay(image, result, labels, bars, candidate_id),
        draw_selection_overlay(image, result),
    )
    for path, overlay in zip(paths[:4], overlays, strict=True):
        if not cv2.imwrite(str(path), overlay):
            raise OSError(f"could not write Y-axis debug overlay to {path}")
    report = {
        "image_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "configuration": {"cv": asdict(cv_config), "y_axis": asdict(axis_config)},
        "labels": [asdict(label) for label in labels],
        "bars": [asdict(bar.bbox) for bar in bars],
        "candidate_evidence": [_evidence_json(item) for item in result.evidence],
        "selected_candidate": result.selected_candidate.id if result.selected_candidate else None,
        "failure_reason": result.failure_reason,
        "runtime_ms": runtime_ms,
    }
    paths[4].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result, paths
