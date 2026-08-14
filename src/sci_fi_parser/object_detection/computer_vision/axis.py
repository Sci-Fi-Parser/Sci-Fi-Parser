"""Experimental ranking of vertical lines as possible left Y-axes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.axis_candidates import (
    CandidateDetection,
    VerticalCandidate,
    detect_vertical_candidates,
)
from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate
from sci_fi_parser.object_detection.computer_vision.config import CvConfig, YAxisConfig
from sci_fi_parser.object_detection.computer_vision.geometry import Box, interval_overlap
from sci_fi_parser.object_detection.computer_vision.lines import to_grayscale


@dataclass(frozen=True, slots=True)
class NumericAxisLabel:
    """A numeric OCR label classified by an upstream component."""

    value: float
    bbox: tuple[float, float, float, float]

    @property
    def box(self) -> Box:
        return Box(*self.bbox)


@dataclass(slots=True)
class CandidateEvidence:
    candidate: VerticalCandidate
    features: dict[str, Any] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)
    associated_labels: list[NumericAxisLabel] = field(default_factory=list)
    has_numeric_scale: bool = False
    score: float = 0.0


@dataclass(slots=True)
class YAxisDetectionResult:
    detection: CandidateDetection
    evidence: list[CandidateEvidence]
    eligible_evidence: list[CandidateEvidence]
    selected_candidate: VerticalCandidate | None
    failure_reason: str | None
    selection_threshold: float
    margin_threshold: float

    @property
    def best_evidence(self) -> CandidateEvidence | None:
        return self.eligible_evidence[0] if self.eligible_evidence else None

    @property
    def runner_up_evidence(self) -> CandidateEvidence | None:
        return self.eligible_evidence[1] if len(self.eligible_evidence) > 1 else None

    @property
    def score_margin(self) -> float | None:
        if self.best_evidence is None:
            return None
        if self.runner_up_evidence is None:
            return self.best_evidence.score
        return self.best_evidence.score - self.runner_up_evidence.score


class BarFeatures(TypedDict):
    bar_overlap_ratio: float
    inside_bar_interior: bool
    on_bar_edge: bool
    bars_right: float
    bars_left: float
    first_bar_left: int | None


def _foreground_support(image: np.ndarray, candidate: VerticalCandidate) -> float:
    gray = to_grayscale(image)
    _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    x = min(max(round(candidate.x), 0), foreground.shape[1] - 1)
    top = min(max(round(candidate.top), 0), foreground.shape[0])
    bottom = min(max(round(candidate.bottom), top), foreground.shape[0])
    if bottom <= top:
        return 0.0
    left, right = max(0, x - 1), min(foreground.shape[1], x + 2)
    return float(np.mean(np.any(foreground[top:bottom, left:right] > 0, axis=1)))


def _label_features(
    candidate: VerticalCandidate,
    labels: list[NumericAxisLabel],
    image_width: int,
    config: YAxisConfig,
) -> tuple[list[NumericAxisLabel], dict[str, float]]:
    maximum_distance = max(config.label_max_distance_pixels, image_width * config.label_max_distance_ratio)
    associated = [
        label
        for label in labels
        if label.box.right <= candidate.x + config.label_x_tolerance_pixels
        and candidate.x - label.box.right <= maximum_distance
        and candidate.top - label.box.height <= label.box.center_y <= candidate.bottom + label.box.height
    ]
    associated.sort(key=lambda label: label.box.center_y)
    if not associated:
        return associated, {
            "label_count": 0.0,
            "label_vertical_span_ratio": 0.0,
            "label_value_correlation": 0.0,
        }

    y_values = np.asarray([label.box.center_y for label in associated])
    values = np.asarray([label.value for label in associated])
    span_ratio = float(np.ptp(y_values) / max(1.0, candidate.length)) if len(associated) > 1 else 0.0
    correlation = 0.0
    if len(associated) > 1 and np.ptp(y_values) > 0 and np.ptp(values) > 0:
        correlation = float(np.corrcoef(y_values, values)[0, 1])
    return associated, {
        "label_count": float(len(associated)),
        "label_vertical_span_ratio": span_ratio,
        "label_value_correlation": correlation,
        "label_boundary": max(label.box.right for label in associated),
    }


def _bar_features(candidate: VerticalCandidate, bars: list[BarCandidate]) -> BarFeatures:
    overlap_lengths = []
    interior = False
    edge = False
    for bar in bars:
        box = bar.bbox
        overlap = interval_overlap(candidate.top, candidate.bottom, box.y, box.bottom)
        if overlap <= 1:
            continue
        if box.x + 1 < candidate.x < box.right - 1:
            interior = True
            overlap_lengths.append(overlap)
        elif abs(candidate.x - box.x) <= 1 or abs(candidate.x - box.right) <= 1:
            edge = True
            overlap_lengths.append(overlap)
    right_bars = [bar for bar in bars if bar.bbox.x > candidate.x]
    left_bars = [bar for bar in bars if bar.bbox.right < candidate.x]
    return {
        "bar_overlap_ratio": max(overlap_lengths, default=0.0) / max(1.0, candidate.length),
        "inside_bar_interior": interior,
        "on_bar_edge": edge,
        "bars_right": float(len(right_bars)),
        "bars_left": float(len(left_bars)),
        "first_bar_left": min((bar.bbox.x for bar in right_bars), default=None),
    }


def calculate_candidate_evidence(
    image: np.ndarray,
    candidate: VerticalCandidate,
    labels: list[NumericAxisLabel],
    bars: list[BarCandidate],
    config: YAxisConfig,
) -> CandidateEvidence:
    height, width = image.shape[:2]
    evidence = CandidateEvidence(candidate)
    span_ratio = candidate.length / max(1, height)
    support_ratio = candidate.supported_length / max(1.0, candidate.length)
    foreground = _foreground_support(image, candidate)
    associated, label_features = _label_features(candidate, labels, width, config)
    bar_features = _bar_features(candidate, bars)
    evidence.associated_labels = associated
    evidence.features.update(
        span_ratio=span_ratio,
        supported_span_ratio=support_ratio,
        foreground_support=foreground,
        detector_agreement=candidate.sources == "both",
        **label_features,
        **bar_features,
    )

    if candidate.x <= config.border_tolerance_pixels or candidate.x >= width - config.border_tolerance_pixels:
        evidence.rejection_reasons.append("image border")
    if (bar_features["inside_bar_interior"] or bar_features["on_bar_edge"]) and bar_features[
        "bar_overlap_ratio"
    ] > config.maximum_bar_overlap_ratio:
        evidence.rejection_reasons.append("material bar overlap")

    correlation = label_features["label_value_correlation"]
    evidence.has_numeric_scale = (
        len(associated) >= config.label_min_count
        and label_features["label_vertical_span_ratio"] >= config.label_min_span_ratio
        and correlation <= -config.minimum_label_correlation
    )
    evidence.contributions = {
        "vertical_span": min(2.0, span_ratio * 3.0),
        "foreground_support": foreground,
        "continuity": max(-1.0, (support_ratio - 0.5) * 1.5),
        "detector_agreement": 0.6 if candidate.sources == "both" else 0.0,
        "numeric_label_count": min(1.5, len(associated) * 0.4),
        "numeric_label_span": min(1.2, label_features["label_vertical_span_ratio"] * 2.0),
        "numeric_scale_order": max(0.0, -correlation) * 1.2,
        "bar_interior": -4.0 if bar_features["inside_bar_interior"] else 0.0,
        "bar_edge": -3.0 if bar_features["on_bar_edge"] else 0.0,
        "bars_left": -min(2.0, bar_features["bars_left"] * 0.4),
    }
    label_boundary = label_features.get("label_boundary")
    first_bar = bar_features["first_bar_left"]
    corridor = (
        label_boundary is not None and first_bar is not None and label_boundary < candidate.x < first_bar
    )
    evidence.features["valid_label_axis_bar_corridor"] = corridor
    evidence.contributions["label_axis_bar_corridor"] = 1.8 if corridor else 0.0
    evidence.score = sum(evidence.contributions.values())
    return evidence


def detect_y_axis(
    image: np.ndarray,
    labels: list[NumericAxisLabel] | None = None,
    bars: list[BarCandidate] | None = None,
    cv_config: CvConfig | None = None,
    axis_config: YAxisConfig | None = None,
) -> YAxisDetectionResult:
    """Rank candidates using labels and bars supplied by upstream components."""
    labels = labels or []
    bars = bars or []
    axis_config = axis_config or YAxisConfig()
    detection = detect_vertical_candidates(image, cv_config)
    evidence = [
        calculate_candidate_evidence(image, candidate, labels, bars, axis_config)
        for candidate in detection.candidates
    ]
    eligible = sorted(
        (item for item in evidence if not item.rejection_reasons and item.has_numeric_scale),
        key=lambda item: (-item.score, item.candidate.x),
    )

    best = eligible[0] if eligible else None
    runner_up = eligible[1] if len(eligible) > 1 else None
    margin = best.score - runner_up.score if best and runner_up else best.score if best else None
    selected = best.candidate if best else None
    failure_reason = None
    if not detection.candidates:
        failure_reason, selected = "no vertical candidates", None
    elif best is None:
        failure_reason, selected = "no eligible candidate with a numeric scale", None
    elif best.score < axis_config.minimum_score:
        failure_reason, selected = "best candidate below score threshold", None
    elif runner_up is not None and margin is not None and margin < axis_config.minimum_margin:
        failure_reason, selected = "best candidate margin below threshold", None

    return YAxisDetectionResult(
        detection=detection,
        evidence=evidence,
        eligible_evidence=eligible,
        selected_candidate=selected,
        failure_reason=failure_reason,
        selection_threshold=axis_config.minimum_score,
        margin_threshold=axis_config.minimum_margin,
    )
