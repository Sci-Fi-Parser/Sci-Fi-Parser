from __future__ import annotations

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision import axis
from sci_fi_parser.object_detection.computer_vision.axis import NumericAxisLabel
from sci_fi_parser.object_detection.computer_vision.axis_candidates import (
    CandidateDetection,
    VerticalObservation,
    fuse_vertical_observations,
)
from sci_fi_parser.object_detection.computer_vision.axis_ocr import (
    numeric_labels_from_ocr,
    parse_numeric_value,
)
from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate, BoundingBox
from sci_fi_parser.object_detection.computer_vision.lines import MergedLine


def _observation(source: str, x: int, top: int, bottom: int) -> VerticalObservation:
    line = MergedLine((x, top), (x, bottom - 1), "vertical", bottom - top, ())
    return VerticalObservation(source, x, top, bottom, ((top, bottom),), line)


def _labels(right: int, values: list[int]) -> list[NumericAxisLabel]:
    ys = [30 + index * 45 for index in range(len(values))]
    return [
        NumericAxisLabel(value, (right - 28, y - 6, right, y + 6))
        for value, y in zip(values, ys, strict=True)
    ]


def test_fusion_pairs_once_and_preserves_unmatched_provenance() -> None:
    candidates = fuse_vertical_observations(
        [_observation("hough", 20, 10, 100), _observation("hough", 70, 10, 100)],
        [_observation("morphology", 21, 20, 95)],
        image_width=200,
    )

    assert [candidate.sources for candidate in candidates] == ["both", "hough"]
    assert candidates[0].supported_intervals == ((10, 100),)


def test_selector_skips_rejected_high_scoring_candidate(monkeypatch) -> None:
    observations = [_observation("hough", 70, 20, 220), _observation("hough", 150, 20, 220)]
    candidates = fuse_vertical_observations(observations, [], image_width=320)
    detection = CandidateDetection((), (), tuple(observations), (), tuple(candidates))
    monkeypatch.setattr(axis, "detect_vertical_candidates", lambda image, config: detection)
    image = np.full((240, 320), 255, dtype=np.uint8)
    labels = _labels(58, [100, 75, 50, 25, 0]) + _labels(138, [100, 75, 50, 25, 0])
    bars = [BarCandidate(BoundingBox(66, 20, 8, 200))]

    result = axis.detect_y_axis(image, labels, bars)

    assert result.selected_candidate is not None
    assert result.selected_candidate.x == 150
    assert result.evidence[0].rejection_reasons == ["material bar overlap"]


def test_selector_requires_three_labels_decreasing_downward() -> None:
    image = np.full((240, 320), 255, dtype=np.uint8)
    cv2.line(image, (70, 20), (70, 220), 0, 3)

    two_labels = axis.detect_y_axis(image, _labels(58, [100, 0]))
    reversed_scale = axis.detect_y_axis(image, _labels(58, [0, 50, 100]))
    valid_scale = axis.detect_y_axis(image, _labels(58, [100, 50, 0]))

    assert two_labels.selected_candidate is None
    assert reversed_scale.selected_candidate is None
    assert valid_scale.selected_candidate is not None


def test_numeric_ocr_adapter_parses_common_labels_and_skips_bad_rows() -> None:
    ocr = {
        "labels": ["1,250", "50%", "-2.5e1", "category", "10"],
        "confidence": [0.99, 0.98, 0.97, 0.99, "bad"],
        "bbox": [
            [10, 10, 30, 20],
            [10, 30, 30, 40],
            [10, 50, 30, 60],
            [10, 70, 30, 80],
            [10, 90, 30, 100],
        ],
    }

    labels = numeric_labels_from_ocr(ocr)

    assert [label.value for label in labels] == [1250.0, 0.5, -25.0]
    assert parse_numeric_value("1,5") == 1.5
    assert parse_numeric_value("12K") == 12000.0
    assert parse_numeric_value("1.2.3") is None
