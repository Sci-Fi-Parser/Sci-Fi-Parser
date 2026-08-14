"""Minimal adapter from raw OCR output to numeric Y-axis labels."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sci_fi_parser.object_detection.computer_vision.axis import NumericAxisLabel
from sci_fi_parser.object_detection.computer_vision.geometry import normalize_bbox

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?[kKmMbB%]?$")


def parse_numeric_value(text: str) -> float | None:
    """Parse common chart-label numbers without attempting general OCR correction."""
    normalized = (
        text.strip().replace("\N{MINUS SIGN}", "-").replace(" ", "").replace("\N{NO-BREAK SPACE}", "")
    )
    if "," in normalized:
        if "." in normalized or re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:[kKmMbB%])?", normalized):
            normalized = normalized.replace(",", "")
        else:
            normalized = normalized.replace(",", ".")
    if not _NUMBER.fullmatch(normalized):
        return None
    suffix = normalized[-1] if normalized[-1] in "kKmMbB%" else ""
    number = normalized[:-1] if suffix else normalized
    try:
        value = float(number)
    except ValueError:
        return None
    multiplier = {"": 1.0, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "b": 1e9, "B": 1e9, "%": 0.01}
    return value * multiplier[suffix]


def _values(ocr_result: dict[str, Any], key: str) -> list[Any]:
    value = ocr_result.get(key)
    if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []
    return list(value)


def numeric_labels_from_ocr(
    ocr_result: dict[str, Any],
    minimum_confidence: float = 0.8,
) -> list[NumericAxisLabel]:
    """Extract numeric labels from PaddleOCR's parallel result arrays."""
    labels = []
    for text, raw_confidence, raw_bbox in zip(
        _values(ocr_result, "labels"),
        _values(ocr_result, "confidence"),
        _values(ocr_result, "bbox"),
        strict=False,
    ):
        value = parse_numeric_value(str(text))
        try:
            confidence = float(raw_confidence)
            box = normalize_bbox(raw_bbox)
        except (TypeError, ValueError):
            continue
        if value is None or confidence < minimum_confidence or box.width <= 0 or box.height <= 0:
            continue
        labels.append(NumericAxisLabel(value, (box.left, box.top, box.right, box.bottom)))
    return labels
