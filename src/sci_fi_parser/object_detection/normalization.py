"""Canonical OCR boxes and cautious numeric parsing."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Literal, cast

from .models import OcrBox, OcrOutput, OcrToken, ParsedNumber

_NUMBER_RE = re.compile(
    r"^(?P<sign>[+-]?)(?P<number>(?:\d[\d.,]*|[.,]\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>[kKmMbB]?)(?P<percent>%?)$"
)
_PAREN_RE = re.compile(r"^\((.+)\)$")
_CORRECTIONS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})
_MAGNITUDES = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without changing recognized characters."""

    return " ".join(unicodedata.normalize("NFKC", text).replace("−", "-").split())


def parse_numeric_token(text: str, *, cautious_corrections: bool = False) -> ParsedNumber | None:
    """Parse chart-number syntax while preserving percentage display semantics.

    Percent values stay in displayed units: ``10%`` becomes ``10``, not ``0.1``.
    Character corrections are opt-in and are intended only for numeric-axis hypotheses.
    """

    candidate = normalize_text(text).replace(" ", "")
    negative_parentheses = False
    parenthesized = _PAREN_RE.fullmatch(candidate)
    if parenthesized:
        negative_parentheses = True
        candidate = parenthesized.group(1)

    corrected = False
    match = _NUMBER_RE.fullmatch(candidate)
    if match is None and cautious_corrections:
        corrected_candidate = candidate.translate(_CORRECTIONS)
        corrected = corrected_candidate != candidate
        candidate = corrected_candidate
        match = _NUMBER_RE.fullmatch(candidate)
    if match is None:
        return None

    number_text = match.group("number")
    if "," in number_text and "." in number_text:
        if number_text.rfind(",") > number_text.rfind("."):
            number_text = number_text.replace(".", "").replace(",", ".")
        else:
            number_text = number_text.replace(",", "")
    elif "," in number_text:
        groups = number_text.split(",")
        if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]) and len(groups[0]) <= 3:
            number_text = "".join(groups)
        else:
            number_text = number_text.replace(",", ".")

    try:
        value = float(f"{match.group('sign')}{number_text}")
    except ValueError:
        return None
    if negative_parentheses:
        value = -abs(value)

    suffix_text = match.group("suffix").upper()
    suffix = cast(Literal["K", "M", "B"], suffix_text) if suffix_text else None
    if suffix:
        value *= _MAGNITUDES[suffix]
    return ParsedNumber(
        value=value,
        parsed_text=candidate,
        is_percentage=bool(match.group("percent")),
        magnitude_suffix=suffix,
        correction_applied=corrected,
    )


def normalize_ocr_box(raw_box: Any) -> OcrBox:
    """Accept Paddle rectangles or polygons and return ordered coordinates."""

    values = raw_box.tolist() if hasattr(raw_box, "tolist") else raw_box
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"OCR box is not a coordinate sequence: {raw_box!r}")

    if len(values) == 4 and all(not isinstance(value, Sequence) for value in values):
        x1, y1, x2, y2 = (float(value) for value in values)
        return OcrBox(left=min(x1, x2), top=min(y1, y2), right=max(x1, x2), bottom=max(y1, y2))

    points = [point.tolist() if hasattr(point, "tolist") else point for point in values]
    if not points or any(not isinstance(point, Sequence) or len(point) < 2 for point in points):
        raise ValueError(f"OCR polygon is malformed: {raw_box!r}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return OcrBox(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


def normalize_ocr_output(labels: Any, confidences: Any, boxes: Any) -> OcrOutput:
    """Convert Paddle parallel arrays into the canonical typed representation."""

    label_list = list(labels) if labels is not None else []
    confidence_list = list(confidences) if confidences is not None else []
    box_list = list(boxes) if boxes is not None else []
    lengths = {len(label_list), len(confidence_list), len(box_list)}
    if len(lengths) != 1:
        raise ValueError(
            "PaddleOCR returned mismatched labels, confidences, and boxes: "
            f"{len(label_list)}, {len(confidence_list)}, {len(box_list)}"
        )

    tokens = []
    for index, (label, confidence, box) in enumerate(zip(label_list, confidence_list, box_list, strict=True)):
        original = str(label)
        normalized = normalize_text(original)
        tokens.append(
            OcrToken(
                token_id=f"ocr-{index}",
                original_text=original,
                normalized_text=normalized,
                confidence=max(0.0, min(1.0, float(confidence))),
                box=normalize_ocr_box(box),
                parsed_number=parse_numeric_token(normalized),
            )
        )
    return OcrOutput(tokens=tokens)
