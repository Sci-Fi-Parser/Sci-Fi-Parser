"""Shared half-open geometry primitives for OCR and chart elements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


def normalize_bbox(values: Sequence[float]) -> Box:
    """Normalize arbitrary endpoint order to a half-open box."""
    if len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in values)
    return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def interval_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Return ordinary overlap between two half-open intervals."""
    return max(0.0, min(end1, end2) - max(start1, start2))
