"""Initial computer-vision proposals for vertical bar-like regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.config import CvConfig

MaskSource = Literal["saturation", "grayscale", "outline"]


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box for a detected bar region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


@dataclass(slots=True)
class BarCandidate:
    """Accepted initial proposal; grouped-bar interpretation is deferred."""

    bbox: BoundingBox
    source: MaskSource | None = None


@dataclass(slots=True)
class ContourDiagnostic:
    """Decision evidence for one contour before or after deduplication."""

    contour_id: str
    source: MaskSource
    bbox: BoundingBox
    contour_area: float
    rectangularity: float
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    deduplicated_to: str | None = None


@dataclass(slots=True)
class BarDetectionDiagnostics:
    """Runtime-only masks and JSON-safe contour evidence for debugging."""

    mask_source: Literal["saturation", "grayscale"]
    effective_saturation_threshold: float | None
    morphology: dict[str, int]
    saturation_mask: np.ndarray | None
    grayscale_mask: np.ndarray | None
    outline_mask: np.ndarray
    processed_mask: np.ndarray
    contours: list[ContourDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mask_source": self.mask_source,
            "effective_saturation_threshold": self.effective_saturation_threshold,
            "morphology": self.morphology,
            "contours": [
                {
                    "contour_id": contour.contour_id,
                    "source": contour.source,
                    "bbox": {
                        "x": contour.bbox.x,
                        "y": contour.bbox.y,
                        "width": contour.bbox.width,
                        "height": contour.bbox.height,
                    },
                    "contour_area": contour.contour_area,
                    "rectangularity": contour.rectangularity,
                    "accepted": contour.accepted,
                    "rejection_reasons": contour.rejection_reasons,
                    "deduplicated_to": contour.deduplicated_to,
                }
                for contour in self.contours
            ],
        }


@dataclass(slots=True)
class BarDetectionResult:
    bars: list[BarCandidate]
    diagnostics: BarDetectionDiagnostics


def detect_bars(image: np.ndarray, config: CvConfig | None = None) -> list[BarCandidate]:
    """Return accepted initial proposals using the compatibility API."""

    return detect_bars_with_diagnostics(image, config).bars


def detect_bars_with_diagnostics(
    image: np.ndarray,
    config: CvConfig | None = None,
) -> BarDetectionResult:
    """Detect initial bars and retain mask, rejection, and deduplication evidence."""

    config = config or CvConfig()
    image_height, image_width = image.shape[:2]
    if image_width == 0 or image_height == 0:
        empty = np.zeros((image_height, image_width), dtype=np.uint8)
        return BarDetectionResult(
            bars=[],
            diagnostics=BarDetectionDiagnostics(
                mask_source="grayscale",
                effective_saturation_threshold=None,
                morphology={
                    "width": config.bar_morphology_width,
                    "height": config.bar_morphology_height,
                },
                saturation_mask=None,
                grayscale_mask=empty,
                outline_mask=empty,
                processed_mask=empty,
            ),
        )

    raw_mask, mask_source, saturation_mask, grayscale_mask, effective_threshold = _foreground_mask(
        image, config
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, config.bar_morphology_width), max(1, config.bar_morphology_height)),
    )
    processed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    diagnostics = BarDetectionDiagnostics(
        mask_source=mask_source,
        effective_saturation_threshold=effective_threshold,
        morphology={
            "width": max(1, config.bar_morphology_width),
            "height": max(1, config.bar_morphology_height),
        },
        saturation_mask=saturation_mask,
        grayscale_mask=grayscale_mask,
        outline_mask=raw_mask.copy(),
        processed_mask=processed_mask,
    )

    min_area = image_height * image_width * config.min_bar_area_ratio
    min_height = max(1, int(image_height * config.min_bar_height_ratio))
    max_width = max(config.min_bar_width_pixels, int(image_width * config.max_bar_width_ratio))
    proposals: list[tuple[ContourDiagnostic, BarCandidate]] = []
    passes: tuple[tuple[MaskSource, np.ndarray], ...] = (
        (mask_source, processed_mask),
        ("outline", raw_mask),
    )
    for source, mask in passes:
        retrieval = cv2.RETR_LIST if source == "outline" else cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, width=width, height=height)
            contour_area = float(cv2.contourArea(contour))
            # OpenCV contour coordinates span width-1 by height-1 for raster rectangles.
            contour_box_area = max(float((width - 1) * (height - 1)), 1.0)
            rectangularity = min(1.0, contour_area / contour_box_area)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            reasons = []
            if width * height < min_area:
                reasons.append("bounding-box area below minimum")
            if width < config.min_bar_width_pixels:
                reasons.append("width below minimum")
            if width > max_width:
                reasons.append("width above maximum")
            if height < min_height:
                reasons.append("height below minimum")
            if rectangularity < config.min_bar_rectangularity:
                reasons.append("rectangularity below minimum")
            if source == "outline" and (len(polygon) != 4 or not cv2.isContourConvex(polygon)):
                reasons.append("raw outline is not a closed rectangle")
            if source != "outline" and width * height < max(100.0, min_area * 10.0) and rectangularity < 0.9:
                reasons.append("small convex contour does not fill its rectangle")
            diagnostic = ContourDiagnostic(
                contour_id=f"contour-{len(diagnostics.contours)}",
                source=source,
                bbox=bbox,
                contour_area=contour_area,
                rectangularity=rectangularity,
                accepted=not reasons,
                rejection_reasons=reasons,
            )
            diagnostics.contours.append(diagnostic)
            if not reasons:
                proposals.append((diagnostic, BarCandidate(bbox=bbox, source=source)))

    kept: list[tuple[ContourDiagnostic, BarCandidate]] = []
    proposals.sort(
        key=lambda item: (
            item[0].source != "outline",
            item[0].rectangularity,
            item[0].contour_area,
        ),
        reverse=True,
    )
    for diagnostic, proposal in proposals:
        duplicate = next(
            (
                kept_diagnostic
                for kept_diagnostic, kept_proposal in kept
                if _box_iou(proposal.bbox, kept_proposal.bbox) >= config.bar_deduplication_iou
                or (
                    diagnostic.source == "outline"
                    and kept_diagnostic.source != "outline"
                    and _box_iou(proposal.bbox, kept_proposal.bbox) >= 0.50
                )
                or _same_outlined_rectangle(
                    diagnostic,
                    proposal.bbox,
                    kept_diagnostic,
                    kept_proposal.bbox,
                )
            ),
            None,
        )
        if duplicate is not None:
            diagnostic.accepted = False
            diagnostic.rejection_reasons.append("high-IoU duplicate")
            diagnostic.deduplicated_to = duplicate.contour_id
            continue
        kept.append((diagnostic, proposal))

    accepted = [proposal for _, proposal in kept]
    accepted.sort(key=lambda bar: (bar.bbox.x, bar.bbox.y))
    return BarDetectionResult(bars=accepted, diagnostics=diagnostics)


def _foreground_mask(
    image: np.ndarray,
    config: CvConfig,
) -> tuple[
    np.ndarray,
    Literal["saturation", "grayscale"],
    np.ndarray | None,
    np.ndarray | None,
    float | None,
]:
    gray = _to_grayscale(image)
    if image.ndim == 2:
        _, grayscale = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return grayscale, "grayscale", None, grayscale, None

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    high_saturation = float(np.percentile(saturation, 95))
    if high_saturation <= config.achromatic_saturation_percentile:
        _, grayscale = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return grayscale, "grayscale", None, grayscale, None

    otsu_threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, saturation_mask = cv2.threshold(saturation, otsu_threshold, 255, cv2.THRESH_BINARY)
    return saturation_mask, "saturation", saturation_mask, None, float(otsu_threshold)


def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _same_outlined_rectangle(
    first_diagnostic: ContourDiagnostic,
    first: BoundingBox,
    second_diagnostic: ContourDiagnostic,
    second: BoundingBox,
) -> bool:
    if "outline" not in {first_diagnostic.source, second_diagnostic.source}:
        return False
    first_contains_second = (
        first.x <= second.x
        and first.y <= second.y
        and first.right >= second.right
        and first.bottom >= second.bottom
    )
    second_contains_first = (
        second.x <= first.x
        and second.y <= first.y
        and second.right >= first.right
        and second.bottom >= first.bottom
    )
    return (
        (first_contains_second or second_contains_first)
        and abs(first.center_x - second.center_x) <= 2.0
        and abs(first.center_y - second.center_y) <= 2.0
        and max(
            abs(first.x - second.x),
            abs(first.y - second.y),
            abs(first.right - second.right),
            abs(first.bottom - second.bottom),
        )
        <= 12
        and _box_iou(first, second) >= 0.30
    )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
