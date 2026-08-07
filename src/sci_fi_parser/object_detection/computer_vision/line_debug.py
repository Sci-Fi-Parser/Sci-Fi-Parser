"""Debug overlays for inspecting structural line detection."""

from pathlib import Path

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.config import CvConfig
from sci_fi_parser.object_detection.computer_vision.lines import (
    LineSegment,
    MergedLine,
    detect_raw_line_segments,
    merge_line_segments,
)


def draw_raw_line_overlay(
    image: np.ndarray, segments: list[LineSegment]
) -> np.ndarray:
    """Draw raw Hough segments over a copy of the input image."""

    overlay = _to_bgr(image)
    for segment in segments:
        color = (255, 180, 0) if segment.orientation == "horizontal" else (180, 0, 255)
        cv2.line(overlay, segment.p1, segment.p2, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, segment.p1, 2, color, thickness=-1)
        cv2.circle(overlay, segment.p2, 2, color, thickness=-1)
    return overlay


def draw_merged_line_overlay(
    image: np.ndarray, lines: list[MergedLine]
) -> np.ndarray:
    """Draw consolidated lines and their source counts over the input image."""

    overlay = _to_bgr(image)
    for line in lines:
        color = (0, 180, 0) if line.orientation == "horizontal" else (0, 90, 255)
        cv2.line(overlay, line.p1, line.p2, color, 2, cv2.LINE_AA)
        label_position = (line.p1[0] + 3, max(12, line.p1[1] - 4))
        cv2.putText(
            overlay,
            f"x{line.source_count}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_line_detection_overlays(
    image: np.ndarray,
    output_directory: str | Path,
    stem: str,
    config: CvConfig | None = None,
) -> tuple[Path, Path]:
    """Detect lines and write separate raw and merged PNG overlays."""

    config = config or CvConfig()
    raw_segments = detect_raw_line_segments(image, config)
    merged_lines = merge_line_segments(raw_segments, config)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / f"{stem}_raw_lines.png"
    merged_path = output_directory / f"{stem}_merged_lines.png"

    if not cv2.imwrite(str(raw_path), draw_raw_line_overlay(image, raw_segments)):
        raise OSError(f"could not write line overlay to {raw_path}")
    if not cv2.imwrite(str(merged_path), draw_merged_line_overlay(image, merged_lines)):
        raise OSError(f"could not write line overlay to {merged_path}")
    return raw_path, merged_path


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError("image must be grayscale, BGR, or BGRA")
