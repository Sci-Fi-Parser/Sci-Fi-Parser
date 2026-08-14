"""Detection and consolidation of horizontal and vertical chart lines."""

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.config import CvConfig


@dataclass(slots=True)
class LineSegment:
    p1: tuple[int, int]
    p2: tuple[int, int]
    orientation: Literal["horizontal", "vertical"]
    length: float
    angle_degrees: float


@dataclass(slots=True)
class MergedLine:
    """One physical line consolidated from one or more Hough segments."""

    p1: tuple[int, int]
    p2: tuple[int, int]
    orientation: Literal["horizontal", "vertical"]
    length: float
    source_segments: tuple[LineSegment, ...]

    @property
    def source_count(self) -> int:
        return len(self.source_segments)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a grayscale image without converting an already-gray input."""

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def detect_line_segments(image: np.ndarray, config: CvConfig | None = None) -> list[LineSegment]:
    """Compatibility alias for raw line-segment detection."""

    return detect_raw_line_segments(image, config)


def detect_raw_line_segments(image: np.ndarray, config: CvConfig | None = None) -> list[LineSegment]:
    """Detect unmerged horizontal and vertical probabilistic Hough segments."""

    config = config or CvConfig()
    gray = to_grayscale(image)
    if gray.size == 0:
        return []

    edges = cv2.Canny(gray, config.canny_threshold1, config.canny_threshold2, apertureSize=3)

    image_height, image_width = gray.shape[:2]
    min_line_length = max(
        config.line_min_length_pixels,
        round(max(image_height, image_width) * config.line_min_length_ratio),
    )
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=config.max_line_gap,
    )

    if raw_lines is None:
        return []

    line_segments: list[LineSegment] = []
    for raw_line in np.asarray(raw_lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw_line)
        normalized = normalize_line_segment(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            tolerance_degrees=config.angle_tolerance_degrees,
        )
        if normalized is not None:
            line_segments.append(normalized)

    return line_segments


def detect_merged_lines(image: np.ndarray, config: CvConfig | None = None) -> list[MergedLine]:
    """Detect raw segments and merge fragments of the same physical line."""

    config = config or CvConfig()
    return merge_line_segments(detect_raw_line_segments(image, config), config)


def detect_directional_lines(image: np.ndarray, config: CvConfig | None = None) -> list[MergedLine]:
    """Detect long physical lines as foreground regions in each direction."""

    config = config or CvConfig()
    gray = to_grayscale(image)
    if gray.size == 0:
        return []

    _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    image_height, image_width = gray.shape[:2]
    lines: list[MergedLine] = []

    directions: tuple[tuple[Literal["horizontal", "vertical"], int], ...] = (
        ("horizontal", image_width),
        ("vertical", image_height),
    )
    for orientation, image_length in directions:
        minimum_length = max(
            config.directional_line_min_length_pixels,
            round(image_length * config.directional_line_min_length_ratio),
        )
        kernel_shape = (minimum_length, 1) if orientation == "horizontal" else (1, minimum_length)
        directional = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape),
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(directional, connectivity=8)

        for x, y, width, height, _ in stats[1:component_count]:
            component_length = width if orientation == "horizontal" else height
            if component_length < minimum_length:
                continue
            if orientation == "horizontal":
                coordinate = y + (height - 1) // 2
                p1 = (int(x), int(coordinate))
                p2 = (int(x + width - 1), int(coordinate))
            else:
                coordinate = x + (width - 1) // 2
                p1 = (int(coordinate), int(y))
                p2 = (int(coordinate), int(y + height - 1))
            lines.append(
                MergedLine(
                    p1=p1,
                    p2=p2,
                    orientation=orientation,
                    length=segment_length(*p1, *p2),
                    source_segments=(),
                )
            )

    lines.sort(key=_merged_line_sort_key)
    return lines


def merge_line_segments(segments: list[LineSegment], config: CvConfig | None = None) -> list[MergedLine]:
    """Merge compatible segments using order-independent connected components."""

    config = config or CvConfig()
    parent = list(range(len(segments)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_index, left in enumerate(segments):
        for right_index in range(left_index + 1, len(segments)):
            if _segments_are_mergeable(left, segments[right_index], config):
                union(left_index, right_index)

    groups: dict[int, list[LineSegment]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(root(index), []).append(segment)

    merged = [_merge_component(component) for component in groups.values()]
    merged.sort(key=_merged_line_sort_key)
    return merged


def _segments_are_mergeable(left: LineSegment, right: LineSegment, config: CvConfig) -> bool:
    if left.orientation != right.orientation:
        return False
    if _angle_distance(left.angle_degrees, right.angle_degrees) > config.line_merge_angle_tolerance_degrees:
        return False

    if left.orientation == "horizontal":
        left_cross = (left.p1[1] + left.p2[1]) / 2
        right_cross = (right.p1[1] + right.p2[1]) / 2
        left_interval = sorted((left.p1[0], left.p2[0]))
        right_interval = sorted((right.p1[0], right.p2[0]))
    else:
        left_cross = (left.p1[0] + left.p2[0]) / 2
        right_cross = (right.p1[0] + right.p2[0]) / 2
        left_interval = sorted((left.p1[1], left.p2[1]))
        right_interval = sorted((right.p1[1], right.p2[1]))

    if abs(left_cross - right_cross) > config.line_collinearity_tolerance_pixels:
        return False

    along_gap = max(
        0,
        left_interval[0] - right_interval[1],
        right_interval[0] - left_interval[1],
    )
    return along_gap <= config.line_along_gap_tolerance_pixels


def _merge_component(segments: list[LineSegment]) -> MergedLine:
    segments = sorted(segments, key=_segment_sort_key)
    orientation = segments[0].orientation
    weights = np.asarray([segment.length for segment in segments], dtype=float)

    if orientation == "horizontal":
        coordinates = np.asarray([(segment.p1[1] + segment.p2[1]) / 2 for segment in segments])
        coordinate = round(float(np.average(coordinates, weights=weights)))
        start = min(min(segment.p1[0], segment.p2[0]) for segment in segments)
        end = max(max(segment.p1[0], segment.p2[0]) for segment in segments)
        p1, p2 = (start, coordinate), (end, coordinate)
    else:
        coordinates = np.asarray([(segment.p1[0] + segment.p2[0]) / 2 for segment in segments])
        coordinate = round(float(np.average(coordinates, weights=weights)))
        start = min(min(segment.p1[1], segment.p2[1]) for segment in segments)
        end = max(max(segment.p1[1], segment.p2[1]) for segment in segments)
        p1, p2 = (coordinate, start), (coordinate, end)

    return MergedLine(
        p1=p1,
        p2=p2,
        orientation=orientation,
        length=segment_length(*p1, *p2),
        source_segments=tuple(segments),
    )


def _angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180
    return min(difference, 180 - difference)


def _segment_sort_key(segment: LineSegment) -> tuple[str, int, int, int, int]:
    return (segment.orientation, *segment.p1, *segment.p2)


def _merged_line_sort_key(line: MergedLine) -> tuple[str, int, int, int, int]:
    return (line.orientation, *line.p1, *line.p2)


def normalize_line_segment(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    tolerance_degrees: float,
) -> LineSegment | None:
    angle = segment_angle_degrees(x1, y1, x2, y2)
    orientation = classify_orientation(angle, tolerance_degrees)
    if orientation is None:
        return None

    if orientation == "horizontal" and x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    if orientation == "vertical" and y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    return LineSegment(
        p1=(x1, y1),
        p2=(x2, y2),
        orientation=orientation,
        length=segment_length(x1, y1, x2, y2),
        angle_degrees=angle,
    )


def segment_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def segment_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def classify_orientation(
    angle_degrees: float, tolerance_degrees: float
) -> Literal["horizontal", "vertical"] | None:
    normalized = abs(angle_degrees)
    if normalized > 90:
        normalized = abs(normalized - 180)
    if normalized <= tolerance_degrees:
        return "horizontal"
    if abs(normalized - 90) <= tolerance_degrees:
        return "vertical"
    return None
