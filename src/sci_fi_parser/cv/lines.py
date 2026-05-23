from config import CvConfig

import cv2
import numpy as np
import math
from dataclasses import dataclass


@dataclass(slots=True)
class LineSegment:
    p1: tuple[int, int]
    p2: tuple[int, int]
    orientation: str
    length: float
    angle_degrees: float

def to_grayscale(image: np.ndarray) -> np.ndarray: # This should probably be done in the image preprocessing part
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_line_segments(image: np.ndarray, config: CvConfig | None = None) -> list[LineSegment]:
    config = config or CvConfig()
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, config.canny_threshold1, config.canny_threshold2, apertureSize=3)

    image_height, image_width = gray.shape[:2]
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=10,
        maxLineGap=config.max_line_gap,
    )

    if raw_lines is None:
        return []

    max_length = float(max(image_height, image_width))
    line_segments: list[LineSegment] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
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

def classify_orientation(angle_degrees: float, tolerance_degrees: float) -> str | None:
    normalized = abs(angle_degrees)
    if normalized > 90:
        normalized = abs(normalized - 180)
    if normalized <= tolerance_degrees:
        return "horizontal"
    if abs(normalized - 90) <= tolerance_degrees:
        return "vertical"
    return None

