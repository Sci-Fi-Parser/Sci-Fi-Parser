from dataclasses import dataclass


@dataclass(slots=True)
class CvConfig:
    canny_threshold1: int = 10
    canny_threshold2: int = 30
    hough_threshold: int = 4
    max_line_gap: int = 2
    angle_tolerance_degrees: float = 10.0
    line_min_length_ratio: float = 0.010
    line_min_length_pixels: int = 10
    line_merge_angle_tolerance_degrees: float = 3.0
    line_collinearity_tolerance_pixels: float = 5.0
    line_along_gap_tolerance_pixels: float = 8.0
    directional_line_min_length_ratio: float = 0.05
    directional_line_min_length_pixels: int = 10
    min_bar_area_ratio: float = 0.0008
    min_bar_height_ratio: float = 0.04
    min_bar_width_pixels: int = 5
    max_bar_width_ratio: float = 0.50


@dataclass(slots=True)
class YAxisConfig:
    """Thresholds and weights for Y-axis evidence and abstention."""

    numeric_ocr_min_confidence: float = 0.80
    label_x_tolerance_pixels: float = 1.5
    label_max_distance_ratio: float = 0.18
    label_max_distance_pixels: float = 55.0
    label_min_count: int = 3
    label_min_span_ratio: float = 0.20
    minimum_label_correlation: float = 0.75
    maximum_bar_overlap_ratio: float = 0.10
    border_tolerance_pixels: float = 2.0
    minimum_score: float = 4.0
    minimum_margin: float = 0.60
