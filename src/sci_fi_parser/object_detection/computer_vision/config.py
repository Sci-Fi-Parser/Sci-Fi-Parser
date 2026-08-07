from dataclasses import dataclass


### AI generated default config, you can and should experiment with changing values
@dataclass(slots=True)
class CvConfig:
    canny_threshold1: int = 10
    canny_threshold2: int = 30
    hough_threshold: int | None = None
    max_line_gap: int | None = None
    angle_tolerance_degrees: float = 10.0
    line_hough_threshold: int = 4
    line_hough_max_gap_pixels: int = 2
    line_min_length_ratio: float = 0.010
    line_min_length_pixels: int = 10
    line_merge_angle_tolerance_degrees: float = 3.0
    line_collinearity_tolerance_pixels: float = 5.0
    line_along_gap_tolerance_pixels: float = 8.0
    min_bar_area_ratio: float = 0.0008
    min_bar_height_ratio: float = 0.04
    min_bar_width_pixels: int = 5
    max_bar_width_ratio: float = 0.50
