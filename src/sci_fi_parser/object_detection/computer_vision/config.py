from dataclasses import dataclass


### AI generated default config, you can and should experiment with changing values
@dataclass(slots=True)
class CvConfig:
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 60
    max_line_gap: int = 2
    angle_tolerance_degrees: float = 10.0
    min_bar_area_ratio: float = 0.0008
    min_bar_height_ratio: float = 0.04
    min_bar_width_pixels: int = 5
    max_bar_width_ratio: float = 0.50
