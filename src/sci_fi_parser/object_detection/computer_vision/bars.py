from dataclasses import dataclass

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision.config import CvConfig


@dataclass(slots=True)
class BoundingBox:
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
    bbox: BoundingBox


def detect_bars(
    image: np.ndarray,
    config: CvConfig | None = None,
) -> list[BarCandidate]:
    config = config or CvConfig()

    image_height, image_width = image.shape[:2]

    if image_width == 0 or image_height == 0:
        return []

    binary = threshold_foreground(image)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_bar_area = image_height * image_width * config.min_bar_area_ratio
    max_bar_width = max(
        config.min_bar_width_pixels,
        int(image_width * config.max_bar_width_ratio),
    )
    min_bar_height = max(
        1,
        int(image_height * config.min_bar_height_ratio),
    )

    bars: list[BarCandidate] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height

        if area < min_bar_area:
            continue

        if width < config.min_bar_width_pixels or width > max_bar_width:
            continue

        if height < min_bar_height:
            continue

        box = BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        bars.append(
            BarCandidate(
                bbox=box,
            )
        )

    bars.sort(key=lambda bar: bar.bbox.x)

    return bars


def threshold_foreground(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        _, binary = cv2.threshold(
            saturation,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        # Remove thin label connections while preserving vertical bar blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    gray = to_grayscale(image)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    return binary


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
