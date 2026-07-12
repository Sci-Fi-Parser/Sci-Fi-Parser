from __future__ import annotations

import cv2
import numpy as np

from sci_fi_parser.object_detection.computer_vision import bars
from sci_fi_parser.object_detection.computer_vision.config import CvConfig


def _make_grayscale_chart() -> np.ndarray:
    image = np.full((180, 180), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 50), (40, 160), 0, thickness=-1)
    cv2.rectangle(image, (95, 30), (125, 160), 0, thickness=-1)
    cv2.rectangle(image, (150, 150), (151, 151), 0, thickness=-1)
    return image


def _make_color_chart() -> np.ndarray:
    image = np.full((180, 180, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (20, 50), (40, 160), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (95, 30), (125, 160), (255, 0, 0), thickness=-1)
    cv2.rectangle(image, (150, 150), (151, 151), (0, 255, 0), thickness=-1)
    return image


def _make_three_bar_chart() -> np.ndarray:
    image = np.full((150, 220), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 40), (30, 140), 0, thickness=-1)
    cv2.rectangle(image, (90, 20), (110, 140), 0, thickness=-1)
    cv2.rectangle(image, (170, 60), (190, 140), 0, thickness=-1)
    return image


def _assert_bbox_covers_region(
    bbox: bars.BoundingBox,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    assert bbox.x <= left
    assert bbox.y <= top
    assert bbox.right >= right
    assert bbox.bottom >= bottom


def test_bounding_box_properties() -> None:
    bbox = bars.BoundingBox(x=10, y=20, width=30, height=40)

    assert bbox.right == 40
    assert bbox.bottom == 60
    assert bbox.center_x == 25.0
    assert bbox.center_y == 40.0


def test_bounding_box_zero_dimensions() -> None:
    bbox = bars.BoundingBox(x=5, y=5, width=0, height=0)

    assert bbox.right == 5
    assert bbox.bottom == 5
    assert bbox.center_x == 5.0
    assert bbox.center_y == 5.0


def test_detect_bars_returns_no_candidates_for_zero_sized_image() -> None:
    zero_width = np.zeros((50, 0), dtype=np.uint8)
    zero_height = np.zeros((0, 50), dtype=np.uint8)

    assert bars.detect_bars(zero_width) == []
    assert bars.detect_bars(zero_height) == []


def test_detect_bars_returns_no_candidates_for_near_blank_image() -> None:
    rng = np.random.default_rng(seed=0)
    image = np.full((120, 120), 255, dtype=np.uint8)
    noise_mask = rng.random((120, 120)) < 0.02
    image[noise_mask] = 0

    assert bars.detect_bars(image) == []


def test_detect_bars_finds_bars_in_grayscale_chart() -> None:
    image = _make_grayscale_chart()

    candidates = bars.detect_bars(image)

    assert len(candidates) == 2
    assert candidates[0].bbox.x < candidates[1].bbox.x
    _assert_bbox_covers_region(
        candidates[0].bbox,
        left=20,
        top=50,
        right=40,
        bottom=160,
    )
    _assert_bbox_covers_region(
        candidates[1].bbox,
        left=95,
        top=30,
        right=125,
        bottom=160,
    )


def test_detect_bars_finds_bars_in_color_chart() -> None:
    image = _make_color_chart()

    candidates = bars.detect_bars(image)

    assert len(candidates) == 2
    assert candidates[0].bbox.x < candidates[1].bbox.x
    _assert_bbox_covers_region(
        candidates[0].bbox,
        left=20,
        top=50,
        right=40,
        bottom=160,
    )
    _assert_bbox_covers_region(
        candidates[1].bbox,
        left=95,
        top=30,
        right=125,
        bottom=160,
    )


def test_detect_bars_orders_multiple_bars_left_to_right() -> None:
    image = _make_three_bar_chart()

    candidates = bars.detect_bars(image)

    assert len(candidates) == 3
    xs = [candidate.bbox.x for candidate in candidates]
    assert xs == sorted(xs)
    _assert_bbox_covers_region(
        candidates[0].bbox,
        left=10,
        top=40,
        right=30,
        bottom=140,
    )
    _assert_bbox_covers_region(
        candidates[1].bbox,
        left=90,
        top=20,
        right=110,
        bottom=140,
    )
    _assert_bbox_covers_region(
        candidates[2].bbox,
        left=170,
        top=60,
        right=190,
        bottom=140,
    )


def test_detect_bars_ignores_small_noise() -> None:
    image = np.full((100, 100), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (11, 11), 0, thickness=-1)

    assert bars.detect_bars(image) == []


def test_detect_bars_respects_configuration_thresholds() -> None:
    # Width of ~2px is intentionally below CvConfig's default
    # min_bar_width_pixels, so the default config must reject it while a
    # relaxed config (lower width/height/area minimums) accepts it.
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (22, 100), 0, thickness=-1)

    default_candidates = bars.detect_bars(image)
    assert default_candidates == []

    relaxed_config = CvConfig(
        min_bar_area_ratio=0.0,
        min_bar_height_ratio=0.01,
        min_bar_width_pixels=1,
        max_bar_width_ratio=1.0,
    )
    relaxed_candidates = bars.detect_bars(image, config=relaxed_config)

    assert len(relaxed_candidates) == 1
    _assert_bbox_covers_region(
        relaxed_candidates[0].bbox,
        left=20,
        top=20,
        right=22,
        bottom=100,
    )