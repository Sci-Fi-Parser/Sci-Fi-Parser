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
    # Width of 2px is intentionally below CvConfig's default
    # min_bar_width_pixels, so the default config must reject it while a
    # relaxed config (lower width/height/area minimums) accepts it.
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (21, 100), 0, thickness=-1)

    default_candidates = bars.detect_bars(image)
    assert default_candidates == []

    relaxed_config = CvConfig(
        min_bar_area_ratio=0.0,
        min_bar_height_ratio=0.01,
        min_bar_width_pixels=1,
        max_bar_width_ratio=1.0,
        min_bar_rectangularity=0.0,
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


def test_detects_black_and_gray_bars_in_bgr_image() -> None:
    image = np.full((180, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (25, 40), (50, 160), (0, 0, 0), -1)
    cv2.rectangle(image, (120, 70), (150, 160), (130, 130, 130), -1)

    result = bars.detect_bars_with_diagnostics(image)

    assert len(result.bars) == 2
    assert result.diagnostics.mask_source == "grayscale"
    assert result.diagnostics.grayscale_mask is not None


def test_otsu_threshold_ignores_low_saturation_jpeg_noise() -> None:
    image = np.full((180, 240, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (25, 35), (55, 160), (30, 170, 210), -1)
    cv2.rectangle(image, (130, 65), (165, 160), (30, 170, 210), -1)
    encoded, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert encoded
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)

    result = bars.detect_bars_with_diagnostics(image)

    assert len(result.bars) == 2
    assert result.diagnostics.mask_source == "saturation"
    assert result.diagnostics.effective_saturation_threshold > 3
    assert result.diagnostics.saturation_mask is not None
    foreground_ratio = cv2.countNonZero(result.diagnostics.saturation_mask) / image.shape[0] / image.shape[1]
    assert foreground_ratio < 0.2


def test_pale_unusual_bars_remain_visible() -> None:
    image = np.full((180, 260, 3), 250, dtype=np.uint8)
    colors = ((220, 235, 245), (235, 220, 245), (240, 235, 215))
    for index, color in enumerate(colors):
        left = 25 + index * 80
        cv2.rectangle(image, (left, 50 + index * 10), (left + 25, 160), color, -1)

    assert len(bars.detect_bars(image)) == 3


def test_gray_fill_with_black_outline_produces_one_proposal() -> None:
    image = np.full((180, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 35), (105, 160), (0, 0, 0), 2)
    cv2.rectangle(image, (62, 37), (103, 158), (170, 170, 170), -1)

    result = bars.detect_bars_with_diagnostics(image)

    assert len(result.bars) == 1
    assert any(contour.deduplicated_to for contour in result.diagnostics.contours)


def test_low_saturation_fill_with_colored_outline_produces_one_proposal() -> None:
    image = np.full((180, 220, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (65, 40), (110, 160), (0, 0, 230), 2)
    cv2.rectangle(image, (67, 42), (108, 158), (220, 220, 240), -1)

    assert len(bars.detect_bars(image)) == 1


def test_one_and_multi_pixel_outlines_are_preserved_by_raw_pass() -> None:
    image = np.full((180, 280, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (30, 35), (70, 160), (0, 0, 230), 1)
    cv2.rectangle(image, (150, 55), (195, 160), (230, 0, 0), 3)

    result = bars.detect_bars_with_diagnostics(image)

    assert len(result.bars) == 2
    assert any(contour.source == "outline" and contour.accepted for contour in result.diagnostics.contours)


def test_short_bar_near_baseline_requires_relaxed_size_config() -> None:
    image = np.full((200, 200, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (80, 175), (95, 180), (0, 0, 230), -1)

    assert bars.detect_bars(image) == []
    relaxed = CvConfig(min_bar_area_ratio=0.0002, min_bar_height_ratio=0.01)
    assert len(bars.detect_bars(image, relaxed)) == 1


def test_one_and_two_pixel_group_gaps_remain_separate() -> None:
    image = np.full((180, 220, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (30, 50), (49, 160), (0, 0, 230), -1)
    cv2.rectangle(image, (51, 70), (70, 160), (0, 180, 0), -1)
    cv2.rectangle(image, (120, 40), (139, 160), (230, 0, 0), -1)
    cv2.rectangle(image, (142, 65), (161, 160), (0, 180, 180), -1)

    assert len(bars.detect_bars(image)) == 4


def test_touching_grouped_bars_are_explicitly_merged() -> None:
    image = np.full((180, 180, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (40, 45), (69, 160), (0, 0, 230), -1)
    cv2.rectangle(image, (70, 75), (99, 160), (230, 0, 0), -1)

    candidates = bars.detect_bars(image)

    assert len(candidates) == 1
    assert candidates[0].bbox.x <= 40
    assert candidates[0].bbox.right >= 100


def test_colored_text_grid_ticks_and_legend_markers_are_negative_controls() -> None:
    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    for y in (40, 80, 120, 160):
        cv2.line(image, (40, y), (290, y), (220, 220, 220), 1)
        cv2.line(image, (34, y), (40, y), (0, 0, 0), 1)
    cv2.line(image, (40, 30), (40, 180), (0, 0, 0), 1)
    cv2.putText(image, "Legend", (190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    cv2.circle(image, (180, 20), 3, (220, 0, 0), -1)
    cv2.putText(image, "100", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    assert bars.detect_bars(image) == []


def test_diagnostics_include_sources_rejections_and_morphology() -> None:
    image = np.full((160, 200, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (30, 35), (55, 145), (0, 0, 230), -1)
    cv2.line(image, (120, 30), (120, 145), (0, 180, 0), 1)

    result = bars.detect_bars_with_diagnostics(image)
    payload = result.diagnostics.to_dict()

    assert len(result.bars) == 1
    assert payload["effective_saturation_threshold"] is not None
    assert payload["morphology"] == {"width": 1, "height": 3}
    assert any(contour["accepted"] for contour in payload["contours"])
    assert any(contour["rejection_reasons"] for contour in payload["contours"])
    assert any(contour["deduplicated_to"] for contour in payload["contours"])
