from pathlib import Path

import cv2
import numpy as np

DEBUG_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def draw_bar_ocr_matches(
    image: np.ndarray,
    matched_bars_and_ocr: list,
    ocr_json: dict,
    output_path: str | Path,
) -> Path:
    canvas = image.copy()
    matched_ocr_boxes = set()

    for index, (bar_candidate, ocr_boxes) in enumerate(matched_bars_and_ocr):
        color = DEBUG_COLORS[index % len(DEBUG_COLORS)]
        bbox = bar_candidate.bbox

        cv2.rectangle(
            canvas,
            (bbox.x, bbox.y),
            (bbox.right, bbox.bottom),
            color,
            2,
        )

        cv2.putText(
            canvas,
            f"bar {index}",
            (bbox.x, max(20, bbox.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        for ocr_box in ocr_boxes:
            matched_ocr_boxes.add(tuple(ocr_box))
            left, top, right, bottom = normalize_ocr_bbox(ocr_box)
            ocr_index = get_ocr_index(ocr_json, ocr_box)
            label = get_ocr_label(ocr_json, ocr_index)
            confidence = get_ocr_confidence(ocr_json, ocr_index)
            cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
            cv2.putText(
                canvas,
                f"{label} ({confidence:.2f})",
                (left, bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    for index, ocr_box in enumerate(ocr_json.get("bbox", [])):
        normalized_box = tuple(ocr_box)
        if normalized_box in matched_ocr_boxes:
            continue

        left, top, right, bottom = normalize_ocr_bbox(ocr_box)
        label = get_ocr_label(ocr_json, index)
        confidence = get_ocr_confidence(ocr_json, index)

        cv2.rectangle(canvas, (left, top), (right, bottom), (160, 160, 160), 1)
        cv2.putText(
            canvas,
            f"unmatched: {label} ({confidence:.2f})",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def normalize_ocr_bbox(
    bbox: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


def get_ocr_label(ocr_json: dict, index: int) -> str:
    if index < 0:
        return "?"

    labels = ocr_json.get("labels", [])
    if index >= len(labels):
        return "?"
    return str(labels[index])


def get_ocr_confidence(ocr_json: dict, index: int) -> float:
    if index < 0:
        return 0.0

    confidence = ocr_json.get("confidence", [])
    if index >= len(confidence):
        return 0.0
    return float(confidence[index])


def get_ocr_index(ocr_json: dict, target_bbox: list[int] | tuple[int, int, int, int]) -> int:
    for index, bbox in enumerate(ocr_json.get("bbox", [])):
        if tuple(bbox) == tuple(target_bbox):
            return index

    return -1
