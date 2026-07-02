from dataclasses import dataclass
from pathlib import Path

import cv2

from sci_fi_parser.object_detection.computer_vision.bars import detect_bars
from sci_fi_parser.object_detection.ocr import Ocr
from sci_fi_parser.schema import ImageSet


SUPPORTED_CHARTS = ["bar_chart"]


@dataclass
class OcrExtractionResult:
    image_name: str
    bar_candidates: object
    ocr_result: object
    matched: object


def extract_ocr_data(paths: list[tuple[str, Path]]) -> list[OcrExtractionResult]:
    ocr = Ocr()
    results = []

    for image in paths:
        image_path = image[1]
        image_array = cv2.imread(str(image_path))

        bar_candidates = detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_result = ocr.run_ocr()

        matched = match_bars_and_ocr(bar_candidates, ocr_result)

        results.append(
            OcrExtractionResult(
                image_name=image_path.name,
                bar_candidates=bar_candidates,
                ocr_result=ocr_result,
                matched=matched,
            )
        )

    return results


def match_bars_and_ocr(bars: list, ocr_json: dict) -> list:
    linked = []
    for bar in bars:
        left = bar.bbox.x
        right = bar.bbox.right
        # print(f"left: {left}, right: {right}")
        matching_ocr = []
        for i, bbox in enumerate(ocr_json["bbox"]):
            if ocr_json["confidence"][i] < 0.90:
                #  print("low conf")
                continue
            ocr_max_x, _, ocr_min_x, _ = bbox
            # print(f"max: {ocr_max_x}, min: {ocr_min_x}")
            if (ocr_min_x <= left and ocr_max_x >= right) or (
                ocr_min_x >= left and ocr_max_x <= right
            ):
                matching_ocr.append(bbox)
        linked.append((bar, matching_ocr))

    # print(f"linked: {linked}")
    return linked


def format_ocr_output(result: OcrExtractionResult) -> str:
    return f"{result.bar_candidates}{result.ocr_result}{result.matched}"


def start_ocr(image_set: ImageSet, batch_size=100) -> None:
    if batch_size <= 0:
        print("[!] OCR/CV batch size is less than 0. Skipping.")
        return

    for chart_type in SUPPORTED_CHARTS:
        chart_ids = image_set.filter_by_type(chart_type, batch_size)
        if not chart_ids:
            continue

        image_paths: list[tuple[str, Path]] = []
        for image_id in chart_ids:
            image_paths.append((image_id, image_set.get_image_path(image_id)))

        results = extract_ocr_data(image_paths)
        for image_id, result in zip(chart_ids, results):
            image_set.add_ocrcv_result(image_id, format_ocr_output(result))
            image_set.add_ocrcv_raw(image_id, result)
