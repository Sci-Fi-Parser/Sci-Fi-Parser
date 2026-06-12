from dataclasses import dataclass
from pathlib import Path

from sci_fi_parser.cv.ocr import Ocr

import sci_fi_parser.cv.bars as bars
from sci_fi_parser.cv.debug_draw import draw_bar_ocr_matches

from sci_fi_parser.data_pipeline import ImageSet

import cv2


SUPPORTED_CHARTS = ["bar"]


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

        bar_candidates = bars.detect_bars(image_array)

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
    for i in range(len(bars)):
        left = bars[i].bbox.x
        right = bars[i].bbox.right
        # print(f"left: {left}, right: {right}")
        matching_ocr = []
        for j in range(len(ocr_json["bbox"])):
            if ocr_json["confidence"][j] < 0.90:
                #  print("low conf")
                continue
            ocr_max_x, _, ocr_min_x, _ = ocr_json["bbox"][j]
            # print(f"max: {ocr_max_x}, min: {ocr_min_x}")
            if (ocr_min_x <= left and ocr_max_x >= right) or (
                ocr_min_x >= left and ocr_max_x <= right
            ):
                matching_ocr.append(ocr_json["bbox"][j])
        linked.append((bars[i], matching_ocr))

    # print(f"linked: {linked}")
    return linked


def format_ocr_output(result: OcrExtractionResult) -> str:
    return f"{result.bar_candidates}{result.ocr_result}{result.matched}"


def start_ocr(image_set: ImageSet, batch_size=100) -> None:
    if batch_size <= 0:
        print("[!] OCR/CV batch size is less than 0. Skipping.")
        return

    for chart_type in SUPPORTED_CHARTS:
        # For now we assume all images are a single type.
        # chart_ids = image_set.filter_by_type(chart_type, batch_size)
        # if not chart_ids:
        # continue

        chart_ids: list[str] = [image_id for image_id in image_set]

        image_paths: list[tuple[str, Path]] = []
        for image_id in chart_ids:
            image_paths.append((image_id, image_set.get_image_path(image_id)))

        results = extract_ocr_data(image_paths)
        for image_id, result in zip(chart_ids, results):
            image_set.add_ocrcv_result(image_id, format_ocr_output(result))
            image_set.add_ocrcv_raw(image_id, result)
