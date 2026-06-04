from dataclasses import dataclass
from pathlib import Path

from sci_fi_parser.cv.ocr import Ocr

import sci_fi_parser.cv.bars as bars
from sci_fi_parser.cv.debug_draw import draw_bar_ocr_matches

import cv2

@dataclass
class OcrExtractionResult:
    image_name: str
    bar_candidates: object
    ocr_result: object
    matched: object


def get_image_paths(folder: Path) -> list[Path]:
    return [
        p
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for p in folder.glob(pattern)
    ]


def extract_ocr_data(folder: Path) -> list[OcrExtractionResult]:
    paths = get_image_paths(folder)
    ocr = Ocr()

    results = []

    for image_path in paths:
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


def format_ocr_output(result: OcrExtractionResult) -> str:
    return f"{result.bar_candidates}{result.ocr_result}{result.matched}"


def start_ocr(folder: Path, ocr_set) -> None:
    results = extract_ocr_data(folder)

    for result in results:
        output_string = format_ocr_output(result)
        ocr_set.add(result.image_name, output_string)

def match_bars_and_ocr(bars: list, ocr_json: dict) -> list:
    linked = []
    for i in range(len(bars)):
        left = bars[i].bbox.x
        right = bars[i].bbox.right
        # print(f"left: {left}, right: {right}")
        matching_ocr = []
        for j in range(len(ocr_json["bbox"])):
            if ocr_json["confidence"][j] < 0.90:
                # print("low conf")
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


if __name__ == "__main__":
    input_path = input("Enter input path: ")
    ocr_set = OCRSet()
    start_ocr(input_path, ocr_set)
    print(ocr_set)


