from pathlib import Path

from sci_fi_parser.cv.ocr import Ocr

import sci_fi_parser.cv.bars as bars
from sci_fi_parser.cv.debug_draw import draw_bar_ocr_matches

import cv2

def start_ocr(folder: Path, ocr_set: dict) -> None:
    path = [p for paths in ("*.png", "*.jpg", "*.jpeg") for p in Path(folder).glob(paths)]    
    ocr = Ocr()
    for image in path:
        image_array = cv2.imread(image)
        bar_candidates = bars.detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_res = ocr.run_ocr()

        matched = match_bars_and_ocr(bar_candidates, ocr_res)

        # draw_bar_ocr_matches(
        #     image_array,
        #     matched,
        #     ocr_res,
        #     image.with_name(f"{image.stem}_debug{image.suffix}"),
        # )

        output_string = str(bar_candidates) + str(ocr_res) + str(matched)
        # print("run")

        ocr_set.add(image.name, output_string)

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



