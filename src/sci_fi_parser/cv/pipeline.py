from pathlib import Path

from cv.ocr import Ocr

import cv.bars as bars
from cv.debug_draw import draw_bar_ocr_matches

from sci_fi_parser.cv.serializers import (
    serialize_bar_candidate,
    serialize_ocr_result,
    serialize_matched,
)

import cv2

def start_ocr(folder: Path, ocr_set: dict, writer, run_id) -> None:
    paths_list = [p for paths in ("*.png", "*.jpg", "*.jpeg") for p in Path(folder).glob(paths)]    

    ocr = Ocr()

    for image_path in paths_list:

        chart_id = image_path.stem
        
        image_array = cv2.imread(image_path)
        bar_candidates = bars.detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_res = ocr.run_ocr()

        matched = match_bars_and_ocr(bar_candidates, ocr_res)

        overlay_image = draw_bar_ocr_matches(
            image_array,
            matched,
            ocr_res,
            image_path.with_name(f"{image_path.stem}_debug{image_path.suffix}"))

        serialized_ocr = {
            "bar_candidates": [
                serialize_bar_candidate(candidate)
                for candidate in bar_candidates
            ],
            "ocr_result": serialize_ocr_result(ocr_res),
            "matched": serialize_matched(matched),
        }

        # SAVING DATA TO OUTPUT FOLDER WITH WRITER
        raw_ocr_path = writer.save_raw_ocr(chart_id, serialized_ocr)
        chart_crop_path = writer.save_chart_crop(image_array, chart_id)
        overlay_path = writer.save_overlay(overlay_image, chart_id)

        writer.write_ocr_result({
            "run_id": run_id,
            "chart_id": chart_id,
            "image_name": image_path.name,
            "chart_crop_path": chart_crop_path,
            "overlay_path": overlay_path,
            "raw_ocr_path": raw_ocr_path,
            "bar_count": len(serialized_ocr["bar_candidates"]),
            "ocr_item_count": len(serialized_ocr["ocr_result"]["labels"]),
            "matched_count": len(serialized_ocr["matched"]),
        })

        output_string = str(bar_candidates) + str(ocr_res) + str(matched)
        ocr_set.add(image_path.name, output_string)



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



