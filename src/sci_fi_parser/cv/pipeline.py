from pathlib import Path

from ocr import Ocr

import bars

import cv2

def start_ocr(folder: str, ocr_set: dict):
    path = Path(folder).glob("*.jpg")
    ocr = Ocr()
    for image in path:
        image_array = cv2.imread(image)
        bar_candidates = bars.detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_res = ocr.run_ocr()
        ocr_set.add(image , (bar_candidates, ocr_res))


if __name__ == "__main__":
    input_path = input("Enter input path: ")
    ocr_set = OCRSet()
    start_ocr(input_path, ocr_set)
    print(ocr_set)
