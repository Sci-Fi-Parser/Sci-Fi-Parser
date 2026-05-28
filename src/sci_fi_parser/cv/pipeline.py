from pathlib import Path

from ocr import Ocr
import bars

import cv2

def start_ocr(folder: str, ocr_set: dict) -> dict:
    path = Path(folder).glob("*.jpg")
    result = {}
    ocr = Ocr()
    for i, image in enumerate(path):
        image_array = cv2.imread(image)
        bar_candidates = bars.detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_json = ocr.run_ocr()

        result[i] = bar_candidates, ocr_json

    return result

if __name__ == "__main__":
    input_path = input("Enter input path: ")
    print(start_ocr(input_path, {}))
