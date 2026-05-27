from pathlib import Path

from ocr import Ocr
import bars

import cv2

def start_ocr(folder: str) -> list:
    path = Path(folder).glob("*.jpg")
    result = []
    for image in path:
        print(image)
        image_array = cv2.imread(image)
        bar_candidates = bars.detect_bars(image_array)
        result.append(bar_candidates)

        ocr = Ocr(image_array)
        ocr_json = ocr.run_ocr()
        result.append(ocr_json)

    return result

if __name__ == "__main__":
    input_path = input("Enter input path: ")
    print(start_ocr(input_path))
