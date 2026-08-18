"""OCR wrapper around PaddleOCR.

Provides a small helper class for loading an input image path and running OCR
with predefined PaddleOCR settings.
"""

from typing import Any

from paddleocr import PaddleOCR  # type: ignore[import-untyped]

from sci_fi_parser.object_detection.models import OcrOutput
from sci_fi_parser.object_detection.normalization import normalize_ocr_output


class Ocr:
    """Wrapper around PaddleOCR.

    Attributes:
        input_data: Data to be predicted.
    """

    def __init__(self, input_data=None):
        """Create an OCR helper and optionally set the input path.

        Attributes:
            input_data (optional): Data to be predicted. Can be `MatLike`, `numpy.ndarray`,
            `str` of local path of an image file or PDf file. See PaddleOCR
            documentation for predict() for more.
        """
        if input_data:
            self.input_data = input_data

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            # Disables document orientation classification model
            use_doc_unwarping=False,  # Disables text image rectification model
            use_textline_orientation=False,
            # Disables text line orientation classification model
            lang="en",
        )

    def read_image(self, input_data) -> None:
        """Store the image data that should be processed next."""
        self.input_data = input_data

    def run_ocr(self) -> OcrOutput:
        """Run OCR and immediately convert Paddle's parallel arrays to typed tokens."""
        result = self.ocr.predict(self.input_data)
        labels: list[Any] = []
        confidences: list[Any] = []
        boxes: list[Any] = []
        for res in result:
            raw_labels = res.get("rec_texts")
            raw_confidences = res.get("rec_scores")
            labels.extend(raw_labels.tolist() if hasattr(raw_labels, "tolist") else raw_labels or [])
            confidences.extend(
                raw_confidences.tolist()
                if hasattr(raw_confidences, "tolist")
                else raw_confidences or []
            )
            raw_boxes = res.get("rec_boxes")
            boxes.extend(raw_boxes.tolist() if hasattr(raw_boxes, "tolist") else raw_boxes or [])
        return normalize_ocr_output(labels, confidences, boxes)


if __name__ == "__main__":
    input_path = input("Enter path for input: ")
    ocr = Ocr(input_path)
    result = ocr.run_ocr()
    print(result)
