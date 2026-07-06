"""OCR wrapper around PaddleOCR.

Provides a small helper class for loading an input image path and running OCR
with predefined PaddleOCR settings.
"""

from typing import Any

from paddleocr import PaddleOCR


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

    def run_ocr(self) -> dict[str, Any]:
        """Run OCR on the stored input and return extracted data.

        Returns:
            dict[str, Any]: A dict containing the results from the OCR:

            - "labels" `list[str]`: List of text recognition results.
            - "confidence" `list[float]`: List of text recognition confidence scores.
            - "bbox" `list[numpy.ndarray]`: List of text detection boxes filtered by confidence.
        """
        result = self.ocr.predict(self.input_data)
        extracted = {}
        for res in result:
            # res.save_to_img("output")
            # res.save_to_json("output")
            extracted["labels"] = res.get("rec_texts")
            extracted["confidence"] = res.get("rec_scores")
            extracted["bbox"] = res.get("rec_boxes")

        return extracted


if __name__ == "__main__":
    input_path = input("Enter path for input: ")
    ocr = Ocr(input_path)
    result = ocr.run_ocr()
    print(result)
