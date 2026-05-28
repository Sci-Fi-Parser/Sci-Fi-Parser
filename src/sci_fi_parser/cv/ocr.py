from paddleocr import PaddleOCR

# PNG/JPG --> labels, values, confidence score.


class Ocr:
    def __init__(self, input_path=None):
        if input_path:
            self.input_path = input_path

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            # Disables document orientation classification model
            use_doc_unwarping=False,  # Disables text image rectification model
            use_textline_orientation=False,
            # Disables text line orientation classification model
            lang="en",
        )

    def read_image(self, input_path) -> dict:
        self.input_path = input_path

    def run_ocr(self) -> dict:
        result = self.ocr.predict(self.input_path)
        extracted = {}
        for res in result:
            # res.save_to_img("output")
            # res.save_to_json("output")
            extracted["labels"] = res.get("rec_texts")
            extracted["confidence"] = res.get("rec_scores")

        return extracted


if __name__ == "__main__":
    input_path = input("Enter path for input: ")
    ocr = Ocr(input_path)
    result = ocr.run_ocr()
    print(result)
