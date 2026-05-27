import json

from paddleocr import PaddleOCR

# PNG/JPG --> labels, values, confidence score.

class Ocr:
    def __init__(self, input_path=None):
        if input_path:
            self.input_path = input_path

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
            use_doc_unwarping=False, # Disables text image rectification model via this parameter
            use_textline_orientation=False, # Disables text line orientation classification model via this parameter
            lang="en",
        )

    def read_image(self, input_path) -> None:
        self.input_path = input_path

    def run_ocr(self) -> str:
        result = self.ocr.predict(self.input_path)
        extracted = {}
        for res in result:
            # res.save_to_img("output")  
            # res.save_to_json("output")
            extracted["input_path"] = res.get("input_path")
            extracted["labels"] = res.get("rec_texts")
            extracted["confidence"] = res.get("rec_scores")

        return json.dumps(extracted, indent=4)


if __name__ == "__main__":
    input_path = input("Enter path for input: ")
    ocr = Ocr(input_path)
    result = ocr.run_ocr()
    print(result)
