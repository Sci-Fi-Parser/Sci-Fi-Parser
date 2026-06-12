"""Run the OCR -> VLM -> offload pipeline on a folder of chart images.

Edit ``TARGET_FOLDER`` / ``OUTPUT_FOLDER`` below; no CLI flags. The work
lives in :mod:`sci_fi_parser.data_pipeline` (data containers + offloader),
:mod:`sci_fi_parser.cv.pipeline` (OCR stage), and
:mod:`sci_fi_parser.vlm.pipeline` (VLM stage).
"""

from pathlib import Path

from sci_fi_parser.data_pipeline import OCRSet, VLMSet, ImageSet, PdfSet
from sci_fi_parser.vlm.pipeline import start_vlm
from sci_fi_parser.cv.pipeline import start_ocr
from sci_fi_parser.image_extraction.pipeline import start_extraction

PDF_INPUT_FOLDER = Path("train_data/small_pdfs")
EXTRACTED_IMAGE_FOLDER = Path("temp/extracted_images")
OUTPUT_FOLDER = Path("output")
VLM_CONFIG = Path("config/vlm.toml")

def main() -> None:
    image_set = ImageSet()
    pdf_set = PdfSet()

    start_extraction(PDF_INPUT_FOLDER, image_set, pdf_set, EXTRACTED_IMAGE_FOLDER)
    start_ocr(image_set)
    start_vlm(image_set, VLM_CONFIG)
    for image_id, _ in image_set.items():
        print("##########")
        print(image_set.get_ocrcv_result(image_id))
        print(image_set.get_vlm_result(image_id))

if __name__ == "__main__":
    main()
