"""Run the OCR -> VLM -> offload pipeline on a folder of chart images.

Edit ``TARGET_FOLDER`` / ``OUTPUT_FOLDER`` below; no CLI flags. The work
lives in :mod:`sci_fi_parser.data_pipeline` (data containers + offloader),
:mod:`sci_fi_parser.cv.pipeline` (OCR stage), and
:mod:`sci_fi_parser.vlm.pipeline` (VLM stage).
"""

from pathlib import Path

from sci_fi_parser.classifier.classifier_pipeline import start_classification
from sci_fi_parser.object_detection.detection_pipeline import start_ocr
from sci_fi_parser.schema import ImageSet, PdfSet
from sci_fi_parser.image_extraction.extraction_pipeline import start_extraction
from sci_fi_parser.storage.writer import save_image_set
from sci_fi_parser.vlm.vlm_pipeline import start_vlm

PDF_INPUT_FOLDER = Path("train_data/small_pdfs")
EXTRACTED_IMAGE_FOLDER = Path("temp/extracted_images")
OUTPUT_FOLDER = Path("output")
VLM_CONFIG = Path("config/vlm.toml")


def main() -> None:
    image_set = ImageSet()
    pdf_set = PdfSet()

    start_extraction(PDF_INPUT_FOLDER, image_set, pdf_set, EXTRACTED_IMAGE_FOLDER)
    start_classification(image_set)
    start_ocr(image_set)
    start_vlm(image_set, VLM_CONFIG)

    save_image_set(image_set=image_set, output_dir=Path(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
