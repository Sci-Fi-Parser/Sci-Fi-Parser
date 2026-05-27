"""Run the OCR -> VLM -> offload pipeline on a folder of chart images.

Edit ``TARGET_FOLDER`` / ``OUTPUT_FOLDER`` below; no CLI flags. The work
lives in :mod:`sci_fi_parser.pipeline`.
"""

from pathlib import Path

from sci_fi_parser.pipeline import (
    OCRSet,
    VLMSet,
    data_offloader,
    start_ocr,
    start_vlm,
)

TARGET_FOLDER = Path("train_data/synthetic/images")
OUTPUT_FOLDER = Path("output")


def main() -> None:
    ocr_set = OCRSet()
    vlm_set = VLMSet()
    start_ocr(TARGET_FOLDER, ocr_set)
    start_vlm(TARGET_FOLDER, ocr_set, vlm_set)
    data_offloader(vlm_set, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
