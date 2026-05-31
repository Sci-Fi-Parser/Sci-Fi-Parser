"""Run the OCR -> VLM -> offload pipeline on a folder of chart images.

Edit ``TARGET_FOLDER`` / ``OUTPUT_FOLDER`` below; no CLI flags. The work
lives in :mod:`sci_fi_parser.data_pipeline` (data containers + offloader),
:mod:`sci_fi_parser.cv.pipeline` (OCR stage), and
:mod:`sci_fi_parser.vlm.pipeline` (VLM stage).
"""

from pathlib import Path

from sci_fi_parser.data_pipeline import OCRSet, VLMSet
from sci_fi_parser.vlm.pipeline import start_vlm
from sci_fi_parser.cv.pipeline import start_ocr

TARGET_FOLDER = Path("train_data/synthetic/images")

OUTPUT_FOLDER = Path("output")

VLM_CONFIG = Path("config/vlm.toml")

def main() -> None:
    ocr_set = OCRSet()
    vlm_set = VLMSet()
    start_ocr(TARGET_FOLDER, ocr_set)
    start_vlm(TARGET_FOLDER, ocr_set, vlm_set, VLM_CONFIG)
   
   
if __name__ == "__main__":
    main()
