"""
Public API for the Sci-Fi Parser package.
"""

from pathlib import Path

from sci_fi_parser.classifier.classifier_pipeline import start_classification
from sci_fi_parser.image_extraction.extraction_pipeline import start_extraction
from sci_fi_parser.object_detection.detection_pipeline import start_ocr
from sci_fi_parser.schema import ImageSet, PdfSet
from sci_fi_parser.storage.writer import save_image_set
from sci_fi_parser.vlm.vlm_pipeline import start_vlm


DEFAULT_VLM_CONFIG = Path("config/vlm.toml")
DEFAULT_EXTRACTED_IMAGE_DIR = Path("temp/extracted_images")


class ParseResult:
    """
    Represents the result of one parsing run.
    """

    def __init__(self, image_set: ImageSet, pdf_set: PdfSet):
        self._image_set = image_set
        self._pdf_set = pdf_set

    @property
    def images(self) -> ImageSet:
        return self._image_set

    @property
    def pdfs(self) -> PdfSet:
        return self._pdf_set

    def summary(self) -> dict:
        return {
            "pdf_count": len(self._pdf_set),
            "image_count": len(self._image_set),
        }

    def save(self, output_dir: str | Path):
        """
        Save the parsed dataset (JSONL + Parquet).
        """
        save_image_set(
            image_set=self._image_set,
            output_dir=Path(output_dir),
        )


def parse_folder(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    extracted_image_dir: str | Path = DEFAULT_EXTRACTED_IMAGE_DIR,
    vlm_config: str | Path = DEFAULT_VLM_CONFIG,
    classify: bool = True,
    ocr: bool = True,
    vlm: bool = True,
) -> ParseResult:
    """
    Parse all PDFs inside a folder.

    Parameters
    ----------
    input_dir
        Folder containing PDF files.

    output_dir
        If provided, save JSONL and Parquet outputs.

    extracted_image_dir
        Temporary folder used during image extraction.

    vlm_config
        Path to the VLM configuration file.

    classify
        Run chart classification.

    ocr
        Run OCR.

    vlm
        Run VLM extraction.
    """

    image_set = ImageSet()
    pdf_set = PdfSet()

    start_extraction(
        Path(input_dir),
        image_set,
        pdf_set,
        Path(extracted_image_dir),
    )

    if classify:
        start_classification(image_set)

    if ocr:
        start_ocr(image_set)

    if vlm:
        start_vlm(
            image_set,
            Path(vlm_config),
        )

    if output_dir is not None:
        save_image_set(
            image_set=image_set,
            output_dir=Path(output_dir),
        )

    return ParseResult(image_set, pdf_set)