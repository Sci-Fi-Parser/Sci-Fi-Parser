"""
Public API for the Sci-Fi Parser package.

This module provides the main entry points for users of the package.
"""

from pathlib import Path

from sci_fi_parser.schema import ImageSet, PdfSet
from sci_fi_parser.object_detection.detection_pipeline import start_ocr
from sci_fi_parser.image_extraction.extraction_pipeline import start_extraction
from sci_fi_parser.storage.writer import save_image_set
from sci_fi_parser.vlm.vlm_pipeline import start_vlm


DEFAULT_VLM_CONFIG = Path("config/vlm.toml")
DEFAULT_EXTRACTED_IMAGE_DIR = Path("temp/extracted_images")


class ParseResult:
    """
    Represents the output of a parsing run.
    """

    def __init__(self, image_set: ImageSet, pdf_set: PdfSet):
        self._image_set = image_set
        self._pdf_set = pdf_set

    @property
    def images(self):
        """Return the extracted ImageSet."""
        return self._image_set

    @property
    def pdfs(self):
        """Return the PdfSet."""
        return self._pdf_set

    def summary(self) -> dict:
        """Return a simple summary of the parsing results."""
        return {
            "pdfs": len(self._pdf_set),
            "images": len(self._image_set),
        }

    def save(self, output_dir: str | Path):
        """
        Save the results (JSONL + Parquet) using the existing storage layer.
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
    run_ocr: bool = True,
    run_vlm: bool = True,
) -> ParseResult:
    """
    Parse every PDF inside a directory.
    """

    image_set = ImageSet()
    pdf_set = PdfSet()

    start_extraction(
        pdf_input_folder=Path(input_dir),
        image_set=image_set,
        pdf_set=pdf_set,
        extracted_image_folder=Path(extracted_image_dir),
    )

    if run_ocr:
        start_ocr(image_set)

    if run_vlm:
        start_vlm(image_set, Path(vlm_config))

    if output_dir is not None:
        save_image_set(
            image_set=image_set,
            output_dir=Path(output_dir),
        )

    return ParseResult(image_set, pdf_set)


def parse_pdf(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    extracted_image_dir: str | Path = DEFAULT_EXTRACTED_IMAGE_DIR,
    vlm_config: str | Path = DEFAULT_VLM_CONFIG,
    run_ocr: bool = True,
    run_vlm: bool = True,
) -> ParseResult:
    """
    Parse a single PDF.

    Currently this simply processes the directory containing the PDF.
    Once the extraction pipeline supports single-file processing,
    this implementation can be updated without changing the public API.
    """

    pdf_path = Path(pdf_path)

    return parse_folder(
        input_dir=pdf_path.parent,
        output_dir=output_dir,
        extracted_image_dir=extracted_image_dir,
        vlm_config=vlm_config,
        run_ocr=run_ocr,
        run_vlm=run_vlm,
    )