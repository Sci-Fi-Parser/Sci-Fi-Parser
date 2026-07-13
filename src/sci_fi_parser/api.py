from pathlib import Path

import pandas as pd

from sci_fi_parser.classifier.classifier_pipeline import start_classification
from sci_fi_parser.image_extraction.extraction_pipeline import start_extraction
from sci_fi_parser.object_detection.detection_pipeline import start_ocr
from sci_fi_parser.schema import ImageSet, PdfSet
from sci_fi_parser.storage.writer import save_image_set
from sci_fi_parser.vlm.vlm_pipeline import start_vlm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXTRACTED_IMAGE_DIR = PROJECT_ROOT / "temp" / "extracted_images"
DEFAULT_VLM_CONFIG = PROJECT_ROOT / "src" / "sci_fi_parser" /"config" / "vlm.toml"


class ParseResult:
    """
    Represents the result of one parsing run.
    """

    def __init__(
        self,
        image_set: ImageSet,
        pdf_set: PdfSet,
        output_dir: str | Path | None = None,
    ):
        self._image_set = image_set
        self._pdf_set = pdf_set
        self._output_dir = Path(output_dir) if output_dir else None

    @property
    def images(self) -> ImageSet:
        return self._image_set

    @property
    def pdfs(self) -> PdfSet:
        return self._pdf_set

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    def summary(self) -> dict:
        return {
            "pdf_count": len(self._pdf_set),
            "image_count": len(self._image_set),
        }

    def save(self, output_dir: str | Path) -> None:
        """
        Save the parsed dataset as JSONL and Parquet.
        """

        output_dir = Path(output_dir)

        save_image_set(
            image_set=self._image_set,
            output_dir=output_dir,
        )

        self._output_dir = output_dir

    def _require_saved_output(self) -> Path:
        if self._output_dir is None:
            raise ValueError(
                "No output directory available. "
                "Pass output_dir to parse_folder() "
                "or call result.save(...)."
            )

        return self._output_dir

    def charts_dataframe(self) -> pd.DataFrame:
        """
        Load charts.parquet as a DataFrame.
        """

        output_dir = self._require_saved_output()

        return pd.read_parquet(
            output_dir / "tables" / "charts.parquet"
        )

    def series_dataframe(self) -> pd.DataFrame:
        """
        Load series.parquet as a DataFrame.
        """

        output_dir = self._require_saved_output()

        return pd.read_parquet(
            output_dir / "tables" / "series.parquet"
        )

    def points_dataframe(self) -> pd.DataFrame:
        """
        Load points.parquet as a DataFrame.
        """

        output_dir = self._require_saved_output()

        return pd.read_parquet(
            output_dir / "tables" / "points.parquet"
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

    return ParseResult(
        image_set=image_set,
        pdf_set=pdf_set,
        output_dir=output_dir,
    )