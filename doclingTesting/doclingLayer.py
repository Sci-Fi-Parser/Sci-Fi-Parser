"""
doclingLayer.py
---------------
Wraps Docling to extract and classify figures from PDF documents.

Accepts a single file or a folder of files.
Returns a list of ClassifiedFigure objects — one per extracted figure.

CHART_LABELS defines which Docling classifier labels count as charts.
Edit this set to add or remove chart types without touching any other code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — edit these to change behaviour
# ---------------------------------------------------------------------------

# Docling figure classifier labels that count as "a chart worth parsing".
# All other labels (logo, photograph, stamp, etc.) are treated as non-charts.
CHART_LABELS: set[str] = {
    "bar_chart",
    "line_chart",
    "pie_chart",
    "scatter_plot",
    "flow_chart",
    "box_plot",
    "table",
}

# Supported input file extensions.
SUPPORTED_EXTENSIONS: set[str] = {".pdf"}

# Image export scale (2.0 = double resolution, better quality at cost of memory).
IMAGE_SCALE: float = 2.0


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedFigure:
    """A single figure extracted and classified from a document."""

    source_file: str        # absolute path to the source document
    page: int               # page number (1-indexed)
    figure_index: int       # index of this figure on the page
    label: str              # Docling classifier label e.g. "bar_chart"
    confidence: float       # classifier confidence 0.0–1.0
    is_chart: bool          # True if label is in CHART_LABELS
    image_path: str | None  # path to saved image file, or None if not saved


@dataclass
class ExtractionSummary:
    """Summary of processing one document."""

    source_file: str
    total_figures: int
    charts_found: int
    figures: list[ClassifiedFigure] = field(default_factory=list)

    def charts_only(self) -> list[ClassifiedFigure]:
        """Return only figures classified as charts."""
        return [f for f in self.figures if f.is_chart]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_converter():
    """Build and return a Docling DocumentConverter with figure classification enabled."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions(
        do_picture_classification=True,  # enable DocumentFigureClassifier
        generate_picture_images=True,    # needed to access image data per figure
        images_scale=IMAGE_SCALE,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _get_label_and_confidence(picture) -> tuple[str, float]:
    """Extract the top label and confidence from a Docling picture element."""
    try:
        if picture.annotations:
            top = picture.annotations[0].predicted_classes[0]
            return top.class_name, float(top.confidence)
    except (AttributeError, IndexError):
        pass
    return "unknown", 0.0


def _save_image(picture, doc, output_dir: Path, filename: str) -> str | None:
    """Save a picture's image to disk. Returns the saved path or None on failure."""
    try:
        img = picture.get_image(doc)
        if img is None:
            return None
        save_path = output_dir / filename
        img.save(save_path)
        return str(save_path)
    except Exception as exc:
        logger.warning("Could not save image %s: %s", filename, exc)
        return None


def _process_file(
    file_path: Path,
    converter,
    output_dir: Path | None,
    save_images: bool,
) -> ExtractionSummary:
    """Run Docling on one file and return an ExtractionSummary."""
    logger.info("Processing: %s", file_path)

    try:
        result = converter.convert(str(file_path))
    except Exception as exc:
        logger.error("Docling failed on %s: %s", file_path, exc)
        return ExtractionSummary(
            source_file=str(file_path),
            total_figures=0,
            charts_found=0,
        )

    doc = result.document
    figures: list[ClassifiedFigure] = []

    # Resolve output directory for images
    if save_images and output_dir is None:
        img_dir = file_path.parent / "chartparse_output" / file_path.stem
    elif save_images:
        img_dir = output_dir / file_path.stem
    else:
        img_dir = None

    if img_dir is not None:
        img_dir.mkdir(parents=True, exist_ok=True)

    # Group pictures by page for indexing
    page_counters: dict[int, int] = {}

    for picture in doc.pictures:
        page = getattr(picture.prov[0], "page_no", 0) if picture.prov else 0
        fig_index = page_counters.get(page, 0)
        page_counters[page] = fig_index + 1

        label, confidence = _get_label_and_confidence(picture)
        is_chart = label in CHART_LABELS

        image_path = None
        if save_images and img_dir is not None:
            filename = f"page_{page:03d}_fig_{fig_index:02d}_{label}.png"
            image_path = _save_image(picture, doc, img_dir, filename)

        fig = ClassifiedFigure(
            source_file=str(file_path.resolve()),
            page=page,
            figure_index=fig_index,
            label=label,
            confidence=confidence,
            is_chart=is_chart,
            image_path=image_path,
        )
        figures.append(fig)

        logger.debug(
            "  page=%d fig=%d label=%-25s confidence=%.2f is_chart=%s",
            page, fig_index, label, confidence, is_chart,
        )

    charts_found = sum(1 for f in figures if f.is_chart)
    logger.info(
        "  → %d figures found, %d classified as charts",
        len(figures), charts_found,
    )

    return ExtractionSummary(
        source_file=str(file_path.resolve()),
        total_figures=len(figures),
        charts_found=charts_found,
        figures=figures,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    path: str | Path,
    output_dir: str | Path | None = None,
    save_images: bool = True,
    recursive: bool = False,
) -> list[ExtractionSummary]:
    """
    Extract and classify figures from a file or folder of documents.

    Args:
        path:        Path to a single file or a directory.
        output_dir:  Where to save extracted images. Defaults to a
                     'chartparse_output/' folder next to the source file.
        save_images: Set False to skip saving images to disk.
        recursive:   If path is a folder, search subfolders too.

    Returns:
        List of ExtractionSummary — one per processed file.
    """
    path = Path(path)
    out = Path(output_dir) if output_dir else None

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect files to process
    if path.is_file():
        files = [path]
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            logger.warning("No supported files found in %s", path)
            return []
    else:
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    logger.info("Found %d file(s) to process", len(files))
    converter = _build_converter()

    return [
        _process_file(f, converter, out, save_images)
        for f in sorted(files)
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Classify figures in PDF documents using Docling."
    )
    parser.add_argument("path", help="PDF file or folder of PDFs")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory for saved images")
    parser.add_argument("--no-images", action="store_true", help="Skip saving images")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search subfolders")
    parser.add_argument("--charts-only", action="store_true", help="Print only chart figures")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    summaries = classify(
        path=args.path,
        output_dir=args.output_dir,
        save_images=not args.no_images,
        recursive=args.recursive,
    )

    # Print results as JSON
    output = []
    for summary in summaries:
        figures = summary.charts_only() if args.charts_only else summary.figures
        output.append({
            "source_file": summary.source_file,
            "total_figures": summary.total_figures,
            "charts_found": summary.charts_found,
            "figures": [
                {
                    "page": f.page,
                    "figure_index": f.figure_index,
                    "label": f.label,
                    "confidence": round(f.confidence, 4),
                    "is_chart": f.is_chart,
                    "image_path": f.image_path,
                }
                for f in figures
            ],
        })

    print(json.dumps(output, indent=2))
