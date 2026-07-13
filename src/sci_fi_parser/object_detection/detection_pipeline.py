"""Detection pipeline helpers for OCR and chart element extraction.

This module coordinates computer vision and OCR for images contained in an
`ImageSet`. It exposes a small set of utilities used by higher-level
processing: extracting OCR and CV results from images, matching detected
bars with OCR bounding boxes, formatting results for storage, and running
the whole pipeline over an `ImageSet` in batches.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from sci_fi_parser.object_detection.computer_vision.bars import detect_bars
from sci_fi_parser.object_detection.ocr import Ocr
from sci_fi_parser.schema import ImageSet

SUPPORTED_CHARTS = ["bar_chart"]


@dataclass
class OcrExtractionResult:
    """Container for OCR and CV extraction results for a single image.

    Attributes:
        image_name: The filename of the processed image.
        bar_candidates: Raw output from the bar detector.
        ocr_result: OCR output in a dict.
        matched: List of tuples linking each bar candidate to OCR bboxes that
            overlap it.
    """

    image_name: str
    bar_candidates: object
    ocr_result: object
    matched: object


def extract_ocr_data(paths: list[tuple[str, Path]]) -> list[OcrExtractionResult]:
    """Run CV and OCR on a list of images.

    Args:
        paths: A list of tuples `(image_id, image_path)` where `image_path` is
            a `Path` pointing to the image file to process.

    Returns:
        A list of `OcrExtractionResult` instances, in the same order as
        `paths`, containing detected bar candidates, OCR output, and the
        matched associations between them.
    """
    ocr = Ocr()
    results = []

    for image in paths:
        image_path = image[1]
        image_array = cv2.imread(str(image_path))

        bar_candidates = detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_result = ocr.run_ocr()

        matched = match_bars_and_ocr(bar_candidates, ocr_result)

        results.append(
            OcrExtractionResult(
                image_name=image_path.name,
                bar_candidates=bar_candidates,
                ocr_result=ocr_result,
                matched=matched,
            )
        )

    return results


def match_bars_and_ocr(bars: list, ocr_json: dict) -> list:
    """Associate detected bars with OCR bounding boxes.

    The function iterates over detected `bars` and finds OCR bboxes from
    `ocr_json` that spatially overlap each bar horizontally. OCR results
    with confidence lower than 0.90 are ignored.

    Args:
        bars: A list of bar candidate objects, each expected to have a
            `bbox` attribute with `x` and `right` attributes.
        ocr_json: OCR output dictionary containing at least the keys
            `'bbox'` (list of [x_max, y_max, x_min, y_min] boxes) and
            `'confidence'` (parallel list of confidences).

    Returns:
        A list of tuples `(bar, matching_bboxes)` where `matching_bboxes`
        is a list of OCR bboxes that overlap the bar horizontally.
    """
    linked = []
    for bar in bars:
        left = bar.bbox.x
        right = bar.bbox.right
        matching_ocr = []
        for i, bbox in enumerate(ocr_json["bbox"]):
            if ocr_json["confidence"][i] < 0.90:
                continue
            ocr_max_x, _, ocr_min_x, _ = bbox
            if (ocr_min_x <= left and ocr_max_x >= right) or (ocr_min_x >= left and ocr_max_x <= right):
                matching_ocr.append(bbox)
        linked.append((bar, matching_ocr))

    return linked


def format_ocr_output(result: OcrExtractionResult) -> str:
    """Create a compact string representation of OCR/CV results."""
    return f"{result.bar_candidates}{result.ocr_result}{result.matched}"


def start_ocr(image_set: ImageSet, batch_size=100) -> None:
    """Run OCR + CV pipeline over an `ImageSet` and persist results.

    The function processes images in `image_set` by chart type defined in
    `SUPPORTED_CHARTS`, in batches of `batch_size`. For each image it runs
    bar detection and OCR, formats a compact result string and saves both
    the compact string and the raw extraction as a dictionary using
    `ImageSet`'s storage methods.

    Args:
        image_set: An `ImageSet` instance.
        batch_size: Maximum number of images to process per chart type in
            one invocation. If `<= 0`, the function returns immediately.

    Side effects:
        Updates the provided `image_set` by adding OCR/CV results for
        processed images.
    """
    if batch_size <= 0:
        logging.info("Object detection batch size is 0 or less. Skipping stage.")
        return

    chart_ids = image_set.filter_by_type(SUPPORTED_CHARTS, batch_size)

    image_paths: list[tuple[str, Path]] = []
    for image_id in chart_ids:
        image_paths.append((image_id, image_set.get_image_path(image_id)))

    results = extract_ocr_data(image_paths)
    for image_id, result in zip(chart_ids, results, strict=True):
        image_set.add_ocrcv_result(image_id, format_ocr_output(result))
        image_set.add_ocrcv_raw(image_id, asdict(result))
