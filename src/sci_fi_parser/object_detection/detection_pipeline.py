"""Detection pipeline helpers for OCR and chart element extraction.

This module coordinates computer vision and OCR for images contained in an
`ImageSet`. It exposes a small set of utilities used by higher-level
processing: extracting OCR and CV results from images, matching detected
bars with OCR bounding boxes, formatting results for storage, and running
the whole pipeline over an `ImageSet` in batches.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import cv2

from sci_fi_parser.object_detection.computer_vision.bars import detect_bars
from sci_fi_parser.object_detection.models import ChartType, OcrCvStageResult, OcrOutput
from sci_fi_parser.object_detection.normalization import normalize_ocr_output
from sci_fi_parser.object_detection.ocr import Ocr
from sci_fi_parser.object_detection.pipeline import analyze_ocr_cv
from sci_fi_parser.schema import ImageSet

SUPPORTED_CHARTS: tuple[ChartType, ...] = ("bar_chart", "line_chart")


@dataclass
class OcrExtractionResult:
    """Container for OCR and CV extraction results for a single image.

    Attributes:
        image_name: The filename of the processed image.
        chart_type: Normalized classifier chart type.
        element_candidates: Raw chart-family-specific element proposals.
        ocr_result: Canonical typed OCR output.
        element_ocr_matches: Tuples linking bar candidates to OCR boxes that
            overlap it.
    """

    image_name: str
    chart_type: ChartType
    element_candidates: object
    ocr_result: OcrOutput
    element_ocr_matches: object
    stage_result: OcrCvStageResult | None = None

    def to_dict(self) -> dict:
        """Return a JSON-safe stage payload for ImageSet storage."""

        payload = asdict(self)
        payload["ocr_result"] = self.ocr_result.model_dump(mode="json")
        payload["stage_result"] = self.stage_result.model_dump(mode="json") if self.stage_result else None
        return payload


def extract_ocr_data(
    paths: list[tuple[str, Path, ChartType]], ocr: Ocr
) -> list[OcrExtractionResult]:
    """Run CV and OCR on a list of images.

    Args:
        paths: Tuples of image ID, path, and normalized chart type.
        ocr: Reusable OCR instance.

    Returns:
        A list of `OcrExtractionResult` instances, in the same order as
        `paths`, containing chart-family-specific candidates, OCR output, and
        any applicable element/OCR associations.
    """
    results = []

    for image in paths:
        _, image_path, chart_type = image
        image_array = cv2.imread(str(image_path))
        if image_array is None:
            raise ValueError(f"could not read image: {image_path}")

        element_candidates = detect_bars(image_array) if chart_type == "bar_chart" else []

        ocr.read_image(image_array)
        ocr_result = ocr.run_ocr()

        element_ocr_matches = match_bars_and_ocr(element_candidates, ocr_result)
        stage_result = analyze_ocr_cv(image_array, ocr_result, chart_type, element_candidates)

        results.append(
            OcrExtractionResult(
                image_name=image_path.name,
                chart_type=chart_type,
                element_candidates=element_candidates,
                ocr_result=ocr_result,
                element_ocr_matches=element_ocr_matches,
                stage_result=stage_result,
            )
        )

    return results


def match_bars_and_ocr(bars: list, ocr_result: OcrOutput | dict) -> list:
    """Associate detected bars with OCR bounding boxes.

    The function iterates over detected `bars` and finds OCR bboxes from
    `ocr_json` that spatially overlap each bar horizontally. OCR results
    with confidence lower than 0.90 are ignored.

    Args:
        bars: A list of bar candidate objects, each expected to have a
            `bbox` attribute with `x` and `right` attributes.
        ocr_result: Canonical OCR output. Legacy parallel arrays are normalized
            before matching for compatibility with persisted inputs.

    Returns:
        A list of tuples `(bar, matching_bboxes)` where `matching_bboxes`
        is a list of OCR bboxes that overlap the bar horizontally.
    """
    if isinstance(ocr_result, dict):
        ocr_result = normalize_ocr_output(
            ocr_result.get("labels", [""] * len(ocr_result.get("bbox", []))),
            ocr_result.get("confidence", []),
            ocr_result.get("bbox", []),
        )

    linked = []
    for bar in bars:
        left = bar.bbox.x
        right = bar.bbox.right
        matching_ocr = []
        for token in ocr_result.tokens:
            if token.confidence < 0.90:
                continue
            bbox = token.box
            if (bbox.left <= left and bbox.right >= right) or (bbox.left >= left and bbox.right <= right):
                matching_ocr.append(bbox.model_dump(mode="json"))
        linked.append((bar, matching_ocr))

    return linked


def format_ocr_output(result: OcrExtractionResult) -> str:
    """Create concise, structured evidence for the VLM without claiming final data."""

    if result.stage_result is None:
        return ""
    stage = result.stage_result
    assignments = {assignment.token_id: assignment.role for assignment in stage.roles.assignments}
    labels = [
        {
            "text": token.original_text,
            "role": assignments[token.token_id],
            "confidence": round(token.confidence, 3),
        }
        for token in stage.ocr.tokens
        if assignments[token.token_id] != "other"
    ]
    calibration = None
    if stage.calibration.succeeded:
        calibration = {
            "slope": stage.calibration.slope,
            "intercept": stage.calibration.intercept,
            "visible_labeled_range": stage.calibration.visible_labeled_range,
            "confidence": round(stage.calibration.confidence, 3),
        }
    context = {
        "scope": "OCR/CV evidence only; VLM remains responsible for final ChartData",
        "chart_type": stage.chart_type,
        "initial_element_counts": {
            kind: sum(element.kind == kind for element in stage.initial_elements)
            for kind in {element.kind for element in stage.initial_elements}
        },
        "axis_labels": labels,
        "axis_role_confidence": round(stage.roles.confidence, 3),
        "axis_abstention": stage.roles.abstention_reason,
        "x_axis_abstention": stage.roles.x_abstention_reason,
        "y_axis_abstention": stage.roles.y_abstention_reason,
        "y_calibration": calibration,
        "calibration_failure": stage.calibration.failure_reason,
    }
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def start_ocr(image_set: ImageSet, ocr: Ocr) -> None:
    """Run OCR + CV pipeline over an `ImageSet` and persist results.

    The function processes images in `image_set` by chart type defined in
    `SUPPORTED_CHARTS`. For each image it runs applicable chart-family CV and
    OCR, formats a compact result string, and saves both
    the compact string and the raw extraction as a dictionary using
    `ImageSet`'s storage methods.

    Args:
        image_set: An `ImageSet` instance.
        ocr: Reusable OCR instance.

    Side effects:
        Updates the provided `image_set` by adding OCR/CV results for
        processed images.
    """
    chart_ids = image_set.filter_by_type(list(SUPPORTED_CHARTS))
    image_paths = [
        (
            image_id,
            image_set.get_image_path(image_id),
            cast(ChartType, image_set.get_classification_result(image_id)),
        )
        for image_id in chart_ids
    ]
    results = extract_ocr_data(image_paths, ocr)
    for image_id, result in zip(chart_ids, results, strict=True):
        image_set.add_ocrcv_result(image_id, format_ocr_output(result))
        image_set.add_ocrcv_raw(image_id, result.to_dict())
