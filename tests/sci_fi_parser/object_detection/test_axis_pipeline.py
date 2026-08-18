import json

import numpy as np

from sci_fi_parser.object_detection.axis_analysis import analyze_ocr_cv
from sci_fi_parser.object_detection.detection_pipeline import OcrExtractionResult, format_ocr_output
from sci_fi_parser.object_detection.normalization import normalize_ocr_output


def test_line_chart_uses_chart_neutral_axis_evidence():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr = normalize_ocr_output(
        ["100", "50", "0", "A", "B"],
        [0.99] * 5,
        [
            [5, 15, 25, 25],
            [5, 45, 25, 55],
            [5, 75, 25, 85],
            [40, 88, 50, 98],
            [70, 88, 80, 98],
        ],
    )

    result = analyze_ocr_cv(image, ocr, "line_chart")

    assert result.chart_type == "line_chart"
    assert result.initial_elements == []
    assert result.roles.selected_y_candidate_id is not None
    selected = next(
        candidate
        for candidate in result.roles.y_candidates
        if candidate.candidate_id == result.roles.selected_y_candidate_id
    )
    assert "elements_on_right" in selected.score_components
    assert result.calibration.succeeded
    assert result.calibration.slope is not None
    assert result.calibration.slope < 0

    context = json.loads(
        format_ocr_output(
            OcrExtractionResult(
                image_name="line.png",
                chart_type="line_chart",
                element_candidates=[],
                ocr_result=ocr,
                element_ocr_matches=[],
                stage_result=result,
            )
        )
    )
    calibration = context["y_calibration"]
    assert calibration["detected_tick_value_range"] == [0.0, 100.0]
    assert calibration["approximate_supported_value_range"] == [-25.0, 125.0]
    assert "normally remain near this range" in calibration["range_guidance"]
