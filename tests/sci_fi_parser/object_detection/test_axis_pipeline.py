import json

import numpy as np
import pytest

from sci_fi_parser.object_detection.axis_analysis import analyze_ocr_cv
from sci_fi_parser.object_detection.detection_pipeline import OcrExtractionResult, format_ocr_output
from sci_fi_parser.object_detection.normalization import normalize_ocr_output


def test_line_chart_uses_chart_neutral_axis_evidence():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr = normalize_ocr_output(
        ["100", "50", "0", "A", "B", "Quarterly results"],
        [0.99] * 6,
        [
            [5, 15, 25, 25],
            [5, 45, 25, 55],
            [5, 75, 25, 85],
            [40, 88, 50, 98],
            [70, 88, 80, 98],
            [35, 2, 75, 12],
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
    assert calibration["visible_labeled_range"] == [0.0, 100.0]
    assert calibration["extrapolation_limits"]["value_at_pixel_y_min"] == pytest.approx(125.0)
    assert calibration["extrapolation_limits"]["value_at_pixel_y_max"] == pytest.approx(-25.0)
    assert "normally remain near" in context["calibration_guidance"]
    assert context["image_size"] == {"width": 100, "height": 100}
    assert len(context["ocr_tokens"]) == 6
    title = next(token for token in context["ocr_tokens"] if token["text"] == "Quarterly results")
    assert title["role"] == "other"
    assert title["bbox"] == {"left": 35.0, "top": 2.0, "right": 75.0, "bottom": 12.0}
