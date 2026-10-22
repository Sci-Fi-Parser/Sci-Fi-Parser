import pytest

from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate, BoundingBox
from sci_fi_parser.object_detection.computer_vision.calibration import calibrate_y_axis
from sci_fi_parser.object_detection.computer_vision.roles import infer_axis_roles
from sci_fi_parser.object_detection.models import AxisCandidate, RoleInferenceResult
from sci_fi_parser.object_detection.normalization import normalize_ocr_output


def _bars():
    return [
        BarCandidate(BoundingBox(75, 90, 30, 110)),
        BarCandidate(BoundingBox(165, 65, 30, 135)),
        BarCandidate(BoundingBox(255, 115, 30, 85)),
    ]


def _chart_ocr(*, x_labels=("A", "B", "C"), y_values=("100", "50", "0")):
    labels = [*y_values, *x_labels, "Revenue"]
    boxes = [
        [15, 34, 48, 46],
        [20, 124, 48, 136],
        [35, 214, 48, 226],
        *[[x - 15, 214, x + 15, 226] for x in (90, 180, 270)[: len(x_labels)]],
        [120, 10, 210, 25],
    ]
    return normalize_ocr_output(labels, [0.99] * len(labels), boxes)


def _roles_for_all_tokens(ocr, *, values=None):
    parsed_values = values or {token.token_id: token.parsed_number for token in ocr.tokens}
    candidate = AxisCandidate(
        candidate_id="y-0",
        kind="y_column",
        token_ids=[token.token_id for token in ocr.tokens],
        score=0.8,
        score_components={},
        bounds={"left": 10, "top": 10, "right": 40, "bottom": 150},
        parsed_values=parsed_values,
    )
    return RoleInferenceResult(
        y_candidates=[candidate],
        selected_y_candidate_id="y-0",
        confidence=0.8,
    )


def test_joint_inference_assigns_axes_and_leaves_annotations_unselected():
    roles = infer_axis_roles(_chart_ocr(), (360, 280), _bars())
    assigned = {assignment.token_id: assignment.role for assignment in roles.assignments}

    assert roles.abstention_reason is None
    assert roles.selected_y_candidate_id is not None
    assert roles.selected_x_candidate_id is not None
    assert [assigned[f"ocr-{index}"] for index in range(3)] == ["y_tick"] * 3
    assert [assigned[f"ocr-{index}"] for index in range(3, 6)] == ["x_label"] * 3
    assert assigned["ocr-6"] == "other"
    assert any(pair.accepted and pair.shared_token_ids == ["ocr-2"] for pair in roles.pair_diagnostics)
    assert roles.winner_runner_up_margin is not None


def test_inference_without_tokens_reports_both_abstentions():
    roles = infer_axis_roles(normalize_ocr_output([], [], []), (200, 100))

    assert roles.confidence == 0.0
    assert roles.abstention_reason == "OCR returned no tokens"
    assert roles.x_abstention_reason == "OCR returned no tokens"
    assert roles.y_abstention_reason == "OCR returned no tokens"


def test_inference_without_numeric_column_keeps_tokens_as_other():
    ocr = normalize_ocr_output(
        ["A", "B"],
        [0.9, 0.9],
        [[50, 200, 70, 215], [150, 200, 170, 215]],
    )

    roles = infer_axis_roles(ocr, (240, 240))

    assert roles.selected_y_candidate_id is None
    assert roles.selected_x_candidate_id is None
    assert roles.x_abstention_reason == "no x/y pair passed endpoint ownership checks"
    assert all(assignment.role == "other" for assignment in roles.assignments)


def test_inference_without_element_anchors_uses_chart_geometry():
    roles = infer_axis_roles(_chart_ocr(), (360, 280))

    assert roles.selected_y_candidate_id is not None
    selected = next(
        candidate
        for candidate in roles.y_candidates
        if candidate.candidate_id == roles.selected_y_candidate_id
    )
    assert selected.score_components["elements_on_right"] > 0


def test_scientific_axis_multiplier_is_applied_to_tick_values():
    ocr = normalize_ocr_output(
        ["1.0", "0.5", "0.0", "1e6", "A", "B"],
        [1.0] * 6,
        [
            [20, 35, 48, 45],
            [20, 95, 48, 105],
            [20, 155, 48, 165],
            [50, 15, 80, 25],
            [100, 220, 120, 235],
            [200, 220, 220, 235],
        ],
    )

    roles = infer_axis_roles(ocr, (300, 260), _bars())
    selected = next(
        candidate
        for candidate in roles.y_candidates
        if candidate.candidate_id == roles.selected_y_candidate_id
    )

    assert selected.score_components["axis_multiplier"] == 1_000_000
    assert selected.parsed_values["ocr-0"].value == 1_000_000
    assert selected.parsed_values["ocr-0"].parsed_text == "1.0 * 1e+06"


def test_explicit_multiplier_joined_to_column_is_removed_from_ticks():
    ocr = normalize_ocr_output(
        ["0.0", "0.5", "1.0", "1e6", "A", "B"],
        [1.0] * 6,
        [
            [16, 155, 26, 165],
            [16, 95, 26, 105],
            [16, 35, 26, 45],
            [30, 15, 42, 25],
            [100, 220, 120, 235],
            [200, 220, 220, 235],
        ],
    )

    roles = infer_axis_roles(ocr, (300, 260), _bars())
    selected = next(
        candidate
        for candidate in roles.y_candidates
        if candidate.candidate_id == roles.selected_y_candidate_id
    )

    assert "ocr-3" not in selected.token_ids
    assert selected.score_components["axis_multiplier"] == 1_000_000


def test_large_tick_values_are_not_scaled_by_a_separate_multiplier():
    ocr = normalize_ocr_output(
        ["1000", "500", "0", "1e6", "A", "B"],
        [1.0] * 6,
        [
            [20, 35, 48, 45],
            [20, 95, 48, 105],
            [20, 155, 48, 165],
            [50, 15, 80, 25],
            [100, 220, 120, 235],
            [200, 220, 220, 235],
        ],
    )

    roles = infer_axis_roles(ocr, (300, 260), _bars())
    selected = next(
        candidate
        for candidate in roles.y_candidates
        if candidate.candidate_id == roles.selected_y_candidate_id
    )

    assert "axis_multiplier" not in selected.score_components
    assert selected.parsed_values["ocr-0"].value == 1000


def test_non_decreasing_and_duplicate_y_candidates_explain_rejection():
    non_decreasing = infer_axis_roles(
        _chart_ocr(y_values=("0", "50", "100")),
        (360, 280),
        _bars(),
    )
    duplicate = infer_axis_roles(
        _chart_ocr(y_values=("10", "10", "10")),
        (360, 280),
        _bars(),
    )

    assert any(
        "numeric values do not decrease down the image" in candidate.rejection_reasons
        for candidate in non_decreasing.y_candidates
    )
    assert any(
        "fewer than two distinct numeric values" in candidate.rejection_reasons
        for candidate in duplicate.y_candidates
    )


def test_pair_rejects_more_than_one_shared_token():
    ocr = normalize_ocr_output(
        ["100", "50", "A"],
        [1.0] * 3,
        [
            [20, 95, 48, 105],
            [20, 99, 48, 109],
            [100, 97, 130, 107],
        ],
    )

    roles = infer_axis_roles(ocr, (200, 200))

    assert any(
        pair.rejection_reason == "x/y candidates share more than one token" for pair in roles.pair_diagnostics
    )


def test_middle_y_tick_cannot_be_shared_with_an_x_label_row():
    ocr = normalize_ocr_output(
        ["100", "50", "0", "A", "B"],
        [1.0] * 5,
        [
            [15, 34, 48, 46],
            [20, 124, 48, 136],
            [35, 214, 48, 226],
            [100, 124, 130, 136],
            [200, 124, 230, 136],
        ],
    )

    roles = infer_axis_roles(ocr, (360, 280), _bars())

    assert roles.selected_x_candidate_id is None
    assert any(
        pair.shared_token_ids == ["ocr-1"]
        and pair.rejection_reason == "shared token is not the bottommost y tick"
        for pair in roles.pair_diagnostics
    )


def test_y_endpoint_alone_does_not_supply_x_axis_evidence():
    roles = infer_axis_roles(_chart_ocr(x_labels=()), (360, 280), _bars())

    assert roles.selected_y_candidate_id is not None
    assert roles.selected_x_candidate_id is None
    assert any(
        pair.rejection_reason == "shared y endpoint leaves no x-only evidence"
        for pair in roles.pair_diagnostics
    )


def test_x_evidence_must_extend_right_of_y_column():
    ocr = normalize_ocr_output(
        ["100", "50", "0", "left"],
        [1.0] * 4,
        [
            [15, 34, 48, 46],
            [20, 124, 48, 136],
            [35, 214, 48, 226],
            [0, 214, 25, 226],
        ],
    )

    roles = infer_axis_roles(ocr, (360, 280), _bars())

    assert any(
        pair.rejection_reason == "x-only evidence does not extend right of the y-label column"
        for pair in roles.pair_diagnostics
    )


def test_calibration_without_selected_candidate_preserves_role_failure():
    roles = RoleInferenceResult(confidence=0.0, abstention_reason="ambiguous labels")

    calibration = calibrate_y_axis(normalize_ocr_output([], [], []), roles, 200)

    assert not calibration.succeeded
    assert calibration.failure_reason == "ambiguous labels"


def test_calibration_requires_at_least_two_parsed_labels():
    ocr = normalize_ocr_output(
        ["100", "label"],
        [1.0, 1.0],
        [[10, 15, 40, 25], [10, 55, 40, 65]],
    )
    parsed_values = {"ocr-0": ocr.tokens[0].parsed_number}

    calibration = calibrate_y_axis(
        ocr,
        _roles_for_all_tokens(ocr, values=parsed_values),
        200,
    )

    assert calibration.failure_reason == "at least two numeric y labels are required"


def test_calibration_requires_distinct_positions_and_values():
    ocr = normalize_ocr_output(
        ["10", "10"],
        [1.0, 1.0],
        [[10, 15, 40, 25], [10, 55, 40, 65]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.failure_reason == "y labels do not contain two distinct positions and values"


def test_calibration_rejects_non_decreasing_values():
    ocr = normalize_ocr_output(
        ["0", "50", "100"],
        [1.0] * 3,
        [[10, 15, 40, 25], [10, 75, 40, 85], [10, 135, 40, 145]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.failure_reason == "all y-label pairs imply a non-negative slope"
    assert calibration.inlier_threshold_pixels == 7.5


def test_calibration_rejects_material_disagreement():
    ocr = normalize_ocr_output(
        ["100", "20", "90", "10"],
        [1.0] * 4,
        [[10, 15, 40, 25], [10, 55, 40, 65], [10, 95, 40, 105], [10, 135, 40, 145]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.failure_reason == "numeric labels disagree materially with a linear scale"


def test_calibration_rejects_three_high_confidence_inconsistent_labels():
    ocr = normalize_ocr_output(
        ["100", "10", "1"],
        [0.99] * 3,
        [[10, 15, 40, 25], [10, 75, 40, 85], [10, 135, 40, 145]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.failure_reason == "three high-confidence labels do not agree on a linear scale"


def test_calibration_reports_fit_diagnostics_and_extrapolation_limits():
    ocr = normalize_ocr_output(
        ["100", "50", "0"],
        [0.99] * 3,
        [[10, 15, 40, 25], [10, 75, 40, 85], [10, 135, 40, 145]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.succeeded
    assert calibration.slope == pytest.approx(-50 / 60)
    assert calibration.visible_labeled_range == (0.0, 100.0)
    assert calibration.residual_pixel_mae == pytest.approx(0.0, abs=1e-12)
    assert all(item.fit_available for item in calibration.leave_one_out)
    assert calibration.extrapolation_limits is not None
    assert calibration.extrapolation_limits.pixel_y_min == 0.0
    assert calibration.extrapolation_limits.pixel_y_max == 170.0


def test_two_label_calibration_marks_leave_one_out_fits_unavailable():
    ocr = normalize_ocr_output(
        ["100", "0"],
        [1.0, 1.0],
        [[10, 15, 40, 25], [10, 135, 40, 145]],
    )

    calibration = calibrate_y_axis(ocr, _roles_for_all_tokens(ocr), 200)

    assert calibration.succeeded
    assert [item.fit_available for item in calibration.leave_one_out] == [False, False]
