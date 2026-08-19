import numpy as np
import pytest

from sci_fi_parser.object_detection.normalization import (
    normalize_ocr_box,
    normalize_ocr_output,
    normalize_text,
    parse_numeric_token,
)


@pytest.mark.parametrize(
    ("text", "value", "percentage", "suffix"),
    [
        ("1,234.5", 1234.5, False, None),
        ("1.234,5", 1234.5, False, None),
        ("12,34", 12.34, False, None),
        ("1,234,567", 1_234_567.0, False, None),
        ("−2.5e2", -250.0, False, None),
        ("(42)", -42.0, False, None),
        ("3.2M", 3_200_000.0, False, "M"),
        ("10%", 10.0, True, None),
    ],
)
def test_parse_numeric_token_formats(text, value, percentage, suffix):
    parsed = parse_numeric_token(text)

    assert parsed is not None
    assert parsed.value == value
    assert parsed.is_percentage is percentage
    assert parsed.magnitude_suffix == suffix


@pytest.mark.parametrize("text", ["", "not a number", "1.2.3", "(12", "--5"])
def test_parse_numeric_token_rejects_invalid_input(text):
    assert parse_numeric_token(text) is None


def test_numeric_corrections_are_only_applied_when_requested():
    assert parse_numeric_token("O.5") is None

    corrected = parse_numeric_token("O.5", cautious_corrections=True)

    assert corrected is not None
    assert corrected.value == 0.5
    assert corrected.parsed_text == "0.5"
    assert corrected.correction_applied


def test_normalize_text_canonicalizes_unicode_and_whitespace():
    assert normalize_text("  −１２\tkg\n") == "-12 kg"


def test_normalize_box_accepts_reversed_rectangle_and_numpy_polygon():
    rectangle = normalize_ocr_box([40, 30, 10, 20])
    polygon = normalize_ocr_box(np.asarray([[4, 8], [10, 7], [11, 15], [3, 16]]))

    assert rectangle.model_dump() == {"left": 10.0, "top": 20.0, "right": 40.0, "bottom": 30.0}
    assert polygon.model_dump() == {"left": 3.0, "top": 7.0, "right": 11.0, "bottom": 16.0}


@pytest.mark.parametrize("box", [[], [[1]], [1, 2, [3, 4], 5]])
def test_normalize_box_rejects_malformed_polygon(box):
    with pytest.raises(ValueError, match="polygon is malformed"):
        normalize_ocr_box(box)


def test_normalize_output_preserves_text_parses_numbers_and_clamps_confidence():
    output = normalize_ocr_output(
        [" 10% ", "label"],
        [1.2, -0.1],
        [[40, 30, 10, 20], [0, 0, 5, 5]],
    )

    number, label = output.tokens
    assert number.token_id == "ocr-0"
    assert number.original_text == " 10% "
    assert number.normalized_text == "10%"
    assert number.confidence == 1.0
    assert number.parsed_number is not None
    assert number.parsed_number.value == 10.0
    assert label.confidence == 0.0
    assert label.parsed_number is None


def test_normalize_output_accepts_none_and_rejects_mismatched_arrays():
    assert normalize_ocr_output(None, None, None).tokens == []

    with pytest.raises(ValueError, match="mismatched labels, confidences, and boxes: 1, 0, 1"):
        normalize_ocr_output(["x"], [], [[0, 0, 1, 1]])
