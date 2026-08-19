from types import SimpleNamespace

from sci_fi_parser.benchmark.ocr_cv import _tick_recall
from sci_fi_parser.object_detection.normalization import normalize_ocr_output


def test_tick_recall_requires_matching_text_and_location():
    ocr = normalize_ocr_output(
        ["100", "wrong", "50"],
        [0.99, 0.99, 0.99],
        [
            [0, 0, 20, 10],
            [0, 20, 20, 30],
            [80, 80, 100, 90],
        ],
    )
    result = SimpleNamespace(ocr=ocr)
    truth = [
        {"text": "100", "role": "y_tick", "bbox_px": [0, 0, 20, 10]},
        {"text": "50", "role": "y_tick", "bbox_px": [0, 20, 20, 30]},
    ]

    assert _tick_recall(result, truth) == (1, 2)
