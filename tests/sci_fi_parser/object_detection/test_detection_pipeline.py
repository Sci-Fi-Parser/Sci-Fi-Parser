from __future__ import annotations

from pathlib import Path

import numpy as np

from sci_fi_parser.object_detection import detection_pipeline
from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate, BoundingBox
from sci_fi_parser.object_detection.normalization import normalize_ocr_output
from sci_fi_parser.schema import ImageSet


def _bar(x: int, y: int, width: int, height: int) -> BarCandidate:
    return BarCandidate(bbox=BoundingBox(x=x, y=y, width=width, height=height))


def _empty_image_record(image_path: Path, chart_type: str = "bar_chart") -> dict:
    record = ImageSet._empty_record(image_path)
    record["classification"]["result"] = chart_type
    return record


def test_match_bars_and_ocr_filters_low_confidence_and_outside_boxes():
    bar = _bar(10, 5, 12, 30)
    ocr_json = {
        "confidence": [0.95, 0.91, 0.5],
        "bbox": [
            [25, 0, 8, 0],
            [18, 0, 12, 0],
            [20, 0, 10, 0],
        ],
    }

    assert detection_pipeline.match_bars_and_ocr([bar], ocr_json) == [
        (
            bar,
            [
                {"left": 8.0, "top": 0.0, "right": 25.0, "bottom": 0.0},
                {"left": 12.0, "top": 0.0, "right": 18.0, "bottom": 0.0},
            ],
        )
    ]


def test_extract_ocr_data_uses_detected_bars_and_ocr_output(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.png"
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    bar = _bar(4, 0, 6, 20)
    ocr_output = normalize_ocr_output(["Label"], [0.99], [[12, 0, 2, 0]])

    calls: list[object] = []

    class FakeOcr:
        def read_image(self, input_image):
            calls.append(input_image)

        def run_ocr(self):
            return ocr_output

    monkeypatch.setattr(detection_pipeline.cv2, "imread", lambda path: image_array)
    monkeypatch.setattr(detection_pipeline, "detect_bars", lambda image: [bar])

    results = detection_pipeline.extract_ocr_data(
        [("chart-1", image_path, "bar_chart")], FakeOcr()
    )

    assert calls == [image_array]
    assert len(results) == 1
    assert results[0].image_name == "chart.png"
    assert results[0].chart_type == "bar_chart"
    assert results[0].element_candidates == [bar]
    assert results[0].ocr_result == ocr_output
    assert results[0].element_ocr_matches == [
        (bar, [{"left": 2.0, "top": 0.0, "right": 12.0, "bottom": 0.0}])
    ]
    assert results[0].stage_result is not None
    assert results[0].to_dict()["stage_result"]["schema_version"] == "ocr-cv-stage-v2"


def test_extract_ocr_data_skips_bar_detection_for_line_charts(monkeypatch, tmp_path):
    image_path = tmp_path / "line.png"
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr_output = normalize_ocr_output([], [], [])

    class FakeOcr:
        def read_image(self, input_image):
            pass

        def run_ocr(self):
            return ocr_output

    monkeypatch.setattr(detection_pipeline.cv2, "imread", lambda path: image_array)
    monkeypatch.setattr(
        detection_pipeline,
        "detect_bars",
        lambda image: (_ for _ in ()).throw(AssertionError("bar detector called for line chart")),
    )
    result = detection_pipeline.extract_ocr_data(
        [("line-1", image_path, "line_chart")], FakeOcr()
    )[0]

    assert result.chart_type == "line_chart"
    assert result.element_candidates == []
    assert result.element_ocr_matches == []
    assert result.stage_result is not None
    assert result.stage_result.chart_type == "line_chart"
    assert result.stage_result.initial_elements == []


def test_start_ocr_writes_results_for_each_matching_image(monkeypatch, tmp_path):
    image_set = ImageSet()
    first_path = tmp_path / "chart-1.png"
    second_path = tmp_path / "chart-2.png"
    image_set.add("chart-1", _empty_image_record(first_path))
    image_set.add("chart-2", _empty_image_record(second_path))

    first_result = detection_pipeline.OcrExtractionResult(
        image_name="chart-1.png",
        chart_type="bar_chart",
        element_candidates=["bar-1"],
        ocr_result=normalize_ocr_output(["one"], [0.99], [[12, 0, 2, 0]]),
        element_ocr_matches=[("bar-1", [[12, 0, 2, 0]])],
    )
    second_result = detection_pipeline.OcrExtractionResult(
        image_name="chart-2.png",
        chart_type="bar_chart",
        element_candidates=["bar-2"],
        ocr_result=normalize_ocr_output(["two"], [0.98], [[22, 0, 14, 0]]),
        element_ocr_matches=[("bar-2", [[22, 0, 14, 0]])],
    )

    def fake_extract(image_paths, _):
        assert image_paths == [
            ("chart-1", first_path, "bar_chart"),
            ("chart-2", second_path, "bar_chart"),
        ]
        return [first_result, second_result]

    monkeypatch.setattr(detection_pipeline, "extract_ocr_data", fake_extract)

    calls: list[object] = []

    class FakeOcr:
        def read_image(self, input_image):
            calls.append(input_image)

        def run_ocr(self):
            return []

    detection_pipeline.start_ocr(image_set, FakeOcr())

    first_record = image_set.get("chart-1")
    second_record = image_set.get("chart-2")

    assert first_record["ocrcv"]["raw"] == first_result.to_dict()
    assert second_record["ocrcv"]["raw"] == second_result.to_dict()
    assert first_record["ocrcv"]["result"] == detection_pipeline.format_ocr_output(first_result)
    assert second_record["ocrcv"]["result"] == detection_pipeline.format_ocr_output(second_result)


def test_start_ocr_processes_bar_and_line_charts(monkeypatch, tmp_path):
    image_set = ImageSet()
    for index, chart_type in enumerate(("bar_chart", "line_chart", "line_chart")):
        image_set.add(
            f"chart-{index}",
            _empty_image_record(tmp_path / f"chart-{index}.png", chart_type),
        )

    calls = []

    def fake_extract(image_paths, ocr):
        calls.append((image_paths, ocr))
        return [
            detection_pipeline.OcrExtractionResult(
                image_name=path.name,
                chart_type=chart_type,
                element_candidates=[],
                ocr_result=normalize_ocr_output([], [], []),
                element_ocr_matches=[],
            )
            for _, path, chart_type in image_paths
        ]

    monkeypatch.setattr(detection_pipeline, "extract_ocr_data", fake_extract)

    ocr = object()
    detection_pipeline.start_ocr(image_set, ocr)

    assert [item[0] for item in calls[0][0]] == ["chart-0", "chart-1", "chart-2"]
    assert calls[0][1] is ocr
    assert image_set.get_ocrcv_raw("chart-2")["chart_type"] == "line_chart"


def test_image_set_stores_ocrcv_results(tmp_path):
    image_set = ImageSet()
    image_path = tmp_path / "chart.png"
    image_set.add("chart-1", _empty_image_record(image_path))

    image_set.add_ocrcv_result("chart-1", "ocr text")
    image_set.add_ocrcv_raw("chart-1", {"confidence": [1.0]})

    record = image_set.get("chart-1")
    assert record["ocrcv"]["result"] == "ocr text"
    assert record["ocrcv"]["raw"] == {"confidence": [1.0]}
    assert "chart-1" in repr(image_set)
