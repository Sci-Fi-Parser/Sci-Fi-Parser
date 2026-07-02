from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

from sci_fi_parser.object_detection import detection_pipeline
from sci_fi_parser.object_detection.computer_vision.bars import BarCandidate, BoundingBox
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

    assert detection_pipeline.match_bars_and_ocr([bar], ocr_json) == [(bar, [[25, 0, 8, 0], [18, 0, 12, 0]])]


def test_extract_ocr_data_uses_detected_bars_and_ocr_output(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.png"
    image_array = object()
    bar = _bar(4, 0, 6, 20)
    ocr_output = {
        "confidence": [0.99],
        "bbox": [[12, 0, 2, 0]],
        "labels": ["Label"],
    }

    calls: list[object] = []

    class FakeOcr:
        def read_image(self, input_image):
            calls.append(input_image)

        def run_ocr(self):
            return ocr_output

    monkeypatch.setattr(detection_pipeline.cv2, "imread", lambda path: image_array)
    monkeypatch.setattr(detection_pipeline, "detect_bars", lambda image: [bar])
    monkeypatch.setattr(detection_pipeline, "Ocr", FakeOcr)

    results = detection_pipeline.extract_ocr_data([("chart-1", image_path)])

    assert calls == [image_array]
    assert len(results) == 1
    assert results[0].image_name == "chart.png"
    assert results[0].bar_candidates == [bar]
    assert results[0].ocr_result == ocr_output
    assert results[0].matched == [(bar, [[12, 0, 2, 0]])]


def test_start_ocr_writes_results_for_each_matching_image(monkeypatch, tmp_path):
    image_set = ImageSet()
    first_path = tmp_path / "chart-1.png"
    second_path = tmp_path / "chart-2.png"
    image_set.add("chart-1", _empty_image_record(first_path))
    image_set.add("chart-2", _empty_image_record(second_path))

    first_result = detection_pipeline.OcrExtractionResult(
        image_name="chart-1.png",
        bar_candidates=["bar-1"],
        ocr_result={"confidence": [0.99], "bbox": [[12, 0, 2, 0]]},
        matched=[("bar-1", [[12, 0, 2, 0]])],
    )
    second_result = detection_pipeline.OcrExtractionResult(
        image_name="chart-2.png",
        bar_candidates=["bar-2"],
        ocr_result={"confidence": [0.98], "bbox": [[22, 0, 14, 0]]},
        matched=[("bar-2", [[22, 0, 14, 0]])],
    )

    def fake_extract(image_paths):
        assert image_paths == [("chart-1", first_path), ("chart-2", second_path)]
        return [first_result, second_result]

    monkeypatch.setattr(detection_pipeline, "extract_ocr_data", fake_extract)

    detection_pipeline.start_ocr(image_set, batch_size=2)

    first_record = image_set.get("chart-1")
    second_record = image_set.get("chart-2")

    assert first_record["ocrcv"]["raw"] == asdict(first_result)
    assert second_record["ocrcv"]["raw"] == asdict(second_result)
    assert first_record["ocrcv"]["result"] == detection_pipeline.format_ocr_output(first_result)
    assert second_record["ocrcv"]["result"] == detection_pipeline.format_ocr_output(second_result)


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
