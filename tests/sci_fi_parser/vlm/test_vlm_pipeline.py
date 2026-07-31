from __future__ import annotations

import logging
from pathlib import Path

from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm_pipeline import start_vlm


def _image_record(image_path: Path, chart_type: str, ocr_result: str = "") -> dict:
    record = ImageSet._empty_record(image_path)
    record["classification"]["result"] = chart_type
    record["ocrcv"]["result"] = ocr_result
    return record


class _FakeVLM:
    def __init__(self, failing_paths: set[Path] | None = None):
        self.failing_paths = failing_paths or set()
        self.calls: list[tuple[Path, str]] = []

    def extract(self, image_path: Path, prompt_suffix: str = "") -> tuple[dict, dict]:
        self.calls.append((image_path, prompt_suffix))
        if image_path in self.failing_paths:
            raise RuntimeError("extraction failed")
        return ({"chart_type": "vertical_bar"}, {"choices": []})


def test_start_vlm_extracts_only_supported_charts(tmp_path):
    image_set = ImageSet()
    bar_path = tmp_path / "bar.png"
    image_set.add("bar-1", _image_record(bar_path, "bar_chart", "ocr text"))
    image_set.add("line-1", _image_record(tmp_path / "line.png", "line_chart"))

    vlm = _FakeVLM()
    start_vlm(image_set, vlm)

    assert vlm.calls == [(bar_path, "ocr text")]
    assert image_set.get_vlm_result("bar-1") == {"chart_type": "vertical_bar"}
    assert image_set.get_vlm_raw("bar-1") == {"choices": []}
    assert image_set.get("line-1")["vlm"]["result"] == {}


def test_start_vlm_continues_after_extraction_failure(tmp_path, caplog):
    image_set = ImageSet()
    failing_path = tmp_path / "broken.png"
    working_path = tmp_path / "working.png"
    image_set.add("broken-1", _image_record(failing_path, "bar_chart"))
    image_set.add("working-1", _image_record(working_path, "bar_chart"))

    with caplog.at_level(logging.WARNING):
        start_vlm(image_set, _FakeVLM(failing_paths={failing_path}))

    assert "broken-1" in caplog.text
    assert image_set.get("broken-1")["vlm"]["result"] == {}
    assert image_set.get_vlm_result("working-1") == {"chart_type": "vertical_bar"}
