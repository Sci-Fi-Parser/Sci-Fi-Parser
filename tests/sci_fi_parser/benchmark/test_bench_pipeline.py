from __future__ import annotations

import json

from PIL import Image

from sci_fi_parser.benchmark.bench_pipeline import run_benchmark
from sci_fi_parser.vlm.vlm_config import VLMProfile


def _write_dataset(root):
    image_dir = root / "images"
    image_dir.mkdir()
    records = [
        {"image": "bars.png", "chart_type": "vertical_bar", "preset": "simple"},
        {"image": "pie.png", "chart_type": "none", "preset": "other"},
    ]
    with (root / "truth.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            Image.new("RGB", (16, 16), "white").save(image_dir / record["image"])
            truth = {
                "image": record["image"],
                "chart_type": record["chart_type"],
                "series": [
                    {"name": "Revenue", "points": [{"x": "2018", "y": 100.0}, {"x": "2019", "y": 150.0}]}
                ],
                "data_range": [0.0, 200.0],
                "geometry": None,
                "metadata": {"preset": record["preset"]},
            }
            fh.write(json.dumps(truth) + "\n")


def test_run_benchmark_scores_vlm_output_and_writes_reports(mocker, tmp_path):
    _write_dataset(tmp_path)
    prediction = {
        "chart_type": "vertical_bar",
        "log_scale": False,
        "series": [{"name": "sales", "points": [{"x": 2018.0, "y": 110.0}, {"x": "2020", "y": 1.0}]}],
    }
    extract = mocker.patch(
        "sci_fi_parser.vlm.vlm.ChatCompletionsVLM.extract",
        return_value=(prediction, {}),
    )
    out = tmp_path / "report"

    payload = run_benchmark(tmp_path, out, profile=VLMProfile(model="test-model"), classify=False, ocr=False)

    # Only the chart whose true type maps to a supported class reaches the VLM.
    assert extract.call_count == 1
    assert extract.call_args.args[0].name == "bars.png"

    summary = payload["summary"]
    assert summary["charts"] == 2
    assert summary["no_prediction"] == 1
    assert summary["points"] == 4
    assert summary["recall"] == 0.25
    assert summary["precision"] == 0.5
    assert summary["mean_pct"] == 5.0
    assert set(payload["breakdowns"]["preset"]) == {"simple", "other"}
    assert payload["run"]["vlm"]["model"] == "test-model"
    assert payload["run"]["stages"] == {"classify": False, "ocr": False}

    assert json.loads((out / "results.json").read_text(encoding="utf-8")) == payload
    assert "test-model" in (out / "report.html").read_text(encoding="utf-8")


def test_run_benchmark_limit_keeps_first_images_by_name(mocker, tmp_path):
    _write_dataset(tmp_path)
    mocker.patch("sci_fi_parser.vlm.vlm.ChatCompletionsVLM.extract", side_effect=RuntimeError("down"))

    payload = run_benchmark(
        tmp_path, tmp_path / "report", profile=VLMProfile(), limit=1, classify=False, ocr=False
    )

    assert [chart["image"] for chart in payload["charts"]] == ["bars.png"]
    assert payload["summary"]["no_prediction"] == 1
    assert payload["summary"]["recall"] == 0.0
