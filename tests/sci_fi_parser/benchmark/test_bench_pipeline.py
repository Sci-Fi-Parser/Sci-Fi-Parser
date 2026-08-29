from __future__ import annotations

import json

import pytest
from PIL import Image

from sci_fi_parser.benchmark import bench_pipeline
from sci_fi_parser.benchmark.bench_pipeline import PipelineInputs, run_pipeline
from sci_fi_parser.benchmark.truth import ChartTruth
from sci_fi_parser.schema import ImageSet


def _write_tiny_dataset(root):
    image_dir = root / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "chart.png")
    (root / "truth.jsonl").write_text(
        json.dumps(
            {
                "image": "chart.png",
                "chart_type": "vertical_bar",
                "series": [
                    {
                        "name": "Revenue",
                        "points": [{"x": "2018", "y": 100.0}],
                    }
                ],
                "data_range": [50.0, 150.0],
                "geometry": None,
                "metadata": {"preset": "tiny", "density": 1, "labels_on": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_pipeline_runs_noisy_oracle_on_truth_json(tmp_path):
    _write_tiny_dataset(tmp_path)
    out = tmp_path / "report"

    agg = run_pipeline(
        data=tmp_path,
        out=out,
        extractor_name="noisy-oracle",
        seed=0,
        print_summary=False,
    )

    assert agg["n_charts"] == 1
    assert agg["n_bars_true"] == 1
    assert (out / "results.json").is_file()
    assert (out / "report.html").is_file()

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["per_chart"][0]["meta"]["preset"] == "tiny"


@pytest.mark.parametrize(
    ("ocr_mode", "expected_modes"),
    [
        ("both", ("oracle", "detected")),
        ("detected", ("detected",)),
    ],
)
def test_ocr_cv_stage_expands_requested_modes(mocker, tmp_path, ocr_mode, expected_modes):
    run_benchmark = mocker.patch(
        "sci_fi_parser.benchmark.ocr_cv.run_benchmark",
        return_value={"summary_by_mode": {}},
    )
    out = tmp_path / "report"

    result = bench_pipeline.run_ocr_cv_stage(
        data=tmp_path,
        out=out,
        dataset="benetech",
        ocr_mode=ocr_mode,
        limit=3,
    )

    assert result == {"summary_by_mode": {}}
    run_benchmark.assert_called_once_with(
        tmp_path,
        out,
        dataset="benetech",
        modes=expected_modes,
        limit=3,
    )


def test_ocr_cv_context_stage_supplies_missing_vertical_bar_classification(mocker, tmp_path):
    image_set = ImageSet()
    image_set.add_extracted_image("vertical", tmp_path / "vertical.png")
    image_set.add_extracted_image("line", tmp_path / "line.png")
    image_set.add_classification_result("line", "line_chart")
    inputs = PipelineInputs(
        image_set=image_set,
        truth_by_image_id={
            "vertical": ChartTruth(chart_type="vertical_bar", series=[], data_range=(0.0, 1.0)),
            "line": ChartTruth(chart_type="line", series=[], data_range=(0.0, 1.0)),
        },
        image_dir=tmp_path,
    )
    ocr_instance = object()
    mocker.patch("sci_fi_parser.object_detection.ocr.Ocr", return_value=ocr_instance)
    start_ocr = mocker.patch("sci_fi_parser.object_detection.detection_pipeline.start_ocr")

    bench_pipeline.run_ocr_cv_context_stage(inputs)

    assert image_set.get_classification_result("vertical") == "bar_chart"
    assert image_set.get_classification_result("line") == "line_chart"
    scoped, passed_ocr = start_ocr.call_args.args
    assert set(scoped) == {"vertical", "line"}
    assert passed_ocr is ocr_instance


def test_ocr_cv_target_bypasses_vlm_pipeline(mocker, tmp_path):
    expected = {"summary_by_mode": {"oracle": {"samples": 2}}}
    run_ocr_cv_stage = mocker.patch.object(
        bench_pipeline,
        "run_ocr_cv_stage",
        return_value=expected,
    )
    load_inputs = mocker.patch.object(bench_pipeline, "load_inputs")
    out = tmp_path / "report"

    result = run_pipeline(
        data=tmp_path,
        out=out,
        dataset="synthetic",
        target="ocr-cv",
        ocr_mode="oracle",
        limit=2,
    )

    assert result == expected
    run_ocr_cv_stage.assert_called_once_with(
        data=tmp_path,
        out=out,
        dataset="synthetic",
        ocr_mode="oracle",
        limit=2,
    )
    load_inputs.assert_not_called()
