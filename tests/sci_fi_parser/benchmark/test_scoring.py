from __future__ import annotations

import json
from pathlib import Path

import pytest

from sci_fi_parser.benchmark.dataset import Sample, load_samples
from sci_fi_parser.benchmark.scoring import breakdowns, score_chart, summarize
from sci_fi_parser.vlm.vlm_schema import ChartData, Series


def _sample(series: list[dict], metadata: dict | None = None) -> Sample:
    return Sample(
        image_path=Path("chart.png"),
        chart_type="grouped_bar",
        series=[Series.model_validate(s) for s in series],
        data_range=(0.0, 200.0),
        metadata=metadata or {},
    )


def _pred(series: list[dict]) -> ChartData:
    return ChartData(chart_type="grouped_bar", log_scale=False, series=series)


def test_matching_ignores_case_and_whitespace_but_not_series_when_multi_series():
    sample = _sample(
        [
            {"name": "A", "points": [{"x": "Q1", "y": 10.0}]},
            {"name": "B", "points": [{"x": "Q1", "y": 20.0}]},
        ]
    )
    pred = _pred(
        [
            {"name": " a ", "points": [{"x": "q1", "y": 14.0}]},
            {"name": "C", "points": [{"x": "Q1", "y": 20.0}]},
        ]
    )

    score = score_chart(sample, pred)

    assert [p.pred for p in score.points] == [14.0, None]
    assert score.errors_pct == [pytest.approx(2.0)]
    assert (score.matched, score.missed, score.extra) == (1, 1, 1)


def test_single_series_names_are_ignored_and_integer_floats_match_labels():
    sample = _sample([{"name": "Revenue", "points": [{"x": "2018", "y": 100.0}]}])
    pred = _pred([{"name": "series", "points": [{"x": 2018.0, "y": 100.0}]}])

    score = score_chart(sample, pred)

    assert score.matched == 1
    assert score.errors_pct == [0.0]


def test_missing_prediction_counts_every_point_as_missed():
    sample = _sample([{"name": "A", "points": [{"x": "Q1", "y": 10.0}]}], {"preset": "simple"})

    score = score_chart(sample, None)
    summary = summarize([score])

    assert score.type_pred is None
    assert summary["no_prediction"] == 1
    assert summary["recall"] == 0.0
    assert summary["precision"] is None
    assert summary["mean_pct"] is None
    assert summary["type_accuracy"] == 0.0
    assert breakdowns([score], ("preset", "density")) == {"preset": {"simple": summary}}


def test_load_benetech_samples_skips_nonfinite_values_and_missing_images(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"")
    annotation = {
        "chart-type": "vertical_bar",
        "data-series": [{"x": "x1", "y": 2.0}, {"x": "x2", "y": "nan"}],
    }
    (tmp_path / "annotations" / "a.json").write_text(json.dumps(annotation), encoding="utf-8")
    (tmp_path / "annotations" / "no_image.json").write_text(json.dumps(annotation), encoding="utf-8")

    (sample,) = load_samples(tmp_path, "benetech")

    assert sample.image_path.name == "a.jpg"
    assert sample.chart_type == "vertical_bar"
    assert [(p.x, p.y) for p in sample.series[0].points] == [("x1", 2.0)]
    assert sample.metadata["skipped_nonfinite_y"] == 1
