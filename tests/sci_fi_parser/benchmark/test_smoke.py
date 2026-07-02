from __future__ import annotations

import json

from PIL import Image


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
    from sci_fi_parser.accuracy.bench_pipeline import run_pipeline

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
