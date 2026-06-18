"""Typed benchmark truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sci_fi_parser.vlm.vlm_schema import Series


@dataclass(slots=True)
class ChartTruth:
    chart_type: str | None
    series: list[Series]
    value_range: tuple[float, float]
    geometry: object | None = None


def truth_to_json(truth: ChartTruth, metadata: dict | None = None) -> dict:
    return {
        "chart_type": truth.chart_type,
        "series": [s.model_dump(mode="json") for s in truth.series],
        "value_range": [float(truth.value_range[0]), float(truth.value_range[1])],
        "geometry": truth.geometry,
        "metadata": metadata or {},
    }


def truth_from_json(raw: dict) -> tuple[ChartTruth, dict]:
    lo, hi = raw["value_range"]
    return (
        ChartTruth(
            chart_type=raw.get("chart_type"),
            series=[Series.model_validate(s) for s in raw["series"]],
            value_range=(float(lo), float(hi)),
            geometry=raw.get("geometry"),
        ),
        raw.get("metadata", {}),
    )


def load_synthetic_truth(data_dir: Path) -> tuple[dict[str, ChartTruth], dict[str, dict]]:
    truth_by_image: dict[str, ChartTruth] = {}
    metadata_by_image: dict[str, dict] = {}
    with (data_dir / "truth.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            truth, metadata = truth_from_json(rec)
            truth_by_image[rec["image"]] = truth
            metadata_by_image[rec["image"]] = metadata
    return truth_by_image, metadata_by_image
