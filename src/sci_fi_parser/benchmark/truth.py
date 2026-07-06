"""Typed benchmark truth."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sci_fi_parser.vlm.vlm_schema import Point, Series


@dataclass(slots=True)
class ChartTruth:
    chart_type: str | None
    series: list[Series]
    data_range: tuple[float, float]
    geometry: object | None = None


def derive_data_range(series: list[Series]) -> tuple[float, float]:
    values = [float(point.y) for item in series for point in item.points]
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    return (lo - 0.10 * lo, hi + 0.10 * hi)


def truth_from_json(raw: dict) -> tuple[ChartTruth, dict]:
    series = [Series.model_validate(s) for s in raw["series"]]
    lo, hi = raw["data_range"]
    return (
        ChartTruth(
            chart_type=raw.get("chart_type"),
            series=series,
            data_range=(float(lo), float(hi)),
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


def truth_from_benetech_annotation(raw: dict) -> tuple[ChartTruth, dict]:
    points: list[Point] = []
    skipped = 0
    for item in raw.get("data-series", []):
        y = float(item["y"])
        if not math.isfinite(y):
            skipped += 1
            continue
        points.append(Point(x=item["x"], y=y))

    series = [
        Series(
            name="series",
            points=points,
        )
    ]
    return (
        ChartTruth(
            chart_type=raw.get("chart-type"),
            series=series,
            data_range=derive_data_range(series),
            geometry=None,
        ),
        {
            "source": "benetech",
            "chart_type": raw.get("chart-type"),
            "skipped_nonfinite_y": skipped,
        },
    )


def load_benetech_truth(data_dir: Path) -> tuple[dict[str, ChartTruth], dict[str, dict]]:
    truth_by_stem: dict[str, ChartTruth] = {}
    metadata_by_stem: dict[str, dict] = {}
    annotation_dir = data_dir / "annotations"
    if not annotation_dir.is_dir():
        raise ValueError(f"Benetech data must contain annotations/: {data_dir}")
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        rec = json.loads(annotation_path.read_text(encoding="utf-8"))
        truth, metadata = truth_from_benetech_annotation(rec)
        truth_by_stem[annotation_path.stem] = truth
        metadata_by_stem[annotation_path.stem] = metadata
    return truth_by_stem, metadata_by_stem
