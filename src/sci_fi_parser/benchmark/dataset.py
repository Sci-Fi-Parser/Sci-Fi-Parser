"""Load annotated chart datasets for benchmarking.

Both supported layouts keep chart images in ``<data>/images/``:

* ``synthetic``: ``<data>/truth.jsonl`` written by the generator in ``scripts/synthetic``.
* ``benetech``: ``<data>/annotations/<image stem>.json`` in the Benetech competition format.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sci_fi_parser.vlm.vlm_schema import ChartType, Point, Series

DatasetKind = Literal["synthetic", "benetech"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(slots=True)
class Sample:
    """One chart image with its ground truth.

    Attributes:
        image_path: Path to the chart image.
        chart_type: True chart type, if annotated.
        series: True data series.
        data_range: Value-axis range used to normalise value errors.
        geometry: Pixel geometry (bars, ticks, OCR tokens); synthetic data only.
        metadata: Free-form dataset metadata, e.g. the synthetic preset.
    """

    image_path: Path
    chart_type: ChartType | None
    series: list[Series]
    data_range: tuple[float, float]
    geometry: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_samples(data: Path, dataset: DatasetKind) -> list[Sample]:
    """Load every annotated sample whose image exists, sorted by image name."""

    samples = _load_synthetic(data) if dataset == "synthetic" else _load_benetech(data)
    return sorted(samples, key=lambda sample: sample.image_path.name)


def _load_synthetic(data: Path) -> list[Sample]:
    samples = []
    with (data / "truth.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            image_path = data / "images" / record["image"]
            if not image_path.is_file():
                continue
            lo, hi = record["data_range"]
            samples.append(
                Sample(
                    image_path=image_path,
                    chart_type=record.get("chart_type"),
                    series=[Series.model_validate(series) for series in record["series"]],
                    data_range=(float(lo), float(hi)),
                    geometry=record.get("geometry"),
                    metadata=record.get("metadata", {}),
                )
            )
    return samples


def _load_benetech(data: Path) -> list[Sample]:
    annotation_dir = data / "annotations"
    if not annotation_dir.is_dir():
        raise ValueError(f"Benetech data must contain annotations/: {data}")
    samples = []
    for annotation_path in annotation_dir.glob("*.json"):
        image_path = _find_image(data / "images", annotation_path.stem)
        if image_path is None:
            continue
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        points = [Point(x=item["x"], y=float(item["y"])) for item in record.get("data-series", [])]
        finite = [point for point in points if math.isfinite(point.y)]
        series = [Series(name="series", points=finite)]
        samples.append(
            Sample(
                image_path=image_path,
                chart_type=record.get("chart-type"),
                series=series,
                data_range=_derive_data_range(series),
                metadata={"source": "benetech", "skipped_nonfinite_y": len(points) - len(finite)},
            )
        )
    return samples


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        if (path := image_dir / f"{stem}{suffix}").is_file():
            return path
    return None


def _derive_data_range(series: list[Series]) -> tuple[float, float]:
    values = [point.y for item in series for point in item.points]
    if not values:
        return (0.0, 1.0)
    lo, hi = min(values), max(values)
    return (lo - 0.10 * lo, hi + 0.10 * hi)
