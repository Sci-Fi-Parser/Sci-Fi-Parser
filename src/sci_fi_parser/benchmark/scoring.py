"""Score extracted ``ChartData`` against benchmark truth.

Points are matched on (series name, x label), ignoring case and surrounding
whitespace. If both truth and prediction have exactly one series, the series
names are ignored: single-series charts rarely have a legend to read a name from.

Value error is ``|pred - true|`` as a percentage of the truth data range, so a
misread of the same pixel distance weighs the same on small and large values.
It only covers matched points, so always read it together with recall and
precision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sci_fi_parser.benchmark.dataset import Sample
from sci_fi_parser.vlm.vlm_schema import ChartData, ChartType


@dataclass(slots=True)
class PointScore:
    series: str
    x: str
    true: float
    pred: float | None


@dataclass(slots=True)
class ChartScore:
    """Per-chart result. ``pred`` values are ``None`` for missed points."""

    image_path: Path
    type_true: ChartType | None
    type_pred: ChartType | None
    points: list[PointScore]
    extra: int
    span: float
    metadata: dict[str, Any]

    @property
    def errors_pct(self) -> list[float]:
        return [abs(p.pred - p.true) / self.span * 100.0 for p in self.points if p.pred is not None]

    @property
    def matched(self) -> int:
        return sum(p.pred is not None for p in self.points)

    @property
    def missed(self) -> int:
        return len(self.points) - self.matched

    @property
    def mean_pct(self) -> float | None:
        errors = self.errors_pct
        return float(np.mean(errors)) if errors else None

    @property
    def max_pct(self) -> float | None:
        errors = self.errors_pct
        return max(errors) if errors else None


def _x_label(x: str | float) -> str:
    # VLMs often return years as JSON numbers: 2018.0 should match "2018".
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def _key(series: str, x: str) -> tuple[str, str]:
    return series.strip().casefold(), x.strip().casefold()


def score_chart(sample: Sample, pred: ChartData | None) -> ChartScore:
    """Match predicted points to truth points and score them.

    Args:
        sample: The ground-truth sample.
        pred: The extracted chart, or ``None`` if extraction failed.
    """

    pred_series = pred.series if pred else []
    rename = len(sample.series) == 1 and len(pred_series) == 1
    pred_values = {
        _key(sample.series[0].name if rename else series.name, _x_label(point.x)): point.y
        for series in pred_series
        for point in series.points
    }
    points = [
        PointScore(
            series.name,
            _x_label(point.x),
            point.y,
            pred_values.get(_key(series.name, _x_label(point.x))),
        )
        for series in sample.series
        for point in series.points
    ]
    lo, hi = sample.data_range
    matched = sum(p.pred is not None for p in points)
    return ChartScore(
        image_path=sample.image_path,
        type_true=sample.chart_type,
        type_pred=pred.chart_type if pred else None,
        points=points,
        extra=len(pred_values) - matched,
        span=abs(hi - lo) or 1.0,
        metadata=sample.metadata,
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize(scores: list[ChartScore]) -> dict[str, Any]:
    """Aggregate metrics over charts. Undefined metrics are ``None``."""

    errors = np.array([e for s in scores for e in s.errors_pct], dtype=float)
    n_true = sum(len(s.points) for s in scores)
    matched = sum(s.matched for s in scores)
    extra = sum(s.extra for s in scores)
    typed = [s for s in scores if s.type_true is not None]
    return {
        "charts": len(scores),
        "no_prediction": sum(s.type_pred is None for s in scores),
        "points": n_true,
        "mean_pct": float(errors.mean()) if errors.size else None,
        "median_pct": float(np.median(errors)) if errors.size else None,
        "p95_pct": float(np.percentile(errors, 95)) if errors.size else None,
        "within_5pct": float((errors <= 5).mean()) if errors.size else None,
        "recall": _ratio(matched, n_true),
        "precision": _ratio(matched, matched + extra),
        "missed": n_true - matched,
        "extra": extra,
        "type_accuracy": _ratio(sum(s.type_pred == s.type_true for s in typed), len(typed)),
    }


def breakdowns(scores: list[ChartScore], keys: tuple[str, ...]) -> dict[str, dict[str, dict[str, Any]]]:
    """``summarize`` per value of each metadata key present in any chart."""

    result = {}
    for key in keys:
        groups: dict[str, list[ChartScore]] = defaultdict(list)
        for score in scores:
            if key in score.metadata:
                groups[str(score.metadata[key])].append(score)
        if groups:
            result[key] = {value: summarize(group) for value, group in sorted(groups.items())}
    return result
