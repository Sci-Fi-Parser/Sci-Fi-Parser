"""Extraction contract — the shape every extractor returns.

Shared by every extractor (VLM today, CV+OCR later) and consumed by the
writer / analytics layer downstream. Kept deliberately small: only what the
pipeline needs to flatten a chart into JSONL rows (one `charts.jsonl` record
+ N `datapoints.jsonl` records). No confidence field — VLM self-reported
confidence is not calibrated.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel


ChartType = Literal[
    "bar_chart",
    "grouped_bar_chart",
    "stacked_bar_chart",
    "horizontal_bar_chart",
    "line_chart",
]


class Point(BaseModel):
    x: str | float
    y: float


class Series(BaseModel):
    name: str = "series"
    points: list[Point]


class ChartData(BaseModel):
    """The whole extraction: chart type plus one or more named series.

    `chart_type` is nullable-but-required so a schema-constrained VLM always
    emits the field even when it can't decide.
    """

    chart_type: ChartType | None
    series: list[Series]


def parse_chartdata(raw: str | dict) -> ChartData:
    """Validate (and lightly repair) raw VLM output into ChartData.

    Strips ```json fences and any prose before/after the JSON object before
    validating — the repair half of the standardization layer, since even
    schema-constrained models sometimes wrap their answer.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            text = text[4:] if text.lstrip().lower().startswith("json") else text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        raw = json.loads(text)
    return ChartData.model_validate(raw)
