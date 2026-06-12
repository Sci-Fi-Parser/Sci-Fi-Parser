"""Canonical chart-extraction contract — the one schema everything speaks.

Why this is its own module
--------------------------
The pipeline has several roles that all need to agree on what a "chart
extraction" looks like:

* VLM and CV+OCR extractors produce :class:`ChartData`.
* The benchmark (:mod:`sci_fi_parser.accuracy.benchmark`) scores it against
  ground truth.
* The pipeline entry point (:mod:`sci_fi_parser.main`) hands extractor output
  downstream to storage / analysis.

Putting the contract here — separate from any extractor or scorer — lets each
of those depend on the schema *only*, not on each other. The cv subpackage
shouldn't have to import the benchmark module just to know what a ``Point`` is.

Kept deliberately light: pydantic + stdlib only, no matplotlib/opencv/ollama.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


ChartType = Literal[
    "bar_chart",
    "grouped_bar_chart",
    "stacked_bar_chart",
    "horizontal_bar_chart",
    "line_chart",
]


class Point(BaseModel):
    """One data point: a category label (bars) or numeric x, plus the value."""

    x: str | float
    y: float


class Series(BaseModel):
    """A named series of points (one line, one bar group, etc.)."""

    name: str = "series"
    points: list[Point]


class ChartData(BaseModel):
    """The canonical extraction output. Every extractor returns this shape.

    ``chart_type`` and ``confidence`` are *required but nullable*: a VLM
    constrained by ``format=ChartData.model_json_schema()`` will always emit
    both fields, but is allowed to report ``null`` when it can't determine the
    chart type or its own confidence. Without ``required``-ness the model
    silently omits them.
    """

    chart_type: ChartType | None
    series: list[Series]
    confidence: float | None


def parse_chartdata(raw: str | dict) -> ChartData:
    """Validate (and lightly repair) extractor output into :class:`ChartData`.

    Real VLM output is messy — code fences, prose, trailing commas. This is the
    repair half of the standardization layer; constrained decoding is the other.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if "```" in text:                       # strip ```json ... ``` fences
            text = text.split("```")[1]
            text = text[4:] if text.lstrip().lower().startswith("json") else text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        raw = json.loads(text)
    return ChartData.model_validate(raw)


@runtime_checkable
class Extractor(Protocol):
    """The single interface every extractor must satisfy.

    ``name`` shows up in benchmark reports; ``extract`` is the workhorse.
    Runtime-checkable so ``isinstance(x, Extractor)`` works in registries.
    """

    name: str

    def extract(self, image_path: Path, prompt_suffix: str = "") -> tuple[dict, dict]:
        ...
