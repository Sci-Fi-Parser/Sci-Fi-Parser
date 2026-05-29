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

import copy
import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


ChartType = Literal[
    "bar_chart",
    "grouped_bar_chart",
    "stacked_bar_chart",
    "horizontal_bar_chart",
    "line_chart",
]


def _cat_key(x: str | float) -> str:
    """String key for category matching; integer-valued floats lose the .0."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def normalize_key(series: str, category: str) -> tuple[str, str]:
    """Whitespace- and case-insensitive form of a (series, category) match key."""
    return (series.strip().casefold(), category.strip().casefold())


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

    def series_map(self) -> dict[tuple[str, str], float]:
        """Flatten to ``{(series_name, category): value}`` for matching.

        Normalises integer-valued category keys (``2018.0`` -> ``"2018"``) so a
        VLM returning JSON numbers matches truth labels stored as strings.
        """
        return {(s.name, _cat_key(p.x)): float(p.y)
                for s in self.series for p in s.points}


def chartdata_schema() -> dict[str, Any]:
    """ChartData's JSON schema with all ``$ref``/``$defs`` inlined.

    llama.cpp's schema-to-grammar converter does not resolve ``$ref``, so the
    nested ``Series``/``Point`` definitions pydantic emits as references are
    left ungrammared and the model invents field names. Inlining the refs makes
    the whole structure constrainable.
    """
    schema = ChartData.model_json_schema()
    defs = schema.get("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: inline(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return inline(schema)


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

    def extract(self, image_path: Path) -> ChartData:
        ...
