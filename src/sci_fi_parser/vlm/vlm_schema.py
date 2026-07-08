from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

ChartType = Literal[
    "vertical_bar", "grouped_bar", "stacked_bar", "horizontal_bar", "line", "scatter", "dot", "none"
]


class Point(BaseModel):
    """A data point consisting of label `x` and value `y`."""

    x: str | float
    y: float


class Series(BaseModel):
    """A named series of data points."""

    name: str = "series"
    points: list[Point]


class ChartData(BaseModel):
    chart_type: ChartType | None = None
    log_scale: bool | None = None
    series: list[Series] = []


@runtime_checkable
class Extractor(Protocol):
    """The single interface every extractor must satisfy.

    ``name`` shows up in benchmark reports; ``extract`` is the workhorse.
    Runtime-checkable so ``isinstance(x, Extractor)`` works in registries.
    """

    name: str

    def extract(self, image_path: Path, prompt_suffix: str = "") -> tuple[dict, dict]: ...
