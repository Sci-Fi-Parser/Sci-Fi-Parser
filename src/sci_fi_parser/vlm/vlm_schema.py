from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ChartType = Literal[
    "vertical_bar", "grouped_bar", "stacked_bar", "horizontal_bar", "line", "scatter", "dot", "none"
]


class Point(BaseModel):
    x: str | float
    y: float


class Series(BaseModel):
    name: str = "series"
    points: list[Point]


class ChartData(BaseModel):
    chart_type: ChartType = "none"
    log_scale: bool = False
    series: list[Series] = []
