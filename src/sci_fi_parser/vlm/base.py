"""The VLM call interface — what `build_vlm` returns and the pipeline calls.

A structural protocol, not a base class: backends satisfy it by having the
right shape, not by inheriting. Lets the type checker verify both concrete
backends (and any future one) without forcing a class hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from sci_fi_parser.schema import ChartData


@runtime_checkable
class VLMBackend(Protocol):
    name: str

    def extract(self, image_path: Path) -> ChartData:
        ...
