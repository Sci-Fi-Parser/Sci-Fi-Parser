"""VLM backends for the Sci-Fi-Parser pipeline.

Two interchangeable backends behind the `VLMBackend` protocol: `APIVLM`
(OpenAI-compatible HTTP) and `HFTransformersVLM` (HuggingFace transformers
library, in-process). Pick one via `VLMConfig.backend` and the `build_vlm`
factory.
"""

from __future__ import annotations

from sci_fi_parser.schema import ChartData, ChartType, Point, Series, parse_chartdata
from sci_fi_parser.vlm.backends.api import APIVLM
from sci_fi_parser.vlm.base import VLMBackend
from sci_fi_parser.vlm.config import APIConfig, HFTransformersConfig, VLMConfig


def build_vlm(config: VLMConfig) -> VLMBackend:
    if config.backend == "api":
        return APIVLM(config)
    if config.backend == "hf_transformers":
        from sci_fi_parser.vlm.backends.hf_transformers import HFTransformersVLM
        return HFTransformersVLM(config)
    raise ValueError(
        f"unknown backend {config.backend!r}; "
        f"expected 'api' or 'hf_transformers'")


__all__ = [
    "APIConfig",
    "APIVLM",
    "ChartData",
    "ChartType",
    "HFTransformersConfig",
    "Point",
    "Series",
    "VLMBackend",
    "VLMConfig",
    "build_vlm",
    "parse_chartdata",
]
