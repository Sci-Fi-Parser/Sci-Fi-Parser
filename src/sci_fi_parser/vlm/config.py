"""VLM configuration objects.

Plain dataclasses the calling application builds and hands to `build_vlm`.
Two backends today: `api` (OpenAI-compatible HTTP, covers ollama / llama.cpp
server / vLLM / LM Studio / OpenAI / OpenRouter) and `hf_transformers` (HF
transformers library, in-process). Pick one with `backend = "..."`; the
matching sub-dataclass holds backend-specific knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_PROMPT = (
    "Extract the data from this chart as JSON.\n"
    "Rules:\n"
    "- chart_type: one of bar_chart, grouped_bar_chart, stacked_bar_chart, "
    "horizontal_bar_chart, line_chart.\n"
    "- Use the x-axis category labels EXACTLY as printed. Do not invent "
    "dates, years, or names.\n"
    "- Series naming: if there is a legend, use the legend labels. If there "
    "is no legend, use the y-axis title as the series name.\n"
    "- Read each y-value from the y-axis scale and the bar / marker height. "
    "If numeric labels are printed on the bars, prefer those.\n"
    "- Watch y-axis units: '200K' = 200000, '1.5M' = 1500000, "
    "'2.3B' = 2300000000. Return plain numbers, no suffixes.\n"
    "- Do not output series, categories, or values that do not appear on "
    "the chart."
)


@dataclass(slots=True)
class APIConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    schema_mode: str = "object"  # "object" (ollama-safe) | "schema" (strict)
    timeout_s: int = 600


@dataclass(slots=True)
class HFTransformersConfig:
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1024


@dataclass(slots=True)
class VLMConfig:
    backend: str = "api"
    model: str = "qwen2.5vl:7b"
    prompt: str = DEFAULT_PROMPT
    api: APIConfig = field(default_factory=APIConfig)
    hf_transformers: HFTransformersConfig = field(default_factory=HFTransformersConfig)
