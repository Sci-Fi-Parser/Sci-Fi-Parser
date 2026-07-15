from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_PROMPT = (
    "Extract the data from this chart.\n"
    "Rules:\n"
    "- chart_type: one of vertical_bar, grouped_bar, stacked_bar, "
    "horizontal_bar, line, scatter, dot, none\n"
    "- log_scale: true if the chart is on the log scale, else false.\n"
    "- Use the x-axis category labels EXACTLY as printed. Do not invent "
    "dates, years, or names.\n"
    "- Series naming: if there is a legend, use the legend labels. "
    "If there is NO legend (single-series chart), use the y-axis title "
    "as the series name. Never leave the name blank.\n"
    "- Read each y-value from the y-axis scale and the bar / marker height. "
    "If numeric labels are printed on the bars, prefer those.\n"
    "- Watch y-axis units: '200K' = 200000, '1.5M' = 1500000, "
    "'2.3B' = 2300000000. Return plain numbers, no suffixes, no extra zeros.\n"
    "- Do not output series, categories, or values that do not appear on "
    'the chart. If the image is not a chart, set chart_type to "none" and series to an empty list.'
)


@dataclass(slots=True)
class VLMProfile:
    """Connection and prompt settings for a VLM endpoint.

    Defaults target a local Ollama instance.

    Attributes:
        model: Name of the model to request.
        prompt: Prompt sent with each extraction request.
        base_url: Base URL of the OpenAI-compatible endpoint.
        api_key: API key for the endpoint; a dummy value if none is required.
    """

    model: str = "qwen2.5vl:7b"
    prompt: str = DEFAULT_PROMPT
    base_url: str = "http://localhost:11434"
    api_key: str = "sk-no-key"


def load_profile(path: Path) -> VLMProfile:
    """Loads a `VLMProfile` from a .toml file.

    Keys missing from the file keep their `VLMProfile` defaults.

    Args:
        path: Path to the config .toml file.

    Returns:
        A `VLMProfile` with the loaded settings.

    Raises:
        ValueError: If the file contains keys that are not `VLMProfile` fields.
    """
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    valid = {f.name for f in fields(VLMProfile)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown key(s) in {path}: {sorted(unknown)}; valid: {sorted(valid)}")
    return VLMProfile(**raw)
