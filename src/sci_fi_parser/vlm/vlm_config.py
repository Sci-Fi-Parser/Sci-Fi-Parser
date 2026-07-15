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
    """VLM profile with defaults.

    Args:
        model: requested model, for Ollama endpoint
        prompt: prompt for a VLM generation request
        base_url: base url of the VLM endpoint
        api_key: API key of the endpoint, if any
    """

    model: str = "qwen2.5vl:7b"
    prompt: str = DEFAULT_PROMPT
    base_url: str = "http://localhost:11434"
    api_key: str = "sk-no-key"


def load_profile(path: Path) -> VLMProfile:
    """Loads a VLM config from a .toml file and checks if its valid.

    Args:
        path: `Path` to the config .toml file.

    Returns:
        VLMProfile instance with the specified config.
    """
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    valid = {f.name for f in fields(VLMProfile)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown key(s) in {path}: {sorted(unknown)}; valid: {sorted(valid)}")
    return VLMProfile(**raw)
