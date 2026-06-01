"""VLM extractor profile — model tag + prompt + ollama options.

A profile lives in a TOML file (by convention ``config/vlm.toml``). Loaded
once at benchmark startup so a researcher can swap models or tweak the prompt
without editing Python.

Resolution order in :func:`sci_fi_parser.accuracy.benchmark.main`:

1. ``--vlm-config <path>``  → that file.
2. ``config/vlm.toml`` relative to cwd  → auto-loaded if present.
3. ``DEFAULT_PROFILE``  → the values baked in here.

The CLI ``--extractor ollama:<tag>`` still wins over the profile's ``model``
field (for one-shot model swaps without editing the file); env vars
``BENCH_NUM_CTX`` / ``BENCH_NUM_GPU`` still override the corresponding
options (for ad-hoc experiments).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


DEFAULT_PROMPT = (
    "Extract the data from this chart.\n"
    "Rules:\n"
    "- chart_type: one of bar_chart, grouped_bar_chart, stacked_bar_chart, "
    "horizontal_bar_chart, line_chart.\n"
    "- Use the x-axis category labels EXACTLY as printed. Do not invent "
    "dates, years, or names.\n"
    "- Series naming: if there is a legend, use the legend labels. "
    "If there is NO legend (single-series chart), use the y-axis title "
    "as the series name. Never leave the name blank.\n"
    "- Read each y-value from the y-axis scale and the bar / marker height. "
    "If numeric labels are printed on the bars, prefer those.\n"
    "- Watch y-axis units: '200K' = 200000, '1.5M' = 1500000, "
    "'2.3B' = 2300000000. Return plain numbers, no suffixes, no extra zeros.\n"
    "- confidence: a number from 0.0 to 1.0 reflecting how certain you are "
    "that the extracted values are correct. Lower it for charts without "
    "printed value labels or with hard-to-read axes.\n"
    "- Do not output series, categories, or values that do not appear on "
    "the chart."
)


@dataclass(slots=True)
class VLMProfile:
    """Everything the OllamaVLM extractor needs. Edit the TOML, not the code."""
    
    model: str = "qwen2.5vl:3b"
    prompt: str = DEFAULT_PROMPT
    num_ctx: int = 2048
    num_gpu: int | None = None


def load_profile(path: Path) -> VLMProfile:
    """Read a TOML profile. Missing fields fall back to dataclass defaults.

    Unknown keys raise :class:`ValueError` so typos don't silently disappear.
    """
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    valid = {f.name for f in fields(VLMProfile)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {sorted(unknown)}; valid: {sorted(valid)}")
    return VLMProfile(**raw)
