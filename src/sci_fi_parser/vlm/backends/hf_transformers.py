"""In-process VLM backend via the HuggingFace transformers library.

Lazy-imports the heavy deps (`transformers`, `torch`, `PIL`, `outlines`)
inside __init__ so importing `sci_fi_parser.vlm` stays cheap for users who
only need the API backend. Install the extras with
`pip install sci-fi-parser[hf_transformers]`.

Decoding is constrained to ChartData by `outlines`: the model is wrapped
with `outlines.from_transformers` and given `output_type=ChartData`, so the
decoder can only emit tokens that keep the output a valid match for the
schema.
"""

from __future__ import annotations

from pathlib import Path

from sci_fi_parser.schema import ChartData, parse_chartdata
from sci_fi_parser.vlm.config import VLMConfig


class HFTransformersVLM:
    name: str

    def __init__(self, config: VLMConfig):
        import outlines
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = config.model
        self._prompt = config.prompt
        self._max_new_tokens = config.hf_transformers.max_new_tokens
        dtype = _resolve_dtype(config.hf_transformers.dtype)
        device_map = config.hf_transformers.device
        self._processor = AutoProcessor.from_pretrained(config.model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            config.model, torch_dtype=dtype, device_map=device_map)
        self._outlines = outlines.from_transformers(self._model, self._processor)

    def extract(self, image_path: Path) -> ChartData:
        from outlines.inputs import Chat, Image
        from PIL import Image as PILImage

        image = PILImage.open(image_path)
        chat = Chat([{
            "role": "user",
            "content": [
                {"type": "image", "image": Image(image)},
                {"type": "text", "text": self._prompt},
            ],
        }])
        text = self._outlines(
            chat, output_type=ChartData, max_new_tokens=self._max_new_tokens)
        return parse_chartdata(text)


def _resolve_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of "
            f"{sorted(mapping) + ['auto']}")
    return mapping[name]
