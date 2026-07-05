from __future__ import annotations

import base64
import copy
import os
from pathlib import Path
from typing import Any

from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.vlm.vlm_schema import ChartData, parse_chartdata

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# VLM inference can take minutes per chart on CPU; give the request plenty of room.
_REQUEST_TIMEOUT = 600.0


def _prompt_with_suffix(prompt: str, suffix: str) -> str:
    return prompt + (f"\n\n{suffix}" if suffix else "")


def chartdata_schema() -> dict[str, Any]:
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


class ChatCompletionsVLM:
    def __init__(self, profile: VLMProfile | None = None, model_override: str | None = None):
        profile = profile or VLMProfile()
        self.name = model_override or profile.model
        self._model = self.name
        self._prompt = profile.prompt
        self._base_url = profile.base_url.rstrip("/")
        self._api_key = os.environ.get(profile.api_key_env) or "sk-no-key"
        self._response_format = profile.response_format

    def _make_response_format(self) -> dict:
        if self._response_format == "json_object":
            return {"type": "json_object"}
        if self._response_format == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "ChartData", "schema": chartdata_schema()},
            }
        raise ValueError(
            f"unknown response_format {self._response_format!r} (expected 'json_schema' or 'json_object')"
        )

    def extract(self, image_path: Path, prompt_suffix: str = "") -> tuple[dict, dict]:
        import httpx  # pylint: disable=import-outside-toplevel

        mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _prompt_with_suffix(self._prompt, prompt_suffix),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                    ],
                }
            ],
            "response_format": self._make_response_format(),
            "seed": 1,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw_data = resp.json()
        content = raw_data["choices"][0]["message"]["content"]
        return (parse_chartdata(content).model_dump(), raw_data)
