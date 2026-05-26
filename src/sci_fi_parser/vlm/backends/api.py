"""OpenAI-compatible HTTP VLM backend.

Works against any server exposing `/v1/chat/completions`: ollama, llama.cpp
server, vLLM, LM Studio, OpenAI, OpenRouter, etc. Uses stdlib `urllib` so the
default install needs no extra dependencies.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path

from sci_fi_parser.schema import ChartData, parse_chartdata
from sci_fi_parser.vlm.config import VLMConfig


class APIVLM:
    name: str

    def __init__(self, config: VLMConfig):
        self.name = config.model
        self._model = config.model
        self._prompt = config.prompt
        self._base_url = config.api.base_url.rstrip("/")
        self._api_key = os.environ.get("SCIFI_VLM_API_KEY") or config.api.api_key
        self._schema_mode = config.api.schema_mode
        self._timeout = config.api.timeout_s

    def extract(self, image_path: Path) -> ChartData:
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt},
                    {"type": "image_url",
                     "image_url": {"url": _data_url(image_path)}},
                ],
            }],
            "response_format": self._response_format(),
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return parse_chartdata(content)

    def _response_format(self) -> dict:
        if self._schema_mode == "schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "ChartData",
                    "schema": ChartData.model_json_schema(),
                    "strict": True,
                },
            }
        return {"type": "json_object"}


def _data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
