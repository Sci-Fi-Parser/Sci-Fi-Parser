"""VLM extractors — two backends behind one ``extract`` interface.

* :class:`OllamaVLM` talks to a local ollama server with schema-enforced JSON.
* :class:`ChatCompletionsVLM` talks to any OpenAI-compatible chat-completions
  endpoint (OpenAI, vLLM, LM Studio, llama.cpp) via a plain ``httpx`` POST.

Kept in its own module so anyone who only needs an extractor (the pipeline,
a one-off script) can ``from sci_fi_parser.vlm.vlm import build_vlm`` without
pulling in the rest of the benchmark machinery.

Configuration comes from a
:class:`~sci_fi_parser.vlm.vlm_config.VLMProfile`; ``profile.backend`` picks
which class :func:`build_vlm` returns. Env vars ``BENCH_NUM_CTX`` and
``BENCH_NUM_GPU`` override the ollama options for ad-hoc experiments.
"""

from __future__ import annotations

import base64
import copy
import os
from pathlib import Path
from typing import Any

from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.schema import ChartData, parse_chartdata


_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# VLM inference can take minutes per chart on CPU; give the request plenty of room.
_REQUEST_TIMEOUT = 600.0


def _prompt_with_suffix(prompt: str, suffix: str) -> str:
    """Append ``suffix`` (OCR text or any side signal) to the base prompt.

    Empty suffix -> the prompt unchanged, for back-compat with callers that
    pass nothing (benchmark, comparison runner).
    """
    return prompt + (f"\n\n{suffix}" if suffix else "")


def chartdata_schema() -> dict[str, Any]:
    """ChartData's JSON schema with all ``$ref``/``$defs`` inlined.

    llama.cpp's schema-to-grammar converter does not resolve ``$ref``, so the
    nested ``Series``/``Point`` definitions pydantic emits as references are
    left ungrammared and the model invents field names. Inlining the refs makes
    the whole structure constrainable.
    """
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


class OllamaVLM:
    """Local VLM via Ollama with schema-enforced JSON output.

    Ollama's ``format=`` accepts a JSON schema and guarantees the reply
    conforms to it — standardization-at-source for the VLM path. Needs an
    ollama server reachable on localhost and the model pulled
    (``ollama pull <tag>``).
    """

    def __init__(self, profile: VLMProfile | None = None,
                 model_override: str | None = None):
        profile = profile or VLMProfile()
        self.name = model_override or profile.model
        self._model = self.name
        self._prompt = profile.prompt
        self._options: dict = {
            "num_ctx": int(os.environ.get("BENCH_NUM_CTX", str(profile.num_ctx))),
        }
        env_gpu = os.environ.get("BENCH_NUM_GPU")
        if env_gpu is not None:
            self._options["num_gpu"] = int(env_gpu)
        elif profile.num_gpu is not None:
            self._options["num_gpu"] = profile.num_gpu

    def extract(self, image_path: Path, prompt_suffix: str = "") -> ChartData:
        """Run the VLM on one image. ``prompt_suffix`` is appended to the base
        prompt -- used by the pipeline to inject OCR text (or any other side
        signal) as additional context.
        """
        import ollama  # pylint: disable=import-outside-toplevel
        content = _prompt_with_suffix(self._prompt, prompt_suffix)
        resp = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": content,
                       "images": [str(image_path)]}],
            format=chartdata_schema(),
            options=self._options,
        )
        return parse_chartdata(resp["message"]["content"])


class ChatCompletionsVLM:
    """VLM via any OpenAI-compatible chat-completions endpoint.

    Targets OpenAI, vLLM, LM Studio, or a local llama.cpp ``llama-server`` --
    whichever ``profile.base_url`` points at. The model is sent the chart image
    as a base64 data URL and asked for JSON matching :class:`ChartData`.

    ``profile.response_format`` picks the JSON-constraint strategy:

    * ``"json_schema"`` (default) sends the ChartData schema as a
      ``response_format`` -- servers like llama.cpp turn it into a decoding
      grammar, the same guarantee ollama's ``format=`` gives.
    * ``"json_object"`` only asks for valid JSON and leans on
      :func:`~sci_fi_parser.schema.parse_chartdata` to repair the reply -- the
      fallback for servers without json_schema support.

    The API key is read from the env var named by ``profile.api_key_env``
    (default ``OPENAI_API_KEY``); a dummy is sent when it is unset so a keyless
    local server still works.
    """

    def __init__(self, profile: VLMProfile | None = None,
                 model_override: str | None = None):
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
            return {"type": "json_schema",
                    "json_schema": {"name": "ChartData",
                                    "schema": chartdata_schema()}}
        raise ValueError(
            f"unknown response_format {self._response_format!r} "
            "(expected 'json_schema' or 'json_object')")

    def extract(self, image_path: Path, prompt_suffix: str = "") -> ChartData:
        """Run the VLM on one image. ``prompt_suffix`` is appended to the base
        prompt -- used by the pipeline to inject OCR text (or any other side
        signal) as additional context.
        """
        import httpx  # pylint: disable=import-outside-toplevel
        mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": _prompt_with_suffix(self._prompt, prompt_suffix)},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{data}"}},
            ]}],
            "response_format": self._make_response_format(),
            "chat_template_kwargs": {"enable_thinking": False}
        }
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return parse_chartdata(resp.json()["choices"][0]["message"]["content"])


def build_vlm(profile: VLMProfile | None = None,
              model_override: str | None = None):
    """Return the extractor for ``profile.backend`` (``"ollama"`` | ``"api"``)."""
    profile = profile or VLMProfile()
    if profile.backend == "ollama":
        return OllamaVLM(profile=profile, model_override=model_override)
    if profile.backend == "api":
        return ChatCompletionsVLM(profile=profile, model_override=model_override)
    raise ValueError(
        f"unknown backend {profile.backend!r} (expected 'ollama' or 'api')")
