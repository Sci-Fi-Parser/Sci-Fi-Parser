"""VLM extractor — talks to a local ollama server with schema-enforced JSON.

Kept in its own module so anyone who only needs the extractor (the pipeline,
a one-off script) can ``from sci_fi_parser.vlm.vlm import OllamaVLM``
without pulling in the rest of the benchmark machinery.

Configuration (model tag, prompt, ollama options) comes from a
:class:`~sci_fi_parser.vlm.vlm_config.VLMProfile`. Env vars
``BENCH_NUM_CTX`` and ``BENCH_NUM_GPU`` override the profile values for
ad-hoc experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.schema import ChartData, parse_chartdata


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
        signal) as additional context. Empty by default for back-compat with
        every existing caller (benchmark, comparison runner).
        """
        import ollama  # pylint: disable=import-outside-toplevel
        content = self._prompt + (f"\n\n{prompt_suffix}" if prompt_suffix else "")
        resp = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": content,
                       "images": [str(image_path)]}],
            format=ChartData.model_json_schema(),
            options=self._options,
        )
        return parse_chartdata(resp["message"]["content"])
