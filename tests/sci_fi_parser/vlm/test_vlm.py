from __future__ import annotations

import base64
import sys
import types

import pytest

from sci_fi_parser.vlm import vlm
from sci_fi_parser.vlm.vlm import (
    ChatCompletionsVLM,
    OllamaVLM,
    build_vlm,
    chartdata_schema,
)
from sci_fi_parser.vlm.vlm_config import VLMProfile


def _chartdata_dict() -> dict:
    return {
        "chart_type": "vertical_bar",
        "series": [{"name": "s", "points": [{"x": "a", "y": 1.0}]}],
        "confidence": 0.5,
    }


def test_prompt_with_suffix_appends_when_present():
    assert vlm._prompt_with_suffix("base", "ocr") == "base\n\nocr"


def test_prompt_with_suffix_unchanged_when_empty():
    assert vlm._prompt_with_suffix("base", "") == "base"


def test_chartdata_schema_inlines_all_refs():
    schema = chartdata_schema()

    def has_ref(node) -> bool:
        if isinstance(node, dict):
            return "$ref" in node or any(has_ref(v) for v in node.values())
        if isinstance(node, list):
            return any(has_ref(item) for item in node)
        return False

    assert not has_ref(schema)
    assert "$defs" not in schema


def test_build_vlm_returns_ollama_for_ollama_backend():
    assert isinstance(build_vlm(VLMProfile(backend="ollama")), OllamaVLM)


def test_build_vlm_returns_chat_completions_for_api_backend():
    profile = VLMProfile(backend="api", base_url="http://x")
    assert isinstance(build_vlm(profile), ChatCompletionsVLM)


def test_build_vlm_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        build_vlm(VLMProfile(backend="nonsense"))


def test_ollama_init_applies_defaults_and_seed():
    extractor = OllamaVLM(VLMProfile(model="m", num_ctx=4096))

    assert extractor.name == "m"
    assert extractor._options["num_ctx"] == 4096
    assert extractor._options["seed"] == 1
    assert extractor._options["temperature"] == 0
    assert "num_gpu" not in extractor._options


def test_ollama_init_model_override_wins():
    extractor = OllamaVLM(VLMProfile(model="m"), model_override="override")
    assert extractor.name == "override"


def test_ollama_init_env_overrides(monkeypatch):
    monkeypatch.setenv("BENCH_NUM_CTX", "8192")
    monkeypatch.setenv("BENCH_NUM_GPU", "2")
    extractor = OllamaVLM(VLMProfile(num_ctx=2048, num_gpu=None))

    assert extractor._options["num_ctx"] == 8192
    assert extractor._options["num_gpu"] == 2


def test_ollama_init_profile_num_gpu_used_without_env():
    extractor = OllamaVLM(VLMProfile(num_gpu=1))
    assert extractor._options["num_gpu"] == 1


def test_ollama_extract_passes_image_and_parses_reply(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"img")
    captured: dict = {}

    class FakeResp:
        def model_dump(self):
            return {"message": {"content": _chartdata_dict()}}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return FakeResp()

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=fake_chat))

    extractor = OllamaVLM(VLMProfile(model="m", prompt="P"))
    parsed, raw = extractor.extract(image_path, prompt_suffix="ocr")

    assert captured["model"] == "m"
    assert captured["messages"][0]["content"] == "P\n\nocr"
    assert captured["messages"][0]["images"] == [str(image_path)]
    assert captured["format"] == chartdata_schema()
    assert parsed["chart_type"] == "vertical_bar"
    assert raw == {"message": {"content": _chartdata_dict()}}


def test_chat_completions_init_api_key_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extractor = ChatCompletionsVLM(VLMProfile(backend="api", base_url="http://x/"))

    assert extractor._api_key == "sk-no-key"
    assert extractor._base_url == "http://x"


def test_chat_completions_init_reads_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    extractor = ChatCompletionsVLM(
        VLMProfile(backend="api", base_url="http://x", api_key_env="MY_KEY"))
    assert extractor._api_key == "secret"


def test_make_response_format_json_object():
    extractor = ChatCompletionsVLM(
        VLMProfile(backend="api", base_url="http://x", response_format="json_object"))
    assert extractor._make_response_format() == {"type": "json_object"}


def test_make_response_format_json_schema():
    extractor = ChatCompletionsVLM(
        VLMProfile(backend="api", base_url="http://x", response_format="json_schema"))
    rf = extractor._make_response_format()
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "ChartData"
    assert rf["json_schema"]["schema"] == chartdata_schema()


def test_make_response_format_rejects_unknown():
    extractor = ChatCompletionsVLM(
        VLMProfile(backend="api", base_url="http://x", response_format="bogus"))
    with pytest.raises(ValueError, match="unknown response_format"):
        extractor._make_response_format()


def test_chat_completions_extract_builds_payload_and_parses(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"img")
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": _chartdata_dict()}}]}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResp()

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=fake_post))

    extractor = ChatCompletionsVLM(
        VLMProfile(backend="api", model="m", prompt="P", base_url="http://x"))
    parsed, raw = extractor.extract(image_path, prompt_suffix="ocr")

    assert captured["url"] == "http://x/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-no-key"
    content = captured["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "P\n\nocr"}
    expected_data = base64.b64encode(b"img").decode("ascii")
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{expected_data}"
    assert parsed["chart_type"] == "vertical_bar"
    assert raw["choices"][0]["message"]["content"] == _chartdata_dict()
