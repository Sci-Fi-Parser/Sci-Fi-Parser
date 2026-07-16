from __future__ import annotations

import base64
import json

import httpx
import pytest

from sci_fi_parser.vlm.vlm import ChatCompletionsVLM, _inlined_chartdata_schema, _prompt_with_suffix
from sci_fi_parser.vlm.vlm_config import VLMProfile


class _FakeResponse:
    def __init__(self, raw: dict, error: Exception | None = None):
        self._raw = raw
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._raw


def test_prompt_with_suffix_appends_after_blank_line():
    assert _prompt_with_suffix("prompt", "suffix") == "prompt\n\nsuffix"


def test_prompt_with_suffix_returns_prompt_unchanged_without_suffix():
    assert _prompt_with_suffix("prompt", "") == "prompt"


def test_inlined_chartdata_schema_has_no_refs():
    schema = _inlined_chartdata_schema()
    serialized = json.dumps(schema)

    assert "$ref" not in serialized
    assert "$defs" not in serialized
    assert set(schema["properties"]) == {"chart_type", "log_scale", "series"}
    assert '"points"' in serialized


def test_model_override_takes_precedence_over_profile():
    vlm = ChatCompletionsVLM(VLMProfile(model="profile-model"), model_override="override-model")
    assert vlm._model == "override-model"


def test_api_key_falls_back_when_env_missing(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    vlm = ChatCompletionsVLM(VLMProfile())
    assert vlm._api_key == "sk-no-key"


def test_extract_posts_image_and_parses_response(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.jpg"
    image_path.write_bytes(b"jpeg-bytes")
    chart = {
        "chart_type": "vertical_bar",
        "log_scale": False,
        "series": [{"name": "Sales", "points": [{"x": "2020", "y": 1.5}]}],
    }
    raw = {"choices": [{"message": {"content": json.dumps(chart)}}]}

    requests = []

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return _FakeResponse(raw)

    monkeypatch.setattr(httpx, "post", fake_post)

    profile = VLMProfile(base_url="http://vlm.local/", api_key="secret")
    parsed, returned_raw = ChatCompletionsVLM(profile).extract(image_path, "ocr text")

    assert parsed == chart
    assert returned_raw == raw

    url, kwargs = requests[0]
    payload = kwargs["json"]
    content = payload["messages"][0]["content"]
    image_url = content[1]["image_url"]["url"]

    assert url == "http://vlm.local/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert payload["model"] == profile.model
    assert content[0]["text"] == profile.prompt + "\n\nocr text"
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.removeprefix("data:image/jpeg;base64,")) == b"jpeg-bytes"
    assert payload["response_format"]["json_schema"] == {
        "name": "ChartData",
        "schema": _inlined_chartdata_schema(),
    }


def test_extract_defaults_to_png_mime_for_unknown_suffix(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.bmp"
    image_path.write_bytes(b"bmp-bytes")
    chart = {"chart_type": "none", "log_scale": False, "series": []}
    raw = {"choices": [{"message": {"content": json.dumps(chart)}}]}

    requests = []

    def fake_post(url, **kwargs):
        requests.append(kwargs)
        return _FakeResponse(raw)

    monkeypatch.setattr(httpx, "post", fake_post)

    parsed, _ = ChatCompletionsVLM(VLMProfile()).extract(image_path)

    assert parsed == chart
    image_url = requests[0]["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_extract_raises_on_http_error(monkeypatch, tmp_path):
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png-bytes")
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(500, request=request))

    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: _FakeResponse({}, error=error))

    with pytest.raises(httpx.HTTPStatusError):
        ChatCompletionsVLM(VLMProfile()).extract(image_path)
