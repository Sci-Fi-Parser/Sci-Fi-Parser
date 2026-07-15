from __future__ import annotations

import pytest

from sci_fi_parser.vlm.vlm_config import DEFAULT_PROMPT, VLMProfile, load_profile


def test_load_profile_reads_toml_and_keeps_defaults(tmp_path):
    config = tmp_path / "vlm.toml"
    config.write_text('model = "test-model"\nbase_url = "http://example.test/v1"\n')

    profile = load_profile(config)

    assert profile.model == "test-model"
    assert profile.base_url == "http://example.test/v1"
    assert profile.prompt == DEFAULT_PROMPT
    assert profile.api_key == "sk-no-key"


def test_load_profile_rejects_unknown_keys(tmp_path):
    config = tmp_path / "vlm.toml"
    config.write_text('model = "test-model"\ntypo_key = "oops"\n')

    with pytest.raises(ValueError, match="typo_key"):
        load_profile(config)


def test_profile_defaults():
    profile = VLMProfile()

    assert profile.model == "qwen2.5vl:7b"
    assert profile.prompt == DEFAULT_PROMPT
    assert profile.base_url == "http://localhost:11434"
    assert profile.api_key == "sk-no-key"
