"""严格协议与模型适配器测试；不会访问真实 API。"""
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scopecrit.config import Config, load_config
from scopecrit.providers import DeepSeekProvider, MockProvider
from scopecrit.schemas import SupportCheck, parse_action


def test_action_strict_fields_and_labels():
    action = {"action": "PROPOSE", "reason": "提取评价", "arguments": {"claims": [{
        "text": "整体结构严谨", "aspect": "STRUCTURE", "target_scope": "整体",
        "stance": "POSITIVE", "evidence_ids": ["E001"]}]}}
    assert parse_action(json.dumps(action))["arguments"]["claims"][0]["aspect"] == "STRUCTURE"
    action["arguments"]["claims"][0]["holder"] = "评论者"
    with pytest.raises(ValidationError):
        parse_action(json.dumps(action))
    del action["arguments"]["claims"][0]["holder"]
    action["arguments"]["claims"][0]["stance"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        parse_action(json.dumps(action))


def test_boolean_is_not_top_k_integer():
    with pytest.raises(ValidationError):
        parse_action(json.dumps({"action": "SEARCH", "reason": "查找", "arguments": {"query": "结尾", "top_k": True}}))


def test_support_boolean_is_strict():
    with pytest.raises(ValidationError):
        SupportCheck.model_validate({"candidate_id": "P001", "support": "SUPPORTED",
            "scope_consistent": "true", "voice_preserved": True, "modality_preserved": True,
            "evidence_ids": ["E001"], "reason": "依据"})


def test_mock_is_scripted_and_replayable():
    provider = MockProvider({"Demo:T1": [{"stage": "ACTION", "response": {"action": "YIELD"}}]})
    a = provider.complete("ACTION", {"role": "solo"}, "Demo:T1:0001")
    assert provider.complete("ACTION", {}, "Demo:T1:0001") == a
    assert a["usage"] is None
    with pytest.raises(ValueError, match="阶段不符"):
        provider.complete("SUPPORT_CHECK", {}, "Demo:T1:0001")
    with pytest.raises(ValueError, match="没有"):
        provider.complete("ACTION", {}, "Demo:T1:0002")


def test_config_defaults_and_relative_script(tmp_path):
    file = tmp_path / "mock.yaml"
    file.write_text("fixture: true\nprovider:\n  kind: mock\nmock:\n  script: actions.json\n")
    config = load_config(file)
    assert config["limits"]["max_model_calls_per_episode"] == 80
    assert config["mock"]["script"] == str(tmp_path / "actions.json")
    file.write_text("limits:\n  max_api_retries: -1\n")
    with pytest.raises(ValidationError):
        load_config(file)
    file.write_text("misspelled_field: 42\n")
    with pytest.raises(ValidationError):
        load_config(file)


def test_deepseek_official_sdk_arguments_and_no_reasoning_leak(monkeypatch):
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="request_1", usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok":true}', reasoning_content="PRIVATE_REASONING_CANARY"),
            finish_reason="stop")])
    def fake_openai(**kwargs):
        assert kwargs["max_retries"] == 0
        assert kwargs["api_key"] == "test-secret"
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    provider = DeepSeekProvider(Config().model_dump()["provider"])
    result = provider.complete("ACTION", {"question": "评价是什么？"}, "Demo:T1:0001")
    assert captured["model"] == "deepseek-flash"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"]["thinking"]["type"] == "enabled"
    assert "PRIVATE_REASONING_CANARY" not in json.dumps(result)
    assert "test-secret" not in json.dumps(result)


def test_deepseek_requires_one_key_and_does_not_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: None))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="必须认证"):
        DeepSeekProvider(Config().model_dump()["provider"])


@pytest.mark.parametrize("limit", [0, -2, True, False, "521", "None", 1.5])
def test_max_samples_config_is_strict_positive_or_none(limit):
    with pytest.raises(ValidationError):
        Config.model_validate({"runtime": {"max_samples": limit}})


def test_max_samples_yaml_null_and_integer(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("runtime:\n  max_samples: null\n")
    assert load_config(path)["runtime"]["max_samples"] is None
    path.write_text("runtime:\n  max_samples: 521\n")
    assert load_config(path)["runtime"]["max_samples"] == 521
