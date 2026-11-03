"""新旧裁判配置的真实 SDK 序列化测试；MockTransport 不发送网络请求。"""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import openai
import pytest
import yaml
import zai

from scopecrit.judge_config import JudgeConfig, load_judges, preflight_credentials, resolved_base_url
from scopecrit.judge_providers import make_judge_provider


ROOT = Path(__file__).resolve().parents[1]


def configured():
    return load_judges(ROOT / "configs/judges.yaml")


LEGACY_JUDGES = [JudgeConfig.model_validate(row).model_dump() for row in [
    {"name": "legacy-deepseek", "provider": "openai", "model": "deepseek-flash",
     "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY",
     "reasoning_effort": "high", "thinking_enabled": True, "response_format_json": True},
    {"name": "legacy-glm", "provider": "zai", "model": "glm-4.5-air",
     "api_key_env": "ZHIPU_API_KEY", "temperature": 0.0},
    {"name": "legacy-grok", "provider": "openai", "model": "grok-build-0.1",
     "base_url": "https://api.apikey.fan/v1", "base_url_env": "GROK_BASE_URL",
     "api_key_env": "GROK_API_KEY", "temperature": 0.0},
]]


@pytest.fixture
def credentials(monkeypatch):
    keys = {"DEEPSEEK_API_KEY": "FAKE_DEEPSEEK_CANARY", "ZHIPU_API_KEY": "FAKE_GLM_CANARY",
            "DASHSCOPE_API_KEY": "FAKE_QWEN_CANARY", "GROK_API_KEY": "FAKE_GROK_CANARY",
            "MINIMAX_API_KEY": "FAKE_MINIMAX_CANARY", "GEMINI_API_KEY": "FAKE_GEMINI_CANARY"}
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    for name in ("GROK_BASE_URL", "MINIMAX_BASE_URL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://test-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    return keys


def test_config_defaults_use_requested_three_judges_and_exclude_agent():
    config = configured()
    expected = ["qwen3.8-flash", "minimax-m3.1-flash", "gemini-3.1-flash-lite"]
    assert [j["name"] for j in config["judges"]] == expected
    assert [j["model"] for j in config["judges"]] == expected
    assert all(j["provider"] == "openai" for j in config["judges"])
    assert config["judges"][0]["base_url"] is None
    assert all(not j["response_format_json"] for j in config["judges"])
    assert all(j["reasoning_effort"] is None and j["thinking_enabled"] is None
               for j in config["judges"])
    assert [j["enable_thinking"] for j in config["judges"]] == [False, None, None]
    assert config["max_api_retries"] == 2


@pytest.mark.parametrize("judge", configured()["judges"] + LEGACY_JUDGES,
                         ids=lambda judge: judge["name"])
def test_real_sdk_serializes_correct_endpoint_model_and_parameters(judge, credentials, monkeypatch):
    captured = []

    def transport(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert request.headers["authorization"] == "Bearer " + credentials[judge["api_key_env"]]
        assert payload["model"] == judge["model"]
        assert payload["messages"][0] == {"role": "system", "content": "评分系统提示，JSON 输出"}
        assert json.loads(payload["messages"][1]["content"]) == {"task": "T1", "prediction": "预测"}
        assert payload["stream"] is False
        assert payload["max_tokens"] == 8192
        if judge["name"] == "legacy-deepseek":
            assert request.url.host == "api.deepseek.com"
            assert payload["thinking"] == {"type": "enabled"}
            assert payload["reasoning_effort"] == "high"
            assert payload["response_format"] == {"type": "json_object"}
        else:
            assert payload.get("thinking") is None and "reasoning_effort" not in payload
            assert payload.get("response_format") is None
            if judge["model"] == "qwen3.8-flash":
                assert payload["enable_thinking"] is False
            else:
                assert "enable_thinking" not in payload
            if judge["provider"] == "zai":
                # zai-sdk 官方将 temperature=0 转为 do_sample=False、temperature=0.01。
                assert payload["temperature"] == 0.01 and payload["do_sample"] is False
            elif judge["temperature"] is not None:
                assert payload["temperature"] == judge["temperature"]
            else:
                assert "temperature" not in payload
        if judge["provider"] == "zai":
            assert request.url.host == "open.bigmodel.cn"
        elif judge["model"] == "qwen3.8-flash":
            assert request.url.host == "test-workspace.cn-beijing.maas.aliyuncs.com"
            assert request.url.path == "/compatible-mode/v1/chat/completions"
        elif judge["name"] != "legacy-deepseek":
            assert request.url.host == "api.apikey.fan"
            assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={
            "id": "judge-http-fixture", "created": 0, "object": "chat.completion", "model": judge["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"alignments": []}',
                "reasoning_content": "HIDDEN_REASONING_CANARY"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    actual_openai, actual_zai = openai.OpenAI, zai.ZhipuAiClient

    def wrapped(client):
        def construct(**kwargs):
            assert kwargs["max_retries"] == 0
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(transport))
            return client(**kwargs)
        return construct

    monkeypatch.setattr(openai, "OpenAI", wrapped(actual_openai))
    monkeypatch.setattr(zai, "ZhipuAiClient", wrapped(actual_zai))
    result = make_judge_provider(judge).complete("评分系统提示，JSON 输出", {"task": "T1", "prediction": "预测"}, "call_1")
    assert len(captured) == 1
    assert result["content"] == '{"alignments": []}'
    assert result["usage"]["total_tokens"] == 5
    assert result["model_name"] == judge["model"]
    saved = json.dumps(result)
    assert "HIDDEN_REASONING_CANARY" not in saved
    assert not any(value in saved for value in credentials.values())


def test_preflight_reports_only_names_and_status(credentials, monkeypatch):
    config = configured()
    assert all(r["status"] == "ready" for r in preflight_credentials(config))
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    monkeypatch.delenv("DASHSCOPE_BASE_URL")
    rows = preflight_credentials(config)
    assert rows[0] == {"judge": "qwen3.8-flash", "status": "missing", "missing": ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"]}
    assert not any(value in json.dumps(rows) for value in credentials.values())
    # 解析环境变量地址不改变配置，不把工作空间地址保存进 manifest。
    assert config["judges"][0]["base_url"] is None


@pytest.mark.parametrize("index", range(3))
def test_json_response_format_can_be_explicitly_enabled(index, credentials, monkeypatch):
    """网关确认支持后可独立启用，不必套用 DeepSeek 的 thinking 参数。"""
    judge = deepcopy(configured()["judges"][index])
    judge["response_format_json"] = True
    captured = []

    def transport(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert payload["response_format"] == {"type": "json_object"}
        assert "thinking" not in payload and "reasoning_effort" not in payload
        return httpx.Response(200, json={
            "id": "json-opt-in", "created": 0, "object": "chat.completion", "model": judge["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{}"}}],
        })

    actual_openai = openai.OpenAI

    def construct(**kwargs):
        return actual_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport)))

    monkeypatch.setattr(openai, "OpenAI", construct)
    result = make_judge_provider(judge).complete("返回 JSON", {"task": "T1"}, "call_1")
    assert len(captured) == 1
    assert result["content"] == "{}" and result["usage"] is None


def test_qwen_placeholder_and_credentials_in_url_rejected(credentials, monkeypatch):
    config = configured()
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    outcome = preflight_credentials(config)[0]
    assert outcome["status"] == "invalid" and outcome["error_code"] == "INVALID_BASE_URL"
    with pytest.raises(ValueError, match="DASHSCOPE_BASE_URL") as caught:
        resolved_base_url(config["judges"][0])
    assert "{WorkspaceId}" not in str(caught.value)


def test_missing_key_errors_do_not_include_values(credentials, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY") as caught:
        make_judge_provider(configured()["judges"][1])
    assert not any(value in str(caught.value) for value in credentials.values())


@pytest.mark.parametrize("mutation", ["extra_secret", "duplicate", "bad_model", "negative", "url_secret", "invalid_variable", "mixed_thinking"])
def test_config_rejects_invalid_values_without_echoing_secrets(tmp_path, mutation):
    raw = yaml.safe_load((ROOT / "configs/judges.yaml").read_text())
    secret = "FAKE_SECRET_INPUT_CANARY"
    if mutation == "extra_secret":
        raw["judges"][0]["api_key"] = secret
    elif mutation == "duplicate":
        raw["judges"].append(deepcopy(raw["judges"][0]))
    elif mutation == "bad_model":
        raw["judges"][0]["model"] = " "
    elif mutation == "negative":
        raw["max_api_retries"] = -1
    elif mutation == "url_secret":
        raw["judges"][0]["base_url"] = f"https://user:{secret}@api.example.com/v1"
    elif mutation == "invalid_variable":
        raw["judges"][0]["base_url_env"] = "NOT A VARIABLE NAME"
    else:
        raw["judges"][0]["thinking_enabled"] = False
    path = tmp_path / "judges.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError) as caught:
        load_judges(path)
    assert secret not in str(caught.value)


def test_unknown_provider_and_non_object_config_rejected(tmp_path):
    path = tmp_path / "judges.yaml"
    path.write_text("- not-an-object\n")
    with pytest.raises(ValueError, match="YAML 对象"):
        load_judges(path)
    path.write_text("judges: [{name: x, model: y, provider: unsupported, api_key_env: KEY}]\n")
    with pytest.raises(ValueError, match="provider"):
        load_judges(path)


@pytest.mark.parametrize("index, variable", [(1, "MINIMAX_BASE_URL"), (2, "GEMINI_BASE_URL")])
def test_endpoint_environment_override_wins_with_static_fallback(index, variable, credentials, monkeypatch):
    judge = configured()["judges"][index]
    assert resolved_base_url(judge) == "https://api.apikey.fan/v1"
    assert preflight_credentials(configured())[index]["status"] == "ready"
    monkeypatch.setenv(variable, "https://custom.example.com/v1/")
    assert resolved_base_url(judge) == "https://custom.example.com/v1"
    assert judge["base_url"] == "https://api.apikey.fan/v1"
    monkeypatch.setenv(variable, "http://insecure.example.com")
    with pytest.raises(ValueError):
        resolved_base_url(judge)
    assert preflight_credentials(configured())[index]["status"] == "invalid"

