"""四家裁判的真实 SDK 序列化测试；MockTransport 不发送网络请求。"""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import openai
import pytest
import yaml
import zai

from scopecrit.judge_config import load_judges, preflight_credentials, resolved_base_url
from scopecrit.judge_providers import make_judge_provider


ROOT = Path(__file__).resolve().parents[1]


def configured():
    return load_judges(ROOT / "configs/judges.yaml")


@pytest.fixture
def credentials(monkeypatch):
    keys = {"DEEPSEEK_API_KEY": "FAKE_DEEPSEEK_CANARY", "ZHIPU_API_KEY": "FAKE_GLM_CANARY",
            "DASHSCOPE_API_KEY": "FAKE_QWEN_CANARY", "GROK_API_KEY": "FAKE_GROK_CANARY"}
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GROK_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://test-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    return keys


def test_config_defaults_use_requested_four_models():
    config = configured()
    assert [j["name"] for j in config["judges"]] == ["deepseek-flash", "GLM-4.5-Air", "qwen3.8-flash", "grok-build-0.1"]
    assert [j["model"] for j in config["judges"]] == ["deepseek-flash", "glm-4.5-air", "qwen3.8-flash", "grok-build-0.1"]
    assert config["judges"][1]["provider"] == "zai"
    assert config["judges"][2]["base_url"] is None
    assert config["max_api_retries"] == 2


@pytest.mark.parametrize("index", range(4))
def test_real_sdk_serializes_correct_endpoint_model_and_parameters(index, credentials, monkeypatch):
    judge = configured()["judges"][index]
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
        if index == 0:
            assert request.url.host == "api.deepseek.com"
            assert payload["thinking"] == {"type": "enabled"}
            assert payload["reasoning_effort"] == "high"
            assert payload["response_format"] == {"type": "json_object"}
        else:
            assert payload.get("thinking") is None and "reasoning_effort" not in payload
            assert payload.get("response_format") is None
            if index == 1:
                # zai-sdk 官方将 temperature=0 转为 do_sample=False、temperature=0.01。
                assert payload["temperature"] == 0.01 and payload["do_sample"] is False
            else:
                assert payload["temperature"] == 0.0
        if index == 1:
            assert request.url.host == "open.bigmodel.cn"
        if index == 2:
            assert request.url.host == "test-workspace.cn-beijing.maas.aliyuncs.com"
            assert request.url.path == "/compatible-mode/v1/chat/completions"
        if index == 3:
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
    assert rows[2] == {"judge": "qwen3.8-flash", "status": "missing", "missing": ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"]}
    assert not any(value in json.dumps(rows) for value in credentials.values())
    # 解析环境变量地址不改变配置，不把工作空间地址保存进 manifest。
    assert config["judges"][2]["base_url"] is None


def test_qwen_placeholder_and_credentials_in_url_rejected(credentials, monkeypatch):
    config = configured()
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    outcome = preflight_credentials(config)[2]
    assert outcome["status"] == "invalid" and outcome["error_code"] == "INVALID_BASE_URL"
    with pytest.raises(ValueError, match="DASHSCOPE_BASE_URL") as caught:
        resolved_base_url(config["judges"][2])
    assert "{WorkspaceId}" not in str(caught.value)


def test_missing_key_errors_do_not_include_values(credentials, monkeypatch):
    monkeypatch.delenv("GROK_API_KEY")
    with pytest.raises(ValueError, match="GROK_API_KEY") as caught:
        make_judge_provider(configured()["judges"][3])
    assert not any(value in str(caught.value) for value in credentials.values())


@pytest.mark.parametrize("mutation", ["extra_secret", "duplicate", "bad_model", "negative", "url_secret", "invalid_variable"])
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
    else:
        raw["judges"][0]["base_url_env"] = "NOT A VARIABLE NAME"
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


def test_endpoint_environment_override_wins_with_static_fallback(credentials, monkeypatch):
    judge = configured()["judges"][3]
    assert resolved_base_url(judge) == "https://api.apikey.fan/v1"
    assert preflight_credentials(configured())[3]["status"] == "ready"
    monkeypatch.setenv("GROK_BASE_URL", "https://custom.example.com/v1/")
    assert resolved_base_url(judge) == "https://custom.example.com/v1"
    assert judge["base_url"] == "https://api.apikey.fan/v1"
    monkeypatch.setenv("GROK_BASE_URL", "http://insecure.example.com")
    with pytest.raises(ValueError):
        resolved_base_url(judge)
    assert preflight_credentials(configured())[3]["status"] == "invalid"
