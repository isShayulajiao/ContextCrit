"""模型替换和阶段路由的回归测试；所有 SDK 调用均在本地替身中完成。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scopecrit.config import Config, ProviderConfig, load_config
from scopecrit.providers import OpenAICompatibleProvider, make_provider


def local_options(**kwargs):
    return {"kind": "openai", "model": "local-qwen", "base_url": "http://127.0.0.1:8000/v1",
            **kwargs}


def fake_sdk(monkeypatch):
    clients, calls = [], []

    def openai(**options):
        clients.append(options)

        def create(**request):
            calls.append((options, request))
            return SimpleNamespace(id="local-test", usage=None, choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"result":true}', reasoning_content="PRIVATE_CANARY"),
                finish_reason="stop")])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai))
    return clients, calls


def test_generic_local_backend_does_not_inherit_deepseek_options(monkeypatch):
    clients, calls = fake_sdk(monkeypatch)
    options = ProviderConfig.model_validate(local_options(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})).model_dump()
    provider = OpenAICompatibleProvider(options)
    result = provider.complete("ACTION", {"visible_evidence": ["E001"]}, "Demo:T1:0001")
    assert clients[0]["api_key"] == "local-no-auth"
    assert clients[0]["max_retries"] == 0
    request = calls[0][1]
    assert request["model"] == "local-qwen"
    assert "reasoning_effort" not in request
    assert "response_format" not in request
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert json.loads(request["messages"][1]["content"]) == {"visible_evidence": ["E001"]}
    assert "PRIVATE_CANARY" not in json.dumps(result)
    assert result["provider_name"] == "openai"


def test_stage_routing_changes_action_only_and_preserves_requests(monkeypatch):
    _, calls = fake_sdk(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    config = Config.model_validate({"provider": {"kind": "deepseek"},
        "stage_providers": {"ACTION": local_options()}}).model_dump()
    provider = make_provider(config)
    for stage in ("ACTION", "SUPPORT_CHECK", "SCOPE_COMPARE"):
        request = {"stage_visible_evidence": [stage], "no_full_document": True}
        provider.complete(stage, request, "Demo:T1:0001")
        assert json.loads(calls[-1][1]["messages"][1]["content"]) == request
    assert [call[1]["model"] for call in calls] == ["local-qwen", "deepseek-flash", "deepseek-flash"]
    assert "extra_body" not in calls[0][1]
    assert calls[1][1]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_all_stages_can_override_unused_default_without_its_key(monkeypatch):
    clients, calls = fake_sdk(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = Config.model_validate({"provider": {"kind": "deepseek"},
        "stage_providers": {stage: local_options() for stage in
            ("ACTION", "SUPPORT_CHECK", "SCOPE_COMPARE")}}).model_dump()
    provider = make_provider(config)
    provider.complete("SUPPORT_CHECK", {}, "Demo:T1:0001")
    assert len(clients) == 3
    assert calls[0][1]["model"] == "local-qwen"
    with pytest.raises(ValueError, match="未知模型调用阶段"):
        provider.complete("JUDGE", {}, "Demo:T1:0002")


def test_remote_generic_backend_uses_configured_key_reference(monkeypatch):
    clients, calls = fake_sdk(monkeypatch)
    monkeypatch.setenv("SCOPECRIT_ACTOR_API_KEY", "local-settings-secret")
    options = ProviderConfig.model_validate(local_options(
        base_url="https://example.test/v1", api_key_env="SCOPECRIT_ACTOR_API_KEY")).model_dump()
    provider = OpenAICompatibleProvider(options)
    result = provider.complete("ACTION", {}, "Demo:T1:0001")
    assert clients[0]["api_key"] == "local-settings-secret"
    assert "local-settings-secret" not in json.dumps(result)
    assert "extra_body" not in calls[0][1]


@pytest.mark.parametrize("options", [
    {"kind": "openai", "model": "a"},
    {"kind": "openai", "base_url": "http://localhost:8000/v1"},
    local_options(base_url="http://example.test/v1"),
    local_options(base_url="https://key:secret@example.test/v1"),
    local_options(base_url="https://example.test/v1?api_key=secret"),
    local_options(base_url="https://example.test/v1"),
    local_options(base_url="http://localhost:0/v1"),
    local_options(base_url="http://localhost:8000/ path"),
    local_options(extra_body={"messages": [{"role": "system", "content": "override"}]}),
    local_options(extra_body={"chat_template_kwargs": {"api_key": "secret"}}),
    local_options(extra_body={"temperature": float("nan")}),
    local_options(thinking_enabled=True),
    local_options(api_key_env="sk-direct-secret"),
])
def test_invalid_or_ambiguous_backend_configs_are_rejected(options):
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(options)


@pytest.mark.parametrize("url", ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1",
                                 "http://[::1]:8000/v1"])
def test_auth_free_loopback_backends(url):
    assert ProviderConfig.model_validate(local_options(base_url=url)).api_key_env is None


def test_unknown_stage_is_not_silently_ignored():
    with pytest.raises(ValidationError):
        Config.model_validate({"stage_providers": {"ACTON": local_options()}})


def test_local_config_example_loads_and_defaults_remain_backwards_compatible():
    root = Path(__file__).resolve().parents[1]
    local = load_config(root / "configs" / "local_openai.yaml")
    assert local["provider"]["kind"] == "openai"
    assert local["provider"]["reasoning_effort"] is None
    assert local["provider"]["thinking_enabled"] is None
    deepseek = load_config(root / "configs" / "deepseek.yaml")
    assert deepseek["provider"]["thinking_enabled"] is True
    assert deepseek["provider"]["reasoning_effort"] == "high"


def test_real_sdk_serialization_against_mock_http_transport(monkeypatch):
    """实际 SDK 的 HTTP 请求构造检查；传输层在内存中返回响应，不开网络端口。"""
    import httpx
    import openai

    observed = []
    real_openai = openai.OpenAI

    def handle(request):
        body = json.loads(request.content)
        observed.append({"path": request.url.path, "body": body})
        return httpx.Response(200, json={"id": "local-response", "object": "chat.completion", "created": 0,
            "model": "local-qwen", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

    def create_client(**kwargs):
        return real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(openai, "OpenAI", create_client)
    options = ProviderConfig.model_validate(local_options(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})).model_dump()
    provider = OpenAICompatibleProvider(options)
    try:
        response = provider.complete("ACTION", {"question": "仅分析当前证据"}, "Demo:T1:0001")
        assert response["usage"]["total_tokens"] == 5
        assert response["content"] == '{"ok":true}'
        assert observed[0]["path"] == "/v1/chat/completions"
        body = observed[0]["body"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "thinking" not in body and "reasoning_effort" not in body
        assert json.loads(body["messages"][1]["content"]) == {"question": "仅分析当前证据"}
    finally:
        provider.client.close()
