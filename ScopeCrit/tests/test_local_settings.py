"""本地私有设置的解析和临时凭证生命周期。"""
import json
import os

import pytest
import yaml

from scopecrit.local_settings import configured_api, load_local_settings


def local_file(tmp_path, content):
    path = tmp_path / "settings.local.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_relative_paths_resolve_against_private_file_not_cwd(tmp_path, monkeypatch):
    path = local_file(tmp_path, {"input": "../data/input.json", "run_dir": "output",
                               "judges_config": "configs/judges.yaml", "max_samples": 521})
    monkeypatch.chdir(tmp_path.parent)
    result = load_local_settings(path)
    assert result["input"] == str((tmp_path / "../data/input.json").resolve())
    assert result["run_dir"] == str(tmp_path / "output")
    assert result["judges_config"] == str(tmp_path / "configs/judges.yaml")
    assert result["max_samples"] == 521


def test_default_none_and_empty_keys_are_allowed_until_provider_preflight(tmp_path):
    result = load_local_settings(local_file(tmp_path, {"input": "input.json"}))
    assert result["max_samples"] is None
    assert result["api_keys"] == {name: "" for name in
                                  ("deepseek", "glm", "qwen", "grok", "minimax", "gemini", "actor")}
    assert result["run_dir"] is None and result["judges_config"] is None


def test_configured_api_uses_private_values_then_restores_process(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "PREVIOUS_FAKE_CANARY")
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("GROK_BASE_URL", raising=False)
    private = load_local_settings(local_file(tmp_path, {
        "input": "input.json", "api_keys": {"deepseek": "NEW_FAKE_CANARY", "glm": "FAKE_GLM_CANARY"},
        "qwen_base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    }))
    with configured_api(private):
        assert os.environ["DEEPSEEK_API_KEY"] == "NEW_FAKE_CANARY"
        assert os.environ["ZHIPU_API_KEY"] == "FAKE_GLM_CANARY"
        assert os.environ["DASHSCOPE_BASE_URL"] == private["qwen_base_url"]
        assert os.environ["GROK_BASE_URL"] == "https://api.apikey.fan/v1"
    assert os.environ["DEEPSEEK_API_KEY"] == "PREVIOUS_FAKE_CANARY"
    assert "ZHIPU_API_KEY" not in os.environ and "GROK_BASE_URL" not in os.environ


def test_exception_still_restores_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    private = load_local_settings(local_file(tmp_path, {"input": "input.json", "api_keys": {"grok": "FAKE_KEY"}}))
    with pytest.raises(RuntimeError):
        with configured_api(private):
            assert os.environ["GROK_API_KEY"] == "FAKE_KEY"
            raise RuntimeError("simulated API failure")
    assert "GROK_API_KEY" not in os.environ


@pytest.mark.parametrize("value", [0, -1, True, "521", 1.2])
def test_invalid_limits_rejected_without_secret_disclosure(tmp_path, value):
    path = local_file(tmp_path, {"input": "input.json", "max_samples": value,
                               "api_keys": {"deepseek": "PRIVATE_FAKE_CANARY"}})
    with pytest.raises(ValueError) as caught:
        load_local_settings(path)
    assert "PRIVATE_FAKE_CANARY" not in str(caught.value)


@pytest.mark.parametrize("mutation", ["newline", "space", "unknown", "url", "blank_input"])
def test_invalid_private_values_are_not_echoed(tmp_path, mutation):
    secret = "PRIVATE_FAKE_CANARY"
    content = {"input": "input.json", "api_keys": {"deepseek": secret}}
    if mutation == "newline":
        content["api_keys"]["deepseek"] += "\n"
    elif mutation == "space":
        content["api_keys"]["deepseek"] += " "
    elif mutation == "unknown":
        content["wrong_secret_field"] = secret
    elif mutation == "url":
        content["qwen_base_url"] = f"https://user:{secret}@example.com"
    else:
        content["input"] = " "
    with pytest.raises(ValueError) as caught:
        load_local_settings(local_file(tmp_path, content))
    assert secret not in str(caught.value)


def test_bad_yaml_does_not_echo_line_containing_key(tmp_path):
    path = tmp_path / "settings.local.yaml"
    path.write_text('input: file.json\napi_keys: {deepseek: "PRIVATE_FAKE_CANARY}\n')
    with pytest.raises(ValueError) as caught:
        load_local_settings(path)
    assert "PRIVATE_FAKE_CANARY" not in str(caught.value)


def test_beijing_workspace_id_derives_qwen_address(tmp_path):
    result = load_local_settings(local_file(tmp_path, {"input": "input.json", "qwen_workspace_id": "test-workspace-42"}))
    assert result["qwen_base_url"] == "https://test-workspace-42.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


@pytest.mark.parametrize("workspace", ["bad_id", "-bad", "bad.", "a" * 64, "https://example.com", "BAD_SECRET_CANARY\n"])
def test_invalid_workspace_id_rejected_without_values(tmp_path, workspace):
    with pytest.raises(ValueError) as caught:
        load_local_settings(local_file(tmp_path, {"input": "input.json", "qwen_workspace_id": workspace}))
    assert workspace not in str(caught.value)


def test_workspace_and_explicit_url_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError) as caught:
        load_local_settings(local_file(tmp_path, {"input": "input.json", "qwen_workspace_id": "workspace-secret-canary",
                                                "qwen_base_url": "https://explicit.example.com/v1"}))
    assert "workspace-secret-canary" not in str(caught.value)
    assert "explicit.example.com" not in str(caught.value)


def test_unknown_field_names_do_not_disclose_accidentally_misplaced_keys(tmp_path):
    secret = "PRIVATE_FAKE_CANARY"
    with pytest.raises(ValueError) as caught:
        load_local_settings(local_file(tmp_path, {"input": "input.json", "api_keys": {secret: ""}}))
    assert secret not in str(caught.value)

