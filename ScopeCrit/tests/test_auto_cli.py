"""四服务真实 SDK、本地 HTTP 模拟：从直接填 Key 到汇总均值的验收。"""
import importlib.util
import json
from pathlib import Path

import httpx
import openai
import pytest
import yaml
import zai

from scopecrit import cli

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("scopecrit_launcher", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_transport(monkeypatch, *, bad_glm=False):
    rows = json.loads((ROOT / "examples/merged_demo.json").read_text())
    script = json.loads((ROOT / "examples/mock_script.json").read_text())
    alignment = json.loads((ROOT / "examples/alignment_demo.json").read_text())
    expected = [entry for row in rows for entry in script[row["sample_id"]]]
    infer, judges = [], []

    def respond(request):
        payload = json.loads(request.content)
        content = json.loads(payload["messages"][1]["content"])
        model = payload["model"]
        if "targets" in content:
            judges.append((model, content))
            assert "Bearer " in request.headers["Authorization"]
            if model == "qwen3.8-flash":
                assert request.url.host == "workspace-test.cn-beijing.maas.aliyuncs.com"
            records = []
            for target in content["targets"]:
                record = next(r for r in alignment if r["sample_id"] == target["sample_id"] and r["turn"] == target["turn"])
                record = json.loads(json.dumps(record))
                if model == "glm-4.5-air" and target["sample_id"].endswith(":T1"):
                    # 预先定义一个不同意见，检验均值确实来自四份独立指标。
                    record["matches"] = record["matches"][:1]
                records.append(record)
            response = {"alignments": records}
            if bad_glm and model == "glm-4.5-air":
                return httpx.Response(401, json={"error": {"message": "not authorized", "type": "auth_error"}})
        else:
            infer.append(payload)
            response = expected[len(infer) - 1]["response"]
        return httpx.Response(200, json={
            "id": "local-transport-only", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(response, ensure_ascii=False),
                "reasoning_content": "HIDDEN_REASONING_CANARY"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    actual_openai, actual_zai = openai.OpenAI, zai.ZhipuAiClient

    def client(factory):
        def create(**kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(respond))
            return factory(**kwargs)
        return create

    monkeypatch.setattr(openai, "OpenAI", client(actual_openai))
    monkeypatch.setattr(zai, "ZhipuAiClient", client(actual_zai))
    return rows, infer, judges


def private_settings(tmp_path, rows):
    source = tmp_path / "dataset.json"
    for row in rows:
        row["material_origin"] = "AUTHENTIC"  # 本测试专用合成样本，不用于真实研究结果。
    source.write_text(json.dumps(rows, ensure_ascii=False))
    keys = {name: f"FAKE_{name}_SECRET_CANARY" for name in ("deepseek", "glm", "qwen", "grok")}
    private = tmp_path / "settings.local.yaml"
    private.write_text(yaml.safe_dump({"input": "dataset.json", "max_samples": None,
        "api_keys": keys, "qwen_workspace_id": "workspace-test",
        "run_dir": "results", "judges_config": str(ROOT / "configs/judges.yaml")}))
    return source, private, keys


def test_private_keys_full_four_model_evaluation_and_cache(tmp_path, monkeypatch, capsys):
    rows, infer, judges = fixture_transport(monkeypatch)
    source, private, keys = private_settings(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    for name in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "GROK_API_KEY", "DASHSCOPE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert launcher().main(["--settings", str(private)]) == 0
    assert len(infer) == 35 and len(judges) == 16
    result = tmp_path / "results"
    report = json.loads((result / "judge_summary.json").read_text())
    assert report["judging_complete"] is True
    aspect = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert aspect["mean"] == pytest.approx(0.875)
    assert aspect["valid_judges"] == 4
    assert report["per_task"]["T1"]["metrics"]["evidence_f1"]["mean"] == 1
    for name in ("metrics_summary.csv", "per_sample_metrics.csv", "report.md"):
        assert (result / name).is_file()
    # 只重跑评分：已有完成响应复用，不增加答题或裁判请求。
    assert launcher().main(["--settings", str(private), "--evaluate-only", "--run-dir", str(result)]) == 0
    assert len(infer) == 35 and len(judges) == 16
    outputs = capsys.readouterr().out + capsys.readouterr().err
    for file in result.rglob("*"):
        if file.is_file():
            outputs += file.read_bytes().decode("utf-8", errors="ignore")
    assert "HIDDEN_REASONING_CANARY" not in outputs
    for key in keys.values():
        assert key not in outputs


def test_failed_judge_does_not_become_zero_or_four_model_mean(tmp_path, monkeypatch):
    rows, infer, judges = fixture_transport(monkeypatch, bad_glm=True)
    _, private, _ = private_settings(tmp_path, rows)
    assert launcher().main(["--settings", str(private)]) == 1
    report = json.loads((tmp_path / "results/judge_summary.json").read_text())
    metric = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert metric["mean"] is None and metric["valid_judges"] == 3
    assert metric["available_mean"] == 1
    assert report["judging_complete"] is False
    assert len([item for item in judges if item[0] == "glm-4.5-air"]) == 1


def test_local_settings_errors_do_not_echo_keys(tmp_path, capsys):
    path = tmp_path / "settings.local.yaml"
    path.write_text('api_keys: {deepseek: "SECRET_NOT_IN_ERRORS"}\ninput: []\n')
    assert launcher().main(["--settings", str(path)]) == 2
    assert "SECRET_NOT_IN_ERRORS" not in capsys.readouterr().err


def test_evaluation_only_without_settings_never_starts_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv: pytest.fail("不能启动任何答题流程"))
    assert launcher().main(["--evaluate-only", "--settings", str(tmp_path / "missing.yaml"),
                            "--input", "anything.json", "--run-dir", str(tmp_path / "new")]) == 2
    assert not (tmp_path / "new").exists()
