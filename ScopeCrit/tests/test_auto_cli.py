"""DeepSeek 答题、三个裁判的真实 SDK 本地 HTTP 模拟验收。"""
import importlib.util
import json
from pathlib import Path

import httpx
import openai
import pytest
import yaml

from scopecrit import cli
from scopecrit.providers import prompt_text

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("scopecrit_launcher", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_transport(monkeypatch, *, bad_minimax=False, local_actor=False):
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
                assert payload["enable_thinking"] is False
            else:
                assert model in {"minimax-m3.1-flash", "gemini-3.1-flash-lite"}
                assert request.url.host == "api.apikey.fan"
                assert request.url.path == "/v1/chat/completions"
                assert "thinking" not in payload and "enable_thinking" not in payload
            records = []
            for target in content["targets"]:
                record = next(r for r in alignment if r["sample_id"] == target["sample_id"] and r["turn"] == target["turn"])
                record = json.loads(json.dumps(record))
                if model == "minimax-m3.1-flash" and target["sample_id"].endswith(":T1"):
                    # 预先定义一个不同意见，检验均值确实来自三份独立指标。
                    record["matches"] = record["matches"][:1]
                records.append(record)
            response = {"alignments": records}
            if bad_minimax and model == "minimax-m3.1-flash":
                return httpx.Response(401, json={"error": {"message": "not authorized", "type": "auth_error"}})
        else:
            if local_actor:
                assert model == "local-qwen"
                assert request.url.host == "127.0.0.1" and request.url.port == 8000
                assert request.url.path == "/v1/chat/completions"
                assert request.headers["Authorization"] == "Bearer local-no-auth"
                assert payload["chat_template_kwargs"] == {"enable_thinking": False}
                assert not {"thinking", "reasoning_effort", "response_format"}.intersection(payload)
                entry = expected[len(infer) % len(expected)]
                assert payload["messages"][0]["content"].startswith(prompt_text(entry["stage"]))
            else:
                assert model == "deepseek-flash"
                assert request.url.host == "api.deepseek.com"
            infer.append(payload)
            # 多文件端到端验收复用同一组预先定义的四条任务脚本。
            response = expected[(len(infer) - 1) % len(expected)]["response"]
        return httpx.Response(200, json={
            "id": "local-transport-only", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(response, ensure_ascii=False),
                "reasoning_content": "HIDDEN_REASONING_CANARY"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    actual_openai = openai.OpenAI

    def client(factory):
        def create(**kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(respond))
            return factory(**kwargs)
        return create

    monkeypatch.setattr(openai, "OpenAI", client(actual_openai))
    return rows, infer, judges


def private_settings(tmp_path, rows):
    source = tmp_path / "dataset.json"
    for row in rows:
        row["material_origin"] = "AUTHENTIC"  # 本测试专用合成样本，不用于真实研究结果。
    source.write_text(json.dumps(rows, ensure_ascii=False))
    keys = {name: f"FAKE_{name}_SECRET_CANARY" for name in ("deepseek", "qwen", "minimax", "gemini")}
    private = tmp_path / "settings.local.yaml"
    private.write_text(yaml.safe_dump({"input": "dataset.json", "max_samples": None,
        "api_keys": keys, "qwen_workspace_id": "workspace-test",
        "run_dir": "results", "judges_config": str(ROOT / "configs/judges.yaml")}))
    return source, private, keys


def test_private_keys_deepseek_agent_three_judges_evaluation_and_cache(tmp_path, monkeypatch, capsys):
    rows, infer, judges = fixture_transport(monkeypatch)
    source, private, keys = private_settings(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MINIMAX_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert launcher().main(["--settings", str(private)]) == 0
    assert len(infer) == 35 and len(judges) == 12
    result = tmp_path / "results"
    report = json.loads((result / "judge_summary.json").read_text())
    assert report["judging_complete"] is True
    assert report["expected_judges"] == ["qwen3.8-flash", "minimax-m3.1-flash", "gemini-3.1-flash-lite"]
    aspect = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert aspect["mean"] == pytest.approx(5 / 6)
    assert aspect["valid_judges"] == 3
    assert report["per_task"]["T1"]["metrics"]["evidence_f1"]["mean"] == 1
    for name in ("metrics_summary.csv", "per_sample_metrics.csv", "report.md"):
        assert (result / name).is_file()
    # 只重跑评分：已有完成响应复用，不增加答题或裁判请求。
    assert launcher().main(["--settings", str(private), "--evaluate-only", "--run-dir", str(result)]) == 0
    assert len(infer) == 35 and len(judges) == 12
    captured = capsys.readouterr()
    outputs = captured.out + captured.err
    for file in result.rglob("*"):
        if file.is_file():
            outputs += file.read_bytes().decode("utf-8", errors="ignore")
    assert "HIDDEN_REASONING_CANARY" not in outputs
    for key in keys.values():
        assert key not in outputs


def test_local_qwen_runs_all_agent_stages_without_deepseek_key_and_keeps_three_judges(tmp_path, monkeypatch):
    rows, infer, judges = fixture_transport(monkeypatch, local_actor=True)
    _, private, _ = private_settings(tmp_path, rows)
    local_config = tmp_path / "local_agent.yaml"
    local_config.write_text(yaml.safe_dump({
        "provider": {"kind": "openai", "model": "local-qwen", "base_url": "http://127.0.0.1:8000/v1",
                     "api_key_env": None, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "evaluation": {"semantic_alignment": "llm", "enable_paid_judge": True,
                       "judges_config": str(ROOT / "configs/judges.yaml")},
    }))
    settings = yaml.safe_load(private.read_text())
    settings["agent_config"] = str(local_config.resolve())
    settings["api_keys"]["deepseek"] = ""
    private.write_text(yaml.safe_dump(settings))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert launcher().main(["--settings", str(private)]) == 0
    assert len(infer) == 35 and {payload["model"] for payload in infer} == {"local-qwen"}
    assert len(judges) == 12
    assert {model for model, _ in judges} == {"qwen3.8-flash", "minimax-m3.1-flash", "gemini-3.1-flash-lite"}
    predictions = [json.loads(line) for line in (tmp_path / "results" / "predictions.jsonl").read_text().splitlines()]
    assert {prediction["task"] for prediction in predictions} == {"T1", "T2", "T3", "T4"}
    assert all(prediction["status"] == "COMPLETED" for prediction in predictions)
    report = json.loads((tmp_path / "results" / "judge_summary.json").read_text())
    assert report["judging_complete"] is True


def test_failed_judge_does_not_become_zero_or_three_judge_mean(tmp_path, monkeypatch):
    rows, infer, judges = fixture_transport(monkeypatch, bad_minimax=True)
    _, private, _ = private_settings(tmp_path, rows)
    assert launcher().main(["--settings", str(private)]) == 1
    report = json.loads((tmp_path / "results/judge_summary.json").read_text())
    metric = report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]
    assert metric["mean"] is None and metric["valid_judges"] == 2
    assert metric["available_mean"] == 1
    assert report["judging_complete"] is False
    assert len([item for item in judges if item[0] == "minimax-m3.1-flash"]) == 1


def test_multiple_absolute_inputs_keep_same_ids_isolated_and_cache_independently(tmp_path, monkeypatch):
    rows, infer, judges = fixture_transport(monkeypatch)
    source, private, _ = private_settings(tmp_path, rows)
    second = tmp_path / "second" / "dataset.json"
    second.parent.mkdir()
    other_rows = json.loads(json.dumps(rows))
    for row in other_rows:
        # 原有证据偏移保持有效；文件之间同 ID、不同正文，不能误复用其他文件的缓存。
        row["document"]["text"] += "这是另一份独立的测试材料。"
    second.write_text(json.dumps(other_rows, ensure_ascii=False))
    settings = yaml.safe_load(private.read_text())
    settings.pop("input")
    settings["inputs"] = [str(source.resolve()), str(second.resolve())]
    private.write_text(yaml.safe_dump(settings))

    runner = launcher()
    assert runner.main(["--settings", str(private)]) == 0
    assert len(infer) == 70 and len(judges) == 24
    batch = json.loads((tmp_path / "results" / "batch_manifest.json").read_text())
    jobs = batch["datasets"]
    assert len(jobs) == 2 and all(job["exit_code"] == 0 for job in jobs)
    assert len({job["run_dir"] for job in jobs}) == 2
    fingerprints = set()
    for job, expected_path in zip(jobs, (source, second), strict=True):
        output = Path(job["run_dir"])
        assert output == runner.dataset_run_dir(tmp_path / "results", expected_path)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["input_path"] == str(expected_path.resolve())
        fingerprints.add(manifest["input_fingerprint"])
        report = json.loads((output / "judge_summary.json").read_text())
        assert report["judging_complete"] is True
        assert report["per_task"]["T1"]["metrics"]["aspect_micro_f1"]["mean"] == pytest.approx(5 / 6)
        judge_results = json.loads((output / "judges" / "results.json").read_text())
        assert judge_results["request_stats"]["model_calls"] == 12
        assert judge_results["request_stats"]["completed_calls"] == 12
    assert len(fingerprints) == 2

    assert runner.main(["--settings", str(private), "--evaluate-only"]) == 0
    assert len(infer) == 70 and len(judges) == 24


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

