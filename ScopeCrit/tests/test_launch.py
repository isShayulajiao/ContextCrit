"""模拟官方 SDK 的 HTTP 交互，验收只填输入路径和 Key 的入口。

所有答案来自明确编写的人工 fixture。MockTransport 不监听端口、不访问网络，
只验证 SDK 序列化、三个模型阶段和运行器之间能衔接，不验证远端模型能力。
"""

import json
from pathlib import Path
import subprocess
import sys

import httpx
import openai

from scopecrit import cli


ROOT = Path(__file__).resolve().parents[1]


def test_direct_script_help_from_other_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--max-samples" in result.stdout


def test_direct_script_missing_key_is_clear_and_creates_no_run(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    rows = json.loads((ROOT / "examples/merged_demo.json").read_text(encoding="utf-8"))
    for row in rows:
        row["material_origin"] = "AUTHENTIC"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--input", str(source), "--no-judge"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert "DEEPSEEK_API_KEY" in result.stderr
    assert not list(tmp_path.rglob("state.sqlite3"))


def test_json_path_and_key_use_real_sdk_serialization_for_all_tasks(tmp_path, monkeypatch):
    """真实 OpenAI SDK + 假 HTTP 响应，检查默认运行不依赖 YAML/输出目录参数。"""
    rows = json.loads((ROOT / "examples/merged_demo.json").read_text(encoding="utf-8"))
    for row in rows:
        row["material_origin"] = "AUTHENTIC"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    script = json.loads((ROOT / "examples/mock_script.json").read_text(encoding="utf-8"))
    expected = [(row["sample_id"], entry) for row in rows for entry in script[row["sample_id"]]]
    captured = []
    fake_key = "sk-FAKE_LOCAL_TRANSPORT_ONLY_CANARY"

    def transport(request):
        sample_id, entry = expected[len(captured)]
        payload = json.loads(request.content)
        assert request.url.host == "api.deepseek.com"
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == f"Bearer {fake_key}"
        assert payload["model"] == "deepseek-flash"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        assert payload["stream"] is False
        user = json.loads(payload["messages"][1]["content"])
        if entry["stage"] == "ACTION":
            assert user["episode_id"] == sample_id
        elif entry["stage"] == "SUPPORT_CHECK":
            assert "candidate" in user and "evidence" in user
        else:
            assert "claims" in user and "evidence" in user
        assert "gold" not in user
        captured.append(payload)
        response = entry["response"]
        return httpx.Response(200, json={
            "id": f"test-response-{len(captured)}", "object": "chat.completion", "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(response, ensure_ascii=False),
                "reasoning_content": "PRIVATE_REASONING_NOT_FOR_LOGS_CANARY"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    actual_openai = openai.OpenAI

    def local_openai(**kwargs):
        assert kwargs["max_retries"] == 0
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(transport))
        return actual_openai(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", local_openai)
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_key)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--input", str(source), "--no-judge"]) == 0
    assert len(captured) == len(expected) == 35
    databases = list(tmp_path.rglob("state.sqlite3"))
    assert len(databases) == 1
    run_dir = databases[0].parent
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 4
    assert all(p["status"] == "COMPLETED" for p in predictions)
    for filename in ("manifest.json", "traces.jsonl", "predictions.jsonl", "messages.jsonl"):
        output = (run_dir / filename).read_text()
        assert fake_key not in output
        assert "PRIVATE_REASONING_NOT_FOR_LOGS_CANARY" not in output

