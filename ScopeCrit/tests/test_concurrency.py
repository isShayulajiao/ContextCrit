"""线程屏障验证实际重叠，测试 SQLite、缓存与单样本顺序；不访问 API。"""
import json
import importlib.util
from pathlib import Path
import sqlite3
from threading import Barrier, Event, Lock, get_ident

import pytest
import yaml

from scopecrit import cli, judging
from scopecrit.config import Config
from scopecrit.judge_config import JudgesConfig
from scopecrit.local_settings import LocalSettings
from scopecrit.parallel import bounded_results
from scopecrit.providers import MockProvider
from scopecrit.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def test_real_sdk_deepseek_four_workers_all_tasks(tmp_path, monkeypatch):
    import httpx
    import openai
    from scopecrit.providers import prompt_text

    source, config = mock_input(tmp_path)
    config.write_text(yaml.safe_dump({"fixture": True, "provider": {"kind": "deepseek"}}))
    script = json.loads((ROOT / "examples/mock_script.json").read_text())
    barrier, guard = Barrier(4), Lock()
    active, positions, observed = {}, {}, []
    fake_key = "FAKE_CONCURRENCY_SDK_KEY_CANARY"

    def handle(request):
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        thread = get_ident()
        with guard:
            if "episode_id" in user:
                active[thread] = user["episode_id"]
            sid = active[thread]
            position = positions.get(sid, 0)
            positions[sid] = position + 1
            observed.append((thread, sid))
        entry = script[sid][position]
        assert payload["messages"][0]["content"].startswith(prompt_text(entry["stage"]))
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["stream"] is False
        assert "gold" not in user
        if position == 0:
            barrier.wait(timeout=10)
        return httpx.Response(200, json={"id": f"{sid}-{position}", "object": "chat.completion", "created": 0,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(entry["response"], ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})

    actual = openai.OpenAI

    def client(**kwargs):
        return actual(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_key)
    output = tmp_path / "sdk"
    assert cli.main(["run", "--input", str(source), "--config", str(config), "--run-dir", str(output),
                     "--workers", "4", "--no-judge"]) == 0
    assert len(observed) == 35 and len({thread for thread, sid in observed}) == 4
    results = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(results) == 4 and all(row["status"] == "COMPLETED" for row in results)
    for filename in ("manifest.json", "traces.jsonl", "predictions.jsonl"):
        assert fake_key not in (output / filename).read_text()


def mock_input(tmp_path):
    source = tmp_path / "input.json"
    source.write_bytes((ROOT / "examples/merged_demo.json").read_bytes())
    config = tmp_path / "mock.yaml"
    config.write_text(yaml.safe_dump({"fixture": True, "provider": {"kind": "mock"},
        "mock": {"script": str(ROOT / "examples/mock_script.json")}}), encoding="utf-8")
    return source, config


def test_actor_requests_overlap_all_tasks_and_resume_without_replay(tmp_path, monkeypatch):
    source, config = mock_input(tmp_path)
    barrier, guard = Barrier(4), Lock()
    threads, calls = set(), []

    class ConcurrentMock(MockProvider):
        def complete(self, stage, request, call_id):
            with guard:
                threads.add(get_ident())
                calls.append((stage, call_id))
            if int(call_id.rsplit(":", 1)[1]) == 1:
                barrier.wait(timeout=10)
            return super().complete(stage, request, call_id)

    monkeypatch.setattr(cli, "make_provider", lambda options: ConcurrentMock(options["mock"]["script"]))
    output = tmp_path / "run"
    assert cli.run(source, config, output, no_judge=True, workers=4) == 0
    assert len(threads) == 4 and len(calls) == 35
    before = (output / "traces.jsonl").read_text()
    results = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert {row["task"] for row in results} == {"T1", "T2", "T3", "T4"}
    assert all(row["status"] == "COMPLETED" for row in results)
    t3 = next(row for row in results if row["task"] == "T3")
    assert [turn["turn"] for turn in t3["turns"]] == [1, 2]
    with sqlite3.connect(output / "state.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT count(*) FROM model_calls").fetchone()[0] == 35
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
    assert cli.main(["resume", "--run-dir", str(output), "--no-judge"]) == 0
    assert len(calls) == 35
    assert (output / "traces.jsonl").read_text() == before


def test_three_judges_overlap_with_independent_cache(tmp_path, monkeypatch):
    source, config = mock_input(tmp_path)
    output = tmp_path / "run"
    assert cli.run(source, config, output, no_judge=True) == 0
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    samples = json.loads(source.read_text())
    annotations = json.loads((ROOT / "examples/alignment_demo.json").read_text())
    barrier, guard = Barrier(3), Lock()
    seen = []

    class Judge:
        def __init__(self, name):
            self.name, self.first = name, True

        def complete(self, prompt, request, call_id):
            if self.first:
                self.first = False
                barrier.wait(timeout=10)
            sid = request["targets"][0]["sample_id"]
            with guard:
                seen.append((self.name, sid, get_ident()))
            return {"content": json.dumps({"alignments": [a for a in annotations if a["sample_id"] == sid]}),
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}

    monkeypatch.setattr(judging, "make_judge_provider", lambda judge: Judge(judge["name"]))
    options = {"enabled": True, "fixture": True, "workers": 3,
               "judges": [{"name": name, "model": "fixture", "base_url": "https://fake.invalid"}
                          for name in ("qwen", "minimax", "gemini")]}
    first = judging.run_judges(predictions, samples, output, options, log=lambda _: None)
    assert first["complete"] and len(seen) == 12
    assert len({thread for _, _, thread in seen}) == 3
    assert first["request_stats"]["model_calls"] == 12
    for judge in first["per_judge"].values():
        assert judge["completed_sample_count"] == 4
    with sqlite3.connect(output / "judges/state.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    second = judging.run_judges(predictions, samples, output, options, log=lambda _: None)
    assert second["complete"] and len(seen) == 12


def test_bounded_scheduler_stops_new_jobs_and_drains_started_jobs():
    barrier, guard = Barrier(2), Lock()
    started = []

    def execute(number):
        with guard:
            started.append(number)
        barrier.wait(timeout=10)
        return number

    result = list(bounded_results(execute, range(20), workers=2, should_stop=lambda _: True))
    assert set(started) == {0, 1}
    assert {item for item, value in result} == {0, 1}


def test_fatal_actor_error_keeps_unstarted_samples_pending(tmp_path, monkeypatch):
    source, config = mock_input(tmp_path)
    barrier, released = Barrier(2), Event()
    real_export = Store.export

    class Rejected(Exception):
        status_code = 401

    class FailingMock(MockProvider):
        def complete(self, stage, request, call_id):
            if int(call_id.rsplit(":", 1)[1]) == 1:
                barrier.wait(timeout=10)
                if ":T1:" in call_id:
                    raise Rejected("secret must not enter output")
                assert released.wait(timeout=10)
            return super().complete(stage, request, call_id)

    def export(store):
        result = real_export(store)
        released.set()
        return result

    monkeypatch.setattr(cli, "make_provider", lambda options: FailingMock(options["mock"]["script"]))
    monkeypatch.setattr(Store, "export", export)
    output = tmp_path / "run"
    assert cli.run(source, config, output, no_judge=True, workers=2) == 1
    with sqlite3.connect(output / "state.sqlite3") as connection:
        states = dict(connection.execute("SELECT sample_id,status FROM episodes"))
    assert states == {"Demo:T1": "PARTIAL", "Demo:T2": "COMPLETED", "Demo:T3": "PENDING", "Demo:T4": "PENDING"}


def test_batch_resume_keeps_cli_worker_override_for_unstarted_file(tmp_path, monkeypatch):
    source, config = mock_input(tmp_path)
    second = tmp_path / "second.json"
    second.write_bytes(source.read_bytes())
    settings = tmp_path / "settings.local.yaml"
    values = {"inputs": [str(source), str(second)], "agent_config": str(config),
              "actor_workers": 2, "run_dir": str(tmp_path / "batch")}
    settings.write_text(yaml.safe_dump(values))
    spec = importlib.util.spec_from_file_location("concurrent_launcher", ROOT / "run.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    actual, calls = cli.main, []

    def interrupted(args):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated interruption before file two")
        return actual(args)

    monkeypatch.setattr(cli, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        app.main(["--settings", str(settings), "--max-samples", "1", "--workers", "3", "--no-judge"])
    values["actor_workers"] = 1
    settings.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(cli, "main", actual)
    assert app.main(["--settings", str(settings), "--resume", "--no-judge"]) == 0
    manifests = list((tmp_path / "batch").glob("*/manifest.json"))
    assert len(manifests) == 2
    assert all(json.loads(path.read_text())["config"]["runtime"]["concurrency"] == 3 for path in manifests)


@pytest.mark.parametrize("number", [0, -1, True, "4", 33])
def test_actor_concurrency_config_rejects_invalid_values(number):
    with pytest.raises(ValueError):
        Config.model_validate({"runtime": {"concurrency": number}})
    with pytest.raises(ValueError):
        LocalSettings.model_validate({"input": "test.json", "actor_workers": number})


@pytest.mark.parametrize("number", [0, -1, True, "3", 17])
def test_judge_concurrency_config_rejects_invalid_values(number):
    with pytest.raises(ValueError):
        JudgesConfig.model_validate({"workers": number,
            "judges": [{"name": "fixture", "model": "fixture", "base_url": "https://fake.invalid", "api_key_env": "KEY"}]})
