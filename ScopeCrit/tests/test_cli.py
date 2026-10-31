"""无网络端到端 CLI 验收；所有输入复制到临时目录，真实数据保持不变。"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys

import pytest
import yaml

from scopecrit import cli


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Mock 不允许访问网络，也不依赖开发机上的真实 API 密钥。"""
    def reject_network(*args, **kwargs):
        raise AssertionError("离线 CLI 测试不允许网络请求")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    input_path = tmp_path / "input.json"
    input_path.write_bytes((ROOT / "examples/merged_demo.json").read_bytes())
    config_path = tmp_path / "mock.yaml"
    config = {"fixture": True, "provider": {"kind": "mock"},
              "mock": {"script": str(ROOT / "examples/mock_script.json")}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"input": input_path, "config": config_path, "run": tmp_path / "run"}


def execute(workspace):
    return cli.main(["run", "--input", str(workspace["input"]),
                     "--config", str(workspace["config"]), "--run-dir", str(workspace["run"])])


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def predictions(workspace):
    return [json.loads(line) for line in (workspace["run"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def database_counts(workspace):
    with sqlite3.connect(workspace["run"] / "state.sqlite3") as connection:
        return {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("episodes", "events", "messages", "model_calls", "predictions")}


def test_mock_full_run_produces_four_tasks_and_debug_logs(workspace, capsys):
    assert execute(workspace) == 0
    output = capsys.readouterr().out
    assert "[准备]" in output and "[动作]" in output and "[模型]" in output
    assert all(task in output for task in ("Demo:T1", "Demo:T2", "Demo:T3", "Demo:T4"))
    rows = predictions(workspace)
    assert {row["task"] for row in rows} == {"T1", "T2", "T3", "T4"}
    assert all(row["status"] == "COMPLETED" for row in rows)
    for filename in ("manifest.json", "state.sqlite3", "input_validation.json", "progress.json",
                     "predictions.jsonl", "traces.jsonl", "messages.jsonl", "failures.jsonl", "metrics.json",
                     "alignment_template.json"):
        assert (workspace["run"] / filename).is_file()
    metrics = json_file(workspace["run"] / "metrics.json")
    assert metrics["overall"]["evidence_f1"] == 1.0
    assert metrics["overall"]["needs_alignment_count"] == 4
    for task in ("T1", "T2", "T3", "T4"):
        assert metrics["per_task"][task]["claim_coverage"] is None
    assert metrics["per_task"]["T3"]["memory_state_f1"] is None
    assert metrics["per_task"]["T4"]["task_success_rate"] is None


def test_manual_fixture_alignment_scores_all_tasks(workspace):
    assert execute(workspace) == 0
    assert cli.main(["evaluate", "--run-dir", str(workspace["run"]), "--gold", str(workspace["input"]),
                     "--alignment", str(ROOT / "examples/alignment_demo.json")]) == 0
    task = json_file(workspace["run"] / "metrics.json")["per_task"]
    assert task["T1"]["aspect_micro_f1"] == 1.0
    assert task["T1"]["stance_macro_f1"] == 1.0
    assert task["T2"]["claim_coverage"] == 1.0
    assert task["T3"]["memory_state_f1"] == 1.0
    assert task["T3"]["false_overwrite"] == 0.0
    assert task["T4"]["expected_exchange_recall"] == 1.0
    assert task["T4"]["task_success_rate"] == 1.0


def test_export_inspect_and_resume_reuse_completed_results(workspace, capsys):
    assert execute(workspace) == 0
    original = predictions(workspace)
    before = database_counts(workspace)
    for command in ("export", "inspect", "resume"):
        assert cli.main([command, "--run-dir", str(workspace["run"])]) == 0
    assert database_counts(workspace) == before
    assert predictions(workspace) == original
    assert "不确定调用" in capsys.readouterr().out


def test_malformed_rows_are_isolated_and_valid_tasks_continue(workspace):
    rows = json_file(workspace["input"])
    rows[0]["gold"] = None
    rows.insert(0, None)
    workspace["input"].write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert execute(workspace) == 1
    results = {p["sample_id"]: p for p in predictions(workspace)}
    assert results["__invalid_0"]["status"] == "INPUT_INVALID"
    assert results["Demo:T1"]["status"] == "INPUT_INVALID"
    assert all(results[f"Demo:{task}"]["status"] == "COMPLETED" for task in ("T2", "T3", "T4"))
    assert len(json_file(workspace["run"] / "input_validation.json")["invalid"]) == 2


def test_duplicate_ids_abort_before_any_run_files(workspace, capsys):
    rows = json_file(workspace["input"])
    workspace["input"].write_text(json.dumps(rows + [rows[0]]), encoding="utf-8")
    assert execute(workspace) == 2
    assert "sample_id 重复" in capsys.readouterr().err
    assert not workspace["run"].exists()


@pytest.mark.parametrize("change", ["input", "config", "source"])
def test_resume_refuses_changed_fingerprints(workspace, monkeypatch, capsys, change):
    assert execute(workspace) == 0
    before = database_counts(workspace)
    if change == "input":
        with workspace["input"].open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "config":
        config = yaml.safe_load(workspace["config"].read_text())
        config["runtime"] = {"seed": 123}
        workspace["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    else:
        # 模拟当前代码计算出了新指纹；不在测试过程中修改实际源文件。
        original = cli.make_manifest

        def changed_manifest(*args, **kwargs):
            manifest = original(*args, **kwargs)
            manifest["source_fingerprint"] = "changed-code-fingerprint"
            return manifest

        monkeypatch.setattr(cli, "make_manifest", changed_manifest)
    assert cli.main(["resume", "--run-dir", str(workspace["run"])]) == 2
    assert "RUN_FINGERPRINT_MISMATCH" in capsys.readouterr().err
    assert database_counts(workspace) == before


def test_evaluate_requires_same_input_file_content(workspace, capsys):
    assert execute(workspace) == 0
    other = workspace["input"].with_name("different.json")
    other.write_text(workspace["input"].read_text() + "\n", encoding="utf-8")
    assert cli.main(["evaluate", "--run-dir", str(workspace["run"]), "--gold", str(other)]) == 2
    assert "评分输入与原始运行输入不同" in capsys.readouterr().err


def test_validate_exit_status_and_written_report(workspace):
    report = workspace["input"].with_name("report.json")
    args = ["validate", "--input", str(workspace["input"]), "--report", str(report)]
    assert cli.main(args + ["--fixture"]) == 0
    assert json_file(report)["valid"]
    assert cli.main(args) == 1
    assert not json_file(report)["valid"]


def test_existing_run_directory_requires_resume(workspace, capsys):
    assert execute(workspace) == 0
    before = database_counts(workspace)
    assert execute(workspace) == 2
    assert "请使用 resume" in capsys.readouterr().err
    assert database_counts(workspace) == before


def test_module_entrypoint_runs_outside_checkout(workspace, tmp_path):
    # 子进程验证 python -m scopecrit 的真实退出码和已安装包发现。
    completed = subprocess.run(
        [sys.executable, "-m", "scopecrit", "validate", "--input", str(workspace["input"]),
         "--report", str(tmp_path / "subprocess_report.json"), "--fixture"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "校验 4 条" in completed.stdout
    assert json_file(tmp_path / "subprocess_report.json")["valid"]


def test_prefix_run_evaluation_and_resume_use_same_selected_rows(workspace):
    args = ["run", "--input", str(workspace["input"]), "--config", str(workspace["config"]),
            "--run-dir", str(workspace["run"]), "--max-samples", "2"]
    assert cli.main(args) == 0
    assert [p["sample_id"] for p in predictions(workspace)] == ["Demo:T1", "Demo:T2"]
    manifest = json_file(workspace["run"] / "manifest.json")
    assert manifest["config"]["runtime"]["max_samples"] == 2
    report = json_file(workspace["run"] / "input_validation.json")
    assert report["source_total"] == 4 and report["total"] == 2
    before = database_counts(workspace)
    assert cli.main(["resume", "--run-dir", str(workspace["run"])]) == 0
    assert database_counts(workspace) == before
    assert cli.main(["evaluate", "--run-dir", str(workspace["run"]),
                     "--gold", str(workspace["input"])]) == 0
    metrics = json_file(workspace["run"] / "metrics.json")
    assert metrics["overall"]["input_row_count"] == 2
    assert metrics["per_task"]["T3"]["count"] == metrics["per_task"]["T4"]["count"] == 0


def test_explicit_none_overrides_yaml_limit_and_processes_all(workspace):
    config = yaml.safe_load(workspace["config"].read_text())
    config["runtime"] = {"max_samples": 1}
    workspace["config"].write_text(yaml.safe_dump(config))
    args = ["run", "--input", str(workspace["input"]), "--config", str(workspace["config"]),
            "--run-dir", str(workspace["run"]), "--max-samples", "nOnE"]
    assert cli.main(args) == 0
    assert len(predictions(workspace)) == 4
    assert json_file(workspace["run"] / "manifest.json")["config"]["runtime"]["max_samples"] is None


def test_omitted_limit_uses_yaml_and_bad_tail_does_not_block(workspace):
    rows = json_file(workspace["input"])
    workspace["input"].write_text(json.dumps(rows + [rows[0], None]))
    config = yaml.safe_load(workspace["config"].read_text())
    config["runtime"] = {"max_samples": 1}
    workspace["config"].write_text(yaml.safe_dump(config))
    assert execute(workspace) == 0
    assert len(predictions(workspace)) == 1


@pytest.mark.parametrize("limit", ["0", "-1", "true", "1.5", "invalid"])
def test_bad_cli_limit_aborts_before_any_output(workspace, limit):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--input", str(workspace["input"]), "--max-samples", limit])
    assert exc.value.code == 2
    assert not workspace["run"].exists()


def test_only_input_is_required_and_default_is_deepseek(workspace, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    assert cli.main(["run", "--input", str(workspace["input"])]) == 0
    assert calls[0][0][2] is None  # run() 自动创建目录
    assert Path(calls[0][0][1]).is_file()
    assert yaml.safe_load(Path(calls[0][0][1]).read_text())["provider"]["kind"] == "deepseek"


def test_automatic_output_directory_is_unique():
    first, second = cli.default_run_dir(), cli.default_run_dir()
    assert first != second
    assert first.parent == Path.cwd() / "runs"


def test_api_authentication_failure_stops_batch_and_returns_nonzero(workspace, monkeypatch):
    class Rejected(Exception):
        status_code = 401

    class Provider:
        calls = 0

        def complete(self, stage, request, call_id):
            self.calls += 1
            raise Rejected("untrusted server error")

    provider = Provider()
    monkeypatch.setattr(cli, "make_provider", lambda config: provider)
    assert execute(workspace) == 1
    assert provider.calls == 1
    progress = json_file(workspace["run"] / "progress.json")
    assert progress["statuses"] == {"PARTIAL": 1, "PENDING": 3}
    metrics = json_file(workspace["run"] / "metrics.json")
    assert metrics["overall"]["count"] == 4


def test_config_limit_change_refuses_resume_even_with_original_cli_override(workspace):
    args = ["run", "--input", str(workspace["input"]), "--config", str(workspace["config"]),
            "--run-dir", str(workspace["run"]), "--max-samples", "2"]
    assert cli.main(args) == 0
    config = yaml.safe_load(workspace["config"].read_text())
    config["runtime"] = {"max_samples": 3}
    workspace["config"].write_text(yaml.safe_dump(config))
    assert cli.main(["resume", "--run-dir", str(workspace["run"])]) == 2
