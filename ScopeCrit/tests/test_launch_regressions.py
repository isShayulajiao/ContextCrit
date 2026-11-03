"""输入编码与整批预检的运行器回归检查。"""
import importlib.util
from pathlib import Path

import pytest
import yaml

from scopecrit import cli

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("launch_regressions", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_config(tmp_path):
    path = tmp_path / "mock.yaml"
    path.write_text(yaml.safe_dump({"fixture": True, "provider": {"kind": "mock"},
        "mock": {"script": str(ROOT / "examples/mock_script.json")}}), encoding="utf-8")
    return str(path)


def test_utf8_bom_dataset_runs_after_passing_launcher_preflight(tmp_path):
    data = tmp_path / "bom_dataset.json"
    data.write_text((ROOT / "examples/merged_demo.json").read_text(encoding="utf-8"),
                    encoding="utf-8-sig")
    private = tmp_path / "settings.local.yaml"
    private.write_text(yaml.safe_dump({"input": str(data), "agent_config": mock_config(tmp_path),
        "max_samples": 1, "run_dir": "results"}), encoding="utf-8")
    assert launcher().main(["--settings", str(private), "--no-judge"]) == 0
    assert len((tmp_path / "results/predictions.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_later_dataset_rejected_before_any_job(tmp_path, monkeypatch, constant):
    first = tmp_path / "first.json"
    first.write_text("[]", encoding="utf-8")
    second = tmp_path / "second.json"
    second.write_text("[" + constant + "]", encoding="utf-8")
    private = tmp_path / "settings.local.yaml"
    private.write_text(yaml.safe_dump({"inputs": [str(first), str(second)],
        "agent_config": mock_config(tmp_path), "run_dir": "results"}), encoding="utf-8")
    monkeypatch.setattr(cli, "main", lambda args: pytest.fail("整批预检应在执行首份数据前拒绝非标准 JSON"))
    assert launcher().main(["--settings", str(private), "--no-judge"]) == 2
    assert not (tmp_path / "results").exists()


def test_evaluate_only_does_not_recreate_missing_state_database(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("[]", encoding="utf-8")
    output = tmp_path / "results"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    private = tmp_path / "settings.local.yaml"
    private.write_text(yaml.safe_dump({"input": str(source), "run_dir": str(output)}), encoding="utf-8")
    monkeypatch.setattr(cli, "main", lambda args: pytest.fail("丢失状态库不能启动评分"))
    assert launcher().main(["--settings", str(private), "--evaluate-only"]) == 2
    assert not (output / "state.sqlite3").exists()
