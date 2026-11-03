"""多文件真实运行器集成：限量、同名文件隔离、批次断点与不重跑。"""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from scopecrit import cli
from scopecrit.local_settings import load_local_settings

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("batch_launcher", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_batch(tmp_path):
    inputs = []
    for folder in ("a", "b"):
        path = tmp_path / folder / "dataset.json"
        path.parent.mkdir()
        path.write_text((ROOT / "examples/merged_demo.json").read_text(encoding="utf-8"), encoding="utf-8")
        inputs.append(str(path))
    config = tmp_path / "mock.yaml"
    config.write_text(yaml.safe_dump({"fixture": True, "provider": {"kind": "mock"},
        "mock": {"script": str(ROOT / "examples/mock_script.json")}}), encoding="utf-8")
    settings = tmp_path / "settings.local.yaml"
    settings.write_text(yaml.safe_dump({"inputs": inputs, "agent_config": str(config),
        "max_samples": None, "run_dir": "results"}), encoding="utf-8")
    return settings, inputs


def test_multiple_same_name_files_limit_independent_and_resume(tmp_path):
    settings, inputs = setup_batch(tmp_path)
    app = launcher()
    assert app.main(["--settings", str(settings), "--no-judge", "--max-samples", "2"]) == 0
    root = tmp_path / "results"
    batch = json.loads((root / "batch_manifest.json").read_text())
    assert batch["run_parameters"]["max_samples"] == 2
    dirs = [Path(job["run_dir"]) for job in batch["datasets"]]
    assert len(set(dirs)) == 2
    before = {}
    for directory in dirs:
        predictions = (directory / "predictions.jsonl").read_text()
        assert len(predictions.splitlines()) == 2
        assert all(json.loads(line)["status"] == "COMPLETED" for line in predictions.splitlines())
        before[directory] = (directory / "traces.jsonl").read_text()
    assert app.main(["--settings", str(settings), "--resume", "--no-judge"]) == 0
    for directory in dirs:
        assert (directory / "traces.jsonl").read_text() == before[directory]


def test_resume_keeps_command_line_limit_for_dataset_not_started(tmp_path, monkeypatch):
    settings, _ = setup_batch(tmp_path)
    app = launcher()
    actual = cli.main
    calls = []

    def interrupted(args):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated stop before second dataset")
        return actual(args)

    monkeypatch.setattr(cli, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        app.main(["--settings", str(settings), "--no-judge", "--max-samples=1"])
    root = tmp_path / "results"
    assert len(list(root.glob("*/state.sqlite3"))) == 1
    monkeypatch.setattr(cli, "main", actual)
    assert app.main(["--settings", str(settings), "--resume", "--no-judge"]) == 0
    assert [len(p.read_text().splitlines()) for p in root.glob("*/predictions.jsonl")] == [1, 1]


def test_invalid_later_path_stops_before_first_model_or_output(tmp_path, monkeypatch):
    settings, _ = setup_batch(tmp_path)
    values = yaml.safe_load(settings.read_text())
    values["inputs"][1] = str(tmp_path / "missing.json")
    settings.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(cli, "main", lambda args: pytest.fail("invalid later input must fail before any API"))
    assert launcher().main(["--settings", str(settings), "--no-judge"]) == 2
    assert not (tmp_path / "results").exists()


def test_multi_paths_and_agent_config_resolve_from_settings(tmp_path, monkeypatch):
    settings = tmp_path / "settings.local.yaml"
    settings.write_text(yaml.safe_dump({"inputs": ["a.json", "b.json"], "agent_config": "configs/actor.yaml"}))
    monkeypatch.chdir(tmp_path.parent)
    loaded = load_local_settings(settings)
    assert loaded["inputs"] == [str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    assert loaded["agent_config"] == str(tmp_path / "configs/actor.yaml")


@pytest.mark.parametrize("paths", [[], [" "], ["a.json", "./a.json"]])
def test_empty_blank_and_duplicate_inputs_rejected(tmp_path, paths):
    settings = tmp_path / "settings.local.yaml"
    settings.write_text(yaml.safe_dump({"inputs": paths}))
    with pytest.raises(ValueError):
        load_local_settings(settings)


def test_input_and_inputs_cannot_both_be_set(tmp_path):
    settings = tmp_path / "settings.local.yaml"
    settings.write_text("input: a.json\ninputs: [b.json]\n")
    with pytest.raises(ValueError):
        load_local_settings(settings)
