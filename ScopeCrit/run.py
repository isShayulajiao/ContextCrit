"""用户入口：私有配置、多数据集批次、独立评分与断点恢复。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def dataset_run_dir(root, input_path):
    """同名数据文件也拥有不同输出；与列表顺序无关。"""
    path = Path(input_path).resolve()
    suffix = sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return Path(root) / f"{path.stem}_{suffix}"


def reject_nonfinite(value):
    raise ValueError("JSON 不能含 NaN 或 Infinity")


def main(argv=None):
    if sys.version_info < (3, 11):
        print("[错误] ScopeCrit 需要 Python 3.11 或以上。", file=sys.stderr)
        return 2
    from scopecrit.cli import main as cli_main, default_run_dir, write_json, DEFAULT_CONFIG, parse_max_samples
    from scopecrit.config import load_config
    from scopecrit.local_settings import configured_api, load_local_settings

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--settings", help="本地 YAML，默认 settings.local.yaml")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options, remaining = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))

    def supplied(flag):
        return any(item == flag or item.startswith(flag + "=") for item in remaining)

    def take(flag):
        for index, item in enumerate(remaining):
            if item.startswith(flag + "="):
                return remaining.pop(index).split("=", 1)[1]
            if item == flag:
                if index + 1 >= len(remaining) or remaining[index + 1].startswith("--"):
                    raise ValueError("参数缺少取值")
                remaining.pop(index)
                return remaining.pop(index)
        return None

    if supplied("--help") or "-h" in remaining:
        print("复制 settings.example.yaml 为 settings.local.yaml，填写一个或多个数据集、答题模型和三个裁判的 API Key。")
        print("--settings 文件；--evaluate-only 仅评分；--resume 续跑；后两者需指定已有 --run-dir。")
        return cli_main(["run", *remaining])
    settings_path = Path(options.settings) if options.settings else Path(__file__).parent / "settings.local.yaml"
    if not settings_path.exists():
        if options.evaluate_only or options.resume or options.settings or not supplied("--input"):
            print("[错误] 请先复制 settings.example.yaml 为 settings.local.yaml 并填写配置。", file=sys.stderr)
            return 2
        return cli_main(["run", *remaining])
    try:
        settings = load_local_settings(settings_path)
    except (ValueError, OSError):
        print("[错误] 本地配置读取失败：检查路径、缩进、字段类型、接口地址和业务空间 ID。", file=sys.stderr)
        return 2
    try:
        if options.evaluate_only and options.resume:
            raise ValueError("--evaluate-only 与 --resume 不能同时使用")
        if options.resume and any(supplied(flag) for flag in ("--max-samples", "--config", "--input", "--workers")):
            raise ValueError("续跑必须沿用原数据、模型配置、条数和并发数；请勿指定 --input、--config、--max-samples 或 --workers")
        override_input = take("--input")
        inputs = [str(Path(override_input).expanduser().resolve())] if override_input else settings["inputs"]
        root = take("--run-dir") or settings.get("run_dir")
        if (options.evaluate_only or options.resume) and not root:
            raise ValueError("仅评分或续跑必须填写原运行目录 --run-dir（或配置 run_dir）")
        if not options.resume and not options.evaluate_only:
            if settings.get("actor_workers") is not None and not supplied("--workers"):
                remaining.extend(["--workers", str(settings["actor_workers"])])
            if not supplied("--max-samples"):
                limit = settings["max_samples"]
                remaining.extend(["--max-samples", "None" if limit is None else str(limit)])
            if settings.get("agent_config") and not supplied("--config"):
                remaining.extend(["--config", settings["agent_config"]])
        if settings.get("judges_config") and not supplied("--judges-config") and not supplied("--no-judge"):
            remaining.extend(["--judges-config", settings["judges_config"]])
        if options.evaluate_only and supplied("--no-judge"):
            raise ValueError("--evaluate-only 用于自动裁判评分，不能与 --no-judge 同用")
        multiple = len(inputs) > 1
        if multiple and supplied("--gold"):
            raise ValueError("多数据集评分自动使用各自原始文件；请勿指定 --gold")
        root = Path(root).expanduser().resolve() if root else (default_run_dir() if multiple else None)
        jobs = [{"input": path, "run_dir": str(dataset_run_dir(root, path) if multiple else root) if root else None}
                for path in inputs]
        # 在首次付费调用前检查整批路径和 JSON 容器。
        for job in jobs:
            with Path(job["input"]).open(encoding="utf-8-sig") as file:
                if not isinstance(json.load(file, parse_constant=reject_nonfinite), list):
                    raise ValueError("每个数据集必须是 JSON 数组")
            output = Path(job["run_dir"]) if job["run_dir"] else None
            if options.evaluate_only and not all((output / name).is_file() for name in ("manifest.json", "state.sqlite3")):
                raise ValueError("评分目录缺少 manifest.json 或 state.sqlite3；请使用原运行目录")
            if options.resume and not multiple and not (output / "state.sqlite3").is_file():
                raise ValueError("续跑目录缺少 state.sqlite3；请使用原运行目录")
            if not options.resume and not options.evaluate_only and output and (output / "state.sqlite3").exists():
                raise ValueError("运行目录已存在；请使用 --resume 或新的 --run-dir")
        batch = {"schema_version": "1.0", "datasets": jobs}
        if multiple:
            print(f"[批次] {len(jobs)} 个数据集；条数限制分别应用于每个文件；输出：{root}")
            batch_path = root / "batch_manifest.json"
            if options.resume or options.evaluate_only:
                if not batch_path.is_file():
                    raise ValueError("多文件续跑或评分需要原批次目录的 batch_manifest.json")
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
                if {job["input"] for job in batch["datasets"]} != set(inputs):
                    raise ValueError("数据集清单与原批次不一致")
                jobs = batch["datasets"]
            else:
                # 保留未启动数据集的准确断点；命令行条数也必须继承到后续文件。
                batch_parser = argparse.ArgumentParser(add_help=False)
                batch_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
                batch_parser.add_argument("--max-samples", type=parse_max_samples)
                batch_parser.add_argument("--workers", type=int)
                batch_options, _ = batch_parser.parse_known_args(remaining)
                launch_config = load_config(batch_options.config)
                batch["run_parameters"] = {"config": str(Path(batch_options.config).resolve()),
                                           "max_samples": batch_options.max_samples,
                                           "workers": (batch_options.workers if batch_options.workers is not None
                                                       else launch_config["runtime"]["concurrency"])}
                write_json(batch_path, batch)
        codes = []
        with configured_api(settings):
            for index, job in enumerate(jobs, 1):
                args = list(remaining)
                if job["run_dir"]:
                    args.extend(["--run-dir", job["run_dir"]])
                if multiple:
                    print(f"[数据集 {index}/{len(jobs)}] {job['input']}")
                if options.evaluate_only:
                    if not supplied("--gold"):
                        args.extend(["--gold", job["input"]])
                    code = cli_main(["evaluate", "--auto-judge", *args])
                elif options.resume and (Path(job["run_dir"]) / "state.sqlite3").exists():
                    code = cli_main(["resume", *args])
                else:
                    # 后续数据集尚未创建时，续跑会从该数据集正常开始。
                    if options.resume:
                        parameters = batch.get("run_parameters", {}) if multiple else {}
                        limit = parameters.get("max_samples", settings["max_samples"])
                        args.extend(["--max-samples", "None" if limit is None else str(limit)])
                        config_path = parameters.get("config") or settings.get("agent_config")
                        if config_path:
                            args.extend(["--config", config_path])
                        workers = parameters.get("workers", settings.get("actor_workers"))
                        if workers is not None:
                            args.extend(["--workers", str(workers)])
                    code = cli_main(["run", "--input", job["input"], *args])
                codes.append(code)
                job["evaluation_exit_code" if options.evaluate_only else "exit_code"] = code
                if multiple:
                    write_json(root / "batch_manifest.json", batch)
                if code == 2:
                    break
        return max(codes, default=0)
    except (ValueError, OSError):
        print("[错误] 检查数据文件为 JSON 数组、输出目录及参数组合；续跑/评分需指定原运行目录。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

