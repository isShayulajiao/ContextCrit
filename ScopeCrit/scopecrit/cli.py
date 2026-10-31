"""命令行入口：校验 → 运行 → 导出 → 独立评分。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from tqdm import tqdm

from . import __version__
from .config import load_config
from .data import fingerprint, load_dataset, sample_key
from .engine import Engine, new_state
from .evaluation import alignment_template, evaluate_predictions
from .providers import make_provider
from .storage import Store


UNSET = object()
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "deepseek.yaml"
DEFAULT_JUDGES = DEFAULT_CONFIG.parent / "judges.yaml"


def parse_max_samples(value):
    """命令行允许 --max-samples None 或一个严格正整数。"""
    if value.lower() == "none":
        return None
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("处理条数必须是正整数或 None") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("处理条数必须是正整数或 None")
    return result


def default_run_dir():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / "runs" / f"deepseek_{stamp}_{uuid4().hex[:8]}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def make_manifest(input_path, config_path, config):
    # 提示词、代码、输入、Mock 脚本任一变化，都不能接着旧状态悄悄续跑。
    package = Path(__file__).parent
    sources = {str(p.relative_to(package)): p.read_text(encoding="utf-8")
               for pattern in ("*.py", "prompts/*.txt") for p in sorted(package.glob(pattern))}
    script = config["mock"]["script"]
    return {"schema_version": "1.0", "version": __version__,
            "input_path": str(Path(input_path).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "input_fingerprint": fingerprint(Path(input_path).read_text(encoding="utf-8")),
            "config_file_fingerprint": fingerprint(Path(config_path).read_text(encoding="utf-8")),
            "config": config, "source_fingerprint": fingerprint(sources),
            "mock_fingerprint": fingerprint(Path(script).read_text(encoding="utf-8")) if script else None}


def score(store, rows, run_dir, alignment_path=None):
    predictions = store.list_predictions()
    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8")) if alignment_path else None
    usable_rows = [r for r in rows if isinstance(r, dict) and
                   isinstance(r.get("sample_id"), str) and r.get("task") in ("T1", "T2", "T3", "T4")]
    usable_ids = {row["sample_id"] for row in usable_rows}
    # 没有合法 ID/任务类型的条目不能进行分任务评分，但隔离数量仍要呈现。
    scored_predictions = [p for p in predictions if p["sample_id"] in usable_ids]
    metrics = evaluate_predictions(scored_predictions, usable_rows, alignment)
    metrics["overall"]["input_row_count"] = len(rows)
    metrics["overall"]["unassignable_invalid_count"] = len(rows) - len(usable_rows)
    write_json(Path(run_dir) / "metrics.json", metrics)
    if alignment is None:
        write_json(Path(run_dir) / "alignment_template.json", alignment_template(scored_predictions))
    return metrics


def judge_configuration(config, override=None, force=False):
    """独立评分配置，不把裁判密钥写入答题 manifest。"""
    evaluation = config.get("evaluation", {})
    if not (force or override or (evaluation.get("enable_paid_judge") and evaluation.get("semantic_alignment") == "llm")):
        return None
    from .judge_config import load_judges
    result = load_judges(override or evaluation.get("judges_config") or DEFAULT_JUDGES)
    if not result["enabled"]:
        raise ValueError("裁判配置 enabled=false；请启用评分或显式使用 --no-judge")
    return result


def automatic_score(store, rows, run_dir, judge_config, structural, log=tqdm.write, retry_uncertain=False):
    from .judging import run_judges
    from .reporting import build_report, format_console, write_report
    usable_rows = [r for r in rows if isinstance(r, dict) and isinstance(r.get("sample_id"), str)
                   and r.get("task") in ("T1", "T2", "T3", "T4")]
    ids = {row["sample_id"] for row in usable_rows}
    predictions = [p for p in store.list_predictions() if p["sample_id"] in ids]
    result = run_judges(predictions, usable_rows, run_dir, judge_config, log=log,
                        retry_uncertain=retry_uncertain)
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    report = build_report(result["per_judge"], structural, Path(manifest["input_path"]).stem,
                          [judge["name"] for judge in judge_config["judges"]])
    report["judging_complete"] = result["complete"]
    report["judge_request_stats"] = result.get("request_stats", {})
    write_report(report, run_dir)
    log(format_console(report))
    log("[裁判汇总] 各裁判、各任务和逐样本指标：judge_summary.json / metrics_summary.csv / per_sample_metrics.csv / report.md")
    log("[裁判汇总] 四裁判均完成才给出正式均值；缺失项为 null，另列可用结果及评分覆盖率")
    return result["complete"]


def run(input_path, config_path, run_dir=None, resume=False, retry_uncertain=False, max_samples=UNSET,
        judges_config=None, no_judge=False, retry_uncertain_judges=False):
    config = load_config(config_path)
    if max_samples is not UNSET:
        if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
            raise ValueError("max_samples 必须是正整数或 None")
        config["runtime"]["max_samples"] = max_samples
    selected_limit = config["runtime"]["max_samples"]
    tqdm.write("[准备] 读取配置并选择待处理记录；不拆分 train/test")
    rows, report = load_dataset(input_path, fixture=config["fixture"], max_samples=selected_limit)
    tqdm.write(f'[选择] 原始 {report["source_total"]} 条；本次处理 {len(rows)} 条；max_samples={selected_limit}')
    run_dir = Path(run_dir) if run_dir is not None else default_run_dir()
    if not resume and (run_dir / "state.sqlite3").exists():
        raise ValueError("运行目录已存在；请使用 resume 或指定新的 --run-dir")
    if config["provider"]["kind"] == "mock" and not config["fixture"]:
        raise ValueError("Mock 只验证人工 fixture 流程；请使用 fixture: true，真实数据请选择 DeepSeek")
    judges = None if no_judge else judge_configuration(config, judges_config)
    if judges is not None:
        from .judge_config import preflight_credentials
        missing = [item for item in preflight_credentials(judges) if item["status"] != "ready"]
        if missing:
            descriptions = [item["judge"] + ":" + ",".join(item.get("missing") or [item.get("error_code", "配置无效")]) for item in missing]
            raise ValueError("裁判配置尚未填写完整：" + "; ".join(descriptions) +
                             "。请填写 settings.local.yaml；仅需答题可使用 --no-judge，已有预测可单独 evaluate --auto-judge")
    # 在创建运行目录前发现缺少 API 密钥、脚本等配置错误。
    provider = make_provider(config)
    store = Store(run_dir)
    try:
        store.initialize(make_manifest(input_path, config_path, config))
        write_json(run_dir / "input_validation.json", report)
        for index, row in enumerate(rows):
            key = sample_key(row, index)
            if key not in report["invalid"] and store.get_state(key) is None:
                state = new_state(row)
                state["status"] = "PENDING"
                store.save_state(key, state)
        tqdm.write(f'[校验] 总计 {report["total"]} 条，无效 {len(report["invalid"])} 条')
        for index, row in enumerate(tqdm(rows, desc="数据集进度", unit="条", dynamic_ncols=True)):
            key = sample_key(row, index)
            if key in report["invalid"]:
                state = {"sample_id": key, "doc_id": row.get("doc_id") if isinstance(row, dict) else None,
                         "task": row.get("task") if isinstance(row, dict) else None,
                         "status": "INPUT_INVALID", "termination_reason": report["invalid"][key],
                         "usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0, "usage_complete": True}}
                prediction = {**state, "prediction": {"claims": [], "answer": "", "evidence_ids": [], "relations": []},
                              "turns": [], "messages": [], "access_violations": []}
                store.save_state(key, state, prediction=prediction)
                tqdm.write(f"[隔离] {key}: {'; '.join(report['invalid'][key])}")
                continue
            engine = Engine(row, config, provider, store, log=tqdm.write, retry_uncertain=retry_uncertain)
            result = engine.run()
            tqdm.write(f'[结果] {key} → {result["status"]} ({result["termination_reason"]})')
            store.export()
            if result["termination_reason"] in ("API_400", "API_401", "API_403"):
                tqdm.write("[停止批次] API 请求或认证被拒绝；其余记录保留 PENDING，请检查模型参数与密钥")
                break
            if config["runtime"]["fail_fast"] and result["status"] != "COMPLETED":
                break
        store.export()
        structural = score(store, rows, run_dir)
        judging_complete = True
        if judges is not None:
            judging_complete = automatic_score(store, rows, run_dir, judges, structural,
                                                retry_uncertain=retry_uncertain_judges)
        tqdm.write(f"[完成] 预测、轨迹、消息和分任务指标已写入 {run_dir}")
        if judges is None:
            tqdm.write("[评分] 本次未使用语义裁判；结构指标见 metrics.json，可随后单独自动评分")
        return 0 if rows and judging_complete and all(state["status"] == "COMPLETED" for state in store.list_states()) else 1
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ScopeCrit：单 JSON 输入的 T1–T4 文学评论 Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="只读校验输入")
    validate.add_argument("--input", required=True)
    validate.add_argument("--report", required=True)
    validate.add_argument("--fixture", action="store_true", help="允许明确标记 SYNTHETIC 的演示数据")
    validate.add_argument("--max-samples", type=parse_max_samples, default=None, help="只校验前 N 条；None 校验全部")
    execute = sub.add_parser("run", help="执行并自动导出、评分")
    execute.add_argument("--input", required=True)
    execute.add_argument("--config", default=str(DEFAULT_CONFIG), help="默认使用 DeepSeek，无需填写配置路径")
    execute.add_argument("--run-dir", help="省略时自动创建唯一的 runs/deepseek_* 输出目录")
    execute.add_argument("--max-samples", type=parse_max_samples, default=UNSET,
                         help="处理输入顺序的前 N 条；None 处理全部；省略时使用配置")
    execute.add_argument("--judges-config", help="独立裁判配置，指定即启用自动评分")
    execute.add_argument("--no-judge", action="store_true", help="本次只生成预测和程序结构指标，不调用裁判")
    execute.add_argument("--retry-uncertain-judges", action="store_true", help="显式重试不确定裁判请求，可能重复计费")
    resume_parser = sub.add_parser("resume", help="从 SQLite 状态续跑")
    resume_parser.add_argument("--run-dir", required=True)
    resume_parser.add_argument("--retry-uncertain", action="store_true", help="显式接受不确定调用可能重复计费")
    resume_parser.add_argument("--judges-config")
    resume_parser.add_argument("--no-judge", action="store_true")
    resume_parser.add_argument("--retry-uncertain-judges", action="store_true")
    for name in ("export", "inspect", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name == "evaluate":
            command.add_argument("--gold", required=True, help="同一个输入 JSON，不需要单独拆分 gold 文件")
            command.add_argument("--alignment", help="独立人工语义对齐 JSON")
            command.add_argument("--auto-judge", action="store_true", help="用配置中的独立模型自动对齐与评分")
            command.add_argument("--judges-config", help="指定即启用裁判，不依赖答题时的评分模式")
            command.add_argument("--no-judge", action="store_true", help="只计算结构指标或人工对齐指标")
            command.add_argument("--retry-uncertain-judges", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            _, report = load_dataset(args.input, fixture=args.fixture, max_samples=args.max_samples)
            write_json(args.report, report)
            print(f'校验 {report["total"]} 条；无效 {len(report["invalid"])} 条；报告：{args.report}')
            return 0 if report["valid"] else 1
        if args.command == "run":
            if args.no_judge and args.judges_config:
                raise ValueError("--no-judge 与 --judges-config 不能同时使用")
            return run(args.input, args.config, args.run_dir, max_samples=args.max_samples,
                       judges_config=args.judges_config, no_judge=args.no_judge,
                       retry_uncertain_judges=args.retry_uncertain_judges)
        manifest = json.loads((Path(args.run_dir) / "manifest.json").read_text(encoding="utf-8"))
        if args.command == "resume":
            return run(manifest["input_path"], manifest["config_path"], args.run_dir, True, args.retry_uncertain,
                       max_samples=manifest["config"]["runtime"]["max_samples"],
                       judges_config=args.judges_config, no_judge=args.no_judge,
                       retry_uncertain_judges=args.retry_uncertain_judges)
        store = Store(args.run_dir)
        try:
            if args.command == "export":
                print(json.dumps(store.export(), ensure_ascii=False, indent=2))
            elif args.command == "inspect":
                print(json.dumps(store.export(), ensure_ascii=False, indent=2))
                print(f"不确定调用：{len(store.uncertain_calls())}")
            else:
                rows, _ = load_dataset(args.gold, fixture=manifest["config"]["fixture"],
                                       max_samples=manifest["config"]["runtime"]["max_samples"])
                if fingerprint(Path(args.gold).read_text(encoding="utf-8")) != manifest["input_fingerprint"]:
                    raise ValueError("评分输入与原始运行输入不同；请使用同一个 JSON 文件")
                metrics = score(store, rows, args.run_dir, args.alignment)
                if args.alignment and (args.auto_judge or args.judges_config):
                    raise ValueError("人工对齐与自动裁判应分开执行，不能同时指定")
                if args.no_judge and (args.auto_judge or args.judges_config):
                    raise ValueError("--no-judge 与自动裁判选项不能同时指定")
                judges = None if args.no_judge or args.alignment else judge_configuration(
                    manifest["config"], args.judges_config, args.auto_judge)
                if judges is not None:
                    return 0 if automatic_score(store, rows, args.run_dir, judges, metrics,
                                                retry_uncertain=args.retry_uncertain_judges) else 1
                print(json.dumps(metrics["per_task"], ensure_ascii=False, indent=2))
        finally:
            store.close()
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        # SDK 调用异常已在 Engine 转成脱敏错误码；此处只呈现本地配置错误。
        print(f"[错误] {exc}", file=sys.stderr)
        return 2


def entrypoint():
    sys.exit(main())
