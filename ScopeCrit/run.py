"""填写 settings.local.yaml 后直接运行；密钥配置不进入 Git 或运行记录。"""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    if sys.version_info < (3, 11):
        print("[错误] ScopeCrit 需要 Python 3.11 或以上；建议 conda create -n scopecrit python=3.11", file=sys.stderr)
        return 2
    from scopecrit.cli import main as cli_main
    from scopecrit.local_settings import configured_api, load_local_settings

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--settings", help="直接填写 API Key 的本地 YAML 文件，默认 settings.local.yaml")
    parser.add_argument("--evaluate-only", action="store_true", help="只给已有运行评分，不重新生成答案；需 --run-dir")
    options, remaining = parser.parse_known_args(arguments)

    def supplied(flag):
        return any(item == flag or item.startswith(flag + "=") for item in remaining)

    if supplied("--help") or "-h" in remaining:
        print("本地配置：复制 settings.example.yaml 为 settings.local.yaml，填写数据路径、四个 Key 和 Qwen 业务空间 ID。")
        print("其他入口：--settings 文件路径；--evaluate-only --run-dir 已有输出目录（仅评分）。")
        return cli_main(["run", *remaining])
    settings_path = Path(options.settings) if options.settings else Path(__file__).parent / "settings.local.yaml"
    if not settings_path.exists():
        if options.evaluate_only:
            print("[错误] 单独评分需要 --settings 本地配置文件和 --run-dir；不会启动答题。", file=sys.stderr)
            return 2
        if options.settings or not supplied("--input"):
            print("[错误] 请先复制 settings.example.yaml 为 settings.local.yaml 并填写配置，或使用 --input 指定数据路径。", file=sys.stderr)
            return 2
        return cli_main(["run", *remaining])
    try:
        settings = load_local_settings(settings_path)
        if options.evaluate_only:
            if not supplied("--run-dir"):
                print("[错误] --evaluate-only 必须同时提供 --run-dir 已有运行目录", file=sys.stderr)
                return 2
            if not supplied("--gold"):
                remaining.extend(["--gold", settings["input"]])
            if not supplied("--judges-config") and settings.get("judges_config"):
                remaining.extend(["--judges-config", settings["judges_config"]])
            with configured_api(settings):
                return cli_main(["evaluate", "--auto-judge", *remaining])
        if not supplied("--input"):
            remaining.extend(["--input", settings["input"]])
        if not supplied("--max-samples"):
            value = settings["max_samples"]
            remaining.extend(["--max-samples", "None" if value is None else str(value)])
        for field, flag in (("run_dir", "--run-dir"), ("judges_config", "--judges-config")):
            if settings.get(field) and not supplied(flag) and not (field == "judges_config" and supplied("--no-judge")):
                remaining.extend([flag, settings[field]])
        with configured_api(settings):
            return cli_main(["run", *remaining])
    except (ValueError, OSError):
        # 不输出本地文件内容或原始解析错误，避免用户填写的 Key 进入日志。
        print("[错误] 本地配置读取失败：请检查文件路径、字段类型、HTTPS 地址和业务空间 ID；不要在控制台粘贴 Key。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
