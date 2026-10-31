"""最简运行入口：安装 requirements.txt 后即可直接执行，无需 pip install -e。

示例：
    python run.py --input ../CritPred_test.json
    python run.py --input ../CritPred_train.json --max-samples 521
    python run.py --input /数据/全部任务.json --max-samples None

密钥只从 DEEPSEEK_API_KEY 环境变量读取，不要把真实密钥写进此文件。
"""
import sys


def main(argv=None):
    if sys.version_info < (3, 11):
        print("[错误] ScopeCrit 需要 Python 3.11 或以上；建议 conda create -n scopecrit python=3.11", file=sys.stderr)
        return 2
    from scopecrit.cli import main as cli_main
    return cli_main(["run", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
