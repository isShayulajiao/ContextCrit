"""Check the verified training stack before importing Torch or spawning ranks.

Only standard-library metadata and filesystem reads are used. In particular,
``find_spec('transformers.masking_utils')`` is deliberately avoided because it
imports the parent Transformers package before it can diagnose an old version.
"""
from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import shlex
import sys


REQUIRED_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "5.3.0",
    "peft": "0.18.1",
    "accelerate": "1.12.0",
}


class TrainingEnvironmentError(RuntimeError):
    """The selected Python cannot run the verified training implementation."""


def _failure(problems, versions):
    actual = "、".join(f"{name}={version or '未安装'}" for name, version in versions.items())
    required = "、".join(f"{name}={version}" for name, version in REQUIRED_VERSIONS.items())
    requirements = Path(__file__).resolve().with_name("requirements.txt")
    command = f"{shlex.quote(sys.executable)} -m pip install -r {shlex.quote(str(requirements))}"
    return TrainingEnvironmentError(
        "训练环境预检失败（尚未启动训练或多卡进程）。\n"
        f"当前 Python：{sys.executable}\n"
        f"检测到：{actual}\n"
        f"本实现已验证的版本：{required}（Torch 允许 +cu118 / +cpu 构建）。\n"
        + "\n".join(f"- {problem}" for problem in problems)
        + "\n若已配置 Conda 环境，请先 conda activate scope-qwen35，确认 python 指向该环境；"
        "环境名称本身不能保证依赖版本正确。\n"
        f"在选定解释器中安装依赖：\n{command}\n"
        "这不是 CUDA 显存不足；不要通过修改导入路径或关闭低显存实现来绕过版本检查。"
    )


def check_training_environment(require_model=True):
    """Return safe environment details or raise an actionable RuntimeError.

    Exact versions reflect the tested implementation, rather than claiming
    compatibility with every newer Transformers API. Torch local build suffixes
    are accepted, so the same check supports CUDA 11.8 and tiny CPU tests.
    ``require_model=False`` omits only the native Qwen3.5 model-file check.
    This function never imports the training libraries or initializes CUDA.
    """
    versions, problems = {}, []
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        problems.append("需要 Python 3.10–3.12；推荐 3.11")
    for name, expected in REQUIRED_VERSIONS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        comparable = actual.partition("+")[0] if actual is not None and name == "torch" else actual
        if comparable != expected:
            problems.append(f"{name} 需要 {expected}，实际为 {actual or '未安装'}")
    if problems:
        raise _failure(problems, versions)

    # Inspect the installed distribution directly; do not import its parent.
    sources = ["transformers/masking_utils.py"]
    if require_model:
        sources.append("transformers/models/qwen3_5/modeling_qwen3_5.py")
    try:
        installed = metadata.distribution("transformers")
        for source in sources:
            if not Path(installed.locate_file(source)).is_file():
                problems.append(f"Transformers 安装不完整：缺少 {source}")
    except (metadata.PackageNotFoundError, OSError) as exc:
        problems.append(f"无法检查 Transformers 安装文件：{exc}")
    if problems:
        raise _failure(problems, versions)
    return {
        "python_executable": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "versions": versions,
        "checked_sources": sources,
    }
