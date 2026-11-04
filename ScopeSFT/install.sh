#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scope_env="${1:-${HOME}/.venvs/scope-qwen35}"
scope_python="${SCOPE_PYTHON:-python3}"
"$scope_python" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('请使用 Python 3.10–3.12；推荐 3.11。')
PY
"$scope_python" -m venv "$scope_env"
"$scope_env/bin/python" -m pip install 'pip==25.0.1'
"$scope_env/bin/python" -m pip install 'torch==2.5.1+cu118' --index-url https://download.pytorch.org/whl/cu118
"$scope_env/bin/python" -m pip install -r "$script_dir/requirements.txt"
"$scope_env/bin/python" -m pip check
"$scope_env/bin/python" - <<'PY'
import torch
import transformers
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig

if torch.__version__ != '2.5.1+cu118':
    raise SystemExit(f'需要 torch 2.5.1+cu118，实际为 {torch.__version__}')
print(f'torch={torch.__version__}, transformers={transformers.__version__}; Qwen3.5 类可导入。')
PY
printf '\n安装完成。激活环境：source %q/bin/activate\n' "$scope_env"
