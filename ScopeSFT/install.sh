#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scope_env="${1:-${HOME}/.venvs/scope-qwen3}"
python3 - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('请使用 Python 3.10–3.12；推荐 3.11。')
PY
python3 -m venv "$scope_env"
"$scope_env/bin/python" -m pip install 'pip==25.0.1'
"$scope_env/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
"$scope_env/bin/python" -m pip install -r "$script_dir/requirements.txt"
"$scope_env/bin/python" -m pip check
printf '\n安装完成。激活环境：source %q/bin/activate\n' "$scope_env"
