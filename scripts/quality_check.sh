#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
QUALITY_TOOLS_DIR="${QUALITY_TOOLS_DIR:-$PROJECT_ROOT/.local/quality-tools}"
export PYTHONDONTWRITEBYTECODE=1
export MYPY_CACHE_DIR="$PROJECT_ROOT/.local/cache/mypy"
if [[ -d "$QUALITY_TOOLS_DIR" ]]; then
  export PYTHONPATH="$QUALITY_TOOLS_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

for module in ruff mypy pytest; do
  if ! "$PYTHON" -c "import $module" >/dev/null 2>&1; then
    echo "missing quality tool: $module" >&2
    echo "install the dev extra or set QUALITY_TOOLS_DIR to an isolated tool directory" >&2
    exit 2
  fi
done

"$PYTHON" -m ruff check src tests --cache-dir .local/cache/ruff
"$PYTHON" -m mypy
"$PYTHON" -m pytest -q -o cache_dir=.local/cache/pytest
