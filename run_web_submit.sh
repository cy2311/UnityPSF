#!/usr/bin/env bash
set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PY="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"
HOST="${NEPTUNE_V03_WEB_HOST:-127.0.0.1}"
PORT="${NEPTUNE_V03_WEB_PORT:-8765}"

export PYTHONPATH="$NEPTUNE_DIR/src:$ROOT"
exec "$PY" "$NEPTUNE_DIR/scripts/gui/submit_training_web.py" --host "$HOST" --port "$PORT"
