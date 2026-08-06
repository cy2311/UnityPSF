#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}}}"
UNITY_DIR="$ROOT/unity"
PY="${UNITY_V04_PYTHON:-${NEPTUNE_V04_PYTHON:-${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}}}"
HOST="${UNITY_V04_WEB_HOST:-${NEPTUNE_V04_WEB_HOST:-${NEPTUNE_V03_WEB_HOST:-127.0.0.1}}}"
PORT="${UNITY_V04_WEB_PORT:-${NEPTUNE_V04_WEB_PORT:-${NEPTUNE_V03_WEB_PORT:-8765}}}"

export PYTHONPATH="$UNITY_DIR/src:$ROOT"
exec "$PY" "$UNITY_DIR/scripts/gui/submit_training_web.py" --host "$HOST" --port "$PORT"
