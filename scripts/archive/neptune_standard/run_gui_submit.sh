#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNITY_V04_ROOT:-${NEPTUNE_V04_ROOT:-${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}}}"
UNITY_DIR="$ROOT/unity"
PY="${UNITY_V04_PYTHON:-${NEPTUNE_V04_PYTHON:-${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}}}"

export PYTHONPATH="$UNITY_DIR/src:$ROOT"
exec "$PY" "$UNITY_DIR/scripts/gui/submit_training_gui.py"
