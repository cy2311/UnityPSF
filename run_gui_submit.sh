#!/usr/bin/env bash
set -euo pipefail

ROOT="${NEPTUNE_V03_ROOT:-/home/guest/Others/main/race}"
NEPTUNE_DIR="$ROOT/neptune_v0.3"
PY="${NEPTUNE_V03_PYTHON:-/home/guest/anaconda3/bin/python}"

export PYTHONPATH="$NEPTUNE_DIR/src:$ROOT"
exec "$PY" "$NEPTUNE_DIR/scripts/gui/submit_training_gui.py"
