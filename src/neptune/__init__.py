"""Neptune-side experimental modules."""

from __future__ import annotations

import os
from pathlib import Path

from components.bootstrap import ensure_component_path


if os.getenv("NEPTUNE_SKIP_COMPONENT_BOOTSTRAP", "").strip().lower() not in {"1", "true", "yes"}:
    ensure_component_path(Path(__file__).resolve().parents[1])
