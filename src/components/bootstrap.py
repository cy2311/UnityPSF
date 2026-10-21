"""Path bootstrap for the v0.4 source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_component_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root.is_dir() and root_str not in sys.path:
        sys.path.insert(0, root_str)
    components_dir = root / "components"
    components_str = str(components_dir)
    if components_dir.is_dir() and components_str not in sys.path:
        insert_at = 1 if sys.path and sys.path[0] == root_str else 0
        sys.path.insert(insert_at, components_str)
    return components_dir
